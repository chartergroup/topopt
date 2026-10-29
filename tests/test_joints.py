"""A7b: joints, a beam from a load region to a fixed point, bordered solve (CONTRACTS.md section 38).

The pad case of `case()`: design x 0..20, y 0..10, z 0..10; clamp plate x -2..0 (`clamp`); a pad x 20..22 (a load region
with a joint and no force); a tip box x 8..12, y 4..6, z 10..13 with the force. The joint goes from the pad to the far point
(40, 5, 5) with a round beam d 4 mm. The closed forms use a cube from `tools/joint_check.py`.
"""

import ctypes
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import yaml
from helpers import needs_accelerate, needs_pardiso
from test_gpu_mg_x_merge import fake_cpu_device  # noqa: F401  (a fixture)
from test_springs import box

from topopt import buckling_voxel as bv
from topopt import cli
from topopt import joints as jt
from topopt import solver as solver_module
from topopt import springs as sp
from topopt.cli import main
from topopt.config import JOINT_AXIS_OPTIONS, SectionSpec, UserError, load_config
from topopt.fe import rigid_body_modes
from topopt.loads import check_rigid_modes, extended_facts, extended_lines
from topopt.optimize import Model, make_solver
from topopt.problem import build_problem
from topopt.solver import LinearSolver
from topopt.springs import FACE_UNIT, face_nodes
from topopt.voxelize import FIXED, LOAD

REPO = Path(__file__).resolve().parents[1]
JOINT = {"region": "pad", "to": [40, 5, 5], "section": {"kind": "round", "d": 4.0}}


def tool(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


jc = tool("joint_check")


def sub(tmp_path, name):
    (tmp_path / name).mkdir()
    return tmp_path / name


def case(tmp_path, h=1.0, mutate=None, clamp=True, solver="amg", joint=None):
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "pad.stl", (20, 0, 0), (22, 10, 10))
    box(tmp_path / "tip.stl", (8, 4, 10), (12, 6, 13))
    d = {
        "name": "padcase",
        "regions": {"design": "design.stl", "fixed": ["plate.stl"] if clamp else [],
                    "load": {"pad": "pad.stl", "tip": "tip.stl"}},
        "joints": [dict(JOINT, **(joint or {}))],
        "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
        "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 3, "solver": solver},
    }
    if mutate:
        mutate(d)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def prob_of(tmp_path, **kw):
    return build_problem(load_config(case(tmp_path, **kw)))


def run_args(p, out, *extra):
    return ["run", str(p), "--out", str(out), "--quiet", *extra]


def solved(prob, rtol=1e-12, rho=0.7):
    m = Model(prob)
    m.solver.rtol = rtol
    r = np.full(m.n_design, rho)
    K = m.stiffness(r)
    return m, K, r, m.solve(K)


# --- the YAML ---------------------------------------------------------------------------------------------------


def test_a_plain_yaml_has_no_joints_and_keeps_the_default_path(tmp_path):
    cfg = load_config(case(tmp_path, mutate=lambda d: d.pop("joints") or d["regions"].update(load={"tip": "tip.stl"})))
    assert cfg.joints == () and not cfg.has_joints() and not cfg.uses_extended_loads()
    prob = build_problem(cfg)
    assert prob.joints is None and prob.extended is None
    m = Model(prob)
    assert m.kb is None and type(m.solver) is LinearSolver           # the old objects, not a bordered solver
    m.solver.free()
    s, _ = make_solver("amg", m.free_dofs, prob.shape, prob.h)
    assert type(s) is LinearSolver


def test_a_joint_region_with_no_force_is_valid_and_the_options_load(tmp_path):
    cfg = load_config(case(tmp_path, joint={"origin": [21, 5, 5], "orient": [0, 0, 5], "bond_stiffness_per_area": 7.5,
                                            "E": 2000, "nu": 0.3, "section": {"kind": "tube", "od": 6, "id": 4}}))
    j = cfg.joints[0]
    assert cfg.has_joints() and cfg.uses_extended_loads() and j.region == "pad" and j.to == (40.0, 5.0, 5.0)
    assert j.section == SectionSpec("tube", (6.0, 4.0)) and j.origin == (21.0, 5.0, 5.0) and j.orient == (0.0, 0.0, 1.0)
    assert (j.bond_stiffness_per_area, j.E, j.nu) == (7.5, 2000.0, 0.3)
    assert not any(f.region == "pad" for f in cfg.load_cases[0].forces)      # no force on the joint region


@pytest.mark.parametrize("change, words", [
    (lambda d: d["joints"][0].update(region="nope"), "not a load region"),
    (lambda d: d["joints"][0].pop("to"), "missing key 'to'"),
    (lambda d: d["joints"][0].pop("section"), "missing key 'section'"),
    (lambda d: d["joints"][0].update(color="red"), "unknown key 'color'"),
    (lambda d: d["joints"].append(dict(JOINT)), "again: one joint for each region"),
    (lambda d: d.update(joints={"pad": 1}), "must be a list"),
    (lambda d: d["joints"][0].update(to=[1, 2]), "list of 3 numbers"),
    (lambda d: d["joints"][0].update(section={"kind": "round", "d": 0}), "'joints[0].section.d' must be > 0"),
    (lambda d: d["joints"][0].update(section={"kind": "round", "d": -3.0}), "must be > 0"),
    (lambda d: d["joints"][0].update(section={"kind": "rect", "b": 2, "h": 0}), "'joints[0].section.h' must be > 0"),
    (lambda d: d["joints"][0].update(section={"kind": "props", "A": 1, "Iy": 1, "Iz": 1, "J": -1}), "'joints[0].section.J'"),
    (lambda d: d["joints"][0].update(section={"kind": "tube", "od": 4, "id": 4}), "'joints[0].section.id' is 4, not less than od 4"),
    (lambda d: d["joints"][0].update(section={"kind": "tube", "od": 4, "id": 5}), "not less than od"),
    (lambda d: d["joints"][0].update(section={"kind": "round"}), "missing key 'd'"),
    (lambda d: d["joints"][0].update(section={"kind": "round", "d": 4, "id": 1}), "unknown key 'id'"),
    (lambda d: d["joints"][0].update(section={"kind": "box", "d": 4}), "must be a mapping with a kind"),
    (lambda d: d["joints"][0].update(section="round"), "must be a mapping with a kind"),
    (lambda d: d["joints"][0].update(E=0), "'joints[0].E' must be > 0"),
    (lambda d: d["joints"][0].update(nu=0.5), "'joints[0].nu'"),
    (lambda d: d["joints"][0].update(bond_stiffness_per_area=0), "bond_stiffness_per_area' must be > 0"),
    (lambda d: d["joints"][0].update(orient=[0, 0, 0]), "must not be the zero vector"),
    (lambda d: d["joints"][0].update(origin=[40, 5, 5]), "the beam has length 0 mm and no direction"),
    (lambda d: d["joints"][0].update(origin=[20, 5, 5], orient=[1, 0, 0]), "is parallel to the beam axis"),
    (lambda d: d["joints"][0].update(origin=[20, 5, 5], orient=[-3, 0, 0]), "is parallel to the beam axis"),
])
def test_a_bad_joint_entry_stops_with_the_fact(tmp_path, change, words):
    with pytest.raises(UserError) as e:
        load_config(case(tmp_path, mutate=change))
    assert words in str(e.value) and "(file:" in str(e.value)


def test_a_degenerate_beam_with_the_default_reference_point_stops_when_the_problem_is_built(tmp_path):
    ref = prob_of(tmp_path).joints.joints[0].ref                  # the area centroid of the bonded faces
    with pytest.raises(UserError, match="the beam has length 0 mm and no direction") as e:
        prob_of(sub(tmp_path, "a"), joint={"to": [float(v) for v in ref]})
    assert "joint on load region 'pad'" in str(e.value) and "Options:" in str(e.value)
    with pytest.raises(UserError, match="is parallel to the beam axis"):
        prob_of(sub(tmp_path, "b"), joint={"orient": [1, 0, 0]})                  # the beam runs along +x from the centroid
    inside = sub(tmp_path, "c")

    def inner(d):
        box(inside / "core.stl", (8, 3, 3), (12, 7, 7))
        d["regions"]["load"]["core"] = "core.stl"
        d["joints"] = [{"region": "core", "to": [30, 5, 5], "section": {"kind": "round", "d": 2}}]
    with pytest.raises(UserError, match="no face of the region looks toward empty space"):
        build_problem(load_config(case(inside, mutate=inner)))


def test_neither_a_fixed_region_nor_a_joint_names_the_options(tmp_path):
    def none(d):
        d.pop("joints")
        d["regions"]["fixed"] = []
    with pytest.raises(UserError) as e:
        load_config(case(tmp_path, mutate=none))
    assert "'joints'" in str(e.value) and "rotation about x, y, z" in str(e.value) and "Options:" in str(e.value)


def test_one_options_text_for_a_beam_without_a_direction_at_load_time_and_at_build_time(tmp_path):
    """Review n1: with `origin` given the fact is found when the YAML is loaded, without it when the problem is built. Both have the options."""
    with pytest.raises(UserError) as e1:
        load_config(case(tmp_path, mutate=lambda d: d["joints"][0].update(origin=[40, 5, 5])))
    ref = prob_of(sub(tmp_path, "a")).joints.joints[0].ref
    with pytest.raises(UserError) as e2:
        prob_of(sub(tmp_path, "b"), joint={"to": [float(v) for v in ref]})
    for e in (e1, e2):
        assert "the beam has length 0 mm and no direction" in str(e.value) and ". " + JOINT_AXIS_OPTIONS in str(e.value)


def test_the_zero_vector_tests_use_one_constant(tmp_path):
    """Review n7: `DEGENERATE` is the test of `_vec3`, and a vector of that size is the zero vector, in `orient` and in the build direction."""
    from topopt.config import DEGENERATE
    assert DEGENERATE == 1e-9
    for change in (lambda d: d["joints"][0].update(orient=[0, 0, 0.9 * DEGENERATE]),
                   lambda d: d.update(print={"build_direction": [0.0, 0.9 * DEGENERATE, 0.0]})):
        with pytest.raises(UserError, match="must not be the zero vector"):
            load_config(case(sub(tmp_path, f"z{id(change)}"), mutate=change))
    ok = load_config(case(sub(tmp_path, "ok"), mutate=lambda d: d["joints"][0].update(orient=[0, 0, 1.1 * DEGENERATE])))
    assert ok.joints[0].orient == (0.0, 0.0, 1.0)


def test_the_material_of_the_beam_says_which_key_it_has(tmp_path):
    """Review n2."""
    for key, value, want in (("E", 9999.0, "the joint entry (E) and the run material (nu)"),
                             ("nu", 0.2, "the joint entry (nu) and the run material (E)")):
        prob = prob_of(sub(tmp_path, key), joint={key: value})
        assert prob.extended["joints"]["joints"][0]["beam_material_from"] == want
        assert any(want in line for line in extended_lines(extended_facts(prob)))


def test_the_texts_of_the_stress_report_and_the_support_refusal_name_a_joint(tmp_path):
    """Review n4 and n5."""
    from topopt.stress_report import input_changes
    out = input_changes(prob_of(tmp_path), {})
    assert any("a spring or a joint is not compared with run.json" in t for t in out["not_compared"])
    with pytest.raises(UserError, match="add a joint on a load region"):
        check_rigid_modes(np.zeros(0, dtype=np.int64), (3, 3, 3), 1.0)


# --- sections, the beam and the frame -----------------------------------------------------------------------------


def test_the_section_values():
    r = jt.section_properties(SectionSpec("round", (4.0,)))
    assert r["A"] == pytest.approx(math.pi * 4) and r["Iy"] == r["Iz"] == pytest.approx(math.pi * 4 ** 4 / 64)
    assert r["J"] == pytest.approx(2 * r["Iy"]) and r["depth"] == 4.0
    t = jt.section_properties(SectionSpec("tube", (6.0, 4.0)))
    assert t["A"] == pytest.approx(math.pi * (36 - 16) / 4) and t["Iy"] == pytest.approx(math.pi * (6 ** 4 - 4 ** 4) / 64)
    assert t["J"] == pytest.approx(2 * t["Iy"]) and t["depth"] == 6.0
    q = jt.section_properties(SectionSpec("rect", (4.0, 2.0)))        # b along the section y axis, h along z
    assert q["A"] == 8.0 and q["Iy"] == pytest.approx(4 * 2 ** 3 / 12) and q["Iz"] == pytest.approx(2 * 4 ** 3 / 12) and q["depth"] == 4.0
    p = jt.section_properties(SectionSpec("props", (3.0, 4.0, 5.0, 6.0)))
    assert (p["A"], p["Iy"], p["Iz"], p["J"]) == (3.0, 4.0, 5.0, 6.0) and p["depth"] == pytest.approx(math.sqrt(12 / math.pi))


def test_the_torsion_constant_of_a_rectangle():
    assert jt.rect_torsion_constant(1.0, 1.0) == pytest.approx(0.1405770, rel=1e-6)                 # the square: 0.14058 a^4
    assert jt.rect_torsion_constant(2.0, 1.0) == pytest.approx(0.45776, rel=2e-3)      # Roark: K = a b^3 (16/3 - 3.36 b/a (1 - b^4/12a^4)), a = 1, b = 0.5
    assert jt.rect_torsion_constant(2.0, 1.0) == jt.rect_torsion_constant(1.0, 2.0)
    thin = jt.rect_torsion_constant(100.0, 1.0)
    assert thin == pytest.approx(100.0 / 3.0 * (1 - 0.63 / 100.0), rel=2e-3) and thin < 100.0 / 3.0   # a t^3 / 3 (1 - 0.63 t / a), a thin plate


def beam6(axis, L=10.0, orient=None, E=200.0, nu=0.25, props=None):
    props = props or jt.section_properties(SectionSpec("round", (2.0,)))
    Q = jt.beam_frame(axis, orient)
    return jt.beam_stiffness(E, nu, props, L, Q), Q, props


def test_the_beam_matrix_has_the_five_closed_forms():
    E, nu, L = 200.0, 0.25, 10.0
    B, Q, p = beam6((1, 0, 0), L, E=E, nu=nu)
    assert np.allclose(Q, np.eye(3)) or np.allclose(Q[:, 0], [1, 0, 0])
    EI, G = E * p["Iy"], E / (2 * (1 + nu))
    assert B[0, 0] == pytest.approx(E * p["A"] / L) and B[3, 3] == pytest.approx(G * p["J"] / L)
    for i in (1, 2):
        assert B[i, i] == pytest.approx(12 * EI / L ** 3)
    for i in (4, 5):
        assert B[i, i] == pytest.approx(4 * EI / L)
    assert abs(B[1, 5]) == pytest.approx(6 * EI / L ** 2) and abs(B[2, 4]) == pytest.approx(6 * EI / L ** 2)
    assert B[1, 5] * B[2, 4] < 0                       # the two bending planes have opposite signs of the coupling
    assert np.allclose(B, B.T) and np.linalg.eigvalsh(B).min() > 0


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (0, 0, -1), (1, 1, 1), (1, 2, 2), (-1, 2, -3), (0, 1e-4, 1)])
def test_the_frame_is_a_rotation_and_the_matrix_follows_it(axis):
    B, Q, _ = beam6(axis, props=jt.section_properties(SectionSpec("rect", (4.0, 2.0))))
    assert np.allclose(Q.T @ Q, np.eye(3)) and np.linalg.det(Q) == pytest.approx(1.0)
    assert np.allclose(Q[:, 0], np.asarray(axis, float) / np.linalg.norm(axis))
    B0, _, _ = beam6((1, 0, 0), props=jt.section_properties(SectionSpec("rect", (4.0, 2.0))), orient=(0, 1, 0))
    T = np.zeros((6, 6))
    T[:3, :3] = T[3:, 3:] = Q
    local = T.T @ B @ T                                 # the matrix in the frame of the beam: the same for every direction
    assert np.allclose(local, B0, rtol=0, atol=1e-9 * abs(B0).max())
    assert np.allclose(B, B.T) and np.linalg.eigvalsh(B).min() > 0


def test_the_section_y_axis_follows_orient_and_the_default_avoids_the_beam():
    for axis, helper in (((1, 0.1, 0), 2), ((0.1, 1, 0.2), 0), ((0.2, 0.3, 1), 0)):       # default: the global axis least parallel to the beam
        ex = np.asarray(axis, float) / np.linalg.norm(axis)
        want = np.eye(3)[helper] - ex[helper] * ex                                          # that axis with the part along the beam removed
        assert np.allclose(jt.beam_frame(axis)[:, 1], want / np.linalg.norm(want))
    Q = jt.beam_frame((1, 0, 0), (0, 0, 5))
    assert np.allclose(Q[:, 1], (0, 0, 1)) and np.allclose(Q[:, 2], (0, -1, 0))        # y = orient, z = x cross y
    Q = jt.beam_frame((1, 0, 0), (0.7, 0, 3))                                            # the part along the beam is removed
    assert np.allclose(Q[:, 1], (0, 0, 1))
    Q = jt.beam_frame((0, 1, 0))                                                         # the pipeline frame is singular here
    assert np.allclose(Q.T @ Q, np.eye(3)) and abs(Q[:, 1] @ Q[:, 0]) < 1e-15
    r = jt.beam_stiffness(1.0, 0.3, jt.section_properties(SectionSpec("rect", (4.0, 2.0))), 5.0, jt.beam_frame((1, 0, 0), (0, 1, 0)))
    s = jt.beam_stiffness(1.0, 0.3, jt.section_properties(SectionSpec("rect", (4.0, 2.0))), 5.0, jt.beam_frame((1, 0, 0), (0, 0, 1)))
    assert r[1, 1] > s[1, 1] * 3 and r[2, 2] < s[2, 2] / 3      # orient y: the stiff direction of b = 4 flips with orient


# --- the bond -----------------------------------------------------------------------------------------------------


def test_the_bond_triple_is_consistent_and_d_is_summed_with_care():
    c = jc.cube(8)
    nodes = face_nodes(jt.exposed_faces(c.labels, c.ids == 0), c.shape)
    ref = np.array([0.3, -0.2, 0.1])                          # a reference point that is not the centroid
    Kb, C, crow, D = jt.bond_blocks(nodes, c.shape, c.origin, c.h, ref, 12.0)
    assert C.shape == (crow.size, 6) and (np.diff(crow) > 0).all() and crow.size < 3 * np.prod(np.array(c.shape) + 1)    # the bonded rows only
    assert set(crow) == set((3 * nodes[:, :, None] + np.arange(3)).ravel())
    # rigid motion u = R q of every node: the bond has no energy, so C^T u + D q = 0 and Kb u + C q = 0
    q = np.array([0.1, -0.2, 0.3, 0.02, -0.01, 0.03])
    pos = jt.node_positions(np.arange(int(np.prod(np.array(c.shape) + 1))), c.shape, c.origin, c.h) - ref
    u = (q[:3] + np.cross(q[3:], pos)).ravel()
    assert np.abs(C.T @ u[crow] + D @ q).max() <= 1e-12 * np.abs(D @ q).max()
    Cq = np.zeros(len(u))
    Cq[crow] = C @ q
    g = np.bincount(Kb[0], weights=Kb[2] * u[Kb[1]], minlength=len(u)) + Cq
    assert np.abs(g).max() <= 1e-12 * np.abs(Kb[2]).max() * np.abs(u).max()
    # D against the exactly rounded sum of the face matrices
    R = np.zeros(nodes.shape + (3, 6))
    R[..., :3] = np.eye(3)
    R[..., 3:] = -jt._skew(jt.node_positions(nodes.ravel(), c.shape, c.origin, c.h).reshape(nodes.shape + (3,)) - ref)
    Df = np.einsum("fiad,ij,fjae->fde", R, FACE_UNIT, R)
    exact = 12.0 * np.array([[math.fsum(Df[:, d, e]) for e in range(6)] for d in range(6)])
    assert np.abs(D - exact).max() <= 4e-16 * np.abs(D).max() * 4
    assert np.allclose(D, D.T) and np.linalg.eigvalsh(D).min() > 0


def test_c_on_the_free_dofs_cuts_the_fixed_rows_and_keeps_the_order_of_the_dofs():
    """`Joint.C` holds the rows of the bonded dofs only (review n10); `C_free` gives the rows of any list of dofs, also when some
    bonded dofs are fixed (not in the list) and when the list is not sorted."""
    c = jc.cube(4)
    j = jc.make_joint(c, (14.0, 6.0, -9.0), ("tube", 3.0, 1.0), E_beam=2.0, k_a=50.0).joints[0]
    n_dof = 3 * int(np.prod(np.array(c.shape) + 1))
    full = np.zeros((n_dof, 6))
    full[j.crow] = j.C.toarray()
    assert j.C.shape[0] == j.crow.size < n_dof                    # the cube has an inner node row set, so fewer rows than all dofs
    fd = c.free[np.arange(c.free.size) % 7 != 3]                 # some bonded dofs are fixed now
    assert np.array_equal(j.C_free(fd).toarray(), full[fd]) and 0 < np.abs(full[np.setdiff1d(c.free, fd)]).sum()
    perm = np.random.default_rng(1).permutation(fd)
    assert np.array_equal(j.C_free(perm).toarray(), full[perm])
    assert np.array_equal(j.C_free(np.zeros(0, dtype=np.int64)).toarray(), np.zeros((0, 6)))
    q = np.arange(1.0, 7.0)
    # The limit is for each row, from the sum of |C_ij q_j|: some rows cancel to about 1e-13 (for example -50 x 2 - 50 x 4
    # + 49.99999999999999 x 6), and there the last bits depend on the BLAS of numpy. Measured on Apple silicon (Accelerate):
    # 2.8e-14 between `C_dot` and `full @ q`, on rows with a sum of 600 to 800.
    assert np.all(np.abs(j.C_dot(q, n_dof) - full @ q) <= 1e-13 * (np.abs(full) @ np.abs(q)))


def test_d_equals_the_two_by_two_gauss_rule_on_each_face():
    c = jc.cube(4)
    nodes = face_nodes(jt.exposed_faces(c.labels, c.ids == 0), c.shape)
    ref = np.array([0.2, 0.1, -0.3])
    _, _, _, D = jt.bond_blocks(nodes, c.shape, c.origin, c.h, ref, 5.0)
    pos = jt.node_positions(nodes.ravel(), c.shape, c.origin, c.h).reshape(nodes.shape + (3,)) - ref
    R = np.zeros(nodes.shape + (3, 6))
    R[..., :3] = np.eye(3)
    R[..., 3:] = -jt._skew(pos)
    g = 0.5 / math.sqrt(3.0)
    want = np.zeros((6, 6))
    for s in (-1, 1):
        for t in (-1, 1):
            a, b = 0.5 + s * g, 0.5 + t * g
            N = np.array([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b])        # nodes 00 01 10 11
            Rq = np.einsum("i,fiad->fad", N, R)
            want += 5.0 * 0.25 * np.einsum("fad,fae->de", Rq, Rq)
    assert np.abs(D - want).max() <= 1e-13 * np.abs(want).max()


def test_the_default_bond_stiffness_and_the_overrides(tmp_path):
    f = prob_of(tmp_path).extended["joints"]["joints"][0]
    assert f["bond_stiffness_per_area_n_per_mm3"] == pytest.approx(jt.BOND_C * 3500.0 / 1.0) and jt.BOND_C == 30.0
    assert f["c_equivalent"] == pytest.approx(30.0) and "c E / h" in f["bond_stiffness_from"]
    g = prob_of(sub(tmp_path, "a"), joint={"bond_stiffness_per_area": 123.0}).extended["joints"]["joints"][0]
    assert g["bond_stiffness_per_area_n_per_mm3"] == 123.0 and "bond_stiffness_per_area" in g["bond_stiffness_from"]
    e = prob_of(sub(tmp_path, "b"), joint={"E": 9999.0, "nu": 0.2, "origin": [21, 5, 5]}).extended["joints"]["joints"][0]
    assert (e["beam_E_mpa"], e["beam_nu"], e["beam_material_from"]) == (9999.0, 0.2, "the joint entry")
    assert e["reference_mm"] == [21.0, 5.0, 5.0] and "origin" in e["reference_from"]
    assert f["beam_material_from"] == "the run material" and f["beam_E_mpa"] == 3500.0 and "area centroid" in f["reference_from"]


def test_the_facts_have_the_beam_the_frame_and_the_bond(tmp_path):
    prob = prob_of(tmp_path)
    f = prob.extended["joints"]["joints"][0]
    assert f["faces"] == 180 and sum(f["faces_by_direction"].values()) == 180 and f["area_mm2"] == 180.0
    assert f["length_mm"] == pytest.approx(np.linalg.norm(np.array([40, 5, 5]) - np.array(f["reference_mm"])))
    assert f["length_over_depth"] == pytest.approx(f["length_mm"] / 4.0) and f["section"] == {"kind": "round", "d": 4.0}
    assert f["frame"]["y_from"].startswith("default") and f["axis"] == pytest.approx([1.0, 0.0, 0.0])
    assert f["rho"] == pytest.approx(min(np.array(f["bond_diagonal"]) / np.array(f["beam_diagonal"])))
    assert 0.0 < f["bond_share_max"] < 1.0 and prob.extended["joints"]["bordered_dofs"] == 6
    assert prob.extended["joints"]["bond"]["c"] == 30.0 and "hook grid" in prob.extended["joints"]["bond"]["source"]
    assert prob.extended["supports"]["rigid_body_test"].startswith("skipped") and "free_modes" not in prob.extended["supports"]
    json.dumps(extended_facts(prob))
    lines = extended_lines(extended_facts(prob))
    assert any(l.startswith("joint:") and "180 faces" in l and "L/d" in l and "Euler-Bernoulli" in l and "round d 4" in l for l in lines)
    assert any(l.startswith("joint frame:") and "section y axis" in l for l in lines)
    assert any(l.startswith("joint bond:") and "rho" in l and "percent of the joint flexibility" in l for l in lines)
    assert any(l.startswith("supports:") and "skipped" in l for l in lines)
    assert not any("rho <" in l or "threshold" in l for l in lines)                   # facts, no threshold text


# --- a very soft beam: the smallest beam stiffness is a fact, and a warning below `springs.NEAR_SINGULAR` ----------


def soft(L, d=0.5):
    """The pad case with no clamp and a thin beam of length `L` along +x from the pad (the case of the A7b review, question 5)."""
    return {"to": [21.5 + L, 5.0, 5.0], "section": {"kind": "round", "d": d}}


def test_the_smallest_beam_stiffness_is_a_fact_for_each_joint(tmp_path):
    prob = prob_of(tmp_path)
    f = prob.extended["joints"]["joints"][0]
    B = prob.joints.joints[0].B
    E, h = prob.config.material.E, prob.h
    assert f["beam_min_translation_over_Eh"] == pytest.approx(np.linalg.eigvalsh(B[:3, :3])[0] / (E * h), rel=1e-12)
    assert f["beam_min_rotation_over_Eh3"] == pytest.approx(np.linalg.eigvalsh(B[3:, 3:])[0] / (E * h ** 3), rel=1e-12)
    # a round beam d 4 mm, L 18.44 mm: 12 E I / L^3 is the smallest translational entry (bending)
    L = f["length_mm"]
    assert f["beam_min_translation_over_Eh"] == pytest.approx(12 * np.pi * 4.0 ** 4 / 64 / L ** 3 / h, rel=1e-9)
    lines = extended_lines(extended_facts(prob))
    assert any(l.startswith("joint beam:") and "of E h (translation)" in l and "of E h^3 (rotation)" in l for l in lines)
    assert cli.check_data(case(sub(tmp_path, "c")))["extended"]["joints"]["joints"][0]["beam_min_rotation_over_Eh3"] == f["beam_min_rotation_over_Eh3"]
    assert not [w for w in cli.check_data(case(sub(tmp_path, "d")))["warnings"] if w.startswith("joint on ")]


@pytest.mark.parametrize("L", [1000.0, 100.0])
def test_a_floating_part_on_a_very_soft_beam_gets_the_warning(tmp_path, L):
    """The review case: pad, no clamp, round d 0.5 mm, L 1000 mm (3.7e-11 of E h) and 100 mm (3.7e-8). Both are below
    `springs.NEAR_SINGULAR` = 1e-6; the pad case without a clamp has h = 1 mm."""
    p = case(tmp_path, clamp=False, joint=soft(L))
    d = cli.check_data(p)
    f = d["extended"]["joints"]["joints"][0]
    assert 0.0 < f["beam_min_translation_over_Eh"] < sp.NEAR_SINGULAR and f["beam_min_translation_over_Eh"] == pytest.approx(3.7e-8 * (100.0 / L) ** 3, rel=0.1)
    w = [x for x in d["warnings"] if x.startswith("joint on 'pad'")]
    assert len(w) == 1 and "below 1e-06" in w[0] and "(translation)" in w[0] and "E 3500 MPa x voxel 1 mm" in w[0]
    assert "(rotation)" not in w[0] and "Options: raise the section or E, shorten the beam, or add a fixed region" in w[0]
    assert w == jt.near_singular_lines(d["extended"]["joints"], 3500.0, 1.0)


def test_the_run_log_has_the_same_warning_for_a_soft_beam(tmp_path):
    out = tmp_path / "out"
    assert main(run_args(case(tmp_path, clamp=False, joint=soft(100.0), mutate=lambda d: d["settings"].update(iterations=1)), out)) == 0
    log = (out / "run.log").read_text()
    assert "warning: joint on 'pad': its smallest beam stiffness is" in log and "joint beam:" in log
    run = json.loads((out / "run.json").read_text())
    assert run["joints"]["joints"][0]["beam_min_translation_over_Eh"] < sp.NEAR_SINGULAR


def test_the_hook_joint_of_the_a7b_run_gives_no_warning(tmp_path):
    """The joint of CONTRACTS.md section 38: round d 2 mm in the run material, 20 mm along -z from the tip: 1e-3 of E h."""
    d = yaml.safe_load((REPO / "examples" / "hook" / "hook.yaml").read_text())
    hook = REPO / "examples" / "hook"
    d["regions"] = {"design": str(hook / "design.stl"), "fixed": [str(hook / "wall_plate.stl")],
                    "keep_out": [str(hook / "screw_hole_top.stl"), str(hook / "screw_hole_bottom.stl")],
                    "load": {"tip": str(hook / "hook_tip.stl")}}
    d["joints"] = [{"region": "tip", "to": [70.8247, -0.1, -3.7034], "section": {"kind": "round", "d": 2.0}}]
    p = tmp_path / "hook_joint.yaml"
    p.write_text(yaml.safe_dump(d))
    res = cli.check_data(p)
    f = res["extended"]["joints"]["joints"][0]
    assert f["beam_min_translation_over_Eh"] > 1e-4 and f["beam_min_rotation_over_Eh3"] > 1e-4
    assert not [w for w in res["warnings"] if w.startswith("joint on ")]


# --- memory: the estimate counts the bordered copy of K ----------------------------------------------------------


def test_the_stiffness_pattern_is_counted_exactly(tmp_path):
    from topopt.optimize import reduced_system
    for clamp in (True, False):
        prob = prob_of(sub(tmp_path, f"c{int(clamp)}"), h=2.0, clamp=clamp)
        edof, _active, free = reduced_system(prob)
        m = Model(prob)
        K = m.stiffness(np.full(m.n_design, 0.5))
        assert jt.stiffness_nnz(edof, free) == K.nnz
        m.solver.free()


def test_the_estimate_counts_the_bordered_copy_and_only_with_joints(tmp_path):
    from topopt.cli import bordered_bytes, estimate_memory_bytes
    assert jt.bordered_copy_bytes(1000) == 1000 * (8 + 4 + 8 + 4 + 4)         # the CSC copy and the COO list of `bmat`
    g, f, e = 10_000, 30_000, 5_000
    assert estimate_memory_bytes(g, f, e, 0) == estimate_memory_bytes(g, f, e)           # without joints: the old number
    assert estimate_memory_bytes(g, f, e, 123_456) == estimate_memory_bytes(g, f, e) + 123_456
    prob = prob_of(tmp_path)
    from topopt.optimize import reduced_system
    edof, _a, free = reduced_system(prob)
    nnz = jt.stiffness_nnz(edof, free)
    assert bordered_bytes(prob, edof, free) == jt.bordered_copy_bytes(nnz) > 0
    plain = prob_of(sub(tmp_path, "plain"), mutate=lambda d: d.pop("joints") or d["regions"].update(load={"tip": "tip.stl"}))
    assert bordered_bytes(plain, edof, free) == 0


def test_check_and_the_run_log_name_the_bordered_copy(tmp_path):
    from topopt.cli import estimate_memory_bytes, run_memory_lines
    from topopt.members import filter_size
    d = cli.check_data(case(tmp_path, h=2.0))
    me = d["memory_estimate"]
    prob = prob_of(sub(tmp_path, "p"), h=2.0)
    from topopt.optimize import reduced_system
    edof, _a, free = reduced_system(prob)
    want = jt.bordered_copy_bytes(jt.stiffness_nnz(edof, free))
    assert me["bordered_copy_bytes"] == want
    fsz = filter_size(float(prob.config.settings.filter_radius), float(prob.h), int((prob.labels == 1).sum()), prob.shape)
    assert me["bytes"] == estimate_memory_bytes(int(np.prod(prob.shape)), int(free.size), fsz["held_bytes_estimate"], want)
    assert me["bytes"] > estimate_memory_bytes(int(np.prod(prob.shape)), int(free.size), fsz["held_bytes_estimate"])
    lines = run_memory_lines(prob, {"size": fsz})
    assert "the bordered copy of K for the joints" in lines[0]
    plain = cli.check_data(case(sub(tmp_path, "q"), h=2.0, mutate=lambda d: d.pop("joints") or d["regions"].update(load={"tip": "tip.stl"})))
    assert "bordered_copy_bytes" not in plain["memory_estimate"]


# --- the closed forms ---------------------------------------------------------------------------------------------

AXES = {"x": (1, 0, 0), "oblique": (1, 2, 2)}


def loads_of(axis):
    xp, p, q = jc.perpendicular_pair(axis)
    return {"axial": (5 * xp, np.zeros(3)), "bending p": (5 * p, np.zeros(3)), "bending q": (5 * q, np.zeros(3)),
            "torsion": (np.zeros(3), 5 * xp), "force p + moment q": (3 * p, 7 * q), "force q + moment p": (3 * q, 7 * p)}


@pytest.mark.parametrize("solver", ["splu", "amg", pytest.param("pardiso", marks=needs_pardiso)])
@pytest.mark.parametrize("name", ["x", "oblique"])
def test_the_closed_forms_hold_for_an_aligned_and_an_oblique_beam(name, solver):
    """Axial, bending in both planes, torsion, and a force with a moment (the cross term): within 1e-3. Measured: the axial
    error is +2.9e-4 (the flexibility of the block and the bond in series), the others 4e-7 to 5e-6."""
    for lname, load in loads_of(AXES[name]).items():
        got, ref = jc.case(4, AXES[name], load, solver=solver)
        assert abs(got / ref - 1.0) < 1e-3, (name, lname, got / ref - 1.0)
        if lname != "axial":
            assert abs(got / ref - 1.0) < 3e-5, (name, lname)


def prandtl_j(b, h, n):
    """Torsion constant of a b x h rectangle by a finite-difference solve of lap(phi) = -2 (phi = 0 on the edge), J = 2 int phi."""
    nx, ny = n, int(round(n * h / b))
    dx, dy = b / nx, h / ny
    tx = sps.diags([-1, 2, -1], [-1, 0, 1], shape=(nx - 1, nx - 1)) / dx ** 2
    ty = sps.diags([-1, 2, -1], [-1, 0, 1], shape=(ny - 1, ny - 1)) / dy ** 2
    A = sps.kron(tx, sps.eye(ny - 1)) + sps.kron(sps.eye(nx - 1), ty)
    return 2.0 * spla.spsolve(A.tocsc(), np.full(A.shape[0], 2.0)).sum() * dx * dy


def rect_case(tmp_path, solver):
    """The full YAML chain of the A7b review (question 2): a stiff 6 mm block (one load region, no fixed region, no spring), a beam of
    E 2000 MPa with a `rect` section b 3 mm along the section y axis and h 1.5 mm along z, L 40 mm, an oblique axis (-1, 2, -3), and
    an `orient` (1, 1, 0) that is not perpendicular to the axis. Return the problem, the engine compliance of each case, the frame and
    the closed form. The closed form is the energy of N, T, My and Mz along the beam by Gauss quadrature, with the frame by Gram-Schmidt
    and the torsion constant from a finite-difference Prandtl solve: it does not use the 6 x 6 matrix, the frame or the series of `joints.py`."""
    E_block, E_beam, nu, b, h_sec, L = 2.0e5, 2.0e3, 0.3, 3.0, 1.5, 40.0
    a = np.array([-1.0, 2.0, -3.0])
    a /= np.linalg.norm(a)
    orient = np.array([1.0, 1.0, 0.0])
    ref = np.array([3.0, 3.0, 3.0])                       # the centre of the block 0..6 mm
    box(tmp_path / "blk.stl", (0, 0, 0), (6, 6, 6))
    box(tmp_path / "des.stl", (-2, 0, 0), (0, 6, 6))
    ey = orient - (orient @ a) * a
    ey /= np.linalg.norm(ey)
    ez = np.cross(a, ey)
    rng = np.random.default_rng(7)
    loads = {"axial": (5 * a, None), "bend_y": (5 * ey, None), "bend_z": (5 * ez, None), "torsion": (None, 5 * a),
             "Fy_Mz": (3 * ey, 7 * ez), "Fz_My": (3 * ez, 7 * ey), "random": (rng.normal(size=3) * 3, rng.normal(size=3) * 10)}
    cases = []
    for name, (F, M) in loads.items():
        e = {"region": "blk", "force": [0.0, 0.0, 0.0] if F is None else [float(v) for v in F]}
        if M is not None:
            e["moment"] = [float(v) for v in M]
        cases.append({"name": name, "forces": [e]})
    doc = {"name": "cf", "material": {"E": E_block, "nu": nu},
           "regions": {"design": "des.stl", "fixed": [], "load": {"blk": "blk.stl"}},
           "joints": [{"region": "blk", "to": [float(v) for v in ref + L * a], "origin": [float(v) for v in ref],
                       "orient": [float(v) for v in orient], "E": E_beam, "nu": nu, "section": {"kind": "rect", "b": b, "h": h_sec}}],
           "load_cases": cases, "settings": {"voxel_size": 1.0, "volume_fraction": 0.5, "iterations": 1, "solver": solver}}
    (tmp_path / "cf.yaml").write_text(yaml.safe_dump(doc))
    prob = build_problem(load_config(tmp_path / "cf.yaml"))
    m = Model(prob)
    if m.solver_name == "amg":
        m.solver.rtol = 1e-12
    comp = np.einsum("ij,ij->j", m.f_red, m.solve(m.stiffness(np.ones(m.n_design))))
    m.solver.free()
    j1, j2 = prandtl_j(b, h_sec, 100), prandtl_j(b, h_sec, 200)
    J = j2 + (j2 - j1) / 3.0                              # Richardson, O(h^2)
    area, Iy, Iz, G = b * h_sec, b * h_sec ** 3 / 12.0, h_sec * b ** 3 / 12.0, E_beam / (2 * (1 + nu))
    gx, gw = np.polynomial.legendre.leggauss(4)

    def closed(F, M, y_axis=ey):
        z_axis = np.cross(a, y_axis)
        u = 0.0
        for x, w in zip(gx, gw):
            s_, ws = 0.5 * L * (x + 1), 0.5 * L * w
            Mi = M - s_ * np.cross(a, F)                  # moment of the wrench about the section at ref + s a
            u += ws * 0.5 * ((a @ F) ** 2 / (E_beam * area) + (a @ Mi) ** 2 / (G * J) + (y_axis @ Mi) ** 2 / (E_beam * Iy)
                             + (z_axis @ Mi) ** 2 / (E_beam * Iz))
        return 2 * u
    nodes = np.stack(np.unravel_index(np.arange(prob.forces.shape[1] // 3), tuple(np.array(prob.shape) + 1)), axis=1)
    pos = prob.origin + prob.h * nodes
    wrench = [(f3.sum(0), np.cross(pos - ref, f3).sum(0)) for f3 in (prob.forces[c].reshape(-1, 3) for c in range(len(cases)))]
    return prob, dict(zip(loads, comp)), ey, J, closed, wrench


@pytest.mark.parametrize("solver", ["amg", pytest.param("pardiso", marks=needs_pardiso)])
def test_closed_forms_hold_through_the_yaml_chain_for_a_rect_section_and_an_oblique_orient(tmp_path, solver):
    """Review m8. Measured (pardiso, amg): axial +3.3e-6, bending and force with moment up to 7e-6, torsion 4e-8 or less, a random wrench
    1e-6 or less. The review's own script gave the same size (3.3e-6 and below). The tolerance is 1e-4."""
    prob, comp, ey, J, closed, wrench = rect_case(tmp_path, solver)
    jf = prob.joints.joints[0].facts
    assert jf["j_mm4"] == pytest.approx(J, rel=1e-6)                                  # the series against the Prandtl solve
    assert np.allclose(jf["frame"]["y"], ey, atol=1e-12) and jf["frame"]["y_from"] == "orient"      # the frame of Gram-Schmidt
    for (name, c), (F, M) in zip(comp.items(), wrench):
        ref = closed(F, M)
        assert abs(c / ref - 1.0) < 1e-4, (name, c / ref - 1.0)
    # the check has teeth: with the default section y axis (orient ignored) the two bending cases are far from the engine
    default_y = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ np.array([-1, 2, -3]) / 14.0) * np.array([-1, 2, -3.0])
    default_y /= np.linalg.norm(default_y)
    names = list(comp)
    F, M = wrench[names.index("bend_y")]
    assert abs(comp["bend_y"] / closed(F, M, default_y) - 1.0) > 1e-2


def test_the_force_with_a_moment_fixes_the_sign_of_the_cross_term():
    xp, p, q = jc.perpendicular_pair((1, 0, 0))
    got, ref = jc.case(4, (1, 0, 0), (3 * p, 7 * q))
    c = jc.cube(4)
    props = jt.section_properties(SectionSpec("round", (2.0,)))
    wrong = jc.closed_form((1, 0, 0), 3 * p, -7 * q, 1.0, 1.0 / 2.6, props["A"], props["Iy"], props["J"], 20.0 - 0.0)
    assert abs(got / ref - 1.0) < 3e-5 and abs(got / wrong - 1.0) > 0.5 and c.E == 1.0     # the other sign is 13 times off


def test_the_axial_error_falls_with_the_stiffness_of_the_block():
    """The axial error is the flexibility of the block and the bond in series with the beam: it falls with E_block / E_beam."""
    errors = []
    for ratio in (1.0, 10.0, 100.0):
        got, ref = jc.case(4, (1, 0, 0), (np.array([5.0, 0, 0]), np.zeros(3)), ratio=ratio)
        errors.append(got / ref - 1.0)
    assert errors[0] == pytest.approx(2.9e-4, rel=0.05) and errors[1] == pytest.approx(errors[0] / 10, rel=0.1)
    assert errors[2] == pytest.approx(errors[0] / 100, rel=0.2)


def test_the_bordered_solve_equals_the_dense_schur_form_and_the_reactions_come_from_u(tmp_path):
    c = jc.cube(4)
    set_ = jc.make_joint(c, (14.0, 6.0, -9.0), ("tube", 3.0, 1.0), E_beam=2.0, k_a=50.0)
    f = jc.nodal_load(c, np.array([1.0, -2.0, 0.5]), np.array([0.3, 1.0, -0.4]))
    u, q = jc.solve(c, set_, f, "splu")
    C, D = set_.border(c.free)
    K = jc.kb_added(c, set_).toarray()
    S = K - C.toarray() @ np.linalg.solve(D.toarray(), C.T.toarray())            # the dense Schur form
    us = np.linalg.solve(S, f)
    assert np.abs(u - us).max() <= 1e-9 * np.abs(us).max() and float(f @ u) == pytest.approx(float(f @ us), rel=1e-8)
    # q from the grid solution: -D^-1 C^T u equals the border slice of the bordered solution
    assert np.allclose(-np.linalg.solve(D.toarray(), C.T @ u), q, rtol=1e-8, atol=1e-12 * np.abs(q).max())


@pytest.mark.parametrize("solver", ["amg", pytest.param("pardiso", marks=needs_pardiso)])
def test_a_run_with_a_joint_gives_the_same_displacement_as_the_dense_schur_form(tmp_path, solver):
    prob = prob_of(tmp_path, h=2.0, solver=solver)
    m, K, rho, U = solved(prob)
    C, D = m.solver.C, m.solver.D
    S = K.toarray() - C.toarray() @ np.linalg.solve(D.toarray(), C.T.toarray())
    ref = np.linalg.solve(S, m.f_red)
    assert isinstance(m.solver, jt.BorderedSolver) and U.shape == m.f_red.shape
    assert np.abs(U - ref).max() <= (1e-8 if solver == "pardiso" else 1e-6) * np.abs(ref).max()
    m.solver.free()


# --- the solvers --------------------------------------------------------------------------------------------------


def _compliance(prob, solver):
    from dataclasses import replace
    m = Model(replace(prob, config=replace(prob.config, settings=replace(prob.config.settings, solver=solver))))
    if solver == "amg":
        m.solver.rtol = 1e-12
    c = float(np.sum(m.evaluate(np.full(m.n_design, 0.5), 1.0)[0]))
    m.solver.free()
    return c


@needs_pardiso
def test_pardiso_and_amg_agree_with_a_joint(tmp_path):
    prob = prob_of(tmp_path, h=2.0, clamp=False)
    assert _compliance(prob, "pardiso") == pytest.approx(_compliance(prob, "amg"), rel=1e-6)


@needs_accelerate
def test_cholesky_and_amg_agree_with_a_joint(tmp_path):
    """macOS only (the Accelerate shim): the bordered matrix in the lower triangle, the same answer."""
    prob = prob_of(tmp_path, h=2.0, clamp=False)
    assert _compliance(prob, "cholesky") == pytest.approx(_compliance(prob, "amg"), rel=1e-6)


class FakeAccelerate:
    """The five entry points of the Accelerate shim on scipy `splu`. It checks the input that the real library needs
    (lower triangle only, sorted rows in each column). The numbers of a real Accelerate factor are not tested."""

    def __init__(self):
        self.h, self.calls = {}, []

    @staticmethod
    def _arr(ptr, n, ctype):
        return np.ctypeslib.as_array((ctype * n).from_address(ptr)).copy()

    def _build(self, n, cs, ri, va):
        low = sps.csc_matrix((va, ri, cs), shape=(n, n))
        assert (low.indices >= np.repeat(np.arange(n), np.diff(low.indptr))).all(), "an entry above the diagonal"
        assert all((np.diff(low.indices[low.indptr[j]:low.indptr[j + 1]]) > 0).all() for j in range(n)), "unsorted rows"
        return spla.splu((low + sps.tril(low, -1).T).tocsc())

    def af_factor_ex(self, n, cs_ptr, ri_ptr, va_ptr, order_method, order, bs, status):
        cs = self._arr(cs_ptr, n + 1, ctypes.c_long)
        nnz = int(cs[n])
        ri, va = self._arr(ri_ptr, nnz, ctypes.c_int), self._arr(va_ptr, nnz, ctypes.c_double)
        self.h[len(self.h) + 1] = dict(f=self._build(n, cs, ri, va), n=n, cs=cs, ri=ri)
        status._obj.value = 0
        self.calls.append("factor")
        return len(self.h)

    def af_refactor(self, hid, va_ptr):
        d = self.h[hid]
        d["f"] = self._build(d["n"], d["cs"], d["ri"], self._arr(va_ptr, int(d["cs"][d["n"]]), ctypes.c_double))
        self.calls.append("refactor")
        return 0

    def af_solve(self, hid, n, nrhs, b_ptr):
        b = np.ctypeslib.as_array((ctypes.c_double * (n * nrhs)).from_address(b_ptr))
        b[:] = np.asarray(self.h[hid]["f"].solve(b.reshape(nrhs, n).T.copy())).T.reshape(-1)

    def af_free(self, hid):
        self.h.pop(hid, None)


def test_cholesky_path_runs_on_the_bordered_matrix_with_a_fake_accelerate_library(monkeypatch):
    """The Python side of the cholesky path (lower-triangle pattern of the bordered matrix, scaling, refactor of the same
    pattern, padding) on Linux. The Accelerate factor itself is run only by the Mac test above."""
    fake = FakeAccelerate()
    monkeypatch.setattr(solver_module, "_accel_lib", lambda: fake)
    monkeypatch.setattr(solver_module, "resolve_solver", lambda name: name)
    c = jc.cube(4)
    set_ = jc.make_joint(c, (9.0, 5.0, -6.0), ("round", 2.0), E_beam=1.0, k_a=100.0)
    f = jc.nodal_load(c, np.array([0.0, 5.0, 1.0]), np.array([0.0, 0.0, 2.0]))
    K = jc.kb_added(c, set_)
    s = set_.make_solver("cholesky", c.free, c.shape, c.h, None, True)
    s.factor(K)
    s.factor(K * 1.0)                                              # new index arrays, same pattern: the refactor call
    u = s.solve(f)
    ref, _ = jc.solve(c, set_, f, "splu")
    assert fake.calls == ["factor", "refactor"] and u.shape == f.shape
    assert float(f @ u) == pytest.approx(float(f @ ref), rel=1e-8)
    s.free()


def test_the_amg_rows_of_a_body_move_with_the_part():
    """The augmented near-null space: a rigid motion of the grid and the bodies that it carries is in the null space of the
    bordered matrix (the energy of the bond is zero)."""
    c = jc.cube(4)
    set_ = jc.make_joint(c, (9.0, 5.0, -6.0), ("round", 2.0), E_beam=1.0, k_a=100.0)
    Bg = rigid_body_modes(c.free, c.shape, c.h)
    B = set_.augmented_modes(Bg, c.free, c.shape, c.h)
    assert B.shape == (Bg.shape[0] + 6, 6) and np.allclose(B[:Bg.shape[0]], Bg)
    C, _ = set_.border(c.free)
    K = jc.kb_added(c, set_)
    top = K @ B[:Bg.shape[0]] + C @ B[Bg.shape[0]:]                     # grid rows of A B: the part moves with its body
    assert np.abs(top).max() <= 1e-12 * np.abs(K.toarray()).max()
    bottom = C.T @ B[:Bg.shape[0]] + set_.joints[0].D_bond @ B[Bg.shape[0]:]     # border rows without the beam
    assert np.abs(bottom).max() <= 1e-10 * np.abs(set_.joints[0].D_bond).max()


# --- the rigid-body test, floating parts --------------------------------------------------------------------------


def test_a_joint_alone_holds_the_part_and_the_run_works(tmp_path):
    prob = prob_of(tmp_path, h=2.0, clamp=False)                      # no fixed region, no spring
    assert prob.fixed_dofs.size == 0 and prob.springs is None and prob.joints is not None
    m, K, rho, U = solved(prob)
    c = float(np.einsum("ij,ij->j", m.f_red, U)[0])
    assert np.isfinite(c) and c > 0
    rf = m.reaction_facts(U, rho)["cases"][0]
    assert "clamp" not in rf and rf["force_error"] < 1e-8 and rf["moment_error"] < 1e-8
    assert rf["joints"]["pad"]["part"]["force_n"] == pytest.approx([0.0, 0.0, 2.0], abs=1e-7)    # the joint carries the whole load
    m.solver.free()
    out = tmp_path / "out"
    assert main(run_args(case(sub(tmp_path, "r"), h=2.0, clamp=False), out)) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["supports"]["rigid_body_test"].startswith("skipped") and run["joints"]["bordered_dofs"] == 6


def test_the_rigid_body_test_is_skipped_for_a_joint_with_a_one_axis_spring(tmp_path):
    """A spring on one axis alone leaves modes free (A7a). With a joint the beam blocks all six, so the test does not run."""
    def spring(d):
        d["springs"] = [{"region": "pad", "stiffness": [100.0, 0.0, 0.0]}]
    prob = prob_of(tmp_path, clamp=False, mutate=spring)
    assert prob.springs is not None and prob.joints is not None and "free_modes" not in prob.extended["supports"]
    with pytest.raises(UserError, match="leave"):
        check_rigid_modes(prob.fixed_dofs, prob.shape, prob.h, prob.springs.dofs)      # the face-dof test alone would refuse it


def test_a_region_with_a_joint_and_a_spring_adds_the_face_matrices_and_lists_both(tmp_path):
    def spring(d):
        d["springs"] = [{"region": "pad", "stiffness": [100.0, 100.0, 100.0]}]
    prob = prob_of(tmp_path, h=2.0, mutate=spring)
    m, K, rho, U = solved(prob)
    K0 = m.assembler.K(m.element_E(rho))
    rows, cols, vals = sp.sum_duplicates(*(np.concatenate(p) for p in zip(prob.springs.triplets(), prob.joints.triplets())),
                                         prob.joints.n_dof)
    dense = sps.csr_matrix((vals, (rows, cols)), shape=(prob.joints.n_dof,) * 2)[m.free_dofs][:, m.free_dofs]
    assert abs(K - (K0 + dense)).max() <= 1e-9 * abs(K).max()                      # the two face matrices add on the same nodes
    c = m.reaction_facts(U, rho)["cases"][0]
    assert set(c["springs"]) == {"pad"} and set(c["joints"]) == {"pad"} and c["force_error"] < 1e-8 and c["moment_error"] < 1e-8
    m.solver.free()


def test_two_joints_on_two_regions_balance_the_load(tmp_path):
    def two(d):
        box(tmp_path / "foot.stl", (0, 0, -2), (4, 10, 0))
        d["regions"]["load"]["foot"] = "foot.stl"
        d["joints"].append({"region": "foot", "to": [2, 5, -20], "section": {"kind": "tube", "od": 3, "id": 2}})
    prob = prob_of(tmp_path, h=2.0, clamp=False, mutate=two)
    assert len(prob.joints.joints) == 2 and prob.extended["joints"]["bordered_dofs"] == 12
    m, K, rho, U = solved(prob)
    c = m.reaction_facts(U, rho)["cases"][0]
    assert set(c["joints"]) == {"pad", "foot"} and c["force_error"] < 1e-8 and c["moment_error"] < 1e-8
    total = np.array(c["joints"]["pad"]["part"]["force_n"]) + np.array(c["joints"]["foot"]["part"]["force_n"])
    assert total == pytest.approx([0.0, 0.0, 2.0], abs=1e-7)
    m.solver.free()


# --- reactions ----------------------------------------------------------------------------------------------------


def test_the_reactions_balance_the_load_with_a_clamp_and_a_joint(tmp_path):
    prob = prob_of(tmp_path)
    m, K, rho, U = solved(prob)
    c = m.reaction_facts(U, rho)["cases"][0]
    assert c["force_error"] < 1e-9 and c["moment_error"] < 1e-9 and c["applied"]["force_n"][2] == pytest.approx(-2.0)
    j = c["joints"]["pad"]
    clamp = np.array(c["clamp"]["force_n"])
    assert np.allclose(clamp + np.array(j["nodal"]["force_n"]) + [0, 0, -2.0], 0, atol=1e-8)      # applied + clamp + joint = 0
    assert clamp[2] > 0 and j["part"]["force_n"][2] > 0                                          # both hold the load up
    assert 0.0 < j["bond_energy_share"] < 1e-3
    m.solver.free()


def test_the_wrench_on_the_part_is_the_wrench_of_the_nodal_forces_and_the_ground_reaction_is_the_same_wrench_moved(tmp_path):
    prob = prob_of(tmp_path, clamp=False)
    m, K, rho, U = solved(prob)
    c = m.reaction_facts(U, rho)["cases"][0]["joints"]["pad"]
    jo = prob.joints.joints[0]
    p, g, nodal = c["part"], c["ground"], c["nodal"]
    assert p["about_mm"] == pytest.approx(list(jo.ref)) and g["about_mm"] == pytest.approx([40, 5, 5])
    assert np.allclose(p["force_n"], nodal["force_n"], atol=1e-9)                        # -B q equals the sum of the bond forces
    about_origin = np.array(p["moment_nmm"]) + np.cross(jo.ref - prob.origin, p["force_n"])
    assert np.allclose(about_origin, nodal["moment_nmm"], atol=1e-7)                       # -B q about ref, moved to the grid origin
    assert np.allclose(g["force_n"], p["force_n"])
    assert np.allclose(g["moment_nmm"], np.array(p["moment_nmm"]) + np.cross(jo.ref - jo.far, p["force_n"]), atol=1e-9)
    assert p["force_n"] == pytest.approx([0.0, 0.0, 2.0], abs=1e-7)                       # a floating part: the joint holds the load
    m.solver.free()


def test_the_ground_reaction_of_a_tip_load_is_f_and_f_times_l():
    """A cantilever: a transverse force F at the joint of a floating cube. The ground gives -F and the moment F L about the far point."""
    c = jc.cube(4)
    L, F = 20.0, 5.0
    set_ = jc.make_joint(c, (0.0, L, 0.0), ("round", 2.0), E_beam=1.0, k_a=100.0)
    f = jc.nodal_load(c, np.array([F, 0.0, 0.0]), np.zeros(3))
    u, q = jc.solve(c, set_, f, "splu")
    w = jt.joint_wrenches(set_.joints[0], q)
    # The zero parts of the force: within 1e-6 F, as the other checks here. They are the solve noise of a soft beam (E 1) on a stiff
    # bond: 1.2e-9 N in y on Apple silicon (Accelerate), and 1e-10 to 4e-9 N after steps of iterative refinement. So 1e-9 N is too small.
    assert np.allclose(w["part"]["force_n"], [-F, 0, 0], rtol=1e-6, atol=1e-6 * F)          # the support pushes the part back
    assert np.allclose(w["ground"]["force_n"], [-F, 0, 0], rtol=1e-6, atol=1e-6 * F)
    assert abs(w["ground"]["moment_nmm"][2]) == pytest.approx(F * L, rel=1e-6) and abs(w["part"]["moment_nmm"][2]) < 1e-6 * F * L
    assert w["ground"]["moment_nmm"][2] < 0 < -w["ground"]["moment_nmm"][2] + 1                # about +z for the moment of F on the far side
    assert w["translation_mm"][0] == pytest.approx(F * L ** 3 / (3 * 1.0 * math.pi / 4), rel=1e-3)   # F L^3 / (3 E I), I = pi d^4 / 64 = pi/4


def test_the_bond_share_of_the_energy_follows_the_stiffness_of_the_bond(tmp_path):
    shares = []
    for k in (1e6, 1e7):
        m, K, rho, U = solved(prob_of(sub(tmp_path, f"k{int(k)}"), h=2.0, joint={"bond_stiffness_per_area": k}))
        shares.append(m.reaction_facts(U, rho)["cases"][0]["joints"]["pad"]["bond_energy_share"])
        m.solver.free()
    assert shares[0] / shares[1] == pytest.approx(10.0, rel=0.05)       # a stiff bond: the share falls as 1 / k (measured 9.93)
    assert 0.0 < shares[1] < shares[0] < 1e-5


# --- the gradient -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["auto", "amg"])
def test_gradient_matches_finite_differences_with_a_joint(tmp_path, solver):
    """A stiff beam (E 210000), as in the A4b and A7a checks: the solve noise of a soft beam limits a small step. Measured
    worst relative difference at the step 1e-3: 1.4e-7 (direct), 1.5e-7 (amg). A step of 1e-3 and not 1e-4 for the direct solver: the
    error of the difference quotient from the solve noise goes as 1 / step (review n9: 1.1e-5 at 1e-4 and 6.4e-5 at 1e-5 in a case
    with two joints and a spring), and the margin of 1e-3 is larger."""
    m = Model(prob_of(tmp_path, h=2.0, solver=solver, joint={"E": 210000.0}))
    eps, tol = 1e-3, 1e-5
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13
        tol = 1e-4
    rng = np.random.default_rng(3)
    x = rng.uniform(0.3, 0.7, m.n_design)
    c0, dc, _, _ = m.evaluate(x, 2.0)
    assert np.isfinite(c0) and dc.shape == (m.n_design,)
    floor = 1e-2 * np.abs(dc).max()
    for e in rng.choice(m.n_design, size=min(12, m.n_design), replace=False):
        xp, xm = x.copy(), x.copy()
        xp[e] += eps
        xm[e] -= eps
        fd = (m.evaluate(xp, 2.0)[0] - m.evaluate(xm, 2.0)[0]) / (2 * eps)
        assert abs(fd - dc[e]) < tol * max(abs(dc[e]), floor)
    m.solver.free()


# --- the readers and the run --------------------------------------------------------------------------------------


def test_check_counts_a_joint_region_as_a_support_and_gives_the_facts(tmp_path):
    d = cli.check_data(case(tmp_path, clamp=False))
    assert not any("no fixed voxel" in w for w in d["warnings"])
    assert d["extended"]["joints"]["joints"][0]["faces"] == 180 and d["extended"]["supports"]["axes"] == {}
    lines = extended_lines(d["extended"])
    assert any(l.startswith("joint:") for l in lines) and any(l.startswith("joints:") and "6 border dofs" in l for l in lines)
    json.dumps(d, default=str)


def apart_case(tmp_path, holders):
    """A part where no support touches a design voxel: the design space (x 4..20) has a gap of 2 mm to the fixed plate (x -2..0) and to the
    mount (x 0..2). `holders` is a list of "spring" and "joint" entries on the mount (load region `mount`)."""
    box(tmp_path / "design.stl", (4, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "mount.stl", (0, 0, 0), (2, 10, 10))
    box(tmp_path / "tip.stl", (8, 4, 10), (12, 6, 13))
    d = {"name": "apart", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"mount": "mount.stl", "tip": "tip.stl"}},
         "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
         "settings": {"voxel_size": 1.0, "volume_fraction": 0.3, "iterations": 3, "solver": "amg"}}
    if "spring" in holders:
        d["springs"] = [{"region": "mount", "stiffness": [100.0, 100.0, 100.0]}]
    if "joint" in holders:
        d["joints"] = [{"region": "mount", "to": [-30.0, 5.0, 5.0], "section": {"kind": "round", "d": 4.0}}]
    (tmp_path / "apart.yaml").write_text(yaml.safe_dump(d))
    return tmp_path / "apart.yaml"


@pytest.mark.parametrize("holders, touch_text", [
    ([], "no fixed voxel touches a design voxel"),
    (["spring"], "no fixed voxel and no spring region touches a design voxel"),
    (["joint"], "no fixed voxel and no joint region touches a design voxel"),
    (["spring", "joint"], "no fixed voxel and no spring region and no joint region touches a design voxel"),
])
def test_check_names_what_holds_the_part_when_no_support_touches_a_design_voxel(tmp_path, holders, touch_text):
    """Guard for the merge of A7b with F2a (#53; review m5). Both change the `touches` block of `cli.check_data`, and the `held_by` list
    that it needs is defined in the conflicting hunk. A resolution that drops `held_by` raises a `NameError` here, and no other test has a part
    that no support touches. The start of the text of the `touches` warning is pinned (P1 adds the distance, the meaning and the options
    after it). The warning about the separate pieces names each holder too; its wording is not pinned, because F2a (#53) changes it."""
    d = cli.check_data(apart_case(tmp_path, holders))               # a NameError would stop here
    touch = [x for x in d["warnings"] if "touches a design voxel" in x]
    assert len(touch) == 1 and touch[0].startswith(touch_text + ": the nearest design voxel is ")
    pieces = [x for x in d["warnings"] if "separate pieces" in x or "pieces" in x]
    assert pieces and all(any(h in x for x in pieces) for h in holders)


def test_the_support_mask_has_the_joint_region(tmp_path):
    prob = prob_of(tmp_path)
    mask = sp.support_mask(prob)
    ids = {lr.name: i for i, lr in enumerate(prob.config.regions.load)}
    pad = prob.load_region_ids == ids["pad"]
    assert (mask == ((prob.labels == FIXED) | (pad & (prob.labels == LOAD)))).all() and not mask[prob.load_region_ids == ids["tip"]].any()


def test_a_joint_only_region_is_used_by_the_builder_and_the_peak_scope(tmp_path):
    from test_springs_x_merge import sr
    from test_stress_report import fake_stress
    from topopt.regions_builder import build_in_workspace
    spec = {"name": "mount", "regions": {"design": [{"box": {"min": [0, 0, 0], "max": [30, 10, 10]}}],
                                         "load": {"pad": [{"box": {"min": [30, 0, 0], "max": [34, 10, 10]}}],
                                                  "tip": [{"box": {"min": [10, 3, 10], "max": [14, 7, 13]}}]}},
            "joints": [{"region": "pad", "to": [60, 5, 5], "section": {"kind": "round", "d": 3}}],
            "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2]}]}],
            "settings": {"volume_fraction": 0.3, "voxel_size": 2.0}}
    (tmp_path / "ws").mkdir()
    r = build_in_workspace(spec, tmp_path / "ws")
    assert r["ok"] and r["builder_ok"], r
    doc = yaml.safe_load((tmp_path / "ws" / r["yaml_rel"]).read_text())
    assert doc["regions"]["fixed"] == [] and doc["joints"][0]["region"] == "pad" and r["check"]["extended"]["joints"]["joints"]
    bad = build_in_workspace({**spec, "joints": [dict(spec["joints"][0], region="tip2")]}, tmp_path / "ws", "bad")
    assert bad["ok"] is False and "joints use region 'tip2'" in bad["error"]
    idle = build_in_workspace(dict(spec, joints=None), tmp_path / "ws", "idle")
    assert idle["ok"] is False
    problem = build_problem(load_config(tmp_path / "ws" / r["yaml_rel"]))
    stress, label = fake_stress(problem, {1: 1.0, int(FIXED): 50.0, int(LOAD): 30.0})
    assert (sr.peak_scope(stress, problem) == (label == 1)).all()          # the joint region is left out, as a fixed region


def test_topopt_verify_takes_a_joint_now(tmp_path):
    """K1-A: `topopt verify` models the joint (`verify/supports_p2.py`, `tests/test_verify_supports.py`); the guard no longer refuses it."""
    from topopt.verify.run import _check_loads_and_supports
    _check_loads_and_supports(load_config(case(sub(tmp_path, "y"))))


def test_check_does_not_warn_that_gpu_mg_cannot_take_a_joint(tmp_path, fake_cpu_device):
    """P2b: gpu-mg takes joints (`gpu_mg/border.py`, `tests/test_gpu_mg_border.py`). The refusal texts and `gpu_mg_problem` are gone."""
    d = cli.check_data(case(tmp_path, solver="gpu-mg"))
    assert d["solver"]["resolved"] == "gpu-mg"
    assert not [w for w in d["warnings"] if "cannot start here" in w or "no stiffness matrix to border" in w]
    assert not hasattr(sp, "GPU_MG_JOINT_REFUSAL") and not hasattr(sp, "gpu_mg_problem")


def test_run_with_gpu_mg_does_not_stop_at_the_start_for_a_joint(tmp_path, fake_cpu_device, capsys, monkeypatch):
    """The run gets as far as the solver: a stand-in for `GpuSolver.__init__` (the fake device has no OpenCL behind it) says that it was called. Any
    refusal before it (`check`, `cmd_run`, the old refusal of the joints) would stop the run with its own text and not with this one (review R1, N3)."""
    from topopt.gpu_mg.model import GpuSolver

    def reached(self, *args, **kwargs):
        raise RuntimeError("the solver of gpu-mg was made")
    monkeypatch.setattr(GpuSolver, "__init__", reached)
    out = tmp_path / "out"
    assert main(run_args(case(tmp_path, solver="gpu-mg"), out)) == 2
    err = capsys.readouterr().err
    assert "the solver of gpu-mg was made" in err and "no stiffness matrix to border" not in err and "cannot start here" not in err


def test_the_voxel_buckling_check_is_not_skipped_for_a_joint_run(tmp_path):
    """PR 4: the check runs with the condensed stiffness S (tests/test_buckling_joints.py has the numbers). Here: the preflight and the facts."""
    prob = prob_of(tmp_path, solver="auto")
    info = bv.preflight(prob.config, "pardiso", 1000, 500, 1000)
    assert info["will_run"] is True and info["reason"] is None
    assert any("no geometric stiffness" in f for f in info["facts"])
    info = bv.preflight(prob.config, None, 1000, 500, 1000)       # a solver that cannot start is still a skip, with its own reason
    assert info["will_run"] is False and info["reason"] == "solver_unavailable"


def test_the_refine_step_solves_with_the_joint(tmp_path):
    from topopt.refine.stress import compute_stress
    prob = prob_of(tmp_path, h=2.0, clamp=False)
    m = Model(prob)
    m.solver.rtol = 1e-12
    c_model = float(m.evaluate(np.ones(m.n_design), 1.0)[0])
    m.solver.free()
    res = compute_stress(prob, np.full(int(np.prod(prob.shape)), 255, dtype=np.uint8))
    assert float(res["compliance_cases"][0]) == pytest.approx(c_model, rel=1e-3)


def test_the_run_writes_the_joints_the_reactions_and_the_amg_iterations(tmp_path):
    out = tmp_path / "out"
    assert main(run_args(case(tmp_path, h=2.0), out)) == 0
    run = json.loads((out / "run.json").read_text())
    f = run["joints"]["joints"][0]
    assert f["region"] == "pad" and f["rho"] > 0 and run["joints"]["bond"]["c"] == 30.0 and "source" in run["joints"]["bond"]
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert all("reactions" not in r for r in hist[:-1]) and "reactions" in hist[-1]
    assert all(len(r["amg_iterations"]) == 1 and r["amg_iterations"][0] > 0 for r in hist)         # amg: the CG iterations of each pass
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    j = metrics["reactions"]["cases"][0]["joints"]["pad"]
    assert metrics["reactions"]["cases"][0]["force_error"] < 1e-4 and j["part"]["force_n"][2] > 0 and "ground" in j
    assert metrics["joints"]["amg_iterations"]["passes"] == len(hist) and metrics["joints"]["amg_iterations"]["max"] > 0
    assert metrics["joints"]["joints"][0]["bond_share_max"] == f["bond_share_max"] and metrics["stiffness_cases"][0]["N_per_mm"] > 0
    log = (out / "run.log").read_text()
    assert "joint:          'pad'" in log and "joint frame:" in log and "joint bond:" in log and "reactions:      down:" in log
    assert "joint 'pad'" in log and "ground reaction" in log and "percent of the strain energy" in log


@needs_pardiso
def test_a_direct_run_has_no_amg_iterations_and_the_reactions_balance(tmp_path):
    out = tmp_path / "out"
    assert main(run_args(case(tmp_path, h=2.0, solver="pardiso", clamp=False), out)) == 0
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert all("amg_iterations" not in r for r in hist)
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    assert "amg_iterations" not in metrics["joints"] and metrics["reactions"]["cases"][0]["force_error"] < 1e-9


def test_a_resume_of_a_finished_joint_run_keeps_the_reactions(tmp_path):
    p = case(tmp_path, h=2.0)
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    history = (out / "history.jsonl").read_bytes()
    before = json.loads((out / "result" / "metrics.json").read_text())["reactions"]
    assert main(run_args(p, out, "--resume")) == 0
    assert (out / "history.jsonl").read_bytes() == history
    assert json.loads((out / "result" / "metrics.json").read_text())["reactions"] == before


def test_a_resume_after_a_stop_rule_continues_with_the_bordered_warm_start(tmp_path):
    """The checkpoint holds the AMG warm start of the bordered size; the same class restores it."""
    def rules(d):
        d["settings"].update(iterations=8, stop_rules={"time_budget_s": 0.05})
    p = case(tmp_path, h=2.0, mutate=rules)
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    assert json.loads((out / "verdict.json").read_text())["state"] == "stopped_by_rule"
    ck = np.load(out / "checkpoint.npz")
    assert any(k.endswith("x_prev") for k in ck.files) and ck[[k for k in ck.files if k.endswith("x_prev")][0]].shape[0] == \
        Model(build_problem(load_config(p))).n_free + 6
    assert main(run_args(p, out, "--resume")) == 0


def test_a_changed_joint_is_a_changed_input_and_the_feedback_comparison_sees_it(tmp_path):
    from topopt import feedback
    p = case(tmp_path, h=2.0)
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    run = json.loads((out / "run.json").read_text())
    doc = yaml.safe_load(p.read_text())
    diffs, compared, _ = feedback._yaml_vs_run(run, doc)
    assert not [d for d in diffs if d["key"].startswith("joints")] and "joints (region, to, section)" in compared
    doc["joints"][0]["to"] = [40, 5, 9]
    assert [d["key"] for d in feedback._yaml_vs_run(run, doc)[0] if d["key"].startswith("joints")] == ["joints[pad].to"]
    doc["joints"][0]["to"] = [40, 5, 5]
    doc["joints"][0]["section"] = {"kind": "round", "d": 5.0}
    assert [d["key"] for d in feedback._yaml_vs_run(run, doc)[0] if d["key"].startswith("joints")] == ["joints[pad].section"]
    doc["joints"] = []
    assert [d["key"] for d in feedback._yaml_vs_run(run, doc)[0] if d["key"].startswith("joints")] == ["joints[pad]"]
    assert feedback._settings(run)["joints"][0]["region"] == "pad"


def test_robust_mode_gives_the_joint_reactions_of_the_nominal_design(tmp_path):
    p = case(tmp_path, h=2.0, mutate=lambda d: d["settings"].update(robust="eroded"))
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    assert metrics["reactions"]["cases"][0]["force_error"] < 1e-4 and "pad" in metrics["reactions"]["cases"][0]["joints"]


def test_a_mirror_run_keeps_the_joint_and_its_reactions(tmp_path):
    """A4a mirror: no refusal. The pad case is symmetric about y = 5, so the joint is symmetric too and carries no force along y."""
    p = case(tmp_path, h=2.0, mutate=lambda d: d.update(mirror=[{"normal": "y", "at_mm": 5.0}]))
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    c = json.loads((out / "result" / "metrics.json").read_text())["reactions"]["cases"][0]
    assert c["force_error"] < 1e-4 and abs(c["joints"]["pad"]["part"]["force_n"][1]) < 1e-6 * abs(c["joints"]["pad"]["part"]["force_n"][0] + 1)


def test_the_texts_name_the_joints():
    contracts = (REPO / "CONTRACTS.md").read_text()
    for word in ("## 38.", "BorderedSolver", "bond_stiffness_per_area", "rho", "Euler-Bernoulli", "ground reaction", "gpu-mg",
                 "Rigid-body test", "buckling"):
        assert word in contracts, word
    for skill in (".claude/skills/part-to-topopt/SKILL.md", ".agents/skills/part-to-topopt/SKILL.md"):
        text = (REPO / skill).read_text()
        assert "joints" in text and "section" in text and "to:" in text
    assert "joints" in (REPO / "docs" / "reference" / "problem-and-run.md").read_text()
    from topopt_mcp import server
    assert "joints" in server.build_regions.__doc__
