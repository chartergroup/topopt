"""A7a: springs on load regions (CONTRACTS.md section 35).

The pad case of `case()`: design x 0..20, y 0..10, z 0..10; clamp plate x -2..0; spring pad x 20..22 (a load region
with a spring and no force); tip box x 8..12, y 4..6, z 10..13 with the force. At voxel size 1 the pad has
10 x 10 faces on its free end and 4 x 20 on its sides: 180 faces, 180 mm2.
"""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps
import trimesh
import yaml
from helpers import needs_accelerate, needs_pardiso

from topopt import springs as sp
from topopt.config import UserError, load_config
from topopt.filter import simp_E
from topopt.loads import check_rigid_modes
from topopt.optimize import E_MIN, Model
from topopt.problem import build_problem
from topopt.voxelize import EMPTY, KEEP_OUT

REPO = Path(__file__).resolve().parents[1]


def box(path, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def case(tmp_path, h=1.0, mutate=None, clamp=True, k=(100.0, 100.0, 100.0), solver="amg"):
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "pad.stl", (20, 0, 0), (22, 10, 10))
    box(tmp_path / "tip.stl", (8, 4, 10), (12, 6, 13))
    d = {
        "name": "padcase",
        "regions": {"design": "design.stl", "fixed": ["plate.stl"] if clamp else [],
                    "load": {"pad": "pad.stl", "tip": "tip.stl"}},
        "springs": [{"region": "pad", "stiffness": list(k)}],
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


def sub(tmp_path, name):
    (tmp_path / name).mkdir()
    return tmp_path / name


def dense_kb(prob, name, k_total):
    """Independent Kb on grid dofs: loops over the faces of the region and the explicit 4 x 4 matrix h^2/36 [4 2 2 1 ...]."""
    ids = prob.load_region_ids
    idx = [lr.name for lr in prob.config.regions.load].index(name)
    ny1, nz1 = prob.shape[1] + 1, prob.shape[2] + 1
    faces = []
    for ijk in np.argwhere(ids == idx):
        for axis in range(3):
            for sign in (-1, 1):
                nb = ijk.copy()
                nb[axis] += sign
                if not (0 <= nb[axis] < prob.shape[axis]) or prob.labels[tuple(nb)] in (EMPTY, KEEP_OUT):
                    faces.append((ijk, axis, sign))
    M = np.array([[4, 2, 2, 1], [2, 4, 1, 2], [2, 1, 4, 2], [1, 2, 2, 4]]) / 36.0
    rows, cols, vals = [], [], []
    for ijk, axis, sign in faces:
        a, b = [d for d in range(3) if d != axis]
        nodes = []
        for da in (0, 1):
            for db in (0, 1):
                c = ijk.copy()
                c[axis] += 1 if sign > 0 else 0
                c[a] += da
                c[b] += db
                nodes.append((c[0] * ny1 + c[1]) * nz1 + c[2])
        for d in range(3):
            for p in range(4):
                for q in range(4):
                    rows.append(3 * nodes[p] + d)
                    cols.append(3 * nodes[q] + d)
                    vals.append(k_total[d] / len(faces) * M[p, q] * prob.h ** 2)
    n = 3 * int(np.prod(np.array(prob.shape) + 1))
    return sps.csr_matrix((vals, (rows, cols)), shape=(n, n)), len(faces)


def tool(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the YAML ----------------------------------------------------------------------------


def test_a_plain_yaml_has_no_springs(tmp_path):
    cfg = load_config(case(tmp_path, mutate=lambda d: d.pop("springs")))
    assert cfg.springs == () and not cfg.has_springs() and not cfg.uses_extended_loads()
    prob = build_problem(cfg)
    assert prob.springs is None and prob.extended is None
    m = Model(prob)
    assert m.kb is None
    m.solver.free()


def test_a_spring_region_with_no_force_is_valid_and_both_forms_load(tmp_path):
    cfg = load_config(case(tmp_path))
    assert cfg.has_springs() and cfg.uses_extended_loads()
    assert cfg.springs[0].stiffness == (100.0, 100.0, 100.0) and cfg.springs[0].per_area is None
    assert not any(f.region == "pad" for f in cfg.load_cases[0].forces)       # no force on the spring region
    one = load_config(case(sub(tmp_path, "a"), mutate=lambda d: d["springs"][0].update(
        {"stiffness_per_area": 2.5}) or d["springs"][0].pop("stiffness")))
    assert one.springs[0].per_area == (2.5, 2.5, 2.5)


@pytest.mark.parametrize("change, words", [
    (lambda d: d["springs"][0].update(stiffness_per_area=[1, 1, 1]), "has both"),
    (lambda d: d["springs"][0].pop("stiffness"), "has neither"),
    (lambda d: d["springs"][0].update(region="nope"), "not a load region"),
    (lambda d: d["springs"][0].update(stiffness=[1, -1, 1]), "must not be negative"),
    (lambda d: d["springs"][0].update(stiffness=[0, 0, 0]), "all zero"),
    (lambda d: d["springs"][0].update(stiffness=[1, 1]), "list of 3 numbers"),
    (lambda d: d["springs"][0].update(stiffness=[1, 1, 1, 1, 1, 1]), "rotational spring"),
    (lambda d: d["springs"][0].update(rotational_stiffness=[1, 1, 1]), "rotational spring"),
    (lambda d: d["springs"][0].update(krx=5.0), "rotational spring"),
    (lambda d: d["springs"][0].update(color="red"), "unknown key 'color'"),
    (lambda d: d["springs"].append({"region": "pad", "stiffness": [1, 1, 1]}), "again: one entry for each region"),
    (lambda d: d.update(springs={"pad": 1}), "must be a list"),
])
def test_a_bad_spring_entry_stops_with_the_fact(tmp_path, change, words):
    with pytest.raises(UserError) as e:
        load_config(case(tmp_path, mutate=change))
    assert words in str(e.value) and "(file:" in str(e.value)


def test_neither_a_fixed_region_nor_a_spring_names_the_modes_and_the_options(tmp_path):
    def empty(d):
        d["regions"]["fixed"] = []
        d.pop("springs")
    def missing(d):
        d["regions"].pop("fixed")
        d.pop("springs")
    for start, mutate in (("needs at least one STL file", empty), ("missing key 'fixed'", missing)):
        with pytest.raises(UserError) as e:
            load_config(case(sub(tmp_path, start[:4]), mutate=mutate))
        text = str(e.value)
        assert start in text and "rotation about x, y, z" in text and "Options:" in text and "springs" in text
    with pytest.raises(UserError, match="translation along x"):
        check_rigid_modes(np.zeros(0, dtype=np.int64), (4, 4, 4), 1.0)


def test_a_spring_region_with_no_free_face_is_refused(tmp_path):
    def inner(d):
        box(tmp_path / "core.stl", (8, 3, 3), (12, 7, 7))        # inside the design: every face looks at a design voxel
        d["regions"]["load"]["core"] = "core.stl"
        d["springs"].append({"region": "core", "stiffness": [1, 1, 1]})
    with pytest.raises(UserError, match="no face of the region looks toward empty space"):
        prob_of(tmp_path, mutate=inner)


# --- the face matrix and the seam -------------------------------------------------------------


def test_the_face_matrix_is_consistent():
    M = sp.FACE_UNIT
    assert np.allclose(M, M.T) and M.sum() == pytest.approx(1.0) and np.allclose(M.sum(axis=1), 0.25)
    assert np.linalg.eigvalsh(M).min() > 0
    x = np.array([0.0, 0.0, 1.0, 1.0])           # u = x on the unit square: the energy is the integral of x^2 = 1/3
    assert x @ M @ x == pytest.approx(1.0 / 3.0)


def test_kb_is_added_in_the_pattern_and_equals_the_independent_matrix(tmp_path):
    prob = prob_of(tmp_path)
    m = Model(prob)
    rho = np.full(m.n_design, 0.6)
    K = m.stiffness(rho)
    Ee = np.full(m.n_el, m.E)
    Ee[m.design_pos] = m.E * simp_E(rho, m.penal, 1.0, E_MIN)
    K0 = m.assembler.K(Ee)
    Kb, n_faces = dense_kb(prob, "pad", (100.0, 100.0, 100.0))
    ref = K0 + Kb[m.free_dofs][:, m.free_dofs]
    assert n_faces == 180
    assert abs(K - ref).max() <= 1e-12 * abs(ref).max()
    assert np.shares_memory(K.indices, K0.indices) and np.shares_memory(K.indptr, K0.indptr)     # the pattern is the assembler's
    m.solver.free()


def test_the_total_of_the_faces_is_the_stiffness(tmp_path):
    prob = prob_of(tmp_path, k=(30.0, 0.0, 70.0))
    reg = prob.springs.regions[0]
    for d, k in enumerate((30.0, 0.0, 70.0)):
        assert reg.vals[reg.rows % 3 == d].sum() == pytest.approx(k, rel=1e-13)
    f = prob.extended["springs"]["regions"][0]
    assert f["faces"] == 180 and f["area_mm2"] == 180.0 and f["stiffness_per_area_n_per_mm3"][0] == pytest.approx(30.0 / 180)
    assert set(prob.springs.dofs % 3) == {0, 2}          # no spring on y: no y dof carries one


def test_total_and_per_area_give_the_same_matrix(tmp_path):
    a = prob_of(sub(tmp_path, "a"), k=(90.0, 90.0, 90.0)).springs.triplets()
    b = prob_of(sub(tmp_path, "b"), mutate=lambda d: d["springs"][0].update(
        stiffness_per_area=[0.5, 0.5, 0.5]) or d["springs"][0].pop("stiffness")).springs.triplets()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.abs(a[2] - b[2]).max() <= 1e-14 * np.abs(a[2]).max()


# --- analytic values ---------------------------------------------------------------------------


def test_series_spring_compliance_is_analytic():
    got, ref = tool("spring_check").series()
    assert got == pytest.approx(ref, rel=1e-10)
    got, ref = tool("spring_check").series(kx=1e9)         # a very stiff spring: the clamped bar plus F^2 / k
    assert got == pytest.approx(ref, rel=1e-8)


@pytest.mark.parametrize("n, bound", [(2, 0.2), (4, 0.06), (8, 0.02)])
def test_a_cantilever_on_a_face_spring_matches_beam_theory(n, bound):
    """Tip deflection against beam theory with a Winkler root: measured -0.095, -0.027, -0.007 percent (a lumped
    face rule gives -33, -11, -3 percent). The bound is about 2 times the measured error."""
    tool_ = tool("spring_check")
    got, ref = tool_.cantilever(n)
    assert abs(100 * (got / ref - 1)) < bound
    if n == 4:        # the lumped rule over-stiffens the rotation of the support by 1 + 2 / n^2
        lump, _ = tool_.cantilever(n, lumped=True)
        assert 100 * (lump / ref - 1) < -9.0


def _rigid_case(tmp_path, h, spring):
    """A body on springs only (no fixed region). A force along x on the axis of the body: it translates."""
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "pad.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "cap.stl", (16, 4, 4), (20, 6, 6))
    d = {"name": "rigid", "regions": {"design": "design.stl", "fixed": [], "load": {"pad": "pad.stl", "cap": "cap.stl"}},
         "springs": [{"region": "pad", **spring}],
         "load_cases": [{"name": "pull", "forces": [{"region": "cap", "force": [2.0, 0, 0]}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.5, "iterations": 1, "solver": "amg"}}
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(d))
    prob = build_problem(load_config(tmp_path / "r.yaml"))
    m = Model(prob)
    m.solver.rtol = 1e-12
    c = float(m.evaluate(np.ones(m.n_design), 1.0)[0])
    m.solver.free()
    return c, prob


# Two AMG-CG solves of one matrix give two numbers that differ a little. PyAMG starts the estimate of the spectral radius of its smoother
# from a random vector, so each solve gets its own multigrid and CG stops at its own point. The body is stiff and the springs are weak, so a
# residual of 1e-12 shows as about 1e-9 in the compliance. Measured on a busy host, 840 pairs: the largest gap of the two compliances was
# 9.5e-10 (voxel size 0.5; 5.2e-10 at 1.0), with 1 or 2 solver threads alike. The bound has a margin of 10 over that gap. It is far below
# the bound of the total stiffness (5e-4).
SOLVE_NOISE = 1e-8


@pytest.mark.parametrize("h", [1.0, 0.5])
def test_a_body_on_springs_only_has_the_compliance_of_the_total_stiffness(tmp_path, h):
    """c = F^2 / k for a stiff body: the total k is the number, whatever the voxel size and the number of faces. The
    elastic part is F^2 L / (E A) = 6e-4 of it. Both forms, at two voxel sizes."""
    k = 0.5
    c, prob = _rigid_case(sub(tmp_path, "t"), h, {"stiffness": [k, k, k]})
    assert c == pytest.approx(4.0 / k, rel=5e-4)
    area = prob.extended["springs"]["regions"][0]["area_mm2"]
    c2, _ = _rigid_case(sub(tmp_path, "a"), h, {"stiffness_per_area": [k / area] * 3})
    assert c2 == pytest.approx(c, rel=SOLVE_NOISE)


def test_the_bound_of_the_two_forms_sees_a_total_stiffness_that_is_1e_6_higher(tmp_path):
    """The bound of the test above is not too wide: a total k that is 1e-6 higher (100 times the bound) gives another compliance."""
    k = 0.5
    c, prob = _rigid_case(sub(tmp_path, "t"), 1.0, {"stiffness": [k, k, k]})
    area = prob.extended["springs"]["regions"][0]["area_mm2"]
    c2, _ = _rigid_case(sub(tmp_path, "a"), 1.0, {"stiffness_per_area": [k * (1 + 1e-6) / area] * 3})
    assert c2 != pytest.approx(c, rel=SOLVE_NOISE)
    assert c2 / c - 1 == pytest.approx(-1e-6, rel=0.05)          # the compliance follows the stiffness: c is F^2 / k


# --- the rigid-body test ------------------------------------------------------------------------


def test_springs_count_as_supports_in_the_rigid_body_test(tmp_path):
    prob = prob_of(tmp_path, clamp=False)                     # springs on three axes: all six modes blocked
    s = prob.extended["supports"]
    assert s["free_modes"] == 0 and s["fixed_dofs"] == 0 and s["spring_dofs"] > 0
    assert prob_of(sub(tmp_path, "c")).extended["supports"]["free_modes"] == 0         # a clamp and springs
    with pytest.raises(UserError) as e:
        prob_of(sub(tmp_path, "x"), clamp=False, k=(100.0, 0.0, 0.0))
    text = str(e.value)
    assert "supports and springs leave 3 of the 6" in text and "translation along y" in text and "translation along z" in text
    assert "rotation about x" in text and "Options:" in text


def test_a_roller_and_springs_together_block_the_modes(tmp_path):
    def roller(d):
        d["regions"]["fixed"] = [{"file": "plate.stl", "axes": ["y"]}]
    prob = prob_of(tmp_path, mutate=roller, k=(100.0, 0.0, 100.0))
    assert prob.extended["supports"]["free_modes"] == 0
    with pytest.raises(UserError, match="leave"):                 # the roller alone does not
        check_rigid_modes(prob.fixed_dofs, prob.shape, prob.h)


# --- reactions and equilibrium ------------------------------------------------------------------


def _solve(prob, rtol=1e-12):
    m = Model(prob)
    m.solver.rtol = rtol
    rho = np.full(m.n_design, 0.7)
    K = m.stiffness(rho)
    U = m.solve(K)
    return m, U


def test_the_reactions_balance_the_load_with_a_clamp_and_a_spring(tmp_path):
    def two_loads(d):          # a load region on the clamp: its force on the fixed nodes is lost, the clamp takes it
        box(tmp_path / "root.stl", (0, 4, 4), (2, 6, 6))
        d["regions"]["load"]["root"] = "root.stl"
        d["load_cases"][0]["forces"].append({"region": "root", "force": [0, 0, -1.0]})
    prob = prob_of(tmp_path, mutate=two_loads)
    m, U = _solve(prob)
    rf = m.reaction_facts(U, np.full(m.n_design, 0.7))     # the density of _solve
    c = rf["cases"][0]
    assert c["force_error"] < 1e-9 and c["moment_error"] < 1e-9
    assert c["applied"]["force_n"][2] == pytest.approx(-3.0)
    # the spring reaction from the independent matrix, the clamp reaction from the balance
    Kb, _ = dense_kb(prob, "pad", (100.0, 100.0, 100.0))
    u = np.zeros(Kb.shape[0])
    u[m.free_dofs] = U[:, 0]
    spring = -(Kb @ u).reshape(-1, 3).sum(axis=0)
    assert np.allclose(c["springs"]["pad"]["force_n"], spring, rtol=1e-8, atol=1e-10)
    assert np.allclose(c["clamp"]["force_n"], -(np.array([0, 0, -3.0]) + spring), rtol=1e-8, atol=1e-10)
    assert c["clamp"]["force_n"][2] > 0 and spring[2] > 0                       # both hold the load up
    m.solver.free()


def test_the_reactions_with_springs_only_equal_the_load(tmp_path):
    m, U = _solve(prob_of(tmp_path, clamp=False))
    c = m.reaction_facts(U, np.full(m.n_design, 0.7))["cases"][0]
    assert "clamp" not in c and c["force_error"] < 1e-9 and c["moment_error"] < 1e-9
    assert c["springs"]["pad"]["force_n"] == pytest.approx([0.0, 0.0, 2.0], abs=1e-8)
    m.solver.free()


# --- the gradient and the solvers ----------------------------------------------------------------


@pytest.mark.parametrize("solver", ["auto", "amg"])      # auto: the direct solver of the host; amg: always, with the tight solve
def test_gradient_matches_finite_differences_with_springs(tmp_path, solver):
    m = Model(prob_of(tmp_path, h=2.0, solver=solver))
    eps, tol = 1e-4, 1e-5            # c is about 0.004 here: a step of 1e-6 would be below the solve noise (pitfall fd-check-noise-floor)
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13        # the check needs a tight solve (KB pitfall fd-check-noise-floor)
        eps, tol = 1e-3, 1e-4
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


def _compliance(prob, solver):
    cfg = replace(prob.config, settings=replace(prob.config.settings, solver=solver))
    m = Model(replace(prob, config=cfg))
    if solver == "amg":
        m.solver.rtol = 1e-12
    c = float(np.sum(m.evaluate(np.full(m.n_design, 0.5), 1.0)[0]))
    m.solver.free()
    return c


@needs_pardiso
def test_pardiso_and_amg_agree_with_springs(tmp_path):
    prob = prob_of(tmp_path, h=2.0, clamp=False)
    assert _compliance(prob, "pardiso") == pytest.approx(_compliance(prob, "amg"), rel=1e-6)


@needs_accelerate
def test_cholesky_and_amg_agree_with_springs(tmp_path):
    """macOS only (the Accelerate shim): K + Kb in the lower triangle, the same pattern, the same answer."""
    prob = prob_of(tmp_path, h=2.0, clamp=False)
    assert _compliance(prob, "cholesky") == pytest.approx(_compliance(prob, "amg"), rel=1e-6)


# --- the readers --------------------------------------------------------------------------------


def test_check_counts_a_spring_region_as_a_support_and_gives_the_facts(tmp_path):
    from topopt.cli import check_data
    from topopt.loads import extended_lines
    d = check_data(case(tmp_path, clamp=False))
    assert not any("no fixed voxel" in w for w in d["warnings"])
    assert d["extended"]["springs"]["regions"][0]["faces"] == 180
    assert d["extended"]["supports"]["free_modes"] == 0 and d["extended"]["supports"]["axes"] == {}
    lines = extended_lines(d["extended"])
    assert any(l.startswith("spring:") and "180 faces" in l and "N/mm in total" in l and "solid" in l for l in lines)
    assert any(l.startswith("supports:") and "6 of 6 blocked" in l for l in lines)
    json.dumps(d, default=str)
    tiny = check_data(case(sub(tmp_path, "t"), k=(1e-9, 1e-9, 1e-9)))
    assert any("nearly singular" in w for w in tiny["warnings"])
    assert not any("nearly singular" in w for w in d["warnings"])


def test_a_spring_only_region_is_used_by_the_builder(tmp_path):
    from topopt.regions_builder import build_in_workspace
    spec = {"name": "mount", "regions": {"design": [{"box": {"min": [0, 0, 0], "max": [30, 10, 10]}}],
                                         "load": {"pad": [{"box": {"min": [30, 0, 0], "max": [34, 10, 10]}},],
                                                  "tip": [{"box": {"min": [10, 3, 10], "max": [14, 7, 13]}}]}},
            "springs": [{"region": "pad", "stiffness": [100, 100, 100]}],
            "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2]}]}],
            "settings": {"volume_fraction": 0.3, "voxel_size": 2.0}}
    (tmp_path / "ws").mkdir()
    r = build_in_workspace(spec, tmp_path / "ws")
    assert r["ok"] and r["builder_ok"], r
    doc = yaml.safe_load((tmp_path / "ws" / r["yaml_rel"]).read_text())
    assert doc["regions"]["fixed"] == [] and doc["springs"][0]["region"] == "pad"
    assert not any("fixed" in w for w in r["warnings"]) and r["check"]["extended"]["springs"]["regions"]
    bad = build_in_workspace({**spec, "springs": [{"region": "tip2", "stiffness": [1, 1, 1]}]}, tmp_path / "ws", "bad")
    assert bad["ok"] is False and "springs use region 'tip2'" in bad["error"]
    idle = dict(spec, springs=None)       # without the spring the pad has no force: the old refusal stays
    assert build_in_workspace(idle, tmp_path / "ws", "idle")["ok"] is False


def test_topopt_verify_takes_springs_now(tmp_path):
    """K1-A: `topopt verify` models the spring (`verify/supports_p2.py`, `tests/test_verify_supports.py`); the guard no longer refuses it."""
    from topopt.verify.run import _check_loads_and_supports
    _check_loads_and_supports(load_config(case(sub(tmp_path, "y"))))


def test_the_run_writes_the_springs_and_the_reactions(tmp_path):
    from topopt.cli import main
    p = case(tmp_path, h=2.0)
    out = tmp_path / "out"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["springs"]["regions"][0]["region"] == "pad" and run["supports"]["free_modes"] == 0
    hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    assert all("reactions" not in r for r in hist[:-1]) and "reactions" in hist[-1]
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    assert metrics["reactions"]["cases"][0]["force_error"] < 1e-4
    assert metrics["stiffness_cases"][0]["N_per_mm"] > 0               # plain forces: |F|^2 / c still holds
    log = (out / "run.log").read_text()
    assert "spring:         'pad'" in log and "reactions:      down:" in log and "spring 'pad'" in log
    plain = tmp_path / "plain"
    plain.mkdir()
    assert main(["run", str(case(plain, h=2.0, mutate=lambda d: d.pop("springs")), ), "--out", str(plain / "o"), "--quiet"]) == 0
    hist = [json.loads(l) for l in (plain / "o" / "history.jsonl").read_text().splitlines()]
    assert all("reactions" not in r for r in hist) and "reactions" not in json.loads((plain / "o/result/metrics.json").read_text())
    assert "springs" not in json.loads((plain / "o/run.json").read_text())


def test_the_refine_step_solves_with_the_springs(tmp_path):
    from topopt.refine.stress import compute_stress
    prob = prob_of(tmp_path, h=2.0, clamp=False)
    m = Model(prob)
    m.solver.rtol = 1e-12
    c_model = float(m.evaluate(np.ones(m.n_design), 1.0)[0])
    m.solver.free()
    res = compute_stress(prob, np.full(int(np.prod(prob.shape)), 255, dtype=np.uint8))
    assert float(res["compliance_cases"][0]) == pytest.approx(c_model, rel=1e-3)


def test_the_texts_name_the_springs():
    contracts = (REPO / "CONTRACTS.md").read_text()
    for word in ("## 35.", "stiffness_per_area", "consistent", "solid", "nearly singular", "gpu-mg", "feedback.py"):
        assert word in contracts, word
    assert "no spring support" not in contracts
    skill = (REPO / ".claude/skills/part-to-topopt/SKILL.md").read_text()
    assert "springs" in skill and "stiffness" in skill
    from topopt_mcp import server
    assert "springs" in server.build_regions.__doc__
