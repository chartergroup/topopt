"""The linear buckling check of a run with joints (CONTRACTS.md sections 33 and 38): the eigen problem uses S = K + Kb - C D^-1 C^T.

The run holds the factor of the bordered matrix `B = [[K + Kb, C], [C^T, D]]`; its grid block inverse is `S^-1`. The check gives `eigsh`
`M = S` (a `LinearOperator`) and `Minv` = the grid block of the bordered solve. The beams have no geometric stiffness in this model.
The pad case of `test_joints.py` (h = 2 mm, 1206 free dofs) is the model.
"""
import json

import numpy as np
import pytest
import scipy.linalg as sla

from helpers import needs_direct_solver
from test_joints import case as joint_case
from test_joints import prob_of
from topopt import buckling as bk
from topopt import buckling_voxel as bv
from topopt.cli import main
from topopt.config import load_config
from topopt.joints import condensed_border, support_matrix
from topopt.problem import build_problem


def gray(prob, seed=1):
    """A gray density (0.3 to 1) for the snapshot: no symmetric (repeated) modes. The `gray` density rule keeps it."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.3, 1.0, int(np.prod(prob.shape))) * 255).astype(np.uint8)


def dense_reference(prob, snap, density, modes):
    """Load factors from `scipy.linalg.eigh` of the whole bordered matrix against `-Kg` padded with zeros on the beam dofs."""
    model = bv.build_model(prob, snap, density)
    K = model.asm.K(model.Ee)
    support_matrix(prob, model.asm, model.free).add_to(K)
    C, D = prob.joints.border(model.free)
    n, nb = C.shape
    B = np.block([[K.toarray(), C.toarray()], [C.toarray().T, D.toarray()]])
    f = np.ascontiguousarray(prob.forces[:, model.free].T)
    out = []
    for i in range(f.shape[1]):
        u = np.linalg.solve(B, np.vstack([f[:, i:i + 1], np.zeros((nb, 1))]))[:n, 0]
        u_all = np.zeros(model.n_dof_all)
        u_all[model.free] = u
        Kg = bv.hex8_kg(model.asm, model.edof, u_all, model.h, model.E, model.nu, model.weight).toarray()
        Kgp = np.zeros((n + nb, n + nb))
        Kgp[:n, :n] = Kg
        out.append(1.0 / sla.eigh(-Kgp, B, eigvals_only=True)[::-1][:modes])
    return out


@needs_direct_solver
@pytest.mark.parametrize("density", ["printed", "gray"])
def test_the_check_with_a_joint_matches_the_dense_bordered_problem(tmp_path, density):
    """(a) the check against (b) the dense generalized eigenproblem of the whole bordered matrix. A beam of E 210000 (the bond has a share).
    Measured (Apple Cholesky): relative difference of the lowest 4 factors at most 1.3e-14 (printed) and 2.4e-14 (gray) for the cases below.
    Bar 1e-8, as the tolerance of ARPACK (`bk.TOL`) asks."""
    prob = prob_of(tmp_path, h=2.0, solver="auto", joint={"E": 210000.0})
    snap = gray(prob)
    res = bv.check_problem(prob, snap, density=density, modes=4)
    assert res["status"] == "done" and res["eigen_solver"] == bk.EIGEN_SOLVER_JOINTS and "skipped" not in res["status"]
    ref = dense_reference(prob, snap, density, 4)
    for rec, lam in zip(res["cases"], ref):
        assert rec["status"] == bk.OK and len(rec["load_factors"]) == 4
        rel = np.abs(np.array(rec["load_factors"]) - lam) / lam
        print("relative difference", density, rel.max())
        assert rel.max() < 1e-8
        assert rec["residual_max"] < 1e-8


@needs_direct_solver
def test_the_condensed_stiffness_is_the_matrix_that_the_bordered_solve_inverts(tmp_path):
    prob = prob_of(tmp_path, h=2.0, solver="auto", joint={"E": 210000.0})
    snap = gray(prob)
    model = bv.build_model(prob, snap)
    K = model.asm.K(model.Ee)
    support_matrix(prob, model.asm, model.free).add_to(K)
    S = condensed_border(prob.joints, model.free).condense(K)
    f = np.ascontiguousarray(prob.forces[:, model.free].T)
    from topopt.optimize import make_solver
    solver, _ = make_solver("auto", model.free, prob.shape, prob.h, prob.joints)
    solver.factor(K)
    u = solver.solve(f)[:, 0]
    solver.free()
    assert np.linalg.norm(S @ u - f[:, 0]) / np.linalg.norm(f) < 1e-12           # S u = f
    assert np.linalg.norm(K @ u - f[:, 0]) / np.linalg.norm(f) > 1e-3             # K u = f is false: K is the wrong matrix
    Sd = S.toarray()
    assert np.abs(Sd - Sd.T).max() / np.abs(Sd).max() < 1e-12
    v = np.random.default_rng(0).standard_normal(S.shape[0])
    assert np.linalg.norm(S @ v - Sd @ v) / np.linalg.norm(Sd @ v) < 1e-12


@needs_direct_solver
def test_the_dense_path_of_buckling_modes_builds_s_densely(tmp_path):
    """`ncv >= n` takes the dense solver: `K.toarray()` of the operator is S, and the pairs are the same as ARPACK gives."""
    prob = prob_of(tmp_path, h=2.0, solver="auto", joint={"E": 210000.0})
    snap = gray(prob)
    model = bv.build_model(prob, snap)
    K = model.asm.K(model.Ee)
    support_matrix(prob, model.asm, model.free).add_to(K)
    S = condensed_border(prob.joints, model.free).condense(K)
    from topopt.optimize import make_solver
    solver, _ = make_solver("auto", model.free, prob.shape, prob.h, prob.joints)
    solver.factor(K)
    f = np.ascontiguousarray(prob.forces[:, model.free].T)
    u_all = np.zeros(model.n_dof_all)
    u_all[model.free] = solver.solve(f)[:, 0]
    Kg = bv.hex8_kg(model.asm, model.edof, u_all, model.h, model.E, model.nu, model.weight)
    arpack = bk.buckling_modes(Kg, S, solver, k=4)
    dense = bk.buckling_modes(Kg, S, solver, k=4, ncv=S.shape[0])
    solver.free()
    assert arpack["method"] == "arpack" and dense["method"] == "dense"
    rel = np.abs(arpack["lam"] - dense["lam"]) / dense["lam"]
    print("dense against arpack", rel.max())
    assert rel.max() < 1e-8 and dense["residual"].max() < 1e-10


@needs_direct_solver
def test_a_very_stiff_short_beam_gives_the_factors_of_a_part_that_is_fixed_at_the_joint(tmp_path):
    """Limit test. The joint: a beam of 2.4 mm (to x = 24; the reference point is at x = 21.56), round d 10 mm, E 2.1e9 MPa (1e4 times the
    part), and no other support. The reference: the same part with the pad as a fixed region and no joint.
    Measured at h = 2 mm: lowest factor 3904.49 against 3912.52 (0.21 percent). With a bond 100 times stiffer than the default: 3905.83
    (0.17 percent). At h = 1 mm: 2172.12 against 2176.29 (0.19 percent). So the remaining difference is not the bond or the beam: it
    is the pad. In the joint model the pad voxels (solid, same E as the part) are held only by the bond on their outer faces, and in the
    reference all pad nodes are fixed. Bar 0.5 percent for the lowest factor. The higher modes differ more (3.8 percent for the 4th), because
    they involve the pad."""
    sub1, sub2 = tmp_path / "joint", tmp_path / "fixed"
    sub1.mkdir()
    sub2.mkdir()
    jprob = prob_of(sub1, h=2.0, clamp=False, solver="auto",
                    joint={"E": 2.1e9, "to": [24, 5, 5], "section": {"kind": "round", "d": 10.0}})

    def fixed_pad(d):
        d.pop("joints")
        d["regions"]["fixed"] = ["pad.stl"]
        d["regions"]["load"] = {"tip": "tip.stl"}
    fprob = build_problem(load_config(joint_case(sub2, h=2.0, clamp=False, solver="auto", mutate=fixed_pad)))
    got = bv.check_problem(jprob, np.full(int(np.prod(jprob.shape)), 255, np.uint8), modes=4)["cases"][0]["load_factors"]
    ref = bv.check_problem(fprob, np.full(int(np.prod(fprob.shape)), 255, np.uint8), modes=4)["cases"][0]["load_factors"]
    print("limit", got, ref)
    assert abs(got[0] - ref[0]) / ref[0] < 5e-3
    assert abs(got[0] - ref[0]) / ref[0] > 1e-4          # not equal: the pad is flexible in the joint model (the number above is real)


@needs_direct_solver
def test_the_record_says_that_the_beams_have_no_geometric_stiffness(tmp_path):
    prob = prob_of(tmp_path, h=2.0, solver="auto")
    res = bv.check_problem(prob, gray(prob), modes=2)
    assert res["status"] == "done" and res["joints"]["joints"] == 1 and res["joints"]["beams_have_geometric_stiffness"] is False
    assert any("no geometric stiffness" in f and "not for buckling of the beams" in f for f in res["facts"])
    assert any("Euler" in o for o in res["options"])
    json.dumps(res)
    assert "not of the beams" in res["joints"]["meaning"]


@needs_direct_solver
def test_a_run_without_joints_has_no_joint_key_and_the_old_eigen_solver_text(tmp_path):
    plain = build_problem(load_config(joint_case(tmp_path, h=2.0, solver="auto",
                                                 mutate=lambda d: d.pop("joints") or d["regions"].update(load={"tip": "tip.stl"}))))
    assert plain.joints is None
    res = bv.check_problem(plain, np.full(int(np.prod(plain.shape)), 255, np.uint8), modes=2)
    assert "joints" not in res and res["eigen_solver"] == bk.EIGEN_SOLVER
    assert not any("geometric stiffness" in f and "beams" in f for f in res["facts"])


@needs_direct_solver
def test_topopt_buckling_on_a_finished_run_with_a_joint_writes_the_key(tmp_path, capsys):
    out = tmp_path / "out"
    p = joint_case(tmp_path, h=2.0, solver="auto")
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert main(["buckling", str(out), "--modes", "2"]) == 0
    key = json.loads((out / "result" / "metrics.json").read_text())["buckling"]
    assert key["status"] == "done" and key["lowest_load_factor"]["value"] > 0
    assert key["joints"]["beams_have_geometric_stiffness"] is False
    assert any("no geometric stiffness" in f for f in key["facts"])
    assert "skipped" not in capsys.readouterr().out
