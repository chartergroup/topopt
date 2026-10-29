"""K1-A: joints and springs in `topopt verify` (`topopt/verify/supports_p2.py`, CONTRACTS.md sections 16, 35 and 38).

The physics tests use a structured tet block (`verify_mesh.box_mesh`, no gmsh) and closed forms:
- joint: a block bonded to a joint beam, a tip load. The deflection of the load face is `F (L^3 / 3 + a L^2 + a^2 L) / (E I)` (a cantilever of
  length L with a rigid arm a). The beam is soft (E 2100) and the block is stiff (E 210000), so the block is nearly rigid.
- spring: a block on a spring of `k` N/mm3 on one face, a force F on the other face. The displacement is `F / (k A)`.
The tests with a YAML and a run folder need gmsh (`needs_gmsh`): a part with `fixed: []` and two joints, and the parity of the tet
stiffness against the voxel stiffness of a real run of the optimizer (11 s on a 2024 Mac, so not marked slow).
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sps
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_accelerate, needs_gmsh, needs_skfem  # noqa: E402
from test_optimize import box  # noqa: E402
from verify_mesh import box_mesh  # noqa: E402

from topopt.config import JointSpec, SectionSpec, SpringSpec, UserError, load_config  # noqa: E402

pytestmark = needs_skfem


# --------------------------------------------------------------------------- #
# Physics on a structured block (no gmsh)
# --------------------------------------------------------------------------- #

def block(cells=(4, 4, 4), size=(10.0, 10.0, 10.0)):
    """`(model, left, right)`: a P2 block and the selections of its faces x = 0 and x = size[0]."""
    from topopt.verify import regions
    from topopt.verify.p2 import P2Model
    p, t = box_mesh(size, cells)
    m = P2Model(p, t)
    bf = regions.boundary_facets(m.t.T)
    c = m.p.T[bf].mean(axis=1)

    def sel(mask):
        f = bf[mask]
        return SimpleNamespace(facets=f, facet_area=regions.facet_areas(m.p.T, f))
    return m, sel(np.abs(c[:, 0]) < 1e-9), sel(np.abs(c[:, 0] - size[0]) < 1e-9)


def solve(m, sup, E, nu, f):
    """The grid displacement (N,) for the load vector `f`, with the bordered system and the refined factor of `topopt.verify`."""
    from topopt.verify.factor import RefinedFactor, make_factor
    K = (m.stiffness(E, nu) + sup.Ks).tocsr()
    free = np.arange(m.N)
    ss = sup.system(K, free)
    rf = ss.padded(RefinedFactor(ss.A, make_factor(ss.A, "superlu"), 3))
    return rf.solve(f[:, None])[:, 0], K


def test_joint_deflection_matches_the_beam_with_a_rigid_block():
    from topopt.verify import bc, supports_p2
    m, left, right = block()
    E, F, L, a, d = 210000.0, 0.1, 50.0, 10.0, 2.0
    cfg = SimpleNamespace(material=SimpleNamespace(E=E, nu=0.3), springs=(),
                          joints=(JointSpec("left", (-L, 5.0, 5.0), SectionSpec("round", (d,)), None, None, None, 2100.0, 0.3),))
    sup = supports_p2.build_supports(m, cfg, {"left": left}, 2.0)
    f, _ = bc.traction_load(m, right.facets, [0.0, 0.0, -F])
    x, K = solve(m, sup, E, 0.3, f)
    X = m.node_xyz.T
    face = np.flatnonzero(np.abs(X[:, 0] - 10.0) < 1e-9)
    got = -x.reshape(-1, 3)[face, 2].mean()
    EI = 2100.0 * np.pi * d ** 4 / 64.0
    want = F * (L ** 3 / 3 + a * L ** 2 + a ** 2 * L) / EI
    print(f"\njoint tip deflection: tet {got:.6g} mm, beam formula {want:.6g} mm, ratio {got / want:.6f}")
    assert abs(got / want - 1.0) < 0.02
    rec = sup.case_record(m, x, f)["joints"]["left"]
    assert rec["bond_force_N"] == pytest.approx([0.0, 0.0, F], abs=1e-6)                 # the bond holds the load
    assert rec["bond_moment_Nmm"] == pytest.approx([0.0, -F * a, 0.0], abs=1e-5)         # about the reference point (0, 5, 5)
    assert rec["about_mm"] == pytest.approx([0.0, 5.0, 5.0])
    assert rec["bond_wrench_vs_beam_relative"] < 1e-4                                    # the nodal wrench is B q
    eq = bc.equilibrium(m, K, x, f, np.zeros(0, np.int64), support=sup.terms(x))
    assert eq["force_relative_residual"] < 1e-5 and eq["moment_relative_residual"] < 1e-5    # a stiff block on a soft beam: ill-conditioned


@pytest.mark.parametrize("entry", [SpringSpec("left", None, (10.0, 10.0, 10.0)), SpringSpec("left", (1000.0, 1000.0, 1000.0), None)])
def test_spring_displacement_is_force_over_k_a(entry):
    from topopt.verify import bc, supports_p2
    m, left, right = block()
    E, F = 210000.0, 50.0
    cfg = SimpleNamespace(material=SimpleNamespace(E=E, nu=0.3), joints=(), springs=(entry,))
    sup = supports_p2.build_supports(m, cfg, {"left": left}, 2.0)
    k = 10.0                                              # N/mm3 on a face of 100 mm2: 1000 N/mm
    f, _ = bc.traction_load(m, right.facets, [F, 0.0, 0.0])
    x, _ = solve(m, sup, E, 0.3, f)
    X = m.node_xyz.T
    got = x.reshape(-1, 3)[np.flatnonzero(np.abs(X[:, 0] - 10.0) < 1e-9), 0].mean()
    want = F / (k * 100.0)
    print(f"\nspring displacement: tet {got:.6g} mm, F/(k A) {want:.6g} mm, ratio {got / want:.6f}")
    assert abs(got / want - 1.0) < 0.01
    rec = sup.case_record(m, x, f)["springs"]["left"]
    assert rec["force_N"] == pytest.approx([-F, 0.0, 0.0], abs=1e-6)                     # the spring pulls the part back


def test_the_spring_matrix_equals_a_scikit_fem_facet_assembly():
    from skfem import BilinearForm, ElementTetP2, ElementVector, FacetBasis
    from skfem.helpers import dot

    from topopt.verify import supports_p2
    m, left, _right = block((3, 3, 3))
    k = 7.0
    got = supports_p2.spring_matrix_p2(m, left.facets, (k, k, k))
    want_keys = {tuple(sorted(r)) for r in left.facets.tolist()}
    idx = [i for i, fc in enumerate(m.mesh.facets.T.tolist()) if tuple(sorted(fc)) in want_keys]
    assert len(idx) == len(left.facets)
    fb = FacetBasis(m.mesh, ElementVector(ElementTetP2()), facets=np.array(idx), intorder=4)
    ref = BilinearForm(lambda u, v, w: k * dot(u, v)).assemble(fb)
    assert abs(got - ref).max() < 1e-9 * abs(ref).max()


def test_bordered_csr_equals_bmat():
    from topopt.verify.supports_p2 import bordered_csr
    rng = np.random.default_rng(1)
    K = sps.random(40, 40, density=0.2, random_state=1, format="csr")
    K = (K + K.T + 40 * sps.identity(40)).tocsr()
    C = sps.random(40, 6, density=0.4, random_state=2, format="csr")
    D = sps.csr_matrix(np.diag(rng.uniform(1, 2, 6)))
    want = sps.bmat([[K, C], [C.T, D]], format="csr")
    got = bordered_csr(K, C, D, chunk_rows=7)
    assert abs(got - want).max() == 0.0 and got.shape == (46, 46)


def test_a_rigid_motion_of_the_part_costs_no_energy_in_the_bond():
    """The bond is exact for a rigid motion: `Kb u + C q = 0` when `u = R q` (the P2 field holds a linear field)."""
    from topopt.verify import supports_p2
    m, left, _right = block((3, 3, 3))
    ref = np.array([0.0, 5.0, 5.0])
    Kb, C, D = supports_p2.bond_blocks_p2(m, left.facets, ref, 3.0)
    q = np.array([0.1, -0.2, 0.3, 0.01, -0.02, 0.03])
    u = np.zeros(m.N)
    X = m.node_xyz.T
    u.reshape(-1, 3)[:] = q[:3] + np.cross(q[3:], X - ref)
    assert np.abs(Kb @ u + C @ q).max() < 1e-10
    assert abs(q @ D @ q + 2 * q @ (C.T @ u) + u @ (Kb @ u)) < 1e-10                    # the energy 1/2 k int |u - R q|^2 is zero


# --------------------------------------------------------------------------- #
# The facets of a support region: the rule of the voxel model (REVIEW-135 M1)
# --------------------------------------------------------------------------- #

def _box_mesh(lo, hi):
    import trimesh
    m = trimesh.creation.box(extents=np.asarray(hi, float) - np.asarray(lo, float))
    m.apply_translation((np.asarray(lo, float) + np.asarray(hi, float)) / 2)
    return m


def _pad_block():
    """A 4 x 10 x 10 mm block (the pad, x 0..4) and the selection of all its boundary facets, as a region that covers the block."""
    from topopt.verify import regions
    from topopt.verify.p2 import P2Model
    p, t = box_mesh((4.0, 10.0, 10.0), (2, 5, 5))
    m = P2Model(p, t)
    bf, be = regions.boundary_facets(m.t.T, return_elems=True)
    sel = SimpleNamespace(facets=bf, facet_area=regions.facet_areas(m.p.T, bf), elems=be)
    return m, sel


def test_a_facet_that_looks_toward_the_design_is_not_bonded():
    """The voxel model bonds a face toward empty space and no face toward design, fixed or load voxels. The tet rule does the same."""
    from topopt.verify import supports_p2
    m, sel = _pad_block()
    design = _box_mesh((4, 0, 0), (14, 10, 10))                  # next to the pad face x = 4
    got = supports_p2.bonded_facets(m, sel, [design], [], step=0.5)
    assert got.band_area_mm2 == pytest.approx(2 * 100.0 + 4 * 40.0) and got.band_facets == len(sel.facets)
    assert got.toward_design_area_mm2 == pytest.approx(100.0)                           # the face x = 4
    assert got.facet_area.sum() == pytest.approx(100.0 + 4 * 40.0)                      # the end face x = 0 and the four sides
    c = m.p.T[got.facets].mean(axis=1)
    assert not np.any(np.abs(c[:, 0] - 4.0) < 1e-9) and np.any(np.abs(c[:, 0]) < 1e-9)
    assert got.toward_design == len(sel.facets) - len(got.facets)
    gap = supports_p2.bonded_facets(m, sel, [_box_mesh((6, 0, 0), (14, 10, 10))], [], step=0.5)     # a gap of 2 mm: the step does not reach it
    assert gap.toward_design == 0 and gap.facet_area.sum() == pytest.approx(got.band_area_mm2)


def test_a_keep_out_region_wins_over_the_design_like_the_label_keep_out():
    from topopt.verify import supports_p2
    m, sel = _pad_block()
    design = _box_mesh((4, 0, 0), (14, 10, 10))
    keep_out = _box_mesh((4, 0, 0), (7, 10, 10))
    got = supports_p2.bonded_facets(m, sel, [design], [keep_out], step=0.5)
    assert got.toward_design == 0 and got.facet_area.sum() == pytest.approx(got.band_area_mm2)     # the neighbour voxel is KEEP_OUT: exposed


def test_a_fixed_or_load_region_next_to_a_facet_counts_like_the_design():
    from topopt.verify import supports_p2
    m, sel = _pad_block()
    other = _box_mesh((4, 0, 0), (8, 10, 10))                    # another region (label FIXED or LOAD) at the face x = 4
    got = supports_p2.bonded_facets(m, sel, [_box_mesh((30, 0, 0), (40, 10, 10)), other], [], step=0.5)
    assert got.toward_design_area_mm2 == pytest.approx(100.0)


def test_a_region_with_no_facet_to_bond_stops_with_the_options():
    from topopt.verify import supports_p2
    m, sel = _pad_block()
    surround = _box_mesh((-5, -5, -5), (10, 15, 15))               # the pad lies inside the design on every side
    got = supports_p2.bonded_facets(m, sel, [surround], [], step=0.5)
    assert len(got.facets) == 0 and got.toward_design == got.band_facets
    cfg = SimpleNamespace(joints=(), springs=(SpringSpec("pad", (1.0, 1.0, 1.0), None),))
    with pytest.raises(UserError) as e:
        supports_p2.check_bonded(cfg, {"pad": got})
    text = str(e.value)
    assert "spring on load region 'pad'" in text and "look toward the design" in text and "Options:" in text


# --------------------------------------------------------------------------- #
# The refusal
# --------------------------------------------------------------------------- #

def _yaml_case(folder: Path, mutate=None, joints=True) -> Path:
    """The two-pad bar: design x 0..20, pads x -2..0 and 20..22 (joints), a tip box on top with the force, and `fixed: []`."""
    folder.mkdir(parents=True, exist_ok=True)
    box(folder / "design.stl", (0, 0, 0), (20, 10, 10))
    box(folder / "padl.stl", (-2, 0, 0), (0, 10, 10))
    box(folder / "padr.stl", (20, 0, 0), (22, 10, 10))
    box(folder / "tip.stl", (8, 4, 10), (12, 6, 13))
    doc = {"name": "two_joints",
           "regions": {"design": "design.stl", "fixed": [], "load": {"padl": "padl.stl", "padr": "padr.stl", "tip": "tip.stl"}},
           "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
           "settings": {"voxel_size": 1.0, "volume_fraction": 0.5, "iterations": 12}}
    if joints:
        doc["joints"] = [{"region": "padl", "to": [-30, 5, 5], "section": {"kind": "round", "d": 4.0}},
                         {"region": "padr", "to": [50, 5, 5], "section": {"kind": "round", "d": 4.0}}]
    if mutate:
        mutate(doc)
    p = folder / "c.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def test_joints_and_springs_pass_the_guard_and_a_moment_is_still_refused(tmp_path):
    from topopt.verify.run import _check_loads_and_supports
    _check_loads_and_supports(load_config(_yaml_case(tmp_path / "a")))               # two joints, fixed: []
    spring = _yaml_case(tmp_path / "b", lambda d: d.update(springs=[{"region": "padl", "stiffness": [100, 100, 100]}], joints=None))
    _check_loads_and_supports(load_config(spring))
    both = _yaml_case(tmp_path / "c", lambda d: d["load_cases"][0]["forces"][0].update(moment=[0.0, 5.0, 0.0]))
    with pytest.raises(UserError) as e:
        _check_loads_and_supports(load_config(both))
    text = str(e.value)
    assert "a moment" in text and "beam support" not in text and "does not model" in text and "Options:" in text
    assert "plain forces and full clamps" in text


# --------------------------------------------------------------------------- #
# A run folder: verify with joints (gmsh)
# --------------------------------------------------------------------------- #

def _bar_run(tmp_path: Path) -> Path:
    """A run folder (no optimizer): `final.stl` is the bar with both pads, x -2..22, y 0..10, z 0..10."""
    yml = _yaml_case(tmp_path / "case")
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    box(run / "result" / "final.stl", (-2, 0, 0), (22, 10, 10))
    d = yaml.safe_load(yml.read_text())
    (run / "run.json").write_text(json.dumps({"name": d["name"], "yaml_path": str(yml), "h": 1.0}))
    return run


@needs_gmsh
def test_a_part_with_no_fixed_region_and_two_joints_runs(tmp_path):
    from topopt.verify import verify_run
    run = _bar_run(tmp_path)
    res = verify_run(run, mesh_size=2.0, solver="superlu", buckling=True, load_sensitivity=False)
    assert res["ok"] and res["exit_code"] == 0, res
    sup = res["supports"]
    assert [j["region"] for j in sup["joints"]] == ["padl", "padr"] and sup["bordered_dofs"] == 12 and sup["springs"] == []
    assert res["regions"]["fixed"] == [] and res["mesh"]["fixed_dofs"] == 0
    c = res["cases"]["down"]
    assert c["equilibrium"]["force_relative_residual"] < 1e-8 and c["equilibrium"]["moment_relative_residual"] < 1e-5
    joints = sup["cases"]["down"]["joints"]
    assert sorted(joints) == ["padl", "padr"]
    fz = 0.0
    for name, r in joints.items():
        assert len(r["bond_force_N"]) == 3 and len(r["bond_moment_Nmm"]) == 3           # the 6 resultants of a bond
        assert r["beam"]["part"]["force_n"] == pytest.approx(r["bond_force_N"], rel=1e-3, abs=1e-6)
        fz += r["bond_force_N"][2]
    assert fz == pytest.approx(2.0, rel=1e-6)                                            # the two bonds carry the whole load
    bk = res["buckling"]
    assert bk["status"] == "done" and bk["supports"]["beams_have_geometric_stiffness"] is False
    assert any("beam has no Kg" in t for t in bk["facts"]) and bk["lowest_load_factor"]["value"] > 0
    assert "A body that a joint or a spring holds is held" in bk["loose_piece"]["reason"] and "condensed stiffness" in bk["factor"]
    from topopt.ratchet import verify_facts                              # the facts that the ratchet judge reads from a verify report (CONTRACTS 36.10)
    f = verify_facts(res)["cases"]["down"]
    assert f["stiffness_N_per_mm"] > 0 and f["p_by_volume_MPa"] > 0 and f["max_displacement_mm"] > 0


@needs_gmsh
def test_a_part_held_by_springs_only_runs(tmp_path):
    from topopt.verify import verify_run
    yml = _yaml_case(tmp_path / "springs", lambda d: (d.update(joints=None, springs=[
        {"region": "padl", "stiffness": [2000, 2000, 2000]}, {"region": "padr", "stiffness_per_area": 20.0}])))
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    box(run / "result" / "final.stl", (-2, 0, 0), (22, 10, 10))
    (run / "run.json").write_text(json.dumps({"name": "springs", "yaml_path": str(yml), "h": 1.0}))
    res = verify_run(run, mesh_size=2.0, solver="superlu", load_sensitivity=False)
    assert res["ok"] and res["supports"]["joints"] == [] and res["supports"]["bordered_dofs"] == 0
    springs = res["supports"]["cases"]["down"]["springs"]
    assert sorted(springs) == ["padl", "padr"]
    assert springs["padl"]["force_N"][2] + springs["padr"]["force_N"][2] == pytest.approx(2.0, rel=1e-6)
    from topopt.verify.report import summary_lines
    text = "\n".join(summary_lines(res))
    assert "supports: 0 joint(s) and 2 spring(s)" in text and "case down: spring padl: force [" in text and "case down: spring padr: force [" in text
    assert res["cases"]["down"]["equilibrium"]["force_relative_residual"] < 1e-8


SPRINGS_ONLY = (lambda d: d.update(joints=None, springs=[{"region": "padl", "stiffness": [2000, 2000, 2000]},
                                                        {"region": "padr", "stiffness_per_area": 20.0}]))


@needs_gmsh
@needs_accelerate
@pytest.mark.parametrize("kind", ["joints", "springs"])
def test_the_mac_factor_cholesky_gives_the_superlu_answer_with_joints_and_springs(tmp_path, kind):
    """macOS arm64 only (REVIEW-135 M4): `cholesky` (Apple Accelerate), the default factor on the Mac, solves the bordered matrix of the
    joints and `K + Kb` of the springs of `topopt verify` with the answer of SuperLU. On Linux only the fake library test of
    `test_joints.py` runs the input checks of that path; this test runs the real library."""
    from topopt.verify import verify_run
    out = {}
    for solver in ("superlu", "cholesky"):
        run = _bar_run_with(tmp_path / solver, None if kind == "joints" else SPRINGS_ONLY)
        res = verify_run(run, mesh_size=2.0, solver=solver, load_sensitivity=False)
        assert res["ok"] and res["exit_code"] == 0, res
        out[solver] = res
    a, b = out["superlu"]["cases"]["down"], out["cholesky"]["cases"]["down"]
    assert b["equilibrium"]["force_relative_residual"] < 1e-8 and b["linear_solve"]["relative_residual"] < 1e-8
    for key in ("compliance_Nmm", "stiffness_N_per_mm", "max_displacement_mm"):
        assert b[key] == pytest.approx(a[key], rel=1e-6), key
    sa, sb = out["superlu"]["supports"]["cases"]["down"], out["cholesky"]["supports"]["cases"]["down"]
    for name, r in sa[kind].items():
        got = sb[kind][name]["bond_force_N" if kind == "joints" else "force_N"]
        assert got == pytest.approx(r["bond_force_N" if kind == "joints" else "force_N"], rel=1e-6, abs=1e-9), name


@needs_gmsh
def test_a_fixed_region_and_a_joint_hold_the_part_together(tmp_path):
    """The mixed case of the review (L5): `padl` is a clamp, `padr` a joint. The clamp reaction and the bond force add up to the load."""
    from topopt.verify import verify_run
    yml = _yaml_case(tmp_path / "mixed", lambda d: (d.update(joints=d["joints"][1:]), d["regions"].update(fixed=["padl.stl"]), d["regions"]["load"].pop("padl")))
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    box(run / "result" / "final.stl", (-2, 0, 0), (22, 10, 10))
    (run / "run.json").write_text(json.dumps({"name": "mixed", "yaml_path": str(yml), "h": 1.0}))
    res = verify_run(run, mesh_size=2.0, solver="superlu", load_sensitivity=False)
    assert res["ok"] and [j["region"] for j in res["supports"]["joints"]] == ["padr"] and res["mesh"]["fixed_dofs"] > 0
    eq = res["cases"]["down"]["equilibrium"]
    assert eq["force_relative_residual"] < 1e-8 and eq["moment_relative_residual"] < 1e-5
    bond = res["supports"]["cases"]["down"]["joints"]["padr"]["bond_force_N"]
    assert 0.0 < bond[2] < 2.0                                          # the joint carries a part of the load; the clamp carries the rest
    assert eq["support_force_N"][2] == pytest.approx(bond[2], rel=1e-6)
    assert eq["support_force_N"][2] + eq["reaction_force_N"][2] == pytest.approx(2.0, rel=1e-6)
    from topopt.verify.report import summary_lines                      # Q5: the text summary has the 6 bond numbers of the joint
    text = "\n".join(summary_lines(res))
    assert "supports: 1 joint(s) and 0 spring(s)" in text and re.search(r"case down: joint padr: bond force \[[-0-9.e, ]+\] N, moment \[[-0-9.e, ]+\] N mm about \[", text)


@needs_gmsh
def test_a_plain_run_has_no_supports_key(tmp_path):
    from topopt.verify import verify_run
    yml = _yaml_case(tmp_path / "plain", lambda d: (d.update(joints=None), d["regions"].update(fixed=["padl.stl"]),
                                                    d["regions"]["load"].pop("padl")))
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    box(run / "result" / "final.stl", (-2, 0, 0), (22, 10, 10))
    (run / "run.json").write_text(json.dumps({"name": "plain", "yaml_path": str(yml), "h": 1.0}))
    res = verify_run(run, mesh_size=2.0, solver="superlu", load_sensitivity=False)
    assert res["ok"] and "supports" not in res and "support_force_N" not in res["cases"]["down"]["equilibrium"]


@needs_gmsh
def test_parity_of_the_tet_stiffness_with_the_voxel_optimizer(tmp_path, capsys):
    """One small part with two joints: `topopt run` (voxel, 12 passes) and then `topopt verify`. The ratio of the stiffness is a fact."""
    from topopt.cli import main
    yml = _yaml_case(tmp_path / "case")
    out = tmp_path / "run"
    assert main(["run", str(yml), "--out", str(out)]) == 0
    from topopt.verify import verify_run
    res = verify_run(out, mesh_size=1.5, solver="superlu", load_sensitivity=False)
    c = res["cases"]["down"]
    ratio = c["stiffness_over_optimizer"]
    print(f"\nparity: tet {c['stiffness_N_per_mm']:.4f} N/mm, voxel {c['optimizer_stiffness_N_per_mm']:.4f} N/mm, ratio {ratio:.4f}")
    assert 0.9 < ratio < 1.1


@needs_gmsh
@pytest.mark.parametrize("entry", ["stiffness_per_area", "stiffness"])
def test_parity_of_a_spring_part_with_the_voxel_optimizer(tmp_path, entry):
    """The two-pad bar on springs instead of joints, optimized (12 passes) and then verified at the mesh size 2 mm (REVIEW-135 M1).

    Measured on one Linux host (a pad has 180 mm2 of voxel faces; the same bar with both pads FIXED gives the tet over voxel ratio 1.065):
    - the rule of the load region (every boundary facet in the band) bonded 216 to 218 mm2 per pad, +20 %, and the ratio was 1.129 (total
      stiffness) and 1.212 (per area): +6 % and +14 % above the baseline;
    - the rule of the voxel model (facets toward the design are not bonded) bonds 178 to 180 mm2 (-1.1 % to +0.1 %) and the ratio is 1.099
      (total) and 1.095 (per area), +3 % above the baseline. The excess that is left is the discretisation of the spring face (Q1 against P2).
    The bands are wide enough for the host, the mesher and the BLAS to change the third digit and too narrow for the old rule.
    """
    from topopt.cli import main
    k = {"stiffness_per_area": 20.0, "stiffness": [3600.0, 3600.0, 3600.0]}[entry]
    yml = _yaml_case(tmp_path / "case", lambda d: d.update(joints=None, springs=[{"region": r, entry: k} for r in ("padl", "padr")]))
    out = tmp_path / "run"
    assert main(["run", str(yml), "--out", str(out)]) == 0
    from topopt.verify import verify_run
    res = verify_run(out, mesh_size=2.0, solver="superlu", load_sensitivity=False)
    c = res["cases"]["down"]
    ratio = c["stiffness_over_optimizer"]
    pads = {s["region"]: s for s in res["supports"]["springs"]}
    print(f"\nspring parity ({entry}): tet {c['stiffness_N_per_mm']:.2f} N/mm, voxel {c['optimizer_stiffness_N_per_mm']:.2f} N/mm, ratio {ratio:.4f}; "
          + ", ".join(f"{n}: tet {s['area_mm2']:.1f} mm2 over voxel {s['voxel_area_mm2']:.1f} = {s['area_over_voxel_area']:.4f}" for n, s in pads.items()))
    for name, s in pads.items():
        assert s["voxel_area_mm2"] == 180.0 and 0.95 < s["area_over_voxel_area"] < 1.05, name
        assert s["facets_toward_design"] > 0 and s["band_area_mm2"] > s["area_mm2"] + 20.0         # the inner face of the pad was dropped
    assert 1.0 < ratio < 1.15


# --------------------------------------------------------------------------- #
# yaml_vs_run compares the joints and the springs (REVIEW-135 M2)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def mixed_run(tmp_path_factory):
    """A real `topopt run` (2 passes) of the two-pad bar with a joint on `padl` and a spring on `padr`: the `run.json` of the voxel model."""
    from topopt.cli import main

    def mut(d):
        d["joints"] = d["joints"][:1]
        d["springs"] = [{"region": "padr", "stiffness": [2000.0, 2000.0, 2000.0]}]
        d["settings"]["iterations"] = 2
    root = tmp_path_factory.mktemp("mixed")
    yml = _yaml_case(root / "case", mut)
    out = root / "run"
    assert main(["run", str(yml), "--out", str(out)]) == 0
    return yml, json.loads((out / "run.json").read_text())


def _compare(run, yml, edit=None):
    from topopt.verify.run import compare_yaml_run
    if edit is not None:
        d = yaml.safe_load(Path(yml).read_text())
        edit(d)
        yml = Path(yml).with_name("edited.yaml")
        yml.write_text(yaml.safe_dump(d, sort_keys=False))
    return compare_yaml_run(SimpleNamespace(run=run, cfg=load_config(yml), yaml_path=yml))


def test_the_unchanged_yaml_of_a_run_with_a_joint_and_a_spring_matches(mixed_run):
    yml, run = mixed_run
    res = _compare(run, yml)
    assert res["matches"] and res["differences"] == []
    assert "The material (E, nu) is not in run.json: it is not compared" in res["note"]       # the text of compare_yaml_problem says so too


@pytest.mark.parametrize("edit, want", [
    (lambda d: d["joints"][0].update(to=[-40, 5, 5]), "joint padl: far point [-30.0, 5.0, 5.0] in run.json, [-40.0, 5.0, 5.0] in the YAML"),
    (lambda d: d["joints"][0].update(section={"kind": "round", "d": 5.0}), "joint padl: section {'kind': 'round', 'd': 4.0} in run.json, {'kind': 'round', 'd': 5.0} in the YAML"),
    (lambda d: d["joints"][0].update(section={"kind": "rect", "b": 4.0, "h": 4.0}), "joint padl: section"),
    (lambda d: d["joints"][0].update(E=7000.0), "joint padl: beam E 3500 in run.json, 7000 in the YAML"),
    (lambda d: d["joints"][0].update(nu=0.3), "joint padl: beam nu 0.35 in run.json, 0.3 in the YAML"),
    (lambda d: d["joints"][0].update(bond_stiffness_per_area=1000.0), "joint padl: bond stiffness per area 105000 in run.json, 1000 in the YAML"),
    (lambda d: d["joints"][0].update(origin=[0, 5, 5]), "joint padl: origin"),
    (lambda d: d["joints"][0].update(orient=[0, 0, 1]), "joint padl: section y axis [0.0, 1.0, 0.0] in run.json, [0.0, 0.0, 1.0] in the YAML (orient)"),
    (lambda d: d["joints"].append({"region": "padr", "to": [50, 5, 5], "section": {"kind": "round", "d": 4.0}}), "joints: run.json has ['padl'], the YAML has ['padl', 'padr']"),
    (lambda d: d.pop("joints"), "joints: run.json has ['padl'], the YAML has []"),
    (lambda d: d["springs"][0].update(stiffness=[2000.0, 2000.0, 3000.0]), "spring padr: stiffness [2000.0, 2000.0, 2000.0] in run.json, [2000.0, 2000.0, 3000.0] in the YAML"),
    (lambda d: d["springs"].__setitem__(0, {"region": "padr", "stiffness_per_area": 11.0}), "spring padr: stiffness per area"),
    (lambda d: d["springs"].append({"region": "padl", "stiffness": [1.0, 1.0, 1.0]}), "springs: run.json has ['padr'], the YAML has ['padl', 'padr']"),
    (lambda d: d.pop("springs"), "springs: run.json has ['padr'], the YAML has []"),
])
def test_a_changed_joint_or_spring_is_a_difference(mixed_run, edit, want):
    yml, run = mixed_run
    res = _compare(run, yml, edit)
    assert not res["matches"] and any(want in t for t in res["differences"]), res["differences"]


def test_a_joint_in_the_yaml_but_not_in_a_plain_run_is_a_difference():
    """The case of the review: the recorded hook run has no joint and no spring; a YAML that adds a spring gave `matches` True."""
    from topopt.verify.run import compare_yaml_run
    run = {"name": "two_joints", "h": 1.0}
    cfg = SimpleNamespace(name="two_joints", load_cases=[], regions=SimpleNamespace(design=Path("design.stl"), fixed=[], keep_out=[], solid=[], load=[]),
                          joints=(), springs=(SpringSpec("tip", (1.0, 2.0, 3.0), None),), material=SimpleNamespace(E=3500.0, nu=0.35),
                          settings=SimpleNamespace(voxel_size=1.0))
    res = compare_yaml_run(SimpleNamespace(run=run, cfg=cfg, yaml_path=Path("c.yaml")))
    assert not res["matches"] and res["differences"] == ["springs: run.json has [], the YAML has ['tip']"]
    cfg.springs = ()
    plain = compare_yaml_run(SimpleNamespace(run=run, cfg=cfg, yaml_path=Path("c.yaml")))
    assert plain["matches"] and "Joints and springs" not in plain["note"]                  # a plain run keeps its text


# --------------------------------------------------------------------------- #
# A joint or spring region with no force is a region of the report (REVIEW-135 M3)
# --------------------------------------------------------------------------- #

def _bar_run_with(tmp_path: Path, mutate) -> Path:
    """Like `_bar_run`, with a changed YAML."""
    yml = _yaml_case(tmp_path / "case", mutate)
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    box(run / "result" / "final.stl", (-2, 0, 0), (22, 10, 10))
    d = yaml.safe_load(yml.read_text())
    (run / "run.json").write_text(json.dumps({"name": d["name"], "yaml_path": str(yml), "h": 1.0}))
    return run


@needs_gmsh
@pytest.mark.parametrize("kind", ["joint", "spring"])
def test_a_support_region_with_no_force_is_in_regions_near_regions_and_the_load_band_note(tmp_path, kind):
    from topopt.verify import verify_run
    mutate = None if kind == "joint" else (lambda d: d.update(joints=None, springs=[{"region": r, "stiffness": [2000.0] * 3} for r in ("padl", "padr")]))
    res = verify_run(_bar_run_with(tmp_path, mutate), mesh_size=2.0, solver="superlu")          # the force acts on `tip` only
    rows = {r["region"]: r for r in res["regions"]["load"]}
    assert {n: r["role"] for n, r in rows.items()} == {"tip": "load", "padl": kind, "padr": kind}
    for n in ("padl", "padr"):
        assert rows[n]["facets"] > 0 and rows[n]["area_mm2"] > 0 and rows[n]["normal_check"]["coherence"] is not None
        assert set(rows[n]["counts_by_band"]) == {"0", "0.5x", "1x"} and rows[n]["file"].endswith(f"{n}.stl")
    near = res["cases"]["down"]["von_mises"]["near_regions"]
    assert sorted(near["regions"]) == ["padl", "padr", "tip"]
    assert set(near["top_volume_share_near"]) == {"padl", "padr", "tip", "any", "none"}
    sens = res["load_band_sensitivity"]
    assert "A joint or spring region (padl, padr) keeps the facets of the base band" in sens["note"]
    assert all(set(row["load_regions"]) == {"tip"} for row in sens["bands"] if "load_regions" in row)    # only a load vector changes with the band


@needs_gmsh
def test_a_support_region_with_a_force_stays_a_load_region(tmp_path):
    from topopt.verify import verify_run
    res = verify_run(_bar_run_with(tmp_path, lambda d: d["load_cases"][0].update(forces=[{"region": "padr", "force": [0, 0, -2.0]}])),
                     mesh_size=2.0, solver="superlu", load_sensitivity=False)
    assert {r["region"]: r["role"] for r in res["regions"]["load"]} == {"padr": "load", "padl": "joint"}
    assert res["supports"]["bordered_dofs"] == 12


@needs_gmsh
def test_a_plain_run_keeps_the_note_of_the_load_band_check(tmp_path):
    from topopt.verify import verify_run
    yml = _yaml_case(tmp_path / "plain", lambda d: (d.update(joints=None), d["regions"].update(fixed=["padl.stl"]), d["regions"]["load"].pop("padl")))
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    box(run / "result" / "final.stl", (-2, 0, 0), (22, 10, 10))
    (run / "run.json").write_text(json.dumps({"name": "plain", "yaml_path": str(yml), "h": 1.0}))
    res = verify_run(run, mesh_size=2.0, solver="superlu")
    assert "joint or spring" not in res["load_band_sensitivity"]["note"]
    assert {r["role"] for r in res["regions"]["load"]} == {"load"}


# --------------------------------------------------------------------------- #
# The singular-system texts know the supports (REVIEW-135 L1)
# --------------------------------------------------------------------------- #

@needs_gmsh
def test_a_spring_on_one_axis_gives_a_singular_text_that_names_the_supports(tmp_path):
    """Two springs that act on z only: the part can move along x and y. The text says which bodies the supports hold and what to change."""
    from topopt.verify import verify_run
    from topopt.verify.errors import SingularSystem
    run = _bar_run_with(tmp_path, lambda d: d.update(joints=None, springs=[{"region": r, "stiffness": [0.0, 0.0, 1000.0]} for r in ("padl", "padr")]))
    with pytest.raises(SingularSystem) as e:
        verify_run(run, mesh_size=2.0, solver="superlu", load_sensitivity=False)
    text = str(e.value)
    assert "(each has a fixed node or a facet of a joint or spring; 1 is held by a joint or spring facet) and 0 fixed nodes" in text
    assert "a spring or a joint that holds fewer than the six rigid-body motions" in text and "stiffness on one axis leaves the other two axes free" in text
    assert "a stiffness on all three axes of a spring, a joint, a second spring on another face or a fixed region" in text
    assert "take more nodes into a fixed region" in text and "Options:" in text
    assert e.value.facts["fixed_nodes"] == 0 and e.value.facts["bodies"] == 1


def test_the_singular_text_of_a_plain_part_is_the_old_text():
    from topopt.verify.factor import singular_message
    plain = singular_message("superlu", "zero pivot", 100, "The mesh has 1 body (each has a fixed node) and 5 fixed nodes. ")
    assert "joint" not in plain and "spring" not in plain and "take more nodes into a fixed region;" in plain
    held = singular_message("superlu", "zero pivot", 100, "", supports=True)
    assert "a spring with stiffness on one axis leaves the other two axes free" in held and "a joint or spring region with few facets" in held


# --------------------------------------------------------------------------- #
# The memory estimate counts the bordered copy (REVIEW-135 L3)
# --------------------------------------------------------------------------- #

def test_the_bordered_copy_is_a_value_and_an_index_for_each_entry_of_kff():
    from topopt.verify.supports_p2 import bordered_copy_bytes
    assert bordered_copy_bytes(1000) == 12_000 and bordered_copy_bytes(0) == 0


@needs_gmsh
def test_the_memory_estimate_grows_by_the_bordered_copy_for_joints_and_not_for_springs(tmp_path):
    from topopt.verify import memory, verify_run
    res = verify_run(_bar_run_with(tmp_path / "j", None), mesh_size=2.0, solver="superlu", load_sensitivity=False)
    mem = res["memory"]
    plain = memory.estimate_bytes(res["mesh"]["dofs"], "superlu", res["estimates_before_meshing"][0]["compactness"])
    assert mem["estimate_bordered_copy_bytes"] > 0 and mem["estimate_bordered_copy_bytes"] % 12 == 0
    assert mem["estimate_after_meshing_bytes"] == plain + mem["estimate_bordered_copy_bytes"]
    assert mem["measured_over_estimate"] == pytest.approx(mem["peak_rss_bytes"] / mem["estimate_after_meshing_bytes"])
    sp = verify_run(_bar_run_with(tmp_path / "s", lambda d: d.update(joints=None, springs=[{"region": r, "stiffness": [2000.0] * 3} for r in ("padl", "padr")])),
                    mesh_size=2.0, solver="superlu", load_sensitivity=False)                # a spring has no border: no copy
    assert "estimate_bordered_copy_bytes" not in sp["memory"]
    assert sp["memory"]["estimate_after_meshing_bytes"] == memory.estimate_bytes(sp["mesh"]["dofs"], "superlu", sp["estimates_before_meshing"][0]["compactness"])


@needs_gmsh
def test_a_limit_below_the_estimate_with_the_bordered_copy_stops_with_the_options(tmp_path):
    """The estimate of the plain model passes the guard and the one with the bordered copy does not: the run stops with the usual memory text."""
    from topopt.verify import memory, verify_run
    from topopt.verify.errors import MemoryRefused
    run = _bar_run_with(tmp_path, None)
    res = verify_run(run, mesh_size=2.0, solver="superlu", load_sensitivity=False)
    plain, extra = res["memory"]["estimate_after_meshing_bytes"] - res["memory"]["estimate_bordered_copy_bytes"], res["memory"]["estimate_bordered_copy_bytes"]
    limit_gb = (plain + extra / 2) / 1e9                                    # between the two estimates
    with pytest.raises(MemoryRefused) as e:
        verify_run(run, mesh_size=2.0, solver="superlu", load_sensitivity=False, max_gb=limit_gb)
    assert "with the bordered copy of K" in str(e.value) and "Options:" in str(e.value)


# --------------------------------------------------------------------------- #
# The near-free facts of a joint (REVIEW-135 L4)
# --------------------------------------------------------------------------- #

@needs_gmsh
def test_the_near_free_facts_of_a_joint_are_those_of_the_voxel_model(tmp_path, mixed_run):
    """verify and the voxel model call `joints.beam_facts`, so the numbers that depend on the beam only are equal, and `rho` (the bond over the
    beam) follows the bond area. The `run.json` is the one of a real voxel run: `yaml_vs_run` of the same YAML must match."""
    from topopt.verify import verify_run
    yml, run_json = mixed_run
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    box(run / "result" / "final.stl", (-2, 0, 0), (22, 10, 10))
    (run / "run.json").write_text(json.dumps(dict(run_json, yaml_path=str(yml))))
    res = verify_run(run, mesh_size=2.0, solver="superlu", load_sensitivity=False)
    v, t = run_json["joints"]["joints"][0], res["supports"]["joints"][0]
    assert t["depth_mm"] == v["depth_mm"] and "voxel run" in t["frame"]["y_from"]              # the section; the y axis is the one of the voxel run
    assert t["frame"]["y"] == pytest.approx(v["frame"]["y"], abs=0.02)
    for axis in "xyz":                                                                                # the beam axis follows the reference point
        assert t["frame"][axis] == pytest.approx(v["frame"][axis], abs=0.02), axis
    print(f"\nreference voxel {v['reference_mm']}, tet {t['reference_mm']}")
    for key in ("length_over_depth", "beam_min_translation_over_Eh", "beam_min_rotation_over_Eh3", "rho"):
        print(f"{key}: tet {t[key]:.6g}, voxel {v[key]:.6g}, ratio {t[key] / v[key]:.4f}")
        # The reference point (and so the beam length) and the bonded area differ a little; `rho` is a ratio of the order of 1e4 and the
        # rotation about the beam axis follows the second moment of the bonded area (measured: tet / voxel 0.835), so the band is 25 %.
        assert t[key] == pytest.approx(v[key], rel=0.25), key
    assert res["supports"]["near_singular_lines"] == []
    assert res["yaml_vs_run"]["matches"], res["yaml_vs_run"]["differences"]


def test_a_nearly_free_beam_gives_the_line_of_topopt_check_in_the_record():
    from topopt.verify import supports_p2
    m, left, _right = block()
    cfg = SimpleNamespace(material=SimpleNamespace(E=3500.0, nu=0.35), springs=(),
                          joints=(JointSpec("left", (-100.0, 5.0, 5.0), SectionSpec("round", (0.2,)), None, None, None, None, None),))
    sup = supports_p2.build_supports(m, cfg, {"left": left}, 1.0)
    f = sup.joints[0].facts
    assert f["beam_min_translation_over_Eh"] < 1e-6 and f["length_over_depth"] == pytest.approx(105.0 / 0.2, rel=0.05)
    lines = sup.facts["near_singular_lines"]
    assert len(lines) == 1 and "joint on 'left'" in lines[0] and "nearly free" in lines[0] and "Options:" in lines[0]
    stiff = SimpleNamespace(material=SimpleNamespace(E=3500.0, nu=0.35), springs=(),
                            joints=(JointSpec("left", (-30.0, 5.0, 5.0), SectionSpec("round", (8.0,)), None, None, None, None, None),))
    assert supports_p2.build_supports(m, stiff, {"left": left}, 1.0).facts["near_singular_lines"] == []


def test_a_joint_with_no_orient_takes_the_section_axis_of_the_voxel_run():
    """The default rule ties when the beam lies along an axis; the tet reference point is not the voxel one, so it can pick the other axis.
    A rect beam 1 x 6 mm is 36 times stiffer in one bending plane: the axis of the voxel run must hold (found while copying the L4 facts)."""
    from topopt.joints import beam_frame, beam_stiffness, section_properties
    from topopt.verify import supports_p2
    m, left, _right = block()
    sec = SectionSpec("rect", (1.0, 6.0))
    spec = JointSpec("left", (-30.0, 5.02, 5.0), sec, None, None, None, None, None)           # the beam is 0.02 mm off the x axis in y: the tie breaks to z
    cfg = SimpleNamespace(material=SimpleNamespace(E=3500.0, nu=0.35), springs=(), joints=(spec,))
    default = supports_p2.build_supports(m, cfg, {"left": left}, 1.0).joints[0]
    held = supports_p2.build_supports(m, cfg, {"left": left}, 1.0, vox_y={"left": [0.0, 1.0, 0.0]}).joints[0]
    assert abs(default.facts["frame"]["y"][2]) > 0.99 and "default" in default.facts["frame"]["y_from"]            # the noise decided
    assert abs(held.facts["frame"]["y"][1]) > 0.99 and "voxel run" in held.facts["frame"]["y_from"]
    want = beam_stiffness(3500.0, 0.35, section_properties(sec), held.facts["length_mm"], beam_frame(held.far - held.ref, [0.0, 1.0, 0.0]))
    assert held.B == pytest.approx(want, rel=1e-12) and not np.allclose(default.B, held.B, rtol=0.1)
    own = JointSpec("left", (-30.0, 5.02, 5.0), sec, None, (0.0, 0.0, 1.0), None, None, None)                    # an `orient` of the entry wins
    kept = supports_p2.build_supports(m, SimpleNamespace(material=cfg.material, springs=(), joints=(own,)), {"left": left}, 1.0,
                                      vox_y={"left": [0.0, 1.0, 0.0]}).joints[0]
    assert abs(kept.facts["frame"]["y"][2]) > 0.99 and kept.facts["frame"]["y_from"] == "orient"
    bad = supports_p2.build_supports(m, cfg, {"left": left}, 1.0, vox_y={"left": [-1.0, 0.0, 0.0]}).joints[0]    # parallel to the beam: the default rule
    assert "default" in bad.facts["frame"]["y_from"]


# --------------------------------------------------------------------------- #
# The press-fit stage with springs and with joints (REVIEW-135 L5, Q4)
# --------------------------------------------------------------------------- #

def _hook_with_supports(tmp_path: Path, supports: dict, name: str) -> Path:
    """The hook YAML with `fixed: []`, the wall plate as a load region, the given `joints` or `springs`, and a press-fit bore on the channel."""
    from test_verify_bore_run import PRESS, block
    from test_verify_run import HOOK_YAML
    d = yaml.safe_load(HOOK_YAML.read_text())
    base = HOOK_YAML.parent
    r = d["regions"]
    r["design"] = str(base / r["design"])
    plate = str(base / r["fixed"][0])
    r.update(fixed=[], keep_out=[str(base / x) for x in r["keep_out"]])
    r["load"] = {**{k: str(base / v) for k, v in r["load"].items()}, "plate": plate}
    d.update(supports)
    d["bore_checks"] = block(press_fit=PRESS)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d))
    return p


def _hook_verify(tmp_path: Path, yml: Path, *argv):
    import contextlib
    import io

    from test_verify_run import copy_run
    from topopt.cli import main
    run = copy_run(tmp_path / "run")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(run), "--yaml", str(yml), "--mesh-size", "3", "--no-sensitivity", "--solver", "superlu", *argv])
    return code, json.loads((run / "verify" / "verify.json").read_text()), out.getvalue(), err.getvalue()


@needs_gmsh
def test_a_part_held_by_springs_only_keeps_the_press_fit_stage(tmp_path):
    """The stiffness matrix K of the stage holds the springs, so the stage solves the part that the springs hold (measured: the relative residual
    of the variant is 3e-17). The record says that the clamp reaction is that of the clamps only, which is zero here."""
    from test_verify_run import needs_recording
    if needs_recording.args[0]:
        pytest.skip("needs the recorded hook run")
    yml = _hook_with_supports(tmp_path, {"springs": [{"region": "plate", "stiffness_per_area": 2000.0}]}, "springs.yaml")
    code, res, text, err = _hook_verify(tmp_path, yml)
    assert code == 0, err
    pf = res["pressfit"]
    assert pf["status"] == "done" and [v["name"] for v in pf["variants"]] == ["rigid"]
    v = pf["variants"][0]
    assert v["solve"]["relative_residual"] < 1e-9 and v["solve"]["clamp_reaction_N"] == [0.0, 0.0, 0.0]
    assert np.linalg.norm(v["bores"][0]["net_spring_force_N"]) > 1.0                       # the cup pushes the part; the springs carry it
    assert any("with a spring support it does not balance net_spring_force_N" in t for t in pf["facts"])
    assert res["goodman"]["status"] == "done" and res["goodman"]["missing_rows"] == []     # sm comes from the press fit again
    assert "pressfit: skipped" not in text and "pressfit (1 cup variant(s)" in text and "press fit rigid: channel_top" in err
    assert "The springs of the YAML are in K" in text


@needs_gmsh
def test_a_part_with_a_joint_skips_the_press_fit_stage_with_a_record_and_says_it_before_the_mesh(tmp_path):
    from test_verify_run import copy_run, needs_recording
    from topopt.verify.report import estimate_lines
    from topopt.verify.run import verify_run
    if needs_recording.args[0]:
        pytest.skip("needs the recorded hook run")
    yml = _hook_with_supports(tmp_path, {"joints": [{"region": "plate", "to": [-30.0, 0.0, 3.0], "section": {"kind": "round", "d": 6.0}}]}, "joint.yaml")
    code, res, text, err = _hook_verify(tmp_path, yml)
    assert code == 0, err
    pf = res["pressfit"]
    assert pf["status"] == "skipped" and pf["reason"] == "supports" and pf["variants"] == []
    assert any("does not take the border of the joints" in t for t in pf["facts"]) and any("without joints" in t for t in pf["options"])
    assert "pressfit: skipped (supports)" in text and "press fit: skipped (the YAML has a joint)" in err
    assert res["goodman"]["status"] == "done"
    # the estimate (no mesh yet) says it too, in the text and in the key of the entry
    est = verify_run(copy_run(tmp_path / "run2"), yaml_path=yml, mesh_size=3, estimate_only=True)
    e = est["estimates_before_meshing"][0]["pressfit"]
    assert "does not take the border of the joints" in e["skipped_with_joints"]
    assert any("with the press-fit check: skipped, the YAML has a joint." in x for x in estimate_lines(est))
