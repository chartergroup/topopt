"""K1-C: the mode facts, the rigid-body test and the alignment of a floating part (`topopt/verify/buckling_p2.py`, `motion.py`).

The meshes are structured tets (`verify_mesh.box_mesh`, no gmsh), so these tests run in a few seconds:
- an Euler column (clamped, free end loaded): a global mode, the largest motion in the region of the free end;
- a thin fin on a block, loaded at the fin top: a local mode, the largest motion in the fin region;
- a column held by one joint and nothing else (`fixed: []`): the soft beam gives modes that are a motion of the supports. The bordered stiffness
  `S` is positive definite and still has them, so the rigid-body test is needed. A stiff beam gives none and the test does not run;
- the alignment of two seat regions of a part that floats: a rigid motion gives no tilt and no shift, a known relative tilt is found within 1 %.
"""
import itertools
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as sla

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_gmsh, needs_skfem  # noqa: E402
from verify_mesh import box_mesh  # noqa: E402

from topopt.config import Buckling, JointSpec, SectionSpec, SpringSpec  # noqa: E402

pytestmark = needs_skfem

E, NU = 110000.0, 0.34
LENGTH, SIDE = 100.0, 5.0


def check(model, free, Kff, fac, f_full, region_masks, supports=None, ss=None, sup=None, modes=4, held=None):
    """`run_check` on one case with the load vector `f_full` (all dofs). Return the case record and the whole record.

    `held` is as in `run_check`; the default is as in `verify.run`: the part is held when it has a support set (joints or springs).
    """
    from topopt.verify import buckling_p2
    from topopt.verify.p2 import STRESS_ORDER
    pad = np.zeros(Kff.shape[0] if ss is None else ss.A.shape[0])
    pad[:len(free)] = f_full[free]
    u = np.zeros(model.N)
    u[free] = np.asarray(fac.solve(pad))[:len(free)]
    sig = model.stress_qp(u, E, NU, STRESS_ORDER)[0]
    tmp = Path(tempfile.mkdtemp())
    (tmp / "result").mkdir()
    cfg = SimpleNamespace(buckling=Buckling(check=True, modes=modes, max_seconds=120.0), load_cases=[SimpleNamespace(name="push")])
    rec = buckling_p2.run_check(model, cfg, SimpleNamespace(run_dir=tmp), fac=fac if ss is None else ss.grid_solver(fac),
                                Kff=Kff if ss is None else ss.stiffness(), free=free, sigs={"push": sig}, cases={}, solver="superlu",
                                E=E, nu=NU, mem={}, supports=supports, region_masks=region_masks, held=sup is not None if held is None else held)
    return rec["cases"][0], rec


def top_facets(model, z):
    from topopt.verify import regions
    bf = regions.boundary_facets(model.t.T)
    return bf[np.all(model.p[2][bf] > z - 1e-9, axis=1)]


def clamped(model, f_value, masks, modes=4):
    """A part clamped at z = 0 with the traction `f_value` on the facets at the top z."""
    from topopt.verify import bc
    from topopt.verify.factor import make_factor
    fixed = bc.fixed_dofs(np.flatnonzero(model.node_xyz[2] < 1e-9))
    free = np.setdiff1d(np.arange(model.N), fixed)
    Kff = model.stiffness(E, NU)[free][:, free].tocsr()
    fac = make_factor(Kff, "superlu")
    try:
        return check(model, free, Kff, fac, f_value, masks, modes=modes)[0]
    finally:
        fac.free()


# --------------------------------------------------------------------------- #
# Global and local modes
# --------------------------------------------------------------------------- #

def test_the_euler_column_has_a_global_mode_and_the_largest_motion_is_at_the_free_end():
    from topopt.verify import bc
    from topopt.verify.p2 import P2Model
    p, t = box_mesh((SIDE, SIDE, LENGTH), (2, 2, 40))
    m = P2Model(p, t)
    pcr = np.pi ** 2 * E * (SIDE ** 4 / 12.0) / (2.0 * LENGTH) ** 2
    f, _ = bc.traction_load(m, top_facets(m, LENGTH), [0.0, 0.0, -pcr])
    X = m.node_xyz.T
    rec = clamped(m, f, {"clamp": X[:, 2] < 1e-9, "tip": X[:, 2] > LENGTH - 1e-9})
    assert rec["status"] == "ok" and rec["load_factors"][0] == pytest.approx(1.0, abs=3e-3)
    mf = rec["mode_facts"]
    assert len(mf) == len(rec["load_factors"]) == 4
    low = mf[0]
    print(f"\ncolumn: mode 1 {low['kind']}, energy volume share {low['energy_volume_share']}, largest motion {low['largest_motion']}")
    assert low["kind"] == "global" and low["energy_volume_share"] > 0.25
    assert low["load_factor"] == rec["load_factors"][0]
    lm = low["largest_motion"]
    assert lm["region"] == "tip" and lm["position_mm"][2] == pytest.approx(LENGTH)
    assert lm["region_rms_over_peak"]["tip"] > 0.99 and lm["region_rms_over_peak"]["clamp"] == 0.0
    assert "rigid_body_fraction" not in low and "support_energy_share" not in low           # a clamped part has no support facts
    assert "rigid_body" not in rec


def grid_mesh(cells, keep):
    """Cubes of size 1 mm, `cells` = (nx, ny, nz), kept where `keep(i, j, k)`; 6 tets each. Return `p` (3, nv) and `t` (4, nel)."""
    from verify_mesh import CUBE_TETS
    ids, tets = {}, []
    for i, j, k in itertools.product(*(range(n) for n in cells)):
        if not keep(i, j, k):
            continue
        corner = [(i + (v & 1), j + ((v >> 1) & 1), k + ((v >> 2) & 1)) for v in range(8)]
        for c in corner:
            ids.setdefault(c, len(ids))
        tets.extend([[ids[corner[v]] for v in tet] for tet in CUBE_TETS])
    return np.array(list(ids), float).T.copy(), np.array(tets, np.int64).T.copy()


def test_a_thin_fin_on_a_block_has_a_local_mode_at_the_fin():
    """A 16 x 16 x 4 mm block and a fin of 16 x 1 x 10 mm, loaded at the fin top. The fin is 13 % of the volume."""
    from topopt.verify import bc
    from topopt.verify.p2 import P2Model
    p, t = grid_mesh((16, 16, 14), lambda i, j, k: k < 4 or j == 7)
    m = P2Model(p, t)
    f, _ = bc.traction_load(m, top_facets(m, 14.0), [0.0, 0.0, -1.0])
    X = m.node_xyz.T
    rec = clamped(m, f, {"foot": X[:, 2] < 1e-9, "fin": X[:, 2] > 4.5, "block": X[:, 2] < 4.0 + 1e-9})
    assert rec["status"] == "ok"
    low = rec["mode_facts"][0]
    print(f"\nfin: lowest factor {rec['load_factors'][0]:.4g}, mode 1 {low['kind']}, energy volume share {low['energy_volume_share']}, "
          f"largest motion {low['largest_motion']}")
    assert low["kind"] == "local" and low["energy_volume_share"] < 0.25
    lm = low["largest_motion"]
    assert lm["region"] == "fin" and lm["position_mm"][2] > 12.0
    assert lm["region_rms_over_peak"]["fin"] > lm["region_rms_over_peak"]["block"]


def test_a_node_in_no_named_region_is_the_design_region():
    from topopt.verify import bc
    from topopt.verify.p2 import P2Model
    p, t = box_mesh((SIDE, SIDE, LENGTH), (2, 2, 20))
    m = P2Model(p, t)
    f, _ = bc.traction_load(m, top_facets(m, LENGTH), [0.0, 0.0, -1.0])
    rec = clamped(m, f, {"clamp": m.node_xyz[2] < 1e-9}, modes=2)
    assert rec["mode_facts"][0]["largest_motion"]["region"] == "design"
    assert set(rec["mode_facts"][0]["largest_motion"]["region_rms_over_peak"]) == {"clamp", "design"}


# --------------------------------------------------------------------------- #
# A part that floats on a joint: the rigid-body test
# --------------------------------------------------------------------------- #

class Floating:
    """A 5 x 5 x 60 column, a joint on its bottom face, the Euler load at the top, no fixed node.

    With `spring_k` (N/mm, the total in each axis) a spring on the bottom face holds it in place of the joint (the beam arguments are not used).
    """

    def __init__(self, section_d, beam_len, beam_E, cells=(2, 2, 24), spring_k=None):
        from topopt.verify import bc, regions, supports_p2
        from topopt.verify.factor import make_factor
        from topopt.verify.p2 import P2Model
        p, t = box_mesh((SIDE, SIDE, 60.0), cells)
        self.m = m = P2Model(p, t)
        bf = regions.boundary_facets(m.t.T)
        c = m.p.T[bf].mean(axis=1)

        def sel(mask):
            fc = bf[mask]
            return SimpleNamespace(facets=fc, facet_area=regions.facet_areas(m.p.T, fc))
        bot, top = sel(np.abs(c[:, 2]) < 1e-9), sel(np.abs(c[:, 2] - 60.0) < 1e-9)
        if spring_k is None:
            cfg = SimpleNamespace(material=SimpleNamespace(E=E, nu=NU), springs=(), joints=(
                JointSpec("bot", (2.5, 2.5, -beam_len), SectionSpec("round", (section_d,)), None, None, None, beam_E, 0.3),))
        else:
            cfg = SimpleNamespace(material=SimpleNamespace(E=E, nu=NU), joints=(), springs=(SpringSpec("bot", stiffness=(spring_k,) * 3),))
        self.sup = supports_p2.build_supports(m, cfg, {"bot": bot}, 1.0)
        self.K = (m.stiffness(E, NU) + self.sup.Ks).tocsr()
        self.free = np.arange(m.N)
        self.ss = self.sup.system(self.K, self.free)
        self.fac = make_factor(self.ss.A, "superlu")
        pcr = np.pi ** 2 * E * (SIDE ** 4 / 12.0) / (2.0 * 60.0) ** 2
        self.f, _ = bc.traction_load(m, top.facets, [0.0, 0.0, -pcr])
        X = m.node_xyz.T
        self.masks = {"bot": X[:, 2] < 1e-9, "top": X[:, 2] > 60.0 - 1e-9}

    def run(self, **kw):
        return check(self.m, self.free, self.ss.Kff, self.fac, self.f, self.masks, self.sup.buckling_note(), self.ss, self.sup, **kw)

    def close(self):
        self.fac.free()


@pytest.fixture(scope="module")
def soft():
    s = Floating(10.0, 30.0, 2100.0)
    yield s
    s.close()


@pytest.fixture(scope="module")
def stiff():
    s = Floating(40.0, 10.0, 110000.0)
    yield s
    s.close()


def test_the_bordered_stiffness_is_positive_definite_and_the_soft_beam_still_gives_motions_of_the_supports():
    """S has no zero eigenvalue, but a part on a soft beam moves as one body at a small load: the lowest modes are not modes of the part."""
    s = Floating(10.0, 30.0, 2100.0, cells=(1, 1, 8))
    try:
        S = s.ss.stiffness().toarray()
        smallest = sla.eigvalsh(S, subset_by_index=[0, 0])[0]
        rec, _ = s.run()
        s_rigid = rec["mode_facts"][0]
    finally:
        s.close()
    print(f"\nsmallest eigenvalue of S: {smallest:.4g} N/mm; lowest mode: rigid {s_rigid['rigid_body_fraction']}, "
          f"support share {s_rigid['support_energy_share']}, factor {rec['load_factors'][0]:.4g}")
    assert smallest > 0.0
    assert s_rigid["rigid_body_fraction"] > 0.99 and s_rigid["support_energy_share"] > 0.5
    assert rec["load_factors"][0] < 0.3                    # the Euler factor of the column on a stiff support is 1
    assert rec["rigid_body"]["applied"] is True


def test_the_soft_beam_runs_the_rigid_body_test_and_the_stiff_beam_does_not(soft, stiff):
    rec, whole = soft.run()
    rb = rec["rigid_body"]
    low, projected = rec["load_factors"][0], rb["load_factors"][0]
    print(f"\nsoft beam: lowest factor {low:.5g}, without the rigid-body motions {projected:.5g} (ratio {rb['lowest_over_unprojected']:.1f})")
    assert rb["status"] == "ok" and projected > 10.0 * low
    assert rb["mode_facts"][0]["support_energy_share"] < 0.01              # with the projection the energy is in the part
    low_facts = rec["mode_facts"][0]
    assert low_facts["support_energy_share"] > 0.5 and low_facts["kind"] is None and "supports hold" in low_facts["kind_note"]       # D8: the tets hold the rest only
    assert low_facts["energy_volume_share"] is not None and rb["mode_facts"][0]["kind"] in ("local", "global") and "kind_note" not in rb["mode_facts"][0]
    assert rb["mode_facts"][0]["rigid_body_fraction"] > 0.5               # D8: orthogonal in the S inner product, not in the plain one; the rule text says so
    assert "can be far from zero" in whole["rigid_body_rule"]
    assert any("motion of the part on its supports" in x for x in rec["facts"])
    assert whole["rigid_body_rule"].startswith("The test runs when the part has joints or springs and no fixed node")
    assert "geometric stiffness" in whole["rigid_body_rule"]
    stiff_rec, stiff_whole = stiff.run()
    sf = stiff_rec["mode_facts"][0]
    print(f"stiff beam: lowest factor {stiff_rec['load_factors'][0]:.5g}, rigid {sf['rigid_body_fraction']}, support share {sf['support_energy_share']}")
    assert sf["support_energy_share"] < 0.01 and "rigid_body" not in stiff_rec
    assert sf["kind"] in ("local", "global") and "kind_note" not in sf     # the supports hold little of the energy: kind describes the mode
    assert stiff_rec["load_factors"][0] == pytest.approx(1.0, abs=2e-2)    # the clamped Euler column
    assert not any("motion of the part on its supports" in x for x in stiff_rec["facts"])
    assert stiff_whole["rigid_body_rule"].startswith("The test runs")      # the rule is in the record, the test did not start


def test_a_part_that_springs_hold_runs_the_rigid_body_test_too():
    """Q3 of the review of #137: the swing on springs is the same swing. `buckling_note` is None without a joint, so `run_check` takes `held`."""
    s = Floating(None, None, None, cells=(1, 1, 8), spring_k=1000.0)
    try:
        rec, whole = s.run()
    finally:
        s.close()
    low = rec["mode_facts"][0]
    print(f"\nsprings only: lowest factor {rec['load_factors'][0]:.4g}, support share {low['support_energy_share']}, "
          f"without the rigid-body motions {rec['rigid_body']['load_factors'][0]:.4g}")
    assert "supports" not in whole           # no beam: no `supports` note (it is about the beams)
    assert whole["rigid_body_rule"].startswith("The test runs when the part has joints or springs and no fixed node")
    assert low["support_energy_share"] > 0.5 and low["rigid_body_fraction"] > 0.99 and low["kind"] is None
    rb = rec["rigid_body"]
    assert rb["applied"] is True and rb["status"] == "ok" and rb["load_factors"][0] > 10.0 * rec["load_factors"][0]
    assert any("motion of the part on its supports" in x for x in rec["facts"])


def test_the_thresholds_are_named_once_and_the_rule_texts_are_built_from_them():
    """D3 of the review of #137: no number typed twice, and a change of a constant changes the text."""
    import re
    from topopt.verify import buckling_p2 as b
    numbers = lambda text: sorted(set(re.findall(r"\d+(?:\.\d+)?", text)))      # noqa: E731
    assert numbers(b.MODE_RULE) == sorted({f"{100 * b.MODE_ENERGY_SHARE:g}", f"{100 * b.LOCAL_VOLUME_SHARE:g}", f"{b.SUPPORT_ENERGY_SHARE:g}"})
    assert numbers(b.RIGID_RULE) == [f"{b.SUPPORT_ENERGY_SHARE:g}"]
    assert b.MODE_RULE == b.mode_rule() and b.RIGID_RULE == b.rigid_rule()
    other = b.mode_rule(0.8, 0.3, 0.6)
    assert "hold 80 % of its strain energy" in other and "at most 30 %" in other and "0.6 or more" in other and "90 %" not in other
    assert "support_energy_share >= 0.7" in b.rigid_rule(0.7) and "0.5" not in b.rigid_rule(0.7)
    assert "first values from the author's measurements on two test parts" in b.MODE_RULE
    assert (b.MODE_ENERGY_SHARE, b.LOCAL_VOLUME_SHARE, b.SUPPORT_ENERGY_SHARE) == (0.9, 0.25, 0.5)         # the first values; a change is a decision
    assert len({"MODE_ENERGY_SHARE", "LOCAL_VOLUME_SHARE", "SUPPORT_ENERGY_SHARE"} & set(vars(b))) == 3
    assert not any(hasattr(b, old) for old in ("RIGID_SHARE", "ENERGY_SHARE", "LOCAL_VOLUME"))              # one constant for two quantities is gone


def test_the_fact_paths_of_the_buckling_symptom_are_keys_of_a_real_record_and_the_documents_say_what_the_code_does(soft):
    """Scheme fit of the review of #137: the symptom on `verify_run` names paths that a real verify.json has (CONTRACTS.md 41.7); the texts match the code."""
    from pathlib import Path as P
    from topopt import levers_cards as lc
    from topopt import levers_run as lr
    from topopt.verify import buckling_p2 as b
    rec, whole = soft.run()
    doc = {"buckling": dict(whole, cases=[dict(rec, name="push")])}
    card = next(c for c in lc.CARDS if c.key == "verify_run" and c.group == "tool")
    symptom = next(sy for sy in card.symptoms if "buckling mode" in sy.words)
    assert symptom.facts == ("verify.buckling.lowest_load_factor", "verify.buckling.cases[].mode_facts", "verify.buckling.cases[].rigid_body")
    for path in symptom.facts:
        source, segs = lr._split(path)
        assert source == "verify" and list(lr._match(doc, segs)), path
    assert "`verify.buckling.cases[].rigid_body` exists" in symptom.when
    assert set(symptom.facts) <= {x for paths in lr.NOT_FLAGGED.values() for x in paths}          # facts of the shape of a mode: no engine flag
    root = P(__file__).resolve().parent.parent
    contracts = (root / "CONTRACTS.md").read_text(encoding="utf-8")
    section = " ".join(contracts[contracts.index("*Mode facts (added with K1-C).*"):contracts.index("*Loose piece.*")].split())
    for text in (f"{100 * b.MODE_ENERGY_SHARE:g} %", f"{100 * b.LOCAL_VOLUME_SHARE:g} %", f"support_energy_share >= {b.SUPPORT_ENERGY_SHARE:g}",
                 "MODE_ENERGY_SHARE", "LOCAL_VOLUME_SHARE", "SUPPORT_ENERGY_SHARE", "first values from the author's measurements on two test parts",
                 "they are the numbers that the gate", "a fact next to them"):
        assert text in section, text
    assert "not tuned" not in section and "rigid_body_fraction >= 0.5" not in section and "read `rigid_body.load_factors[0]`" not in contracts
    assert "3 to 4 %" in section and "2.7 % and 4.2 %" in section and "can be far from zero" in section                  # D7 and D8
    assert "springs alone hold runs it" in section and "(`Kg` of the stress state acts on a rigid motion" in section
    readme = " ".join((root / "docs" / "reference" / "verify.md").read_text(encoding="utf-8").split())
    assert "`rigid_body.load_factors` is the factor of the part itself, a fact next to it" in readme and "3 to 4 %" in readme


def test_the_trigger_reads_the_support_energy_share_only():
    """The rigid_body_fraction was 0.98 or more on every part of the review, so a condition on it decided nothing."""
    from topopt.verify import buckling_p2 as b
    assert b.rule_trigger([{"support_energy_share": 0.5}]) and b.rule_trigger([{"support_energy_share": 0.9, "rigid_body_fraction": 0.1}])
    assert not b.rule_trigger([{"support_energy_share": 0.49, "rigid_body_fraction": 1.0}])
    assert not b.rule_trigger([]) and not b.rule_trigger([{}]) and not b.rule_trigger([{"rigid_body_fraction": 1.0}])


def test_a_projected_solve_that_gives_no_factor_or_fails_leaves_the_measured_factors_and_says_why(soft, monkeypatch):
    """D6 of the review of #137: the projected solve is an extra. It must not take the measured factors of the case away."""
    from topopt.verify import buckling_p2, gate
    real = buckling_p2.bk.buckling_modes
    want, _ = soft.run()

    def not_converged(Kg, *a, **kw):
        res = real(Kg, *a, **kw)
        return dict(res, status=buckling_p2.bk.NOT_CONVERGED) if isinstance(Kg, buckling_p2._Projected) else res

    def boom(Kg, *a, **kw):
        if isinstance(Kg, buckling_p2._Projected):
            raise RuntimeError("test failure")
        return real(Kg, *a, **kw)

    for fake, status, text in ((not_converged, "not_converged", "gave the status not_converged"), (boom, "error", "RuntimeError: test failure")):
        monkeypatch.setattr(buckling_p2.bk, "buckling_modes", fake)
        rec, _ = soft.run()
        assert rec["status"] == "ok" and rec["load_factors"] == want["load_factors"]            # the measured factors stay
        assert rec["rigid_body"]["applied"] is True and rec["rigid_body"]["status"] == status and "mode_facts" not in rec["rigid_body"]
        assert any(text in x for x in rec["facts"])
        lv = dict(size_mm=1.0, cases={}, buckling=dict(status="done", cases=[dict(rec, name="a")]))
        lim = dict(stress={"a": dict(limit=None, source=None)}, rows=[], buckling=dict(limit=0.05, source="buckling.min_factor"), applies_load={"a": True})
        entry = gate.evaluate(lv, lim, percentile=99.9)["checks"]["buckling"]["a"]
        assert entry["value"] == want["load_factors"][0] and entry["met"] is True
        assert entry["projected_value"] is None and status in entry["projected_reason"] and "no projected value" in entry["projected_reason"]


def test_the_projected_factors_equal_a_dense_solve_with_the_projector():
    """`P Kg P` against `S` on a small mesh: the factors of the check are the positive eigenvalues of the dense problem."""
    from topopt.verify import buckling_p2
    from topopt.verify.p2 import STRESS_ORDER
    s = Floating(10.0, 30.0, 2100.0, cells=(1, 1, 8))
    try:
        rec, _ = s.run(modes=4)
        pad = np.zeros(s.ss.A.shape[0])
        pad[:s.m.N] = s.f
        u = np.asarray(s.fac.solve(pad))[:s.m.N]
        Kg = s.m.geometric_stiffness(s.m.stress_qp(u, E, NU, STRESS_ORDER)[0]).toarray()
        R = buckling_p2.rigid_body_basis(s.m.node_xyz)
        P = np.eye(s.m.N) - R @ np.linalg.solve(R.T @ R, R.T)
        mu = np.sort(sla.eigh(-(P @ Kg @ P), s.ss.stiffness().toarray(), eigvals_only=True))[::-1]
        want = 1.0 / mu[:4]
    finally:
        s.close()
    got = rec["rigid_body"]["load_factors"][:4]
    assert got == pytest.approx(list(want), rel=1e-6)


def test_the_rigid_body_basis_spans_the_six_motions():
    from topopt.verify import buckling_p2
    X = np.random.default_rng(3).uniform(0, 50, size=(3, 40))
    R = buckling_p2.rigid_body_basis(X)
    assert R.shape == (120, 6) and np.linalg.matrix_rank(R) == 6
    t, th, c = np.array([0.2, -0.1, 0.3]), np.array([1e-3, 2e-3, -1e-3]), X.mean(axis=1)
    u = (t + np.cross(th, (X.T - c))).ravel()
    coef, *_ = np.linalg.lstsq(R, u, rcond=None)
    assert np.linalg.norm(u - R @ coef) < 1e-12 * np.linalg.norm(u)


def test_a_part_with_a_fixed_node_does_not_run_the_rigid_body_test():
    """A joint on the top and a clamp at the bottom: the part does not float, so the key and the test are absent."""
    from topopt.verify import bc, regions, supports_p2
    from topopt.verify.factor import make_factor
    from topopt.verify.p2 import P2Model
    p, t = box_mesh((SIDE, SIDE, 60.0), (2, 2, 12))
    m = P2Model(p, t)
    bf = regions.boundary_facets(m.t.T)
    c = m.p.T[bf].mean(axis=1)
    fc = bf[np.abs(c[:, 2] - 60.0) < 1e-9]
    top = SimpleNamespace(facets=fc, facet_area=regions.facet_areas(m.p.T, fc))
    cfg = SimpleNamespace(material=SimpleNamespace(E=E, nu=NU), springs=(), joints=(
        JointSpec("top", (2.5, 2.5, 90.0), SectionSpec("round", (10.0,)), None, None, None, 2100.0, 0.3),))
    sup = supports_p2.build_supports(m, cfg, {"top": top}, 1.0)
    K = (m.stiffness(E, NU) + sup.Ks).tocsr()
    fixed = bc.fixed_dofs(np.flatnonzero(m.node_xyz[2] < 1e-9))
    free = np.setdiff1d(np.arange(m.N), fixed)
    ss = sup.system(K[free][:, free], free)
    fac = make_factor(ss.A, "superlu")
    try:
        f, _ = bc.traction_load(m, fc, [0.0, 0.0, -50.0])
        rec, whole = check(m, free, ss.Kff, fac, f, {}, sup.buckling_note(), ss, sup)
    finally:
        fac.free()
    assert rec["status"] == "ok" and "rigid_body" not in rec
    assert "rigid_body_fraction" not in rec["mode_facts"][0] and "support_energy_share" in rec["mode_facts"][0]
    assert whole["rigid_body_rule"].startswith("The rigid-body test did not run: the part has a fixed node")


# --------------------------------------------------------------------------- #
# Alignment of two seats on a part that floats (motion.py, bind_nodes)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def floating_part(tmp_path_factory):
    """The cantilever of `test_verify_motion` with no fixed dof: all dofs are free, as for a part held only by joints."""
    from test_verify_motion import Cantilever
    c = Cantilever(tmp_path_factory.mktemp("seats"))
    c.free = np.arange(c.model.N)
    return c


def rigid_field(floating, t, theta, centre):
    return np.array([t + np.cross(theta, x - centre) for x in floating.X]).ravel()


def test_a_rigid_motion_of_a_floating_part_gives_zero_tilt_and_zero_shift(floating_part):
    c = floating_part
    t, theta, centre = np.array([0.4, -0.3, 0.2]), np.array([0.01, -0.02, 0.015]), np.array([3.0, 7.0, -2.0])
    U = np.column_stack([rigid_field(c, t, theta, centre)] * 2)
    rows = {r["name"]: r for r in c.record(U)["rows"]}
    print(f"\nrigid motion: shift {rows['shift:a-b:down']['tet_value']:.3g} mm, tilt {rows['tilt:a-b:down']['tet_value']:.3g} rad "
          f"(translation {np.linalg.norm(t):.3g} mm, rotation {np.linalg.norm(theta):.3g} rad)")
    assert rows["shift:a-b:down"]["tet_value"] < 1e-10 and rows["tilt:a-b:down"]["tet_value"] < 1e-12


def test_a_known_relative_tilt_of_two_seats_is_found_within_1_percent(floating_part):
    """The right half turns by `rel` about its own centre on top of a rigid motion. The twist about the row axis x is not a tilt."""
    c = floating_part
    rel = np.array([2e-3, 3e-3, 4e-3])                   # 2e-3 about x (twist), 5e-3 across x
    cb = np.array([15.0, 5.0, 5.0])
    t, theta, centre = np.array([0.1, 0.2, -0.3]), np.array([-0.01, 0.02, 0.005]), np.array([0.0, 0.0, 0.0])
    base = rigid_field(c, t, theta, centre)
    extra = np.array([np.cross(rel, x - cb) if x[0] > 10.0 else np.zeros(3) for x in c.X]).ravel()
    U = np.column_stack([base + extra] * 2)
    rows = {r["name"]: r for r in c.record(U)["rows"]}
    tilt = rows["tilt:a-b:down"]["tet_value"]
    want = float(np.hypot(rel[1], rel[2]))
    print(f"\nknown tilt: want {want:.6g} rad, got {tilt:.6g} rad, error {100 * (tilt / want - 1):.4f} %")
    assert abs(tilt / want - 1.0) < 0.01


# --------------------------------------------------------------------------- #
# The names of the regions of the mode facts (`verify.run._region_masks`)
# --------------------------------------------------------------------------- #

def sel(name, role, mask):
    return SimpleNamespace(name=name, role=role, node_mask=np.array(mask, bool))


def test_a_region_that_is_listed_twice_is_one_name_and_a_measure_region_is_a_name():
    """D4 and item 3 of the review of #137: a joint region that carries a force is the same selection in `load_sel` and `sup_sel`."""
    from topopt.verify.run import _region_masks
    plate, padl, tip, padr = sel("plate", "fixed", [1, 0, 0, 0]), sel("padl", "load", [0, 1, 0, 0]), sel("tip", "load", [0, 0, 1, 0]), sel("padr", "load", [0, 0, 0, 1])
    seat = sel("seat", "measure", [0, 1, 1, 0])
    other_tip = sel("tip", "measure", [1, 1, 0, 0])               # another region with the name of a load region
    out = _region_masks([plate], {"padl": padl, "tip": tip}, {"padl": padl, "padr": padr}, {"seat": seat, "tip": other_tip})
    assert list(out) == ["plate", "padl", "tip", "padr", "seat", "tip#measure"]       # no 'padl#load'
    assert out["padl"] is padl.node_mask and out["seat"] is seat.node_mask and out["tip#measure"] is other_tip.node_mask
    assert list(_region_masks([plate], {"tip": tip}, {})) == ["plate", "tip"]            # the old call still works


@needs_gmsh
def test_a_floating_part_with_a_loaded_joint_and_two_seats_names_them_and_gives_the_alignment_rows(tmp_path):
    """Through `verify_run`: the joint region `padl` also carries a force; two measure regions have an alignment row; no fixed node."""
    from test_verify_supports import _bar_run, _yaml_case, box
    import json
    import yaml
    from topopt.verify import verify_run

    def mutate(d):
        d["load_cases"][0]["forces"].append({"region": "padl", "force": [0, 0, -0.5]})
        d["regions"]["measure"] = {"seat_a": "seat_a.stl", "seat_b": "seat_b.stl"}
        d["limits"] = {"alignment": [{"region_a": "seat_a", "region_b": "seat_b", "axis": [1, 0, 0], "shift_limit_mm": 1.0,
                                       "tilt_limit_rad": 0.1, "cases": ["down"]}]}
        d["buckling"] = {"modes": 3}

    folder = tmp_path / "case"
    yml = _yaml_case(folder, mutate)
    box(folder / "seat_a.stl", (2, 0, 0), (4, 10, 10))
    box(folder / "seat_b.stl", (16, 0, 0), (18, 10, 10))
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    box(run / "result" / "final.stl", (-2, 0, 0), (22, 10, 10))
    (run / "run.json").write_text(json.dumps({"name": "seats", "yaml_path": str(yml), "h": 1.0}))
    res = verify_run(run, mesh_size=2.0, solver="superlu", buckling=True, load_sensitivity=False)
    assert res["ok"] and res["exit_code"] == 0 and res["regions"]["fixed"] == [], res.get("error")
    mo = res["motion"]
    assert mo["status"] == "done" and {r["kind"] for r in mo["rows"]} == {"shift", "tilt"}, mo.get("facts")
    assert all(r["tet_value"] >= 0.0 and r["tet_value"] == r["tet_value"] for r in mo["rows"])       # finite on a part with all dofs free
    case = res["buckling"]["cases"][0]
    names = set(case["mode_facts"][0]["largest_motion"]["region_rms_over_peak"])
    print(f"\nregions of the mode facts: {sorted(names)}; largest motion in {case['mode_facts'][0]['largest_motion']['region']}")
    assert names == {"padl", "padr", "tip", "seat_a", "seat_b", "design"}                  # the loaded joint once, the two seats named

