"""The press-fit check of `topopt verify` at module level (B4b, plan sections 6 and 10.3): the Lame stiffness, the assembly, the closed form of a thick ring.

The ring tests use a structured ring (`annulus_mesh`, no gmsh) and one linear solve for each cup variant; one test meshes the same ring with gmsh.
The bounds are test numbers over the values that the plan measured (section 10.4); they are not constants of the product.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import _HAS_PARDISO, needs_gmsh, needs_skfem, tet_probe  # noqa: E402
from verify_mesh import annulus_mesh  # noqa: E402

pytestmark = needs_skfem

A, B, L = 21.9875, 25.25, 30.0
E, NU = 197000.0, 0.28                 # the ring (steel)
E_CUP, NU_CUP = 70000.0, 0.33
DR = 0.05                              # the radial interference; the YAML value is the diametral one
NET_OVER_SCALE = 1e-3                  # a test bound: the pins of the supports break the symmetry of the ring a little
SOLVER = "pardiso" if _HAS_PARDISO else "superlu"
NO_LIMIT = {"limit_bytes": None, "limit_source": "no limit"}


def closed_form(k):
    """Thick ring (plane stress, free ends) with a cup of stiffness k: the contact pressure and the hoop stress at the bore."""
    c_h = A / E * ((B * B + A * A) / (B * B - A * A) + NU)
    p = DR / (1.0 / k + c_h)
    return p, p * (B * B + A * A) / (B * B - A * A)


def spec(name="ring", dr=DR, radius=A, length=L, point=(0.0, 0.0, 0.0)):
    from topopt.config import BoreSpec
    return BoreSpec(name, point, (0.0, 0.0, 1.0), radius, length, "yaml", diametral_interference_mm=2 * dr)


def cups():
    from topopt.config import Cup
    return (Cup("rigid", k_mpa_per_mm=1.0e6, sources=(("k_mpa_per_mm", "numbers for a test"),)),
            Cup("cup_al", e_mpa=E_CUP, nu=NU_CUP, wall_mm=2.0), Cup("cup_al4", e_mpa=E_CUP, nu=NU_CUP, wall_mm=4.0))


def checks(bores, cup_list=None):
    from topopt.config import BoreChecks
    return BoreChecks(bores=tuple(bores), cups=tuple(cup_list or cups()), press_fit=True)


def ring_supports(model, tol=1e-9):
    """u_z on the whole face z = 0, and u_y at 0 and 180 degrees, u_x at 90 degrees on that face: the exact solution of the long ring has no load there."""
    X = model.node_xyz.T
    on0 = np.abs(X[:, 2]) < tol
    fixed = list(3 * np.flatnonzero(on0) + 2)
    cand = np.flatnonzero(on0 & (np.abs(np.hypot(X[:, 0], X[:, 1]) - A) < 1e-3))
    ang = np.arctan2(X[cand, 1], X[cand, 0])

    def at(a):
        return cand[np.argmin(np.abs(np.angle(np.exp(1j * (ang - a)))))]
    fixed += [3 * at(0.0) + 1, 3 * at(np.pi) + 1, 3 * at(np.pi / 2) + 0]
    fixed = np.array(sorted(set(fixed)))
    return fixed, np.setdiff1d(np.arange(model.N), fixed)


def stage(model, bores, bc, fixed, free, *, lim=NO_LIMIT, force=False, after_bytes=0, say=lambda _m: None):
    from topopt.verify import pressfit
    K = model.stiffness(E, NU)
    return pressfit.run_stage(bc, bores, model, K, free=free, fixed=fixed, solver=SOLVER, E=E, nu=NU, refine_steps=3, lim=lim, force=force,
                              after_bytes=after_bytes, size=0.0, say=say), K


@pytest.fixture(scope="module")
def ring():
    from topopt.verify import bore
    from topopt.verify.p2 import P2Model
    from topopt.config import BoreKnobs
    p, t = annulus_mesh(A, B, L, 2, 72, 8)
    model = P2Model(p, t)
    b = bore.build_bore(model, spec(), BoreKnobs())
    fixed, free = ring_supports(model)
    return model, b, fixed, free


@pytest.fixture(scope="module")
def ring_record(ring):
    model, b, fixed, free = ring
    rec, _K = stage(model, [b], checks([spec()]), fixed, free)
    return rec


# --------------------------------------------------------------------------- #
# The Lame stiffness
# --------------------------------------------------------------------------- #

def test_the_lame_stiffness_is_the_value_of_the_pipeline_bit_for_bit():
    from topopt.verify.pressfit import cup_k
    # pf_info.<variant>.<seat>.k_MPa_per_mm of pressfit.json of the pipeline (E 70000, nu 0.33; walls 2 and 4 mm; bore radii 21.9875 and 27.9625)
    assert cup_k(21.9875, 2.0, 70000.0, 0.33) == 312.50100889055324       # cup_al, upper
    assert cup_k(27.9625, 2.0, 70000.0, 0.33) == 190.0829012470297        # cup_al, lower
    assert cup_k(21.9875, 4.0, 70000.0, 0.33) == 674.9396487664678        # cup_al4, upper
    assert cup_k(27.9625, 4.0, 70000.0, 0.33) == 403.8224430702899        # cup_al4, lower
    R, E_, nu_ = 21.9875, 70000.0, 0.33                   # a solid disc: the wall is the radius
    assert cup_k(R, R, E_, nu_) == 1.0 / (R / E_ * (1.0 - nu_))


def test_the_stored_pipeline_record_gives_the_same_stiffness():
    found = sorted((Path(__file__).parent.parent / "acceptance").glob("*/data/released/pressfit.json"))
    if not found:
        pytest.skip("the case data of the acceptance runner is not in this checkout")
    path = found[0]
    from topopt.verify.pressfit import cup_k
    info = json.loads(path.read_text())["pf_info"]
    n = 0
    for var, wall in (("cup_al", 2.0), ("cup_al4", 4.0)):
        for seat, R in (("upper", 21.9875), ("lower", 27.9625)):
            assert cup_k(R, wall, 70000.0, 0.33) == info[var][seat]["k_MPa_per_mm"]
            n += 1
    assert n == 4


def test_the_stiffness_of_a_cup_is_its_k_or_the_lame_value_at_the_bore():
    from topopt.config import Cup
    from topopt.verify.pressfit import cup_k, stiffness
    assert stiffness(Cup("a", k_mpa_per_mm=123.5), spec()) == 123.5
    c = Cup("b", e_mpa=E_CUP, nu=NU_CUP, wall_mm=2.0)
    assert stiffness(c, spec(radius=A)) == cup_k(A, 2.0, E_CUP, NU_CUP) != stiffness(c, spec(radius=A + 5.0))


# --------------------------------------------------------------------------- #
# The assembly
# --------------------------------------------------------------------------- #

def test_the_spring_direction_is_the_radial_unit_vector_not_the_facet_normal(ring):
    from topopt.verify.pressfit import FACET_QUAD_ORDER, Spring
    model, b, _fixed, _free = ring
    sp = Spring(model, b)
    assert FACET_QUAD_ORDER == 4 and sp.n.shape == (3, len(b.ids), 6) and sp.dx.shape == (len(b.ids), 6)
    assert np.allclose(np.linalg.norm(sp.n, axis=0), 1.0, rtol=0, atol=1e-12) and np.abs(sp.n[2]).max() < 1e-12      # unit, and no axial part
    assert sp.area == pytest.approx(b.entry["area_mm2"], rel=1e-12)          # the facet basis has the area of the facets
    x = np.asarray(sp.basis.global_coordinates().value)
    assert np.allclose(sp.n[0] * np.hypot(x[0], x[1]), x[0], rtol=0, atol=1e-9)    # n is x / r at the quadrature point
    assert len(sp.nodes) == 72 * 9 and sp.nbytes() > 0                      # 9 rows of 72 vertices on the wall


def test_assembly_identities_for_a_rigid_translation(ring):
    """u = e_x everywhere: u . Ks u = k area / 2 (the mean of cos^2 over a full ring), f . u = 0 and sum f = 0 (the mean of cos over a full ring)."""
    from topopt.verify.pressfit import Spring, assemble
    model, b, _fixed, _free = ring
    sp = Spring(model, b)
    k, dr = 312.5, DR
    Ks, f, net = assemble([sp], {"ring": k}, {"ring": dr})
    u = np.zeros(model.N)
    u[0::3] = 1.0
    assert u @ (Ks @ u) == pytest.approx(0.5 * k * sp.area, rel=1e-9)
    scale = k * dr * sp.area
    assert abs(f @ u) < 1e-9 * scale and np.abs(f.reshape(-1, 3).sum(axis=0)).max() < 1e-9 * scale
    assert np.abs(net["ring"]).max() < 1e-9 * scale
    assert Ks.shape == (model.N, model.N) and abs(Ks - Ks.T).max() < 1e-9 * k           # symmetric
    # the nonzeros of Ks lie inside the pattern of K: the nodes of a facet are nodes of one tet
    K = model.stiffness(E, NU).tocsr()
    kp, sp_ = K.copy(), Ks.copy()
    kp.data[:] = 1.0
    sp_.data[:] = 1.0
    assert (kp + sp_).nnz == kp.nnz


def test_the_stiffness_scales_the_spring_matrix_and_the_load_vector_only_by_k(ring):
    from topopt.verify.pressfit import Spring, assemble
    model, b, _fixed, _free = ring
    sp = Spring(model, b)
    K1, f1, _ = assemble([sp], {"ring": 10.0}, {"ring": 0.1})
    K2, f2, _ = assemble([sp], {"ring": 30.0}, {"ring": 0.2})
    assert abs(K2 - 3.0 * K1).max() < 1e-12 * abs(K2).max()
    assert np.abs(f2 - 6.0 * f1).max() < 1e-12 * np.abs(f2).max()


# --------------------------------------------------------------------------- #
# A thick ring against the closed form
# --------------------------------------------------------------------------- #

def test_a_thick_ring_matches_the_closed_form(ring_record):
    rec = ring_record
    assert rec["status"] == "done" and rec["kind"] == "pressfit" and rec["cups"] == ["rigid", "cup_al", "cup_al4"]
    assert [v["name"] for v in rec["variants"]] == ["rigid", "cup_al", "cup_al4"] and rec["dropped_bores"] == []
    for v in rec["variants"]:
        (bo,) = v["bores"]
        p, hoop = closed_form(bo["k_mpa_per_mm"])
        cp = bo["contact_pressure_MPa"]
        assert bo["hoop_p99_9_MPa"] == pytest.approx(hoop, rel=0.03)                 # measured +1.2 % (plan 10.4)
        assert bo["hoop_max_MPa"] >= bo["hoop_p99_9_MPa"] >= bo["hoop_median_MPa"] >= bo["hoop_min_MPa"]
        assert bo["radial_interference_mm"] == DR and bo["diametral_interference_mm"] == 2 * DR
        if v["name"] == "rigid":
            assert cp["area_mean"] == pytest.approx(p, rel=0.005)                    # measured +0.03 %; the vertex values of the rigid variant are not tested
            assert cp["mean"] > cp["area_mean"]                                      # the vertex mean of a stiff spring is above the area mean (plan 6.7)
        else:
            assert cp["mean"] == pytest.approx(p, rel=0.001) and cp["area_mean"] == pytest.approx(p, rel=0.001)
        assert cp["negative_nodes"] == 0 and cp["nodes"] == 72 * 9 and cp["min"] <= cp["mean"] <= cp["max"]


def test_the_cups_are_the_lame_form_and_the_rigid_variant_is_the_k_form(ring_record):
    from topopt.verify.pressfit import cup_k
    rigid, al, al4 = ring_record["variants"]
    assert rigid["k_form"] == "k" and rigid["sources"] == {"k_mpa_per_mm": "numbers for a test"} and "wall_mm" not in rigid
    assert al["k_form"] == "lame" and al["wall_mm"] == 2.0 and al["e_mpa"] == E_CUP and al["nu"] == NU_CUP
    assert al["bores"][0]["k_mpa_per_mm"] == cup_k(A, 2.0, E_CUP, NU_CUP) and al4["bores"][0]["k_mpa_per_mm"] == cup_k(A, 4.0, E_CUP, NU_CUP)


def test_a_softer_cup_gives_a_lower_pressure_and_a_lower_hoop_stress(ring_record):
    rigid, al, al4 = ring_record["variants"]
    h = [v["bores"][0]["hoop_p99_9_MPa"] for v in (rigid, al, al4)]
    p = [v["bores"][0]["contact_pressure_MPa"]["area_mean"] for v in (rigid, al, al4)]
    assert h[0] > h[2] > h[1] > 0 and p[0] > p[2] > p[1] > 0                     # k: rigid 1e6, then wall 4 mm (675 MPa/mm at this bore), then wall 2 mm (312): a thicker wall is a stiffer cup


def test_the_solve_facts_and_the_clamp_reaction(ring_record):
    for v in ring_record["variants"]:
        s = v["solve"]
        assert s["relative_residual"] < 1e-9 and s["backward_error"] < 1e-9 and s["refinement_steps"] == 3 and s["seconds"] > 0
        assert s["max_displacement_mm"] > 0
        # a full ring is balanced by its own symmetry: the net of the spring force and the reaction of the clamps are small against p x area
        (bo,) = v["bores"]
        scale = closed_form(bo["k_mpa_per_mm"])[0] * bo["area_mm2"]
        assert np.abs(bo["net_spring_force_N"]).max() < NET_OVER_SCALE * scale and np.abs(s["clamp_reaction_N"]).max() < NET_OVER_SCALE * scale


def test_the_record_has_the_keys_of_the_plan_and_the_facts(ring_record):
    rec = ring_record
    assert rec["model"]["facet_quadrature_order"] == 4 and rec["model"]["solver"] == SOLVER and rec["seconds"] > 0
    m = rec["memory"]
    assert m["extra_bytes"] > 0 and m["estimate_with_stage_bytes"] == m["estimate_without_stage_bytes"] + m["extra_bytes"]
    assert m["peak_rss_after_stage_bytes"] >= m["peak_rss_before_stage_bytes"] > 0 and m["limit_bytes"] is None
    text = " ".join(rec["facts"])
    for needle in ("p = k (d_r - u_r)", "diametral value of the YAML", "supports are those of the static solve", "solved with no cup spring",
                   "net_spring_force_N is the force of the spring on the part", "not a stress measure", "within 0.4 %"):
        assert needle in text
    assert rec["options"] and rec["statistics"] == ["p99_9", "max"]
    (b0,) = rec["variants"][0]["bores"]
    assert set(b0) >= {"name", "k_mpa_per_mm", "diametral_interference_mm", "radial_interference_mm", "facets", "area_mm2", "net_spring_force_N",
                       "load_vector_net_N", "contact_pressure_MPa", "hoop_p99_9_MPa", "hoop_max_MPa", "hoop_median_MPa", "hoop_min_MPa",
                       "vm_p99_9_MPa", "vm_max_MPa", "vm_median_MPa", "vm_min_MPa", "n_qp"}
    assert set(b0["contact_pressure_MPa"]) == {"area_mean", "mean", "max", "min", "nodes", "negative_nodes", "mean_radial_displacement_mm"}
    assert set(rec["variants"][0]["solve"]) >= {"relative_residual", "backward_error", "refinement_steps", "seconds", "max_displacement_mm", "clamp_reaction_N"}


def test_hoop_values_feed_the_goodman_check(ring_record):
    from topopt.verify import pressfit
    h = pressfit.hoop_values(ring_record)
    assert set(h) == {"rigid", "cup_al", "cup_al4"} and set(h["rigid"]) == {"ring"} and set(h["rigid"]["ring"]) == {"p99_9", "max"}
    b0 = ring_record["variants"][1]["bores"][0]
    assert h["cup_al"]["ring"] == {"p99_9": b0["hoop_p99_9_MPa"], "max": b0["hoop_max_MPa"]}
    assert pressfit.hoop_values(None) is None and pressfit.hoop_values({"status": "skipped"}) is None


# --------------------------------------------------------------------------- #
# Forces on a half ring: the sum of the spring forces is minus the reaction of the clamps
# --------------------------------------------------------------------------- #

def test_the_net_spring_force_is_minus_the_clamp_reaction(ring):
    from topopt.config import BoreKnobs
    from topopt.verify import bore, pressfit
    from topopt.verify.p2 import P2Model
    p, t = annulus_mesh(A, B, L, 2, 36, 6, sweep=math.pi)                  # a half ring: the spring force does not cancel
    model = P2Model(p, t)
    b = bore.build_bore(model, spec(), BoreKnobs())
    assert b.ok and b.entry["facets"] > 100
    X = model.node_xyz.T
    on0 = np.abs(X[:, 2]) < 1e-9
    fixed = np.array(sorted(set(np.concatenate([3 * np.flatnonzero(on0) + c for c in range(3)]))))      # the whole face z = 0 is clamped
    free = np.setdiff1d(np.arange(model.N), fixed)
    bc = checks([spec()], cups()[1:2])
    rec, _K = stage(model, [b], bc, fixed, free)
    (v,) = rec["variants"]
    (bo,) = v["bores"]
    net = np.array(bo["net_spring_force_N"])
    react = np.array(v["solve"]["clamp_reaction_N"])
    assert np.abs(net).max() > 1.0                                              # a half ring pushes to one side: there is a force to balance
    assert np.abs(net + react).max() < 1e-8 * np.abs(net).max()                # the sum of the spring forces is minus the reaction of the clamps
    # the load vector net is k d_r times the integral of the radial unit vector (the pipeline's net_force_N)
    sp = pressfit.Spring(model, b)
    want = bo["k_mpa_per_mm"] * bo["radial_interference_mm"] * np.array([(sp.n[c] * sp.dx).sum() for c in range(3)])
    assert np.array(bo["load_vector_net_N"]) == pytest.approx(want, rel=1e-9, abs=1e-9)
    # the cup is a spring: the physical force is below the load-vector net by the factor of the wall movement
    assert np.linalg.norm(net) < np.linalg.norm(want)


# --------------------------------------------------------------------------- #
# The combined case (plan 6.8, F7): the stress of a load case plus the stress of the press fit at the sample points
# --------------------------------------------------------------------------- #

def test_the_combined_stress_is_the_statistics_of_the_sum_at_the_sample_points(ring):
    from topopt.verify import pressfit
    model, b, fixed, free = ring
    bc = checks([spec()], cups()[:2])
    rng = np.random.default_rng(7)
    case_a = rng.normal(scale=20.0, size=(b.sample.n_qp, 6))
    sp = {"zero": {"ring": np.zeros((b.sample.n_qp, 6))}, "random": {"ring": case_a}}
    K = model.stiffness(E, NU)
    rec = pressfit.run_stage(bc, [b], model, K, free=free, fixed=fixed, solver=SOLVER, E=E, nu=NU, refine_steps=3, lim=NO_LIMIT, force=False, after_bytes=0,
                             size=0.0, service_points=sp)
    assert rec["status"] == "done" and list(rec["combined"]) == ["rigid", "cup_al"]
    springs = [pressfit.Spring(model, b)]
    for cup, var in zip(bc.cups, rec["variants"]):
        by_case = rec["combined"][cup.name]["ring"]
        assert list(by_case) == ["zero", "random"] and set(by_case["zero"]) == {f"{m}_{s}_MPa" for m in ("vm", "hoop") for s in ("p99_9", "max")}
        (bo,) = var["bores"]
        for key, val in by_case["zero"].items():                                          # a case with no stress: the stress of the press fit alone
            assert val == pytest.approx(bo[key], rel=1e-9)
        _v, pts = pressfit._solve_variant(model, K, springs, cup, bc, None, free=free, fixed=fixed, solver=SOLVER, E=E, nu=NU, refine_steps=3)
        want = b.sample.stats_at(case_a + pts["ring"])
        for key, val in by_case["random"].items():
            assert val == pytest.approx(want[key], rel=1e-9)
        vm_case = b.sample.stats_at(case_a)["vm_max_MPa"]
        assert by_case["random"]["vm_max_MPa"] <= vm_case + bo["vm_max_MPa"] + 1e-9           # von Mises is a norm of the stress: the sum is not above the sum of the maxima
    assert any("The cup spring does not act in the solve of the load case" in t for t in rec["facts"])


def test_no_service_points_no_combined_key_and_the_stress_of_the_cases_is_counted_in_the_memory_term(ring):
    from topopt.verify import pressfit
    model, b, fixed, free = ring
    bc = checks([spec()], cups()[:1])
    plain, K = stage(model, [b], bc, fixed, free)
    assert "combined" not in plain and not any("combined is the stress" in t for t in plain["facts"])
    sp = {c: {"ring": np.zeros((b.sample.n_qp, 6))} for c in ("c1", "c2", "c3")}
    rec = pressfit.run_stage(bc, [b], model, K, free=free, fixed=fixed, solver=SOLVER, E=E, nu=NU, refine_steps=3, lim=NO_LIMIT, force=False, after_bytes=0,
                             size=0.0, service_points=sp)
    assert rec["memory"]["extra_bytes"] - plain["memory"]["extra_bytes"] == 3 * b.sample.n_qp * 6 * 8 and "combined" in rec
    assert "the stress of each load case at the sample points" in rec["memory"]["model"]
    assert rec["dropped_bores"] == plain["dropped_bores"] and rec["model"] == plain["model"]
    for a_, b_ in zip(plain["variants"], rec["variants"]):
        for ba, bb in zip(a_["bores"], b_["bores"]):
            assert ba["hoop_p99_9_MPa"] == pytest.approx(bb["hoop_p99_9_MPa"], rel=1e-12) and ba["k_mpa_per_mm"] == bb["k_mpa_per_mm"]


def test_a_skipped_stage_has_no_combined_key(ring):
    model, b, fixed, free = ring
    from topopt.verify import pressfit
    K = model.stiffness(E, NU)
    sp = {"c1": {"ring": np.zeros((b.sample.n_qp, 6))}}
    rec = pressfit.run_stage(checks([spec()]), [b], model, K, free=free, fixed=fixed, solver=SOLVER, E=E, nu=NU, refine_steps=3,
                             lim={"limit_bytes": 1, "limit_source": "test"}, force=False, after_bytes=5, size=0.0, service_points=sp)
    assert rec["status"] == "skipped" and "combined" not in rec


# --------------------------------------------------------------------------- #
# Memory decision, one factor at a time, dropped bores
# --------------------------------------------------------------------------- #

def test_the_memory_decision_skips_above_the_limit_and_force_runs_the_stage(ring):
    from topopt.verify import pressfit
    from topopt.verify.names import use_style
    model, b, fixed, free = ring
    bc = checks([spec()])
    lim = {"limit_bytes": 1000, "limit_source": "--max-gb 1e-6"}
    rec, _K = stage(model, [b], bc, fixed, free, lim=lim, after_bytes=999)
    assert rec["status"] == "skipped" and rec["reason"] == "memory" and rec["variants"] == []
    m = rec["memory"]
    assert m["limit_bytes"] == 1000 and m["estimate_without_stage_bytes"] == 999 and m["extra_bytes"] > 1
    assert m["estimate_with_stage_bytes"] == 999 + m["extra_bytes"]
    text = " ".join(rec["facts"]) + " ".join(rec["options"])
    assert "above the limit" in text and "--max-gb" in text and "--force" in text and "--mesh-size" in text and "--no-bore-checks" in text
    with use_style("mcp"):
        skip, _m = pressfit.memory_decision(bc, 10, 999, lim, False, 3.0)
    assert "max_gb" in " ".join(skip["options"]) and "--max-gb" not in " ".join(skip["options"]) and skip["size_mm"] == 3.0
    assert pressfit.memory_decision(bc, 10, 999, lim, True, 3.0)[0] is None                      # force: the stage runs
    assert pressfit.memory_decision(bc, 10, 980, lim, False, 3.0)[0] is None                     # 990 is inside 1000
    assert pressfit.memory_decision(bc, 10, 999, {"limit_bytes": None, "limit_source": "none"}, False, 3.0)[0] is None


def test_a_skipped_stage_makes_no_factor(ring, monkeypatch):
    import topopt.verify.factor as vf
    model, b, fixed, free = ring
    made = []
    real = vf.make_factor
    monkeypatch.setattr(vf, "make_factor", lambda A_, name: (made.append(name), real(A_, name))[1])
    rec, _K = stage(model, [b], checks([spec()]), fixed, free, lim={"limit_bytes": 1, "limit_source": "test"}, after_bytes=5)
    assert rec["reason"] == "memory" and made == []


def test_one_factor_is_alive_at_a_time(ring, monkeypatch):
    import topopt.verify.factor as vf
    model, b, fixed, free = ring
    live, peak, made = set(), [0], [0]
    real = vf.make_factor

    def spy(A_, name):
        fac = real(A_, name)
        made[0] += 1
        live.add(id(fac))
        peak[0] = max(peak[0], len(live))
        orig = fac.free

        def free_():
            live.discard(id(fac))
            return orig()
        fac.free = free_
        return fac
    monkeypatch.setattr(vf, "make_factor", spy)
    rec, _K = stage(model, [b], checks([spec()]), fixed, free)
    assert rec["status"] == "done" and made[0] == 3 and peak[0] == 1 and not live


def test_a_failing_factor_is_freed_and_the_exception_goes_to_the_caller(ring, monkeypatch):
    import topopt.verify.factor as vf
    model, b, fixed, free = ring
    live = set()
    real = vf.make_factor

    def spy(A_, name):
        fac = real(A_, name)
        live.add(id(fac))
        orig, solve = fac.free, fac.solve

        def free_():
            live.discard(id(fac))
            return orig()

        def boom(*a, **kw):
            raise np.linalg.LinAlgError("test failure")
        fac.free, fac.solve = free_, boom
        return fac
    monkeypatch.setattr(vf, "make_factor", spy)
    with pytest.raises(np.linalg.LinAlgError, match="test failure"):
        stage(model, [b], checks([spec()]), fixed, free)
    assert not live
    from topopt.verify import pressfit
    rec = pressfit.error_record(checks([spec()]), "LinAlgError: test failure")
    assert rec["status"] == "error" and rec["reason"] == "exception" and rec["variants"] == [] and "--max-gb" in rec["options"][0]
    assert pressfit.hoop_values(rec) is None


def test_a_bore_with_no_wall_is_dropped_and_the_other_bore_goes_on(ring):
    from topopt.config import BoreKnobs
    from topopt.verify import bore
    model, b, fixed, free = ring
    far = bore.build_bore(model, spec("far", point=(50.0, 0.0, 0.0)), BoreKnobs())
    assert not far.ok
    rec, _K = stage(model, [far, b], checks([spec("far", point=(50.0, 0.0, 0.0)), spec()], cups()[1:2]), fixed, free)
    assert rec["status"] == "done" and rec["dropped_bores"] == ["far"] and [x["name"] for x in rec["variants"][0]["bores"]] == ["ring"]
    assert any("dropped bore(s) far" in f for f in rec["facts"])
    rec, _K = stage(model, [far], checks([spec("far", point=(50.0, 0.0, 0.0))]), fixed, free)
    assert rec["status"] == "skipped" and rec["reason"] == "no_bore" and rec["facts"] and rec["options"]


def test_two_bores_share_one_solve_and_each_has_its_own_stiffness(ring):
    """Two bores on the two halves of one ring (the same radius and the same interference): each has the facets of its own range and the two have the same stiffness.

    The mapping of `k` and `d_r` to each bore is tested with two radii and two interferences in the next test."""
    from topopt.config import BoreKnobs
    from topopt.verify import bore
    model, _b, fixed, free = ring
    lo = spec("lo", length=L / 2)
    hi = spec("hi", length=L / 2, point=(0.0, 0.0, L / 2))
    bl, bh = (bore.build_bore(model, s, BoreKnobs()) for s in (lo, hi))
    assert bl.ok and bh.ok and set(bl.ids).isdisjoint(set(bh.ids))
    rec, _K = stage(model, [bl, bh], checks([lo, hi], cups()[1:2]), fixed, free)
    (v,) = rec["variants"]
    assert [x["name"] for x in v["bores"]] == ["lo", "hi"] and v["bores"][0]["k_mpa_per_mm"] == v["bores"][1]["k_mpa_per_mm"]
    p, hoop = closed_form(v["bores"][0]["k_mpa_per_mm"])
    assert all(x["contact_pressure_MPa"]["area_mean"] == pytest.approx(p, rel=0.01) for x in v["bores"])


A2, B2, X2 = 14.0, 17.0, 100.0         # a second ring in the same mesh: another radius, another wall, its axis at x = X2


def closed_form_of(k, a, b, dr):
    """The closed form of `closed_form` for a ring of inner radius `a`, outer radius `b` and the radial interference `dr` (the same steel)."""
    c_h = a / E * ((b * b + a * a) / (b * b - a * a) + NU)
    p = dr / (1.0 / k + c_h)
    return p, p * (b * b + a * a) / (b * b - a * a)


def two_ring_model():
    """Two separate steel rings in one P2 mesh, with the supports of `ring_supports` on each. The bores have different radii (A and A2)."""
    from topopt.verify.p2 import P2Model
    p1, t1 = annulus_mesh(A, B, L, 2, 72, 8)
    p2, t2 = annulus_mesh(A2, B2, L, 2, 72, 8)
    p2 = p2 + np.array([[X2], [0.0], [0.0]])
    model = P2Model(np.hstack([p1, p2]), np.hstack([t1, t2 + p1.shape[1]]))
    X = model.node_xyz.T
    on0 = np.abs(X[:, 2]) < 1e-9
    fixed = list(3 * np.flatnonzero(on0) + 2)
    for cx, a in ((0.0, A), (X2, A2)):
        cand = np.flatnonzero(on0 & (np.abs(np.hypot(X[:, 0] - cx, X[:, 1]) - a) < 1e-3))
        ang = np.arctan2(X[cand, 1], X[cand, 0] - cx)

        def at(t):
            return cand[np.argmin(np.abs(np.angle(np.exp(1j * (ang - t)))))]
        fixed += [3 * at(0.0) + 1, 3 * at(math.pi) + 1, 3 * at(math.pi / 2) + 0]
    fixed = np.array(sorted(set(fixed)))
    return model, fixed, np.setdiff1d(np.arange(model.N), fixed)


def test_each_bore_has_its_own_stiffness_and_interference_when_the_radius_and_the_interference_differ():
    """Two bores in one mesh (radius 21.9875 and 14, diametral interference 0.1 and 0.04 mm): the Lame `k` of a cup is `cup_k` of the radius of its own bore, `d_r` is half
    of the interference of its own bore, and the area-mean pressure and the hoop stress of each bore follow the closed form of its own ring (review m-4: a stage that takes
    the `k` or the `d_r` of the first bore for every bore fails this test)."""
    from topopt.config import BoreKnobs
    from topopt.verify import bore, pressfit
    model, fixed, free = two_ring_model()
    s1 = spec("one", dr=0.05, radius=A)
    s2 = spec("two", dr=0.02, radius=A2, point=(X2, 0.0, 0.0))
    b1, b2 = (bore.build_bore(model, s, BoreKnobs()) for s in (s1, s2))
    assert b1.ok and b2.ok and set(b1.ids).isdisjoint(set(b2.ids))
    cup_list = cups()[:2]                                                      # the rigid cup (the same k for both bores) and the Lame cup (a k for each radius)
    rec, _K = stage(model, [b1, b2], checks([s1, s2], cup_list), fixed, free)
    assert rec["status"] == "done" and rec["dropped_bores"] == []
    rigid, lame = rec["variants"]
    assert [x["name"] for x in rigid["bores"]] == ["one", "two"] and [x["name"] for x in lame["bores"]] == ["one", "two"]
    want_k = {"rigid": (1.0e6, 1.0e6), "cup_al": (pressfit.cup_k(A, 2.0, E_CUP, NU_CUP), pressfit.cup_k(A2, 2.0, E_CUP, NU_CUP))}
    assert want_k["cup_al"][0] != want_k["cup_al"][1]
    for v in rec["variants"]:
        for bo, k, (a, b, dr) in zip(v["bores"], want_k[v["name"]], ((A, B, 0.05), (A2, B2, 0.02))):
            assert bo["k_mpa_per_mm"] == k                                     # cup_k of the radius of this bore
            assert bo["radial_interference_mm"] == dr and bo["diametral_interference_mm"] == 2 * dr
            p, hoop = closed_form_of(k, a, b, dr)
            cp = bo["contact_pressure_MPa"]
            assert cp["area_mean"] == pytest.approx(p, rel=0.005 if v["name"] == "rigid" else 0.001)      # measured +0.03 % (rigid), +0.008 % (Lame)
            assert bo["hoop_p99_9_MPa"] == pytest.approx(hoop, rel=0.03)                                  # measured +1.1 % to +1.3 %
        net = np.array([bo["net_spring_force_N"] for bo in v["bores"]]).sum(axis=0)
        assert np.abs(net + np.array(v["solve"]["clamp_reaction_N"])).max() < 1e-6                         # the two forces still balance


def test_the_memory_term_counts_the_stress_array_of_all_tets(ring):
    """The stage computes the stress at the quadrature points of every tet for each variant; `extra_bytes` counts that array."""
    from topopt.verify import pressfit
    model, b, fixed, free = ring
    rec, _K = stage(model, [b], checks([spec()], cups()[:1]), fixed, free)
    sig = model.stress_qp(np.zeros(model.N), E, NU)[0]
    spring = pressfit.Spring(model, b)
    Ks, f, _net = pressfit.assemble([spring], {"ring": 1.0e6}, {"ring": DR})
    assert pressfit.stress_array_bytes(model) == sig.nbytes
    assert rec["memory"]["extra_bytes"] == pressfit.csr_bytes(Ks) + f.nbytes + spring.nbytes() + sig.nbytes
    assert "stress of one variant at the quadrature points of every tet" in rec["memory"]["model"]


def test_a_bore_that_failed_is_dropped_with_its_reason_and_the_stage_goes_on(ring):
    """A bore with an error entry (`Bore.fail`, no sample set, no `facets` key in its entry) is not in the stage; the fact names it with the reason `error` (review m-5)."""
    from topopt.config import BoreKnobs
    from topopt.verify import bore
    model, _b, fixed, free = ring
    lo = spec("lo", length=L / 2)
    hi = spec("hi", length=L / 2, point=(0.0, 0.0, L / 2))
    bl, bh = (bore.build_bore(model, s, BoreKnobs()) for s in (lo, hi))
    bl.fail(FloatingPointError("a made up failure"), "stats")
    assert not bl.ok and bl.drop_reason == "error" and "facets" not in bl.entry
    rec, _K = stage(model, [bl, bh], checks([lo, hi], cups()[1:2]), fixed, free)
    assert rec["status"] == "done" and rec["dropped_bores"] == ["lo"]
    (v,) = rec["variants"]
    assert [x["name"] for x in v["bores"]] == ["hi"]
    text = " ".join(rec["facts"])
    assert "dropped bore(s) lo (error)" in text and "no_facets" in text and "no_sample_set" in text


# --------------------------------------------------------------------------- #
# The estimate before meshing
# --------------------------------------------------------------------------- #

def test_the_estimate_before_meshing_counts_the_variants_and_cuts_the_factor_time_out_of_the_old_fit():
    from topopt.verify import memory, pressfit
    bc = checks([spec()])
    est = pressfit.estimate_before_meshing(bc, 123456789, 1.0e5, "pardiso", None)
    fs = memory.factor_seconds(1.0e5, "pardiso")
    assert est["variants"] == est["factorizations"] == 3 and est["estimate_bytes"] == 123456789
    assert est["estimated_seconds"] == 3 * (fs + memory.TIME_CASE_PER_1E5_S) and est["note"]
    assert pressfit.estimate_before_meshing(bc, 1, 1.0e5, "cholesky", None)["estimated_seconds"] is None
    for solver in ("pardiso", "cholespy", "superlu", "cholesky"):               # the old function gives the same numbers as before the cut
        old = memory.estimate_seconds(2.5e5, solver, 2, True, 0.03)
        if old is None:
            assert memory.factor_seconds(2.5e5, solver, 0.03) is None
        else:
            x = 2.5e5 / 1e5
            assert old == float(memory.TIME_BASE_S + memory.TIME_PER_1E5_S * x + memory.factor_seconds(2.5e5, solver, 0.03)
                                + memory.TIME_CASE_PER_1E5_S * x * 6)


# --------------------------------------------------------------------------- #
# The same ring on a gmsh mesh
# --------------------------------------------------------------------------- #

def face_pin_supports(model):
    """The supports of the research ring run for an unstructured mesh: u_z at three nodes of the face z = 0 (away from the bore and the outer wall), u_y at one.

    The exact solution of the long ring has u_z = 0 on that face and u_y = 0 on the line y = 0. A pin at the bore edge, or on a node that is a few
    degrees off an axis, puts a point force on a node of the bore and moves the tail of the stress statistics (the maximum is above the closed form by 30 %).
    """
    X = model.node_xyz.T
    r = np.hypot(X[:, 0], X[:, 1])
    face = np.flatnonzero((np.abs(X[:, 2]) < 1e-6) & (r > A + 0.1) & (r < B - 0.1))
    ang = np.arctan2(X[face, 1], X[face, 0])
    fixed = []
    for t in (0.0, 2 * np.pi / 3, -2 * np.pi / 3):
        j = face[np.argmin(np.abs(np.angle(np.exp(1j * (ang - t)))))]
        fixed.append(3 * j + 2)
        if t == 0.0:
            fixed.append(3 * j + 1)
    fixed = np.array(sorted(set(fixed)))
    return fixed, np.setdiff1d(np.arange(model.N), fixed)


@needs_gmsh
def test_a_thick_ring_on_a_gmsh_mesh_matches_the_closed_form(tmp_path):
    """The ring of the research runs (`trimesh.creation.annulus`, 360 sections, z from 0 to 30) meshed at 2.0 mm: 8,049 tets, plan section 10.4.

    The hoop p99.9 is a statistic of the stress at the surface, which P2 does not smooth: it depends on the mesh. The same ring from another builder
    (manifold3d, 360 sections) at the same size gives +5 % to +7 % (B4a: the ring surface p99.9 scatters by 5 % to 18 %, the median by less than 1.4 %).
    The cause is the P2 surface error at the bore mouths (the edge between the bore wall and the end face), not the supports: all the top 0.1 % points of that ring are
    at z below 1.25 mm or above 29.4 mm, the points with 2 mm < z < 28 mm give +1.4 % to +1.5 %, and three sets of supports move the p99.9 by about 1 point (B4b review m-2).
    The bound of 3 % is for this ring (the trimesh ring) and the pinned supports above, as the plan measured it.
    """
    probe = tet_probe()
    if not probe.ok:
        pytest.skip(f"needs gmsh (the 'tet' extra): {probe.state}")
    trimesh = pytest.importorskip("trimesh")
    from topopt.config import BoreKnobs
    from topopt.tet import mesh_stl
    from topopt.verify import bore
    from topopt.verify.p2 import P2Model
    ring = trimesh.creation.annulus(r_min=A, r_max=B, height=L, sections=360)
    ring.apply_translation([0, 0, L / 2])
    assert ring.is_watertight
    stl = tmp_path / "ring.stl"
    ring.export(stl)
    model = P2Model.from_tetmesh(mesh_stl(stl, size=2.0, threads=1, optimize="Gmsh+Netgen", log=None))
    b = bore.build_bore(model, spec(), BoreKnobs())
    assert b.ok
    fixed, free = face_pin_supports(model)
    rec, _K = stage(model, [b], checks([spec()]), fixed, free)
    for v in rec["variants"]:
        (bo,) = v["bores"]
        p, hoop = closed_form(bo["k_mpa_per_mm"])
        cp = bo["contact_pressure_MPa"]
        assert bo["hoop_p99_9_MPa"] == pytest.approx(hoop, rel=0.03)                 # measured +1.4 % to +1.5 %
        assert bo["hoop_median_MPa"] == pytest.approx(hoop, rel=0.03)                # measured -1.6 % to -1.8 %
        if v["name"] == "rigid":
            assert cp["area_mean"] == pytest.approx(p, rel=0.005)                    # measured +0.15 %
        else:
            assert cp["mean"] == pytest.approx(p, rel=0.001) and cp["area_mean"] == pytest.approx(p, rel=0.001)     # measured 0.03 % to 0.07 %
