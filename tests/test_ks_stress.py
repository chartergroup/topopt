"""Tests for the stress constraint (A2): aggregation, gradients, rows, history, YAML keys.

Tolerances (each one is stated at the test that uses it):

- EQUIV_RTOL 1e-12: aggregation functions against the fixture that the pipeline lines made. The error is
  element by element (relative, no norm, no floor above 1e-250). D0 comes from the fixture (the
  pipeline lines 312-314); the demo's own D0 is checked on its own (1e-14).
- FD: central difference, best of three steps, relative to the gradient of the voxel (with a floor of
  1e-3 of the largest gradient of the row). Direct solver: FD_TOL 1e-5. AMG: 1e-4.
  Noise floor (PARDISO, cantilever h = 2 mm, 249 voxels, combined rows, 8 voxels, worst row; measured):
  step 1e-3: 1e-6 (truncation, falls with the step squared), 3e-4: 1.3e-7, 1e-4: 1.3e-6 (beta 1) and
  2.8e-8 (beta 4), 3e-5: 2.3e-6, 1e-5: 5.7e-6, 1e-6: 1.1e-4, 1e-7: 1.3e-4. Below 1e-4 the roundoff of
  two solves divided by the step wins (KB pitfall fd-check-noise-floor). The test takes the best of the
  steps 1e-4, 3e-5 and 1e-5 for each voxel. The best error is at most 1.3e-6, so the bar 1e-5 is
  about 8 times above it and 10 times below the noise at 1e-6.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from test_multirow_mma import case_yaml, evaluate, fd_settings, make_model, make_model_in
from topopt.config import UserError, load_config
from topopt.fe import isotropic_D
from topopt.formulation import (NOT_MET, Formulation, compliance_constraint, mass_objective, volume_constraint,
                                weighted_compliance)
from topopt.ks_stress import (PEAK_ABOVE_TOL, XMIN_STRESS, adjoint_strain_load, explicit_rho_term, ks_aggregate, ks_scale,
                              peak_above_limit, relaxed_von_mises, stress_ks_constraint)
from topopt.optimize import Model, formulation_from_config, minimize
from topopt.problem import build_problem
from beta_fact_helpers import problem_facts_without_beta_fact as problem_facts   # the short fixtures end before the last beta stage: that fact (F4-18) is not the subject here

# Every test of this file compares the demo with a record of the old pipeline: pytest -m equivalence.
pytestmark = pytest.mark.equivalence

FIXTURE = Path(__file__).resolve().parent / "data" / "ks_pipeline_fixture.json"
EQUIV_RTOL = 1e-12
FD_BETAS = (1.0, 4.0)


# --- equivalence with the pipeline ------------------------------------------


TINY = 1e-250   # below this a reference value counts as zero (the weights of far elements underflow to 0 or denormals)


def _err(a, b):
    """Largest error over the elements: |a - b| / |b|, for the elements with |b| above `TINY`.

    The comparison is element by element, with no floor and no norm. A norm hides an error in a few small
    elements (REVIEW m1: a floor of the explicit term 1000 times larger changed it by 99.9 % in the elements with
    tiny rho, and a norm test passed). An element with |b| <= TINY must have |a| <= TINY (else the error is inf).
    """
    a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
    big = np.abs(b) > TINY
    if np.any(np.abs(a[~big]) > TINY):
        return float("inf")
    return float(np.max(np.abs(a[big] - b[big]) / np.abs(b[big]), initial=0.0))


def check_equivalence(doc):
    """Run the demo functions on the arrays of the fixture. Return the worst element error. Raise on a miss."""
    D0 = np.array(doc["D0"])                       # the matrix of the pipeline (lines 312-314), not the demo's
    worst = 0.0
    for case in doc["cases"]:
        eps, rho = np.array(case["eps"]), np.array(case["rho"])
        vm, dvm_ds, fac = relaxed_von_mises(eps, D0, rho, case["power"])
        S, w, zmax = ks_aggregate(vm, case["limit"], case["q"])
        c = ks_scale(zmax, S) if case["c_from_update_rule"] else case["c"]
        explicit = explicit_rho_term(w, vm, case["limit"], rho, case["power"])
        g6 = adjoint_strain_load(w, case["limit"], fac, dvm_ds, D0)     # the line of StressRow.__call__
        errs = {"vm": _err(vm, case["vm"]), "dvm_ds": _err(dvm_ds, case["dvm_ds"]), "fac": _err(fac, case["fac"]),
                "S": _err(S, case["S"]), "zmax": _err(zmax, case["zmax"]), "w": _err(w, case["w"]),
                "c": _err(c, case["c"]), "explicit": _err(explicit, case["explicit"]), "g6": _err(g6, case["g6"]),
                "row": _err(c * S - 1.0, case["row"])}
        for key, e in errs.items():
            assert e <= EQUIV_RTOL, f"{case['name']}: {key} element error {e:.3e}"
            worst = max(worst, e)
        assert abs(w.sum() - 1.0) < 1e-12
    return worst


def test_aggregation_equals_pipeline_fixture():
    """vm, dvm/dsigma, S, w, c, the explicit term, the adjoint right-hand side g6 and the row equal the pipeline lines.

    Element by element, error at most 1e-12 (relative). The pipeline
    matrix D0 comes from the fixture, so the arrays are the same operations on the same numbers.
    """
    doc = json.loads(FIXTURE.read_text())
    worst = check_equivalence(doc)
    print(f"equivalence: worst element error {worst:.3e}")


def test_d0_of_the_demo_equals_the_pipeline_matrix():
    """The demo builds D0 from E and nu, the pipeline from the Lame constants. They agree to the last bits."""
    doc = json.loads(FIXTURE.read_text())
    D0 = isotropic_D(doc["E"], doc["nu"])
    ref = np.array(doc["D0"])
    assert np.max(np.abs(D0 - ref)) / np.max(np.abs(ref)) <= 1e-14


def test_the_equivalence_test_catches_a_changed_floor(monkeypatch):
    """Mutation check (REVIEW m1): a floor of 1e-9 instead of 1e-12 must fail the test, also in the explicit term only."""
    import topopt.ks_stress as ks
    doc = json.loads(FIXTURE.read_text())
    assert check_equivalence(doc) <= EQUIV_RTOL
    with monkeypatch.context() as m:                      # the explicit term alone, as the reviewer did
        m.setattr(sys.modules[__name__], "explicit_rho_term",
                  lambda w, vm, limit, rho, power=0.5: w * vm / limit * power / np.maximum(rho, 1e-9))
        with pytest.raises(AssertionError, match="explicit element error"):
            check_equivalence(doc)
    monkeypatch.setattr(ks, "RHO_FLOOR", 1e-9)           # the floor of the whole row
    with pytest.raises(AssertionError, match="element error"):
        check_equivalence(doc)


def test_fixture_covers_the_regimes():
    """The fixture holds a small c (stress far below the limit), c = 1, tiny rho and another q and power."""
    doc = json.loads(FIXTURE.read_text())
    by = {c["name"]: c for c in doc["cases"]}
    assert by["far_below_limit_c_small"]["c"] < 0.5
    assert by["scale_one"]["c"] == 1.0 and not by["scale_one"]["c_from_update_rule"]
    assert min(by["tiny_rho_floor"]["rho"]) < 1e-10
    assert by["q_20_power_0p6"]["q"] == 20.0 and by["q_20_power_0p6"]["power"] == 0.6


def test_ks_is_an_upper_bound_of_the_peak_and_c_makes_it_equal():
    rng = np.random.default_rng(0)
    vm = rng.uniform(0.5, 3.0, 400)
    S, w, zmax = ks_aggregate(vm, 2.0, 40.0)
    assert zmax <= S <= zmax + np.log(400) / 40.0 + 1e-15
    assert abs(ks_scale(zmax, S) * S - zmax) < 1e-15      # c * S = the peak at the stage start


# --- gradients ---------------------------------------------------------------


def fd_check(m, form, beta, names, samples=8, seed=3, steps=(1e-4, 3e-5, 1e-5)):
    """Compare the rows and the objective with central differences. Best of `steps` for each entry."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.3, 0.7, m.n_design)
    _, tol = fd_settings(m)
    if m.solver_name == "amg":
        steps = (1e-3, 3e-4, 1e-4)
    ev = evaluate(m, form, x, beta)
    assert ev.g.shape[0] == len(names)
    worst = 0.0
    for e in rng.choice(m.n_design, size=min(samples, m.n_design), replace=False):
        best = np.full(len(names) + 1, np.inf)
        for eps in steps:
            xp, xm = x.copy(), x.copy()
            xp[e] += eps
            xm[e] -= eps
            a, b = evaluate(m, form, xp, beta), evaluate(m, form, xm, beta)
            fd = np.append((a.g - b.g) / (2 * eps), (a.f0 - b.f0) / (2 * eps))
            an = np.append(ev.dg[:, e], ev.df0[e])
            ref = np.append(np.maximum(np.abs(ev.dg[:, e]), 1e-3 * np.abs(ev.dg).max(axis=1)),
                            max(abs(ev.df0[e]), 1e-3 * np.abs(ev.df0).max(), 1e-300))
            best = np.minimum(best, np.abs(fd - an) / ref)
        assert np.all(best < tol), f"beta {beta} voxel {e}: errors {dict(zip(names + ['objective'], best))}"
        worst = max(worst, float(best.max()))
    return worst


@pytest.mark.parametrize("beta", FD_BETAS)
def test_fd_mass_objective_and_stress_rows(tmp_path, beta):
    """The mass objective and the stress row of each case, through the filter and projection.

    Row 1 has the default scale (auto: c is set at the first design of the beta stage and then kept),
    row 2 has c = 1.
    """
    m = make_model(tmp_path, h=2.0)
    form = Formulation(mass_objective(), [stress_ks_constraint("down", 0.5),
                                          stress_ks_constraint("side", 0.3, scale=1.0)])
    assert form.xmin == XMIN_STRESS
    worst = fd_check(m, form, beta, ["stress:down", "stress:side"])
    print(f"fd beta {beta}: worst {worst:.2e}")
    m.solver.free()


@pytest.mark.parametrize("beta", FD_BETAS)
def test_fd_stress_compliance_limit_and_volume_rows_together(tmp_path, beta):
    """One problem with the mass objective, two stress rows, a compliance limit and a volume row."""
    m = make_model(tmp_path, h=2.0)
    form = Formulation(mass_objective(), [
        stress_ks_constraint("down", 0.5, scale=0.8),
        stress_ks_constraint(1, 0.3),
        compliance_constraint("down", 5.0),
        volume_constraint(m.n_design, 0.3)])
    worst = fd_check(m, form, beta, ["stress:down", "stress:side", "compliance:down", "volume"])
    print(f"fd combined beta {beta}: worst {worst:.2e}")
    m.solver.free()


def test_fd_stress_row_with_the_compliance_objective(tmp_path):
    """A stress row next to the weighted compliance and the volume row (the compliance objective path)."""
    m = make_model(tmp_path, h=2.0)
    form = Formulation(weighted_compliance(), [volume_constraint(m.n_design, 0.3),
                                               stress_ks_constraint("side", 0.4, scale=1.0)])
    fd_check(m, form, 2.0, ["volume", "stress:side"])
    m.solver.free()


def test_gradient_is_finite_as_rho_tends_to_zero(tmp_path):
    """x = 0 gives rho = 0 in the void (and the floor 1e-12 under the power): the gradient stays finite.

    Also at x = xmin (0.02), the lower bound of a run with a stress row, and at the FD level.
    """
    m = make_model(tmp_path, h=2.0)
    form = Formulation(mass_objective(), [stress_ks_constraint("down", 0.5, scale=1.0),
                                          stress_ks_constraint("side", 0.3)])
    for value in (0.0, 1e-9, XMIN_STRESS):
        x = np.full(m.n_design, value)
        x[: m.n_design // 4] = 0.8       # some solid, so the stiffness matrix is not singular
        for beta in (1.0, 8.0):
            ev = evaluate(m, form, x, beta)
            assert np.all(np.isfinite(ev.dg)) and np.all(np.isfinite(ev.g)) and np.all(np.isfinite(ev.df0))
    # rho of the void voxels is tiny, the relaxed stress there is tiny, and the explicit term is bounded
    rho_tilde, rho = m.density(np.full(m.n_design, 0.0), 8.0)
    assert rho.min() < 1e-6
    m.solver.free()


def test_fd_at_the_xmin_design(tmp_path):
    """FD against the analytic gradient at x = xmin for most voxels (rho near 0): the row is still right."""
    m = make_model(tmp_path, h=2.0)
    form = Formulation(mass_objective(), [stress_ks_constraint("down", 0.5, scale=1.0)])
    x = np.full(m.n_design, XMIN_STRESS)
    x[: m.n_design // 3] = 0.9
    beta = 4.0
    ev = evaluate(m, form, x, beta)
    eps, tol = fd_settings(m)
    rng = np.random.default_rng(1)
    big = np.argsort(-np.abs(ev.dg[0]))[:30]      # voxels with a visible gradient
    for e in rng.choice(big, size=6, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[e] += 1e-4
        xm[e] -= 1e-4
        fd = (evaluate(m, form, xp, beta).g[0] - evaluate(m, form, xm, beta).g[0]) / 2e-4
        assert abs(fd - ev.dg[0, e]) / abs(ev.dg[0, e]) < 1e-4, (e, fd, ev.dg[0, e])
    m.solver.free()


# --- the row: value, info, stage scale ---------------------------------------


def test_row_value_info_and_stage_scale(tmp_path):
    m = make_model(tmp_path, h=2.0)
    row = stress_ks_constraint("down", 0.5)
    form = Formulation(mass_objective(), [row])
    x = np.full(m.n_design, 0.5)
    ev = evaluate(m, form, x, 1.0)
    e = ev.constraints[0]
    assert e["name"] == "stress:down" and e["limit"] == 1.0
    assert e["c_changed"] and 0 < e["c"] <= 1
    # at the stage start the row equals the relaxed peak over the limit (c = zmax / KS)
    assert abs(e["value"] - e["relaxed_peak_MPa"] / e["stress_limit_MPa"]) < 1e-12
    assert e["ks_MPa"] >= e["relaxed_peak_MPa"]
    ev2 = evaluate(m, form, x * 0.9, 1.0)         # same beta: c is kept
    assert not ev2.constraints[0]["c_changed"] and ev2.constraints[0]["c"] == e["c"]
    ev3 = evaluate(m, form, x * 0.9, 2.0)         # new beta: c is set again
    assert ev3.constraints[0]["c_changed"]
    assert abs(ev3.constraints[0]["value"] - ev3.constraints[0]["relaxed_peak_MPa"] / 0.5) < 1e-12
    m.solver.free()


def test_scale_one_keeps_c_and_makes_a_bound(tmp_path):
    m = make_model(tmp_path, h=2.0)
    form = Formulation(mass_objective(), [stress_ks_constraint("down", 0.5, scale=1)])
    for beta in (1.0, 2.0, 8.0):
        e = evaluate(m, form, np.full(m.n_design, 0.5), beta).constraints[0]
        assert e["c"] == 1.0 and not e["c_changed"]
        assert e["value"] >= e["relaxed_peak_MPa"] / e["stress_limit_MPa"]    # KS >= peak
    m.solver.free()


def test_row_validation():
    for bad in (dict(limit=0.0), dict(limit=-1.0), dict(limit=float("nan")), dict(limit=1.0, q=0.0),
                dict(limit=1.0, power=0.0), dict(limit=1.0, power=1.5), dict(limit=1.0, scale=0.0),
                dict(limit=1.0, scale="never")):
        with pytest.raises(ValueError):
            stress_ks_constraint("down", **bad)


def test_formulation_xmin_rule():
    """xmin is 0.02 with a stress row and 0 without. An explicit xmin wins."""
    obj = weighted_compliance()
    assert Formulation(obj, [volume_constraint(10, 0.3)]).xmin == 0.0
    assert Formulation(obj, [volume_constraint(10, 0.3), stress_ks_constraint(0, 1.0)]).xmin == XMIN_STRESS
    assert Formulation(obj, [stress_ks_constraint(0, 1.0)], xmin=0.0).xmin == 0.0
    assert Formulation(obj, [stress_ks_constraint(0, 1.0)], xmin=0.05).xmin == 0.05


def test_mass_objective_value_and_gradient(tmp_path):
    m = make_model(tmp_path, h=2.0)
    x = np.full(m.n_design, 0.4)
    rho_tilde, rho = m.density(x, 1.0)
    point = m.design_point(x, 1.0, rho_tilde, rho, m.solve(m.stiffness(rho)))
    (t,) = mass_objective().terms(point)
    h, dens = m.problem.h, m.problem.config.material.density
    assert abs(t.value - (rho.sum() + m.n_el - m.n_design) * h ** 3 / 1000.0 * dens) < 1e-12
    assert np.allclose(t.grad, h ** 3 / 1000.0 * dens)
    m.solver.free()


# --- YAML --------------------------------------------------------------------


def yaml_with(tmp_path, settings=None, case0=None, case1=None):
    p = case_yaml(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["settings"].update(settings or {})
    d["load_cases"][0].update(case0 or {})
    d["load_cases"][1].update(case1 or {})
    p.write_text(yaml.safe_dump(d))
    return p


def test_yaml_defaults_have_no_stress(tmp_path):
    cfg = load_config(case_yaml(tmp_path))
    st = cfg.settings
    assert (st.objective, st.stress_q, st.stress_power, st.stress_scale) == ("compliance", 40.0, 0.5, "auto")
    assert all(c.stress_limit_mpa is None and c.stress_limit_from is None for c in cfg.load_cases)
    form = formulation_from_config(cfg, 10)
    assert [c.name for c in form.constraints] == ["volume"] and form.xmin == 0.0


def test_yaml_limit_rules(tmp_path):
    p = yaml_with(tmp_path, case0={"stress_limit": 12.5},
                  case1={"allowable_mpa": 40, "safety_factor": 2.5, "allowable_source": "datasheet X, 23 C"})
    c0, c1 = load_config(p).load_cases
    assert (c0.stress_limit_mpa, c0.stress_limit_from) == (12.5, "stress_limit")
    assert (c1.stress_limit_mpa, c1.stress_limit_from) == (16.0, "allowable_mpa / safety_factor")
    assert c1.allowable_source == "datasheet X, 23 C"
    # allowable alone: factor 1
    c = load_config(yaml_with(tmp_path, case0={"allowable_mpa": 30.0})).load_cases[0]
    assert c.stress_limit_mpa == 30.0
    # stress_limit is the final limit; allowable_mpa (no factor) is kept for the report
    c = load_config(yaml_with(tmp_path, case0={"stress_limit": 9.0, "allowable_mpa": 40.0})).load_cases[0]
    assert c.stress_limit_mpa == 9.0 and c.allowable_mpa == 40.0 and c.safety_factor is None


def test_yaml_scientific_text_is_read_for_the_stress_keys(tmp_path):
    c = load_config(yaml_with(tmp_path, case0={"stress_limit": "5e1", "allowable_mpa": "1e2"})).load_cases[0]
    assert c.stress_limit == 50.0 and c.allowable_mpa == 100.0


@pytest.mark.parametrize("settings,case0,message", [
    ({"objective": "volume"}, None, "'settings.objective' must be compliance or mass, got 'volume'"),
    ({"stress_q": 0}, None, "'settings.stress_q' must be > 0.0"),
    ({"stress_power": 2.0}, None, "'settings.stress_power' must be <= 1.0"),
    ({"stress_scale": "big"}, None, "'settings.stress_scale' must be auto or a number > 0, got 'big'"),
    ({"stress_scale": "AUTO"}, None, "'settings.stress_scale' must be auto or a number > 0, got 'AUTO'"),
    ({"stress_step_acceptance": "yes"}, None, "'settings.stress_step_acceptance' must be true or false, got 'yes'"),
    ({"stress_scale": 0}, None, "'settings.stress_scale' must be > 0.0"),
    (None, {"stress_limit": -1.0}, "'load_cases[0].stress_limit' must be > 0.0, got -1.0"),
    (None, {"stress_limit": "big"}, "'load_cases[0].stress_limit' must be a number, got 'big'"),
    (None, {"safety_factor": 2.0}, "'load_cases[0].safety_factor' needs 'load_cases[0].allowable_mpa'"),
    (None, {"stress_limit": 40.0, "safety_factor": 2.0},
     "'load_cases[0]' has both safety_factor and stress_limit: stress_limit is the final limit and takes no "
     "safety_factor. Write allowable_mpa + allowable_source + safety_factor, or a final stress_limit with no "
     "safety_factor"),
    (None, {"stress_limit": 5.0, "allowable_mpa": 40.0, "safety_factor": 2.0}, "has both safety_factor and stress_limit"),
    (None, {"allowable_mpa": 0}, "'load_cases[0].allowable_mpa' must be > 0.0"),
    (None, {"allowable_mpa": 5.0, "allowable_source": ""}, "'load_cases[0].allowable_source' must be a non-empty string"),
])
def test_yaml_errors(tmp_path, settings, case0, message):
    with pytest.raises(UserError) as e:
        load_config(yaml_with(tmp_path, settings=settings, case0=case0))
    assert message in str(e.value) and "(file:" in str(e.value)


def test_yaml_formulation_rows_and_order(tmp_path):
    cfg = load_config(yaml_with(tmp_path, case0={"stress_limit": 5.0, "compliance_limit": 3.0},
                                case1={"stress_limit": 7.0}))
    form = formulation_from_config(cfg, 10)
    assert [c.name for c in form.constraints] == ["volume", "compliance:down", "stress:down", "stress:side"]
    assert form.objective.name == "weighted_compliance" and form.xmin == XMIN_STRESS
    cfg = load_config(yaml_with(tmp_path, settings={"objective": "mass"}, case0={"stress_limit": 5.0},
                                case1={"stress_limit": 7.0}))
    form = formulation_from_config(cfg, 10)
    assert form.objective.name == "mass"
    assert [c.name for c in form.constraints] == ["stress:down", "stress:side"]   # no volume row


def test_mass_run_without_a_limit_is_a_fact_not_a_refusal(tmp_path):
    """Objective mass, no stress limit, no compliance limit: the facts are given and the run goes on."""
    p = yaml_with(tmp_path, settings={"objective": "mass", "case_combine": "max", "iterations": 6})
    cfg = load_config(p)
    facts = problem_facts(cfg)
    assert len(facts) == 2 and "no load case has a stress_limit" in facts[0] and "options:" in facts[0]
    assert "case_combine has no effect" in facts[1]
    m = Model(build_problem(cfg))
    form = formulation_from_config(cfg, m.n_design)
    assert [c.name for c in form.constraints] == ["volume"]
    res = minimize(m, form)                       # no error: the run reaches the end
    assert len(res.history) == cfg.settings.iterations + 1
    assert res.history[-1]["volume_fraction"] < cfg.settings.volume_fraction     # the mass went down
    m.solver.free()
    assert problem_facts(load_config(case_yaml(tmp_path))) == []                  # no fact for the default


def test_facts_for_keys_that_do_nothing_or_surprise(tmp_path):
    """REVIEW n3, n4, n5: a fact in `problem_facts` (so in `topopt check` and the run log). Nothing blocks a run."""
    # n3: allowable_source in a case with no stress limit
    f = problem_facts(load_config(yaml_with(tmp_path, case0={"allowable_source": "datasheet X"})))
    assert len(f) == 1 and "'down'" in f[0] and "no stress row" in f[0] and "text is not used" in f[0]
    # no fact when the case has a limit
    assert problem_facts(load_config(yaml_with(tmp_path, case0={"allowable_mpa": 40.0, "allowable_source": "x"}))) == []
    # n4: safety_factor below 1
    f = problem_facts(load_config(yaml_with(tmp_path, case0={"allowable_mpa": 40.0, "safety_factor": 0.5})))
    assert len(f) == 1 and "safety_factor 0.5 is below 1" in f[0] and "80 MPa is above the allowable 40 MPa" in f[0]
    # n5: stress_scale above 1 makes the row stricter than the limit
    f = problem_facts(load_config(yaml_with(tmp_path, settings={"stress_scale": 2.0}, case0={"stress_limit": 5.0})))
    assert len(f) == 1 and "stress_scale 2 is above 1" in f[0] and "S <= 0.5" in f[0] and "stricter" in f[0]
    assert problem_facts(load_config(yaml_with(tmp_path, settings={"stress_scale": 2.0}))) == []      # no stress row
    # the step acceptance switched off is a fact
    f = problem_facts(load_config(yaml_with(tmp_path, settings={"stress_step_acceptance": False},
                                            case0={"stress_limit": 5.0})))
    assert len(f) == 1 and "stress_step_acceptance is false" in f[0] and "takes every MMA step" in f[0]
    assert problem_facts(load_config(yaml_with(tmp_path, settings={"stress_step_acceptance": False}))) == []


def test_check_prints_the_facts_and_the_stress_time(tmp_path, capsys):
    from topopt.cli import main
    p = yaml_with(tmp_path, case0={"allowable_source": "datasheet X"}, case1={"stress_limit": 5.0})
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert "allowable_source but no stress limit" in out
    assert "stress limit:   side: 5 MPa from stress_limit" in out
    assert "stress rows:    1 row, in the time estimate above (estimate)" in out


def test_time_estimate_counts_the_stress_rows():
    """REVIEW m3: per_it * (1 + k n_rows) + n_rows * 0.5e-6 s * n_design, k 0.2 (direct) or 1 (AMG), plus the rejected tries."""
    from topopt.cli import STEP_TRIES_PER_PASS, estimate_seconds, stress_seconds
    n_free, n_design = 115_836, 34_220
    base_it, base_total = estimate_seconds(n_free, 160, "pardiso")
    assert estimate_seconds(n_free, 160, "pardiso", 0, n_design) == (base_it, base_total)     # no row: as before A2
    it, total = estimate_seconds(n_free, 160, "pardiso", 2, n_design)
    assert it == pytest.approx(base_it * (1 + 0.2 * 2 + STEP_TRIES_PER_PASS) + 2 * 0.5e-6 * n_design)
    assert total == pytest.approx(10.0 + it * 161 + 15.0)
    it_off = estimate_seconds(n_free, 160, "pardiso", 2, n_design, step_acceptance=False)[0]
    assert it_off == pytest.approx(base_it * (1 + 0.2 * 2) + 2 * 0.5e-6 * n_design)
    amg_it = estimate_seconds(n_free, 160, "amg")[0]
    assert estimate_seconds(n_free, 160, "amg", 2, n_design, False)[0] == pytest.approx(amg_it * 3 + 2 * 0.5e-6 * n_design)
    assert stress_seconds(1.0, 0, n_design) == 0.0
    assert 0.5 <= STEP_TRIES_PER_PASS <= 0.7                                      # measured: 0.59 and 0.60 tries per pass


def test_check_gives_a_time_range_for_a_run_with_stress_rows(tmp_path, capsys):
    """R1-2: low = the model, high = the model times 1.6. No estimate is called "conservative"; the high end carries the E1 fact (E1-P3)."""
    from topopt.cli import STRESS_RUN_TIME_FACTOR, main
    assert STRESS_RUN_TIME_FACTOR == 1.6
    p = yaml_with(tmp_path, case0={"stress_limit": 5.0})
    assert main(["check", "--json", str(p)]) == 0
    te = json.loads(capsys.readouterr().out)["time_estimate"]
    assert te["range_s"] == [te["total_s"], round(te["total_s"] * 1.6)] or te["range_s"][1] - te["total_s"] * 1.6 < 1
    assert te["range_factor"] == 1.6
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert "time range:     " in out and "Low: the model above. High: the model times 1.6, a fixed factor, not a bound." in out
    assert "In experiment E1, 53 finished PARDISO runs with stress rows in 6 problems on one shared Linux host had a median of 2.23 times " \
           "the model total, and 70 percent took more than 1.6 times it" in out
    assert "1.4 to 1.9" not in out and "conservative" not in out
    (tmp_path / "plain").mkdir()
    plain = case_yaml(tmp_path / "plain")                                           # a YAML in another folder: no stress row
    assert main(["check", "--json", str(plain)]) == 0                               # no stress row: no range
    assert "range_s" not in json.loads(capsys.readouterr().out)["time_estimate"]
    # no budget in the YAML: no budget warning, whatever the estimate is
    assert main(["check", str(p)]) == 0
    cap = capsys.readouterr()
    assert "your time budget" not in cap.out + cap.err and not re.search(r"demo( time)?[ -]budget", cap.out + cap.err, re.IGNORECASE)   # the words, not the paths (N3)
    # a stress run above the budget of the user: the warning gives the range and does not say "conservative"
    (tmp_path / "b1").mkdir()
    (tmp_path / "b2").mkdir()
    budget = {"stop_rules": {"time_budget_s": 1.0}}     # the estimate is at least the 25 s of setup and post-process
    p_budget = yaml_with(tmp_path / "b1", settings=budget, case0={"stress_limit": 5.0})
    assert main(["check", str(p_budget)]) == 0
    cap = capsys.readouterr()
    warn = [l for l in (cap.out + cap.err).splitlines() if l.startswith("warning: the time estimate")]
    assert len(warn) == 1 and "for a run with stress rows is" in warn[0] and " to " in warn[0]
    assert "conservative" not in warn[0] and "model times 1.6" in warn[0] and "1.4 to 1.9" not in warn[0]
    assert "had a median of 2.23 times the model total" in warn[0]
    assert "above your time budget of 1 s (settings.stop_rules.time_budget_s) at the high end" in warn[0] and "demo" not in warn[0]
    assert "; the stop rules are on in this YAML (a stop_rules mapping turns them on), so the run stops at the first pass that ends after 1 s of run time" in warn[0]   # MJ1
    plain_budget = yaml_with(tmp_path / "b2", settings=budget)                      # no stress row: the plain text
    assert main(["check", str(plain_budget)]) == 0
    cap = capsys.readouterr()
    plain_warn = [l for l in (cap.out + cap.err).splitlines() if l.startswith("warning: the time estimate is")]
    assert len(plain_warn) == 1 and "above your time budget of 1 s (settings.stop_rules.time_budget_s)" in plain_warn[0]
    assert "conservative" not in cap.out + cap.err and "demo" not in plain_warn[0]


def test_check_json_has_the_stress_part_of_the_estimate(tmp_path, capsys):
    from topopt.cli import main
    p = yaml_with(tmp_path, case0={"stress_limit": 5.0}, case1={"stress_limit": 7.0})
    assert main(["check", "--json", str(p)]) == 0
    te = json.loads(capsys.readouterr().out)["time_estimate"]
    st = te["stress"]
    assert st["rows"] == 2 and st["label"] == "estimate" and st["step_acceptance"] is True
    assert 0 < st["s_per_iteration"] < te["s_per_iteration"] and st["total_s"] > 0
    assert main(["check", "--json", str(case_yaml(tmp_path))]) == 0
    assert json.loads(capsys.readouterr().out)["time_estimate"]["stress"] is None


# --- history, stage facts, unmet lines ----------------------------------------


def test_history_stage_notes_and_unmet_lines(tmp_path):
    """A beta stage that ends with a row not met is recorded as a fact. The unmet rows give facts and options."""
    p = yaml_with(tmp_path, settings={"objective": "mass", "iterations": 12, "volume_fraction": 0.5,
                                      "move_limit": 0.05, "beta_schedule": [[0, 1], [6, 2]]},
                  case0={"stress_limit": 1e-3}, case1={"stress_limit": 1e-3})   # far below any stress: always unmet
    cfg = load_config(p)
    m = Model(build_problem(cfg))
    res = minimize(m, formulation_from_config(cfg, m.n_design))
    m.solver.free()
    assert [s["iteration"] for s in res.stages] == [0, 6]
    first, second = res.stages
    assert first["from_beta"] is None and second["from_beta"] == 1.0 and second["to_beta"] == 2.0
    assert second["row"].startswith("stress:") and second["feasible"] is (second["max_g"] <= 5e-3)
    assert "stage" in res.history[0] and "stage" in res.history[6] and "stage" not in res.history[3]
    last = res.history[-1]["constraints"]
    assert {"stress_limit_MPa", "ks_MPa", "relaxed_peak_MPa", "c", "c_changed", "q"} <= set(last[0])
    from topopt.formulation import Formulation
    f = formulation_from_config(cfg, m.n_design)
    assert Formulation.stage_lines(second)[0].startswith("beta stage 1 -> 2 at pass 6")
    ok = dict(second, feasible=True)                        # R1-1: "met" is said for the relaxed measure
    assert "all rows met (a stress row is the relaxed measure of the optimizer, not the stress of the part)" in \
        Formulation.stage_lines(ok)[0]
    text = "\n".join(res.unmet)
    assert all(c["g"] > 5e-3 for c in last)                   # the limit is out of reach: both rows are not met
    assert "constraint not met: stress:" in text and "relaxed peak" in text and "options:" in text
    assert "does not fit at this volume" not in text          # the volume text of A1 is for compliance rows
    assert second["feasible"] is False and "NOT feasible" in Formulation.stage_lines(second)[0]
    assert f.n_rows == 2


def test_a_stress_limit_above_the_stress_is_met(tmp_path):
    """A loose limit: the run ends with every row met and no unmet line."""
    p = yaml_with(tmp_path, settings={"objective": "mass", "iterations": 10, "volume_fraction": 1.0,
                                      "move_limit": 0.05, "beta_schedule": [[0, 1]]},
                  case0={"stress_limit": 1000.0}, case1={"stress_limit": 1000.0})
    cfg = load_config(p)
    m = Model(build_problem(cfg))
    res = minimize(m, formulation_from_config(cfg, m.n_design))
    m.solver.free()
    assert res.unmet == [] and all(c["g"] < 0 for c in res.constraints)
    assert res.history[-1]["objective"]["value"] < res.history[0]["objective"]["value"]   # mass goes down


# --- default path ---------------------------------------------------------------


def test_default_run_has_no_stress_keys(tmp_path):
    """A run with no stress key writes the history and run.json keys of A1 and the same metrics keys."""
    from topopt.cli import main
    out = tmp_path / "r"
    assert main(["run", str(case_yaml(tmp_path, iterations=4)), "--out", str(out), "--quiet"]) == 0
    lines = (out / "history.jsonl").read_text().splitlines()
    first, rec = json.loads(lines[0]), json.loads(lines[-1])
    a1_keys = {"iteration", "elapsed_s", "compliance", "compliance_cases", "volume_fraction", "beta",
               "change", "grayness", "t_iter_s", "objective", "constraints"}
    assert set(first) == a1_keys                      # a pass that does not end a beta stage
    assert set(rec) == a1_keys | {"stage_end"}        # the last pass has the volume and load path facts of the end of the run (P3a, section 13)
    assert set(rec["constraints"][0]) == {"name", "value", "limit", "g", "violation"}
    run = json.loads((out / "run.json").read_text())
    assert "stress" not in run and run["objective"] == "weighted_compliance"
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    assert "stress" not in metrics
    assert "stage" not in rec and "beta stage" not in (out / "run.log").read_text()
    assert "obj=" not in (out / "run.log").read_text()      # the log line of a default run is unchanged


# --- F4-34: the relaxed peak above the limit, with the row met -------------------------------------------------------------------------------------------------

def test_the_tolerance_of_the_peak_above_the_limit_has_one_name_and_is_the_not_met_value():
    """The run log line and the flag in metrics.json use `PEAK_ABOVE_TOL`, which is the tolerance of "not met" (`NOT_MET`): with the KS scale 1 a peak above
    the limit by more than it is a row that is not met, so a met row with such a peak has a scale below 1."""
    assert PEAK_ABOVE_TOL == NOT_MET == 5e-3
    lim = 0.5
    assert peak_above_limit(lim * (1 + PEAK_ABOVE_TOL), lim) is False and peak_above_limit(lim * (1 + PEAK_ABOVE_TOL) * (1 + 1e-9), lim) is True


@pytest.mark.parametrize("peak", [0.05, 0.4999, 0.5, 0.5024, 0.5025, 0.50251, 0.6, 0.957, 92.5])
@pytest.mark.parametrize("g", [-0.2, 0.0, 0.005, 0.2])
def test_the_run_log_line_and_the_flag_in_metrics_json_are_one_comparison(peak, g):
    """F4-34: `stress peak above the limit` (end of the run log) and `relaxed.peak_above_limit` (metrics.json) cannot disagree, also when `met` is true or false."""
    from topopt import stress_report as sr
    lim = 0.5
    entry = {"name": "stress:pull_off", "relaxed_peak_MPa": peak, "ks_MPa": peak * 1.1, "stress_limit_MPa": lim, "value": 1.0 + g, "g": g, "limit": 1.0, "c": 0.52,
             "c_changed": False, "q": 40.0}
    lines = stress_ks_constraint("pull_off", lim).facts(entry)
    rel = sr.relaxed_numbers({"iteration": 7, "constraints": [entry]}, "pull_off")
    assert rel["peak_above_limit"] is peak_above_limit(peak, lim) and bool(lines) is rel["peak_above_limit"]
    assert rel["met"] is (g <= NOT_MET)
    assert rel["peak_above_limit_tol"] == PEAK_ABOVE_TOL and "peak_above_limit_tol" in rel["peak_above_limit_means"] and "whatever `met` says" in rel["peak_above_limit_means"]
    if lines:
        assert lines[0].startswith("stress peak above the limit: stress:pull_off") and f"{peak:.4g} MPa" in lines[0]


def test_the_flag_is_true_for_a_met_row_with_the_peak_above_the_limit_and_null_where_the_history_has_no_limit():
    """E2/SS05: `stress_scale` 0.5 gave the relaxed peak 0.957 MPa against 0.5 MPa and `met` true."""
    from topopt import stress_report as sr
    row = {"name": "stress:pull_off", "relaxed_peak_MPa": 0.957, "ks_MPa": 1.8, "stress_limit_MPa": 0.5, "value": 0.99, "g": -0.01, "c": 0.5, "q": 40.0}
    rel = sr.relaxed_numbers({"iteration": 220, "constraints": [row]}, "pull_off")
    assert rel["met"] is True and rel["peak_above_limit"] is True
    old = {k: v for k, v in row.items() if k != "stress_limit_MPa"}
    assert sr.relaxed_numbers({"iteration": 220, "constraints": [old]}, "pull_off")["peak_above_limit"] is None


# --- F4-28: stress_scale moves the stress limit of the user, below 1 and above 1 --------------------------------------------------------------------------------

def scale_facts(tmp_path, value, **settings):
    return problem_facts(load_config(yaml_with(tmp_path, settings={"stress_scale": value, **settings}, case0={"stress_limit": 5.0})))


def test_a_stress_scale_below_1_has_a_fact_in_the_form_of_the_one_above_1(tmp_path):
    """F4-28 (E2/SS05): 0.5 gave mass x0.761 and a relaxed peak of 0.957 MPa against 0.5 MPa with the row met, and `topopt check` said nothing. The two facts
    say the same things in the same order: the row, what it allows or holds, which way it moves the limit of the user (a requirement), the options."""
    from topopt.optimize import STRESS_SCALE_OPTIONS
    (above,), (below,) = scale_facts(tmp_path, 2.0), scale_facts(tmp_path, 0.5)
    assert above.startswith("fact: settings.stress_scale 2 is above 1. The row c*S - 1 <= 0 then asks for KS S <= 0.5, so the relaxed peak is held to at most 0.5 times the limit")
    assert below.startswith("fact: settings.stress_scale 0.5 is below 1. The row c*S - 1 <= 0 then asks for KS S <= 2, and KS is at or above the relaxed peak, "
                            "so the row allows a relaxed peak up to 2 times the limit and is met there")
    assert "(stricter than the limit)" in above and "(looser than the limit)" in below
    assert "This tightens the stress limit of the user, a requirement" in above
    assert "This loosens the stress limit of the user, a requirement" in below and "Only the user can allow it (the tool does not change the limit)" in below
    for text in (above, below):
        assert text.endswith(STRESS_SCALE_OPTIONS) and "\n" not in text
        assert "options: 1 (the KS row is then a bound of the relaxed peak" in text and "or auto (the default" in text
        assert not re.search(r"\b(must|never|always)\b", text, re.I)


@pytest.mark.parametrize("value, expected", [(0.999, 1), (1.001, 1), (1e-3, 1), (50.0, 1), (1.0, 0), (1, 0), ("auto", 0)])
def test_the_stress_scale_fact_is_for_a_number_that_is_not_1_and_not_auto(tmp_path, value, expected):
    """1 is the limit as the user wrote it (a bound of the relaxed peak) and `auto` is the default (the run log has the stage-start fact): no fact on every stress run."""
    assert len(scale_facts(tmp_path, value)) == expected


def test_the_stress_scale_fact_needs_a_stress_row_and_gives_the_ratio_of_the_number(tmp_path):
    assert problem_facts(load_config(yaml_with(tmp_path, settings={"stress_scale": 0.5}))) == []                 # no stress limit: no row, the key has no effect
    assert "up to 4 times the limit" in scale_facts(tmp_path, 0.25)[0] and "KS S <= 4," in scale_facts(tmp_path, 0.25)[0]
    assert "at most 0.1 times the limit" in scale_facts(tmp_path, 10.0)[0]


def test_check_prints_the_stress_scale_fact_below_1_and_json_has_it_in_facts(tmp_path, capsys):
    from topopt.cli import main
    p = yaml_with(tmp_path, settings={"stress_scale": 0.5}, case0={"stress_limit": 5.0})
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.count("fact: settings.stress_scale 0.5 is below 1.") == 1 and "loosens the stress limit of the user" in out
    assert main(["check", "--json", str(p)]) == 0
    facts = [f for f in json.loads(capsys.readouterr().out)["facts"] if "stress_scale" in f]
    assert len(facts) == 1 and facts[0].startswith("fact: settings.stress_scale 0.5 is below 1.")


def test_the_run_log_starts_with_the_stress_scale_fact_below_1_and_ends_with_the_peak_line_when_the_peak_is_above_the_limit(tmp_path):
    """A short real run with stress_scale 0.5: the fact is in the log before pass 0, and the flag in metrics.json is the line at the end of the log (F4-34)."""
    from topopt.cli import main
    p = yaml_with(tmp_path, settings={"stress_scale": 0.5, "iterations": 10, "objective": "mass"}, case0={"stress_limit": 5.0}, case1={"stress_limit": 5.0})
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text().splitlines()
    first = next(i for i, t in enumerate(log) if t.startswith("fact: settings.stress_scale 0.5 is below 1."))
    assert first < next(i for i, t in enumerate(log) if t.startswith("it "))       # before pass 0
    cases = json.loads((out / "result" / "metrics.json").read_text())["stress"]["cases"]
    lines = [t for t in log if t.startswith("stress peak above the limit:")]
    flagged = [n for n, c in cases.items() if c["relaxed"]["peak_above_limit"]]
    assert len(lines) == len(flagged) and all(f"stress:{n} " in line for n, line in zip(flagged, lines))
