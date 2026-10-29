"""Tests for step acceptance of the stress rows (A2 fixes, round 1: REVIEW M1) and the optimizer state (m5).

The rule (`topopt.optimize.StepAcceptance`): after an MMA step, solve at the trial design and read the stress
rows. Accept if each row has `g_new <= max(g_old, 0) + 1e-3`. Else restore the MMA state, halve the move limit
and try again (6 tries at most; the last step is taken). After an accepted step the move limit grows by 1.5, up to
the limit of the settings. The solve of the accepted trial design is the solve of the next pass.

Tolerances: the run tests use the cantilever of test_optimize (1988 design voxels at 1 mm) with a mass objective and
two stress limits, 24 passes. Measured (PARDISO, 1 thread): with the rule the worst row ends at g +0.0014, without it
at +1.12 and the rows swing between 0.05 and 1.6. The tests use the margins 5e-3 and 5e-2.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_stress_report import stress_yaml
from topopt.cli import main
from topopt.config import load_config
from topopt.formulation import Constraint, Formulation, Response, mass_objective
from topopt.ks_stress import stress_ks_constraint
from topopt.mma import Mma, MmaState
from topopt.optimize import (STEP_GROW, STEP_HALVE, STEP_TOL, STEP_TRIES, Model, StepAcceptance,
                             formulation_from_config, minimize)
from topopt.problem import build_problem


# --- the rule on its own (scripted rows, no solve) ----------------------------------------


class FakeMma:
    """Records the move limit of each call and keeps a state that the rule can restore."""

    def __init__(self, n=4):
        self.state = MmaState(n)
        self.moves = []

    def step(self, x, f0, df0, g, dg, a0, a, c, move=None):
        self.moves.append(move)
        self.state.it += 1
        self.state.x = np.asarray(x, float) * 0.9
        return self.state.x.copy()


class FakeModel:
    def __init__(self):
        self.solves = 0

    def density(self, x, beta):
        return x, x

    def stiffness_density(self, rho_tilde, rho, beta):      # A3b: the density that the stiffness sees (rho in a default run)
        return rho

    def stiffness(self, rho):
        return rho

    def solve(self, K):
        self.solves += 1
        return f"U{self.solves}"

    def design_point(self, x, beta, rho_tilde, rho, U, rho_stiff=None):
        return SimpleNamespace(U=U)


class ScriptedForm:
    """`measure` returns the next scripted g for the one stress row. The rule reads `constraints` and `measure`."""

    def __init__(self, g_script):
        self.constraints = [Constraint("stress:a", lambda p: Response(0.0, np.zeros(4)), limit=1.0, scale=1.0,
                                       stage_aware=True, measure=lambda p: 0.0),
                            Constraint("volume", lambda p: Response(0.0, np.zeros(4)), limit=1.0)]
        self.script = list(g_script)

    def measure(self, point, names=None):
        return {"stress:a": self.script.pop(0)}


def make_rule(g_script, move_max=0.03):
    mma = FakeMma()
    rule = StepAcceptance(FakeModel(), ScriptedForm(g_script), mma, move_max)
    ev = SimpleNamespace(f0=1.0, df0=np.zeros(4), g=np.zeros(2), dg=np.zeros((2, 4)), a0=1.0, a=np.zeros(2),
                         c=np.full(2, 1000.0), constraints=[{"name": "stress:a", "g": None}, {"name": "volume", "g": 0.0}])
    return rule, mma, ev


def test_only_stage_aware_rows_with_a_measure_are_checked():
    rule, _, _ = make_rule([0.0])
    assert rule.names == ["stress:a"] and rule.active
    none = StepAcceptance(FakeModel(), Formulation(mass_objective(), [Constraint("v", lambda p: (0.0, np.zeros(4)))]),
                          FakeMma(), 0.03)
    assert none.names == [] and not none.active
    off = StepAcceptance(FakeModel(), ScriptedForm([]), FakeMma(), 0.03, enabled=False)
    assert not off.active


def test_a_step_that_keeps_the_row_is_accepted_with_one_solve():
    rule, mma, ev = make_rule([0.0015])                 # g_old 0.001: allowed up to 0.001 + STEP_TOL = 0.002
    ev.constraints[0]["g"] = 0.001
    x = np.ones(4)
    x_new, info = rule.step(x, ev, 1.0)
    assert info["accepted"] and info["tries"] == 1 and info["rejected"] == [] and mma.moves == [0.03]
    assert info["move_next"] == 0.03 and mma.state.move == 0.03              # grown by 1.5, capped at the limit
    assert rule.solved(x_new, 1.0) == "U1" and rule.solved(x_new, 1.0) is None      # used once
    assert rule.model.solves == 1 and rule.lines(3, info) == []


def test_a_rejected_step_halves_the_move_and_restores_the_state():
    # g_old 0.002 -> the bound is 0.003: tries 1, 2 and 3 fail, the 4th passes
    rule, mma, ev = make_rule([0.05, 0.03, 0.01, 0.0025])
    ev.constraints[0]["g"] = 0.002
    x = np.ones(4)
    x_new, info = rule.step(x, ev, 2.0)
    assert info["accepted"] and info["tries"] == 4
    assert mma.moves == pytest.approx([0.03, 0.015, 0.0075, 0.00375])
    assert [r["move"] for r in info["rejected"]] == pytest.approx([0.03, 0.015, 0.0075])
    assert [r["g"]["stress:a"] for r in info["rejected"]] == [0.05, 0.03, 0.01]
    assert mma.state.it == 1                                   # the three rejected steps left no trace in the state
    assert info["move"] == pytest.approx(0.00375) and info["move_next"] == pytest.approx(0.00375 * STEP_GROW)
    assert mma.state.move == pytest.approx(0.005625)
    assert rule.model.solves == 4 and rule.solved(x_new, 1.0) is None   # another beta: the solve cannot be used
    lines = rule.lines(7, info)
    assert len(lines) == 1 and "pass 7" in lines[0] and "3 of 4 tries rejected" in lines[0]
    assert "stress:a +0.0500" in lines[0] and "Accepted with move limit 0.00375" in lines[0]
    assert rule.n_rejected == 3 and rule.n_rejected_passes == 1 and rule.n_forced == 0


def test_after_the_last_try_the_step_is_taken_and_recorded():
    rule, mma, ev = make_rule([0.5] * STEP_TRIES)       # every try makes the row worse
    ev.constraints[0]["g"] = 0.0
    x_new, info = rule.step(np.ones(4), ev, 1.0)
    assert not info["accepted"] and info["tries"] == STEP_TRIES and len(info["rejected"]) == STEP_TRIES - 1
    assert info["g_taken"] == {"stress:a": 0.5} and mma.state.it == 1       # the last step stays in the state
    assert info["move"] == pytest.approx(0.03 * STEP_HALVE ** (STEP_TRIES - 1)) and info["move_next"] == info["move"]
    assert rule.solved(x_new, 1.0) == f"U{STEP_TRIES}"
    line = rule.lines(2, info)[0]
    assert f"All {STEP_TRIES} tries were rejected" in line and "the step with move limit" in line
    assert rule.n_rejected == STEP_TRIES and rule.n_forced == 1
    assert "1 passes took the step after 6 rejected tries" in rule.summary()[0]


def test_the_move_limit_grows_again_after_accepted_steps_and_is_capped():
    rule, mma, ev = make_rule([0.5, 0.0] + [0.0] * 8)
    ev.constraints[0]["g"] = 0.0
    x = np.ones(4)
    moves = []
    for _ in range(9):
        x, info = rule.step(x, ev, 1.0)
        moves.append(info["move"])
    # pass 1: the tries 0.03 (rejected) and 0.015 (accepted); pass 2 starts at 0.015 * 1.5; pass 3 at 0.0225 * 1.5, capped
    assert moves[:3] == pytest.approx([0.015, 0.0225, 0.03])
    assert max(moves) <= 0.03 and moves[-1] == pytest.approx(0.03)   # back at the limit of the settings


def test_the_tolerance_is_in_g_and_a_row_above_zero_may_not_rise_by_more():
    rule, _, ev = make_rule([0.0125, 0.0105])
    ev.constraints[0]["g"] = 0.01                       # the bound is 0.01 + 0.001
    _, info = rule.step(np.ones(4), ev, 1.0)
    assert info["tries"] == 2 and STEP_TOL == 1e-3
    rule, _, ev = make_rule([-0.5])
    ev.constraints[0]["g"] = -0.9                       # a met row: the bound is max(g, 0) + tol, so it may rise to 0.001
    _, info = rule.step(np.ones(4), ev, 1.0)
    assert info["accepted"] and info["tries"] == 1


def test_mma_step_takes_a_move_and_checks_it():
    mma = Mma(3, 1, move=0.2)
    args = (np.full(3, 0.5), 1.0, np.ones(3), np.array([0.1]), np.ones((1, 3)))
    small = mma.step(*args, move=0.01)
    assert np.max(np.abs(small - 0.5)) <= 0.01 + 1e-12
    big = Mma(3, 1, move=0.2).step(*args)
    assert np.max(np.abs(big - 0.5)) > 0.01
    with pytest.raises(ValueError, match="move limit"):
        Mma(3, 1).step(*args, move=0.0)


# --- runs ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """The same 24-pass mass run with the rule on and off."""
    tmp = tmp_path_factory.mktemp("accept")
    p = stress_yaml(tmp, iterations=24)
    out = {}
    for on in (True, False):
        m = Model(build_problem(load_config(p)))
        solves = {"n": 0}
        original = m.solve

        def counted(K, original=original, solves=solves):
            solves["n"] += 1
            return original(K)

        m.solve = counted
        res = minimize(m, formulation_from_config(m.problem.config, m.n_design), step_acceptance=on)
        m.solver.free()
        out[on] = (res, solves["n"])
    return out


def worst_g(res, last=4):
    return max(c["g"] for r in res.history[-last:] for c in r["constraints"])


def test_the_rule_holds_the_rows_that_the_plain_run_does_not(pair):
    on, off = pair[True][0], pair[False][0]
    assert worst_g(on) <= 5e-3                                   # met at the end (measured +0.0014)
    assert worst_g(off) > 5e-2                                   # not met without the rule (measured +1.12)
    assert max(max(c["g"] for c in r["constraints"]) for r in off.history[5:]) > 0.3      # it swings far above the limit
    assert max(max(c["g"] for c in r["constraints"]) for r in on.history[5:]) < 0.2       # and stays near it with the rule
    assert not any("step_acceptance" in r for r in off.history) and off.step_facts == []


def test_the_history_and_the_facts_record_the_rejected_tries(pair):
    res, _ = pair[True]
    steps = [r["step_acceptance"] for r in res.history[:-1]]
    assert all(set(s) >= {"tries", "accepted", "move", "move_next", "g_old", "rejected"} for s in steps)
    assert "step_acceptance" not in res.history[-1]                         # the last pass only measures
    rejected = sum(len(s["rejected"]) + (0 if s["accepted"] else 1) for s in steps)
    assert rejected > 0 and all(len(s["rejected"]) == s["tries"] - 1 for s in steps if s["accepted"])
    for r in res.history[:-1]:
        for rej in r["step_acceptance"]["rejected"]:
            assert set(rej["g"]) == {"stress:down", "stress:side"} and 0 < rej["move"] <= 0.05
    assert res.step_facts[-1].startswith(f"fact: step acceptance (stress rows): {rejected} tries rejected")
    assert any(l.startswith("fact: step acceptance, pass ") and "tries rejected" in l for l in res.step_facts)
    assert res.mma.state.move is not None and 0 < res.mma.state.move <= 0.05


def test_the_solve_of_the_accepted_trial_design_is_used_by_the_next_pass(pair):
    """One solve for pass 0, one for each trial of each step, and one more after a beta change (the density differs)."""
    res, solves = pair[True]
    steps = [r["step_acceptance"] for r in res.history[:-1]]
    tries = sum(s["tries"] for s in steps)
    beta_changes = sum(1 for a, b in zip(res.history[:-1], res.history[1:]) if a["beta"] != b["beta"])
    assert solves == 1 + tries + beta_changes
    plain = pair[False][1]
    assert plain == 25                                           # the plain loop: 24 steps + the last pass + ... = n_iter + 1


def test_a_run_without_a_stress_row_does_not_use_the_rule(tmp_path):
    from test_multirow_mma import make_model
    m = make_model(tmp_path, iterations=6)
    form = formulation_from_config(m.problem.config, m.n_design)
    res = minimize(m, form)
    m.solver.free()
    assert not any("step_acceptance" in r for r in res.history) and res.step_facts == [] and res.mma.state.move is None


def test_the_cli_run_logs_and_reports_the_step_acceptance(tmp_path):
    p = stress_yaml(tmp_path, iterations=24)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text()
    assert "fact: step acceptance, pass " in log and "fact: step acceptance (stress rows):" in log
    hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    assert all("step_acceptance" in r for r in hist[:-1]) and "step_acceptance" not in hist[-1]
    run = json.loads((out / "run.json").read_text())
    assert run["stress"]["step_acceptance"] is True
    s = json.loads((out / "result" / "metrics.json").read_text())["stress"]
    assert s["parameters"]["step_acceptance"] is True
    assert s["step_acceptance"]["passes"] == 24 and s["step_acceptance"]["tries_rejected"] > 0
    assert all(c["relaxed"]["met"] for c in s["cases"].values())      # met at the end (measured +0.0014)


def test_the_setting_turns_the_rule_off(tmp_path):
    p = stress_yaml(tmp_path, iterations=8)
    d = yaml.safe_load(p.read_text())
    d["settings"]["stress_step_acceptance"] = False
    p.write_text(yaml.safe_dump(d))
    assert load_config(p).settings.stress_step_acceptance is False
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text()
    assert "step acceptance" not in log.replace("settings.stress_step_acceptance is false", "")
    assert "settings.stress_step_acceptance is false" in log
    hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    assert not any("step_acceptance" in r for r in hist)
    s = json.loads((out / "result" / "metrics.json").read_text())["stress"]
    assert s["parameters"]["step_acceptance"] is False and s["step_acceptance"] is None


# --- the optimizer state (REVIEW m5) ------------------------------------------------------


def test_the_ks_scale_and_the_stage_beta_are_in_the_optimizer_state(tmp_path):
    """A copy of the state restores `c`: a new row object continues the stage, a plain new row object starts a new one."""
    from test_multirow_mma import make_model
    m = make_model(tmp_path, iterations=6)
    rng = np.random.default_rng(4)
    x = rng.uniform(0.3, 0.7, m.n_design)
    beta = 2.0

    def point_at(x):
        rho_tilde, rho = m.density(x, beta)
        return m.design_point(x, beta, rho_tilde, rho, m.solve(m.stiffness(rho)))

    form_a = Formulation(mass_objective(), [stress_ks_constraint("down", 0.5)])
    state = MmaState(m.n_design)
    ev1 = form_a.evaluate(point_at(x), state)
    assert state.rows["stress:down"]["beta"] == beta and state.rows["stress:down"]["c"] == ev1.constraints[0]["c"]
    assert ev1.constraints[0]["c_changed"] is True
    x2 = np.clip(x + 0.1 * rng.normal(size=x.size), 0.05, 1.0)
    resumed = Formulation(mass_objective(), [stress_ks_constraint("down", 0.5)])    # a new row object, as after a resume
    saved = copy.deepcopy(state)
    ev_straight = form_a.evaluate(point_at(x2), state)                  # the straight run
    ev_resumed = resumed.evaluate(point_at(x2), saved)                  # the resumed run: c comes from the state
    assert ev_straight.constraints[0]["c"] == ev1.constraints[0]["c"] == ev_resumed.constraints[0]["c"]
    assert ev_resumed.constraints[0]["c_changed"] is False
    np.testing.assert_allclose(ev_straight.g, ev_resumed.g, rtol=1e-12, atol=0)     # AMG: the solves differ at 1e-14
    np.testing.assert_allclose(ev_straight.dg, ev_resumed.dg, rtol=1e-8, atol=1e-12)   # PARDISO: noise of 1e-16
    fresh = Formulation(mass_objective(), [stress_ks_constraint("down", 0.5)]).evaluate(point_at(x2), MmaState(m.n_design))
    assert fresh.constraints[0]["c"] != ev1.constraints[0]["c"]          # without the state c is set again
    m.solver.free()


def test_the_state_docstring_names_the_new_fields():
    doc = MmaState.__doc__
    assert "`rows`" in doc and "`move`" in doc and "KS scale" in doc
    assert MmaState(3).rows == {} and MmaState(3).move is None


def test_run_does_not_assemble_the_stiffness_when_the_solve_is_reused(tmp_path, monkeypatch):
    """R1-N2: `run` assembles K for pass 0, for each trial of the step acceptance, and after a beta change only."""
    import topopt.optimize as opt
    calls = {"n": 0}
    original = opt.Model.stiffness

    def counted(self, rho):
        calls["n"] += 1
        return original(self, rho)

    monkeypatch.setattr(opt.Model, "stiffness", counted)
    out = tmp_path / "r"
    assert main(["run", str(stress_yaml(tmp_path, iterations=24)), "--out", str(out), "--quiet"]) == 0
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    tries = sum(r["step_acceptance"]["tries"] for r in hist if "step_acceptance" in r)
    beta_changes = sum(1 for a, b in zip(hist[:-1], hist[1:]) if a["beta"] != b["beta"])
    assert beta_changes == 2
    assert calls["n"] == 1 + tries + beta_changes          # before the fix: tries + 25
