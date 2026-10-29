"""Robust mode with a stress row (merge of A3b and A2, CONTRACTS.md sections 17 and 22b).

A stress row reads `point.rho` (nominal) in its explicit term and `stiffness_derivative` (slope to `rho_stiff`) in its
adjoint term. In robust mode these are two densities, and the chain rule of one `wrt` cannot give both: the gradient is
wrong with no error (`stress_fd_probe.py` of the merge gate measured 4e-3 to 1e-1 against 1e-5 for a right gradient).
So robust mode with a stress limit stops at the start with the facts and the options. Options (a) and (b) are not built.
"""

import numpy as np
import pytest
import yaml

from test_multirow_mma import case_yaml
from test_stress_report import stress_yaml
from topopt import robust
from topopt.cli import check_data, main
from topopt.config import UserError, load_config
from topopt.filter import heaviside_project
from topopt.formulation import Formulation, mass_objective
from topopt.ks_stress import stress_ks_constraint
from topopt.mma import Mma
from topopt.optimize import Model, StepAcceptance
from topopt.problem import build_problem


def with_settings(path, **settings):
    d = yaml.safe_load(path.read_text())
    d["settings"].update(settings)
    path.write_text(yaml.safe_dump(d))
    return path


def test_robust_with_a_stress_limit_stops_before_the_solve_with_facts_and_options(tmp_path):
    p = with_settings(stress_yaml(tmp_path, iterations=3), robust="eroded")
    with pytest.raises(UserError) as e:
        Model(build_problem(load_config(p)))
    text = str(e.value)
    assert "settings.robust is eroded, and 2 load case(s) have a stress limit" in text
    assert "gradient mixes two densities" in text and "4e-3 to 1e-1" in text
    assert "Options: set settings.robust to off; or remove the stress limits" in text
    assert "(a) the stress of the eroded design" in text and "(b) a second solve of the nominal design" in text
    assert text.endswith("The run stops at the start")
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 1          # a stop, not a crash (exit 1)
    assert not list(out.glob("snapshots/*.bin"))                              # no pass ran
    assert "have a stress limit" in (out / "status.json").read_text()


def test_check_gives_the_same_text_as_a_warning_and_does_not_stop(tmp_path, capsys):
    p = with_settings(stress_yaml(tmp_path, iterations=3), robust="eroded")
    d = check_data(p)
    fact = [w for w in d["warnings"] if "have a stress limit" in w]
    assert len(fact) == 1 and fact[0].endswith("The run stops at the start")
    assert fact[0] in d["derived"]["robust"]["facts"]
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert [ln for ln in out.splitlines() if ln.startswith("warning: settings.robust is eroded, and 2 load case(s)")]


def test_no_stop_without_robust_or_without_a_stress_limit(tmp_path):
    off = Model(build_problem(load_config(stress_yaml(tmp_path, iterations=3))))       # stress rows, robust off
    assert off.robust is None
    off.solver.free()
    other = tmp_path / "plain"
    other.mkdir()
    plain = with_settings(case_yaml(other), robust="eroded")
    m = Model(build_problem(load_config(plain)))                                     # robust, no stress limit
    assert m.robust is not None
    m.solver.free()
    assert not [w for w in check_data(plain)["warnings"] if "have a stress limit" in w]


def test_step_acceptance_solves_the_trial_design_with_the_eroded_density(tmp_path):
    """S1: `StepAcceptance.step` gives `stiffness_density` to `stiffness`. The stop keeps a run from reaching it, so the
    model gets the robust object by hand here (the rule is for the day that option (a) or (b) is built)."""
    m = Model(build_problem(load_config(case_yaml(tmp_path, h=2.0))))
    m.robust = robust.Robust(0.5, 0.55, "default")
    form = Formulation(mass_objective(), [stress_ks_constraint("down", 0.5, scale=1.0)])
    mma = Mma(m.n_design, form.n_rows, 0.2, xmin=form.xmin)
    x = np.random.default_rng(3).uniform(0.3, 0.7, m.n_design)
    beta = 4.0
    rho_tilde, rho = m.density(x, beta)
    rho_stiff = m.stiffness_density(rho_tilde, rho, beta)
    point = m.design_point(x, beta, rho_tilde, rho, m.solve(m.stiffness(rho_stiff)), rho_stiff)
    ev = form.evaluate(point, mma.state)
    seen = []
    stiffness = m.stiffness
    m.stiffness = lambda r: (seen.append(np.array(r)), stiffness(r))[1]
    rule = StepAcceptance(m, form, mma, 0.2)
    x_new, _info = rule.step(x, ev, beta)
    tilde_new, rho_new = m.density(x_new, beta)
    assert np.array_equal(seen[-1], heaviside_project(tilde_new, beta, 0.55))     # the eroded density of the trial design
    assert not np.array_equal(seen[-1], rho_new)
    m.solver.free()
