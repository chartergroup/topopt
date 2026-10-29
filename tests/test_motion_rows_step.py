"""A5 rows with the stress rows and the mass objective of #22 (A2, CONTRACTS.md sections 17 and 31): merge-gate tests.

The rows are `stage_aware` and have no `measure`: the step acceptance of the stress rows does not check them. The mass objective has facts for a probe region
(it does not hold material) and for a `max` or `min` row (the other motion meets it).
"""

import json

import numpy as np
import pytest
import yaml

from test_motion_rows import ALL_ROWS, limits_case, sub
from topopt import rows_motion as rm
from topopt.cli import main
from topopt.config import load_config
from topopt.mma import Mma
from topopt.optimize import Model, StepAcceptance, formulation_from_config
from topopt.problem import build_problem
from beta_fact_helpers import problem_facts_without_beta_fact as problem_facts   # the short fixtures end before the last beta stage: that fact (F4-18) is not the subject here

HOLDS_MATERIAL = "No row holds material"
ONE_SIDED = "is also met when the region moves the other way"


def history(path):
    return [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]


def test_rows_are_stage_aware_and_the_step_acceptance_does_not_check_them(tmp_path):
    m = Model(build_problem(load_config(limits_case(tmp_path, ALL_ROWS))))
    form = formulation_from_config(m.problem.config, m.n_design, m)
    rows = [c for c in form.constraints if c.name != "volume"]
    assert len(rows) == 10 and all(c.stage_aware and c.measure is None for c in rows)
    assert not StepAcceptance(m, form, Mma(m.n_design, len(form.constraints)), 0.2).active
    m.solver.free()


def test_a_run_with_the_rows_has_no_step_acceptance(tmp_path):
    p = limits_case(tmp_path, ALL_ROWS, h=2.0, iterations=6)
    out = sub(tmp_path, "r")
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert not any("step_acceptance" in r for r in history(out))
    assert "step acceptance" not in (out / "run.log").read_text()
    assert len(json.loads((out / "result" / "metrics.json").read_text())["rows"]) == 10


# --- the mass objective ------------------------------------------------------------------


def mass_case(tmp_path, limits):
    p = limits_case(tmp_path, limits, h=2.0, iterations=6)
    d = yaml.safe_load(p.read_text())
    d["settings"]["objective"] = "mass"
    p.write_text(yaml.safe_dump(d))
    return p


PROBE_ROW = {"displacement": [{"region": "mid", "limit_mm": 0.02, "direction": [0, 0, -1]}]}
TIP_ROW = {"displacement": [{"region": "tip", "limit_mm": 0.03}]}


def facts(tmp_path, limits, name="c", compliance_limit=None):
    p = mass_case(sub(tmp_path, name), limits)
    if compliance_limit is not None:
        d = yaml.safe_load(p.read_text())
        d["load_cases"][0]["compliance_limit"] = compliance_limit
        p.write_text(yaml.safe_dump(d))
    return problem_facts(load_config(p))


def test_a_mass_run_with_only_a_probe_row_has_the_fact_that_no_row_holds_material(tmp_path):
    lines = facts(tmp_path, PROBE_ROW)
    assert len(lines) == 1 and HOLDS_MATERIAL in lines[0]
    assert "a limit on a probe region does not" in lines[0] and "remove the material between the load and the probe" in lines[0]
    assert "give a limit a load region" in lines[0] and "The run goes on with the displacement and alignment rows only" in lines[0]
    cfg = load_config(mass_case(sub(tmp_path, "form"), PROBE_ROW))
    assert [c.name for c in formulation_from_config(cfg, 10).constraints] == ["displacement:mid:down", "displacement:mid:side"]


def test_alignment_rows_on_probe_regions_have_the_fact_too(tmp_path):
    lines = facts(tmp_path, {"alignment": ALL_ROWS["alignment"]})
    assert len(lines) == 1 and HOLDS_MATERIAL in lines[0]


def test_a_mass_run_with_no_limits_keeps_the_text_of_a2(tmp_path):
    lines = facts(tmp_path, None)
    assert len(lines) == 1 and HOLDS_MATERIAL in lines[0] and "limit a load region" not in lines[0]
    assert "The run goes on with a volume ceiling at settings.volume_fraction" in lines[0]


def test_a_row_on_a_load_region_holds_material_and_a_limit_on_a_case_does_too(tmp_path):
    assert facts(tmp_path, TIP_ROW, "tip") == []                                                    # a displacement row on a load region
    al = {"alignment": [{"region_a": "tip", "region_b": "a", "axis": [1, 0, 0], "tilt_limit_rad": 1e-3}]}
    assert facts(tmp_path, al, "align") == []                                                       # one load region of two is enough
    assert facts(tmp_path, PROBE_ROW, "compliance", compliance_limit=5.0) == []                     # a compliance limit holds it


def test_a_one_sided_row_has_a_fact_under_the_mass_objective_only(tmp_path):
    one = {"displacement": [{"name": "up", "region": "tip", "limit_mm": 0.03, "direction": [0, 0, -1], "sense": "max"}]}
    lines = facts(tmp_path, one, "max")
    assert len(lines) == 1 and "displacement limit 'up' has the sense max" in lines[0] and ONE_SIDED in lines[0]
    assert "use the sense abs (the default)" in lines[0]
    both = {"displacement": ALL_ROWS["displacement"][2:]}       # a max and a min row on the probe 'mid'
    lines = facts(tmp_path, both, "both")
    assert [HOLDS_MATERIAL in t for t in lines] == [True, False, False] and ONE_SIDED in lines[1] and ONE_SIDED in lines[2]
    # the compliance objective: no fact
    cfg = load_config(limits_case(sub(tmp_path, "comp"), one))
    assert problem_facts(cfg) == []


def test_the_unmet_text_names_the_load_path_and_has_no_volume_claim_for_these_rows(tmp_path):
    entry = {"name": "displacement:mid:down", "value": 0.03, "limit": 0.02, "g": 0.5, "violation": 0.5}
    comp = formulation_from_config(load_config(limits_case(sub(tmp_path, "comp"), PROBE_ROW)), 10)
    text = "\n".join(comp.unmet([entry]))
    assert "this limit is not met at the end of the run" in text and "does not fit at this volume" not in text
    assert "options: raise settings.volume_fraction, relax the limit, or move the region onto the load path" in text
    mass = formulation_from_config(load_config(mass_case(sub(tmp_path, "mass"), PROBE_ROW)), 10)
    text = "\n".join(mass.unmet([entry]))
    assert "this limit is not met at the end of the run" in text and "does not fit at this volume" not in text
    assert "volume_fraction" not in text and "move the region onto the load path" in text and "compliance_limit" in text


def test_a_mass_run_with_a_displacement_limit_runs(tmp_path):
    out = sub(tmp_path, "r")
    assert main(["run", str(mass_case(tmp_path, PROBE_ROW)), "--out", str(out), "--quiet"]) == 0
    assert (out / "result" / "metrics.json").exists()
    assert HOLDS_MATERIAL in (out / "run.log").read_text()
