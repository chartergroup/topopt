"""Merge gate A2 (stress report) x A4b (load and support types): a moment or a pressure is not a false change of the forces.

`run.json` holds a pressure as an entry of the load case and the moment of a force entry; the forces of the YAML (`LoadCase.forces`) hold
neither. `stress_report.input_changes` therefore skips the force comparison for a YAML with extended loads and says so (CONTRACTS.md
sections 17 and 26.2).
"""

import yaml

from test_stress_report import stress_yaml
from topopt import stress_report as sr
from topopt.cli import main


def test_a_moment_and_a_pressure_are_not_a_false_change_of_the_forces(tmp_path):
    p = stress_yaml(tmp_path, iterations=6)
    d = yaml.safe_load(p.read_text())
    d["load_cases"][1]["forces"][0]["moment"] = [0.0, 5.0, 0.0]
    d["load_cases"][0]["pressures"] = [{"region": "tip", "value_mpa": 0.05}]
    p.write_text(yaml.safe_dump(d))
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    sec = sr.update_metrics(out)
    check = sec["input_check"]
    assert check["changed"] is False, check
    assert any("not compared with run.json" in t for t in check["not_compared"])
    for case in sec["cases"].values():
        assert "input_changed" not in case and case["physical"]["voxels"] > 0
    assert check["compared_with_run_json"] == "material"      # the text names only what was compared
