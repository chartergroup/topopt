"""Merge gate A4b x A3b (#35): the stiffness rule of A4b sits in `post.case_stiffness`, so both lists get it."""

import json

import yaml

from test_robust import settings_of
from topopt.cli import main


def test_a_moment_or_a_pressure_case_has_a_null_stiffness_with_a_reason_in_both_lists(tmp_path):
    p = settings_of(tmp_path, robust="eroded", eta_eroded=0.6, iterations=4)
    d = yaml.safe_load(p.read_text())
    assert [c["name"] for c in d["load_cases"]] == ["down", "side"]
    d["load_cases"][1]["forces"] = [{"region": "tip", "moment": [0.0, 5.0, 0.0]}]      # a pure moment
    d["load_cases"].append({"name": "push", "pressures": [{"region": "tip", "value_mpa": 0.1}]})
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    nominal, eroded = metrics["stiffness_cases"], metrics["robust"]["stiffness_cases_eroded"]
    for lst in (nominal, eroded):
        assert [c["name"] for c in lst] == ["down", "side", "push"]
        assert lst[0]["N_per_mm"] > 0 and "reason" not in lst[0]
        for c in lst[1:]:
            assert c["N_per_mm"] is None and "plain forces" in c["reason"], c
    assert "moment" in nominal[1]["reason"] and "pressure" in nominal[2]["reason"]
