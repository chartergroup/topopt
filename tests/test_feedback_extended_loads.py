"""Merge gate A4b x V3 (#28): `topopt/feedback.py` and the load and support types of A4b (CONTRACTS.md section 26.2).

The tests need the A4b code in the tree (a moment, a remote force, a pressure, a per-axis support). They skip without it.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import feedback  # noqa: E402
from topopt.config import Config  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(Config, "uses_extended_loads"), reason="needs A4b (load and support types)")

from test_feedback import box, fake_run, make_case  # noqa: E402


def extended_case(tmp_path):
    """The V3 test case with a moment, a remote force, a pressure and a roller (a solid slab at the side face)."""
    ypath = make_case(tmp_path / "in")
    d = ypath.parent
    box(d / "side.stl", (0, -8, 0), (24, -6, 12))
    doc = yaml.safe_load(ypath.read_text())
    doc["regions"]["fixed"] = ["plate.stl", {"file": "side.stl", "axes": ["Y"]}]
    doc["load_cases"] += [
        {"name": "twist", "weight": 1.0, "forces": [{"region": "tip", "moment": [0.0, 3.0, 0.0]}]},
        {"name": "remote", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -1.0], "at": [30.0, 0.0, 2.0],
                                                      "moment": [0.0, 2.0, 0.0]}]},
        {"name": "push", "weight": 1.0, "pressures": [{"region": "tip", "value_mpa": 0.1}]},
    ]
    ypath.write_text(yaml.safe_dump(doc, sort_keys=False))
    return ypath


def differs(r, side="a"):
    return {x["key"]: x for x in r["inputs"][side]["yaml_vs_run_json"]["differs"]}


def edited(ypath, change):
    doc = yaml.safe_load(ypath.read_text())
    change(doc)
    ypath.write_text(yaml.safe_dump(doc, sort_keys=False))


def test_an_unchanged_yaml_has_no_difference_from_run_json(tmp_path):
    ypath = extended_case(tmp_path)
    a = fake_run(tmp_path / "runs" / "a", ypath)
    b = fake_run(tmp_path / "runs" / "b", ypath, compliance=0.5)
    r = feedback.compare_runs(a, b)
    for side in "ab":
        v = r["inputs"][side]["yaml_vs_run_json"]
        assert v["differs"] == [], v["differs"]       # a pure moment, a remote force, a pressure, a roller: no false difference
        assert "load_cases (names, weights, forces, moments, pressures)" in v["compared"]
        assert "regions.fixed (axes)" in v["compared"] and "regions.fixed (file names)" in v["compared"]


def test_a_changed_moment_at_pressure_or_axes_is_a_difference(tmp_path):
    ypath = extended_case(tmp_path)
    a = fake_run(tmp_path / "runs" / "a", ypath)
    b = fake_run(tmp_path / "runs" / "b", ypath)

    def change(doc):
        doc["load_cases"][1]["forces"][0]["moment"] = [0.0, 4.0, 0.0]            # a pure moment
        doc["load_cases"][2]["forces"][0]["at"] = [31.0, 0.0, 2.0]               # the point of a remote force
        doc["load_cases"][3]["pressures"][0]["value_mpa"] = 0.2
        doc["regions"]["fixed"][1]["axes"] = ["x", "y"]
    edited(ypath, change)
    keys = differs(feedback.compare_runs(a, b))
    assert keys["load_cases[1].forces[0].moment"]["yaml"] == [0.0, 4.0, 0.0]
    assert "load_cases[2].forces[0].at" in keys and "load_cases[2].forces[0].moment" in keys   # the remote term moves too
    assert keys["load_cases[3].pressures"]["run_json"] == [("tip", [0.1])] and keys["load_cases[3].pressures"]["yaml"] == [("tip", [0.2])]
    assert keys["regions.fixed (axes)"]["yaml"]["side.stl"] == "xy" and keys["regions.fixed (axes)"]["run_json"]["side.stl"] == "y"
    assert "load_cases[0].forces" not in keys and "regions.fixed" not in keys


def test_a_moment_only_the_yaml_has_or_a_new_pressure_is_a_difference(tmp_path):
    ypath = extended_case(tmp_path)
    a = fake_run(tmp_path / "runs" / "a", ypath)
    edited(ypath, lambda doc: (doc["load_cases"][0]["forces"][0].update(moment=[0.0, 1.0, 0.0]),
                               doc["load_cases"][1].update(pressures=[{"region": "tip", "value_mpa": 0.5}])))
    keys = differs(feedback.compare_runs(a, a))
    assert "load_cases[0].forces[0].moment" in keys and keys["load_cases[0].forces[0].moment"]["run_json"] is None
    assert "load_cases[1].pressures" in keys


def test_two_runs_that_differ_only_in_a_moment_or_a_pressure_say_so(tmp_path):
    ya = extended_case(tmp_path / "x")
    yb = extended_case(tmp_path / "y")
    edited(yb, lambda doc: (doc["load_cases"][1]["forces"][0].update(moment=[0.0, 9.0, 0.0]),
                            doc["load_cases"][3]["pressures"][0].update(value_mpa=0.3)))
    a = fake_run(tmp_path / "runs" / "a", ya)
    b = fake_run(tmp_path / "runs" / "b", yb)
    r = feedback.compare_runs(a, b)
    assert any("The load cases differ between the runs" in n for n in r["notes"])
    assert {"load_cases[1].forces[0].moment", "load_cases[3].forces[0].pressure_mpa"} <= {x["key"] for x in r["setup"]["run_json_diff"]}
    f = feedback.run_facts(b)
    forces = {c["name"]: c["forces"] for c in f["settings"]["load_cases"]}
    assert forces["twist"][0]["moment"] == [0.0, 9.0, 0.0] and forces["twist"][0]["kind"] == "moment"
    assert forces["push"][0]["pressure_mpa"] == 0.3 and forces["down"][0] == {"region": "tip", "force": [0.0, 0.0, -5.0]}
    assert f["settings"]["supports_axes"] == {"plate.stl": "xyz", "side.stl": "y"}


def test_a_roller_file_is_hashed_and_a_change_of_it_is_seen(tmp_path):
    ya = extended_case(tmp_path / "x")
    yb = extended_case(tmp_path / "y")
    items = {x["key"]: x for x in feedback._region_files(ya, yaml.safe_load(ya.read_text()), with_hash=True)}
    assert items["fixed:side.stl"]["sha256"] and items["fixed:plate.stl"]["sha256"] and items["fixed:side.stl"]["exists"]
    box(yb.parent / "side.stl", (0, -9, 0), (24, -6, 12))      # a thicker roller slab in the second case
    a = fake_run(tmp_path / "runs" / "a", ya)
    b = fake_run(tmp_path / "runs" / "b", yb)
    r = feedback.compare_runs(a, b)
    assert {x["key"]: x["status"] for x in r["setup"]["region_files"]["diff"]} == {"fixed:side.stl": "changed"}


def test_a_plain_yaml_gives_the_same_facts_as_before(tmp_path):
    ypath = make_case(tmp_path / "in")
    a = fake_run(tmp_path / "runs" / "a", ypath)
    f = feedback.run_facts(a)
    assert f["settings"]["load_cases"][0]["forces"] == [{"region": "tip", "force": [0.0, 0.0, -5.0]}]
    assert "supports_axes" not in f["settings"]
    v = feedback.compare_runs(a, a)["inputs"]["a"]["yaml_vs_run_json"]
    assert v["differs"] == [] and "regions.fixed (axes)" not in v["compared"]
