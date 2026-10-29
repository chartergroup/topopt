"""Tests for the lost-force report (B-007): load nodes that are also fixed nodes."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from topopt.cli import check_data, lost_force_lines, main
from topopt.config import load_config
from topopt.loads import lost_force, lost_force_warning
from topopt.optimize import reduced_system
from topopt.problem import build_problem
from test_input import box, make_case

HOOK = Path(__file__).resolve().parent.parent / "examples" / "hook" / "hook.yaml"


def touching_case(tmp_path):
    """Writer test of the KB: load box 3 x 2 x 3 voxels on the plate face, Fz -10 N."""
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "load.stl", (0, 4, 0), (3, 6, 3))
    d = {"name": "touch",
         "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"l": "load.stl"}},
         "load_cases": [{"name": "down", "forces": [{"region": "l", "force": [0, 0, -10.0]}]}],
         "settings": {"voxel_size": 1.0, "volume_fraction": 0.3, "iterations": 5, "solver": "amg"}}
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_no_overlap_reports_zero(tmp_path, capsys):
    p = make_case(tmp_path)
    d = check_data(p)
    lf = d["lost_force"]
    assert lf["max_lost_percent"] == 0
    assert all(c["lost_percent"] == 0 and c["regions"] == [] for c in lf["cases"])
    assert not any("force is lost" in w for w in d["warnings"])
    assert lost_force_lines(lf) == []
    assert main(["check", str(p)]) == 0
    assert "lost force" not in capsys.readouterr().out


def test_overlap_one_quarter(tmp_path, capsys):
    p = touching_case(tmp_path)
    d = check_data(p)
    c = d["lost_force"]["cases"][0]
    assert c["total_n"] == pytest.approx([0, 0, -10.0])
    assert c["lost_n"] == pytest.approx([0, 0, -2.5])      # KB: one quarter
    assert c["lost_percent"] == pytest.approx(25.0)
    assert d["lost_force"]["max_lost_percent"] == pytest.approx(25.0)
    r = c["regions"][0]
    assert r["region"] == "l" and r["lost_magnitude_n"] == pytest.approx(2.5)
    w = [w for w in d["warnings"] if "force is lost" in w]
    assert len(w) == 1 and "25.0 percent" in w[0]
    assert "load nodes touch fixed nodes; their force is dropped by the solver" in w[0]
    assert "move the load region off the fixed region" in w[0]
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.count("lost force:") == 1 and "25.0 percent" in out


def test_solver_reduced_force_unchanged(tmp_path):
    """The solver force is the force array of build_forces on the free dofs; lost_force changes nothing."""
    from topopt.loads import build_forces
    prob = build_problem(load_config(touching_case(tmp_path)))
    before = prob.forces.copy()
    fixed_before = prob.fixed_dofs.copy()
    lf = lost_force(prob.config, prob.load_region_ids, prob.fixed_dofs, prob.forces)
    assert np.array_equal(prob.forces, before) and np.array_equal(prob.fixed_dofs, fixed_before)
    rebuilt = build_forces(prob.config, prob.load_region_ids)    # independent rebuild
    assert np.array_equal(rebuilt, before)
    _e, _a, free = reduced_system(prob)
    f_red = rebuilt[:, free]
    assert f_red.sum() == pytest.approx(-7.5)
    assert f_red.sum() + lf["cases"][0]["lost_n"][2] == pytest.approx(-10.0)


def two_region_case(tmp_path, force_b=(0, 0, -8.0)):
    """Region a touches the plate; region b is far from it. Two load cases."""
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "a.stl", (0, 1, 0), (3, 3, 3))      # touches the plate
    box(tmp_path / "b.stl", (12, 6, 0), (15, 9, 3))    # does not touch
    d = {"name": "two",
         "regions": {"design": "design.stl", "fixed": ["plate.stl"],
                     "load": {"a": "a.stl", "b": "b.stl"}},
         "load_cases": [
             {"name": "both", "forces": [{"region": "a", "force": [0, 0, 10.0]},
                                         {"region": "b", "force": list(force_b)}]},
             {"name": "only_b", "forces": [{"region": "b", "force": [1.0, 0, 0]}]}],
         "settings": {"voxel_size": 1.0, "volume_fraction": 0.3, "iterations": 3, "solver": "amg"}}
    p = tmp_path / "two.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_several_regions_and_cases(tmp_path):
    lf = check_data(two_region_case(tmp_path, (0, 0, 10.0)))["lost_force"]
    both, only_b = lf["cases"]
    assert only_b["lost_magnitude_n"] == 0 and only_b["lost_percent"] == 0
    assert only_b["lost_percent_of_sum"] == 0 and only_b["regions"] == []
    assert [r["region"] for r in both["regions"]] == ["a"]       # b shares no node
    assert both["total_n"] == pytest.approx([0, 0, 20.0])
    summed = np.sum([r["lost_n"] for r in both["regions"]], axis=0)
    assert summed == pytest.approx(both["lost_n"])
    assert both["lost_percent"] == pytest.approx(100 * both["lost_magnitude_n"] / 20.0)
    assert both["lost_percent_of_sum"] == pytest.approx(100 * both["lost_magnitude_n"] / 20.0)
    assert lf["max_lost_percent"] == pytest.approx(both["lost_percent"])


def test_cancelling_forces(tmp_path, capsys):
    d = check_data(two_region_case(tmp_path, (0, 0, -8.0)))      # total 2 N, lost > 2 N
    c = d["lost_force"]["cases"][0]
    assert c["total_magnitude_n"] == pytest.approx(2.0)
    assert c["lost_percent"] > 100
    assert c["lost_percent_of_sum"] == pytest.approx(100 * c["lost_magnitude_n"] / 18.0)
    w = [w for w in d["warnings"] if "force is lost" in w]
    assert len(w) == 1 and "cancel" in w[0] and "of the sum of the region force sizes" in w[0]
    # nearly cancelling: a huge percent, the percent of the sum stays small
    d0 = check_data(two_region_case(tmp_path, (0, 0, -9.999)))
    c0 = d0["lost_force"]["cases"][0]
    assert c0["lost_percent"] > 1000 and c0["lost_percent_of_sum"] < 100
    assert d0["lost_force"]["max_lost_percent"] == pytest.approx(c0["lost_percent"])
    assert main(["check", str(tmp_path / "two.yaml")]) == 0
    assert "lost force:" in capsys.readouterr().out
    # total about 0 (the YAML sum check does not allow an exact 0 total): percent is None
    lf = {"cases": [{"name": "z", "lost_magnitude_n": 1.5, "total_magnitude_n": 0.0,
                     "lost_percent": None, "lost_percent_of_sum": 7.5}]}
    assert "cancel" in lost_force_warning(lf)
    assert "total about 0" in lost_force_lines(lf)[0]


def test_mcp_check_problem_returns_lost_force(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    res = server.check_problem(str(touching_case(tmp_path)))
    assert res["ok"] and res["lost_force"]["cases"][0]["lost_percent"] == pytest.approx(25.0)
    assert any("force is lost" in w for w in res["warnings"])


def test_run_log_line(tmp_path):
    from topopt.cli import cmd_run
    import argparse
    out = tmp_path / "o"
    args = argparse.Namespace(yaml=str(touching_case(tmp_path)), out=str(out), dashboard=False,
                              port=8765, quiet=True, no_wait=True)
    cmd_run(args)
    log = (out / "run.log").read_text().splitlines()
    assert sum(1 for ln in log if ln.startswith("warning: lost force:")) == 1
    assert json.loads((out / "run.json").read_text())["lost_force"]["cases"][0]["lost_percent"] \
        == pytest.approx(25.0)
    ok = tmp_path / "ok"
    ok.mkdir()
    out2 = tmp_path / "o2"
    args.yaml, args.out = str(make_case(ok, h=1.0, mutate=lambda d: d["settings"].update(
        solver="amg", iterations=2))), str(out2)
    cmd_run(args)
    assert "lost force" not in (out2 / "run.log").read_text()


def test_run_json_key(tmp_path):
    from topopt.snapshot import RunWriter
    prob = build_problem(load_config(touching_case(tmp_path)))
    out = tmp_path / "run"
    out.mkdir()
    w = RunWriter(out, 0.0, quiet=True)
    w.write_run_json(prob, 10, 10, "amg")
    run = json.loads((out / "run.json").read_text())
    assert run["lost_force"]["cases"][0]["lost_percent"] == pytest.approx(25.0)
    assert "lost_percent_of_sum" in run["lost_force"]["cases"][0]
    assert run["format"] == 1 and "load_cases" in run and "derived" in run


def test_hook_example(tmp_path):
    """The hook example as shipped has no overlap; a plate that covers the tip face shows a loss."""
    d = check_data(HOOK)
    assert d["lost_force"]["max_lost_percent"] == 0
    # variant: the plate STL is replaced by a box that reaches into the tip region
    work = tmp_path / "hook"
    shutil.copytree(HOOK.parent, work)
    import trimesh
    tip = trimesh.load(work / "hook_tip.stl")
    lo, hi = tip.bounds
    box(work / "wall_plate.stl", lo - [5, 5, 5], [lo[0] + 3, hi[1] + 5, hi[2] + 5])
    d2 = check_data(work / "hook.yaml")
    assert d2["lost_force"]["max_lost_percent"] > 0
    assert any("force is lost" in w for w in d2["warnings"])
