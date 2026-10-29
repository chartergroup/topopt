"""Where springs (A7a, CONTRACTS.md section 35) meet other branches of integration/mvp-1.

The review of A7a found six joins that the branch alone does not show. Each has a test here.

- A6a buckling check: the springs are in `K` (M1).
- D1b resume and stop rule: the run loop gives reactions, also for a finished run and a stop-rule run (M2).
- C2 interfaces and A2 stress report: a spring region holds the part like a fixed region (M3).
- A3b robust mode: the reactions are those of the nominal design (m3).
- Warnings in `run` (m1).
- G1d gpu-mg: three places refused springs (m5). Since P2b gpu-mg takes them (`tests/test_gpu_mg_border.py`): the three tests below say so.
"""

import json

import numpy as np
import pytest
import yaml

from test_checkpoint import needs_direct
from test_gpu_mg_x_merge import fake_cpu_device  # noqa: F401  (a fixture)
from test_springs import box, case, prob_of
from test_stress_report import fake_stress
from topopt import buckling_voxel as bv
from topopt import cli
from topopt import springs as sp
from topopt import stress_report as sr
from topopt.cli import main
from topopt.config import load_config
from topopt.optimize import Model
from topopt.problem import build_problem
from topopt.voxelize import FIXED, LOAD


def run_args(p, out, *extra):
    return ["run", str(p), "--out", str(out), "--quiet", *extra]


def log_of(out):
    return (out / "run.log").read_text()


# --- M1: the buckling check ---------------------------------------------------------------------------------------


def column(tmp_path, clamp=True, spring=True, k=1.0e6, lateral_top=0.0, solver="auto"):
    """A column 4 x 4 x 24 mm along z, 1 mm voxels. The top carries a compressive force of 1 N.

    The base is a clamp plate (`clamp`) or a pad with a spring `k` on all axes (`spring` and no clamp).
    `lateral_top`: a spring [k, k, 0] on the top region (the lateral support of the free end).
    """
    box(tmp_path / "design.stl", (0, 0, 0), (4, 4, 24))
    box(tmp_path / "base.stl", (0, 0, -2), (4, 4, 0))
    box(tmp_path / "top.stl", (0, 0, 24), (4, 4, 25))
    d = {"name": "column", "regions": {"design": "design.stl", "fixed": [], "load": {"top": "top.stl"}},
         "load_cases": [{"name": "push", "forces": [{"region": "top", "force": [0, 0, -1.0]}]}],
         "settings": {"voxel_size": 1.0, "volume_fraction": 0.5, "iterations": 2, "solver": solver},
         "springs": []}
    if clamp:
        d["regions"]["fixed"] = ["base.stl"]
    elif spring:
        d["regions"]["load"]["base"] = "base.stl"
        d["springs"].append({"region": "base", "stiffness": [k, k, k]})
    if lateral_top:
        d["springs"].append({"region": "top", "stiffness": [lateral_top, lateral_top, 0.0]})
    if not d["springs"]:
        del d["springs"]
    p = tmp_path / "column.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def buckling_of(p):
    problem = build_problem(load_config(p))
    snap = np.full(int(np.prod(problem.shape)), 255, np.uint8)
    return problem, bv.check_problem(problem, snap, modes=2)


@needs_direct
def test_the_buckling_check_has_the_springs_in_k(tmp_path):
    """Springs only: the check runs (before, K had the six rigid modes). Clamp and a lateral spring: a higher load factor."""
    problem, only = buckling_of(column(tmp_path, clamp=False, k=1.0e6))
    assert only["status"] == "done" and only["lowest_load_factor"]["value"] > 0
    assert only["loose_solid_voxels"] == 0               # the spring pad is the anchor of the loose pieces
    _, clamp = buckling_of(column(_sub(tmp_path, "clamp"), clamp=True, spring=False))
    lam_spring, lam_clamp = only["lowest_load_factor"]["value"], clamp["lowest_load_factor"]["value"]
    assert lam_spring == pytest.approx(lam_clamp, rel=0.1) and lam_spring <= lam_clamp * 1.001   # a stiff pad is a nearly rigid base
    _, both = buckling_of(column(_sub(tmp_path, "both"), clamp=True, lateral_top=1.0e5))
    assert both["lowest_load_factor"]["value"] > 2.0 * lam_clamp          # a lateral support of the free end: 8 times for a rigid one


@needs_direct
def test_the_buckling_check_in_the_run_gives_a_result_for_springs_only(tmp_path):
    p = column(tmp_path, clamp=False, k=1.0e6)
    d = yaml.safe_load(p.read_text())
    d["buckling"] = {"check": True}
    p.write_text(yaml.safe_dump(d))
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    b = json.loads((out / "result" / "metrics.json").read_text())["buckling"]
    assert b["status"] == "done" and b["lowest_load_factor"]["value"] > 0


# --- M2: the run loop ---------------------------------------------------------------------------------------------


def test_a_resume_of_a_finished_spring_run_keeps_the_reactions(tmp_path):
    p = case(tmp_path, h=2.0)
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    history = (out / "history.jsonl").read_bytes()
    before = json.loads((out / "result" / "metrics.json").read_text())["reactions"]
    assert main(run_args(p, out, "--resume")) == 0                # NameError: react, before the fix
    assert (out / "history.jsonl").read_bytes() == history        # no pass ran
    text = log_of(out)
    assert "No pass is left to run" in text and text.count("reactions:      down:") >= 2   # the log line comes again
    assert json.loads((out / "result" / "metrics.json").read_text())["reactions"] == before


def test_a_resume_of_a_finished_plain_run_works(tmp_path):
    p = case(tmp_path, h=2.0, mutate=lambda d: d.pop("springs"))      # the plate is a full clamp
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    assert main(run_args(p, out, "--resume")) == 0
    assert "reactions" not in json.loads((out / "result" / "metrics.json").read_text())


def test_a_stop_rule_run_with_springs_gives_the_reactions(tmp_path):
    def rules(d):
        d["settings"].update(iterations=60, stop_rules={"time_budget_s": 0.05})
    p = case(tmp_path, h=2.0, mutate=rules)
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    v = json.loads((out / "verdict.json").read_text())
    assert v["state"] == "stopped_by_rule" and v["final_iteration"] < 60
    history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert "reactions" in history[-1] and all("reactions" not in r for r in history[:-1])
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    assert metrics["reactions"] == history[-1]["reactions"] and metrics["reactions"]["cases"][0]["force_error"] < 1e-4
    assert "reactions:      down:" in log_of(out)
    assert main(run_args(p, out, "--resume")) == 0                # the stopped run resumes as a finished run
    assert json.loads((out / "result" / "metrics.json").read_text())["reactions"] == history[-1]["reactions"]


# --- M3: the readers that treat FIXED as the only support ----------------------------------------------------------

BOSS = {"name": "s", "kind": "boss", "point": [15, 5, 9], "axis": [0, 0, 1], "radius_mm": 2.0, "length_mm": 4.0}
FLOATER = {"name": "floater", "kind": "boss", "point": [32, 5, 3], "axis": [0, 0, 1], "radius_mm": 2.0, "length_mm": 6.0}


def with_interfaces(entries):
    return lambda d: d.update(interfaces=entries)


def test_a_union_interface_on_springs_only_passes_check_and_a_floating_one_does_not(tmp_path, capsys):
    p = case(tmp_path, clamp=False, mutate=with_interfaces([BOSS]))
    d = cli.check_data(p)
    assert [f for f in d["interfaces"] if f["name"] == "s"][0]["has_fixed"] is True
    assert not any("no fixed voxel" in w for w in d["warnings"])
    assert main(["check", str(p)]) == 0
    assert "no fixed voxel" not in capsys.readouterr().out
    sub = _sub(tmp_path, "floating")
    q = case(sub, clamp=False, mutate=with_interfaces([BOSS, FLOATER]))
    d = cli.check_data(q)
    assert [f for f in d["interfaces"] if f["name"] == "floater"][0]["has_fixed"] is False       # a spring pad far away does not hold it
    assert any("interface 'floater'" in w and "no fixed voxel" in w for w in d["warnings"])
    assert main(run_args(q, sub / "out")) == 1
    assert "interface 'floater' (boss) holds" in capsys.readouterr().err


def test_a_union_interface_on_springs_only_runs(tmp_path):
    p = case(tmp_path, h=2.0, clamp=False, mutate=with_interfaces([BOSS]))
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    assert json.loads((out / "result" / "metrics.json").read_text())["interfaces"]["applied"]


def snapshot_of(problem, value):
    return np.where(problem.labels == 1, value, 0).astype(np.uint8)


def test_the_load_path_of_a_spring_only_part_is_connected(tmp_path):
    problem = prob_of(tmp_path, clamp=False)
    assert problem.fixed_dofs.size == 0
    facts = sr.load_path_facts(problem, snapshot_of(problem, 255))
    tip = facts["regions"]["tip"]
    assert tip["thresholded_design"]["connected_to_fixed"] is True and tip["used_by"] == ["down"]
    assert facts["regions"]["pad"]["used_by"] == []                      # a spring is not a force: the pad is not "used by" a case
    assert sr.case_load_path(facts, "down")["all_connected"] is True
    gap = snapshot_of(problem, 255).reshape(problem.shape)
    gap[14, :, :] = 0                                                     # a cut across the part between the tip and the pad
    cut = sr.load_path_facts(problem, gap)
    assert cut["regions"]["tip"]["thresholded_design"]["connected_to_fixed"] is False
    assert sr.case_load_path(cut, "down")["not_connected"] == ["tip"]
    whole = box_file(tmp_path / "whole.stl", (0, 0, 0), (22, 10, 13))      # the final mesh joins the tip and the pad
    assert sr.load_path_facts(problem, snapshot_of(problem, 255), whole)["regions"]["tip"]["final_stl"]["connected_to_fixed"] is True
    apart = box_file(tmp_path / "apart.stl", (0, 0, 0), (6, 10, 13))       # no material at the pad
    assert sr.load_path_facts(problem, snapshot_of(problem, 255), apart)["regions"]["tip"]["final_stl"]["connected_to_fixed"] is False


def box_file(path, lo, hi):
    box(path, lo, hi)
    return path


def test_the_peak_scope_leaves_out_the_spring_region(tmp_path):
    problem = prob_of(tmp_path, clamp=False)
    stress, label = fake_stress(problem, {1: 1.0, int(FIXED): 50.0, int(LOAD): 30.0})
    scope = sr.peak_scope(stress, problem)
    assert (scope == (label == 1)).all()                # the pad (spring) and the tip (force) are both left out, as a fixed region would be


def test_the_support_mask_is_fixed_and_spring_voxels(tmp_path):
    problem = prob_of(tmp_path)                           # a clamp and a pad
    mask = sp.support_mask(problem)
    pad = problem.load_region_ids == [lr.name for lr in problem.config.regions.load].index("pad")
    assert (mask == ((problem.labels == FIXED) | (pad & (problem.labels == LOAD)))).all()
    assert not mask[problem.load_region_ids == [lr.name for lr in problem.config.regions.load].index("tip")].any()
    plain = prob_of(_sub(tmp_path, "plain"), mutate=lambda d: d.pop("springs"))
    assert (sp.support_mask(plain) == (plain.labels == FIXED)).all()


def _sub(tmp_path, name):
    (tmp_path / name).mkdir()
    return tmp_path / name


def test_the_comparison_of_a_yaml_and_a_run_sees_a_changed_spring(tmp_path):
    from topopt import feedback
    p = case(tmp_path, h=2.0)
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    run = json.loads((out / "run.json").read_text())
    doc = yaml.safe_load(p.read_text())
    diffs, compared, _ = feedback._yaml_vs_run(run, doc)
    assert not [d for d in diffs if d["key"].startswith("springs")] and "springs (region, stiffness)" in compared
    doc["springs"][0]["stiffness"] = [100.0, 100.0, 250.0]
    diffs, _, _ = feedback._yaml_vs_run(run, doc)
    assert [d["key"] for d in diffs if d["key"].startswith("springs")] == ["springs[pad].stiffness"]
    doc["springs"] = []
    diffs, _, _ = feedback._yaml_vs_run(run, doc)
    assert [d["key"] for d in diffs if d["key"].startswith("springs")] == ["springs[pad]"]
    assert feedback._settings(run)["springs"] == [{"region": "pad", "stiffness_n_per_mm": [100.0, 100.0, 100.0]}]


# --- m3: the reactions of the density of the solve; robust mode -------------------------------------------------------


def test_the_reactions_need_the_density_of_the_solve(tmp_path):
    m = Model(prob_of(tmp_path, h=2.0, solver="amg"))
    m.solver.rtol = 1e-12
    rho = np.full(m.n_design, 0.7)
    U = m.solve(m.stiffness(rho))
    good = m.reaction_facts(U, rho)["cases"][0]
    other = m.reaction_facts(U, np.full(m.n_design, 0.3))["cases"][0]       # another design than the one that was solved
    assert good["force_error"] < 1e-8 and good["residual_rel"] < 1e-8
    assert other["force_error"] > 1e-4 and other["residual_rel"] > 1e-3      # a mismatch shows in the balance and in the residual
    assert not hasattr(m, "Ee_last")


def test_robust_mode_gives_the_reactions_of_the_nominal_design(tmp_path):
    p = case(tmp_path, h=2.0, mutate=lambda d: d["settings"].update(robust="eroded"))
    out = tmp_path / "out"
    assert main(run_args(p, out)) == 0
    history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert all("reactions" not in r for r in history)                        # the passes solve the eroded design
    robust = json.loads((out / "robust.json").read_text())
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    assert metrics["reactions"] == robust["reactions"]
    c = metrics["reactions"]["cases"][0]
    assert c["force_error"] < 1e-4 and "clamp" in c and "pad" in c["springs"]
    assert "reactions:      down:" in log_of(out)


# --- m1 and m4: the warnings and the residual ------------------------------------------------------------------------


def test_a_balance_error_gets_the_warning_prefix_and_the_residual_is_a_fact():
    rf = {"cases": [{"name": "c", "applied": {"force_n": [0, 0, 1.0]}, "springs": {"r": {"force_n": [0, 0, -1.0]}},
                     "force_error": 3e-3, "moment_error": 1e-9, "residual_rel": 0.2}]}
    bad = sp.reaction_lines(rf)[0]
    assert bad.startswith("warning: reactions:      c:") and "residual of the solve 2.0e-01 of the load" in bad
    rf["cases"][0].update(force_error=1e-9, residual_rel=1e-9)
    assert sp.reaction_lines(rf)[0].startswith("reactions:      c:")


def test_the_reactions_have_the_residual_of_the_solve(tmp_path):
    m = Model(prob_of(tmp_path, h=2.0, solver="amg"))
    m.solver.rtol = 1e-5
    rho = np.full(m.n_design, 0.5)
    c = m.reaction_facts(m.solve(m.stiffness(rho)), rho)["cases"][0]
    assert 0.0 < c["residual_rel"] < 1e-4


def test_run_logs_a_near_singular_spring_with_the_warning_prefix(tmp_path):
    p = case(tmp_path, h=2.0, k=(1e-9, 1e-9, 1e-9), mutate=lambda d: d["settings"].update(iterations=1))
    d = cli.check_data(p)
    assert any(w.startswith("spring on 'pad': its smallest stiffness is") for w in d["warnings"])
    out = tmp_path / "out"
    main(run_args(p, out))                       # a soft support: the run may or may not end well, the line is written first
    assert "warning: spring on 'pad': its smallest stiffness is" in log_of(out)


# --- n5: the faces by direction ---------------------------------------------------------------------------------------


def test_the_facts_count_the_faces_by_direction(tmp_path):
    problem = prob_of(tmp_path)
    r = problem.extended["springs"]["regions"][0]
    assert r["faces_by_direction"] == {"+x": 100, "-y": 20, "+y": 20, "-z": 20, "+z": 20} and sum(r["faces_by_direction"].values()) == r["faces"]
    line = [x for x in sp_lines(problem) if x.startswith("spring:")][0]
    assert "180 faces (+x 100, -y 20, +y 20, -z 20, +z 20)" in line


def sp_lines(problem):
    from topopt.loads import extended_facts, extended_lines
    return extended_lines(extended_facts(problem))


# --- m5: gpu-mg refused springs in three places (P2b lifted it) ---------------------------------------------------------


def gpu_case(tmp_path):
    return case(tmp_path, solver="gpu-mg")


def test_check_does_not_warn_that_gpu_mg_cannot_take_springs(tmp_path, fake_cpu_device):
    d = cli.check_data(gpu_case(tmp_path))
    assert d["solver"]["resolved"] == "gpu-mg"
    assert not [w for w in d["warnings"] if "cannot start here" in w or "spring term" in w]


def test_check_has_no_such_warning_for_other_solvers(tmp_path):
    d = cli.check_data(case(tmp_path, solver="amg"))
    assert not [w for w in d["warnings"] if "spring term" in w]


def test_run_does_not_stop_at_the_start_for_springs_and_gpu_mg(tmp_path, fake_cpu_device, capsys, monkeypatch):
    """The run gets as far as the solver: a stand-in for `GpuSolver.__init__` (the fake device has no OpenCL behind it) says that it was called. Any
    refusal before it (`check`, `cmd_run`, the old refusal of the springs) would stop the run with its own text and not with this one (review R1, N3)."""
    from topopt.gpu_mg.model import GpuSolver

    def reached(self, *args, **kwargs):
        raise RuntimeError("the solver of gpu-mg was made")
    monkeypatch.setattr(GpuSolver, "__init__", reached)
    out = tmp_path / "out"
    assert main(run_args(gpu_case(tmp_path), out)) == 2
    err = capsys.readouterr().err
    assert "the solver of gpu-mg was made" in err and "spring term" not in err and "cannot start here" not in err
