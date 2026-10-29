"""A load case that feels no load for a reason other than its YAML numbers (F4-37; `loads.reaches_free_dofs`, `loads.zero_compliance_refusal`,
CONTRACTS.md section 13).

U21 refuses a case with no load only where its YAML numbers are all 0.0. Two more reasons give the same answer, and each test is an exact 0.0
(no tolerance, no new number):

- `lost`: the load vector of the case is exactly 0.0 on every free dof (all its force lands on fixed dofs). `problem.build_problem` knows it, so
  `topopt check`, `check_problem`, `topopt run` (before pass 0), `--resume`, `--continue-from` and the part builder stop or state it.
- `zero`: the load reaches a free dof, but the compliance of the case is exactly 0.0 at run time (a force of `[0, 0, 1e-300]`: c = f.u underflows).
  `optimize.run` stops at the first pass with a user error (exit 1), where the log form, the max form and the buckling row gave an internal error
  (exit 2) or, for the buckling check, an error record.

A case that loses only a part of its force keeps the lost-force fact and is not refused.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from test_lost_force import touching_case
from test_optimize import box
from test_repeated_load_cases import SPEC, write_case
from topopt import loads
from topopt.cli import check_data, derived_numbers, main
from topopt.config import UserError, load_config
from topopt.optimize import reduced_system
from topopt.problem import build_problem
from topopt.regions_builder import build_spec, check_yaml

ZERO = {"name": "zero", "forces": [{"region": "tip", "force": [0, 0, 0]}]}
DOWN = {"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}
TINY = {"name": "tiny", "forces": [{"region": "tip", "force": [0, 0, 1e-300]}]}
LOST = {"name": "lost", "forces": [{"region": "blk", "force": [0, 0, -2.0]}]}
MASS = {"objective": "mass"}
LOST_WORDS = "every force of the case lands on fixed dofs, so the solver drops all of it and no free dof gets a load"
ZERO_WORDS = "the compliance of the case is 0.0 or subnormal in double precision, so its load does no measurable work on the part"
NODAL_WORDS = ("the nodal forces of the case add up to 0.0 at every node (opposite forces on the same nodes cancel, or a force is so small that its "
               "nodal share is 0.0 in double precision)")
NET_WORDS = ("every nodal force of the case lands on fixed dofs, so the solver drops all of it and no free dof gets a load; the forces cancel in the "
             "net sum that the `lost_force` fact measures, so that fact does not say 100 percent")
OPTIONS_NODAL = "Options: put opposite forces of the case on different load regions, or give the case a larger force"
OPTIONS_LOST = "Options: move the load region of 'lost' off the fixed regions, so that its force reaches a free dof"
OPTIONS_TINY = "Options: give the case a larger force, moment or pressure"
CLIMIT = "has a compliance_limit, and the row needs a compliance above 0 (the run would stop at pass 0)"
CLIMIT_RUN = "has a compliance_limit, and the row needs a compliance above 0 (the run stops at the first pass where its compliance is 0.0 or subnormal)"
MAX = "settings.case_combine is max, and the max objective needs a compliance above 0 for every case"
ROW = "buckling.min_factor needs a load in every case (the eigen solve of a case with no load has no start vector)"
CHECK = "buckling.check needs a load in every case (the eigen solve of a case with no load has no start vector)"
EVERY = "no case applies a load, and the objective is compliance: it is 0 for every design, so nothing in the YAML shapes the part"


def with_(case, **keys):
    return dict(case, **keys)


def write_lost(tmp_path, cases, settings=None, fixed_block=None, squeeze=False, **top):
    """The cantilever of `write_case` with a load region `blk`, one voxel between the plate and a fixed block: all 8 nodes of the voxel are
    fixed, so a force on `blk` is lost for 100 percent. `fixed_block` replaces the entry of the block in `regions.fixed` (a roller).
    `squeeze` adds a second load region `blk2` of the same kind next to `blk` (two opposite forces on fixed nodes: the net force is 0)."""
    p = write_case(tmp_path, cases, {"iterations": 4, **(settings or {})}, **top)
    box(tmp_path / "block.stl", (2, 2, 0), (4, 8, 4))
    box(tmp_path / "blk.stl", (0, 4, 0), (2, 6, 2))
    d = yaml.safe_load(p.read_text())
    d["regions"]["fixed"].append(fixed_block or "block.stl")
    d["regions"]["load"]["blk"] = "blk.stl"
    if squeeze:
        box(tmp_path / "blk2.stl", (0, 2, 0), (2, 4, 2))
        d["regions"]["load"]["blk2"] = "blk2.stl"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def write_tiny(tmp_path, cases, settings=None, **top):
    return write_case(tmp_path, cases, {"iterations": 4, **(settings or {})}, **top)


def run_text(path, out, capsys, *, resume=False):
    """`topopt run` ends with exit 1 and one `error:` line, no traceback."""
    assert main(["run", str(path), "--out", str(out), "--quiet", *(["--resume"] if resume else [])]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1 and "Traceback" not in err and "internal error" not in err
    return err


def history(out):
    keep = lambda d: {k: v for k, v in d.items() if not any(t in k for t in ("time", "elapsed", "t_iter_s"))}      # noqa: E731
    return [keep(json.loads(ln)) for ln in (Path(out) / "history.jsonl").read_text().splitlines()]


# --- the test: the load vector on the free dofs -----------------------------------------------------------------------------


def test_the_load_vector_on_the_free_dofs_is_tested_exactly_per_dof():
    forces = np.zeros((4, 12))
    forces[0, [0, 1, 2]] = [1.0, 0.0, -1.0]        # all on the fixed dofs
    forces[1, [1, 7]] = [5e-324, 1e-300]           # a tiny number on a fixed dof, another on a free dof: the free dof counts, however small
    forces[2, [5, 11]] = [3.0, 0.5]                # a part on a fixed dof, the rest on a free dof: it reaches (no share, no threshold)
    fixed = np.array([0, 1, 2, 3, 4, 5])
    assert loads.reaches_free_dofs(forces, fixed) == [False, True, True, False]       # row 3 is all 0.0
    assert loads.reaches_free_dofs(forces, np.array([], dtype=np.int64)) == [True, True, True, False]


def test_a_force_on_the_nodes_between_two_fixed_regions_reaches_no_free_dof(tmp_path):
    prob = build_problem(load_config(write_lost(tmp_path, [LOST, DOWN])))
    assert prob.load_reaches_free_dofs == (False, True) and prob.load_reach == ("fixed", "free")
    free = reduced_system(prob)[2]                                             # the free dofs as the solver takes them
    assert not np.any(prob.forces[0][free])                                    # the reduced vector: exactly 0.0 on every free dof
    assert np.any(prob.forces[0])                                              # the full vector is not 0: the force sits on fixed dofs
    assert np.any(prob.forces[1][free])


def test_a_roller_loses_the_force_along_its_axis_and_not_along_the_free_axis(tmp_path):
    """The same nodes: a block that holds z only loses a z force and passes an x force (the test reads dofs, not nodes)."""
    roller = {"file": "block.stl", "axes": ["z"]}
    z = build_problem(load_config(write_lost(tmp_path / "z", [LOST, DOWN], fixed_block=roller)))
    along_x = with_(LOST, forces=[{"region": "blk", "force": [-2.0, 0, 0]}])
    x = build_problem(load_config(write_lost(tmp_path / "x", [along_x, DOWN], fixed_block=roller)))
    assert z.load_reaches_free_dofs == (False, True) and x.load_reaches_free_dofs == (True, True)


def test_a_case_that_applies_no_load_in_the_yaml_reaches_no_free_dof_either(tmp_path):
    assert build_problem(load_config(write_case(tmp_path, [ZERO, DOWN]))).load_reaches_free_dofs == (False, True)


# --- (b) a force that lands on fixed nodes only: the U21 answer ------------------------------------------------------------

REFUSED = [
    # (id, cases, settings, top, text of the reason)
    ("compliance-limit", [with_(LOST, compliance_limit=50.0), DOWN], {}, {}, f"'lost' {CLIMIT}"),
    ("compliance-limit-mass", [with_(LOST, compliance_limit=50.0)], MASS, {}, f"'lost' {CLIMIT}"),
    ("max-combine", [LOST, DOWN], {"case_combine": "max"}, {}, MAX),
    ("buckling-row", [LOST, DOWN], {}, {"buckling": {"min_factor": 5.0}}, ROW),
    ("buckling-check", [LOST, DOWN], {}, {"buckling": {"check": True}}, CHECK),
    ("all-lost-compliance", [LOST], {}, {}, EVERY),
]


@pytest.mark.parametrize("cases, settings, top, reason", [r[1:] for r in REFUSED], ids=[r[0] for r in REFUSED])
def test_a_case_that_loses_all_its_force_next_to_a_row_that_cannot_work_is_refused_with_options(tmp_path, capsys, monkeypatch, cases, settings, top, reason):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    p = write_lost(tmp_path, cases, settings, **top)
    load_config(p)                                                  # the YAML numbers are a load: the loader does not stop it
    with pytest.raises(UserError) as e:
        check_data(p)
    msg = str(e.value)
    assert f"load case 'lost' applies no load ({LOST_WORDS}; the `lost_force` fact says 100 percent), and the run cannot work: " in msg
    assert reason in msg and "\n" not in msg and msg.endswith(f"(file: {p.resolve()})")
    assert OPTIONS_LOST in msg
    assert main(["check", str(p)]) == 1                             # exit 1, as U21 gives for the same refusal
    assert reason in capsys.readouterr().err
    out = tmp_path / "run"
    assert reason in run_text(p, out, capsys)
    assert not (out / "history.jsonl").exists()                     # before pass 0
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "error" and status["iteration"] == 0 and reason in status["message"]
    assert json.loads((out / "job.json").read_text())["state"] == "failed" and (out / "job.FAILED").exists()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    res = server.check_problem(str(p))
    assert res["ok"] is False and reason in res["error"] and OPTIONS_LOST in res["error"]


def test_each_option_of_the_refusal_works(tmp_path):
    """Move the load region (here: another region), remove the case, remove the row, change the combine: each gives a YAML that builds."""
    limited = with_(LOST, compliance_limit=50.0)
    build_problem(load_config(write_lost(tmp_path / "a", [with_(limited, forces=[{"region": "tip", "force": [0, 0, -2.0]}]), DOWN])))   # a region that is free
    build_problem(load_config(write_lost(tmp_path / "b", [DOWN])))                                                                 # remove the case
    build_problem(load_config(write_lost(tmp_path / "c", [LOST, DOWN])))                                                           # remove the compliance_limit
    build_problem(load_config(write_lost(tmp_path / "d", [LOST, DOWN], {"case_combine": "sum"})))                                  # case_combine: sum
    build_problem(load_config(write_lost(tmp_path / "e", [DOWN], buckling={"min_factor": 5.0})))                                   # the case removed, the row stays


def test_a_lost_case_and_a_yaml_zero_case_are_named_once_each_with_their_own_words(tmp_path):
    """The loader passes the YAML zero case (no row needs it), `build_problem` refuses with both: the zero case is not reported twice."""
    p = write_lost(tmp_path, [ZERO, with_(LOST, compliance_limit=50.0), DOWN])
    load_config(p)
    with pytest.raises(UserError) as e:
        build_problem(load_config(p))
    msg = str(e.value)
    assert msg.startswith("load cases 'zero' and 'lost' apply no load ('zero': every force and moment is exactly 0 and no pressure is other than 0; "
                          f"'lost': {LOST_WORDS}")
    assert f"'lost' {CLIMIT}" in msg and "'zero' has" not in msg and "'zero' have" not in msg    # no row needs 'zero': only the head and the option name it
    assert "Options: give 'zero' a force, a moment or a pressure; move the load region of 'lost' off the fixed regions" in msg


def test_a_lost_case_with_no_row_that_needs_a_load_gets_the_fact_and_a_normal_run(tmp_path, capsys, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    p = write_lost(tmp_path / "p", [LOST, DOWN])
    d = check_data(p)
    facts = [w for w in d["warnings"] if "applies no load" in w]
    assert len(facts) == 1 and facts[0].startswith("fact: load case 'lost' applies no load: " + LOST_WORDS)
    assert "options: move the load region of 'lost' off the fixed regions, so that its force reaches a free dof; or remove the case." in facts[0]
    assert [(c["name"], c["applies_load"], c["load_reaches_free_dofs"]) for c in d["derived"]["load_cases"]] == [("lost", True, False), ("down", True, True)]
    assert any("force is lost" in w for w in d["warnings"])                  # the lost-force warning stays
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.count("fact: load case 'lost' applies no load") == 1 and "lost force:" in out
    assert main(["check", str(p), "--json"]) == 0
    assert [f for f in json.loads(capsys.readouterr().out)["facts"] if "applies no load" in f] == facts
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    res = server.check_problem(str(p))
    assert res["ok"] is True and [w for w in res["warnings"] if "applies no load" in w] == facts
    run, alone = tmp_path / "r", tmp_path / "a"
    assert main(["run", str(p), "--out", str(run), "--quiet"]) == 0
    assert main(["run", str(write_lost(tmp_path / "q", [DOWN])), "--out", str(alone), "--quiet"]) == 0
    log = (run / "run.log").read_text()
    assert log.count("fact: load case 'lost' applies no load") == 1
    assert log.index("fact: load case 'lost' applies no load") < re.search(r"^it +0 ", log, re.M).start()
    first, base = history(run)[0], history(alone)[0]
    assert first["compliance_cases"][0] == 0.0 and first["compliance"] == pytest.approx(base["compliance"], rel=1e-9)     # the case adds nothing
    derived = json.loads((run / "run.json").read_text())["derived"]["load_cases"]
    assert [(c["applies_load"], c["load_reaches_free_dofs"]) for c in derived] == [(True, False), (True, True)]


def test_a_lost_case_under_the_mass_objective_with_a_stress_limit_is_a_fact_and_the_run_goes_on(tmp_path):
    p = write_lost(tmp_path / "p", [with_(LOST, stress_limit=5.0)], MASS)
    d = check_data(p)
    [fact] = [w for w in d["warnings"] if "applies no load" in w]
    assert "The stress limit ('lost' 5 MPa) is met by any design." in fact and "the mass objective gives the lightest design" in fact
    assert main(["run", str(p), "--out", str(tmp_path / "o"), "--quiet"]) == 0


def test_the_part_builder_says_the_fact_or_the_error_for_a_lost_case(tmp_path):
    lines = []
    say = lambda *a: lines.append(" ".join(map(str, a)))      # noqa: E731
    ok, _ = check_yaml(write_lost(tmp_path / "a", [LOST, DOWN]), say=say)
    assert ok and len([ln for ln in lines if "applies no load" in ln]) == 1
    lines.clear()
    ok, _ = check_yaml(write_lost(tmp_path / "b", [with_(LOST, compliance_limit=50.0), DOWN]), say=say)
    assert not ok and any(ln.startswith("ERROR: load case 'lost' applies no load") for ln in lines)


def test_resume_and_continue_of_a_folder_made_with_the_same_lost_yaml_stop_with_the_refusal(tmp_path, capsys, monkeypatch):
    """Every path that builds the problem. The folder is made with the SAME YAML (a lost case and `buckling.check`; the code before F4-37 ran it to the end with
    an error record, exit 0), so the old "the inputs changed" stop cannot be the one that this test sees: only the guard in `build_problem` can give the text."""
    from topopt import problem
    p = write_lost(tmp_path / "p", [LOST, DOWN], {"iterations": 3}, buckling={"check": True})
    first = tmp_path / "first"
    with monkeypatch.context() as m:
        m.setattr(problem, "no_load_refusal", lambda *a: None, raising=False)   # the code before F4-37: no guard in build_problem
        assert main(["run", str(p), "--out", str(first), "--quiet"]) == 0
    capsys.readouterr()
    before = sorted(x.name for x in first.iterdir())
    assert main(["run", str(p), "--out", str(first), "--quiet", "--resume"]) == 1
    err = capsys.readouterr().err
    assert f"'lost' applies no load ({LOST_WORDS}" in err and CHECK in err and "inputs changed" not in err
    assert err.count("\n") == 1 and sorted(x.name for x in first.iterdir()) == before
    assert main(["run", str(p), "--out", str(tmp_path / "next"), "--quiet", "--continue-from", str(first), "--more-iterations", "2"]) == 1
    err = capsys.readouterr().err
    assert f"'lost' applies no load ({LOST_WORDS}" in err and CHECK in err and err.count("\n") == 1 and not (tmp_path / "next").exists()


# --- the words follow the vector (review m1): forces that cancel, a squeeze and a moment on fixed nodes -----------------------------

CANCEL = {"name": "cancel", "forces": [{"region": "tip", "force": [0, 0, -2.0]}, {"region": "tip", "force": [0, 0, 2.0]}]}
SQUEEZE = {"name": "squeeze", "forces": [{"region": "blk", "force": [2.0, 0, 0]}, {"region": "blk2", "force": [-2.0, 0, 0]}]}
TWIST = {"name": "twist", "forces": [{"region": "blk", "moment": [0, 0, 5.0]}]}


def test_opposite_forces_on_a_free_region_get_the_cancel_words_and_no_fixed_option(tmp_path, capsys):
    """The two forces are on the free region `tip`: no node is fixed, but the nodal vector is 0.0 everywhere. The words must not say "fixed"."""
    p = write_tiny(tmp_path / "p", [with_(CANCEL, compliance_limit=50.0), DOWN])
    load_config(p)                                                  # the YAML numbers are a load: the loader does not stop it
    with pytest.raises(UserError) as e:
        check_data(p)
    msg = str(e.value)
    assert f"load case 'cancel' applies no load ({NODAL_WORDS}), and the run cannot work: 'cancel' {CLIMIT}. {OPTIONS_NODAL}; remove the case; " in msg
    assert "fixed" not in msg and "100 percent" not in msg and "lost_force" not in msg
    assert main(["check", str(p)]) == 1
    assert NODAL_WORDS in capsys.readouterr().err
    out = tmp_path / "run"
    err = run_text(p, out, capsys)
    assert NODAL_WORDS in err and "fixed" not in err and not (out / "history.jsonl").exists()
    # with no row that needs a load: the fact, with the same words, and a normal run
    q = write_tiny(tmp_path / "q", [CANCEL, DOWN])
    d = check_data(q)
    [fact] = [w for w in d["warnings"] if "applies no load" in w]
    assert fact.startswith(f"fact: load case 'cancel' applies no load: {NODAL_WORDS}. ")
    assert "options: put opposite forces of the case on different load regions, or give the case a larger force; or remove the case." in fact
    assert "fixed" not in fact and "100 percent" not in fact
    assert [(c["load_reach"], c["load_reaches_free_dofs"]) for c in d["derived"]["load_cases"]] == [("nodal_zero", False), ("free", True)]
    assert main(["check", str(q)]) == 0
    assert capsys.readouterr().out.count("fact: load case 'cancel' applies no load") == 1
    assert main(["run", str(q), "--out", str(tmp_path / "r"), "--quiet"]) == 0
    assert (tmp_path / "r" / "run.log").read_text().count(f"fact: load case 'cancel' applies no load: {NODAL_WORDS}") == 1


def test_a_squeeze_on_fixed_nodes_does_not_say_100_percent(tmp_path, capsys):
    """Two opposite forces on two load regions between the plate and a fixed block: every nodal force is on a fixed dof, but the net force is 0, so
    the `lost_force` fact says 0 percent. The words name that and not "100 percent"."""
    q = write_lost(tmp_path / "q", [SQUEEZE, DOWN], squeeze=True)
    d = check_data(q)
    assert d["lost_force"]["cases"][0]["lost_percent"] == 0.0 and not [w for w in d["warnings"] if "force is lost" in w]
    [fact] = [w for w in d["warnings"] if "applies no load" in w]
    assert fact.startswith(f"fact: load case 'squeeze' applies no load: {NET_WORDS}. ")
    assert "says 100 percent" not in fact and "options: move the load region of 'squeeze' off the fixed regions" in fact
    assert [c["load_reach"] for c in d["derived"]["load_cases"]] == ["fixed_net", "free"]
    p = write_lost(tmp_path / "p", [with_(SQUEEZE, compliance_limit=50.0), DOWN], squeeze=True)
    with pytest.raises(UserError) as e:
        check_data(p)
    msg = str(e.value)
    assert f"load case 'squeeze' applies no load ({NET_WORDS}), and the run cannot work: 'squeeze' {CLIMIT}. " in msg
    assert "says 100 percent" not in msg and "Options: move the load region of 'squeeze' off the fixed regions" in msg
    assert main(["check", str(p)]) == 1 and NET_WORDS in capsys.readouterr().err


def test_a_moment_on_fixed_nodes_says_100_percent_because_the_fact_does(tmp_path):
    """A moment alone has no resultant force, so `lost_percent` is 0.0, and the fact is the lost moment: 100 percent. The `fixed` words stay true."""
    d = check_data(write_lost(tmp_path / "q", [TWIST, DOWN]))
    assert d["lost_force"]["cases"][0]["lost_moment_percent"] == pytest.approx(100.0)
    [fact] = [w for w in d["warnings"] if "applies no load" in w]
    assert fact.startswith(f"fact: load case 'twist' applies no load: {LOST_WORDS}; the `lost_force` fact says 100 percent. ")
    assert [c["load_reach"] for c in d["derived"]["load_cases"]] == ["fixed", "free"]


def test_the_kinds_of_a_case_that_no_free_dof_feels_come_from_the_vectors_and_the_fact(tmp_path):
    forces = np.zeros((4, 12))
    forces[1, [0, 7]] = [1.0, 0.0]                  # on a fixed dof only
    forces[2, [1, 2]] = [1.0, -1.0]                 # on fixed dofs, and the fact says 0 percent (the forces cancel)
    forces[3, 8] = 1e-300                           # on a free dof
    fixed = np.array([0, 1, 2, 3, 4, 5])
    calls = []

    def lost():
        calls.append(1)
        return {"cases": [{"lost_percent": 0.0}, {"lost_percent": 100.0}, {"lost_percent": 0.0}, {"lost_percent": 0.0}]}

    assert loads.reach_kinds(forces, fixed, lost) == ["nodal_zero", "fixed", "fixed_net", "free"]
    assert len(calls) == 1                           # asked once, and only because a case has a force on fixed dofs and none on a free dof
    calls.clear()
    assert loads.reach_kinds(forces[[0, 3]], fixed, lost) == ["nodal_zero", "free"] and not calls


def test_a_lost_case_and_a_cancelled_case_have_a_null_stiffness_with_a_reason_that_names_the_fact(tmp_path):
    """Review n1: `stiffness_cases[].N_per_mm` is null for a case whose load reaches no free dof, and the entry says why (a normal case gets a number)."""
    out = tmp_path / "o"
    assert main(["run", str(write_lost(tmp_path / "p", [LOST, CANCEL, DOWN])), "--out", str(out), "--quiet"]) == 0
    assert json.loads((out / "run.json").read_text())["derived"]["load_cases"][0]["load_reaches_free_dofs"] is False
    stiff = json.loads((out / "result" / "metrics.json").read_text())["stiffness_cases"]
    assert [s["name"] for s in stiff] == ["lost", "cancel", "down"] and [s["N_per_mm"] is None for s in stiff] == [True, True, False]
    reason = "the load of the case reaches no free dof (load_reaches_free_dofs is false), so its compliance is 0 and |F|^2 / c is not defined"
    assert stiff[0]["reason"] == reason and stiff[1]["reason"] == reason and "reason" not in stiff[2]


def test_the_stiffness_reason_for_a_case_with_a_load_on_a_free_dof_and_for_an_old_run_json_is_unchanged():
    from topopt.post import case_stiffness
    force = [{"region": "tip", "force": [0.0, 0.0, -2.0], "kind": "force"}]
    cases = [{"name": "a", "forces": force}, {"name": "b", "forces": force}]
    assert [o.get("reason") for o in case_stiffness(cases, [0.0, 0.0])] == [None, None]                            # an old run.json has no key: no reason
    assert [o.get("reason") for o in case_stiffness(cases, [0.0, 0.0], reaches=[True, None])] == [None, None]      # true or unknown: no reason
    assert case_stiffness(cases, [0.0, 0.5], reaches=[False, True])[1]["N_per_mm"] == 8.0
    assert "reaches no free dof" in case_stiffness(cases, [0.0, 0.5], reaches=[False, True])[0]["reason"]


# --- a partial loss keeps today's behaviour ---------------------------------------------------------------------------------


def test_a_partial_loss_keeps_the_lost_force_fact_and_is_not_refused(tmp_path, capsys):
    """(A guard: it passes before F4-37 too.) A load box on the plate face loses a quarter of its force (the case of test_lost_force). A `compliance_limit` is a row that a case with no load
    refuses, but this case has a load on free dofs: the check says the lost force, exit 0, and the run starts."""
    p = touching_case(tmp_path)
    d0 = yaml.safe_load(p.read_text())
    d0["load_cases"][0]["compliance_limit"] = 500.0
    d0["settings"].update(iterations=3, solver="amg")
    p.write_text(yaml.safe_dump(d0))
    d = check_data(p)
    assert d["lost_force"]["cases"][0]["lost_percent"] == pytest.approx(25.0)
    assert [w for w in d["warnings"] if "force is lost" in w] and not [w for w in d["warnings"] if "applies no load" in w]
    assert main(["check", str(p)]) == 0
    assert "lost force:" in capsys.readouterr().out
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert len(history(out)) == 4 and "applies no load" not in (out / "run.log").read_text()
    assert "warning: lost force" in (out / "run.log").read_text()


def test_a_roller_that_loses_some_nodes_is_a_partial_loss(tmp_path):
    p = write_lost(tmp_path, [with_(LOST, forces=[{"region": "blk", "force": [-2.0, 0, 0]}], compliance_limit=50.0), DOWN],
                   fixed_block={"file": "block.stl", "axes": ["z"]})
    d = check_data(p)                                      # the x=0 face is lost (plate), the x=2 face is free along x: half is lost
    assert d["lost_force"]["cases"][0]["lost_percent"] == pytest.approx(50.0)
    assert [c["load_reaches_free_dofs"] for c in d["derived"]["load_cases"]] == [True, True] and not [w for w in d["warnings"] if "applies no load" in w]


# --- (a) a compliance of exactly 0.0 at run time -----------------------------------------------------------------------------

RUNTIME = [
    ("compliance-limit", [with_(TINY, compliance_limit=50.0), DOWN], {}, {}, f"'tiny' {CLIMIT_RUN}"),
    ("max-combine", [TINY, DOWN], {"case_combine": "max"}, {}, MAX),
    ("buckling-row", [TINY, DOWN], {}, {"buckling": {"min_factor": 5.0}}, ROW),
    ("buckling-check", [TINY, DOWN], {}, {"buckling": {"check": True}}, CHECK),
    ("all-tiny-compliance", [TINY], {}, {}, EVERY),
]


@pytest.mark.parametrize("cases, settings, top, reason", [r[1:] for r in RUNTIME], ids=[r[0] for r in RUNTIME])
def test_a_compliance_of_0_0_next_to_a_row_that_cannot_work_stops_the_run_with_a_user_error(tmp_path, capsys, monkeypatch, cases, settings, top, reason):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    p = write_tiny(tmp_path, cases, settings, **top)
    load_config(p)
    check_data(p)                                                   # the check cannot know: the load is a load, and it reaches a free dof
    out = tmp_path / "run"
    err = run_text(p, out, capsys)                                  # exit 1, not 2
    assert f"load case 'tiny' applies no load ({ZERO_WORDS})" in err and reason in err and OPTIONS_TINY in err
    assert "pass 0" not in err                                      # review m4: the refusal comes at the first pass where it holds, not always at pass 0
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "error" and status["iteration"] == 0 and status["message"] == err[len("error: "):-1]
    job = json.loads((out / "job.json").read_text())
    assert job["state"] == "failed" and job["exit_code"] == 1 and (out / "job.FAILED").exists()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    st = server.run_status(str(out))
    assert st["state"] == "error" and st["job"]["state"] == "failed" and ZERO_WORDS in st["summary"] and reason in st["summary"]


def test_a_compliance_of_0_0_with_no_row_that_needs_it_runs_as_before(tmp_path):
    out = tmp_path / "o"
    assert main(["run", str(write_tiny(tmp_path / "p", [TINY, DOWN])), "--out", str(out), "--quiet"]) == 0
    assert history(out)[0]["compliance_cases"][0] == 0.0 and "applies no load" not in (out / "run.log").read_text()


def test_the_runtime_refusal_answers_only_for_a_case_with_a_load_on_a_free_dof(tmp_path):
    cfg = load_config(write_lost(tmp_path, [ZERO, with_(LOST, compliance_limit=50.0), with_(TINY, compliance_limit=50.0), DOWN]))
    reach = ("nodal_zero", "fixed", "free", "free")
    ask = lambda c: loads.zero_compliance_refusal(cfg, c, reach)      # noqa: E731
    assert ask([0.0, 0.0, 1.0, 1.0]) is None                           # zero and lost: the loader and build_problem answered
    text = ask([0.0, 0.0, 0.0, 1.0])
    assert text.startswith("load cases 'zero', 'lost' and 'tiny' apply no load ('zero': every force and moment is exactly 0") and f"'tiny': {ZERO_WORDS}" in text
    assert ask([1.0, 1.0, 1e-300, 1.0]) is None                        # 1e-300 is a normal double: no refusal


# --- a compliance that is subnormal: the two old windows (review m2, m3) ----------------------------------------------------------

def tiny_force(x):
    return with_(TINY, forces=[{"region": "tip", "force": [0, 0, x]}])


@pytest.mark.parametrize("x, settings, top, reason", [
    (3e-161, {}, {}, f"'tiny' {CLIMIT_RUN}"),                           # the log form: c is about 1e-322, and `d / value` gave "MMA got a value or gradient that is not finite"
    (8e-161, {}, {}, f"'tiny' {CLIMIT_RUN}"),                           # the same window, at its upper end (the base stopped at pass 1)
    (1e-160, {}, {"buckling": {"min_factor": 5.0}}, ROW),               # the buckling row: c is 2.57e-322, and ARPACK gave error -9 (the base stopped at pass 1)
    (3e-161, {"case_combine": "max"}, {}, MAX),
], ids=["log-form-3e-161", "log-form-8e-161", "buckling-row-1e-160", "max-3e-161"])
def test_a_subnormal_compliance_next_to_a_row_that_cannot_work_stops_with_a_user_error_and_not_an_internal_error(tmp_path, capsys, x, settings, top, reason):
    """The compliance is not 0.0 here, but it is subnormal (below `np.finfo(float).tiny`): it has lost its precision, so the row cannot work with it."""
    limited = {"compliance_limit": 50.0} if not settings and not top else {}
    p = write_tiny(tmp_path, [with_(tiny_force(x), **limited), DOWN], settings, **top)
    err = run_text(p, tmp_path / "run", capsys)                       # exit 1, not 2
    assert f"load case 'tiny' applies no load ({ZERO_WORDS})" in err and reason in err and OPTIONS_TINY in err
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["state"] == "error" and status["message"] == err[len("error: "):-1]


def test_a_compliance_that_becomes_subnormal_at_a_later_pass_stops_at_that_pass_with_words_that_do_not_say_pass_0(tmp_path, capsys):
    """1e-153 N: the compliance of the case is 2.4e-308 at pass 0 (normal) and falls below `np.finfo(float).tiny` (2.2e-308) when the design gets stiffer."""
    p = write_tiny(tmp_path, [with_(tiny_force(1e-153), compliance_limit=50.0), DOWN])
    err = run_text(p, tmp_path / "run", capsys)
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["iteration"] > 0 and "pass 0" not in err and f"'tiny' {CLIMIT_RUN}" in err
    assert history(tmp_path / "run")[0]["compliance_cases"][0] > np.finfo(float).tiny      # pass 0 was normal and was run


def test_a_force_whose_compliance_is_a_normal_double_runs_as_before(tmp_path):
    """1e-150 N gives a compliance of about 2.4e-302: small, but normal. The refusal is for 0.0 and subnormal values only."""
    p = write_tiny(tmp_path / "p", [with_(tiny_force(1e-150), compliance_limit=50.0), DOWN], {"iterations": 2})
    out = tmp_path / "o"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    c = history(out)[0]["compliance_cases"][0]
    assert np.finfo(float).tiny < c < 1e-300


def test_the_refusal_threshold_is_the_smallest_normal_double(tmp_path):
    """0.0 and every subnormal value of either sign is refused for a case with a load; the smallest normal double is not; nan is left to the old error."""
    cfg = load_config(write_tiny(tmp_path, [with_(TINY, compliance_limit=50.0), DOWN]))
    tiny = np.finfo(float).tiny
    ask = lambda c: loads.zero_compliance_refusal(cfg, [c, 1.0], ("free", "free"))      # noqa: E731
    for c in (0.0, -0.0, 5e-324, tiny / 2, np.nextafter(tiny, 0.0), -tiny / 2):
        assert ask(c) is not None and ZERO_WORDS in ask(c), c
    for c in (tiny, np.nextafter(tiny, 1.0), 1e-300, 0.5, float("nan")):
        assert ask(c) is None, c


# --- the guard: a normal run does not change --------------------------------------------------------------------------------


def test_a_normal_run_has_no_new_fact_and_the_history_of_the_code_before_F4_37(tmp_path):
    """One small case in which every case has a load on a free dof, run on this tree and compared with the history that `f367b2d` (U21, before F4-37) wrote
    for it: `tests/data/root_guard_history_f367b2d.json` holds, for each pass, the objective, the compliance of each case, the values of the rows, the volume
    fraction and the grayness. A change anywhere in the path of a normal run (a wrong refusal, a dropped case, a changed objective) makes the numbers differ.
    The tolerance (relative 1e-6) is only for the rounding of another BLAS or another solver: a changed case moves these numbers by far more."""
    rec = json.loads((Path(__file__).parent / "data" / "root_guard_history_f367b2d.json").read_text())
    assert rec["recorded_on"].startswith("f367b2d") and len(rec["rows"]) == 5
    p = write_case(tmp_path / "p", [with_(DOWN, compliance_limit=50.0), {"name": "side", "forces": [{"region": "side", "force": [0, 0.3, -0.2]}]}], {"iterations": 4})
    assert not [w for w in check_data(p)["warnings"] if "applies no load" in w]
    out = tmp_path / "on"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    rows = [json.loads(ln) for ln in (out / "history.jsonl").read_text().splitlines()]
    assert len(rows) == len(rec["rows"])
    for got, want in zip(rows, rec["rows"]):
        assert got["iteration"] == want["iteration"]
        assert got["objective"]["value"] == pytest.approx(want["objective"], rel=1e-6)
        assert got["compliance"] == pytest.approx(want["compliance"], rel=1e-6)
        assert got["compliance_cases"] == pytest.approx(want["compliance_cases"], rel=1e-6)
        assert [c["name"] for c in got["constraints"]] == list(want["constraints"])
        for c in got["constraints"]:
            assert [c["value"], c["g"]] == pytest.approx(want["constraints"][c["name"]], rel=1e-6)
        assert got["volume_fraction"] == pytest.approx(want["volume_fraction"], rel=1e-6)
        assert got["grayness"] == pytest.approx(want["grayness"], rel=1e-6)
    log = (out / "run.log").read_text()
    assert "applies no load" not in log and "no free dof" not in log


def test_derived_has_the_key_for_every_case_true_for_a_normal_case_and_null_where_there_is_no_grid(tmp_path):
    cfg = load_config(write_case(tmp_path, [ZERO, DOWN]))
    assert [c["load_reaches_free_dofs"] for c in derived_numbers(cfg)["load_cases"]] == [None, None]
    assert [c["applies_load"] for c in derived_numbers(cfg)["load_cases"]] == [False, True]
    normal = write_case(tmp_path / "n", [DOWN, {"name": "side", "forces": [{"region": "side", "force": [0, 0.3, -0.2]}]}], {"iterations": 2})
    assert [c["load_reaches_free_dofs"] for c in check_data(normal)["derived"]["load_cases"]] == [True, True]
    out = tmp_path / "o"
    assert main(["run", str(normal), "--out", str(out), "--quiet"]) == 0
    assert [c["load_reaches_free_dofs"] for c in json.loads((out / "run.json").read_text())["derived"]["load_cases"]] == [True, True]
