"""End-to-end tests of the ratchet on real short runs (CONTRACTS.md section 36): T4 the hook, T5 the ladle, T6 the marks seam.

Each test makes a short run with `topopt run` (the hook in about 15 s, the ladle in about 10 s), then runs the ratchet with the
real judge (`printcheck`, and `verify` in the process). T4 and T5 take their candidates from the test-only box-cut source
(`tests/ratchet_helpers.py`); T6 uses the marks source (`apply_marks`), whose sewn output `topopt verify` can mesh since the C3a fix.
They need gmsh, scikit-fem and one linear solver.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import trimesh
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import needs_gmsh, needs_skfem  # noqa: E402
from ratchet_helpers import BoxCutSource, far_vertex  # noqa: E402
from test_feedback import mark, marks_doc, tree_hash  # noqa: E402

from topopt import ratchet as rt  # noqa: E402
from topopt.cli import main as topopt_main  # noqa: E402
from topopt.config import load_config  # noqa: E402

EXAMPLES = REPO / "examples"
pytestmark = [needs_gmsh, needs_skfem, pytest.mark.slow]       # the runs take 10 to 100 s (deselect with -m "not slow")


@pytest.fixture(autouse=True, scope="module")
def _need_a_solver():
    from topopt.verify.factor import factor_availability
    if not any(v["available"] for v in factor_availability().values()):
        pytest.skip("no linear solver can run here")


def quiet(*a, **k):
    pass


def absolute_yaml(src: Path, dst: Path, acceptance=None, settings=None, case_edits=None) -> Path:
    """A copy of an example YAML: the region paths made absolute, the settings and load cases edited, an `acceptance` block."""
    doc = yaml.safe_load(src.read_text())

    def ab(v):
        if isinstance(v, list):
            return [ab(x) for x in v]
        if isinstance(v, dict):
            return {k: ab(x) for k, x in v.items()}
        return str(src.parent / v)
    doc["regions"] = ab(doc["regions"])
    doc.setdefault("settings", {}).update(settings or {})
    for case in doc["load_cases"]:
        case.update((case_edits or {}).get(case["name"], {}))
    if acceptance is not None:
        doc["acceptance"] = acceptance
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(doc, sort_keys=False))
    return dst


def make_real_run(yaml_path: Path, out: Path) -> Path:
    assert topopt_main(["run", str(yaml_path), "--out", str(out), "--quiet"]) == 0
    assert (out / "result" / "final.stl").is_file()
    return out


def region_stls(example: str, *names) -> list:
    return [EXAMPLES / example / n for n in names]


def log_rows(root):
    return rt.read_log(root)[0]


def fail_cases(row, rule):
    return sorted(c["case"] for c in row["checks"] if c["rule"] == rule and c["state"] == "fail")


# --- T4: the hook ------------------------------------------------------------------------------------------------------

HOOK_ACCEPTANCE = {"rules": [
    {"rule": "watertight"}, {"rule": "bodies"},
    {"rule": "stiffness", "max_drop_pct": 10, "source": "test: a limit above the drop of a small cut"},
    {"rule": "mass", "must_fall": True}]}
HOOK_REGIONS = ("wall_plate.stl", "hook_tip.stl", "screw_hole_top.stl", "screw_hole_bottom.stl")


@pytest.fixture(scope="module")
def hook(tmp_path_factory):
    d = tmp_path_factory.mktemp("hook")
    yaml_path = absolute_yaml(EXAMPLES / "hook" / "hook_short.yaml", d / "hook_acc.yaml", acceptance=HOOK_ACCEPTANCE)
    run = make_real_run(yaml_path, d / "run")
    return SimpleNamespace(dir=d, run=run, yaml=yaml_path, regions=region_stls("hook", *HOOK_REGIONS))


def hook_cuts(run, regions):
    """A small cube cut at the vertex that is farthest from every frozen region, and a deep notch across the bar.

    The notch is a box that takes away the upper part of the section at a third of the length of the part.
    """
    lo, hi = trimesh.load(str(run / "result" / "final.stl"), force="mesh").bounds
    x = lo[0] + 0.34 * (hi[0] - lo[0])
    zmid = 0.5 * (lo[2] + hi[2])
    notch_centre = (x, 0.0, zmid + 0.5 * (hi[2] - zmid) + 1.0)
    return [("small", lambda p: far_vertex(p, regions), 4.0),
            ("notch", notch_centre, (4.0, 2.0 * (hi[1] - lo[1]), hi[2] - zmid + 2.0)),
            ("same_again", lambda p: far_vertex(p, regions), 4.0)]


def test_hook_the_small_cut_is_promoted_the_deep_notch_is_rejected_and_the_chain_goes_on(hook):
    before = tree_hash(hook.run)
    rep = rt.check_report(hook.run)
    assert rep["ready"] is True and rep["tools"]["verify"]["ready"] is True
    cfg = load_config(hook.yaml)
    rules = rt.rules_from_config(cfg)
    ctx = rt.JudgeContext.from_config(cfg, rules, json.loads((hook.run / "run.json").read_text()))
    root = hook.dir / "ratchet"
    source = BoxCutSource(hook_cuts(hook.run, hook.regions))
    res = rt.run_ratchet(hook.run, source, rules, root=root, ctx=ctx, yaml_path=hook.yaml, say=quiet)
    assert res["exit_code"] == 0 and res["accepted"] == ["c001", "c003"] and res["rejected"] == ["c002"], res
    rows = log_rows(root)
    assert [(r["type"], r["decision"]) for r in rows] == [("baseline", "BASELINE"), ("candidate", "ACCEPT"),
                                                          ("candidate", "REJECT"), ("candidate", "ACCEPT")]
    base, small, notch, again = rows
    assert [r["parent"] for r in rows] == [None, "acc_00", "acc_01", "acc_01"]            # the notch and the next cut: same parent
    assert small["promoted"] == "acc_01" and notch["promoted"] is None and again["promoted"] == "acc_02"
    assert tree_hash(hook.run) == before                                                  # the start run is not written

    # the facts are numbers from the tools, each with its limit and where it came from
    assert base["facts"]["watertight"] is True and base["facts"]["bodies"] == 1 and base["errors"] is None
    assert set(base["facts"]["cases"]) == {"hang", "pull_off"} and base["facts"]["mesh_size_mm"] > 0
    stiff = [c for c in small["checks"] if c["rule"] == "stiffness"]
    assert {c["case"] for c in stiff} == {"hang", "pull_off"} and all(c["state"] == "pass" for c in stiff)
    assert all("test: a limit above the drop of a small cut" in c["limit_source"] for c in stiff)
    assert small["facts"]["mass_g"] < base["facts"]["mass_g"]
    assert "stiffness" in notch["failed"]
    assert notch["facts"]["mass_g"] < small["facts"]["mass_g"]                            # the mass fell: mass is not what failed
    assert [c["state"] for c in notch["checks"] if c["rule"] == "mass"] == ["pass"]
    # one mesh size for the whole ratchet: the size that verify picked for the baseline
    assert {r["facts"]["mesh_size_mm"] for r in rows} == {base["facts"]["mesh_size_mm"]}
    assert all("verify" in r["judge"]["timing_s"] for r in rows)

    # the folders: the parents are run-shaped, the last one is the second cut of the same size
    for name in ("acc_00", "acc_01", "acc_02"):
        assert (root / name / "run.json").is_file() and (root / name / "result" / "final.stl").is_file()
        assert not (root / name / "result" / "metrics.json").exists()
    vols = [trimesh.load(str(root / n / "result" / "final.stl"), force="mesh").volume for n in ("acc_00", "acc_01", "acc_02")]
    assert vols[0] > vols[1] > vols[2]

    # a second call goes on where the first one stopped: no candidate is left, nothing is judged again
    n_rows = len(rows)
    again_res = rt.run_ratchet(hook.run, source, rules, root=root, ctx=ctx, yaml_path=hook.yaml, say=quiet)
    assert again_res["exit_code"] == 0 and again_res["accepted"] == [] and len(log_rows(root)) == n_rows
    st = rt.status(root)
    assert st["parent"] == "acc_02" and st["counts"] == {"candidates": 3, "ACCEPT": 2, "REJECT": 1}


def test_hook_cli_judge_runs_the_real_tools_on_one_stl(hook, capsys, tmp_path):
    cut = hook_cuts(hook.run, hook.regions)[0]
    from ratchet_helpers import cut_box
    stl = cut_box(hook.run / "result" / "final.stl", tmp_path / "cut.stl", cut[1](hook.run / "result" / "final.stl"), cut[2])
    before = tree_hash(hook.run)
    code = rt.main(["judge", str(stl), "--parent", str(hook.run), "--out", str(tmp_path / "judged")])
    out = capsys.readouterr().out
    assert code == 0 and "ACCEPT" in out and "stiffness" in out
    assert tree_hash(hook.run) == before


# --- T5: the ladle, three load cases and a stress limit for each --------------------------------------------------------

LADLE_REGIONS = ("grip.stl", "bowl.stl", "gravy_space.stl", "over_opening.stl")
LADLE_ACCEPTANCE = {"rules": [
    {"rule": "watertight"}, {"rule": "bodies"}, {"rule": "stress"},
    {"rule": "stiffness", "max_drop_pct": 10, "source": "test"}, {"rule": "mass", "must_fall": True}]}


@pytest.fixture(scope="module")
def ladle(tmp_path_factory):
    """A short ladle run. The stress limits are 1.25 times the stress of the finished part, found with `verify`, so the test
    does not depend on the exact shape that a platform finds. A YAML can be changed after the run: the limits are not
    part of the run."""
    from topopt.verify.run import verify_run
    d = tmp_path_factory.mktemp("ladle")
    # The envelope fill is off: this test is about the judgement of a cut, and its 26 mm cut is tuned to the mesh of the run (with the fill on, the
    # mesh differs and the stress of side_stir stays under its limit). Measured on one host at 6.5 mm, 30 passes: the check of that case passes with the fill.
    settings = {"voxel_size": 6.5, "iterations": 30, "envelope_fill": "off"}
    first = absolute_yaml(EXAMPLES / "ladle" / "ladle.yaml", d / "ladle_acc.yaml", acceptance=LADLE_ACCEPTANCE, settings=settings)
    run = make_real_run(first, d / "run")
    probe = rt.make_run_folder(d / "probe", run, run / "result" / "final.stl", first, info={"kind": "probe"})
    base = rt.verify_facts(verify_run(probe, yaml_path=str(first), load_sensitivity=False))["cases"]
    limit = {n: round(1.25 * v["p_by_volume_MPa"], 4) for n, v in base.items()}
    edits = {"full_scoop": {"stress_limit": limit["full_scoop"]},
             "scrape": {"allowable_mpa": round(3.0 * limit["scrape"], 4), "safety_factor": 3.0,
                        "allowable_source": "test table of the material"},
             "side_stir": {"stress_limit": limit["side_stir"]}}
    path = absolute_yaml(EXAMPLES / "ladle" / "ladle.yaml", d / "ladle_acc.yaml", acceptance=LADLE_ACCEPTANCE, settings=settings,
                         case_edits=edits)
    return SimpleNamespace(dir=d, run=run, yaml=path, regions=region_stls("ladle", *LADLE_REGIONS), limit=limit, base=base)


def test_ladle_every_case_has_its_own_stress_limit_and_a_cut_that_loads_one_case_is_rejected(ladle):
    cfg = load_config(ladle.yaml)
    rules = rt.rules_from_config(cfg)
    assert {n: v[0] for n, v in rules.stress_limits.items()} == pytest.approx(ladle.limit, rel=1e-3)
    assert "allowable_mpa / safety_factor" in rules.stress_limits["scrape"][1]
    assert "test table of the material" in rules.stress_limits["scrape"][1]
    ctx = rt.JudgeContext.from_config(cfg, rules, json.loads((ladle.run / "run.json").read_text()))
    before = tree_hash(ladle.run)
    # The big cut is centred on the far vertex of the BASE result, the place of the first small cut. The far vertex of the cut parent
    # can be on another member: with the cut to the envelope (F4-51) it moved from y 21.6 to y -4.0 mm (the middle of the bottom
    # member), and a 22 mm cut there loaded all three cases.
    centre = far_vertex(ladle.run / "result" / "final.stl", ladle.regions)
    cuts = [("small", lambda p: far_vertex(p, ladle.regions), 6.0),
            # 26 mm: the stress of side_stir must end clearly above 1.25 x its base value. With the cut to the envelope (F4-51),
            # 22 mm gave 1.28 x, 24 mm 1.35 x and 26 mm 1.38 x; at 26 mm the stress of the other two cases stays below their base
            # and their stiffness within 3 % (one host). Before F4-51, 22 mm gave 1.38 x (an 18 mm cut tested the mesh noise).
            ("big", centre, 26.0),
            ("small_again", lambda p: far_vertex(p, ladle.regions), 6.0)]
    root = ladle.dir / "ratchet"
    res = rt.run_ratchet(ladle.run, BoxCutSource(cuts), rules, root=root, ctx=ctx, yaml_path=ladle.yaml, say=quiet)
    assert res["exit_code"] == 0 and res["accepted"] == ["c001", "c003"] and res["rejected"] == ["c002"], res
    base, small, big, again = log_rows(root)
    assert tree_hash(ladle.run) == before
    assert [r["parent"] for r in (small, big, again)] == ["acc_00", "acc_01", "acc_01"]
    assert again["promoted"] == "acc_02"
    # three load cases: one check each for stress and for stiffness
    for row in (small, big, again):
        assert [c["case"] for c in row["checks"] if c["rule"] == "stress"] == ["full_scoop", "scrape", "side_stir"]
        assert [c["case"] for c in row["checks"] if c["rule"] == "stiffness"] == ["full_scoop", "scrape", "side_stir"]
    assert all(c["state"] == "pass" for c in small["checks"])
    # the big cut: the load case side_stir gets the stress and the stiffness out of the limits; the other cases do not fail
    assert sorted(big["failed"]) == ["stiffness", "stress"]
    assert fail_cases(big, "stress") == ["side_stir"] and fail_cases(big, "stiffness") == ["side_stir"]
    sc = {c["case"]: c for c in big["checks"] if c["rule"] == "stress"}
    assert sc["side_stir"]["limit_source"].startswith("load_cases[side_stir].stress_limit")
    assert "allowable_mpa / safety_factor" in sc["scrape"]["limit_source"] and sc["scrape"]["state"] == "pass"
    assert sc["side_stir"]["value"] > ladle.limit["side_stir"] > sc["side_stir"]["parent_value"]
    # the record keeps the numbers of every case, and the log has no other file than the ones of the ratchet
    assert set(big["facts"]["cases"]) == {"full_scoop", "scrape", "side_stir"}
    assert {r["facts"]["mesh_size_mm"] for r in (base, small, big, again)} == {base["facts"]["mesh_size_mm"]}


# --- T6: the marks source and verify ---------------------------------------------------------------------------------------

def hook_marks(hook, tmp_path, rules, case_edits=None):
    """Two marks on the finished hook: a cut at the vertex that is farthest from the frozen regions, and one off the part."""
    far = far_vertex(hook.run / "result" / "final.stl", hook.regions)
    path = tmp_path / "marks.json"
    path.write_text(json.dumps(marks_doc(mark(1, "remove", [[float(x) for x in far]], radius=3.0),
                                         mark(2, "add", [[500.0, 500.0, 500.0]], radius=2.0))))
    yaml_path = absolute_yaml(EXAMPLES / "hook" / "hook_short.yaml", hook.dir / f"hook_{tmp_path.name}.yaml",
                              acceptance={"rules": rules}, case_edits=case_edits)
    return path, yaml_path


def run_marks(hook, tmp_path, capsys, rules, case_edits=None):
    marks, yaml_path = hook_marks(hook, tmp_path, rules, case_edits)
    root = tmp_path / "root"
    code = rt.main(["run", str(hook.run), "--source", f"marks:{marks}", "--root", str(root), "--yaml", str(yaml_path)])
    capsys.readouterr()
    assert code == 0
    return root, log_rows(root)


def test_t6_marks_source_works_with_printcheck_rules(hook, tmp_path, capsys):
    """The marks source works with the rules that need printcheck only (closed, bodies, mass)."""
    marks, yaml_path = hook_marks(hook, tmp_path, [{"rule": "watertight"}, {"rule": "bodies"}, {"rule": "mass", "must_fall": True}])
    before = tree_hash(hook.run)
    root = tmp_path / "root"
    code = rt.main(["run", str(hook.run), "--source", f"marks:{marks}", "--root", str(root), "--yaml", str(yaml_path)])
    capsys.readouterr()
    assert code == 0 and tree_hash(hook.run) == before
    base, cut, off = log_rows(root)
    assert (cut["decision"], off["decision"]) == ("ACCEPT", "REJECT") and cut["promoted"] == "acc_01"
    assert cut["facts"]["mass_g"] < base["facts"]["mass_g"] and cut["judge"]["tools"] == {"printcheck": "exit 0"}
    assert cut["source"]["facts"]["marks_status"] == "applied" and cut["source"]["facts"]["frozen_unchanged"] is not False
    assert off["failed"] == ["no_candidate"] and "outside the part" in off["source"]["note"]
    assert (root / "acc_01" / "labels.bin").is_file()                                     # apply_marks can take acc_01 as a parent


def test_t6_stress_on_a_marks_candidate_is_judged_by_the_stress_rule_like_any_other(hook, tmp_path, capsys):
    """Since the C3a fix, `topopt verify` meshes the sewn output of `apply_marks`: no `errors.verify`, no `no_input`.

    The first cut is solved and the stress rule decides. With `not_worse` and no tolerance the cut is REJECTED, because the p99.9
    stress by volume of at least one load case is a little higher than the stress of the parent (a fact, not a mesh error).
    Which cases get worse depends on the optimized hook (P3a `objective_scale: stage` changed it: on main both cases got worse;
    with `stage` the hang case does not), so the test checks the rule on each case, not a fixed set of states.
    """
    root, (base, cut, off) = run_marks(hook, tmp_path, capsys, [{"rule": "watertight"}, {"rule": "stress", "against_parent": "not_worse"}])
    assert base["errors"] is None and base["decision"] == "BASELINE"                      # the finished part meshes
    assert cut["errors"] is None and cut["judge"]["tools"] == {"printcheck": "exit 0", "verify": "exit 0"}
    assert cut["decision"] == "REJECT" and cut["failed"] == ["stress"]
    stress = [c for c in cut["checks"] if c["rule"] == "stress"]
    assert {c["case"] for c in stress} == {"hang", "pull_off"} and any(c["state"] == "fail" for c in stress)
    for c in stress:                                                                      # the state follows the rule; a failed case says by how much
        assert c["form"] == "not_worse"
        if c["state"] == "fail":
            assert c["value"] > c["parent_value"] and c["delta_pct"] > 0 and "moved +" in c["reason"]
            assert "no tolerance_pct is set" in c["reason"]
        else:
            assert c["state"] == "pass" and c["value"] <= c["parent_value"]
    assert cut["facts"]["mesh_size_mm"] == base["facts"]["mesh_size_mm"]                  # one mesh size for the whole ratchet
    assert [c["state"] for c in cut["checks"] if c["rule"] == "watertight"] == ["pass"]
    assert off["failed"] == ["no_candidate"] and not (root / "acc_01").exists()


def test_t6_cap_or_parent_with_a_limit_above_the_parent_accepts_the_same_cut(hook, tmp_path, capsys):
    """The same cut with `cap_or_parent` and a stress limit above the stress of the parent: ACCEPT, and `acc_01` is a parent
    that `apply_marks` and the next judge can use. The limits are 1.25 times the stress of the finished part (found with
    `verify`, like the ladle fixture), so the test does not depend on the exact numbers of a platform."""
    from topopt.verify.run import verify_run
    probe = rt.make_run_folder(tmp_path / "probe", hook.run, hook.run / "result" / "final.stl", hook.yaml, info={"kind": "probe"})
    first = rt.verify_facts(verify_run(probe, yaml_path=str(hook.yaml), load_sensitivity=False))["cases"]
    edits = {n: {"stress_limit": round(1.25 * v["p_by_volume_MPa"], 6)} for n, v in first.items()}
    root, (base, cut, off) = run_marks(hook, tmp_path, capsys, [{"rule": "watertight"}, {"rule": "stress", "against_parent": "cap_or_parent"}],
                                       case_edits=edits)
    assert base["errors"] is None and cut["errors"] is None
    assert cut["judge"]["tools"] == {"printcheck": "exit 0", "verify": "exit 0"}
    assert cut["decision"] == "ACCEPT" and cut["failed"] == [] and cut["promoted"] == "acc_01"
    stress = [c for c in cut["checks"] if c["rule"] == "stress"]
    assert {(c["case"], c["state"]) for c in stress} == {("hang", "pass"), ("pull_off", "pass")}
    for c in stress:
        assert c["form"] == "cap_or_parent" and c["parent_value"] < c["limit_value"] and c["value"] <= c["limit_value"]
    assert (root / "acc_01" / "result" / "final.stl").is_file() and (root / "acc_01" / "labels.bin").is_file()
    assert off["failed"] == ["no_candidate"]
