"""Tests for the facts of a mass run (F1-3a, topopt/stage_facts.py, CONTRACTS.md sections 13 and 24.2).

A run with `settings.objective: mass` and no row named `volume` says how much material it uses (`material`). When the grayness of a pass is above
`settings.stop_rules.grayness_max`, the stage end of that pass also says that the design is grey (`grey`): the grayness, the volume and the body count of the
thresholded part at the cuts 0.4, 0.5 and 0.6, and the volume spread between 0.4 and 0.6 as a share of the volume at 0.5. The post step adds the body count of
`final.stl` (`metrics.grey_design`), because `verdict.json` is written before the post step and cannot have it. All of it is fact: no limit, no gate,
and the state of the verdict does not change. A run with a volume row keeps its texts.
"""

import copy
import dataclasses
import json

import numpy as np
import pytest

from topopt import stage_facts
from topopt.config import StopRules, load_config
from topopt.formulation import Formulation
from topopt.problem import build_problem
from topopt.verdict import build_verdict
from topopt.voxelize import DESIGN, FIXED, LOAD

from test_objective_scale import case_yaml, sub
from test_stage_facts import read_history, run_case

SCHEDULE_ENDS = [4, 9, 14]      # the last pass of each stage of the schedule [[0, 1], [5, 2], [10, 4]] with 14 iterations
CRISP = {"enabled": False, "grayness_max": 1.0}     # the grayness of a design cannot be above 1: no pass is grey


def with_settings(problem, **changes):
    """The same `Problem` with other `settings` values."""
    cfg = problem.config
    return dataclasses.replace(problem, config=dataclasses.replace(cfg, settings=dataclasses.replace(cfg.settings, **changes)))


def metrics_of(out):
    return json.loads((out / "result" / "metrics.json").read_text())


def verdict_of(out):
    return json.loads((out / "verdict.json").read_text())


def log_of(out):
    return (out / "run.log").read_text().splitlines()


@pytest.fixture(scope="module")
def grey_stress(tmp_path_factory):
    """A mass run with a stress row, 14 passes: it ends grey (0.92) with the schedule that stops at beta 4."""
    return run_case(tmp_path_factory.mktemp("mrf_stress"), "stress")


@pytest.fixture(scope="module")
def grey_mass(tmp_path_factory):
    """A mass run held by a compliance limit: grey at the last pass; the thresholded part has 2 bodies and the mesh step keeps 1."""
    return run_case(tmp_path_factory.mktemp("mrf_mass"), "mass")


@pytest.fixture(scope="module")
def crisp_mass(tmp_path_factory):
    """The same run with `grayness_max` 1: the design is never above it, so the run is not grey for the verdict."""
    return run_case(tmp_path_factory.mktemp("mrf_crisp"), "mass", edit=lambda d: d["settings"].update(stop_rules=dict(CRISP)))


@pytest.fixture(scope="module")
def compliance(tmp_path_factory):
    """A run with the compliance objective and a volume row."""
    return run_case(tmp_path_factory.mktemp("mrf_sum"))


# --- the facts of one stage end, by hand ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mass_problem(tmp_path_factory):
    return build_problem(load_config(case_yaml(sub(tmp_path_factory.mktemp("mrf_problem"), "in"), variant="mass")))


def hand_design(problem, n1, n2, n3, top=200):
    """Snapshot bytes of the full grid and the density of the design voxels: n1 voxels at byte `top` (default 200: solid at 0.4, 0.5, 0.6), n2 at 130 (0.4 and 0.5),
    n3 at 110 (0.4 only); the other design voxels are void."""
    labels = problem.labels
    idx = np.flatnonzero(labels.ravel() == DESIGN)
    assert len(idx) >= n1 + n2 + n3
    values = np.zeros(len(idx), np.uint8)
    values[:n1], values[n1:n1 + n2], values[n1 + n2:n1 + n2 + n3] = top, 130, 110
    snap = np.zeros(labels.size, np.uint8)
    snap[idx] = values
    return snap.reshape(labels.shape), values.astype(np.float64) / 255.0


def test_the_material_and_the_grey_fact_of_a_stage_end_are_counted_by_hand(mass_problem):
    p = mass_problem
    n1, n2, n3 = 30, 20, 10
    snap, rho = hand_design(p, n1, n2, n3)
    forced = int(((p.labels == FIXED) | (p.labels == LOAD)).sum())
    voxel = p.h ** 3
    end = stage_facts.stage_end(p, snap, rho, [], 2.0, None)
    assert end["material"] == {"volume_mm3": round((forced + n1 + n2) * voxel, 1), "density_g_cm3": 1.24,
                               "mass_g": round((forced + n1 + n2) * voxel / 1000.0 * 1.24, 3)}
    grey = end["grey"]
    assert grey["grayness"] == pytest.approx(float(np.mean(4 * rho * (1 - rho))))
    assert grey["grayness"] > 0.1 and grey["grayness_max"] == 0.1
    assert [lv["level"] for lv in grey["levels"]] == [0.4, 0.5, 0.6]
    assert [lv["volume_mm3"] for lv in grey["levels"]] == [round((forced + n) * voxel, 1) for n in (n1 + n2 + n3, n1 + n2, n1)]
    assert grey["volume_spread"] == pytest.approx((n2 + n3) / (forced + n1 + n2), rel=1e-3)      # the volume is rounded to 0.1 mm3
    assert "volume" not in end and "bodies" in end["load_path"]


def test_a_design_at_or_under_grayness_max_has_the_material_fact_only(mass_problem):
    p = mass_problem
    snap, rho = hand_design(p, 30, 0, 0, top=255)         # solid or void everywhere: grayness 0
    assert stage_facts.grayness(rho) == 0.0
    end = stage_facts.stage_end(p, snap, rho, [], 2.0, None)
    assert "material" in end and "grey" not in end
    rho = np.full(len(rho), 0.5)                          # grayness 1.0, and grayness_max 1.0 is the largest value the loader accepts
    crisp = with_settings(p, stop_rules=StopRules(grayness_max=1.0))
    assert "grey" not in stage_facts.stage_end(crisp, snap, rho, [], 2.0, None)
    assert "grey" in stage_facts.stage_end(p, snap, rho, [], 2.0, None)


def test_the_volume_row_of_a_run_gives_the_volume_fact_and_no_material_fact(mass_problem):
    p = mass_problem
    snap, rho = hand_design(p, 30, 20, 10)
    rows = [{"name": "volume", "limit": 0.6, "value": 0.1, "g": -0.8}]
    end = stage_facts.stage_end(p, snap, rho, rows, 2.0, None)
    assert "volume" in end and "material" not in end and "grey" in end        # a mass run with the volume fallback: the volume fact stays, the grey fact is added


def test_a_compliance_run_has_neither_new_fact(mass_problem):
    p = mass_problem
    snap, rho = hand_design(p, 30, 20, 10)
    end = stage_facts.stage_end(with_settings(p, objective="compliance"), snap, rho, [], 2.0, None)
    assert set(end) == {"beta", "next_beta", "load_path"}


def test_the_texts_of_the_facts(mass_problem):
    p = mass_problem
    snap, rho = hand_design(p, 30, 20, 10)
    end = stage_facts.stage_end(p, snap, rho, [], 4.0, None)
    assert stage_facts.material_text(end["material"]).startswith("material: the thresholded part (density 0.5) has ")
    assert stage_facts.material_text(end["material"]).endswith("(material density 1.24 g/cm3)")
    grey = end["grey"]
    text = stage_facts.grey_text(grey, 4.0)
    g, lv = grey["grayness"], grey["levels"]
    assert text == (f"grey design: grayness {g:.3g} is above grayness_max 0.1 at beta 4, so the thresholded part depends on the cut. "
                    f"Its volume is {lv[0]['volume_mm3']:.0f} mm3 at the cut 0.4, {lv[1]['volume_mm3']:.0f} mm3 at 0.5 and {lv[2]['volume_mm3']:.0f} mm3 at 0.6, "
                    f"a spread of {grey['volume_spread']:.2f} of the volume at 0.5. "
                    f"Its body count is {lv[0]['bodies']} at 0.4, {lv[1]['bodies']} at 0.5 and {lv[2]['bodies']} at 0.6")
    assert "final.stl" not in text and "options" not in text
    final = stage_facts.grey_text(grey, 4.0, final=True)
    assert final.startswith(text + ". ") and "metrics.grey_design.final_stl_bodies" in final
    lines = stage_facts.stage_end_lines(end, 14)
    assert [ln.split(": ")[2] for ln in lines] == ["material", "grey design", "load path"]
    assert lines[0].startswith("fact: last pass 14: material: ") and stage_facts.GREY_OPTIONS in lines[1]
    assert not any(stage_facts.GREY_OPTIONS in ln for ln in stage_facts.stage_end_lines({**end, "next_beta": 8.0}, 9))   # the options come once, at the last pass


def test_a_stop_by_request_names_no_post_step(mass_problem):
    """A run that a request stopped at a pass that is not the last has no post step: its facts do not point at `metrics.grey_design`, and give no options."""
    p = mass_problem
    snap, rho = hand_design(p, 30, 20, 10)
    end = stage_facts.stage_end(p, snap, rho, [], 2.0, 4.0)
    sentence = stage_facts.verdict_sentence({"iteration": 7, **end})
    assert "grey design: grayness" in sentence and "material: the thresholded part" in sentence and "final.stl" not in sentence
    lines = stage_facts.stage_end_lines(end, 7, where="stop at pass 7")
    assert [ln.split(": ")[1:3] for ln in lines] == [["stop at pass 7", "material"], ["stop at pass 7", "grey design"], ["stop at pass 7", "load path"]]
    assert not any("options:" in ln or "final.stl" in ln for ln in lines)


def test_the_text_of_a_grey_fact_without_a_cut_check_names_the_error():
    grey = {"grayness": 0.5, "grayness_max": 0.1, "error": "ValueError: no grid"}
    assert stage_facts.grey_text(grey, 2.0) == ("grey design: grayness 0.5 is above grayness_max 0.1 at beta 2, so the thresholded part depends on the cut "
                                                 "(the check of the cuts was not made: ValueError: no grid)")
    assert stage_facts.grey_text({"error": "boom"}) == "grey design: the check was not made (boom)"
    assert stage_facts.material_text({"error": "boom"}) == "material: the check was not made (boom)"


def test_the_spread_of_a_part_that_is_void_at_the_middle_cut_is_not_a_division(mass_problem):
    p = mass_problem
    labels = np.where((mass_problem.labels == FIXED) | (mass_problem.labels == LOAD), 0, mass_problem.labels).astype(np.uint8)    # no forced voxel
    void = dataclasses.replace(p, labels=labels)
    snap = np.zeros(labels.shape, np.uint8)
    snap[labels == DESIGN] = 110                          # solid at the cut 0.4 only
    grey = stage_facts.grey_fact(void, snap, np.full(int((labels == DESIGN).sum()), 0.5))
    assert grey["volume_spread"] is None and grey["levels"][1]["volume_mm3"] == 0.0 and grey["levels"][0]["volume_mm3"] > 0
    assert "a spread" not in stage_facts.grey_text(grey, 2.0)


# --- a grey mass run ----------------------------------------------------------------------------------------------

def test_every_stage_end_of_a_grey_mass_run_has_the_material_and_the_grey_fact(grey_stress):
    _, out = grey_stress
    rows = read_history(out)
    ends = [r for r in rows if "stage_end" in r]
    assert [r["iteration"] for r in ends] == SCHEDULE_ENDS
    for r in ends:
        end = r["stage_end"]
        assert "volume" not in end and end["material"]["volume_mm3"] > 0
        grey = end["grey"]
        assert grey["grayness"] == r["grayness"] and grey["grayness_max"] == 0.1          # the number of the verdict, to the last digit
        by = {lv["level"]: lv for lv in grey["levels"]}
        assert grey["volume_spread"] == pytest.approx((by[0.4]["volume_mm3"] - by[0.6]["volume_mm3"]) / by[0.5]["volume_mm3"])
        assert end["material"]["volume_mm3"] == by[0.5]["volume_mm3"]                      # the material is the thresholded part at 0.5
        assert by[0.5]["bodies"] == end["load_path"]["bodies"]                             # the same labeling rule as the load path fact


def test_the_last_stage_end_has_the_cuts_of_the_post_step(grey_stress):
    _, out = grey_stress
    last = read_history(out)[-1]["stage_end"]
    assert last["grey"]["levels"] == metrics_of(out)["threshold_stability"]["levels"]       # one snapshot, one rule: the same numbers


def test_the_run_log_has_a_stage_end_line_for_the_grey_design(grey_stress):
    _, out = grey_stress
    rows = read_history(out)
    lines = log_of(out)
    for it, tag in zip(SCHEDULE_ENDS, ("end of the beta 1 stage, pass 4", "end of the beta 2 stage, pass 9", "last pass 14")):
        grey = next(r for r in rows if r["iteration"] == it)["stage_end"]["grey"]
        line = next(ln for ln in lines if ln.startswith(f"fact: {tag}: grey design: "))
        assert f"grayness {grey['grayness']:.3g} is above grayness_max 0.1" in line
        assert f"a spread of {grey['volume_spread']:.2f} of the volume at 0.5" in line
        assert "Its body count is " in line and "so the thresholded part depends on the cut" in line
        assert any(ln.startswith(f"fact: {tag}: material: ") for ln in lines)
        at = next(k for k, ln in enumerate(lines) if ln.startswith(f"it {it:3d}  beta="))
        assert at < lines.index(line)                                                       # the line follows the line of its own pass
    last = next(ln for ln in lines if ln.startswith("fact: last pass 14: grey design: "))
    assert "the body count of final.stl comes after the post step" in last and stage_facts.GREY_OPTIONS in last
    assert not any(stage_facts.GREY_OPTIONS in ln for ln in lines if "end of the beta" in ln)


def test_the_verdict_text_says_grey_with_the_grayness_value_the_cuts_and_the_bodies(grey_stress):
    _, out = grey_stress
    v, rows = verdict_of(out), read_history(out)
    end = rows[-1]["stage_end"]
    gray = v["measures"]["grayness"]
    assert gray["met"] is False and gray["value"] == end["grey"]["grayness"]
    tests, facts = v["text"].split(" Facts: ")
    assert f"grayness {gray['value']:.3g} (threshold 0.1: not met)" in tests
    assert stage_facts.material_text(end["material"]) in facts
    assert stage_facts.grey_text(end["grey"], end["beta"], final=True) in facts
    assert "a spread of " in facts and "of the volume at 0.5" in facts and "Its body count is " in facts
    assert v["facts"] == {"iteration": 14, **end}
    assert metrics_of(out)["verdict"] == v and "verdict: " + v["text"] in "\n".join(log_of(out))        # metrics.verdict is verdict.json: the post step adds nothing to it


def test_the_new_facts_change_no_state_no_test_no_feasible(grey_stress):
    """The verdict of the run equals the verdict of the same history with the two new keys cut out, except for the text after "Facts:"."""
    _, out = grey_stress
    rows = read_history(out)
    stripped = copy.deepcopy(rows)
    for r in stripped:
        for key in ("material", "grey"):
            r.get("stage_end", {}).pop(key, None)
    cfg = load_config(case_yaml(sub(out.parent, "cfg_in"), variant="stress")).settings
    v, plain = verdict_of(out), build_verdict(stripped, cfg.stop_rules, cfg.beta_schedule, cfg.iterations)
    for key in ("state", "converged", "feasible", "measures", "final_iteration", "stop", "stop_rules"):
        assert v[key] == plain[key]
    assert v["text"].startswith(plain["text"].split(" Facts: ")[0]) and len(v["text"]) > len(plain["text"])


def test_the_stage_note_says_that_met_holds_for_the_grey_design(grey_stress):
    _, out = grey_stress
    rows = read_history(out)
    note = next(ln for ln in log_of(out) if ln.startswith("beta stage 1 -> 2 at pass 5: "))
    old = "all rows met (a stress row is the relaxed measure of the optimizer, not the stress of the part)"
    gray = rows[4]["grayness"]
    assert old in note                                                                        # the old words stay
    assert (f"{old}; the design of the old stage is grey (grayness {gray:.3g}, grayness_max 0.1), "
            "so \"met\" holds for this grey design and not for the thresholded part; the schedule goes by iteration count") in note
    assert rows[5]["stage"]["grey"] == {"grayness": gray, "grayness_max": 0.1}


def test_the_stage_note_of_a_feasible_stage_without_a_grey_fact_keeps_its_words():
    note = {"iteration": 5, "from_beta": 1.0, "to_beta": 2.0, "row": "stress:down", "max_g": -0.01, "feasible": True, "notes": []}
    assert Formulation.stage_lines(note) == [
        "beta stage 1 -> 2 at pass 5: largest g of the old stage -0.0100 (stress:down), all rows met (a stress row is the relaxed measure of the optimizer, "
        "not the stress of the part); the schedule goes by iteration count"]
    not_met = {**note, "feasible": False, "max_g": 0.2, "grey": {"grayness": 0.9, "grayness_max": 0.1}}
    assert "grey" not in Formulation.stage_lines(not_met)[0]                                  # a row that is not met has its own words


# --- the post step ------------------------------------------------------------------------------------------------

def test_the_post_step_gives_the_body_counts_of_the_thresholded_part_and_of_final_stl(grey_mass):
    _, out = grey_mass
    m, rows = metrics_of(out), read_history(out)
    last = rows[-1]["stage_end"]
    gd = m["grey_design"]
    assert last["load_path"]["bodies"] == 2 and gd["thresholded_bodies"] == 2          # the thresholded part has two bodies ...
    assert gd["final_stl_bodies"] == m["mesh"]["stl_roundtrip"]["bodies"] == 1         # ... and final.stl has one: the mesh step took one out
    assert gd["mesh_bodies_removed"] == m["mesh"]["extra_bodies_removed"] == 1
    assert gd["mesh_bodies_removed_mm3"] == m["mesh"]["extra_bodies_volume_mm3"] > 0
    assert gd["grayness"] == verdict_of(out)["measures"]["grayness"]["value"] and gd["grayness_max"] == 0.1
    assert gd["text"] == ("grey design: the thresholded part has 2 bodies at the cut 0.5 (2 at 0.4, 2 at 0.6), and final.stl has 1 body. "
                          "The first count is on the voxel grid and the second on the smoothed mesh. "
                          f"The mesh step keeps only the bodies that touch a frozen region; it removed 1 body ({gd['mesh_bodies_removed_mm3']:.0f} mm3)")
    assert "fact: " + gd["text"] in log_of(out)
    assert list(m)[-2:] == ["grey_design", "verdict"]                                  # `verdict` stays the last key
    assert m["verdict"] == verdict_of(out)                                             # the bytes of verdict.json are not changed after the run: the post step copies them


def test_the_final_stl_number_is_in_no_verdict_text(grey_mass):
    _, out = grey_mass
    v = verdict_of(out)
    assert "final.stl has" not in v["text"] and "final_stl_bodies" in v["text"]        # the verdict says where the number is, not the number
    assert json.loads(json.dumps(metrics_of(out)["verdict"])) == v


def test_the_post_step_record_for_a_blur_mesh_and_for_missing_numbers():
    facts = {"grey": {"grayness": 0.5, "grayness_max": 0.1}}
    stability = {"levels": [{"level": 0.4, "volume_mm3": 3.0, "bodies": 2}, {"level": 0.5, "volume_mm3": 2.0, "bodies": 3},
                            {"level": 0.6, "volume_mm3": 1.0, "bodies": 1}]}
    blur = {"stl_roundtrip": {"bodies": 3}}              # the old mesh method has no `extra_bodies_removed`
    rec = stage_facts.grey_design_record(facts, stability, blur)
    assert rec["final_stl_bodies"] == 3 and "mesh_bodies_removed" not in rec
    assert rec["text"] == ("grey design: the thresholded part has 3 bodies at the cut 0.5 (2 at 0.4, 1 at 0.6), and final.stl has 3 bodies. "
                           "The first count is on the voxel grid and the second on the smoothed mesh")
    assert "body counts were not made" in stage_facts.grey_design_record(facts, {"error": "x"}, blur)["text"]
    assert "body counts were not made" in stage_facts.grey_design_record(facts, stability, {})["text"]
    assert stage_facts.grey_design_record({"volume": {}}, stability, blur) is None       # no grey fact: no record
    assert stage_facts.grey_design_record(None, stability, blur) is None


def test_the_mesh_step_that_removed_no_body_says_so(grey_stress):
    _, out = grey_stress
    gd = metrics_of(out)["grey_design"]
    assert gd["final_stl_bodies"] == 2 == gd["thresholded_bodies"] and gd["mesh_bodies_removed"] == 0
    assert gd["text"].endswith("and final.stl has 2 bodies. The first count is on the voxel grid and the second on the smoothed mesh. "
                               "The mesh step keeps only the bodies that touch a frozen region; it removed no body")


# --- a mass run that is not grey -------------------------------------------------------------------------------------

def test_a_mass_run_that_is_not_grey_has_the_material_fact_only(crisp_mass):
    _, out = crisp_mass
    rows = read_history(out)
    assert rows[-1]["grayness"] > 0.1                                                      # the design is grey by the default limit, and the limit of this run is 1
    assert [r["iteration"] for r in rows if "stage_end" in r] == SCHEDULE_ENDS
    assert all("material" in r["stage_end"] and "grey" not in r["stage_end"] for r in rows if "stage_end" in r)
    v = verdict_of(out)
    assert v["measures"]["grayness"]["met"] is True
    assert " Facts: " + stage_facts.material_text(rows[-1]["stage_end"]["material"]) in v["text"] and "grey design: grayness" not in v["text"]
    assert "grey design: grayness" not in "\n".join(log_of(out))
    assert "grey_design" not in metrics_of(out)
    assert "final.stl" not in v["text"]


def test_the_material_is_the_thresholded_part_of_the_last_snapshot(crisp_mass):
    _, out = crisp_mass
    material = read_history(out)[-1]["stage_end"]["material"]
    at_half = metrics_of(out)["threshold_stability"]["levels"][1]
    assert at_half["level"] == 0.5 and material["volume_mm3"] == at_half["volume_mm3"]
    assert material["mass_g"] == pytest.approx(material["volume_mm3"] / 1000.0 * 1.24, abs=1e-3)


def test_a_mass_run_with_the_volume_fallback_keeps_its_volume_fact(tmp_path):
    """A mass run with no other row gets the volume row (a ceiling at volume_fraction): it has the volume fact already, so no material fact."""
    _, out = run_case(tmp_path, "mass", edit=lambda d: d["load_cases"][0].pop("compliance_limit"))
    rows = read_history(out)
    ends = [r["stage_end"] for r in rows if "stage_end" in r]
    assert all("volume" in e and "material" not in e for e in ends)
    v = verdict_of(out)
    assert " Facts: volume: " in v["text"] and "material:" not in v["text"]
    assert all(("grey" in e) == (r["grayness"] > 0.1) for e, r in zip(ends, [r for r in rows if "stage_end" in r]))


# --- a run that is not a mass run ---------------------------------------------------------------------------------------

def test_a_compliance_run_keeps_its_stage_end_texts(compliance):
    _, out = compliance
    rows = read_history(out)
    ends = [r for r in rows if "stage_end" in r]
    assert [set(r["stage_end"]) for r in ends] == [{"beta", "next_beta", "volume", "load_path"}] * 3
    lines = [ln for ln in log_of(out) if ln.startswith("fact: ") and ("stage, pass" in ln or "last pass" in ln)]
    assert [ln.split(": ")[2].split(" ")[0] for ln in lines] == ["volume", "load"] * 3
    v = verdict_of(out)
    assert v["text"].endswith(" Facts: " + stage_facts.volume_text(rows[-1]["stage_end"]["volume"]) + ".")
    assert set(v["facts"]) == {"iteration", "beta", "next_beta", "volume", "load_path"}
    text = "\n".join(log_of(out)) + v["text"] + json.dumps(metrics_of(out))
    assert "grey design: grayness" not in text and "material: the thresholded" not in text and "grey_design" not in text


def test_a_compliance_run_with_a_stress_row_keeps_its_stage_note(tmp_path):
    _, out = run_case(tmp_path, "stress_compliance")
    rows = read_history(out)
    notes = [ln for ln in log_of(out) if ln.startswith("beta stage 1 -> 2")]
    assert len(notes) == 1 and "grey" not in notes[0] and notes[0].endswith("; the schedule goes by iteration count")
    assert all("grey" not in r.get("stage", {}) for r in rows)
    assert all(set(r["stage_end"]) == {"beta", "next_beta", "volume", "load_path"} for r in rows if "stage_end" in r)


def test_the_facts_of_a_history_from_before_this_change_give_the_same_verdict_sentence():
    """A `stage_end` of the P3a shape (volume and load path) makes the sentence it made before: the new keys are read with `get`."""
    old = {"iteration": 14, "beta": 4.0, "next_beta": None,
           "volume": {"limit": 0.3, "projected": 0.15, "thresholded": 0.3, "used_projected": 0.5, "used_thresholded": 1.0},
           "load_path": {"regions": {"tip": False}, "not_joined": ["tip"], "bodies": 2, "bodies_with_a_support": 1}}
    assert stage_facts.verdict_sentence(old) == (
        " Facts: volume: 0.50 of the allowed volume used (projected design), 1.00 (thresholded part at 0.5); load path: load region 'tip' is not joined "
        "to a fixed, spring or joint region in the thresholded design (density 0.5) at beta 4. A low beta gives a grey design: the thresholded part "
        "is then not the final part.")
    assert stage_facts.verdict_sentence({k: v for k, v in old.items() if k != "volume"}).startswith(" Facts: load path: ")
    assert stage_facts.verdict_sentence({"iteration": 3, "beta": 1.0, "next_beta": None, "load_path": {"regions": {}, "not_joined": [], "bodies": 1,
                                                                                                  "bodies_with_a_support": 1}}) == ""


# --- MCP ------------------------------------------------------------------------------------------------------------------

def test_run_status_and_run_result_carry_the_grey_mass_run_facts(grey_mass):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    _, out = grey_mass
    st = server.run_status(str(out))
    assert "Facts: material: " in st["summary"] and "grey design: grayness " in st["summary"]
    assert st["verdict"]["facts"]["grey"]["grayness"] == verdict_of(out)["facts"]["grey"]["grayness"]
    res = server.run_result(str(out), preview=False)
    assert res["metrics"]["grey_design"]["final_stl_bodies"] == 1
    assert res["metrics"]["verdict"]["facts"] == st["verdict"]["facts"]
    from topopt_mcp import replies
    assert "metrics.grey_design" in replies.guide("run_result") and "the `grey` design" in server.run_status.__doc__
