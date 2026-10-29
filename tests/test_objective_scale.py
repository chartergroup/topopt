"""Tests for `settings.objective_scale` (P3a, CONTRACTS.md section 13): the scale of the objective at each beta stage.

The MMA objective is divided by a scale, `MmaState.f0_ref`. Mode "first" (the code before P3a) takes it at pass 0 and keeps it. Mode "stage" (the default)
takes it again at the first pass of each beta stage, from the design of that pass. The reason: a beta jump removes much of the projected volume, the volume
row is then inactive, only the objective moves the design, and a scale from a near-void start makes its gradient smaller than the absolute MMA
regularization `raa0` (1e-5), so the volume comes back slowly.
"""

import copy

import numpy as np
import pytest
import trimesh
import yaml

from topopt.config import UserError, load_config
from topopt.optimize import Model, beta_at, formulation_from_config, minimize, objective_scale_resets
from topopt.problem import build_problem

from test_optimize import make_case

SCHEDULE = [[0, 1], [5, 2], [10, 4]]
STARTS = [5, 10]     # the first pass of each beta stage after pass 0


def sub(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def case_yaml(tmp_path, scale=None, variant="sum", iterations=14, h=2.0):
    """The 2 mm cantilever with a short schedule. `variant`: the objective and the rows that the scale has to work with."""
    p = make_case(tmp_path, h=h, iterations=iterations)
    d = yaml.safe_load(p.read_text())
    st = d["settings"]
    st["beta_schedule"] = copy.deepcopy(SCHEDULE)
    if scale is not None:
        st["objective_scale"] = scale
    if variant == "max":                                   # the bound form: one row for each case and the variable z
        st["case_combine"] = "max"
    elif variant == "mass":                                # the mass, held by a compliance limit
        st.update(objective="mass", volume_fraction=0.6)
        d["load_cases"][0]["compliance_limit"] = 2.0
    elif variant == "stress":                              # a KS stress row, with step acceptance
        st.update(objective="mass", volume_fraction=0.6, move_limit=0.05)
        d["load_cases"][0].update(stress_limit=10.0)
        d["load_cases"][0]["forces"][0]["force"] = [0, 0, -20.0]
    elif variant == "stress_compliance":                   # the compliance objective with a stress row and the volume row
        d["load_cases"][0].update(stress_limit=10.0)
        d["load_cases"][0]["forces"][0]["force"] = [0, 0, -20.0]
    elif variant == "robust":                              # volume on the nominal design, compliance on the eroded one
        st["robust"] = "eroded"
    else:
        assert variant == "sum"
    p.write_text(yaml.safe_dump(d))
    return p


def same(a, b, rel=1e-8):
    """Equal records, to a relative 1e-8: a direct solver with more than one thread is not bit-reproducible (about 1e-12)."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], rel) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float):
        return a == pytest.approx(b, rel=rel, abs=1e-12)
    return a == b


def run_minimize(path, **kw):
    """`minimize` with a spy on `Formulation.respond`. Return `(result, calls, records)`.

    `calls`: `(pass, the scale was None before the call, the scale after it)` for each pass. `records`: the history of the run.
    """
    model = Model(build_problem(load_config(path)))
    form = formulation_from_config(model.problem.config, model.n_design, model)
    calls = []
    real = form.respond

    def spy(point, state):
        before = state.f0_ref
        out = real(point, state)
        calls.append((point.iteration, before is None, state.f0_ref))
        assert out.f0_ref == state.f0_ref
        return out

    form.respond = spy
    try:
        res = minimize(model, form, **kw)
    finally:
        model.solver.free()
    return res, calls, res.history


# --- the key ----------------------------------------------------------------------------------------------------

def test_the_default_is_stage_and_both_values_load(tmp_path):
    assert load_config(case_yaml(tmp_path)).settings.objective_scale == "stage"
    assert load_config(case_yaml(tmp_path, "stage")).settings.objective_scale == "stage"
    assert load_config(case_yaml(tmp_path, "first")).settings.objective_scale == "first"


@pytest.mark.parametrize("bad", ["every", "Stage", "", 1, True, ["stage"]])
def test_other_values_are_refused_with_the_two_names(tmp_path, bad):
    with pytest.raises(UserError, match=r"settings\.objective_scale' must be stage or first, got "):
        load_config(case_yaml(tmp_path, bad))


def test_the_part_builder_passes_the_key(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    spec = {"name": "w", "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [40, 20, 20]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 20]}}]},
        "load": {"tip": [{"box": {"min": [36, 8, 8], "max": [40, 12, 12]}}]}},
        "load_cases": [{"name": "a", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
        "settings": {"voxel_size": 2.0, "objective_scale": "first"}}
    r = server.build_regions(spec)
    assert r["ok"], r
    doc = yaml.safe_load((tmp_path / "ws" / r["yaml_rel"]).read_text())
    assert doc["settings"]["objective_scale"] == "first"
    assert load_config(tmp_path / "ws" / r["yaml_rel"]).settings.objective_scale == "first"


# --- when the scale is set again --------------------------------------------------------------------------------

def test_the_rule_by_the_pass_number_and_the_schedule():
    sched = ((0, 1.0), (5, 2.0), (10, 4.0))
    assert [i for i in range(14) if objective_scale_resets("stage", sched, i)] == STARTS
    assert [i for i in range(14) if objective_scale_resets("first", sched, i)] == []
    assert not objective_scale_resets("stage", sched, 0)              # pass 0 sets the scale in both modes: it is not a reset
    # a breakpoint that keeps the beta is no new stage: the stage is the beta, as in `Formulation.stage_note`
    same = ((0, 1.0), (4, 1.0), (8, 2.0))
    assert [i for i in range(12) if objective_scale_resets("stage", same, i)] == [8]
    assert all(objective_scale_resets("stage", sched, i) == (beta_at(sched, i) != beta_at(sched, i - 1)) for i in range(1, 14))


@pytest.mark.parametrize("variant", ["sum", "max", "mass", "stress", "stress_compliance", "robust"])
def test_the_scale_is_set_again_at_each_stage_start_and_not_between(tmp_path, variant):
    """Stage mode: the scale is None before the call at pass 0 and at each stage start, and it is the objective of that pass. Between, it stays."""
    res, calls, hist = run_minimize(case_yaml(tmp_path, variant=variant))
    assert [it for it, reset, _ in calls] == list(range(15))
    assert [it for it, reset, _ in calls if reset] == [0] + STARTS
    scales = {}
    for it, reset, ref in calls:
        if reset:
            assert ref == abs(hist[it]["objective"]["value"]) or ref == 1.0, (it, ref)   # the objective of this pass (the physical value)
            assert ref != scales.get("last"), "a new design gives a new scale"
            scales["last"] = ref
        else:
            assert ref == scales["last"], (it, ref)
    assert len({ref for _it, _reset, ref in calls}) == 3                   # one scale for each of the three stages
    assert all(np.isfinite(r["objective"]["value"]) and np.isfinite(r["compliance"]) for r in hist)


@pytest.mark.parametrize("variant", ["sum", "stress"])
def test_first_mode_keeps_the_scale_of_pass_0(tmp_path, variant):
    res, calls, hist = run_minimize(case_yaml(tmp_path, "first", variant))
    assert [it for it, reset, _ in calls if reset] == [0]
    assert len({ref for _it, _reset, ref in calls}) == 1
    assert calls[0][2] == abs(hist[0]["objective"]["value"])
    assert res.mma.state.f0_ref == calls[0][2]


def test_the_keyword_of_minimize_wins_over_the_setting(tmp_path):
    _res, calls, _h = run_minimize(case_yaml(tmp_path, "stage"), objective_scale="first")
    assert [it for it, reset, _ in calls if reset] == [0]
    _res, calls, _h = run_minimize(case_yaml(tmp_path, "first"), objective_scale="stage")
    assert [it for it, reset, _ in calls if reset] == [0] + STARTS


def test_the_history_is_the_same_in_both_modes_up_to_the_first_stage_start(tmp_path):
    """The scale is not in the history (`objective.value` and `compliance` are physical values, and so is `obj=` in the run log).
    The two modes give the same records up to the pass before the first stage start. At the first stage start the design is the same
    (it comes from the step of the pass before, which used the scale of pass 0), so the physical values are the same too: only `change`,
    the size of the step of that pass, is not. After it the records differ. The stop rules read `objective.value`, never the scale."""
    _r, _c, stage = run_minimize(case_yaml(sub(tmp_path, "a"), "stage"))
    _r, _c, first = run_minimize(case_yaml(sub(tmp_path, "b"), "first"))
    s = STARTS[0]
    clean = lambda r: {k: v for k, v in r.items() if k != "t_iter_s"}
    assert same([clean(r) for r in stage[:s]], [clean(r) for r in first[:s]])
    for key in ("compliance", "compliance_cases", "volume_fraction", "objective", "constraints"):
        assert same(stage[s][key], first[s][key]), key
    assert not same(stage[s]["change"], first[s]["change"])
    assert not same(stage[s + 1]["compliance"], first[s + 1]["compliance"])



def test_one_beta_stage_is_the_same_run_in_both_modes(tmp_path):
    """No stage start after pass 0: nothing is set again, so `stage` is `first`, to the last digit of the solver."""
    def one_stage(folder, scale):
        p = case_yaml(folder, scale, iterations=8)
        d = yaml.safe_load(p.read_text())
        d["settings"]["beta_schedule"] = [[0, 2]]
        p.write_text(yaml.safe_dump(d))
        return p
    _r, calls_s, stage = run_minimize(one_stage(sub(tmp_path, "s"), "stage"))
    _r, calls_f, first = run_minimize(one_stage(sub(tmp_path, "f"), "first"))
    assert [it for it, reset, _ in calls_s if reset] == [0] == [it for it, reset, _ in calls_f if reset]
    assert same([{k: v for k, v in r.items() if k != "t_iter_s"} for r in stage], [{k: v for k, v in r.items() if k != "t_iter_s"} for r in first])

def test_a_mirror_case_sets_the_scale_at_each_stage_start_from_the_averaged_design(tmp_path):
    """Two mirror planes (CONTRACTS.md section 26.1). The stage starts are the passes where the beta changes, as in a case with no mirror.
    The scale of a stage start is the physical objective of that pass: the value that the history has, for the mirror average of x. The design stays symmetric."""
    from topopt import symmetry
    from test_mirror import two_planes, write_yaml
    p = write_yaml(tmp_path, two_planes(), iterations=8, beta_schedule=[[0, 1], [3, 2], [6, 4]])
    assert load_config(p).settings.objective_scale == "stage"
    res, calls, hist = run_minimize(p)
    assert [it for it, reset, _ in calls if reset] == [0, 3, 6]
    last = None
    for it, reset, ref in calls:
        if reset:
            assert ref == abs(hist[it]["objective"]["value"]) and ref != last, (it, ref)
            last = ref
        else:
            assert ref == last, (it, ref)
    model = Model(build_problem(load_config(p)))
    perms = symmetry.perms_of(model)
    model.solver.free()
    assert len(perms) == 2
    assert all(np.abs(res.rho - res.rho[perm]).max() < 1e-12 for perm in perms)


# --- nothing else reads the scale -------------------------------------------------------------------------------

def test_the_physical_objective_in_the_history_does_not_depend_on_the_scale(tmp_path):
    """`Formulation.report` keeps `ev.objective` (physical). A rule that reads the history cannot see a step of the scale."""
    model = Model(build_problem(load_config(case_yaml(tmp_path))))
    form = formulation_from_config(model.problem.config, model.n_design, model)
    x = np.full(model.n_design, 0.3)
    rho_tilde, rho = model.density(x, 1.0)
    point = model.design_point(x, 1.0, rho_tilde, rho, model.solve(model.stiffness(rho)))
    from topopt.mma import MmaState
    objs = []
    for ref in (1.0, 123.0):
        state = MmaState(model.n_design)
        state.f0_ref = ref
        ev = form.evaluate(point, state)
        objs.append((ev.objective, ev.constraints, form.report(ev)))
        assert ev.f0 == pytest.approx(ev.objective / ref) and ev.f0_ref == ref        # the scaled value is only in f0 and df0
    assert same(objs[0], objs[1], rel=1e-12)       # the same design point: the same values, whatever the scale
    model.solver.free()



# --- the checkpoint ---------------------------------------------------------------------------------------------

def test_the_effective_settings_leave_out_stage_and_keep_first(tmp_path):
    """The rule of PR #72 for a changed default: the key is left out at `stage`, so a checkpoint of a build before P3a resumes (the resume tests are in
    test_objective_scale_resume.py). `first` is the old behavior and stays in: a change from or to it is a change of the inputs."""
    from topopt import checkpoint as C
    now = C.input_hashes(load_config(case_yaml(tmp_path)))
    assert "objective_scale" not in now["effective"]["settings"]
    explicit = C.input_hashes(load_config(case_yaml(sub(tmp_path, "s"), "stage")))
    assert explicit["effective"] == now["effective"] and explicit["effective_sha256"] == now["effective_sha256"]       # `stage` in the YAML: the same effective settings
    first = C.input_hashes(load_config(case_yaml(sub(tmp_path, "f"), "first")))
    assert first["effective"]["settings"]["objective_scale"] == "first" and first["effective_sha256"] != now["effective_sha256"]
    # the key in the YAML is a change of the parsed YAML, as any other key
    changes, _notes = C.diff_inputs(now, first)
    assert changes == ["settings.objective_scale: <not set> -> first"]
    assert C.diff_inputs(first, now)[0] == ["settings.objective_scale: first -> <not set>"]


# --- the regression: the volume comes back after the last beta jump ------------------------------------------------

def stand_yaml(tmp_path, scale=None, h=2.0):
    """A phone stand: a 120 x 15 x 60 mm triangle, 2 N down on the slope, supported on the desk face. Penal 6, volume fraction 0.0729.

    The CPU copy of a stand from tests with a GPU (5103 design voxels, 6307 active voxels at 2 mm). Those tests had
    about 200 000 active voxels, where the scaled objective gradient is smaller still; the mechanism is the same.
    """
    def hull(pts, name):
        trimesh.convex.convex_hull(np.array(pts, float)).export(str(tmp_path / name))
    hull([(x, y, z) for y in (0, 15) for (x, z) in ((0, 0), (120, 0), (0, 60))], "design.stl")
    hull([(x, y, z) for y in (0, 15) for x in (0, 120) for z in (0, h)], "desk_face.stl")
    t = 2 * h
    hull([(x, y, z) for y in (0, 15) for (x, z) in ((0, 60), (0, 60 - t), (120, 0), (120 - 2 * t, 0))], "phone_surface.stl")
    d = {"name": "stand", "regions": {"design": "design.stl", "fixed": ["desk_face.stl"], "load": {"phone_surface": "phone_surface.stl"}},
         "load_cases": [{"name": "phone_weight", "forces": [{"region": "phone_surface", "force": [0, 0, -2.0]}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.0729, "iterations": 52, "filter_radius": 1.2 * h, "penal": 6.0,
                      "beta_schedule": [[0, 1], [16, 2], [28, 4], [40, 8]]}}
    if scale is not None:
        d["settings"]["objective_scale"] = scale
    p = tmp_path / "stand.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def passes_to_reach(hist, jump, target, limit):
    """The number of passes after the pass `jump` (the first pass of the last beta stage) until the volume is at `target` times the limit."""
    for r in hist[jump:]:
        if r["volume_fraction"] / limit >= target:
            return r["iteration"] - jump
    return None


@pytest.fixture(scope="module")
def stand_runs(tmp_path_factory):
    out = {}
    for scale in ("stage", "first"):
        res, _calls, hist = run_minimize(stand_yaml(tmp_path_factory.mktemp("stand_" + scale), scale))
        out[scale] = hist
    return out


def test_regression_stage_brings_the_volume_back_after_the_last_jump_and_first_does_not(stand_runs):
    """The user tests ended with 0.55 to 0.88 of the allowed volume. On this stand the volume is at 0.73 of the limit after the last jump
    (pass 40) in mode "first", and it needs 5 passes to be at 0.95. In mode "stage" the volume is at 0.94 at the jump and above 0.95 one pass later."""
    limit = 0.0729
    stage, first = stand_runs["stage"], stand_runs["first"]
    assert first[40]["volume_fraction"] / limit < 0.8                       # the jump takes the volume down in both modes
    assert stage[40]["volume_fraction"] / limit > 0.85
    n_stage, n_first = passes_to_reach(stage, 40, 0.95, limit), passes_to_reach(first, 40, 0.95, limit)
    assert n_stage is not None and n_stage <= 2
    assert n_first is None or n_first >= 4
    assert n_first is None or n_first >= n_stage + 3
    # and both end at the limit: the end of the run does not show the difference, the passes after the jump do
    assert stage[-1]["volume_fraction"] / limit == pytest.approx(1.0, abs=0.02)
    assert abs(stage[-1]["volume_fraction"] / limit - 1.0) <= abs(first[-1]["volume_fraction"] / limit - 1.0) + 1e-3


def test_regression_the_earlier_jumps_and_pass_0_are_the_same_in_both_modes_up_to_the_first_stage_start(stand_runs):
    stage, first = stand_runs["stage"], stand_runs["first"]
    assert same([r["compliance"] for r in stage[:17]], [r["compliance"] for r in first[:17]])    # passes 0 to 16: the first stage, and the design at its end
    assert not same(stage[17]["compliance"], first[17]["compliance"])
