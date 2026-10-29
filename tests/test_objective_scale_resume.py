"""Resume and `settings.objective_scale` (P3a review, F1; CONTRACTS.md section 13, "Resume").

One rule for a changed default of topopt, the rule of PR #72 for `gpu_tol` and `gpu_backward_error`: the key is left out of the effective settings at its
default `stage`, so a checkpoint of a build before P3a resumes. The resume says in a `resume:` fact that the run goes on with `stage` and that
`settings.objective_scale: first` keeps the old behavior. `first` stays in the effective settings, so a change from or to it needs the override.
A checkpoint of this build records the value in `meta["objective_scale"]`; a checkpoint with no such key is from before P3a.
"""

import copy
import hashlib
import json

import pytest

from topopt import checkpoint as C
from topopt.config import load_config
from topopt.problem import build_problem
from topopt.solver_choice import resolve_solver

from test_checkpoint import clean, finish, history, killed, log, needs_direct, resume
from test_objective_scale import case_yaml, sub


def as_written_by_the_old_build(run_dir, yaml_without_key):
    """Rewrite the checkpoint of `run_dir` as the build before P3a wrote it: no `objective_scale` in meta, and the inputs of a YAML with no such key."""
    ck = C.read_checkpoint(run_dir)
    meta = copy.deepcopy(ck.meta)
    meta.pop("objective_scale", None)
    cfg = load_config(yaml_without_key)
    inputs = C.input_hashes(cfg, build_problem(cfg), resolve_solver("auto"))
    inputs["effective"]["settings"].pop("objective_scale", None)       # the old build has no such setting (this build leaves `stage` out too)
    inputs["effective_sha256"] = hashlib.sha256(json.dumps(inputs["effective"], sort_keys=True).encode()).hexdigest()
    meta["inputs"] = inputs
    C.write_checkpoint(run_dir, ck.arrays, meta)


@pytest.fixture(scope="module")
def whole_runs(tmp_path_factory):
    """The whole runs in child processes with one thread: the default (`stage`) and `first`. The first part of an old run is a `first` run."""
    base = tmp_path_factory.mktemp("whole")
    out = {}
    for name, scale in (("stage", None), ("first", "first")):
        p = case_yaml(sub(base, name + "_in"), scale)
        finish(p, base / name)
        out[name] = (p, base / name)
    return out


@needs_direct
@pytest.mark.parametrize("scale", [None, "first"])
def test_the_checkpoint_records_the_value_of_the_setting(tmp_path, scale):
    """A checkpoint of this build has `meta["objective_scale"]`, also at the default (the effective settings leave `stage` out). One without it is from before P3a."""
    out = tmp_path / "out"
    killed(case_yaml(sub(tmp_path, "in"), scale), out, "after_checkpoint:3")
    assert C.read_checkpoint(out).meta["objective_scale"] == (scale or "stage")


@needs_direct
def test_a_checkpoint_of_a_build_before_the_key_resumes_with_a_fact_and_goes_on_with_stage(tmp_path, whole_runs):
    """The first part is a `first` run (the old build), killed after pass 7. The resume with no key in the YAML needs no override and says what changes."""
    p_first, _ = whole_runs["first"]
    p_new = case_yaml(sub(tmp_path, "in"))                       # the same case, no key: the default `stage`
    cut = tmp_path / "cut"
    killed(p_first, cut, "after_checkpoint:7")
    as_written_by_the_old_build(cut, p_new)
    plan = C.plan_resume(cut, load_config(p_new))                # no refusal, no changed input
    assert plan.changes == []
    fact = next(f for f in plan.facts if "settings.objective_scale existed" in f)
    assert fact.startswith("resume: the checkpoint was written before settings.objective_scale existed (P3a).")
    assert "settings.objective_scale: stage, the default" in fact and "settings.objective_scale: first in the YAML keeps the old behavior" in fact
    resume(p_new, cut)
    assert fact in log(cut)
    old, new = history(whole_runs["first"][1]), history(cut)
    assert len(old) == len(new) == 15
    assert clean(old[:10]) == clean(new[:10])        # up to the stage start at pass 10 the scale of pass 0 is in force, as in the old run
    assert old[10]["compliance"] == new[10]["compliance"] and old[10]["change"] != new[10]["change"]      # pass 10: the same design, but its step takes the new scale
    assert old[11]["compliance"] != new[11]["compliance"]        # so `stage` applies from the next stage start on


@needs_direct
def test_a_checkpoint_of_a_build_before_the_key_resumed_at_a_stage_start_is_the_stage_run(tmp_path, whole_runs):
    """Killed after pass 4, so the resume starts at the first pass of the stage at 5: the pass number gives the rule, and the run is the default run."""
    p_first, _ = whole_runs["first"]
    p_new, out_stage = whole_runs["stage"]
    cut = tmp_path / "cut"
    killed(p_first, cut, "after_checkpoint:4")
    as_written_by_the_old_build(cut, p_new)
    resume(p_new, cut)
    assert clean(history(cut)) == clean(history(out_stage))
    assert "resume: the checkpoint was written before settings.objective_scale existed (P3a)." in log(cut)


@needs_direct
def test_the_same_checkpoint_with_first_in_the_yaml_needs_the_override_and_is_then_the_old_run(tmp_path, whole_runs):
    p_first, out_first = whole_runs["first"]
    p_new = case_yaml(sub(tmp_path, "in"))
    cut = tmp_path / "cut"
    killed(p_first, cut, "after_checkpoint:7")
    as_written_by_the_old_build(cut, p_new)
    with pytest.raises(C.ResumeRefused) as exc:
        C.plan_resume(cut, load_config(p_first))
    assert exc.value.facts == ["settings.objective_scale: <not set> -> first"]
    assert any("the checkpoint is from before settings.objective_scale" in o and "--allow-changed-inputs to resume with first" in o for o in exc.value.options)
    resume(p_first, cut, "--allow-changed-inputs")
    assert clean(history(cut)) == clean(history(out_first))      # the old behavior: the same run as one that was not stopped
    assert "resume: the checkpoint was written before settings.objective_scale existed (P3a)." in log(cut)
    assert "settings.objective_scale: first does the same, so the run goes on as the old build did" in log(cut)


@needs_direct
@pytest.mark.parametrize("written", [None, "stage"])
def test_a_checkpoint_with_first_refuses_a_change_to_stage_with_a_fact_and_options(tmp_path, whole_runs, written):
    """`first` is in the effective settings. The key taken out of the YAML, or `stage` written in it, is a change from the checkpoint."""
    p_first, _ = whole_runs["first"]
    cut = tmp_path / "cut"
    killed(p_first, cut, "after_checkpoint:7")
    p_new = case_yaml(sub(tmp_path, "in"), written)
    with pytest.raises(C.ResumeRefused) as exc:
        C.plan_resume(cut, load_config(p_new))
    line = f"settings.objective_scale: first -> {written or '<not set>'}"
    assert exc.value.facts == [line] and line in exc.value.reason
    assert any("settings.objective_scale: first in the YAML keeps the scale rule of the checkpoint" in o and "the scale of the checkpoint stays until the next beta stage start" in o
               for o in exc.value.options)
    plan = C.plan_resume(cut, load_config(p_new), allow_changed=True)
    assert plan.changes == [line] and not any("settings.objective_scale existed" in f for f in plan.facts)    # a checkpoint of this build: no "before P3a" fact


@needs_direct
def test_a_checkpoint_of_this_build_with_the_default_resumes_with_no_fact(tmp_path, whole_runs):
    p_new, out_stage = whole_runs["stage"]
    cut = tmp_path / "cut"
    killed(p_new, cut, "after_checkpoint:7")
    plan = C.plan_resume(cut, load_config(p_new))
    assert plan.changes == [] and not any("objective_scale" in f for f in plan.facts)
    resume(p_new, cut)
    assert clean(history(cut)) == clean(history(out_stage))


def test_the_checkpoint_of_a_finished_run_of_an_old_build_has_no_fact(tmp_path, whole_runs):
    """No pass is left, so the fact about the scale would say nothing: a complete checkpoint gets none."""
    p_new, out_stage = whole_runs["stage"]
    done = tmp_path / "done"
    done.mkdir()
    for f in out_stage.iterdir():
        if f.is_file():
            (done / f.name).write_bytes(f.read_bytes())
    as_written_by_the_old_build(done, p_new)
    plan = C.plan_resume(done, load_config(p_new))
    assert plan.changes == [] and not any("settings.objective_scale existed" in f for f in plan.facts)
