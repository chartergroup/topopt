"""Tests of `topopt run --fine-from PARENT` and `--fine-pass`, and of `topopt check` for a fine pass (#318, CONTRACTS.md section 24.6).

The case is the cantilever of tests/test_optimize.py: the plate and the tip load are 2 mm thick, so the part needs a voxel of 0.7 mm (2.0 / 2.8 = 0.714,
rounded down to 0.05 mm: MEASURED_MM) and a run at 2 mm voxels is too coarse. A plain check and the end of a run state that value (CONTRACTS.md section
24.6). A fine pass tests it on its own grid: the grid at 0.7 mm holds only 2 rows of the tip load (2 mm wide, x from 18 to 20, voxel centres at 18.65 and
19.35), so the voxel goes down one step to PART_MM = 0.65 mm, and the log says so.
The parent runs at 2 mm (tiny: 275 elements). Most fine passes use `--fine-voxel 1.0` to stay small; one test takes the voxel
that the part needs (PART_MM). The runs are in this process; the kill and the resume are child processes with one thread (tests/test_checkpoint.py).
"""
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from topopt import fine_pass as F
from topopt.config import load_config
from topopt.cli import fine_folder, main
from test_checkpoint import DIRECT, edit, finish, history, killed, log, same_run, run_args
from test_optimize import make_case

PARENT_PASSES = 6
MEASURED_MM = 0.7     # the voxel of the part for the cantilever at 2 mm, as measured: what a plain check and the end of a run say
PART_MM = 0.65        # the voxel that the fine pass takes: the measured one, tested on the grid of the fine pass (one step down)
CHECK_FACT = "On the grid of the fine pass at that voxel the region 'tip' has too few voxels across"      # the fact of the step down


def read_json(path):
    return json.loads(Path(path).read_text())


def state_of(d):
    """Every file and folder of `d`: the kind, the modification time in ns and the bytes of a file. A parent that was only read has the same state."""
    d = Path(d)
    out = {}
    for p in [d] + sorted(d.rglob("*")):
        st = p.lstat()
        out[str(p.relative_to(d))] = (stat.S_ISDIR(st.st_mode), st.st_mtime_ns, None if stat.S_ISDIR(st.st_mode) else p.read_bytes())
    return out


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    """The finished run at 2 mm voxels with 6 passes: `(yaml, run folder)`. The YAML is alone in its folder, with its STL files."""
    root = tmp_path_factory.mktemp("fine_cli")
    (root / "case").mkdir()
    yaml_path = make_case(root / "case", h=2.0, iterations=PARENT_PASSES)
    out = root / "parent"
    assert main(run_args(yaml_path, out)) == 0
    return yaml_path, out


def fine_run(parent, out, *flags):
    return main(run_args(parent[0], out, "--fine-from", str(parent[1]), *flags))


def refused(capsys, args):
    """The CLI with `args`: exit code 1 and one error on stderr. Return the text of the error."""
    assert main(args) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:"), err
    return err


# --- a fine pass --------------------------------------------------------------------------------------

def test_a_fine_pass_takes_the_voxel_of_the_part_and_the_pass_count_of_the_parent(parent, tmp_path):
    before = state_of(parent[1])
    out = tmp_path / "fine"
    assert fine_run(parent, out) == 0
    assert state_of(parent[1]) == before                       # the parent is only read
    run = read_json(out / "run.json")
    rec = run["fine_from"]
    assert run["h"] == pytest.approx(PART_MM) and run["iterations"] == PARENT_PASSES
    assert rec["run_dir"] == str(parent[1].resolve()) and rec["iteration"] == PARENT_PASSES and rec["parent_voxel_mm"] == 2.0
    assert rec["voxel_mm"] == pytest.approx(PART_MM) and rec["voxel_from"] == "part_voxel" and rec["part_voxel_mm"] == pytest.approx(PART_MM)
    assert rec["limiting"] in ("plate", "tip") and rec["limiting_role"] in ("fixed", "load") and rec["limiting_thickness_mm"] == pytest.approx(2.0, abs=1e-3)
    assert rec["grid_check"] == {"voxel_mm": PART_MM, "start_mm": MEASURED_MM, "steps": 1, "region": "tip", "ok": True, "skipped": False}   # one record, no band
    assert not {"part_voxel_level", "part_voxel_bands"} & set(rec)
    assert rec["max_passes"] == PARENT_PASSES and rec["max_from"] == "parent"                       # the default maximum: the pass count of the parent
    assert rec["tolerance"] == F.FINE_OBJECTIVE_CHANGE and rec["tolerance_from"] == "default" and rec["stop_rule"] == "settled"
    assert rec["schedule_offset"] == PARENT_PASSES and rec["source"] == "x" and rec["window"] == 5
    assert rec["map"]["points"] > 0 and rec["volume"]["mean_after"] == pytest.approx(rec["volume"]["target"], abs=1e-6)
    assert rec["checkpoint"] == "checkpoint.npz" and len(rec["checkpoint_sha256"]) == 64
    verdict = read_json(out / "verdict.json")
    assert verdict["stop_rules"]["enabled"] and verdict["stop_rules"]["objective_change"] == F.FINE_OBJECTIVE_CHANGE
    text = log(out)
    assert "fine pass: new run from the design of" in text and "the voxel that the part needs" in text
    assert CHECK_FACT in text and "so the voxel goes down 1 step of 0.05 mm to 0.65 mm, where every tested region passes" in text      # the test on the grid says so
    assert f"after {PARENT_PASSES} passes at most (the pass count of the parent run)" in text
    assert "fine pass: start design from pass 6" in text and "fitted to the volume target" in text


def test_the_beta_schedule_goes_on_from_the_parent(parent, tmp_path):
    out = tmp_path / "fine"
    assert fine_run(parent, out, "--fine-voxel", "1.0") == 0
    betas = [r["beta"] for r in history(out)]
    assert betas[0] == 1.0 and betas[3] == 1.0 and betas[4] == 2.0       # the parent had 6 passes: the stage of beta 2 starts at pass 10 there, pass 4 here
    assert "fact: settings.iterations is" not in log(out)               # a fine pass has a maximum of passes: no plan of the beta stages


def test_the_loss_settles_before_the_maximum(parent, tmp_path):
    out = tmp_path / "fine"
    assert fine_run(parent, out, "--fine-voxel", "1.0", "--fine-passes", "40", "--fine-tol", "0.5") == 0
    verdict = read_json(out / "verdict.json")
    assert verdict["state"] == "stopped_by_rule" and verdict["stop"]["rule"] == "settled"
    last = verdict["final_iteration"]
    assert 9 <= last < 40                                                # the stage of beta 2 starts at pass 4: the window of 5 needs passes 4 to 9
    assert history(out)[-1]["iteration"] == last
    text = log(out)
    assert f"stop rule 'settled' at pass {last}: the loss has settled" in text and "tolerance 0.5" in text
    assert f"fine pass: the loss settled at pass {last} (the maximum is 40 passes)" in text
    rec = read_json(out / "run.json")["fine_from"]
    assert rec["tolerance"] == 0.5 and rec["tolerance_from"] == "option" and rec["max_passes"] == 40 and rec["max_from"] == "fine_passes"
    ck = read_json_checkpoint_meta(out)
    assert ck["complete"] and ck["stop"]["rule"] == "settled" and ck["fine_from"]["tolerance"] == 0.5


def read_json_checkpoint_meta(run_dir):
    from topopt.checkpoint import read_checkpoint
    return read_checkpoint(run_dir).meta


def test_the_maximum_ends_the_fine_pass_before_the_loss_settles_and_the_log_says_so(parent, tmp_path):
    out = tmp_path / "fine"
    assert fine_run(parent, out, "--fine-voxel", "1.0", "--fine-passes", "3") == 0
    verdict = read_json(out / "verdict.json")
    assert verdict["state"] == "iteration_limit" and verdict["final_iteration"] == 3 and read_json(out / "run.json")["iterations"] == 3
    text = log(out)
    assert "fine pass: the maximum of 3 passes ended the fine pass before the loss settled" in text
    assert "not judged: it needs 6 passes in a row at one beta" in text and "higher --fine-passes" in text


def test_no_stop_rules_switches_off_the_early_stop_of_a_fine_pass_and_it_runs_to_its_maximum(parent, tmp_path):
    out = tmp_path / "fine"
    flags = ("--fine-voxel", "1.0", "--fine-passes", "12", "--fine-tol", "0.5")
    assert fine_run(parent, out, *flags, "--no-stop-rules") == 0
    verdict = read_json(out / "verdict.json")
    assert verdict["state"] == "iteration_limit" and verdict["final_iteration"] == 12 and not verdict["stop_rules"]["enabled"]
    rec = read_json(out / "run.json")["fine_from"]
    assert rec["stop_rule"] is None and rec["max_passes"] == 12 and rec["tolerance"] == 0.5
    assert history(out)[-1]["iteration"] == 12 and read_json_checkpoint_meta(out)["fine_from"]["stop_rule"] is None
    text = log(out)
    assert "All stop rules are off (--no-stop-rules): it runs 12 passes" in text
    assert "fine pass: all stop rules were off (--no-stop-rules), so the fine pass ran its 12 passes." in text
    assert "does not switch off" not in text
    on = tmp_path / "fine-on"
    assert fine_run(parent, on, *flags) == 0                                       # the same command with the rules on settles before the maximum
    assert read_json(on / "verdict.json")["stop"]["rule"] == "settled" and read_json(on / "verdict.json")["final_iteration"] < 12


def test_a_fine_pass_can_be_the_parent_of_the_next_and_the_offsets_add_up(parent, tmp_path):
    first, second = tmp_path / "f1", tmp_path / "f2"
    assert fine_run(parent, first, "--fine-voxel", "1.0", "--fine-passes", "4") == 0
    assert main(run_args(parent[0], second, "--fine-from", str(first), "--fine-voxel", "0.8", "--fine-passes", "3")) == 0
    rec = read_json(second / "run.json")["fine_from"]
    assert rec["iteration"] == 4 and rec["schedule_offset"] == 4 + PARENT_PASSES and rec["parent_voxel_mm"] == 1.0
    assert history(second)[0]["beta"] == 2.0 and rec["max_passes"] == 3
    # the maximum of the next fine pass, when nothing is given, is the pass count of ITS parent (the first fine pass: 4)
    third = tmp_path / "f3"
    assert main(run_args(parent[0], third, "--fine-from", str(first), "--fine-voxel", "0.8")) == 0
    assert read_json(third / "run.json")["fine_from"]["max_passes"] == 4


def continue_args(yaml_path, out, parent, n, *extra):
    return run_args(yaml_path, out, "--continue-from", str(parent), "--more-iterations", str(n), *extra)


def test_a_continuation_of_a_fine_pass_keeps_its_voxel_schedule_and_rule_with_the_same_yaml(parent, tmp_path):
    first, cont, full = tmp_path / "f1", tmp_path / "f2", tmp_path / "full"
    assert fine_run(parent, first, "--fine-voxel", "1.0", "--fine-passes", "4") == 0
    assert main(continue_args(parent[0], cont, first, 5)) == 0                 # the YAML is the one of the first run: voxel 2 mm, 6 passes, beta 1 to 4
    run = read_json(cont / "run.json")
    assert run["h"] == 1.0 and run["iterations"] == 9 and run["continued_from"]["run_dir"] == str(first.resolve())
    assert run["fine_from"] == read_json(first / "run.json")["fine_from"]       # both records are kept, the fine one unchanged
    assert read_json_checkpoint_meta(cont)["fine_from"] == run["fine_from"] and read_json_checkpoint_meta(cont)["complete"]
    verdict = read_json(cont / "verdict.json")
    assert verdict["stop_rules"]["enabled"] and verdict["stop_rules"]["objective_change"] == F.FINE_OBJECTIVE_CHANGE
    betas = [r["beta"] for r in history(cont)]
    assert betas[:4] == [1.0] * 4 and set(betas[4:]) == {2.0}                   # the beta of the fine pass goes on: it does not start again at 1
    text = log(cont)
    assert "continue: the parent is a fine pass from" in text and "It keeps the stop rule 'settled'" in text
    assert "fine pass: all stop rules were off" not in text
    assert fine_run(parent, full, "--fine-voxel", "1.0", "--fine-passes", "9", "--no-stop-rules") == 0     # the run of the full length
    for a, b in zip(history(full), history(cont)):
        assert (a["iteration"], a["beta"]) == (b["iteration"], b["beta"]) and b["compliance"] == pytest.approx(a["compliance"], rel=1e-6)


def test_a_continuation_of_a_settled_fine_pass_is_judged_again_and_goes_on_with_no_stop_rules(parent, tmp_path):
    first, cont, more = tmp_path / "f1", tmp_path / "f2", tmp_path / "f3"
    flags = ("--fine-voxel", "1.0", "--fine-passes", "40", "--fine-tol", "0.5")
    assert fine_run(parent, first, *flags) == 0
    last = read_json(first / "verdict.json")["final_iteration"]
    assert read_json(first / "verdict.json")["stop"]["rule"] == "settled"
    assert main(continue_args(parent[0], cont, first, 3)) == 0                  # the rule is on and the tolerance is the one of the fine pass: it settles again at pass K
    v = read_json(cont / "verdict.json")
    assert v["stop"]["rule"] == "settled" and v["final_iteration"] == last and v["stop_rules"]["objective_change"] == 0.5
    assert "judged again at pass" in log(cont)
    assert main(continue_args(parent[0], more, first, 3, "--no-stop-rules")) == 0
    v = read_json(more / "verdict.json")
    assert v["state"] == "iteration_limit" and v["final_iteration"] == last + 3 and not v["stop_rules"]["enabled"]
    betas = [r["beta"] for r in history(more)]
    assert min(betas[4:]) == 2.0 and betas[-1] == 2.0                           # the beta does not drop to the first stage of the YAML
    assert read_json(more / "run.json")["fine_from"]["stop_rule"] == "settled"  # the record of the fine pass is the parent's; this run has the rules off for its own passes
    assert "fine pass: all stop rules were off" in log(more)


def test_a_continuation_of_a_continuation_of_a_fine_pass_goes_on_with_the_voxel_and_the_schedule(parent, tmp_path):
    first, c1, c2 = tmp_path / "f1", tmp_path / "c1", tmp_path / "c2"
    assert fine_run(parent, first, "--fine-voxel", "1.0", "--fine-passes", "3") == 0
    assert main(continue_args(parent[0], c1, first, 2)) == 0
    assert main(continue_args(parent[0], c2, c1, 3)) == 0
    run = read_json(c2 / "run.json")
    assert run["h"] == 1.0 and run["iterations"] == 8 and run["continued_from"]["run_dir"] == str(c1.resolve())
    assert run["fine_from"]["run_dir"] == str(parent[1].resolve()) and run["fine_from"]["max_passes"] == 3
    betas = [r["beta"] for r in history(c2)]
    assert betas[:4] == [1.0] * 4 and set(betas[4:]) == {2.0}


def test_a_killed_continuation_of_a_fine_pass_resumes_with_the_voxel_and_the_pass_count_of_its_checkpoint(parent, tmp_path):
    first, cont, ref = tmp_path / "f1", tmp_path / "c1", tmp_path / "ref"
    assert fine_run(parent, first, "--fine-voxel", "1.0", "--fine-passes", "3") == 0
    assert main(continue_args(parent[0], ref, first, 4)) == 0
    killed(parent[0], cont, "after_checkpoint:5", "--continue-from", str(first), "--more-iterations", "4")
    finish(parent[0], cont, "--resume")
    assert read_json(cont / "run.json")["h"] == 1.0 and read_json(cont / "run.json")["iterations"] == 7
    assert [r["beta"] for r in history(cont)] == [r["beta"] for r in history(ref)]
    for a, b in zip(history(ref), history(cont)):
        assert b["compliance"] == pytest.approx(a["compliance"], rel=1e-6)


def test_a_mass_run_is_fitted_to_the_final_density_of_the_parent_and_not_to_its_start_density(tmp_path):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=2.0, iterations=20)
    edit(yaml_path, lambda d: (d["settings"].update(objective="mass", volume_fraction=0.5),
                               d["load_cases"][0].update(compliance_limit=0.15), d["load_cases"][1].update(compliance_limit=0.3)))
    first, out = tmp_path / "first", tmp_path / "fine"
    assert main(run_args(yaml_path, first)) == 0
    final = history(first)[-1]["volume_fraction"]
    assert abs(final - 0.5) > 0.1                                              # the mass run left its start density
    assert main(run_args(yaml_path, out, "--fine-from", str(first), "--fine-voxel", "1.0", "--fine-passes", "2")) == 0
    vol = read_json(out / "run.json")["fine_from"]["volume"]
    assert vol["target_from"] == "parent" and vol["target"] == pytest.approx(final, abs=1e-6) and vol["mean_after"] == pytest.approx(final, abs=1e-4)
    assert abs(history(out)[0]["volume_fraction"] - final) < 0.01
    text = log(out)
    assert "fitted to the final mean density of the parent" in text and "settings.volume_fraction is only its start density" in text
    assert "fitted to the volume target" not in text


def yaml_with_tolerance(parent, tmp_path, value):
    """A copy of the case YAML (with its STL files) that has `settings.fine_pass.objective_change`: the parent run is of the YAML without it."""
    shutil.copytree(parent[0].parent, tmp_path / "case2")
    path = tmp_path / "case2" / parent[0].name
    edit(path, lambda d: d["settings"].update(fine_pass={"objective_change": value}))
    return path


def test_the_yaml_gives_the_tolerance_and_the_record_shows_the_source(parent, tmp_path):
    path = yaml_with_tolerance(parent, tmp_path, 0.5)
    out = tmp_path / "fine"
    assert main(run_args(path, out, "--fine-from", str(parent[1]), "--fine-voxel", "1.0", "--fine-passes", "40")) == 0      # no --fine-tol, and the YAML differs from the parent's: a note
    rec = read_json(out / "run.json")["fine_from"]
    assert rec["tolerance"] == 0.5 and rec["tolerance_from"] == "yaml"
    verdict = read_json(out / "verdict.json")
    assert verdict["stop"]["rule"] == "settled" and verdict["stop_rules"]["objective_change"] == 0.5 and verdict["final_iteration"] < 40
    text = log(out)
    assert "(objective change at or below 0.5, settings.fine_pass.objective_change, over 5 passes at one beta)" in text
    assert "settings.fine_pass.objective_change:" in text and "this does not change the numbers of a pass" in text      # the YAML differs from the parent's: a note
    assert read_json_checkpoint_meta(out)["fine_from"]["tolerance_from"] == "yaml"


def test_the_option_wins_over_the_yaml_and_the_default_is_the_source_without_both(parent, tmp_path):
    path = yaml_with_tolerance(parent, tmp_path, 0.5)
    out, plain = tmp_path / "fine", tmp_path / "plain"
    assert main(run_args(path, out, "--fine-from", str(parent[1]), "--fine-voxel", "1.0", "--fine-passes", "3", "--fine-tol", "0.7")) == 0
    rec = read_json(out / "run.json")["fine_from"]
    assert rec["tolerance"] == 0.7 and rec["tolerance_from"] == "option" and "--fine-tol" in log(out)
    assert fine_run(parent, plain, "--fine-voxel", "1.0", "--fine-passes", "3") == 0
    rec = read_json(plain / "run.json")["fine_from"]
    assert rec["tolerance"] == F.FINE_OBJECTIVE_CHANGE and rec["tolerance_from"] == "default"


def test_a_resume_and_a_continuation_keep_the_tolerance_of_the_record_and_not_the_yaml(parent, tmp_path):
    path = yaml_with_tolerance(parent, tmp_path, 0.5)
    first, cont = tmp_path / "f1", tmp_path / "f2"
    assert main(run_args(path, first, "--fine-from", str(parent[1]), "--fine-voxel", "1.0", "--fine-passes", "4")) == 0
    edit(path, lambda d: d["settings"].update(fine_pass={"objective_change": 0.01}))                  # another value later: a note, not a change of the inputs
    assert main(continue_args(path, cont, first, 2)) == 0
    assert read_json(cont / "verdict.json")["stop_rules"]["objective_change"] == 0.5
    rec = read_json(cont / "run.json")["fine_from"]
    assert rec["tolerance"] == 0.5 and rec["tolerance_from"] == "yaml"
    assert main(run_args(path, first, "--resume")) == 0                                              # a resume of the finished fine pass: no refusal
    assert read_json(first / "run.json")["fine_from"]["tolerance"] == 0.5


def test_fine_pass_takes_the_yaml_tolerance_in_the_one_command_form_and_check_shows_it(parent, tmp_path, capsys):
    path = yaml_with_tolerance(parent, tmp_path, 0.5)
    out = tmp_path / "run"
    assert main(run_args(path, out, "--fine-pass", "--fine-voxel", "1.0", "--fine-passes", "40")) == 0
    assert read_json(fine_folder(out) / "run.json")["fine_from"]["tolerance_from"] == "yaml"
    assert main(["check", str(path), "--fine-from", str(parent[1]), "--fine-voxel", "1.0"]) == 0
    assert "settings.fine_pass.objective_change" in capsys.readouterr().out


def test_a_fine_pass_setting_with_a_wrong_value_stops_the_run_before_anything_is_written(parent, tmp_path, capsys):
    path = yaml_with_tolerance(parent, tmp_path, -1)
    out = tmp_path / "fine"
    err = refused(capsys, run_args(path, out, "--fine-from", str(parent[1]), "--fine-voxel", "1.0"))
    assert "'settings.fine_pass.objective_change' must be a finite number above 0" in err and "Example:" in err and not out.exists()


def limited_case(tmp_path, limit=2000):
    """The cantilever at 2 mm with `settings.max_active_voxels` = `limit`: the fine grid of the part (about 8000 active voxels at PART_MM) is above it."""
    (tmp_path / "limited").mkdir()
    path = make_case(tmp_path / "limited", h=2.0, iterations=2)
    edit(path, lambda d: d["settings"].update(max_active_voxels=limit))
    return path


def test_check_says_that_the_fine_grid_fails_a_guard_and_gives_the_finest_voxel_that_passes(tmp_path, capsys):
    path = limited_case(tmp_path)
    assert main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("fine pass: the part needs"))
    # the measured voxel is tested (one value, no band); the finest voxel that passes does not depend on it
    assert f"the fine grid at {MEASURED_MM:g} mm fails settings.max_active_voxels: about {round(275 * (2 / MEASURED_MM) ** 3)} active voxels, the limit is 2000" in line
    assert "lower end" not in line and "band" not in line
    assert "--fine-voxel 1.1 (MCP fine_voxel) is the finest voxel that passes" in line and "raise settings.max_active_voxels in the YAML" in line
    assert main(["check", str(path), "--json"]) == 0
    guards = json.loads(capsys.readouterr().out)["fine_pass"]["guards"]
    assert guards["ok"] is False and guards["voxel_pass_mm"] == 1.1 and guards["failed"][0]["guard"] == "settings.max_active_voxels"
    assert guards["failed"][0]["limit"] == 2000


def test_fine_pass_refuses_before_the_first_run_when_the_fine_grid_fails_a_guard(tmp_path, capsys):
    path = limited_case(tmp_path)
    out = tmp_path / "run"
    err = refused(capsys, run_args(path, out, "--fine-pass"))
    assert "--fine-pass stops before the first run" in err and "--fine-voxel 1.1" in err and "settings.max_active_voxels" in err
    # the pre-check takes the voxel that the fine pass would take: the measured one, tested on the grid of the fine pass
    assert f"the fine grid at {PART_MM:g} mm fails settings.max_active_voxels: about {round(275 * (2 / PART_MM) ** 3)} active voxels" in err
    assert not out.exists() and not fine_folder(out).exists()                  # nothing is written
    err = refused(capsys, run_args(path, out, "--fine-pass", "--fine-voxel", "0.8"))      # the voxel of the user is tested too
    assert "the fine grid at 0.8 mm fails settings.max_active_voxels" in err and not out.exists()
    assert main(run_args(path, out, "--fine-pass", "--fine-voxel", "1.1", "--fine-passes", "1")) == 0     # the finest voxel that passes starts both runs
    assert read_json(fine_folder(out) / "run.json")["fine_from"]["voxel_mm"] == 1.1


def test_the_memory_estimate_above_the_ram_fails_the_fine_grid_guard(tmp_path, capsys, monkeypatch):
    (tmp_path / "case").mkdir()
    path = make_case(tmp_path / "case", h=2.0, iterations=2)
    monkeypatch.setattr("topopt.cli.physical_ram_bytes", lambda: 400 * 2**20)         # 400 MiB: the estimate at PART_MM is about 0.5 GiB
    assert main(["check", str(path), "--json"]) == 0
    guards = json.loads(capsys.readouterr().out)["fine_pass"]["guards"]
    assert guards["ok"] is False and "memory" in [f["guard"] for f in guards["failed"]] and "use a machine with more memory" in guards["text"]
    assert guards["voxel_pass_mm"] is not None and guards["voxel_pass_mm"] > PART_MM
    err = refused(capsys, run_args(path, tmp_path / "run", "--fine-pass"))           # the pre-check: the voxel of the fine pass, the memory guard only
    assert "the memory estimate is" in err and "RAM" in err and "settings.max_active_voxels" not in err and not (tmp_path / "run").exists()


def test_an_error_of_the_fine_grid_names_fine_voxel_and_not_the_voxel_size_of_the_yaml(parent, tmp_path, capsys):
    (tmp_path / "copy").mkdir()
    shutil.copytree(parent[0].parent, tmp_path / "copy" / "case")
    path = tmp_path / "copy" / "case" / parent[0].name
    edit(path, lambda d: d["settings"].update(max_active_voxels=2000))                 # a soft key: the parent run stays valid
    out = tmp_path / "fine"
    err = refused(capsys, run_args(path, out, "--fine-from", str(parent[1]), "--fine-voxel", "0.8"))
    assert "Try --fine-voxel" in err and "(MCP fine_voxel)" in err and "Try voxel_size" not in err and "voxel_size" not in err and not out.exists()
    assert main(["check", str(path), "--fine-from", str(parent[1]), "--fine-voxel", "0.8"]) == 1
    err = capsys.readouterr().err
    assert "Try --fine-voxel" in err and "voxel_size" not in err


def test_a_voxel_coarser_than_the_parent_is_allowed_with_a_fact(parent, tmp_path):
    out = tmp_path / "fine"
    first = tmp_path / "f1"
    assert fine_run(parent, first, "--fine-voxel", "1.0", "--fine-passes", "2") == 0
    assert main(run_args(parent[0], out, "--fine-from", str(first), "--fine-voxel", "1.5", "--fine-passes", "2")) == 0
    assert "is coarser than the voxel of the parent run, 1 mm" in log(out)


def test_the_same_voxel_is_refused_and_the_error_names_continue_from(parent, tmp_path, capsys):
    out = tmp_path / "fine"
    err = refused(capsys, run_args(parent[0], out, "--fine-from", str(parent[1]), "--fine-voxel", "2.0"))
    assert "the same as the voxel of the parent run" in err and "--continue-from PARENT with --more-iterations N" in err
    assert not out.exists()


def test_a_part_voxel_that_is_not_finer_than_the_parent_is_refused_with_options_and_the_one_command_form_ends_after_the_first_run(tmp_path, capsys):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=0.6, iterations=1)                            # the parent (0.6 mm) is not coarser than the voxel of the part
    first = tmp_path / "first"
    assert main(run_args(yaml_path, first)) == 0
    out = tmp_path / "fine"
    err = refused(capsys, run_args(yaml_path, out, "--fine-from", str(first)))
    part = F.part_voxel(load_config(yaml_path), verify=True)["voxel_mm"]                  # the voxel that the fine pass takes
    assert "is not finer than the voxel of the parent run" in err and f"{part:g} mm" in err and "0.6 mm" in err
    assert "--fine-voxel H forces a voxel" in err and "--continue-from PARENT with --more-iterations N" in err and not out.exists()
    assert main(["check", str(yaml_path), "--fine-from", str(first)]) == 1                   # the check gives the refusal of the run
    assert "is not finer than the voxel of the parent run" in capsys.readouterr().err
    one = tmp_path / "one"
    assert main(run_args(yaml_path, one, "--fine-pass")) == 0                                # the run voxel is fine enough: the existing fact, no fine pass
    assert not fine_folder(one).exists() and "fine pass: --fine-pass is on, but there is no fine pass" in log(one)
    assert main(run_args(yaml_path, out, "--fine-from", str(first), "--fine-voxel", "0.8", "--fine-passes", "1")) == 0     # --fine-voxel forces it
    assert read_json(out / "run.json")["fine_from"]["voxel_from"] == "fine_voxel"


def test_fine_pass_with_an_out_folder_that_has_no_name_builds_the_sibling_from_the_resolved_path(tmp_path, monkeypatch):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=2.0, iterations=2)
    out = tmp_path / "run"
    out.mkdir()
    monkeypatch.chdir(out)
    assert fine_folder(".") == tmp_path / "run-fine" and fine_folder("..") == tmp_path.parent / (tmp_path.name + "-fine")
    assert main(run_args(yaml_path, ".", "--fine-pass", "--fine-voxel", "1.0", "--fine-passes", "2")) == 0
    assert (tmp_path / "run-fine" / "result" / "metrics.json").is_file() and read_json(tmp_path / "run-fine" / "run.json")["fine_from"]["run_dir"] == str(out.resolve())
    assert not (tmp_path.parent / "-fine").exists() and not Path("-fine").exists()


@pytest.mark.parametrize("flags, words", [
    (("--fine-tol", "0"), "--fine-tol must be a number above 0"),
    (("--fine-tol", "x"), "--fine-tol must be a number above 0"),
    (("--fine-passes", "0"), "--fine-passes must be a whole number of 1 or more"),
    (("--fine-passes", "2.5"), "--fine-passes must be a whole number of 1 or more"),
    (("--fine-voxel", "-1"), "--fine-voxel must be a number of mm above 0"),
])
def test_a_wrong_value_is_refused_with_options_and_writes_nothing(parent, tmp_path, capsys, flags, words):
    out = tmp_path / "fine"
    err = refused(capsys, run_args(parent[0], out, "--fine-from", str(parent[1]), *flags))
    assert words in err and "Options:" in err and not out.exists()


def test_the_fine_options_need_fine_from_or_fine_pass(parent, tmp_path, capsys):
    out = tmp_path / "x"
    err = refused(capsys, run_args(parent[0], out, "--fine-tol", "0.1", "--fine-passes", "3"))
    assert "--fine-passes, --fine-tol go with --fine-from or --fine-pass" in err and not out.exists()


def test_a_wrong_mix_is_refused_and_writes_nothing(parent, tmp_path, capsys):
    out = tmp_path / "x"
    base = run_args(parent[0], out, "--fine-from", str(parent[1]))
    for extra, flag in ((("--resume",), "--resume"), (("--continue-from", str(parent[1]), "--more-iterations", "2"), "--continue-from")):
        err = refused(capsys, base + list(extra))
        assert f"--fine-from does not go with {flag}" in err and not out.exists()
    for extra, flag in ((("--resume",), "--resume"), (("--allow-changed-inputs",), "--allow-changed-inputs")):
        err = refused(capsys, run_args(parent[0], out, "--fine-pass", *extra))
        assert f"--fine-pass does not go with {flag}" in err and not out.exists()
    err = refused(capsys, run_args(parent[0], out, "--fine-pass", "--fine-from", str(parent[1])))
    assert "--fine-pass does not go with --fine-from" in err
    err = refused(capsys, run_args(parent[0], out, "--allow-changed-inputs"))
    assert "needs --resume, --continue-from or --fine-from" in err


def test_the_new_folder_may_not_be_the_parent_or_inside_it(parent, capsys):
    before = state_of(parent[1])
    err = refused(capsys, run_args(parent[0], parent[1], "--fine-from", str(parent[1])))
    assert "is the parent run folder" in err
    err = refused(capsys, run_args(parent[0], parent[1] / "inside", "--fine-from", str(parent[1])))
    assert "is inside the parent run folder" in err
    assert state_of(parent[1]) == before and not (parent[1] / "inside").exists()


def test_a_parent_that_is_not_a_folder_or_not_finished_is_refused_and_writes_nothing(parent, tmp_path, capsys):
    out = tmp_path / "fine"
    err = refused(capsys, run_args(parent[0], out, "--fine-from", str(tmp_path / "nothing")))
    assert "is not a folder" in err and not out.exists()
    unfinished = tmp_path / "unfinished"
    killed(parent[0], unfinished, "after_checkpoint:3")
    err = refused(capsys, run_args(parent[0], out, "--fine-from", str(unfinished)))
    assert "the run did not finish" in err and "--resume" in err and not out.exists()


def test_changed_inputs_are_refused_and_the_override_goes_on(parent, tmp_path, capsys):
    work = tmp_path / "case"
    shutil.copytree(parent[0].parent, work)
    edit(work / parent[0].name, lambda d: d["settings"].update(volume_fraction=0.35))
    out = tmp_path / "fine"
    err = refused(capsys, run_args(work / parent[0].name, out, "--fine-from", str(parent[1]), "--fine-voxel", "1.0"))
    assert "Fine pass stopped: the inputs changed since the parent run" in err and "settings.volume_fraction: 0.3 -> 0.35" in err
    assert "--allow-changed-inputs" in err and not out.exists()
    assert main(run_args(work / parent[0].name, out, "--fine-from", str(parent[1]), "--fine-voxel", "1.0", "--fine-passes", "2",
                         "--allow-changed-inputs")) == 0
    assert "fine pass: the inputs changed since the parent run: settings.volume_fraction: 0.3 -> 0.35" in log(out)


def test_the_voxel_the_iterations_the_schedule_and_the_default_radius_are_no_change(parent, tmp_path):
    work = tmp_path / "case"
    shutil.copytree(parent[0].parent, work)
    edit(work / parent[0].name, lambda d: d["settings"].update(voxel_size=1.0, iterations=50, beta_schedule=[[0, 1], [20, 2]]))
    out = tmp_path / "fine"
    assert main(run_args(work / parent[0].name, out, "--fine-from", str(parent[1]), "--fine-voxel", "1.0", "--fine-passes", "2")) == 0
    assert read_json(out / "run.json")["iterations"] == 2                 # settings.iterations of the YAML is not used


# --- the resume of a killed fine pass -----------------------------------------------------------------

def test_a_killed_fine_pass_resumes_from_its_own_checkpoint_and_equals_a_run_that_was_not_killed(parent, tmp_path):
    flags = ("--fine-from", str(parent[1]), "--fine-voxel", "1.0", "--fine-passes", "7", "--fine-tol", "1e-9")
    base, out = tmp_path / "base", tmp_path / "out"
    finish(parent[0], base, *flags)
    killed(parent[0], out, "after_checkpoint:3", *flags)
    meta = read_json_checkpoint_meta(out)
    assert meta["iteration"] == 3 and not meta["complete"] and meta["fine_from"]["max_passes"] == 7 and meta["fine_from"]["voxel_mm"] == 1.0
    moved = tmp_path / "parent-moved"                                   # the resume does not read the parent: it can be gone
    shutil.move(str(parent[1]), str(moved))
    try:
        done = finish(parent[0], out, "--resume")
    finally:
        shutil.move(str(moved), str(parent[1]))
    assert "resume: this run is a fine pass from" in log(out) and "its maximum is 7 passes" in log(out)
    assert "settings.voxel_size and settings.iterations of the YAML are not used" in log(out)
    assert done.returncode == 0 and history(out)[-1]["iteration"] == 7
    same_run(base, out, exact=DIRECT)
    assert read_json(out / "run.json")["fine_from"] == read_json(base / "run.json")["fine_from"]


@pytest.mark.parametrize("flag", [(), ("--no-stop-rules",)])
def test_a_killed_continuation_without_stop_rules_resumes_with_the_rules_off_with_or_without_the_flag(parent, tmp_path, flag):
    """The review case (scen8, scen9): the resume of a continuation of a fine pass with --no-stop-rules took the rule 'settled' from the record of the
    fine pass, ended early, and recorded the rules as on. The state of a run comes from its own record (the step settings of the continuation)."""
    first, base, out = tmp_path / "f1", tmp_path / "base", tmp_path / "out"
    assert fine_run(parent, first, "--fine-voxel", "1.0", "--fine-passes", "40", "--fine-tol", "0.5") == 0
    last = read_json(first / "verdict.json")["final_iteration"]
    assert read_json(first / "verdict.json")["stop"]["rule"] == "settled"
    extra = ("--continue-from", str(first), "--more-iterations", "3", "--no-stop-rules")
    finish(parent[0], base, *extra)
    assert history(base)[-1]["iteration"] == last + 3 and read_json(base / "verdict.json")["stop_rules"]["enabled"] is False
    killed(parent[0], out, f"after_checkpoint:{last + 1}", *extra)
    assert read_json_checkpoint_meta(out)["continued_from"]["step_settings"]["stop_rules"]["enabled"] is False
    finish(parent[0], out, "--resume", *flag)
    verdict = read_json(out / "verdict.json")
    assert verdict["stop_rules"]["enabled"] is False and verdict["state"] == "iteration_limit" and history(out)[-1]["iteration"] == last + 3
    text = log(out)
    assert "its maximum is" in text and "all stop rules are off (as the checkpoint has it)" in text and "it ends when the loss has settled" not in text
    same_run(base, out, exact=DIRECT, run_json_drop=("continued_from",))        # the same pass count and the same compliance as the run that was not killed
    assert read_json(out / "run.json")["continued_from"]["step_settings"]["stop_rules"]["enabled"] is False


def test_a_resume_with_an_explicit_stop_rule_flag_wins_over_the_record_and_the_records_follow(parent, tmp_path):
    flags = ("--fine-from", str(parent[1]), "--fine-voxel", "1.0", "--fine-passes", "20", "--fine-tol", "0.5")
    on, off = tmp_path / "on", tmp_path / "off"
    killed(parent[0], on, "after_checkpoint:2", *flags)                              # the rule 'settled' is on in the record
    finish(parent[0], on, "--resume", "--no-stop-rules")                             # the flag of the resume wins: the fine pass runs to its maximum
    assert history(on)[-1]["iteration"] == 20 and read_json(on / "verdict.json")["state"] == "iteration_limit"
    assert read_json(on / "verdict.json")["stop_rules"]["enabled"] is False and read_json(on / "run.json")["fine_from"]["stop_rule"] is None
    assert read_json_checkpoint_meta(on)["fine_from"]["stop_rule"] is None
    assert "the checkpoint has the stop rules on; --no-stop-rules of this resume sets them off" in log(on)
    killed(parent[0], off, "after_checkpoint:2", *flags, "--no-stop-rules")          # all stop rules are off in the record
    finish(parent[0], off, "--resume", "--stop-rules")                               # the flag wins the other way
    assert read_json(off / "verdict.json")["stop_rules"]["enabled"] is True and read_json(off / "run.json")["fine_from"]["stop_rule"] == "settled"
    assert history(off)[-1]["iteration"] < 20 and read_json(off / "verdict.json")["stop"]["rule"] == "settled"
    assert "the checkpoint has the stop rules off; --stop-rules of this resume sets them on" in log(off)


def test_a_continuation_of_a_continuation_keeps_the_state_of_the_stop_rules_of_its_parent(parent, tmp_path):
    first, c1, c2, c3 = tmp_path / "f1", tmp_path / "c1", tmp_path / "c2", tmp_path / "c3"
    assert fine_run(parent, first, "--fine-voxel", "1.0", "--fine-passes", "40", "--fine-tol", "0.5") == 0
    assert main(continue_args(parent[0], c1, first, 2, "--no-stop-rules")) == 0
    assert main(continue_args(parent[0], c2, c1, 2)) == 0                            # no flag: the rules of the parent (off) go on
    assert read_json(c2 / "verdict.json")["stop_rules"]["enabled"] is False and "It keeps the setting that all stop rules are off" in log(c2)
    assert main(continue_args(parent[0], c3, c1, 2, "--stop-rules")) == 0            # the flag wins, and the fact names both
    assert read_json(c3 / "verdict.json")["stop_rules"]["enabled"] is True
    assert "The stop rules are on in this run: the parent has them off, and --stop-rules of this run sets them" in log(c3)


# --- the one command ----------------------------------------------------------------------------------

def test_fine_pass_runs_the_first_run_and_then_the_fine_pass_in_the_sibling_folder(parent, tmp_path, capsys):
    out = tmp_path / "run"
    assert main(run_args(parent[0], out, "--fine-pass", "--fine-voxel", "1.0", "--fine-passes", "3")) == 0
    fine = fine_folder(out)
    assert fine == tmp_path / "run-fine" and (out / "result" / "metrics.json").is_file() and (fine / "result" / "metrics.json").is_file()
    first, second = read_json(out / "run.json"), read_json(fine / "run.json")
    assert "fine_from" not in first and second["fine_from"]["run_dir"] == str(out.resolve())
    assert first["part_voxel"]["recommend"] is True and second["h"] == 1.0 and second["iterations"] == 3
    assert "fine pass: --fine-pass is on. The fine pass starts next, in " + str(fine) in log(out)
    assert f"fine pass: the part needs a voxel of {MEASURED_MM:g} mm or less" in log(out)      # the recommendation line at the end of the log: the measured value
    assert read_json(fine / "verdict.json")["state"] == "iteration_limit"


def test_the_one_command_states_one_voxel_in_the_first_run_and_the_fine_pass_takes_it_after_the_test_on_its_grid(tmp_path, capsys):
    """The end of the first run says the measured voxel (MEASURED_MM, no grid test); the fine pass tests it on its grid and takes PART_MM."""
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=2.0, iterations=2)
    out = tmp_path / "run"
    assert main(run_args(yaml_path, out, "--fine-pass", "--fine-passes", "1")) == 0
    first, second = read_json(out / "run.json")["part_voxel"], read_json(fine_folder(out) / "run.json")
    assert first["voxel_mm"] == MEASURED_MM and first["recommend"] is True and first["run_voxel_mm"] == 2.0
    assert not {"level", "voxel_high_mm", "probe_voxels_max"} & set(first) and "probe_mm" not in first["limiting"]
    rec = second["fine_from"]
    assert second["h"] == pytest.approx(PART_MM) and rec["voxel_from"] == "part_voxel" and rec["part_voxel_mm"] == pytest.approx(PART_MM)
    assert rec["grid_check"]["start_mm"] == MEASURED_MM and rec["grid_check"]["steps"] == 1
    fine_log = log(fine_folder(out))
    assert "at voxel 0.65 mm (the voxel that the part needs)." in fine_log and CHECK_FACT in fine_log


def cantilever_with_a_wide_tip(tmp_path):
    """The cantilever with a tip load of 4 x 4 x 3 mm: the plate (2 mm) sets the voxel alone, and its 3 rows at 0.7 mm pass the test on the grid."""
    from test_optimize import box
    (tmp_path / "wide").mkdir()
    path = make_case(tmp_path / "wide", h=2.0, iterations=2)
    box(tmp_path / "wide" / "tip.stl", (16, 3, 0), (20, 7, 3))
    return path


def test_check_and_the_fine_pass_give_the_same_voxel_when_the_grid_holds_enough_voxels_across(tmp_path, capsys):
    path = cantilever_with_a_wide_tip(tmp_path)
    assert main(["check", str(path), "--json"]) == 0
    block = json.loads(capsys.readouterr().out)["fine_pass"]
    verified = F.part_voxel(load_config(path), verify=True)
    assert block["voxel_mm"] == MEASURED_MM == verified["voxel_mm"] and verified["checked"]["steps"] == 0 and block["limiting"]["name"] == "plate"
    out = tmp_path / "run"
    assert main(run_args(path, out, "--fine-pass", "--fine-passes", "1")) == 0
    rec = read_json(fine_folder(out) / "run.json")["fine_from"]
    assert rec["voxel_mm"] == block["voxel_mm"] and rec["grid_check"]["steps"] == 0
    assert "has too few voxels across" not in log(fine_folder(out))             # no step down: no fact about it


def test_fine_pass_with_a_run_voxel_that_is_fine_enough_ends_after_the_first_run_with_exit_code_0(tmp_path, capsys):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=0.7, iterations=2)                            # the run voxel is the voxel of the part: no recommendation
    out = tmp_path / "run"
    assert main(run_args(yaml_path, out, "--fine-pass")) == 0
    assert (out / "result" / "metrics.json").is_file() and not fine_folder(out).exists()
    block = read_json(out / "run.json")["part_voxel"]
    assert block["recommend"] is False and "fact" not in block
    text = log(out)
    assert "fine pass: --fine-pass is on, but there is no fine pass" in text and "The run ends here" in text
    assert "fine pass: the part needs a voxel" not in text


def test_fine_pass_with_a_run_voxel_that_is_fine_enough_says_so_at_the_start_of_the_first_run(tmp_path):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=0.7, iterations=2)
    out = tmp_path / "run"
    assert main(run_args(yaml_path, out, "--fine-pass")) == 0
    lines = log(out).splitlines()
    start = lines.index("topopt run: cantilever")
    assert lines[start + 1].startswith("fine pass: --fine-pass is on, but the part needs a voxel of 0.7 mm (the ")
    assert "not finer than the voxel of this run, 0.7 mm, by 0.05 mm or more. There will be no fine pass: the run ends after the first run, with exit code 0." in lines[start + 1]
    assert any(ln.startswith("fine pass: --fine-pass is on, but there is no fine pass.") for ln in lines[start + 2:])      # and at the end, as before


def test_fine_pass_with_a_fine_voxel_that_is_finer_than_the_run_runs_at_that_voxel_although_the_part_needs_no_finer_one(tmp_path):
    """Tools inform, the user decides: `--fine-pass --fine-voxel F` agrees with `--fine-from ... --fine-voxel F`."""
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=0.7, iterations=2)                            # the run voxel is the voxel of the part: no recommendation
    out = tmp_path / "run"
    assert main(run_args(yaml_path, out, "--fine-pass", "--fine-voxel", "0.65", "--fine-passes", "1")) == 0
    assert read_json(out / "run.json")["part_voxel"]["recommend"] is False
    fine = fine_folder(out)
    assert (fine / "result" / "metrics.json").is_file() and read_json(fine / "status.json")["state"] == "done"      # the fine pass ran
    rec = read_json(fine / "run.json")["fine_from"]
    assert rec["voxel_mm"] == 0.65 and rec["voxel_from"] == "fine_voxel" and read_json(fine / "run.json")["h"] == 0.65
    fact_start = "fine pass: the part needs about 0.7 mm (the "
    fact_end = "which is not finer than the voxel of this run, 0.7 mm, by 0.05 mm or more. --fine-voxel 0.65 mm is finer, so the fine pass runs at 0.65 mm as asked."
    first = log(out).splitlines()
    start = first.index("topopt run: cantilever")
    assert first[start + 1].startswith(fact_start) and first[start + 1].endswith(fact_end)          # the fact is at the start of the first run
    assert not any("no fine pass" in ln.lower() or "run ends here" in ln.lower() for ln in first)    # the "no fine pass" note is gone
    assert any(ln == f"fine pass: --fine-pass is on. The fine pass starts next, in {fine}, from this run, at 0.65 mm as --fine-voxel says." for ln in first)
    second = log(fine).splitlines()
    assert any(ln.startswith(fact_start) and ln.endswith(fact_end) for ln in second)                # and in the log of the fine run
    assert sum(ln.startswith("fine pass: new run from the design of") for ln in second) == 1


def test_fine_pass_with_a_finer_fine_voxel_tests_the_fine_grid_against_the_guards_before_the_first_run(tmp_path, capsys):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=0.7, iterations=2)
    import yaml
    d = yaml.safe_load(yaml_path.read_text())
    d["settings"]["max_active_voxels"] = 7000                                              # the first run (6414 active voxels) passes, the grid at 0.5 mm does not
    yaml_path.write_text(yaml.safe_dump(d))
    out = tmp_path / "run"
    err = refused(capsys, run_args(yaml_path, out, "--fine-pass", "--fine-voxel", "0.5"))
    assert "the fine grid at 0.5 mm fails settings.max_active_voxels" in err and "stops before the first run" in err
    assert not out.exists() and not fine_folder(out).exists()                               # no first run was made


def test_the_fine_pass_follows_the_first_run_of_fine_pass_when_the_part_needs_it_or_a_fine_voxel_was_given():
    import argparse
    from topopt.cli import _fine_follows
    first = argparse.Namespace(first_of_fine=True, fine_forced_voxel=None)
    forced = argparse.Namespace(first_of_fine=True, fine_forced_voxel=0.5)
    plain = argparse.Namespace(fine_forced_voxel=0.5)                                       # not the first run of --fine-pass
    assert _fine_follows(first, {"recommend": True}) and not _fine_follows(first, {"recommend": False}) and not _fine_follows(first, None)
    assert _fine_follows(forced, {"recommend": False}) and _fine_follows(forced, None) and not _fine_follows(plain, {"recommend": True})


@pytest.mark.parametrize("fine_voxel, words", [("2.0", "the same as the voxel of the first run, 2 mm"), ("2", "the same as the voxel of the first run, 2 mm"),
                                                ("2.5", "coarser than the voxel of the first run, 2 mm")])
def test_fine_pass_refuses_before_the_first_run_a_fine_voxel_that_is_not_finer_than_the_run(tmp_path, capsys, fine_voxel, words):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=2.0, iterations=2)
    out = tmp_path / "run"
    err = refused(capsys, run_args(yaml_path, out, "--fine-pass", "--fine-voxel", fine_voxel))
    assert f"--fine-pass: --fine-voxel {float(fine_voxel):g} mm is {words}" in err and "stops before the first run" in err
    assert "Options: give --fine-voxel a value below 2 (the part needs about 0.7 mm: `topopt check` prints it); remove --fine-voxel to use the voxel that the part needs; " \
           "remove --fine-pass to make the first run only." in err
    assert "--fine-from" not in err and "--continue-from" not in err                      # the options name what the user typed
    assert not out.exists() and not fine_folder(out).exists()                            # nothing is written: no first run


def test_fine_pass_refuses_a_wrong_fine_voxel_before_the_first_run_also_when_the_part_needs_no_finer_voxel(tmp_path, capsys):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=0.7, iterations=2)
    out = tmp_path / "run"
    err = refused(capsys, run_args(yaml_path, out, "--fine-pass", "--fine-voxel", "-1"))
    assert "--fine-voxel must be a number of mm above 0" in err and not out.exists()
    err = refused(capsys, run_args(yaml_path, out, "--fine-pass", "--fine-voxel", "0.7"))
    assert "the same as the voxel of the first run, 0.7 mm" in err and not out.exists()


def one_command_case(tmp_path, h=2.0):
    (tmp_path / "case").mkdir()
    return make_case(tmp_path / "case", h=h, iterations=2), tmp_path / "run"


def test_a_stop_request_in_the_fine_folder_after_the_first_run_prevents_the_fine_pass(tmp_path, capsys):
    """The window between the end of the first run and the first pass of the fine run: the MCP tool `stop_run` writes `job.stop` in the fine folder."""
    from topopt.runjob import STOPPED_EXIT, request_stop
    yaml_path, out = one_command_case(tmp_path)
    fine = fine_folder(out)
    request_stop(fine, "mcp stop_run", os.getpid())               # the request is there before the driver looks (the process is this one)
    assert main(["run", str(yaml_path), "--out", str(out), "--no-wait", "--fine-pass", "--fine-voxel", "1.0", "--fine-passes", "2"]) == STOPPED_EXIT
    text = capsys.readouterr().out
    assert "fine pass: a stop request came after the first run: the fine pass did not start." in text
    assert (out / "result" / "metrics.json").is_file() and read_json(out / "status.json")["state"] == "done"          # the first run is finished and keeps its result
    assert "the fine pass did not start" in log(out) and "is removed" in log(out)
    assert not fine.exists()                                       # the request is removed; the empty folder that it made is removed too


def test_a_stop_request_in_the_first_folder_during_the_post_step_prevents_the_fine_pass(tmp_path, capsys, monkeypatch):
    from topopt import cli
    from topopt.runjob import STOPPED_EXIT, request_stop
    yaml_path, out = one_command_case(tmp_path)
    real = cli.run_refine

    def refine_then_request(*a, **k):                              # the request comes after the last read of the first run
        real(*a, **k)
        request_stop(out, "topopt stop", os.getpid())
    monkeypatch.setattr(cli, "run_refine", refine_then_request)
    assert main(["run", str(yaml_path), "--out", str(out), "--no-wait", "--fine-pass", "--fine-voxel", "1.0", "--fine-passes", "2"]) == STOPPED_EXIT
    text = log(out)
    assert "came after the last pass" in text and "so the fine pass does not start" in text and "a stop request came after the first run: the fine pass did not start" in text
    assert not (out / "job.stop").exists() and not fine_folder(out).exists() and read_json(out / "status.json")["state"] == "done"


def test_topopt_stop_in_the_post_step_of_the_first_run_of_fine_pass_writes_the_request_and_prevents_the_fine_pass(tmp_path, capsys, monkeypatch):
    """`topopt stop <first folder>` in the window where the first run reads no request: `stop_reply` writes it for the first run of `--fine-pass`
    (job.json marks it), because `_stop_before_fine` reads it before the fine pass starts."""
    from topopt import cli
    from topopt.runjob import STOPPED_EXIT
    yaml_path, out = one_command_case(tmp_path)
    real = cli.run_refine
    seen = {}

    def stop_then_refine(*a, **k):                                 # the status is `done` and the job still holds the lock: the refine step
        seen["job"] = read_json(out / "job.json")
        capsys.readouterr()
        seen["rc"] = main(["stop", str(out)])
        seen["text"] = capsys.readouterr().out
        seen["request"] = read_json(out / "job.stop")
        real(*a, **k)
    monkeypatch.setattr(cli, "run_refine", stop_then_refine)
    assert main(["run", str(yaml_path), "--out", str(out), "--no-wait", "--fine-pass", "--fine-voxel", "1.0", "--fine-passes", "2"]) == STOPPED_EXIT
    assert seen["job"]["fine_pass"] == {"first_run": True, "fine_run_dir": str(fine_folder(out)), "forced": True}      # forced: --fine-voxel was given
    assert seen["rc"] == 0 and seen["request"]["by"] == "topopt stop" and seen["request"]["target_pid"] == os.getpid()      # exit 0: a request is in the folder
    assert "The first run of --fine-pass" in seen["text"] and "reads it before it starts the fine pass" in seen["text"] and "Nothing is written" not in seen["text"]
    assert "a stop request came after the first run: the fine pass did not start" in log(out)
    assert not (out / "job.stop").exists() and not fine_folder(out).exists()                  # the driver removed the request; no fine run was made
    assert read_json(out / "status.json")["state"] == "done" and (out / "result" / "metrics.json").is_file()      # the first run is finished and keeps its result


def test_topopt_stop_in_the_post_step_of_a_first_run_that_needs_no_fine_pass_writes_nothing_as_before(tmp_path, capsys, monkeypatch):
    from topopt import cli
    yaml_path, out = one_command_case(tmp_path, h=0.7)             # the run voxel is the voxel of the part: no fine pass follows
    real = cli.run_refine
    seen = {}

    def stop_then_refine(*a, **k):
        capsys.readouterr()
        seen["rc"] = main(["stop", str(out)])
        seen["text"] = capsys.readouterr().out
        seen["file"] = (out / "job.stop").exists()
        real(*a, **k)
    monkeypatch.setattr(cli, "run_refine", stop_then_refine)
    assert main(["run", str(yaml_path), "--out", str(out), "--no-wait", "--fine-pass"]) == 0
    assert seen["rc"] == 1 and seen["file"] is False and "post step or the refine step" in seen["text"] and "Nothing is written" in seen["text"]


def test_the_fine_pass_follows_the_job_of_the_first_run_when_the_part_needs_it_or_a_fine_voxel_was_given(tmp_path):
    from topopt.runjob import _fine_pass_follows
    job = {"fine_pass": {"first_run": True, "fine_run_dir": str(tmp_path / "x-fine"), "forced": False}}
    forced = {"fine_pass": {**job["fine_pass"], "forced": True}}
    assert _fine_pass_follows(tmp_path, job) is False                                          # run.json has no part_voxel block yet
    (tmp_path / "run.json").write_text(json.dumps({"part_voxel": {"recommend": True}}))
    assert _fine_pass_follows(tmp_path, job) and _fine_pass_follows(tmp_path, forced)
    (tmp_path / "run.json").write_text(json.dumps({"part_voxel": {"recommend": False}}))
    assert not _fine_pass_follows(tmp_path, job) and _fine_pass_follows(tmp_path, forced)      # a given fine voxel wins
    assert not _fine_pass_follows(tmp_path, {}) and not _fine_pass_follows(tmp_path, {"fine_pass": {"first_run": False}})      # not the first run of --fine-pass


def test_a_stop_request_during_the_plan_of_the_fine_grid_prevents_the_fine_pass(tmp_path, capsys, monkeypatch):
    from topopt import cli
    from topopt.runjob import STOPPED_EXIT, request_stop
    yaml_path, out = one_command_case(tmp_path)
    real = cli._plan_fine

    def plan_then_request(*a, **k):                                # the plan of the fine grid takes time; the request comes in it
        held = real(*a, **k)
        request_stop(fine_folder(out), "mcp stop_run", os.getpid())
        return held
    monkeypatch.setattr(cli, "_plan_fine", plan_then_request)
    assert main(["run", str(yaml_path), "--out", str(out), "--no-wait", "--fine-pass", "--fine-voxel", "1.0", "--fine-passes", "2"]) == STOPPED_EXIT
    assert "a stop request came after the first run: the fine pass did not start" in log(out)
    assert not fine_folder(out).exists()                           # no fine run was made


def test_a_stop_request_in_the_post_step_of_a_first_run_that_needs_no_fine_pass_is_removed_as_before(tmp_path, monkeypatch):
    from topopt import cli
    from topopt.runjob import request_stop
    yaml_path, out = one_command_case(tmp_path, h=0.7)             # the run voxel is the voxel of the part: no fine pass
    real = cli.run_refine

    def refine_then_request(*a, **k):
        real(*a, **k)
        request_stop(out, "topopt stop", os.getpid())
    monkeypatch.setattr(cli, "run_refine", refine_then_request)
    assert main(["run", str(yaml_path), "--out", str(out), "--no-wait", "--fine-pass"]) == 0
    assert "came after the last pass" in log(out) and "so the request is removed" in log(out)      # the text and the effect of a plain run
    assert not (out / "job.stop").exists() and not fine_folder(out).exists()


def test_the_fine_pass_does_not_start_when_the_first_run_does_not_end_with_exit_code_0(tmp_path, capsys):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=2.0, iterations=2)
    edit(yaml_path, lambda d: d["settings"].update(volume_fraction=7))                       # an error in the first run
    out = tmp_path / "run"
    assert main(run_args(yaml_path, out, "--fine-pass")) == 1
    assert not fine_folder(out).exists()
    assert capsys.readouterr().err.startswith("error:")


# --- check and the recommendation ---------------------------------------------------------------------

def test_a_plain_check_prints_the_fact_and_the_key(parent, capsys):
    assert main(["check", str(parent[0])]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("fine pass: the part needs")]
    assert len(lines) == 1 and f"the part needs a voxel of {MEASURED_MM:g} mm or less" in lines[0] and "between" not in lines[0].split("Option:")[0]
    assert "This run uses 2 mm" in lines[0] and "--fine-from" in lines[0] and "goes down in steps of 0.05 mm" in lines[0]
    assert f"Estimate at {MEASURED_MM:g} mm: about" in lines[0] and "less if the loss settles first" in lines[0] and "estimates" in lines[0]
    assert main(["check", str(parent[0]), "--json"]) == 0
    block = json.loads(capsys.readouterr().out)["fine_pass"]
    assert block["recommend"] and block["voxel_mm"] == pytest.approx(MEASURED_MM) and block["run_voxel_mm"] == 2.0
    assert not {"voxel_high_mm", "level", "probe_voxels_max"} & set(block) and "probe_mm" not in block["limiting"]
    assert block["limiting"]["role"] in ("fixed", "load") and block["limiting"]["across_needed"] == pytest.approx(2.8)
    assert block["estimate"]["free_dofs"] > 1080 and block["max_passes"] == PARENT_PASSES and block["fact"] == lines[0]


def test_a_plain_check_of_a_fine_enough_run_voxel_has_no_fact(tmp_path, capsys):
    (tmp_path / "case").mkdir()
    yaml_path = make_case(tmp_path / "case", h=0.7, iterations=2)
    assert main(["check", str(yaml_path)]) == 0
    assert "fine pass:" not in capsys.readouterr().out
    assert main(["check", str(yaml_path), "--json"]) == 0
    block = json.loads(capsys.readouterr().out)["fine_pass"]
    assert block["recommend"] is False and "fact" not in block


def test_a_run_log_and_a_part_card_carry_the_recommendation(parent):
    assert f"fine pass: the part needs a voxel of {MEASURED_MM:g} mm or less" in log(parent[1])
    card = (parent[1] / "result" / "report.md").read_text()
    assert "## Fine pass" in card and f"fine pass: the part needs a voxel of {MEASURED_MM:g} mm or less" in card
    assert str(parent[1].parent) not in card                                                  # no absolute path in the card


def test_a_fine_pass_has_its_block_in_the_part_card(parent, tmp_path):
    out = tmp_path / "fine"
    assert fine_run(parent, out, "--fine-voxel", "1.0", "--fine-passes", "3", "--fine-tol", "0.01") == 0
    card = (out / "result" / "report.md").read_text()
    assert "## Fine pass" in card and "This run is a fine pass" in card and f"`{parent[1].name}`" in card and str(tmp_path) not in card
    assert "voxel 2 mm)" in card and "on a voxel of 1 mm" in card and "after 3 passes at most" in card and "at or below 0.01" in card
    assert "Here: the maximum of passes ended it at pass 3 before the loss settled." in card


def test_check_fine_from_shows_the_plan_and_writes_nothing(parent, capsys):
    before = state_of(parent[1])
    assert main(["check", str(parent[0]), "--fine-from", str(parent[1])]) == 0
    out = capsys.readouterr().out
    assert "fine pass: new run from the design of" in out
    assert f"at voxel {PART_MM:g} mm (the voxel that the part needs)." in out and CHECK_FACT in out
    assert f"voxel {PART_MM:g} mm" in out and "grid shape:" in out and "This run uses" not in out      # the plan, not the recommendation of a plain check
    assert main(["check", str(parent[0]), "--fine-from", str(parent[1]), "--fine-voxel", "1.0", "--fine-passes", "9", "--fine-tol", "0.01", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    rec = data["fine_from"]
    assert data["grid"]["voxel_mm"] == 1.0 and data["fine_pass"] is None
    assert rec["voxel_mm"] == 1.0 and rec["max_passes"] == 9 and rec["tolerance"] == 0.01 and rec["voxel_from"] == "fine_voxel"
    assert rec["run_dir"] == str(parent[1].resolve()) and any("new run from the design of" in f for f in rec["facts"])
    assert state_of(parent[1]) == before


def test_check_fine_from_gives_the_refusals_of_the_run(parent, tmp_path, capsys):
    err = refused(capsys, ["check", str(parent[0]), "--fine-from", str(parent[1]), "--fine-voxel", "2.0"])
    assert "the same as the voxel of the parent run" in err
    err = refused(capsys, ["check", str(parent[0]), "--fine-voxel", "1.0"])
    assert "go with --fine-from or --fine-pass" in err or "goes with --fine-from or --fine-pass" in err
    work = tmp_path / "case"
    shutil.copytree(parent[0].parent, work)
    edit(work / parent[0].name, lambda d: d["settings"].update(volume_fraction=0.35))
    err = refused(capsys, ["check", str(work / parent[0].name), "--fine-from", str(parent[1])])
    assert "Fine pass stopped: the inputs changed" in err
    assert main(["check", str(work / parent[0].name), "--fine-from", str(parent[1]), "--allow-changed-inputs"]) == 0


# --- a plain run does not change ----------------------------------------------------------------------

def test_a_plain_run_has_no_fine_key_and_no_stop_rule_from_the_fine_pass(parent):
    run = read_json(parent[1] / "run.json")
    assert "fine_from" not in run and "continued_from" not in run
    verdict = read_json(parent[1] / "verdict.json")
    assert verdict["stop_rules"]["enabled"] is False and verdict["stop_rules"]["objective_change"] == 1e-4 and verdict["stop"] is None
    assert "fine_from" not in read_json_checkpoint_meta(parent[1])
