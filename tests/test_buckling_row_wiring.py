"""The buckling row (A6b, CONTRACTS.md section 39): config, refusals, the rows of the formulation, memory facts, the default path.

T9 config, T10 refusals, T11 `formulation_from_config` and `problem_facts`, T12 memory and the lines of `topopt check` and of the run
log, T15 the default path. The resume test (T13) is in `test_buckling_row_resume.py`. Nothing here loads OpenCL.
"""
import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

from helpers import needs_direct_solver
from test_buckling_row import column_model, column_yaml, point_at
from topopt import buckling_row as br
from topopt import buckling_voxel as bv
from topopt import checkpoint, cli, solver_choice
from topopt.cli import check_data, main
from topopt.config import BUCKLING_KS_Q, UserError, load_config
from topopt.optimize import Model, formulation_from_config, problem_facts
from topopt.problem import build_problem


ONE_BETA = {"beta_schedule": [[0, 1]]}      # a short run: the default schedule of `column_yaml` needs 10 passes or more


def edit(path, fn):
    d = yaml.safe_load(path.read_text())
    fn(d)
    path.write_text(yaml.safe_dump(d))
    return path


def with_buckling(path, **keys):
    return edit(path, lambda d: d.setdefault("buckling", {}).update(keys))


def with_settings(path, **keys):
    return edit(path, lambda d: d["settings"].update(keys))


def avail(chol=False, pardiso=False, amg=True):
    """A faked `backend_availability()`."""
    return {"cholesky": {"available": chol, "note": "Apple Accelerate sparse Cholesky" if chol else "Apple silicon only (faked)"},
            "pardiso": {"available": pardiso, "note": "Intel MKL PARDISO (faked)" if pardiso else
                        "pypardiso is not installed. Install it with: uv pip install pypardiso (faked)"},
            "amg": {"available": amg, "note": "PyAMG-preconditioned CG"}}


# --- T9: config --------------------------------------------------------------


def test_the_ks_q_default_of_the_row_is_the_config_default():
    assert br.KS_Q_DEFAULT == BUCKLING_KS_Q == 40.0


def test_config_defaults_and_both_keys(tmp_path):
    b = load_config(column_yaml(tmp_path / "a")).buckling
    assert (b.min_factor, b.ks_q, b.modes, b.check) == (None, None, 6, False)
    p = with_buckling(column_yaml(tmp_path / "b"), min_factor=150, ks_q=20, modes=4)
    b = load_config(p).buckling
    assert (b.min_factor, b.ks_q, b.modes) == (150.0, 20.0, 4)


@pytest.mark.parametrize("key", ["min_factor", "ks_q"])
@pytest.mark.parametrize("value, text", [("x", "number"), (0, "must be > 0"), (-3.0, "must be > 0"), (True, "number")])
def test_config_errors_name_the_key(tmp_path, key, value, text):
    p = with_buckling(column_yaml(tmp_path), **{key: value})
    with pytest.raises(UserError) as e:
        load_config(p)
    assert f"buckling.{key}" in str(e.value) and text in str(e.value)


def test_the_effective_settings_have_the_row_only_when_it_is_on(tmp_path):
    plain = load_config(column_yaml(tmp_path / "a"))
    check_only = load_config(with_buckling(column_yaml(tmp_path / "b"), check=True, modes=4))
    ks_only = load_config(with_buckling(column_yaml(tmp_path / "c"), ks_q=10))
    on = load_config(with_buckling(column_yaml(tmp_path / "d"), min_factor=150))
    for cfg in (plain, check_only, ks_only):
        assert set(checkpoint.effective_settings(cfg)) == {"settings", "material", "load_cases"}
    eff = checkpoint.effective_settings(on)
    assert eff["buckling_row"] == {"min_factor": 150.0, "ks_q": 40.0, "modes": 6}
    assert eff["settings"] == checkpoint.effective_settings(plain)["settings"]          # the other keys are as they were
    # a changed key changes the dict (the hash of the inputs), and the default ks_q is the same as a given 40
    changed = [with_buckling(column_yaml(tmp_path / f"e{i}"), **k)
               for i, k in enumerate(({"min_factor": 151}, {"min_factor": 150, "ks_q": 20}, {"min_factor": 150, "modes": 5}))]
    assert all(checkpoint.effective_settings(load_config(q)) != eff for q in changed)
    same = with_buckling(column_yaml(tmp_path / "f"), min_factor=150, ks_q=40)
    assert checkpoint.effective_settings(load_config(same)) == eff


# --- T10: refusals -----------------------------------------------------------


def run_and_read(p, out, *extra):
    code = main(["run", str(p), "--out", str(out), "--quiet", *extra])
    return code, json.loads((out / "status.json").read_text())


@pytest.mark.parametrize("pardiso", [True, False])
def test_amg_is_refused_before_the_run_with_options_from_the_availability(tmp_path, monkeypatch, pardiso):
    monkeypatch.setattr(cli, "backend_availability", lambda: avail(pardiso=pardiso))
    p = with_buckling(with_settings(column_yaml(tmp_path), solver="amg"), min_factor=150)
    out = tmp_path / "r"
    code, status = run_and_read(p, out)
    text = status["message"]
    assert code == 1 and status["state"] == "error"
    assert "buckling.min_factor needs a direct factor of K" in text and "The solver of this run is amg" in text
    assert "The run is not started" in text and "options:" in text
    assert not (out / "history.jsonl").exists() and not list((out / "snapshots").glob("*.bin"))
    if pardiso:
        assert "Set settings.solver: pardiso in the YAML" in text and "use the check after the run (buckling.check: true)" in text
        assert "No direct solver is available" not in text
    else:
        assert "pardiso: pypardiso is not installed" in text and "Set settings.solver: pardiso" not in text
        assert "No direct solver is available here, so the check after the run (buckling.check) skips too" in text
        assert "use the check after the run" not in text
        assert (solver_choice.LICENSE_NOTE in text) == solver_choice.is_linux_x86_64()


def test_gpu_mg_is_refused_by_the_setting_before_the_backend_problem(tmp_path, monkeypatch):
    def not_called(*a, **k):
        raise AssertionError("backend_problem must not run: gpu-mg needs OpenCL, and the row refusal speaks first")
    monkeypatch.setattr(cli, "backend_problem", not_called)
    monkeypatch.setattr(cli, "backend_availability", lambda: avail(pardiso=True))
    p = with_buckling(with_settings(column_yaml(tmp_path), solver="gpu-mg"), min_factor=150)
    code, status = run_and_read(p, tmp_path / "r")
    assert code == 1 and "The solver of this run is gpu-mg" in status["message"]
    assert br.config_refusal(load_config(p), avail(pardiso=True)).startswith("buckling.min_factor needs a direct factor")


def test_the_check_prints_the_refusal_as_a_warning_and_does_not_stop(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "backend_availability", lambda: avail(pardiso=True))
    p = with_buckling(with_settings(column_yaml(tmp_path), solver="amg"), min_factor=150)
    d = check_data(p)
    assert d["buckling_row"]["will_run"] is False and d["buckling_row"]["reason"] == "solver_not_direct"
    assert [w for w in d["warnings"] if w.startswith("buckling row: buckling.min_factor needs a direct factor")]
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert [ln for ln in out.splitlines() if ln.startswith("warning: buckling row: buckling.min_factor needs a direct factor")]
    assert [ln for ln in out.splitlines() if ln.startswith("buckling row: on in the YAML (min_factor 150")]


def test_a_row_object_on_an_amg_model_refuses_at_its_first_evaluation(tmp_path):
    pytest.importorskip("pyamg")
    m = column_model(tmp_path, extra_settings={"solver": "amg"})
    point = point_at(m, np.full(m.n_design, 0.6))
    with pytest.raises(UserError, match="direct factor of K"):
        br.BucklingRow("axial", 100.0)(point)
    with pytest.raises(UserError, match="direct factor of K"):          # `formulation_from_config` with a model stops at once
        formulation_from_config(with_cfg(m, 100.0), m.n_design, m)


def with_cfg(model, min_factor):
    """The config of the model with `buckling.min_factor`."""
    import dataclasses
    cfg = model.problem.config
    return dataclasses.replace(cfg, buckling=dataclasses.replace(cfg.buckling, min_factor=min_factor))


def test_robust_mode_with_the_row_stops_at_the_start_with_the_text(tmp_path, capsys):
    p = with_buckling(with_settings(column_yaml(tmp_path), robust="eroded"), min_factor=150)
    with pytest.raises(UserError) as e:
        Model(build_problem(load_config(p)))
    text = str(e.value)
    assert text.startswith("settings.robust is eroded, and buckling.min_factor is set") and "Options: set settings.robust to off" in text
    assert text.endswith("The run stops at the start")
    d = check_data(p)
    fact = [w for w in d["warnings"] if w.startswith("settings.robust is eroded, and buckling.min_factor is set")]
    assert len(fact) == 1 and fact[0] in d["derived"]["robust"]["facts"]
    assert main(["check", str(p)]) == 0
    assert "warning: settings.robust is eroded, and buckling.min_factor is set" in capsys.readouterr().out


@needs_direct_solver
def test_robust_mode_with_the_row_ends_a_run_with_exit_1(tmp_path):
    p = with_buckling(with_settings(column_yaml(tmp_path, iterations=3, extra_settings=ONE_BETA), robust="eroded"), min_factor=150)
    out = tmp_path / "r"
    code, status = run_and_read(p, out)
    assert code == 1 and "settings.robust is eroded, and buckling.min_factor is set" in status["message"]
    assert not list((out / "snapshots").glob("*.bin"))


# --- T11: the rows of the formulation and the facts --------------------------


def two_case_yaml(tmp_path, **kw):
    """The column with a second case `side` that has a compliance limit and a stress limit."""
    p = column_yaml(tmp_path, **kw)

    def add(d):
        d["load_cases"].append({"name": "side", "weight": 1.0, "forces": [{"region": "cap", "force": [40.0, 0.0, -200.0]}],
                                "compliance_limit": 1.0e6, "stress_limit": 900.0})
    return edit(p, add)


def test_the_rows_are_in_the_order_volume_compliance_stress_buckling_one_for_each_case(tmp_path):
    p = with_buckling(two_case_yaml(tmp_path), min_factor=150, ks_q=25, modes=3)
    cfg = load_config(p)
    form = formulation_from_config(cfg, 100)
    assert [c.name for c in form.constraints] == ["volume", "compliance:side", "stress:side", "buckling:axial", "buckling:side"]
    rows = [c.evaluate for c in form.constraints[3:]]
    assert [(r.case, r.min_factor, r.q, r.modes) for r in rows] == [(0, 150.0, 25.0, 3), (1, 150.0, 25.0, 3)]
    assert all(c.limit == 1.0 and c.scale == 1.0 and c.stage_aware and c.measure is None for c in form.constraints[3:])
    assert br.INFO_KEYS and form.constraints[3].facts is not None
    # the default ks_q
    form = formulation_from_config(load_config(with_buckling(two_case_yaml(tmp_path / "x"), min_factor=150)), 100)
    assert form.constraints[3].evaluate.q == 40.0


def test_the_rows_without_the_key_are_as_they_are_on_main(tmp_path):
    cfg = load_config(two_case_yaml(tmp_path))
    form = formulation_from_config(cfg, 100)
    assert [c.name for c in form.constraints] == ["volume", "compliance:side", "stress:side"]
    form_ks = formulation_from_config(load_config(with_buckling(two_case_yaml(tmp_path / "k"), ks_q=10)), 100)
    assert [c.name for c in form_ks.constraints] == ["volume", "compliance:side", "stress:side"]


def test_a_mass_run_with_only_buckling_rows_has_no_ceiling_and_a_fact(tmp_path):
    p = with_buckling(with_settings(column_yaml(tmp_path), objective="mass"), min_factor=150)
    cfg = load_config(p)
    form = formulation_from_config(cfg, 100)
    assert [c.name for c in form.constraints] == ["buckling:axial"]
    facts = problem_facts(cfg)
    assert len(facts) == 1 and "the buckling row is the only row that holds material" in facts[0]
    assert "A load case with no compression has no buckling load" in facts[0]
    assert "options:" in facts[0]
    # without the key the old text is there and the ceiling stays
    cfg0 = load_config(with_settings(column_yaml(tmp_path / "o"), objective="mass"))
    assert [c.name for c in formulation_from_config(cfg0, 100).constraints] == ["volume"]
    assert "No row holds material" in problem_facts(cfg0)[0]
    # with a stress limit the buckling row is not the only row: no fact
    cfg2 = load_config(with_buckling(with_settings(two_case_yaml(tmp_path / "s"), objective="mass"), min_factor=150))
    assert not [f for f in problem_facts(cfg2) if "only row that holds material" in f]


def test_the_facts_for_ks_q_without_the_row_and_for_one_mode(tmp_path):
    a = problem_facts(load_config(with_buckling(column_yaml(tmp_path / "a"), ks_q=10)))
    assert len(a) == 1 and a[0].startswith("fact: buckling.ks_q is set, and buckling.min_factor is not set")
    b = problem_facts(load_config(with_buckling(column_yaml(tmp_path / "b"), min_factor=150, modes=1)))
    assert len(b) == 1 and b[0].startswith("fact: buckling.modes is 1 with the buckling row") and "options:" in b[0]
    assert problem_facts(load_config(with_buckling(column_yaml(tmp_path / "c"), min_factor=150))) == []
    assert problem_facts(load_config(column_yaml(tmp_path / "d"))) == []


# --- T12: memory and the lines of the check ---------------------------------


def test_preflight_gives_the_a6a_estimate_and_a_fact_above_the_budget(tmp_path):
    cfg = load_config(with_buckling(column_yaml(tmp_path), min_factor=150))
    grid, free = 120_000, 115_836
    info = br.preflight(cfg, "pardiso", grid, free, ram_bytes=64 * 10 ** 9)
    assert info["will_run"] and info["warnings"] == [] and info["reason"] is None
    assert info["memory_estimate_bytes"] == bv.estimate_bytes(grid, free, cfg.buckling.modes + 1)
    assert info["memory_extra_bytes"] == info["memory_estimate_bytes"] - cli.estimate_memory_bytes(grid, free) > 0
    small = br.preflight(cfg, "pardiso", grid, free, ram_bytes=10 ** 9)
    assert small["will_run"] and len(small["warnings"]) == 1
    w = small["warnings"][0]
    assert w.startswith("buckling row: the memory estimate with the row is") and "above the budget of 1.0 GB" in w
    assert "The run goes on" in w and "options:" in w and "the RAM of this machine" in w
    with_max = br.preflight(load_config(with_buckling(column_yaml(tmp_path / "m"), min_factor=150, max_gb=2.0)), "pardiso",
                            grid, free, ram_bytes=64 * 10 ** 9)
    assert "buckling.max_gb = 2" in with_max["warnings"][0]
    none = br.preflight(cfg, None, grid, free, ram_bytes=64 * 10 ** 9)
    assert not none["will_run"] and none["reason"] == "solver_unavailable" and "buckling row:" in none["warnings"][0]
    line = br.lines(info)[0]
    assert line.startswith("buckling row: on in the YAML (min_factor 150, KS q 40, 6 modes)") and "GB = 1e9 bytes" in line
    assert "time estimate of the run does not include the row" in line


def test_check_has_the_row_key_only_when_the_row_is_on(tmp_path, capsys):
    plain = column_yaml(tmp_path / "a")
    assert "buckling_row" not in check_data(plain)
    on = with_buckling(column_yaml(tmp_path / "b"), min_factor=150)
    d = check_data(on)
    assert d["buckling_row"]["min_factor"] == 150.0 and d["buckling_row"]["memory_estimate_bytes"] > 0
    assert main(["check", str(on), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["buckling_row"]["modes"] == 6
    assert main(["check", str(plain), "--json"]) == 0
    assert "buckling_row" not in json.loads(capsys.readouterr().out)


@needs_direct_solver
def test_a_run_above_the_budget_goes_on_and_logs_the_row_and_the_fact(tmp_path):
    p = with_buckling(column_yaml(tmp_path, iterations=4, extra_settings=ONE_BETA), min_factor=150, max_gb=0.0001)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text()
    assert "buckling row: on in the YAML (min_factor 150" in log
    assert "warning: buckling row: the memory estimate with the row is" in log and "The run goes on" in log
    rows = [json.loads(x) for x in (out / "history.jsonl").read_text().splitlines()]
    entries = [next(c for c in r["constraints"] if c["name"] == "buckling:axial") for r in rows]
    assert [e["evaluated"] for e in entries[:2]] == [True, True]                  # passes 0 and 1 always run
    assert all(e["n_fail"] == 0 for e in entries) and all(set(br.INFO_KEYS) <= set(e) for e in entries)
    assert all(e["residual_adjoint"] <= 1e-6 for e in entries if e["evaluated"])


# --- T15: the default path ---------------------------------------------------


def test_a_default_run_does_not_import_the_row_and_its_entries_keep_their_keys(tmp_path):
    p = column_yaml(tmp_path, iterations=3, extra_settings=ONE_BETA)
    out = tmp_path / "r"
    code = ("import json, sys\nfrom topopt.cli import main\n"
            f"rc = main(['run', {str(p)!r}, '--out', {str(out)!r}, '--quiet'])\n"
            "print(json.dumps({'rc': rc, 'row': 'topopt.buckling_row' in sys.modules}))\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.splitlines()[-1]) == {"rc": 0, "row": False}
    rows = [json.loads(x) for x in (out / "history.jsonl").read_text().splitlines()]
    for r in rows:
        assert [set(c) for c in r["constraints"]] == [{"name", "value", "limit", "g", "violation"}]


def test_the_model_and_a_hand_made_point_keep_their_old_defaults(tmp_path):
    m = column_model(tmp_path)
    assert m.edof.shape == (m.n_el, 24)
    point = point_at(m, np.full(m.n_design, 0.6))
    assert point.iteration is None and point.last is False


def test_the_hook_example_loads_and_checks():
    from pathlib import Path
    example = Path(__file__).resolve().parent.parent / "examples" / "hook" / "hook_buckling.yaml"
    assert load_config(example).buckling.min_factor == 150.0
    d = check_data(example)
    assert d["buckling_row"]["min_factor"] == 150.0 and d["buckling_row"]["modes"] == 6
    assert d["name"] == "tshirt_hook_buckling"
