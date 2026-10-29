"""E1-P3: the measured "up to" value next to the time and memory estimate of `topopt check` (facts only).

The module `topopt.estimate_bounds` is imported inside the tests, so a test that does not need it fails on its own assertion (the
same file runs on the start commit 3e868a7 to show that each test fails there for its reason).
"""
import importlib
import json
import re
from pathlib import Path

import pytest
import yaml

from helpers import needs_pardiso
from test_optimize import make_case
from test_solver_choice import LOAD_ERROR, fake_platform  # noqa: F401  (the fixture and the load error of pypardiso)
from topopt import cli
from topopt.cli import check_data, main

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30
SOURCE = ("experiment E1 (run-time study, 2026-10-23); runs made by the code revisions before main bb6e193; "
          "one shared Linux host; 10,000 free dofs and more")
LINUX = "linux x86_64"
KEYS = ("time_estimate_up_to_s", "memory_estimate_up_to_gib", "estimate_bounds")
BIG = 50_000              # free dofs inside the measured range of E1 (10,000 and more)
SMALL_LINE = "no up-to value: experiment E1 measured problems with 10,000 free dofs and more"


def eb():
    return importlib.import_module("topopt.estimate_bounds")


def plain_case(tmp_path, name="plain", **kw):
    folder = tmp_path / name
    folder.mkdir()
    return make_case(folder, h=kw.pop("h", 2.0), iterations=kw.pop("iterations", 3))


def big_case(tmp_path, name="big", **kw):
    """The case of `plain_case` at 0.8 mm: 12,675 free dofs, inside the measured range of E1 (`plain_case` has 1,080)."""
    return plain_case(tmp_path, name, h=0.8, **kw)


def stress_case(tmp_path):
    p = plain_case(tmp_path, "stress")
    d = yaml.safe_load(p.read_text())
    d["load_cases"][0]["stress_limit"] = 5.0
    p.write_text(yaml.safe_dump(d))
    return p


# --- the module ----------------------------------------------------------------------------------

# (quantity, solver, multiplier, runs, problems) of E1 REPORT 3.3, recomputed in the work of E1-P3: 10,000 free dofs and more.
ROWS = [
    ("time", "pardiso", 3.12, 542, 105), ("time", "amg", 1.60, 22, 12), ("time", "gpu-mg", 2.74, 32, 12),
    ("memory", "pardiso", 1.54, 332, 81), ("memory", "amg", 1.03, 8, 7),
]


@pytest.mark.parametrize("quantity, solver, mult, n_runs, n_problems", ROWS)
def test_each_solver_row_on_linux_gives_the_estimate_times_its_multiplier(quantity, solver, mult, n_runs, n_problems):
    f = eb().bound_facts(solver, LINUX, 100.0, 2.0, free_dofs=BIG)[quantity]
    est = 100.0 if quantity == "time" else 2.0
    assert f["estimate"] == est and f["multiplier"] == mult
    assert f["up_to"] == round(est * mult, 1 if quantity == "time" else 2)
    assert (f["n_runs"], f["n_problems"]) == (n_runs, n_problems)
    assert 0.75 < f["covered_share"] <= 1.0
    assert f["basis"] == SOURCE
    assert "10,000 free dofs and more" in f["note"] and f"{n_runs} E1 runs" in f["note"]
    assert f["line"].startswith("up to about ") and f"{n_runs} runs of experiment E1" in f["line"] and f"{mult:g} times" in f["line"]


@pytest.mark.parametrize("platform", ["darwin arm64", "darwin x86_64", "win32 AMD64"])
@pytest.mark.parametrize("solver", ["cholesky", "pardiso", "amg", "gpu-mg"])
def test_a_platform_other_than_linux_x86_64_has_no_value_and_the_note_says_why(platform, solver):
    facts = eb().bound_facts(solver, platform, 100.0, 2.0, free_dofs=BIG)
    for q in ("time", "memory"):
        f = facts[q]
        assert f["up_to"] is None and f["multiplier"] is None and f["n_runs"] == 0 and f["basis"] == SOURCE
        assert "no finished run on this platform" in f["note"] and platform in f["note"] and "Linux x86_64" in f["note"]
        assert f["line"].startswith("no up-to value: ")


def test_cholesky_on_linux_has_no_value_because_e1_has_no_cholesky_run():
    facts = eb().bound_facts("cholesky", LINUX, 100.0, 2.0, free_dofs=BIG)
    for q in ("time", "memory"):
        f = facts[q]
        assert f["up_to"] is None and f["multiplier"] is None and f["n_runs"] == 0 and f["basis"] == SOURCE
        assert "no finished run with solver cholesky" in f["note"]
    assert eb().bound_facts("superlu", LINUX, 1.0, 1.0, free_dofs=BIG)["time"]["up_to"] is None      # any other name: no E1 run


def test_no_solver_that_can_start_gives_no_value_and_the_note_does_not_name_a_solver():
    facts = eb().bound_facts(None, LINUX, 100.0, 2.0, free_dofs=BIG)
    for q in ("time", "memory"):
        f = facts[q]
        assert f["up_to"] is None and f["n_runs"] == 0 and f["basis"] == SOURCE and "No solver can start" in f["note"]
        assert f["line"] == "no up-to value: no solver can start here" and "cholesky" not in f["note"] + f["line"]
    assert eb().derived_keys(None, LINUX, 100.0, 2.0, free_dofs=BIG)["time_estimate_up_to_s"] is None


def test_a_multiplier_below_one_gives_no_value_and_the_note_gives_the_multiplier_and_n():
    f = eb().bound_facts("gpu-mg", LINUX, 100.0, 2.0, free_dofs=BIG)["memory"]
    assert f["up_to"] is None and f["multiplier"] == 0.73 and f["n_runs"] == 23 and f["n_problems"] == 8
    assert "0.73" in f["note"] and "23 E1 runs" in f["note"] and "below 1" in f["note"]
    assert "Cholesky model value" in f["note"] and "derived.memory_estimate_gib" in f["note"]      # what the multiplier is relative to
    assert "0.73" in f["line"] and "23 runs" in f["line"] and f["basis"] == SOURCE
    assert eb().bound_facts("gpu-mg", LINUX, 100.0, 2.0, free_dofs=BIG)["time"]["up_to"] == 274.0         # the time of gpu-mg is above 1


def test_every_fact_carries_n_and_the_source_text():
    for solver in ("cholesky", "pardiso", "amg", "gpu-mg", "other"):
        for platform in (LINUX, "darwin arm64"):
            facts = eb().bound_facts(solver, platform, 50.0, 1.0, free_dofs=BIG)
            assert set(facts) == {"solver", "platform", "rule", "time", "memory"} and "Facts only" in facts["rule"]
            for q in ("time", "memory"):
                f = facts[q]
                assert isinstance(f["n_runs"], int) and isinstance(f["n_problems"], int) and f["basis"] == SOURCE and f["note"]
                assert json.loads(json.dumps(f)) == f           # plain data: it goes to run.json


def test_free_dofs_is_a_required_keyword_of_the_module_functions():
    import inspect
    for fn in (eb().bound_facts, eb().derived_keys):
        param = inspect.signature(fn).parameters["free_dofs"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is inspect.Parameter.empty
    assert eb().E1_MIN_FREE_DOFS == 10_000


@pytest.mark.parametrize("solver, quantity, n_runs, n_problems, median, q90", [
    ("pardiso", "time", 717, 16, 0.32, 0.71), ("amg", "time", 190, 9, 0.17, 0.74), ("gpu-mg", "time", 105, 5, 0.24, 0.45),
    ("pardiso", "memory", 136, 11, 1.02, 1.34), ("amg", "memory", 32, 3, 0.96, 1.04), ("gpu-mg", "memory", 27, 5, 1.51, 1.86),
])
def test_below_10000_free_dofs_there_is_no_value_and_the_note_gives_the_e1_facts_there(solver, quantity, n_runs, n_problems, median, q90):
    """Review m1: the multipliers are from 10,000 free dofs and more. Below that E1 found the time estimate high (PARDISO median 0.32)."""
    f = eb().bound_facts(solver, LINUX, 100.0, 2.0, stress_rows=1, free_dofs=9_999)[quantity]
    assert f["up_to"] is None and f["multiplier"] is None and f["n_runs"] == 0 and f["covered_share"] is None and f["basis"] == SOURCE
    assert f["line"] == SMALL_LINE
    assert "This problem has 9,999 free dofs." in f["note"] and "measured at 10,000 free dofs and more" in f["note"]
    assert f"E1 has {n_runs} runs with solver {solver} ({n_problems} problems)" in f["note"]
    assert f"was {median:g} times the estimate" in f["note"] and f"at most {q90:g} times it" in f["note"]
    assert "stress rows" not in f["note"]          # the stress facts are of the runs with 10,000 free dofs and more


def test_the_range_starts_at_10000_free_dofs():
    at = eb().bound_facts("pardiso", LINUX, 100.0, 2.0, free_dofs=10_000)
    below = eb().bound_facts("pardiso", LINUX, 100.0, 2.0, free_dofs=9_999)
    assert at["time"]["up_to"] == 312.0 and at["memory"]["up_to"] == 3.08
    assert below["time"]["up_to"] is None and below["memory"]["up_to"] is None
    assert eb().derived_keys("pardiso", LINUX, 100.0, 2.0, free_dofs=9_999)["time_estimate_up_to_s"] is None


def test_the_order_of_the_reasons_the_platform_and_the_solver_come_before_the_size():
    assert "darwin arm64" in eb().bound_facts("pardiso", "darwin arm64", 1.0, 1.0, free_dofs=10)["time"]["line"]
    assert "solver cholesky" in eb().bound_facts("cholesky", LINUX, 1.0, 1.0, free_dofs=10)["time"]["line"]
    assert "no solver can start" in eb().bound_facts(None, LINUX, 1.0, 1.0, free_dofs=10)["time"]["line"]


def test_the_gpu_mg_time_note_says_that_e1_ran_gpu_mg_before_the_deflation_of_case_c():
    """Review m2: the deflation of case C came to gpu-mg after the E1 runs."""
    for dofs in (BIG, 9_999):
        facts = eb().bound_facts("gpu-mg", LINUX, 100.0, 2.0, free_dofs=dofs)
        assert "before the deflation of case C" in facts["time"]["note"] and "before the deflation of case C" not in facts["memory"]["note"]
        assert "before the deflation of case C" not in eb().bound_facts("pardiso", LINUX, 100.0, 2.0, free_dofs=dofs)["time"]["note"]
    assert eb().bound_facts("gpu-mg", LINUX, 100.0, 2.0, free_dofs=BIG)["time"]["up_to"] == 274.0


def test_no_estimate_gives_no_value():
    f = eb().bound_facts("gpu-mg", LINUX, None, 2.0, free_dofs=BIG)["time"]          # a CPU OpenCL device has no seconds
    assert f["up_to"] is None and f["estimate"] is None and f["n_runs"] == 32 and "no point estimate" in f["note"]


def test_stress_rows_add_the_e1_facts_to_the_notes_and_leave_the_value_alone():
    plain = eb().bound_facts("pardiso", LINUX, 100.0, 2.0, free_dofs=BIG)
    stress = eb().bound_facts("pardiso", LINUX, 100.0, 2.0, stress_rows=2, free_dofs=BIG)
    assert stress["time"]["up_to"] == plain["time"]["up_to"] and stress["memory"]["up_to"] == plain["memory"]["up_to"]
    assert "stress rows" not in plain["time"]["note"]
    assert "53 E1 runs with stress rows" in stress["time"]["note"] and "median of 2.23" in stress["time"]["note"] and "4.01" in stress["time"]["note"]
    assert "51 E1 runs with stress rows" in stress["memory"]["note"] and "2.83" in stress["memory"]["note"]
    assert "no run with stress rows and solver amg" in eb().bound_facts("amg", LINUX, 100.0, 2.0, stress_rows=1, free_dofs=BIG)["time"]["note"]


def test_the_contracts_table_has_the_multipliers_and_the_runs_of_the_module():
    text = (ROOT / "CONTRACTS.md").read_text()
    section = text[text.index("### 5.1 The up-to values of the estimates"):text.index("### `status.json`")]
    flat = " ".join(section.split())
    assert "Facts only" in flat and "No warning, refusal, budget or RAM comparison" in flat
    for piece in ("run-time study, 2026-10-23", "main bb6e193", "one shared Linux host", "10,000 free dofs and more", "before the deflation of case C"):
        assert piece in flat, piece
    assert "79fe842" not in flat          # Review n1: a local research repository; a reader cannot find it
    mod = eb()
    for quantity, rows in (("time", mod.TIME_ROWS), ("memory", mod.MEMORY_ROWS)):
        for solver, row in rows.items():
            line = [l for l in section.splitlines() if l.startswith(f"| {quantity} | `{solver}` |")]
            assert len(line) == 1, (quantity, solver)
            cells = [c.strip() for c in line[0].strip("|").split("|")]
            assert (float(cells[2]), int(cells[3]), int(cells[4]), float(cells[5])) == (row.multiplier, row.n_runs, row.n_problems, row.covered_share)
    s = mod.STRESS_RUNS["pardiso"]
    assert f"{s.time_multiplier:g}" in section and f"{s.memory_multiplier:g}" in section
    for quantity, rows in (("time", mod.SMALL_TIME_ROWS), ("memory", mod.SMALL_MEMORY_ROWS)):
        for solver, row in rows.items():
            line = [l for l in section.splitlines() if l.startswith(f"| {quantity} below 10,000 free dofs | `{solver}` |")]
            assert len(line) == 1, (quantity, solver)
            cells = [c.strip() for c in line[0].strip("|").split("|")]
            assert (int(cells[2]), int(cells[3]), float(cells[4]), float(cells[5])) == tuple(row)


def test_no_e1_text_names_the_local_research_repository():
    """Review n1: the texts name experiment E1 by its date, not by the SHA of a local research repository."""
    for name in ("README.md", *(f"docs/reference/{p.name}" for p in sorted((ROOT / "docs" / "reference").glob("*.md"))), "CONTRACTS.md", "topopt/cli.py", "topopt/solver_choice.py", "topopt/estimate_bounds.py",
                 ".claude/skills/part-to-topopt/SKILL.md", ".agents/skills/part-to-topopt/SKILL.md"):
        assert "79fe842" not in (ROOT / name).read_text(), name


# --- the keys: check, check_problem and run.json -----------------------------------------------

def test_check_json_check_problem_and_run_json_have_the_same_keys_and_values(tmp_path):
    from topopt_mcp import server
    p = big_case(tmp_path)
    via_check_data = check_data(str(p))["derived"]
    assert all(k in via_check_data for k in KEYS)
    reply = server.check_problem(str(p))
    assert reply["ok"] and {k: reply["derived"][k] for k in KEYS} == {k: via_check_data[k] for k in KEYS}
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    run_derived = json.loads((out / "run.json").read_text())["derived"]
    assert {k: run_derived[k] for k in KEYS} == {k: via_check_data[k] for k in KEYS}
    assert run_derived["time_estimate_s"] == via_check_data["time_estimate_s"] and run_derived["memory_estimate_gib"] == via_check_data["memory_estimate_gib"]
    eab = run_derived["estimate_bounds"]
    assert eab["time"]["estimate"] == run_derived["time_estimate_s"] and eab["memory"]["estimate"] == run_derived["memory_estimate_gib"]
    assert run_derived["time_estimate_up_to_s"] == eab["time"]["up_to"] and run_derived["memory_estimate_up_to_gib"] == eab["memory"]["up_to"]


def test_check_json_has_the_keys(tmp_path, capsys):
    p = plain_case(tmp_path)
    assert main(["check", "--json", str(p)]) == 0
    derived = json.loads(capsys.readouterr().out)["derived"]
    assert all(k in derived for k in KEYS)
    assert derived["estimate_bounds"]["solver"] == check_data(str(p))["solver"]["resolved"] or derived["estimate_bounds"]["solver"] == "cholesky"


@needs_pardiso
def test_the_values_of_a_pardiso_problem_on_linux(tmp_path):
    d = check_data(str(big_case(tmp_path)))["derived"]
    assert d["estimate_bounds"]["solver"] == "pardiso" and d["estimate_bounds"]["platform"] == LINUX
    assert d["time_estimate_up_to_s"] == round(d["time_estimate_s"] * 3.12, 1)
    assert d["memory_estimate_up_to_gib"] == round(d["memory_estimate_gib"] * 1.54, 2)
    assert d["estimate_bounds"]["time"]["n_runs"] == 542 and d["estimate_bounds"]["memory"]["n_runs"] == 332


def test_a_mac_has_no_value_in_check_check_problem_and_run_json(tmp_path, monkeypatch, capsys):
    """E1 has no Mac run: the keys are there, null, with the note."""
    from topopt_mcp import server
    monkeypatch.setattr(cli, "platform_name", lambda: "darwin arm64")
    p = plain_case(tmp_path)
    d = check_data(str(p))["derived"]
    assert d["time_estimate_up_to_s"] is None and d["memory_estimate_up_to_gib"] is None
    for q in ("time", "memory"):
        f = d["estimate_bounds"][q]
        assert f["up_to"] is None and f["n_runs"] == 0 and "darwin arm64" in f["note"] and f["basis"] == SOURCE
    assert server.check_problem(str(p))["derived"]["estimate_bounds"] == d["estimate_bounds"]
    assert main(["check", str(p)]) == 0
    text = capsys.readouterr().out
    time_line = [l for l in text.splitlines() if l.startswith("time estimate:")][0]
    memory_line = [l for l in text.splitlines() if l.startswith("memory estimate:")][0]
    assert time_line.endswith("(no up-to value: experiment E1 has no run on this platform (darwin arm64))")
    assert memory_line.endswith("(no up-to value: experiment E1 has no run on this platform (darwin arm64))")
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert json.loads((out / "run.json").read_text())["derived"]["estimate_bounds"]["time"]["up_to"] is None


@needs_pardiso
def test_a_small_problem_has_no_value_in_check_check_problem_and_the_text(tmp_path, capsys):
    """Review m1: `plain_case` has 1,080 free dofs, below the measured range of E1."""
    from topopt_mcp import server
    p = plain_case(tmp_path)
    data = check_data(str(p))
    assert data["free_dofs"] < 10_000 and data["derived"]["estimate_bounds"]["solver"] == "pardiso"
    d = data["derived"]
    assert d["time_estimate_up_to_s"] is None and d["memory_estimate_up_to_gib"] is None
    assert d["estimate_bounds"]["time"]["line"] == SMALL_LINE and f"This problem has {data['free_dofs']:,} free dofs" in d["estimate_bounds"]["time"]["note"]
    assert server.check_problem(str(p))["derived"]["estimate_bounds"] == d["estimate_bounds"]
    assert main(["check", str(p)]) == 0
    text = capsys.readouterr().out
    assert text.count(f"({SMALL_LINE})") == 2


def test_check_with_a_pardiso_that_does_not_load_has_no_up_to_value_and_names_no_cholesky(tmp_path, capsys, fake_platform):
    """`solver: none` (exit 1): the estimates still use the model of cholesky, but no run takes them, so the lines say that."""
    fake_platform("linux", "x86_64", pypardiso=LOAD_ERROR)
    p = plain_case(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "pardiso"
    p.write_text(yaml.safe_dump(d))
    data = check_data(str(p))
    assert data["solver"]["resolved"] is None
    assert data["derived"]["time_estimate_up_to_s"] is None and data["derived"]["memory_estimate_up_to_gib"] is None
    assert data["derived"]["estimate_bounds"]["solver"] is None and data["derived"]["estimate_bounds"]["time"]["n_runs"] == 0
    assert main(["check", str(p)]) == 1
    text = capsys.readouterr().out
    assert text.count("(no up-to value: no solver can start here)") == 2 and "cholesky" not in text


def test_the_time_line_and_the_memory_line_of_check_end_with_the_up_to_text(tmp_path, capsys):
    p = plain_case(tmp_path)
    d = check_data(str(p))["derived"]
    assert main(["check", str(p)]) == 0
    text = capsys.readouterr().out
    time_line = [l for l in text.splitlines() if l.startswith("time estimate:")][0]
    memory_line = [l for l in text.splitlines() if l.startswith("memory estimate:")][0]
    assert time_line.endswith(f"({d['estimate_bounds']['time']['line']})")
    assert memory_line.endswith(f"({d['estimate_bounds']['memory']['line']})")
    assert text.count("estimate bounds: measured in " + SOURCE) == 1 and "Facts only: no warning or refusal uses them" in text


def test_the_gpu_mg_memory_line_says_what_the_multiplier_is_relative_to(tmp_path, monkeypatch, capsys):
    from topopt.gpu_mg import device
    from test_gpu_mg_device import dev
    monkeypatch.setattr(device, "pyopencl_problem", lambda: None)
    monkeypatch.setattr(device, "enumerate_devices", lambda: [dev(0, "Fast GPU", "GPU", True, platform_name="P", vendor="V")])
    monkeypatch.setattr(cli, "opencl_facts", lambda settings=None, solver_setting=None: device.facts(getattr(settings, "gpu_device", "auto")))
    p = plain_case(tmp_path, h=0.8, iterations=20)        # 12,675 free dofs: inside the measured range
    d = check_data(str(p), solver_setting="gpu-mg")
    b = d["derived"]["estimate_bounds"]
    assert b["solver"] == "gpu-mg" and d["derived"]["memory_estimate_gib"] != round(d["memory_estimate"]["bytes"] / GIB, 2)   # derived: the Cholesky number
    assert d["derived"]["time_estimate_up_to_s"] == b["time"]["up_to"] == (round(d["derived"]["time_estimate_s"] * 2.74, 1) if b["platform"] == LINUX else None)
    assert main(["check", str(p), "--solver", "gpu-mg"]) == 0
    memory_line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("memory estimate:")][0]
    assert memory_line.endswith(f"({b['memory']['line']})")
    if b["platform"] == LINUX:
        assert d["derived"]["memory_estimate_up_to_gib"] is None and "Cholesky model value" in memory_line


# --- facts only: nothing uses the value ----------------------------------------------------------

def test_no_warning_or_refusal_uses_the_up_to_values(tmp_path, monkeypatch, capsys):
    """A multiplier of 1,000 puts every up-to value far above the budget and the RAM; the warnings and the exit code do not change."""
    monkeypatch.setattr(cli, "physical_ram_bytes", lambda: 16 * GIB)
    p, s = big_case(tmp_path), stress_case(tmp_path)

    def snap():
        out = []
        for q in (p, s):
            code = main(["check", str(q)])
            printed = [l for l in capsys.readouterr().out.splitlines() if l.startswith("warning:")]
            out.append((code, check_data(str(q))["warnings"], printed))
        return out
    before = snap()
    mod = eb()
    monkeypatch.setattr(mod, "TIME_ROWS", {k: mod.Row(1000.0, v.n_runs, v.n_problems, v.covered_share) for k, v in mod.TIME_ROWS.items()})
    monkeypatch.setattr(mod, "MEMORY_ROWS", {k: mod.Row(1000.0, v.n_runs, v.n_problems, v.covered_share) for k, v in mod.MEMORY_ROWS.items()})
    d = check_data(str(p))["derived"]
    if d["estimate_bounds"]["platform"] == LINUX and d["estimate_bounds"]["solver"] in mod.TIME_ROWS:
        assert d["time_estimate_up_to_s"] == round(d["time_estimate_s"] * 1000, 1) and d["memory_estimate_up_to_gib"] > 16
    assert snap() == before


# --- no "conservative" ---------------------------------------------------------------------------

def test_check_text_and_json_do_not_say_conservative_or_safe_side(tmp_path, monkeypatch, capsys):
    p, s = plain_case(tmp_path), stress_case(tmp_path)
    for q in (p, s):                                   # both warn about the time budget of the user (P9-8: no demo budget)
        d = yaml.safe_load(q.read_text())
        d["settings"]["stop_rules"] = {"time_budget_s": 0.001}
        q.write_text(yaml.safe_dump(d))
    for q in (p, s):
        for argv in (["check", str(q)], ["check", "--json", str(q)]):
            assert main(argv) == 0
            cap = capsys.readouterr()
            text = (cap.out + cap.err).lower()
            assert "conservative" not in text and "safe side" not in text, argv
    assert cli.estimate_seconds.__doc__ and "safe side" not in cli.estimate_seconds.__doc__
    assert "conservative" not in json.dumps(cli.backend_estimates(30_000, 40)).lower()


def test_the_part_builder_does_not_say_conservative(tmp_path, monkeypatch, capsys):
    from topopt import regions_builder
    from test_part_skill import SPEC, br, write
    s = json.loads(json.dumps(SPEC))
    s["settings"]["voxel_size"] = 1.0
    s["settings"]["stop_rules"] = {"time_budget_s": 40}   # P9-8: the budget is the user's own; there is no demo budget
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "WARNING: the estimate is above your time budget of 40 s" in out and "(the model)" in out
    assert "conservative" not in out.lower() and "safe side" not in out.lower()


def test_no_text_of_the_package_or_the_docs_calls_a_run_estimate_conservative():
    """The hits that stay: the verify memory model (not measured by E1), the Mac M4 sentence of CONTRACTS, and a mesh note."""
    hits = []
    for path in [*sorted((ROOT / "topopt").rglob("*.py")), *sorted((ROOT / "topopt_mcp").rglob("*.py")), ROOT / "README.md", *sorted((ROOT / "docs" / "reference").glob("*.md")),
                 ROOT / ".claude/skills/part-to-topopt/SKILL.md", ROOT / ".agents/skills/part-to-topopt/SKILL.md"]:
        for no, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"conservative|safe side", line, re.I):
                hits.append(f"{path.relative_to(ROOT)}:{no}")
    assert hits == ["topopt/verify/memory.py:6"], hits


def test_the_99_s_of_the_hook_is_not_called_typical():
    """The 99 s is the best of 62 runs (E1: median 462 s). Each text that quotes it says so."""
    for name in ("docs/reference/install.md", "topopt/cli.py", "topopt/solver_choice.py", "CONTRACTS.md"):
        text = (ROOT / name).read_text()
        assert "462 s" in text and "62 runs" in text, name


def test_the_stress_claim_1_4_to_1_9_is_gone():
    for name in ("topopt/cli.py", "CONTRACTS.md"):
        text = " ".join(re.sub(r"^\s*#", " ", (ROOT / name).read_text(), flags=re.M).split())
        assert "took 1.4 to 1.9 times" not in text and "1.4 to 1.9 times the model" not in text, name
        assert "53 finished PARDISO runs with stress rows in 6 problems" in text and "median of 2.23 times the model" in text, name
    from topopt.cli import STRESS_RUN_TIME_FACTOR
    assert STRESS_RUN_TIME_FACTOR == 1.6          # the factor and the warning point stay


def test_check_problem_description_names_the_key_and_the_key_text_stays_early():
    import asyncio
    from topopt_mcp.server import mcp
    from topopt_mcp import replies
    tool = [t for t in asyncio.run(mcp.list_tools()) if t.name == "check_problem"][0]
    # P9-11: the line that explains the keys of the reply moved to the guide topopt://replies/check_problem; the tool text names the guide
    assert "topopt://replies/check_problem" in tool.description and "`estimate_bounds`" in replies.guide("check_problem")
    assert len(tool.description) <= 4096
    assert tool.description.index("Use it before run_topopt") < 2048
