"""`topopt verify`: the keys `partial` and `coverage` and the line of text (TC-2, CONTRACTS.md section 44), on the recorded hook run.

A sub-check that fails or does not run is a fact in its own record and verify goes on. The keys sum those records up. The tests force
one failure at a time (a monkeypatch of the sub-check) and look at the record, the text and the exit code; a clean run is `partial: false`.
A real run needs gmsh (`needs_gmsh`) and scikit-fem. The runs use the mesh size 3 mm.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_gmsh, needs_skfem  # noqa: E402
from test_verify_bore_run import CHANNEL, block  # noqa: E402
from test_verify_run import HOOK_YAML, copy_run, needs_recording  # noqa: E402

pytestmark = [needs_gmsh, needs_skfem, needs_recording]

SIZE = 3.0


def yaml_with(tmp_path, **blocks) -> Path:
    """The hook YAML with absolute region paths and the given top-level blocks added."""
    d = yaml.safe_load(HOOK_YAML.read_text())
    base = HOOK_YAML.parent
    r = d["regions"]
    r["design"] = str(base / r["design"])
    r["fixed"] = [str(base / x) for x in r["fixed"]]
    r["keep_out"] = [str(base / x) for x in r["keep_out"]]
    r["load"] = {k: str(base / v) for k, v in r["load"].items()}
    d.update(blocks)
    p = tmp_path / "hook_tc2.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def cli(run, *argv, yaml_path=HOOK_YAML):
    from topopt.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(run), "--yaml", str(yaml_path), "--mesh-size", str(SIZE), "--no-sensitivity", *map(str, argv)])
    return code, out.getvalue(), err.getvalue()


def result(run) -> dict:
    return json.loads((Path(run) / "verify" / "verify.json").read_text())


def names(entries) -> list:
    return [x["name"] if isinstance(x, dict) else x for x in entries]


def test_a_clean_run_is_not_partial_and_lists_what_ran_and_what_the_inputs_did_not_ask_for(tmp_path):
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run)
    assert code == 0, err
    res = result(run)
    assert res["partial"] is False and res["coverage"]["ran"] == ["stress"] and res["coverage"]["failed"] == []
    not_run = {x["name"]: x for x in res["coverage"]["not_run"]}
    assert set(not_run) == {"supports", "buckling", "motion", "bore_facts", "goodman", "pressfit", "gate"}
    assert all(x["asked"] is False and x["reason"] for x in not_run.values())
    assert list(res)[:4] == ["format", "kind", "partial", "coverage"]
    lines = out.splitlines()
    assert lines[0].startswith("verify: ") and lines[1] == "partial: no; 1 of 1 checks ran (stress); 7 not asked"      # near the top
    assert not any(x.startswith("plain: partial") for x in lines)                                                    # a clean run adds no plain line
    assert "buckling" not in lines[1]                                                                                 # the text names no check that was not asked for


def test_a_failure_of_the_buckling_check_makes_the_result_partial_and_changes_nothing_else(tmp_path, monkeypatch):
    clean_run = copy_run(tmp_path / "clean")
    assert cli(clean_run, "--buckling")[0] == 0
    clean = result(clean_run)
    assert clean["partial"] is False and clean["coverage"]["ran"] == ["stress", "buckling"]

    from topopt.verify import buckling_p2
    monkeypatch.setattr(buckling_p2, "run_check", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("phase 33 error -4")))
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, "--buckling")
    assert code == 0, err                                                                  # the exit code does not follow partial
    res = result(run)
    assert res["partial"] is True and res["ok"] is True and res["exit_code"] == 0
    assert res["coverage"]["ran"] == ["stress"]
    assert res["coverage"]["failed"] == [{"name": "buckling", "reason": "RuntimeError: phase 33 error -4"}]
    assert res["buckling"]["status"] == "error"                                            # the record keeps its own fact
    assert {k: v["von_mises"]["p99_9_by_volume_MPa"] for k, v in res["cases"].items()} == \
        {k: v["von_mises"]["p99_9_by_volume_MPa"] for k, v in clean["cases"].items()}      # the numbers are the numbers of the clean run
    lines = out.splitlines()
    assert lines[1] == "partial: 1 of 2 checks failed (buckling: RuntimeError: phase 33 error -4); ran: stress; 6 not asked"
    assert [x for x in lines if x.startswith("plain: ")][0] == f"plain: {lines[1]}"      # the first plain line


def test_a_failure_of_the_rows_is_a_failed_check_and_the_rows_are_not_asked_without_them(tmp_path, monkeypatch):
    from topopt.verify import motion
    monkeypatch.setattr(motion, "motion_record", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad measure STL")))
    run = copy_run(tmp_path / "run")
    limits = {"displacement": [{"name": "tip_pull", "region": "tip", "limit_mm": 0.021, "cases": ["pull_off"]}]}
    code, out, err = cli(run, yaml_path=yaml_with(tmp_path, limits=limits))
    assert code == 0, err
    res = result(run)
    assert res["partial"] is True and res["coverage"]["failed"] == [{"name": "motion", "reason": "ValueError: bad measure STL"}]
    assert "motion" not in names(x for x in res["coverage"]["not_run"] if x["asked"] is False)


def test_a_check_skipped_for_memory_is_asked_and_not_run(tmp_path, monkeypatch):
    from topopt.verify import buckling_p2
    real = buckling_p2.memory_decision

    def tight(cfg, solver, term, after_bytes, lim, force, size):
        _, mem = real(cfg, solver, term, after_bytes, lim, force, size)
        return buckling_p2.skip_record(cfg, solver, "memory", ["The memory estimate with the check is 9 GB, above the limit 8 GB. The check is not started."],
                                       ["A larger max_gb."], memory=mem, size_mm=float(size)), mem
    monkeypatch.setattr(buckling_p2, "memory_decision", tight)
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, "--buckling")
    assert code == 0, err
    res = result(run)
    assert res["partial"] is True and res["coverage"]["failed"] == [] and res["coverage"]["ran"] == ["stress"]
    skipped = [x for x in res["coverage"]["not_run"] if x["name"] == "buckling"]
    assert len(skipped) == 1 and skipped[0]["asked"] is True and skipped[0]["reason"].startswith("The memory estimate with the check is 9 GB")
    assert out.splitlines()[1] == ("partial: 1 of 2 checks not run (buckling: The memory estimate with the check is 9 GB, above the limit 8 GB); "
                                   "ran: stress; 6 not asked")


def test_a_bore_that_stops_is_a_failed_check_and_the_goodman_check_that_needs_it_is_not_run(tmp_path, monkeypatch):
    from topopt.verify import bore

    def boom(model, spec, knobs):
        raise ValueError("cannot read the bore spec")
    monkeypatch.setattr(bore, "build_bore", boom)
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, yaml_path=yaml_with(tmp_path, bore_checks=block()))
    assert code == 0, err
    res = result(run)
    assert res["partial"] is True and res["coverage"]["ran"] == ["stress"]
    assert res["coverage"]["failed"] == [{"name": "bore_facts", "reason": "bore 'channel_top' stopped at the build: ValueError: cannot read the bore spec"}]
    asked_not_run = {x["name"]: x["reason"] for x in res["coverage"]["not_run"] if x["asked"]}
    assert list(asked_not_run) == ["goodman"] and asked_not_run["goodman"].startswith("No bore has a sample set")
    assert "pressfit" in names(x for x in res["coverage"]["not_run"] if not x["asked"])      # no press_fit block: not asked, not a defect
    line = out.splitlines()[1]
    assert line.startswith("partial: 1 of 3 checks failed (bore_facts: bore 'channel_top' stopped at the build: ValueError: cannot read the bore spec); "
                           "1 of 3 checks not run (goodman: No bore has a sample set")


def test_a_clean_run_with_a_bore_block_has_the_bore_checks_under_ran(tmp_path):
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, yaml_path=yaml_with(tmp_path, bore_checks=block(bores=(CHANNEL,))))
    assert code == 0, err
    res = result(run)
    assert res["partial"] is False and res["coverage"]["ran"] == ["stress", "bore_facts", "goodman"] and res["coverage"]["failed"] == []
    assert names(x for x in res["coverage"]["not_run"] if x["asked"]) == []


def test_the_gate_keeps_its_rules_and_the_exit_code_when_the_result_is_partial(tmp_path, monkeypatch):
    from topopt.verify import buckling_p2
    monkeypatch.setattr(buckling_p2, "run_check", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, "--gate", yaml_path=yaml_with(tmp_path, buckling={"min_factor": 2.0}))
    res = result(run)
    assert res["gate"]["passed"] is False and code == 3 and res["exit_code"] == 3             # the gate: a check that could not run fails it (as before)
    assert res["partial"] is True and res["coverage"]["failed"] == [{"name": "buckling", "reason": "RuntimeError: boom"}]
    assert res["coverage"]["ran"] == ["stress", "gate"]                                        # the gate itself ran: it gave its verdict


def test_an_estimate_only_result_has_neither_key(tmp_path):
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    res = verify_run(run, estimate_only=True, yaml_path=HOOK_YAML)
    assert res["estimate_only"] is True and "partial" not in res and "coverage" not in res


def test_a_run_with_two_mesh_sizes_has_one_record(tmp_path):
    run = copy_run(tmp_path / "run")
    from topopt.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(run), "--yaml", str(HOOK_YAML), "--levels", "4,3", "--no-sensitivity"])
    assert code == 0, err.getvalue()
    res = result(run)
    assert res["partial"] is False and res["coverage"]["ran"] == ["stress"] and "levels" in res
