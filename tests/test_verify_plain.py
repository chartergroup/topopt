"""P9-17: "will it break?" in one plain line for each load case (`topopt.verify.report.plain_lines`), the `plain` key of the
MCP tool `verify_status`, and the lines of `topopt verify`. The offer in the skills is in `tests/test_verify_plain_offer.py`.

Most tests use small result dicts that are built with the real `limit_block` and `no_allowable_fact` (so the keys are the real keys)
and need no gmsh. One test runs the real command on the recorded hook run (needs gmsh and scikit-fem; about 25 s on CPU).
"""
import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import needs_gmsh, needs_skfem, recorded_hook_run  # noqa: E402
from topopt.verify import names  # noqa: E402
from topopt.verify.report import plain_lines  # noqa: E402
from topopt.verify.run import MATERIAL_REFERENCE_URI, limit_block, no_allowable_fact  # noqa: E402

RECORDING = recorded_hook_run()
PROMISES = ("safe", "will not break", "guaranteed", "guarantee", "certain", "will hold", "won't break", "strong enough",
            "no risk")


def vm(value, peak=None, percentile=99.9):
    """The `von_mises` block of one case, with the keys that `limit_block` and `plain_lines` read."""
    return dict(percentile=percentile, percentile_by_volume_MPa=value, p99_9_by_volume_MPa=value,
                peak_qp_MPa=peak if peak is not None else 1.8 * value)


def result(values, allowable=None, percentile=99.9, peaks=None, **extra):
    """A small verify result: the cases, and `limit` (with an allowable) or `no_allowable` (without one)."""
    cases = {n: {"von_mises": vm(v, (peaks or {}).get(n), percentile)} for n, v in values.items()}
    res = {"run": {"name": "x"}, "cases": cases, **extra}
    if allowable is None:
        res["no_allowable"] = no_allowable_fact()
    else:
        res["limit"] = limit_block({"size_mm": 1.5, "cases": cases}, allowable, percentile)
    return res


def text(lines):
    return "\n".join(lines)


# -- the line function -------------------------------------------------------------------------------------------------

def test_a_case_below_the_allowable_has_the_value_the_source_and_the_margin():
    lines = plain_lines(result({"hang": 2.0}, allowable=5.0))
    assert len(lines) == 2                                           # the case line, and the line for what the check does not cover
    [line] = lines[:1]
    assert line.startswith("case hang: ")
    assert "99.9 % of the volume has von Mises stress at or below 2 MPa (p99.9 by volume, the value that topopt compares)" in line
    assert "The allowable is 5 MPa, given by the user (--allowable-mpa); topopt has no default." in line
    assert "The value is below the allowable, margin 2.50 (allowable divided by value)." in line


def test_a_case_above_the_allowable_says_above():
    [line, _] = plain_lines(result({"pull_off": 6.0}, allowable=5.0))
    assert "The value is above the allowable, margin 0.83 (allowable divided by value)." in line
    assert "below the allowable" not in line


def test_a_value_equal_to_the_allowable_is_not_called_above_or_below():
    [line, _] = plain_lines(result({"c": 5.0}, allowable=5.0))
    assert "The value is equal to the allowable, margin 1.00" in line


def test_the_percentile_of_the_limit_block_is_named():
    [line, _] = plain_lines(result({"c": 2.0}, allowable=5.0, percentile=95.0))
    assert "95 % of the volume" in line and "p95 by volume" in line


def test_a_peak_above_the_allowable_is_named_when_the_compared_value_is_below_it():
    res = result({"a": 2.0, "b": 2.0}, allowable=5.0, peaks={"a": 7.5, "b": 4.0})
    la, lb, _ = plain_lines(res)
    assert "The highest value at one point, 7.5 MPa, is above the allowable; the check compares the p99.9 value only." in la
    assert "highest value" not in lb
    over = plain_lines(result({"a": 6.0}, allowable=5.0, peaks={"a": 9.0}))[0]
    assert "highest value" not in over                              # the compared value is above already


def test_a_case_with_no_load_has_nothing_to_compare():
    res = result({"idle": 1.0, "pull": 1.0}, allowable=5.0)
    res["limit"]["cases"]["idle"].update(value_MPa=0.0, margin=float("inf"), exceeded=False)       # as limit_block gives it for a stress of 0
    res = json.loads(json.dumps(res))                                                             # verify.json holds Infinity: read it back
    idle, pull, _ = plain_lines(res)
    assert idle.startswith("case idle: the stress is 0 MPa: no load reaches the free part in this case")
    assert "0 MPa" in idle and "5 MPa" in idle and "margin" not in idle and "inf" not in idle
    assert "margin 5.00" in pull


def test_several_cases_give_one_line_each_in_the_order_of_the_result():
    lines = plain_lines(result({"hang": 0.08, "pull_off": 2.06, "twist": 7.0}, allowable=5.0))
    assert [ln.split(":")[0] for ln in lines[:3]] == ["case hang", "case pull_off", "case twist"]
    assert len(lines) == 4
    assert "below the allowable" in lines[0] and "below the allowable" in lines[1] and "above the allowable" in lines[2]


def test_no_allowable_gives_the_value_no_yes_or_no_and_where_to_look():
    lines = plain_lines(result({"hang": 2.0, "pull_off": 3.0}))
    assert len(lines) == 4                                          # two cases, the allowable line, the scope line
    for ln, v in zip(lines[:2], ("2", "3")):
        assert f"at or below {v} MPa (p99.9 by volume)." in ln
        assert "No allowable was given, so there is no yes or no for this case." in ln
        assert "margin" not in ln and "given by the user" not in ln
    assert MATERIAL_REFERENCE_URI in lines[2] and "not the values of your material" in lines[2]
    assert "Give an allowable (--allowable-mpa) to get a comparison." in lines[2]


def test_the_option_name_follows_the_style_of_the_caller():
    res_limit, res_none = result({"c": 2.0}, allowable=5.0), result({"c": 2.0})
    with names.use_style("mcp"):
        assert "(allowable_mpa)" in text(plain_lines(res_limit)) and "--allowable-mpa" not in text(plain_lines(res_limit))
        assert "Give an allowable (allowable_mpa)" in text(plain_lines(res_none))
    assert "(--allowable-mpa)" in text(plain_lines(res_limit))


def test_a_result_with_no_allowable_key_at_all_still_points_to_the_reference():
    res = result({"c": 2.0})
    del res["no_allowable"]                                         # a result of an older version
    assert MATERIAL_REFERENCE_URI in text(plain_lines(res))


def test_odd_case_names_stay_on_one_line_each():
    odd = ["pull off", 'say "hi"', "line\nbreak\ttab", "ü-ß_日本", "  padded  ", "a:b"]
    res = result({n: 1.0 for n in odd}, allowable=5.0)
    lines = plain_lines(res)
    assert len(lines) == len(odd) + 1
    assert all("\n" not in ln and "\t" not in ln for ln in lines)
    assert lines[2].startswith("case line break tab: ")
    assert lines[3].startswith("case ü-ß_日本: ") and lines[4].startswith("case padded: ")
    assert lines[1].startswith('case say "hi": ')


@pytest.mark.parametrize("allowable", [None, 5.0, 1.0])
def test_no_line_promises_anything(allowable):
    low = text(plain_lines(result({"a": 0.5, "b": 2.0}, allowable=allowable, peaks={"a": 9.0}))).lower()
    for word in PROMISES:
        assert word not in low, word


def test_the_scope_line_names_what_the_check_does_not_cover():
    scope = plain_lines(result({"c": 2.0}, allowable=5.0))[-1]
    assert scope.startswith("This check is the linear static stress of one mesh, one load case at a time, with one isotropic material.")
    assert "It does not cover fatigue, buckling, printing defects or print direction." in scope


def test_a_check_that_ran_is_not_listed_as_not_covered():
    both = plain_lines(result({"c": 2.0}, allowable=5.0, buckling={"status": "done"}, goodman={"status": "done"}))[-1]
    assert "It does not cover printing defects or print direction." in both
    skipped = plain_lines(result({"c": 2.0}, allowable=5.0, buckling={"status": "skipped"}))[-1]
    assert "fatigue, buckling, printing defects or print direction" in skipped


def test_a_result_with_no_load_case_gives_no_lines():
    assert plain_lines({"run": {"name": "x"}, "cases": {}}) == []


# -- verify_status --------------------------------------------------------------------------------------------------------

@pytest.fixture
def server(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    from topopt_mcp import server as srv
    srv._verify.clear()
    return srv


def make_folder(tmp_path, res, state="done", stl=b"solid x\nendsolid x\n"):
    """A run folder with `verify/verify.json` (the result `res`) and a status file in `state`."""
    from topopt.verify.status import StatusWriter
    d = tmp_path / "runs" / "part"
    (d / "result").mkdir(parents=True)
    (d / "result" / "final.stl").write_bytes(stl)
    out = d / "verify"
    out.mkdir()
    res = dict(res, ok=True, exit_code=0, files={"verify": "verify.json"},
               run={"name": "x", "verified_file": "final.stl", "final_stl_sha256": hashlib.sha256(stl).hexdigest()})
    (out / "verify.json").write_text(json.dumps(res))
    sw = StatusWriter(out, run_dir=str(d), sizes_mm=[1.5], solver="fake", estimated_seconds_total=1.0)
    sw.update(state, exit_code=0, message=state, **({"error": "x"} if state == "error" else {}))
    return d


def test_verify_status_of_a_done_job_has_the_plain_lines(server, tmp_path):
    d = make_folder(tmp_path, result({"hang": 0.5, "pull_off": 6.0}, allowable=5.0))
    st = server.verify_status(run_dir=str(d))
    assert st["state"] == "done" and st["result"]["limit"]["allowable_MPa"] == 5.0
    assert len(st["plain"]) == 3 and st["plain"][0].startswith("case hang: ")
    assert "given by the user (allowable_mpa)" in st["plain"][0]            # the MCP name, not the option of the command
    assert "above the allowable" in st["plain"][1]
    assert st["plain"] == plain_lines_mcp(st["result"])


def plain_lines_mcp(res):
    with names.use_style("mcp"):
        return plain_lines(res)


def test_verify_status_with_no_allowable_has_the_plain_lines_and_the_reference(server, tmp_path):
    d = make_folder(tmp_path, result({"hang": 0.5}))
    st = server.verify_status(run_dir=str(d))
    assert "no yes or no" in st["plain"][0] and MATERIAL_REFERENCE_URI in st["plain"][1]


def test_verify_status_with_result_false_has_no_plain_key(server, tmp_path):
    d = make_folder(tmp_path, result({"hang": 0.5}, allowable=5.0))
    st = server.verify_status(run_dir=str(d), result=False)
    assert st["state"] == "done" and "result" not in st and "plain" not in st


@pytest.mark.parametrize("state", ["solving", "meshing", "error"])
def test_verify_status_of_a_job_that_is_not_done_has_no_plain_key(server, tmp_path, state):
    d = make_folder(tmp_path, result({"hang": 0.5}, allowable=5.0), state=state)      # an old verify.json beside a job that works
    st = server.verify_status(run_dir=str(d))
    assert st["state"] != "done" and "plain" not in st and "result" not in st


def test_verify_status_with_no_job_and_no_result_has_no_plain_key(server, tmp_path):
    d = tmp_path / "runs" / "empty"
    d.mkdir(parents=True)
    st = server.verify_status(run_dir=str(d))
    assert st["state"] == "none" and "plain" not in st


def test_verify_status_of_a_result_with_no_load_case_has_no_plain_key(server, tmp_path):
    d = make_folder(tmp_path, {"cases": {}})                    # what the fake child of tests/test_verify_mcp.py writes
    st = server.verify_status(run_dir=str(d))
    assert st["state"] == "done" and "plain" not in st


def test_a_result_that_plain_lines_cannot_read_still_comes_back_without_plain(server, tmp_path):
    d = make_folder(tmp_path, {"cases": {"c": {}}, "limit": {"cases": {"c": {}}}})        # keys of another version
    st = server.verify_status(run_dir=str(d))
    assert st["state"] == "done" and "result" in st and "plain" not in st


def test_a_result_for_an_older_mesh_file_says_so_in_the_first_line(server, tmp_path):
    d = make_folder(tmp_path, result({"hang": 0.5}, allowable=5.0))
    (d / "result" / "final.stl").write_bytes(b"solid y\nendsolid y\n")                  # the file changed after the check
    st = server.verify_status(run_dir=str(d))
    assert st["result_matches_final_stl"] is False
    assert st["plain"][0].startswith("These lines belong to an older mesh file") and st["plain"][1].startswith("case hang: ")


def test_the_description_of_verify_status_names_the_key_and_stays_short():
    import asyncio
    from topopt_mcp.server import mcp
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    doc = tools["verify_status"].description
    assert "`plain`" in doc and "will it break" in doc and len(doc) <= 2048          # under the cut of an old client
    assert len(tools["verify_run"].description) <= 4096


# -- the command line -----------------------------------------------------------------------------------------------------

def run_cli(argv, monkeypatch, res):
    """`topopt verify` with the verify engine replaced by `res` (a fixture: the summary lines are a stub)."""
    from topopt import cli
    from topopt.verify import report, run
    monkeypatch.setattr(run, "verify_run", lambda *a, **k: res)
    monkeypatch.setattr(report, "summary_lines", lambda r: ["SUMMARY-1", "SUMMARY-2"])
    monkeypatch.setattr(report, "estimate_lines", lambda r: ["ESTIMATE-1"])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(["verify", "somewhere", *argv])
    return code, out.getvalue().splitlines()


def test_the_command_prints_the_plain_lines_after_its_summary(monkeypatch):
    res = dict(result({"hang": 0.5, "pull_off": 6.0}, allowable=5.0), exit_code=3)
    code, out = run_cli(["--allowable-mpa", "5"], monkeypatch, res)
    assert code == 3 and out[:2] == ["SUMMARY-1", "SUMMARY-2"]
    assert out[2:] == ["plain: " + ln for ln in plain_lines(res)]
    assert len(out) == 2 + 3 and "(--allowable-mpa)" in out[2] and "above the allowable" in out[3]


def test_the_command_with_no_allowable_prints_where_values_are_and_what_it_does_not_cover(monkeypatch):
    res = dict(result({"hang": 0.5, "pull_off": 6.0}), exit_code=0)
    code, out = run_cli([], monkeypatch, res)
    assert code == 0 and out[:2] == ["SUMMARY-1", "SUMMARY-2"]
    assert out[2:] == ["plain: " + ln for ln in plain_lines(res)]
    assert len(out) == 2 + 4 and all("no yes or no" in ln for ln in out[2:4])
    assert out[4].startswith("plain: No allowable stress was given.") and "(--allowable-mpa)" in out[4]
    assert out[5].startswith("plain: This check is") and "It does not cover" in out[5]
    low = "\n".join(out[2:]).lower()
    assert not any(w in low for w in PROMISES)


def test_the_command_prints_no_plain_line_for_an_estimate(monkeypatch):
    code, out = run_cli(["--estimate-only"], monkeypatch, {"estimate_only": True, "exit_code": 0, "cases": {}})
    assert code == 0 and out == ["ESTIMATE-1"]


@needs_gmsh
@needs_skfem
@pytest.mark.skipif(not (RECORDING / "result" / "final.stl").is_file(), reason="needs the recorded hook run")
def test_a_real_verify_of_the_recorded_hook_prints_the_plain_lines(tmp_path):
    """The real command on CPU, no mock: the default mesh size of the hook takes about 25 s."""
    from topopt.cli import main
    run = tmp_path / "hook"
    (run / "result").mkdir(parents=True)
    shutil.copy(RECORDING / "run.json", run / "run.json")
    for name in ("final.stl", "metrics.json"):
        shutil.copy(RECORDING / "result" / name, run / "result" / name)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(run), "--allowable-mpa", "5", "--no-sensitivity", "--yaml", str(REPO / "examples" / "hook" / "hook.yaml")])
    assert code == 0, err.getvalue()
    plain = [ln for ln in out.getvalue().splitlines() if ln.startswith("plain: ")]
    res = json.loads((run / "verify" / "verify.json").read_text())
    assert [ln[len("plain: "):] for ln in plain] == plain_lines(res)
    assert [ln.split(":")[1].strip() for ln in plain[:2]] == ["case hang", "case pull_off"]
    assert re.search(r"margin \d+\.\d\d \(allowable divided by value\)", plain[1])
    low = "\n".join(plain).lower()
    assert not any(w in low for w in PROMISES)
