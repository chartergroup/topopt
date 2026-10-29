"""Tests of the CI shards (CONTRACTS.md 27.2): the key `shards`, the matrix, the split of test files, `run_pytest.py --shard`, and the job `report`.

They start no optimizer run and no network access. The tests that start pytest use a tiny suite in `tmp_path`.
"""
import copy
import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "ci"))

import _common  # noqa: E402
import check_skips  # noqa: E402
import make_tiers  # noqa: E402
import plan_legs  # noqa: E402
import report_shards  # noqa: E402
import run_pytest  # noqa: E402
import shards  # noqa: E402

SPEC = _common.load_spec()


# --- the key `shards` and the matrix ------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, "4", 1.5, True, None])
def test_shards_of_refuses_a_value_that_is_not_a_whole_number_of_one_or_more(bad):
    with pytest.raises(_common.CiError, match="x: `shards` in legs.toml must be a whole number of 1 or more"):
        _common.shards_of({"shards": bad}, "x")


def test_shards_of_defaults_to_one_and_the_legs_have_the_designed_numbers():
    assert _common.shards_of({}) == 1 and _common.shards_of({"shards": 3}) == 3
    legs = SPEC["legs"]
    assert {k: _common.shards_of(v, k) for k, v in legs.items()} == {
        "linux-x86_64": 2, "macos-arm64": 2, "linux-x86_64-intel-pins": 2, "macos-x86_64": 1}   # intel-mac.yml has no job report (27.3.2)


def test_the_choice_linux_has_one_leg_one_python_two_shards_and_names_that_hold_the_shard():
    include, _ = plan_legs.plan("workflow_dispatch", SPEC, legs="linux")
    assert len(include) == 1 * 1 * 2
    assert all(tuple(e) == plan_legs.MATRIX_KEYS and e["shards"] == 2 for e in include)
    assert [e["suffix"] for e in include if e["python"] == "3.13"] == ["-s1of2", "-s2of2"]
    assert include[1]["part"] == ", shard 2/2"
    names = {f"ci-{e['label']}-py{e['python']}{e['suffix']}" for e in include}
    assert len(names) == 2 and "ci-linux-x86_64-py3.13-s2of2" in names          # every artifact name is unique


def test_the_choice_all_counts_every_shard():
    include, _ = plan_legs.plan("workflow_dispatch", SPEC, legs="all")
    by = {}
    for e in include:
        by[e["label"]] = by.get(e["label"], 0) + 1
    assert by == {"linux-x86_64": 2, "macos-arm64": 2, "linux-x86_64-intel-pins": 2}      # one Python, 2 shards each


def test_a_leg_with_one_shard_keeps_the_names_that_it_had():
    include, _ = plan_legs.plan("workflow_dispatch", SPEC, workflow="intel-mac")
    assert include[0]["suffix"] == "" and include[0]["part"] == "" and (include[0]["shard"], include[0]["shards"]) == (1, 1)


def test_a_leg_that_runs_no_pytest_in_the_tier_gets_one_shard():
    fast, _ = plan_legs.plan("workflow_dispatch", SPEC, legs="all", tier="fast")
    full, _ = plan_legs.plan("workflow_dispatch", SPEC, legs="all", tier="full")
    count = lambda inc, label: sum(1 for e in inc if e["label"] == label)      # noqa: E731
    assert count(fast, "linux-x86_64-intel-pins") == 1 and count(full, "linux-x86_64-intel-pins") == 2
    assert count(fast, "linux-x86_64") == count(full, "linux-x86_64") == 2


def test_plan_refuses_a_wrong_number_of_shards():
    spec = copy.deepcopy(SPEC)
    spec["legs"]["linux-x86_64"]["shards"] = 0
    with pytest.raises(_common.CiError, match="linux-x86_64: `shards`"):
        plan_legs.plan("workflow_dispatch", spec, legs="linux")


# --- the split ----------------------------------------------------------------------------------------

DUR = {f"tests/test_{c}.py": {"all": a, "not_slow": n} for c, a, n in
       [("a", 100, 10), ("b", 60, 60), ("c", 50, 5), ("d", 40, 40), ("e", 30, 30), ("f", 20, 2), ("g", 10, 10), ("h", 5, 5)]}


def test_parse_shard():
    assert shards.parse_shard("2/4") == (2, 4) and shards.parse_shard("1/1") == (1, 1)
    for bad in ("0/4", "5/4", "1/0", "x", "2", "2/4/6", "-1/2"):
        with pytest.raises(_common.CiError, match="--shard"):
            shards.parse_shard(bad)


def test_every_file_is_in_exactly_one_shard():
    files = list(DUR) + ["tests/test_new.py"]
    for n in (1, 2, 3, 4, 7, 20):
        parts = shards.split_files(files, DUR, n)
        assert len(parts) == n
        flat = [f for p in parts for f in p]
        assert sorted(flat) == sorted(files), n


def test_the_split_is_the_same_for_the_same_input_in_any_order():
    files = list(DUR) + ["tests/test_new.py", "tests/test_newer.py"]
    first = shards.split_files(files, DUR, 4)
    for seed in range(5):
        mixed = files[:]
        random.Random(seed).shuffle(mixed)
        assert shards.split_files(mixed, DUR, 4) == first
        assert shards.split_files(mixed + mixed[:3], DUR, 4) == first          # a repeated name does not change it


def test_a_file_that_is_not_in_the_durations_gets_the_median_and_ties_break_by_name():
    w = shards.weights(["tests/test_a.py", "tests/test_zz.py"], DUR, "all")
    assert w["tests/test_a.py"] == 100 and w["tests/test_zz.py"] == 35.0          # the median of 100 60 50 40 30 20 10 5
    assert shards.weights(["x.py"], {}, "all") == {"x.py": 1.0}                     # no list at all: every file has the same weight
    parts = shards.split_files(["b.py", "a.py", "d.py", "c.py"], {}, 2)
    assert parts == [["a.py", "c.py"], ["b.py", "d.py"]]


def test_one_shard_gets_all_the_files_and_the_load_is_balanced():
    assert shards.split_files(list(DUR), DUR, 1) == [sorted(DUR)]
    parts = shards.split_files(list(DUR), DUR, 3, "all")
    loads = shards.shard_loads(parts, DUR, "all")
    assert max(loads) - min(loads) <= 40 and sum(loads) == 315            # no shard is far from the others: the biggest file is 100 s


def test_the_tier_chooses_the_sum_that_the_split_balances():
    assert shards.duration_key("not slow") == "not_slow" and shards.duration_key("platform_sensitive and not slow") == "not_slow"
    assert shards.duration_key("") == "all" and shards.duration_key(None) == "all"
    # test_a.py has 100 s in all but 10 s without its slow tests: in the fast tier it must not weigh as the biggest file
    all_parts = shards.split_files(list(DUR), DUR, 2, "all")
    fast_parts = shards.split_files(list(DUR), DUR, 2, "not_slow")
    assert all_parts != fast_parts
    loads = shards.shard_loads(fast_parts, DUR, "not_slow")
    assert abs(loads[0] - loads[1]) <= 10


def test_the_checked_in_durations_name_files_that_exist_and_give_a_balanced_split_of_the_real_files():
    d = shards.load_durations()
    assert d and all(set(v) == {"all", "not_slow"} and v["not_slow"] <= v["all"] for v in d.values())
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py"))
    for key in shards.KEYS:
        loads = shards.shard_loads(shards.split_files(files, d, 4, key), d, key)
        assert max(loads) <= 1.1 * (sum(loads) / 4), (key, loads)


def test_the_cli_prints_the_files_of_one_shard(capsys):
    assert shards.main(["split", "--shard", "1/2", "--durations", "/nonexistent.json", "b.py", "a.py", "c.py"]) == 0
    assert capsys.readouterr().out.split() == ["a.py", "c.py"]
    assert shards.main(["show", "--shards", "2", "--durations", "/nonexistent.json", "b.py", "a.py"]) == 0
    assert "shard 1/2: 1 files" in capsys.readouterr().out


# --- make_tiers.py durations --------------------------------------------------------------------------

def test_file_durations_sum_all_tests_and_the_tests_that_are_not_slow():
    r1 = {"tests/test_a.py::t1": ("pass", 2.0), "tests/test_a.py::t2": ("pass", 30.0), "tests/test_b.py::t": ("pass", 1.0)}
    r2 = {"tests/test_a.py::t1": ("pass", 3.0)}                  # the longest time of a test counts
    out = make_tiers.file_durations([r1, r2], {"tests/test_a.py::t2"})
    assert out == {"tests/test_a.py": {"all": 33.0, "not_slow": 3.0}, "tests/test_b.py": {"all": 1.0, "not_slow": 1.0}}


def test_the_durations_command_leaves_out_a_file_that_is_gone(tmp_path, monkeypatch):
    xml = tmp_path / "j.xml"
    xml.write_text('<testsuites><testsuite><testcase classname="tests.test_ci_shards" name="t" time="1.5"/>'
                   '<testcase classname="tests.test_gone_file" name="t" time="9"/></testsuite></testsuites>')
    monkeypatch.setattr(make_tiers, "DURATIONS_FILE", tmp_path / "durations.json")
    assert make_tiers.main(["durations", "--junit", str(xml)]) == 0
    assert json.loads((tmp_path / "durations.json").read_text())["files"] == {"tests/test_ci_shards.py": {"all": 1.5, "not_slow": 1.5}}


def test_make_tiers_check_names_a_durations_file_that_is_gone(tmp_path):
    (tmp_path / "tests" / "tiers").mkdir(parents=True)
    (tmp_path / "tests" / "tiers" / "durations.json").write_text(json.dumps({"files": {"tests/test_gone.py": {"all": 1, "not_slow": 1}}}))
    assert make_tiers.check(tmp_path) == ["durations.json: no file tests/test_gone.py"]


# --- run_pytest.py --shard ----------------------------------------------------------------------------

def test_the_collect_command_and_the_shard_command_have_the_marker_and_the_files():
    extra = ["-m", "not slow"]
    assert run_pytest.collect_command("py", "/r", extra) == ["py", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "--rootdir", "/r", "-m", "not slow"]
    assert run_pytest.collect_command("py", "/r", extra, "/a/collect.xml")[-3:] == ["--junitxml=/a/collect.xml", "-m", "not slow"]
    cmd = run_pytest.pytest_command("py", "/r", "/a/junit.xml", "2", extra + ["tests/test_a.py", "tests/test_c.py"])
    assert cmd[-4:] == ["-m", "not slow", "tests/test_a.py", "tests/test_c.py"] and "--deselect" not in cmd and "-n" in cmd


def test_parse_node_ids_takes_only_the_node_id_lines():
    out = "tests/test_a.py::t1\ntests/test_a.py::t2[x-1]\n\n6642/6693 tests collected (51 deselected) in 8.1s\nwarnings: tests/x.py ...\n"
    assert run_pytest.parse_node_ids(out) == ["tests/test_a.py::t1", "tests/test_a.py::t2[x-1]"]


def test_collect_tier_fails_on_a_collection_error_and_accepts_no_test(monkeypatch):
    def fake(code, out):
        return lambda cmd, **kw: subprocess.CompletedProcess(cmd, code, stdout=out)
    monkeypatch.setattr(run_pytest.subprocess, "run", fake(2, "ERROR collecting tests/test_a.py\nImportError: x\n"))
    with pytest.raises(_common.CiError, match="(?s)collecting the tier failed.*ImportError"):
        run_pytest.collect_tier(["x"], {})
    monkeypatch.setattr(run_pytest.subprocess, "run", fake(5, "no tests collected\n"))
    assert run_pytest.collect_tier(["x"], {}) == []


def test_shard_files_runs_the_split_of_the_files_of_the_ids():
    ids = [f"{f}::t{i}" for f in DUR for i in range(2)]
    got = [run_pytest.shard_files(ids, DUR, i, 3, "all") for i in (1, 2, 3)]
    assert sorted(f for g in got for f in g) == sorted(DUR)


def _suite(root: Path) -> None:
    (root / "test_one.py").write_text("import pytest\ndef test_a(): pass\ndef test_b(): pass\n@pytest.mark.slow\ndef test_slow(): pass\n")
    (root / "test_two.py").write_text("import pytest\n@pytest.mark.parametrize('x', [1, 2, 3])\ndef test_p(x): pass\n")
    (root / "test_three.py").write_text("def test_c(): pass\n")
    (root / "test_four.py").write_text("def test_d(): pass\ndef test_e(): pass\n")


def _run_shards(tmp_path, monkeypatch, n):
    suite = tmp_path / "suite"
    suite.mkdir()
    _suite(suite)
    monkeypatch.setattr(run_pytest, "ROOT", suite)
    monkeypatch.setattr(run_pytest, "get_leg", lambda label: {"pytest_workers": "0", "pytest_fast": "not slow"})
    art = tmp_path / "art"
    for i in range(1, n + 1):
        d = art / f"ci-linux-x86_64-py3.12-s{i}of{n}"
        assert run_pytest.main(["--leg", "linux-x86_64", "--artifacts", str(d), "--tier", "fast", "--shard", f"{i}/{n}", "--workers", "0"]) == 0
    return art


def test_the_shards_of_a_real_pytest_run_together_run_the_tier_once(tmp_path, monkeypatch):
    art = _run_shards(tmp_path, monkeypatch, 3)
    docs = [json.loads((d / "shard.json").read_text()) for d in sorted(art.iterdir())]
    ran = [t for d in docs for t in d["ran"]]
    assert len(ran) == len(set(ran)) == 8 == docs[0]["tier_total"]            # test_slow is not in the fast tier; 2 + 3 + 1 + 2 tests
    assert all(d["tier_total"] == 8 and d["shards"] == 3 and d["python"] == f"{sys.version_info[0]}.{sys.version_info[1]}" for d in docs)
    assert sorted(f for d in docs for f in d["files"]) == ["test_four.py", "test_one.py", "test_three.py", "test_two.py"]
    assert report_shards.main(["--artifacts", str(art)]) == 0


def test_one_shard_of_one_runs_the_whole_suite_and_has_no_tier_total(tmp_path, monkeypatch):
    art = _run_shards(tmp_path, monkeypatch, 1)
    doc = json.loads((next(art.iterdir()) / "shard.json").read_text())
    assert doc["tier_total"] is None and doc["files"] is None and len(doc["ran"]) == 8


def test_a_shard_refuses_a_test_path_after_the_double_dash_and_accepts_a_keyword(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    suite.mkdir()
    _suite(suite)
    monkeypatch.setattr(run_pytest, "ROOT", suite)
    monkeypatch.setattr(run_pytest, "get_leg", lambda label: {"pytest_workers": "0", "pytest_fast": "not slow"})
    assert run_pytest.path_args(["-k", "one or two", "-m", "not slow", "test_one.py", "test_two.py::test_p", "x/test_y.py", "--maxfail", "3"]) == [
        "test_one.py", "test_two.py::test_p", "x/test_y.py"]
    assert run_pytest.path_args(["--deselect", "test_one.py::test_a", "--ignore", "test_two.py", "--basetemp", "x"]) == []   # option values
    base = ["--leg", "linux-x86_64", "--tier", "fast", "--workers", "0"]
    with pytest.raises(_common.CiError, match=r"give no test path after `--` \(test_one\.py\)"):
        run_pytest.main(base + ["--artifacts", str(tmp_path / "a"), "--shard", "1/2", "--", "test_one.py"])
    assert not (tmp_path / "a" / "shard.json").exists()
    art = tmp_path / "k"
    for i in (1, 2):                                   # -k is fine: the collect and the shards see the same filter
        assert run_pytest.main(base + ["--artifacts", str(art / f"ci-linux-x86_64-py3.12-s{i}of2"), "--shard", f"{i}/2", "--", "-k", "one or two"]) == 0
    docs = [json.loads((d / "shard.json").read_text()) for d in sorted(art.iterdir())]
    assert sorted(t for d in docs for t in d["ran"]) == ["test_one.py::test_a", "test_one.py::test_b"] + [f"test_two.py::test_p[{x}]" for x in (1, 2, 3)]
    assert report_shards.main(["--artifacts", str(art)]) == 0
    assert run_pytest.main(base + ["--artifacts", str(tmp_path / "one"), "--", "test_one.py"]) == 0     # one shard: a path is fine


def test_a_file_that_skips_at_import_runs_in_no_shard_and_the_report_checks_its_skip(tmp_path, monkeypatch):
    """A module-level skip gives no node id, so no shard runs the file. The collect step writes the skip to collect.xml, and report_shards gives it
    to the skip check: here the skip is not allowed, so the report fails (without shards the same skip fails check_skips.py)."""
    suite = tmp_path / "suite"
    suite.mkdir()
    _suite(suite)
    (suite / "test_five.py").write_text("import pytest\npytest.importorskip('nosuchmod_xyz')\ndef test_f(): pass\n")
    monkeypatch.setattr(run_pytest, "ROOT", suite)
    monkeypatch.setattr(run_pytest, "get_leg", lambda label: {"pytest_workers": "0", "pytest_fast": "not slow"})
    art = tmp_path / "art"
    for i in (1, 2):
        assert run_pytest.main(["--leg", "linux-x86_64", "--artifacts", str(art / f"ci-linux-x86_64-py3.12-s{i}of2"), "--tier", "fast",
                                "--shard", f"{i}/2", "--workers", "0"]) == 0
    docs = [json.loads((d / "shard.json").read_text()) for d in sorted(art.iterdir())]
    assert "test_five.py" not in [f for d in docs for f in d["files"]] and sum(len(d["ran"]) for d in docs) == 8 == docs[0]["tier_total"]
    skips = [s for d in sorted(art.iterdir()) for s in check_skips.skipped_tests(d / "collect.xml")]
    assert {s["name"] for s in skips} == {"test_five"} and all("nosuchmod_xyz" in s["reason"] for s in skips)
    with pytest.raises(_common.CiError, match="nosuchmod_xyz"):
        report_shards.main(["--artifacts", str(art)])
    (suite / "test_five.py").unlink()                     # the same shards without the file: the report passes
    (tmp_path / "again").mkdir()
    art2 = _run_shards(tmp_path / "again", monkeypatch, 2)
    assert report_shards.main(["--artifacts", str(art2)]) == 0


def test_a_shard_with_no_file_writes_an_empty_junit(tmp_path, monkeypatch):
    art = _run_shards(tmp_path, monkeypatch, 6)               # 4 files in 6 shards
    docs = [json.loads((d / "shard.json").read_text()) for d in sorted(art.iterdir())]
    assert sum(1 for d in docs if d["files"] == []) == 2 and sum(len(d["ran"]) for d in docs) == 8
    assert run_pytest.junit_counts(next(d for d in sorted(art.iterdir()) if json.loads((d / "shard.json").read_text())["files"] == []) / "junit.xml")["tests"] == 0


# --- report_shards.py ----------------------------------------------------------------------------------

def _artifact(root: Path, name: str, shard: int, n: int, ran: list[str], total=None, leg="linux-x86_64", py="3.12", times=None, skips=()):
    d = root / name
    d.mkdir(parents=True)
    (d / "shard.json").write_text(json.dumps({"leg": leg, "python": py, "shard": shard, "shards": n, "tier": "fast", "tier_total": total, "files": [], "ran": ran}))
    cases = "".join(f'<testcase classname="{t.split("::")[0][:-3].replace("/", ".")}" name="{t.split("::")[1]}" time="{(times or {}).get(t, 1)}"/>' for t in ran)
    cases += "".join(f'<testcase classname="{c}" name="{n_}"><skipped message="{m}"/></testcase>' for c, n_, m in skips)
    (d / "junit.xml").write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")


T = [f"tests/test_x.py::t{i}" for i in range(6)]


def test_report_passes_when_the_shards_run_the_tier_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    art = tmp_path / "art"
    art.mkdir()
    _artifact(art, "ci-linux-x86_64-py3.12-s1of2", 1, 2, T[:4], 6, times={T[2]: 9.5})
    _artifact(art, "ci-linux-x86_64-py3.12-s2of2", 2, 2, T[4:], 6)
    assert report_shards.main(["--artifacts", str(art), "--slowest", "3"]) == 0
    text = (tmp_path / "sum.md").read_text()
    assert "linux-x86_64 | 3.12 | 2/2 | 6 | 6 | ok" in text and "The 3 slowest tests" in text
    assert text.index("t2") < text.index("| 1.0 |")                     # the slowest test comes first
    assert "OK: 1 group(s)" in capsys.readouterr().out


def test_report_fails_when_a_test_did_not_run(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    _artifact(art, "ci-a-s1of2", 1, 2, T[:3], 6)
    _artifact(art, "ci-a-s2of2", 2, 2, T[4:], 6)                      # t3 is in no shard
    with pytest.raises(_common.CiError, match="the tier has 6 tests and the shards ran 5 .difference 1"):
        report_shards.main(["--artifacts", str(art)])


def test_report_fails_when_a_test_ran_in_two_shards(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    _artifact(art, "ci-a-s1of2", 1, 2, T[:4], 6)
    _artifact(art, "ci-a-s2of2", 2, 2, T[3:], 6)                      # t3 is in both: the count is 6, but one test ran twice
    with pytest.raises(_common.CiError, match="tests/test_x.py::t3 ran in shard 1 and in shard 2"):
        report_shards.main(["--artifacts", str(art)])


def test_report_fails_when_a_shard_is_missing_or_has_no_shard_json(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    _artifact(art, "ci-a-s1of2", 1, 2, T, 6)
    with pytest.raises(_common.CiError, match=r"shards \[2\] are missing"):
        report_shards.main(["--artifacts", str(art)])
    (art / "ci-a-s2of2").mkdir()                                       # an artifact of a job that stopped before pytest ended
    with pytest.raises(_common.CiError, match="ci-a-s2of2: no shard.json"):
        report_shards.main(["--artifacts", str(art)])


def test_report_groups_by_leg_and_python_and_skips_the_count_of_a_leg_with_one_shard(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    _artifact(art, "ci-l-py3.12-s1of2", 1, 2, T[:3], 6, py="3.12")
    _artifact(art, "ci-l-py3.12-s2of2", 2, 2, T[3:], 6, py="3.12")
    _artifact(art, "ci-l-py3.13-s1of2", 1, 2, T[:3], 6, py="3.13")
    _artifact(art, "ci-l-py3.13-s2of2", 2, 2, T[3:], 6, py="3.13")
    _artifact(art, "ci-m", 1, 1, T[:2], None, leg="macos-x86_64")
    assert report_shards.main(["--artifacts", str(art)]) == 0
    (art / "ci-l-py3.13-s2of2" / "shard.json").unlink()
    with pytest.raises(_common.CiError, match="Python 3.13"):
        report_shards.main(["--artifacts", str(art)])


def test_report_runs_the_skip_check_on_all_junit_files_of_a_leg(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    _artifact(art, "ci-a-s1of2", 1, 2, T[:3], 6, skips=[("tests.test_fe", "test_needs_gpu", "no OpenCL device at all")])
    _artifact(art, "ci-a-s2of2", 2, 2, T[3:], 6)
    with pytest.raises(_common.CiError, match="skipped test.s. that the plan expects to run on linux-x86_64"):
        report_shards.main(["--artifacts", str(art)])


def test_check_skips_takes_several_junit_files(tmp_path):
    import check_skips
    a, b = tmp_path / "a.xml", tmp_path / "b.xml"
    for f, name in ((a, "t1"), (b, "t2")):
        f.write_text(f'<testsuites><testsuite><testcase classname="tests.test_fe" name="{name}"><skipped message="odd reason"/></testcase></testsuite></testsuites>')
    with pytest.raises(_common.CiError, match="2 skipped test"):
        check_skips.main(["--leg", "linux-x86_64", "--junit", str(a), "--junit", str(b)])


# --- the workflow --------------------------------------------------------------------------------------

def _wf(name="tests.yml"):
    import yaml
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


@pytest.mark.parametrize("name,job", [("tests.yml", "test"), ("tests.yml", "test-after"), ("intel-mac.yml", "test")])
def test_the_once_per_leg_steps_run_on_shard_1_only_and_pytest_runs_on_every_shard(name, job):
    steps = {s["name"]: s for s in _wf(name)["jobs"][job]["steps"]}
    for script in ("license_check", "check_locks", "short_run"):
        (st,) = [s for s in steps.values() if script in s.get("run", "")]
        assert "matrix.shard == 1" in st["if"], script
    (py,) = [s for s in steps.values() if "run_pytest" in s.get("run", "")]
    assert "matrix.shard" not in py["if"] and "--shard ${{ matrix.shard }}/${{ matrix.shards }}" in py["run"]
    (inst,) = [s for s in steps.values() if "install_env" in s.get("run", "")]
    assert "if" not in inst
    assert _wf(name)["jobs"][job]["name"].endswith("${{ matrix.part }})")


def test_the_report_job_waits_for_the_tests_runs_after_a_failure_and_downloads_the_ci_artifacts():
    job = _wf()["jobs"]["report"]
    assert job["needs"] == ["plan", "test", "test-after"] and "always()" in job["if"]
    down = [s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/download-artifact@")][0]
    assert down["with"]["pattern"] == "ci-*" and "matrix" not in str(job)
    assert any("report_shards.py" in s.get("run", "") for s in job["steps"])


def test_plan_passes_the_tier_in_both_workflows():
    for name in ("tests.yml", "intel-mac.yml"):
        (st,) = [s for s in _wf(name)["jobs"]["plan"]["steps"] if "plan_legs.py" in s.get("run", "")]
        assert '--tier "${{ inputs.tier }}"' in st["run"], name
