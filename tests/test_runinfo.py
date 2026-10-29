"""Tests for topopt/runinfo.py and the atomic history writer (CONTRACTS.md section 24.1).

Peak memory unit (both platform branches), the one reader of history.jsonl (a partial last line is
skipped and stated, a bad line before the last one is an error), the writer (one write, then a flush),
the post step and run_status on a run whose last history line is cut.
"""
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from topopt import runinfo, snapshot  # noqa: E402
from topopt.runinfo import HistoryError, read_history  # noqa: E402

ALLOC = "b = bytearray(b'x') * (200 * 2**20)\n"   # 200 MB, every page touched


# --- peak memory ------------------------------------------------------------------------------

def test_ru_maxrss_unit_both_platforms(monkeypatch):
    assert runinfo.ru_maxrss_to_bytes(1000, "darwin") == 1000          # macOS: bytes
    assert runinfo.ru_maxrss_to_bytes(1000, "linux") == 1000 * 1024    # Linux: KiB
    assert runinfo.ru_maxrss_to_bytes(1000, "freebsd14") == 1000 * 1024
    monkeypatch.setattr(sys, "platform", "darwin")
    assert runinfo.ru_maxrss_to_bytes(7) == 7
    monkeypatch.setattr(sys, "platform", "linux")
    assert runinfo.ru_maxrss_to_bytes(7) == 7 * 1024


def test_peak_rss_functions_use_the_unit(monkeypatch):
    """No /proc (macOS): `ru_maxrss` with the unit of the platform."""
    import resource

    class Usage:
        ru_maxrss = 5000

    monkeypatch.setattr(resource, "getrusage", lambda who: Usage)
    monkeypatch.setattr(runinfo, "_PROC_STATUS", "/nonexistent/status")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert runinfo.peak_rss_bytes() == 5000 and runinfo.children_peak_rss_bytes() == 5000
    assert runinfo.peak_rss_info() == (5000, "ru_maxrss")
    monkeypatch.setattr(sys, "platform", "linux")
    assert runinfo.peak_rss_bytes() == 5000 * 1024 and runinfo.children_peak_rss_bytes() == 5000 * 1024


def test_vmhwm_comes_from_proc_status_when_it_is_there(tmp_path, monkeypatch):
    import resource

    class Usage:
        ru_maxrss = 999999999

    monkeypatch.setattr(resource, "getrusage", lambda who: Usage)
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmPeak:\t  900000 kB\nVmHWM:\t  123456 kB\nVmRSS:\t   100000 kB\n")
    monkeypatch.setattr(runinfo, "_PROC_STATUS", str(status))
    assert runinfo._vmhwm_bytes() == 123456 * 1024
    assert runinfo.peak_rss_info() == (123456 * 1024, "VmHWM")          # not the ru_maxrss of the fake usage
    for text in ("Name:\tpython\n", "VmHWM:\t  lots kB\n", "VmHWM:\n", ""):
        status.write_text(text)
        assert runinfo._vmhwm_bytes() is None
        assert runinfo.peak_rss_info()[1] == "ru_maxrss"                 # the fallback


needs_proc = pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="needs /proc (Linux)")


# On Linux a child starts with the peak of its parent (`ru_maxrss` is inherited through fork and exec).
# A pytest process can hold more than 1 GB, so the test code runs in a child of a small launcher process.
LAUNCH = ("import subprocess, sys\n"
          "r = subprocess.run([sys.executable, '-c', sys.argv[1]], capture_output=True, text=True)\n"
          "sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); sys.exit(r.returncode)\n")


def _py(code):
    r = subprocess.run([sys.executable, "-c", LAUNCH, code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return int(r.stdout.strip().splitlines()[-1])


def test_peak_rss_unit():
    """A child allocates 200 MB. The helper must say between 0.2 and 1.0 GB. A wrong factor fails."""
    peak = _py(ALLOC + "from topopt.runinfo import peak_rss_bytes\nprint(peak_rss_bytes())")
    assert 0.2e9 < peak < 1.0e9, peak


@needs_proc
def test_peak_rss_has_no_floor_from_a_parent_with_a_big_peak():
    """A parent touches 400 MB and frees it, then starts a child with `subprocess`. `ru_maxrss` of the child starts at
    the peak of the parent (Linux). The peak of the helper is `VmHWM`: it does not."""
    code = ("import subprocess, sys\n"
            "b = bytearray(b'x') * (400 * 2**20); del b\n"
            "child = 'from topopt.runinfo import peak_rss_bytes; print(peak_rss_bytes())'\n"
            "r = subprocess.run([sys.executable, '-c', child], capture_output=True, text=True, check=True)\n"
            "print(r.stdout.strip())")
    assert _py(code) < 150e6


def test_children_peak_rss_unit():
    code = ("import subprocess, sys\n"
            f"subprocess.run([sys.executable, '-c', {ALLOC!r}], check=True)\n"
            "from topopt.runinfo import children_peak_rss_bytes\nprint(children_peak_rss_bytes())")
    assert 0.2e9 < _py(code) < 1.0e9


# --- the one reader of history.jsonl ----------------------------------------------------------

def rows(n):
    return [{"iteration": i, "elapsed_s": 1.5 * i, "compliance": 10.0 / (i + 1)} for i in range(n)]


def write_lines(path, recs, tail=""):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs) + tail)
    return path


def test_read_complete_file(tmp_path):
    p = write_lines(tmp_path / "history.jsonl", rows(5))
    got = read_history(p)
    assert got.rows == rows(5) and got.facts == []
    rows_, facts = read_history(p)          # a NamedTuple: it unpacks
    assert len(rows_) == 5 and facts == []


def test_read_empty_and_blank_lines(tmp_path):
    p = tmp_path / "history.jsonl"
    p.write_text("")
    assert read_history(p) == ([], [])
    p.write_text("\n\n" + json.dumps(rows(1)[0]) + "\n\n   \n")
    assert read_history(p).rows == rows(1)
    with pytest.raises(FileNotFoundError):
        read_history(tmp_path / "missing.jsonl")


@pytest.mark.parametrize("cut", [1, 10, 40, -2])
def test_partial_last_line_is_skipped_with_a_fact(tmp_path, cut):
    full = "".join(json.dumps(r) + "\n" for r in rows(4))
    last = json.dumps({"iteration": 4, "elapsed_s": 6.0, "compliance": 2.0})
    p = tmp_path / "history.jsonl"
    p.write_text(full + (last[:cut] if cut > 0 else last[:cut]))      # no closing brace, no newline
    got = read_history(p)
    assert got.rows == rows(4)
    assert len(got.facts) == 1 and "partial last line" in got.facts[0] and "line 5" in got.facts[0]


def test_last_line_without_newline_that_parses_is_kept(tmp_path):
    p = tmp_path / "history.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows(2)) + json.dumps(rows(3)[2]))
    got = read_history(p)
    assert len(got.rows) == 3 and got.facts == []


def test_last_line_not_an_object_counts_as_partial(tmp_path):
    p = tmp_path / "history.jsonl"
    p.write_text(json.dumps(rows(1)[0]) + "\n12\n")
    got = read_history(p)
    assert len(got.rows) == 1 and "partial last line" in got.facts[0]


def test_bad_line_before_the_last_is_an_error(tmp_path):
    p = tmp_path / "history.jsonl"
    good = [json.dumps(r) for r in rows(4)]
    p.write_text("\n".join([good[0], good[1][:20], good[2], good[3]]) + "\n")
    with pytest.raises(HistoryError, match="line 2"):
        read_history(p)
    assert issubclass(HistoryError, ValueError)


def test_bad_line_before_the_last_and_a_partial_last_line_is_still_an_error(tmp_path):
    p = tmp_path / "history.jsonl"
    good = [json.dumps(r) for r in rows(3)]
    p.write_text("\n".join([good[0], "garbage", good[1], good[2][:15]]))
    with pytest.raises(HistoryError, match="line 2"):
        read_history(p)


def test_tolerant_mode_skips_every_bad_line_and_names_them(tmp_path):
    p = tmp_path / "history.jsonl"
    good = [json.dumps(r) for r in rows(5)]
    p.write_text("\n".join([good[0], "x", good[1], "y", good[2], good[3][:9]]))
    got = read_history(p, strict=False)
    assert got.rows == rows(3)
    assert len(got.facts) == 2
    assert "2 line(s)" in got.facts[0] and "2, 4" in got.facts[0]
    assert "partial last line" in got.facts[1]


def test_invalid_bytes_in_the_last_line_do_not_crash(tmp_path):
    p = tmp_path / "history.jsonl"
    p.write_bytes((json.dumps(rows(1)[0]) + "\n").encode() + b'{"a": "\xff\xfe')
    got = read_history(p)
    assert len(got.rows) == 1 and "partial last line" in got.facts[0]


# --- the writer: one whole line, one write, a flush ---------------------------------------------

class SpyFile:
    """Wrap a real file. Keep the order of the calls."""

    def __init__(self, f, calls):
        self._f, self._calls = f, calls

    def write(self, data):
        self._calls.append(("write", bytes(data)))
        return self._f.write(data)

    def flush(self):
        self._calls.append(("flush", None))
        return self._f.flush()

    def __enter__(self):
        self._f.__enter__()
        return self

    def __exit__(self, *exc):
        return self._f.__exit__(*exc)


def test_add_history_is_one_write_then_a_flush(tmp_path, monkeypatch):
    calls = []
    real_open = open

    def spy_open(path, mode="r", *a, **kw):
        f = real_open(path, mode, *a, **kw)
        return SpyFile(f, calls) if Path(path).name == "history.jsonl" else f

    monkeypatch.setattr(snapshot, "open", spy_open, raising=False)
    w = snapshot.RunWriter(tmp_path, quiet=True)
    rec = {"iteration": 3, "elapsed_s": 1.0, "note": "x" * 20000}     # longer than one 8 KiB buffer
    w.add_history(rec)
    kinds = [c[0] for c in calls]
    assert kinds == ["write", "flush"]
    data = calls[0][1]
    assert data.endswith(b"\n") and data.count(b"\n") == 1
    assert json.loads(data) == rec
    assert (tmp_path / "history.jsonl").read_bytes() == data


def test_a_reader_never_fails_while_the_writer_appends(tmp_path):
    w = snapshot.RunWriter(tmp_path, quiet=True)
    total = 400
    done = threading.Event()

    def write():
        for i in range(total):
            w.add_history({"iteration": i, "elapsed_s": float(i), "pad": "p" * (50 + 60 * (i % 7))})
        done.set()

    t = threading.Thread(target=write)
    t.start()
    seen, last_count = 0, 0
    while not done.is_set() or seen < 3:
        p = tmp_path / "history.jsonl"
        if p.exists():
            got = read_history(p)                      # strict: it must not raise
            assert all(isinstance(r, dict) for r in got.rows)
            assert [r["iteration"] for r in got.rows] == list(range(len(got.rows)))
            assert len(got.rows) >= last_count
            last_count = len(got.rows)
        seen += 1
    t.join()
    final = read_history(tmp_path / "history.jsonl")
    assert len(final.rows) == total and final.facts == []


def test_snapshot_text_does_not_overclaim():
    """Only whole files are called atomic. history.jsonl and run.log are named as append-only."""
    doc = snapshot.__doc__
    assert "history.jsonl" in doc and "only grow" in doc and "not atomic" in doc
    assert "read_history" in doc


def test_write_atomic_and_log_line(tmp_path):
    p = tmp_path / "a.json"
    runinfo.write_atomic(p, b"one")
    runinfo.write_atomic(p, b"two")
    assert p.read_bytes() == b"two" and not list(tmp_path.glob("*.tmp"))
    runinfo.append_log_line(tmp_path, "first")
    runinfo.append_log_line(tmp_path, "second\n")
    assert (tmp_path / "run.log").read_text() == "first\nsecond\n"


@pytest.mark.parametrize("mode", [None, 0o600])
@pytest.mark.parametrize("target", ["file", "missing"])
def test_write_atomic_does_not_follow_a_link_at_the_temporary_name(tmp_path, mode, target):
    """P9-14 R2 R2-wa. A link at `<name>.tmp` must not send the data to its target (a file outside the folder, or a file that does not exist yet)."""
    folder, outside = tmp_path / "run", tmp_path / "elsewhere"
    folder.mkdir()
    outside.mkdir()
    victim = outside / "victim.txt"
    if target == "file":
        victim.write_text("outside")
    link = folder / "a.json.tmp"
    link.symlink_to(victim)
    p = folder / "a.json"
    runinfo.write_atomic(p, b"data", **({} if mode is None else {"mode": mode}))
    assert p.read_bytes() == b"data" and not p.is_symlink()
    assert (victim.read_text() if target == "file" else victim.exists()) == ("outside" if target == "file" else False)
    assert not link.exists() and not link.is_symlink() and not list(folder.glob("*.tmp"))
    if mode is not None:
        assert stat.S_IMODE(p.stat().st_mode) == mode


# --- the post step and run_status on a run with a cut last line --------------------------------

@pytest.fixture(scope="module")
def cut_run(tmp_path_factory):
    import fake_run
    out = tmp_path_factory.mktemp("cutrun")
    fake_run.run(out, spi=0.0, n_it=6)
    shutil.rmtree(out / "result")
    lines = (out / "history.jsonl").read_text().splitlines()
    total = len(lines)
    (out / "history.jsonl").write_text("\n".join(lines[:-1]) + "\n" + lines[-1][: len(lines[-1]) // 2])
    return out, total


def test_post_step_skips_a_cut_last_line_and_states_it(cut_run, caplog):
    from topopt import post
    out, total = cut_run
    with caplog.at_level("WARNING", logger="topopt.post"):
        metrics = post.finish(out)
    assert metrics["watertight"] is True and (out / "result" / "final.stl").exists()
    log = (out / "run.log").read_text()
    assert f"warning: history.jsonl: skipped a partial last line (line {total}" in log
    assert f"The metrics are from iteration {total - 2}; the mesh is from the last snapshot on disk." in log
    assert any("partial last line" in r.message for r in caplog.records)
    assert json.loads((out / "status.json").read_text())["state"] == "done"
    # The metrics come from the last row that parses (iteration total-2). That is the fact the log line gives.
    assert metrics["compliance_final"] == read_history(out / "history.jsonl").rows[-1]["compliance"]


def test_post_step_stops_on_a_bad_line_that_is_not_the_last(tmp_path):
    import fake_run
    from topopt import post
    fake_run.run(tmp_path, spi=0.0, n_it=5)
    lines = (tmp_path / "history.jsonl").read_text().splitlines()
    lines[2] = lines[2][:30]
    (tmp_path / "history.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(HistoryError, match="line 3"):
        post.finish(tmp_path)


def test_run_status_works_on_a_cut_last_line_and_states_it(cut_run):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    out, total = cut_run
    st = server.run_status(str(out))
    assert st["state"] == "done"
    assert st["last_history"]["iteration"] == total - 2          # the last row that parses
    assert len(st["history_facts"]) == 1 and "partial last line" in st["history_facts"][0]
    assert "partial last line" in st["summary"]


def test_run_status_without_a_cut_line_has_no_fact(tmp_path):
    pytest.importorskip("mcp")
    import fake_run
    from topopt_mcp import server
    fake_run.run(tmp_path, spi=0.0, n_it=4)
    st = server.run_status(str(tmp_path))
    assert st["history_facts"] == [] and st["job"] is None


def test_last_history_of_the_mcp_server_uses_the_shared_reader(tmp_path):
    """A byte that is not UTF-8 and a cut last line: no crash, the last row that parses."""
    pytest.importorskip("mcp")
    from topopt_mcp import server
    good = json.dumps({"iteration": 1, "compliance": 2.5}).encode()
    (tmp_path / "history.jsonl").write_bytes(
        json.dumps({"iteration": 0}).encode() + b"\n" + good + b"\n" + b'{"iteration": 2, "note": "\xff"}\n' + b'{"iteration": 3, "comp')
    assert server._last_history(tmp_path) == {"iteration": 2, "note": "�"}
    (tmp_path / "history.jsonl").write_bytes(b'{"iteration": 0, "comp')
    assert server._last_history(tmp_path) is None
    assert server._last_history(tmp_path / "missing") is None


def test_sha256_file_reads_in_blocks_and_matches_hashlib(tmp_path):
    import hashlib
    data = os.urandom(3 * 1024 + 17)
    f = tmp_path / "part.stl"
    f.write_bytes(data)
    want = hashlib.sha256(data).hexdigest()
    assert runinfo.sha256_file(f) == want
    assert runinfo.sha256_file(str(f), block=1000) == want
    (tmp_path / "empty").write_bytes(b"")
    assert runinfo.sha256_file(tmp_path / "empty") == hashlib.sha256(b"").hexdigest()


def test_region_files_keys_by_role_and_relative_path(tmp_path):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Load:
        name: str
        path: Path

    @dataclass(frozen=True)
    class Regions:  # the fields of config.Regions plus a field that a later version adds
        design: Path
        fixed: tuple
        keep_out: tuple
        load: tuple
        solid: tuple
        extra: Path | None = None

    y = tmp_path / "case"
    r = Regions(design=y / "design.stl",
                fixed=(y / "a" / "clamp.stl", y / "b" / "clamp.stl"),
                keep_out=(),
                load=(Load("tip", y / "tip.stl"),),
                solid=(tmp_path / "shared" / "boss.stl",))
    files = runinfo.region_files(r, y)
    assert set(files) == {"design:design.stl", "fixed:a/clamp.stl", "fixed:b/clamp.stl", "load:tip.stl",
                          "solid:../shared/boss.stl"}
    assert files["fixed:b/clamp.stl"] == y / "b" / "clamp.stl"



# --- region_files: the outside rule and the path that the YAML names (added with the D2 refresh) -----------------

def _yaml_with_regions(folder, regions):
    import yaml
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "c.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "regions": regions}))
    return p


def test_region_files_outside_the_root_get_a_hash_key_without_a_folder_name(tmp_path):
    from types import SimpleNamespace
    work, away = tmp_path / "work", tmp_path / "jdoe-private" / "parts"
    y = work / "case"
    r = SimpleNamespace(design=y / "design.stl", fixed=(), keep_out=(), load=(),
                        solid=(work / "shared" / "boss.stl", away / "tip.stl"))
    old = runinfo.region_files(r, y)                                  # no root: the old keys with `..`
    assert "solid:../../jdoe-private/parts/tip.stl" in old
    files = runinfo.region_files(r, y, root=work)
    keys = sorted(files)
    assert keys[0] == "design:design.stl" and keys[1] == "solid:../shared/boss.stl"          # inside the root: as before
    (outside,) = [k for k in keys if "<outside>" in k]
    assert re.fullmatch(r"solid:<outside>/[0-9a-f]{12}/tip\.stl", outside)
    assert "jdoe-private" not in "".join(keys)
    assert files[outside] == away / "tip.stl"                          # the value is still the path to read
    assert runinfo.region_files(r, y, root=work) == files                # the key is stable
    assert runinfo.outside_key(away / "tip.stl") == outside.split(":", 1)[1].rsplit("/", 1)[0]


def test_region_files_key_a_link_by_the_path_that_the_yaml_names(tmp_path):
    y = tmp_path / "case"
    (y / "store").mkdir(parents=True)
    (y / "store" / "d1.stl").write_bytes(b"solid d1")
    (y / "link.stl").symlink_to(Path("store") / "d1.stl")
    (y / "p.stl").write_bytes(b"p")
    p = _yaml_with_regions(y, {"design": "link.stl", "fixed": ["p.stl"], "load": {"tip": "p.stl"}})
    from types import SimpleNamespace
    resolved = SimpleNamespace(design=(y / "link.stl").resolve(), fixed=((y / "p.stl").resolve(),),
                               load=(SimpleNamespace(name="tip", path=(y / "p.stl").resolve()),))
    assert "design:store/d1.stl" in runinfo.region_files(resolved, y)                  # the loader resolves a link
    files = runinfo.region_files(resolved, y, root=tmp_path, yaml_path=p)
    assert sorted(files) == ["design:link.stl", "fixed:p.stl", "load:p.stl"]
    assert runinfo.sha256_file(files["design:link.stl"]) == runinfo.sha256_file(y / "store" / "d1.stl")   # the hash is of the content
    # a role whose number of names differs from the number of paths keeps the paths of `regions`
    two = SimpleNamespace(design=(y / "link.stl").resolve(), fixed=((y / "p.stl").resolve(), (y / "p.stl").resolve()), load=())
    assert "fixed:p.stl" in runinfo.region_files(two, y, root=tmp_path, yaml_path=p)


def test_region_files_without_regions_use_the_names_of_the_yaml_text(tmp_path):
    y = tmp_path / "case"
    p = _yaml_with_regions(y, {"design": "d.stl", "fixed": ["a.stl", "sub/b.stl"], "load": {"tip": "t.stl"}})
    files = runinfo.region_files(None, y, root=tmp_path, yaml_path=p)           # the problem does not load: no regions
    assert sorted(files) == ["design:d.stl", "fixed:a.stl", "fixed:sub/b.stl", "load:t.stl"]
    assert runinfo.region_files(None, y) == {} and runinfo.region_files(None, y, yaml_path=y / "nope.yaml") == {}
    assert runinfo.region_names(y / "nope.yaml") is None
    (y / "bad.yaml").write_text("regions: [unclosed\n")
    assert runinfo.region_names(y / "bad.yaml") is None
    assert runinfo.region_names(p) == {"design": ["d.stl"], "fixed": ["a.stl", "sub/b.stl"], "load": ["t.stl"]}


def test_key_root_is_the_work_folder_when_the_yaml_lies_in_it(tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "runs" / "part").mkdir(parents=True)
    (tmp_path / "other").mkdir()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(work))
    assert runinfo.key_root(work / "runs" / "part" / "c.yaml") == work.resolve()
    assert runinfo.key_root(tmp_path / "other" / "c.yaml") == (tmp_path / "other").resolve()      # not in the work folder
    monkeypatch.delenv("TOPOPT_WORKSPACE")
    monkeypatch.chdir(work)
    assert runinfo.key_root(work / "runs" / "part" / "c.yaml") == work.resolve()
