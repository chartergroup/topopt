"""Tests of tools/bench_gpu_mg.py (the measurement script for a Mac). A small cantilever replaces the hook."""
import json
import sys
from pathlib import Path

import pytest

from test_optimize import make_case

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import bench_gpu_mg  # noqa: E402


def md_text(out):
    return (out / "bench_gpu_mg.md").read_text()


def test_dry_run_skips_a_default_run_that_does_not_fit(tmp_path, monkeypatch, capsys):
    """The memory estimate above the RAM: the default run is skipped, with the numbers as a fact."""
    case = make_case(tmp_path, h=2.0, iterations=2)
    monkeypatch.setattr(bench_gpu_mg, "memory_fact", lambda p: {
        "free_dofs": 905010, "active_voxels": 319432, "estimate_bytes": 23_600_000_000,
        "ram_bytes": 16_000_000_000, "fits": False})
    out = tmp_path / "bench"
    assert bench_gpu_mg.main(["--out", str(out), "--hook", str(case), "--sizes", "2.0", "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "SKIPPED" in text and "memory estimate of topopt check" in text
    report = json.loads((out / "bench_gpu_mg.json").read_text())
    runs = {r["solver"]: r for r in report["runs"]}
    assert runs["default"]["status"] == "skipped" and "above the RAM" in runs["default"]["skipped_reason"]
    assert runs["gpu-mg"]["status"] == "dry-run"
    md = (out / "bench_gpu_mg.md").read_text()
    assert "skipped (memory estimate" in md and "max_active_voxels used: 400000" in md


def test_run_default_always_runs_it(tmp_path, monkeypatch):
    case = make_case(tmp_path, h=2.0, iterations=2)
    monkeypatch.setattr(bench_gpu_mg, "memory_fact", lambda p: {
        "free_dofs": 1, "active_voxels": 1, "estimate_bytes": 10**12, "ram_bytes": 10**9, "fits": False})
    out = tmp_path / "bench"
    bench_gpu_mg.main(["--out", str(out), "--hook", str(case), "--sizes", "2.0", "--dry-run", "--run-default", "always"])
    report = json.loads((out / "bench_gpu_mg.json").read_text())
    assert [r["status"] for r in report["runs"]] == ["dry-run", "dry-run"]


def test_the_yaml_copy_has_absolute_paths_and_the_settings(tmp_path):
    case = make_case(tmp_path, h=2.0, iterations=2)
    target = tmp_path / "copy.yaml"
    bench_gpu_mg.write_yaml(case, target, 1.5, 7, "gpu-mg", 123456, "C", "0")
    import yaml
    d = yaml.safe_load(target.read_text())
    assert Path(d["regions"]["design"]).is_absolute() and Path(d["regions"]["load"]["tip"]).is_absolute()
    st = d["settings"]
    assert (st["voxel_size"], st["iterations"], st["solver"], st["max_active_voxels"]) == (1.5, 7, "gpu-mg", 123456)
    assert (st["gpu_case"], st["gpu_device"]) == ("C", "0")
    bench_gpu_mg.write_yaml(case, target, 1.5, 7, "auto", 123456, "C", "0")
    assert "gpu_case" not in yaml.safe_load(target.read_text())["settings"]


@pytest.fixture(scope="module")
def dev():
    """The device that `auto` takes (the child process of the bench takes it too). The test is skipped when there is none, or without pyopencl."""
    pytest.importorskip("pyopencl")
    from topopt.config import UserError
    from topopt.gpu_mg import device
    try:
        return device.select("auto", "auto")[0]
    except UserError as e:
        pytest.skip(f"no usable OpenCL device: {e}")


@pytest.mark.opencl_context       # the child process makes an OpenCL context (skipped on a Mac, see conftest.py)
def test_a_real_job_one_at_a_time_writes_json_and_a_table(tmp_path, dev):
    pytest.importorskip("pyopencl")
    case = make_case(tmp_path, h=2.0, iterations=2)
    out = tmp_path / "bench"
    assert bench_gpu_mg.main(["--out", str(out), "--hook", str(case), "--sizes", "2.0", "--iterations", "3"]) == 0
    report = json.loads((out / "bench_gpu_mg.json").read_text())
    runs = {r["solver"]: r for r in report["runs"]}
    assert runs["default"]["status"] == runs["gpu-mg"]["status"] == "ok"
    g = runs["gpu-mg"]
    assert g["solver_run"] == "gpu-mg" and g["iterations_done"] == 4 and g["peak_rss_mb"] > 0
    assert len(g["gpu_mg"]["per_iteration"]) == 4 and g["gpu_mg"]["optimizer_iterations"] == 4
    assert {"iteration", "kinds", "setup_s", "upload_s", "solve_s", "map_s", "device_iterations"} <= set(
        g["gpu_mg"]["per_iteration"][0])
    assert "per OPTIMIZER iteration" not in md_text(out) and "OPTIMIZER iteration" in md_text(out)
    assert abs(g["final_compliance"] - runs["default"]["final_compliance"]) / runs["default"]["final_compliance"] < 1e-6
    assert report["opencl"]["device"]["name"] and "fp64" in report["opencl"]["device"]
    md = (out / "bench_gpu_mg.md").read_text()
    assert "| 2.0 | gpu-mg | ok |" in md and "rel. diff vs default" in md
    assert report["machine"]["ram_bytes"] and report["max_active_voxels"] == 400000


# --- review fixes: units, paths, time limit, records per optimizer iteration, dirty tree ----------------------------


@pytest.mark.parametrize("system,factor", [("linux", 1024), ("darwin", 1)])
def test_peak_rss_is_converted_to_bytes_first(monkeypatch, system, factor):
    """KiB on Linux, bytes on macOS (KB `pitfalls/ru-maxrss-units-linux`). 2,005,000 KiB is 2,053 MB, not 2,005 MB."""
    from types import SimpleNamespace
    monkeypatch.setattr(sys, "platform", system)
    ru = SimpleNamespace(ru_maxrss=2_005_000)
    assert bench_gpu_mg.peak_rss_bytes(ru) == 2_005_000 * factor
    assert bench_gpu_mg.child_peak_rss_mb(ru) == pytest.approx(2_005_000 * factor / 1e6)


def test_the_hook_path_in_the_report_is_relative_or_a_file_name(tmp_path):
    assert bench_gpu_mg.hook_label(bench_gpu_mg.DEFAULT_HOOK) == "examples/hook/hook.yaml"
    assert bench_gpu_mg.hook_label(tmp_path / "x" / "c.yaml") == "c.yaml"


def test_scrub_removes_the_paths_of_this_machine(tmp_path):
    out = tmp_path / "bench"
    text = f"error: no STL in {out}/h1.2_default.yaml and {bench_gpu_mg.REPO}/examples and {Path.home()}/x"
    assert bench_gpu_mg.scrub(text, out) == "error: no STL in <out>/h1.2_default.yaml and <repo>/examples and ~/x"


def test_the_report_files_hold_no_absolute_path(tmp_path, capsys):
    case = make_case(tmp_path, h=2.0, iterations=2)
    out = tmp_path / "bench"
    assert bench_gpu_mg.main(["--out", str(out), "--hook", str(case), "--sizes", "2.0", "--dry-run"]) == 0
    for name in ("bench_gpu_mg.json", "bench_gpu_mg.md"):
        text = (out / name).read_text()
        assert str(tmp_path) not in text and str(bench_gpu_mg.REPO) not in text, name
        assert sys.executable not in text, name
    assert json.loads((out / "bench_gpu_mg.json").read_text())["hook"] == "c.yaml"


def test_scrub_tree_removes_the_python_the_repo_and_the_out_folder(tmp_path):
    """The install hint of a missing pyopencl names the Python and the repo (found with the pyopencl stub)."""
    hint = f"uv pip install --python {sys.executable} -e '{bench_gpu_mg.REPO}[gpu]' (log in {tmp_path}/run)"
    obj = {"opencl": {"error": hint, "n": 3, "list": [hint, 1.5, None]}, "ok": True}
    text = json.dumps(bench_gpu_mg.scrub_tree(obj, tmp_path))
    for path in (sys.executable, str(bench_gpu_mg.REPO), str(tmp_path)):
        assert path not in text
    assert "<python>" in text and "<repo>[gpu]" in text and "<out>/run" in text
    assert bench_gpu_mg.scrub_tree({"n": 3, "x": None}, tmp_path) == {"n": 3, "x": None}


def test_the_sample_report_holds_no_absolute_path():
    f = REPO / "tests" / "data" / "bench_gpu_mg_pocl_sample.json"
    text = f.read_text()
    assert "/app" + "/data" not in text and "/Users/" not in text and "/home/" not in text, f


def test_the_default_time_limit_is_four_times_the_estimate_with_a_floor(monkeypatch, tmp_path):
    for estimate, expect in ((10.0, 600.0), (100.0, 600.0), (500.0, 2000.0), (20000.0, 80000.0)):
        monkeypatch.setattr(bench_gpu_mg, "check_data", lambda p, solver_setting=None, e=estimate: {"time_estimate": {"total_s": e}})
        assert bench_gpu_mg.default_timeout_s(tmp_path / "x.yaml") == expect
    monkeypatch.setattr(bench_gpu_mg, "check_data", lambda *a, **k: (_ for _ in ()).throw(bench_gpu_mg.UserError("bad")))
    assert bench_gpu_mg.default_timeout_s(tmp_path / "x.yaml") == 600.0


def test_a_run_over_the_time_limit_is_stopped_and_marked_timed_out(tmp_path):
    case = make_case(tmp_path, h=2.0, iterations=2)
    out = tmp_path / "bench"
    code = bench_gpu_mg.main(["--out", str(out), "--hook", str(case), "--sizes", "2.0", "--solvers", "default",
                              "--iterations", "3", "--timeout-s", "0.05"])
    assert code == 1
    report = json.loads((out / "bench_gpu_mg.json").read_text())
    run = report["runs"][0]
    assert run["status"] == "timed_out" and run["timed_out"] and run["timeout_s"] == 0.05
    assert "time limit" in run["message"]
    assert "timed_out" in md_text(out)


def test_the_default_limit_is_recorded_for_a_job(tmp_path):
    case = make_case(tmp_path, h=2.0, iterations=2)
    out = tmp_path / "bench"
    assert bench_gpu_mg.main(["--out", str(out), "--hook", str(case), "--sizes", "2.0", "--solvers", "default",
                              "--iterations", "2"]) == 0
    run = json.loads((out / "bench_gpu_mg.json").read_text())["runs"][0]
    assert run["timeout_s"] >= bench_gpu_mg.TIMEOUT_FLOOR_S and run["status"] == "ok"


def test_gpu_mg_lines_are_summed_for_each_optimizer_iteration():
    def line(n, kind, it, solve_s, its):
        return {"line": n, "kind": kind, "iteration": it, "solve": 0, "case": "C", "setup_s": 0.1 if kind != "adjoint" else 0.0,
                "upload_s": 0.01, "solve_s": solve_s, "map_s": 0.001, "total_s": 0.111 + solve_s, "solves": 1,
                "iterations": [its]}
    lines = [line(0, "main", 0, 1.0, {"inner": [10, 8], "refine_steps": 2}),
             line(1, "adjoint", 0, 0.5, {"inner": [5]}),
             line(2, "main", 1, 1.0, {"pcg": [20], "executed": 21}),
             line(3, "trial", 1, 1.0, {"pcg": [20], "executed": 21, "inner": [4], "refine_steps": 1}),
             line(4, "trial", 1, 1.0, {"inner": [3]})]
    g = bench_gpu_mg.gpu_summary(lines)
    assert g["records"] == 5 and g["optimizer_iterations"] == 2 and g["records_by_kind"] == {"main": 2, "adjoint": 1, "trial": 2}
    first, second = g["per_iteration"]
    assert first["iteration"] == 0 and first["kinds"] == {"main": 1, "adjoint": 1}
    assert first["solve_s"] == pytest.approx(1.5) and first["device_iterations"] == 10 + 8 + 5
    assert second["kinds"] == {"main": 1, "trial": 2} and second["solve_s"] == pytest.approx(3.0)
    assert second["device_iterations"] == 21 + (21 + 4) + 3
    assert g["median_s"]["solve_s"] == pytest.approx(2.25) and "OPTIMIZER" not in g["per"] and "optimizer iteration" in g["per"]


def test_a_run_summary_has_the_number_of_load_cases(tmp_path):
    """`topopt gpu-check` takes the columns for each launch (1 or 2) from it, for the constants of the time model (P2a)."""
    (tmp_path / "run.json").write_text(json.dumps({"solver": "gpu-mg", "n_free_dofs": 1234, "n_active": 99,
                                                   "load_cases": [{"name": "a"}, {"name": "b"}]}))
    out = bench_gpu_mg.summarize_run(tmp_path, {"h": 1.2})
    assert out["n_load_cases"] == 2 and out["free_dofs"] == 1234 and out["solver_run"] == "gpu-mg"
    (tmp_path / "run.json").write_text(json.dumps({"solver": "gpu-mg", "n_free_dofs": 1234, "n_active": 99}))
    assert bench_gpu_mg.summarize_run(tmp_path, {"h": 1.2})["n_load_cases"] == 0       # unknown: the calibration says so and is not made


def test_the_report_says_when_the_working_tree_was_not_clean(tmp_path, monkeypatch):
    import subprocess
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.txt").write_text("one\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "first"], check=True)
    monkeypatch.setattr(bench_gpu_mg, "REPO", repo)
    rev, dirty = bench_gpu_mg.git_state()
    assert rev and dirty is False
    (repo / "b.txt").write_text("two\n")                      # an untracked file counts
    assert bench_gpu_mg.git_state()[1] is True
    report = {"machine": {"platform": "p", "cpu": "c", "cores": 1, "ram": "1 GiB", "macos": None, "python": "3",
                          "topopt": "0", "git_rev": rev, "git_dirty": True},
              "opencl": {}, "max_active_voxels": 1, "iterations": 1, "note": "n", "runs": []}
    assert "dirty" in bench_gpu_mg.markdown(report)


# --- the parity tool (review n3) ------------------------------------------------------------------------------------------


def _fake_device(fp64):
    from topopt.gpu_mg.device import DeviceInfo
    return DeviceInfo(index=0, platform="Test", name="Test device", kind="CPU", fp64=fp64, version="OpenCL 1.2",
                      max_work_group_size=256, max_mem_alloc=2**30, global_mem=2**32, local_mem=32768,
                      constant_mem=65536, compute_units=4)


def test_parity_tool_fails_when_no_case_ran(monkeypatch, capsys):
    """Every case skipped (case B needs 64-bit floats): the tool does not print PASS."""
    import gpu_mg_parity
    monkeypatch.setattr(gpu_mg_parity.device, "select", lambda d="auto", c="auto": (_fake_device(False), "C"))
    assert gpu_mg_parity.main(["--h", "2.0", "--cases", "B"]) == 1
    text = capsys.readouterr().out
    assert "case B: skipped" in text and "FAIL: no case ran" in text and "PASS" not in text


def test_parity_tool_reports_a_failed_build_as_a_fail_line(monkeypatch, capsys):
    import gpu_mg_parity
    from topopt.config import UserError

    def broken(*a, **k):
        raise UserError("solver 'gpu-mg': the OpenCL device Test device could not build the program")

    monkeypatch.setattr(gpu_mg_parity.device, "select", lambda d="auto", c="auto": (_fake_device(True), "B"))
    monkeypatch.setattr(gpu_mg_parity, "GpuModel", broken)
    assert gpu_mg_parity.main(["--h", "2.0", "--cases", "B,C"]) == 1
    text = capsys.readouterr().out
    assert "case B: FAIL. solver 'gpu-mg': the OpenCL device Test device could not build" in text
    assert "case C: FAIL." in text and "PASS" not in text and "Traceback" not in text


def test_parity_tool_names_the_restart_of_case_b():
    import gpu_mg_parity
    assert gpu_mg_parity._iterations({"pcg": [11, 11], "executed": 12}) == "11 PCG"
    assert gpu_mg_parity._iterations({"inner": [5, 4, 4], "refine_steps": 3}) == "13 inner in 3 refinement steps"
    text = gpu_mg_parity._iterations({"pcg": [None, None], "executed": 300, "inner": [66, 179], "refine_steps": 2})
    assert text == "300 PCG (not converged), then 245 inner in 2 restart steps"
