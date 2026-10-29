"""#447, #449: a resource error in a step after `result/final.stl` is written stops the post step (CONTRACTS.md section 19, "Resource errors").

The part-only cut (`result/part_only.stl`) and the stress report (the stress block of `metrics.json`) used to catch every error. A memory error,
a failed thread start or a native library that could not be mapped under a memory cap then ended the run with exit 0, without the file or the
block. Their handlers now call `reraise_if_resource(exc, step, missing=...)` first: the post step stops with `PostResourceError`, which names
the step, the cause and what this pass did not write. Any other error keeps the old record (`part_only.error`, `stress.error`).
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import topopt.refine.stress as rs  # noqa: E402
from topopt import part_only, post, smooth_sdf  # noqa: E402
from topopt import stress_report as sr  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt.resource_errors import (MeshResourceError, PostResourceError, ResourceStopError, is_resource_error,  # noqa: E402
                                    reraise_if_resource)
from topopt.runjob import exit_code_of  # noqa: E402

from test_mesh_resource_errors import _route_args  # noqa: E402
from test_part_only import ALL_KINDS, write_case  # noqa: E402
from test_stress_report import stress_yaml  # noqa: E402

MEMORY = MemoryError("Unable to allocate 14.7 MiB for an array with shape (642023, 3) and data type float64")
THREAD = RuntimeError("can't start new thread")
BOTH = pytest.mark.parametrize("exc", [MEMORY, THREAD], ids=["MemoryError", "no-thread"])
PART_ONLY = "result/part_only.stl"
ROUTE = "To repeat only the post step on this run: `"


def llvmlite_error():
    """The error of numba's backend library under an address-space cap: llvmlite raises a plain OSError (no errno) from the loader's error."""
    try:
        try:
            raise OSError("<venv>/llvmlite/binding/libllvmlite.so: failed to map segment from shared object")
        except OSError as inner:
            raise OSError("Could not find/load shared object file") from inner
    except OSError as exc:
        return exc


def raising(exc):
    def boom(*a, **k):
        raise exc
    return boom


# -- the helper ----------------------------------------------------------------------------------------------------------------------------
def test_a_native_library_that_cannot_be_mapped_is_a_resource_error():
    exc = llvmlite_error()
    assert exc.errno is None and is_resource_error(exc)
    assert not is_resource_error(OSError("Could not find/load shared object file"))     # a library that is not there: an install problem


def test_reraise_if_resource_with_missing_gives_a_post_resource_error():
    reraise_if_resource(ValueError("x"), "the part-only cut", missing=PART_ONLY)          # returns
    with pytest.raises(PostResourceError) as got:
        reraise_if_resource(MEMORY, "the part-only cut", missing=PART_ONLY)
    err = got.value
    assert isinstance(err, ResourceStopError) and not isinstance(err, MeshResourceError) and err.__cause__ is MEMORY
    assert (err.step, err.missing) == ("the part-only cut", PART_ONLY)
    text = str(err)
    assert "the part-only cut" in text and "MemoryError: Unable to allocate 14.7 MiB" in text and "No part was written" not in text
    assert "result/final.stl is written, but result/part_only.stl is not" in text and "Options:" in text
    with pytest.raises(PostResourceError) as again:                                      # from an inner step: passes on as it is
        reraise_if_resource(err, "the stress report", missing="the stress block")
    assert again.value is err
    assert exit_code_of(err) == 2


def test_a_mesh_resource_error_of_a_helper_becomes_a_post_resource_error_that_names_both_steps():
    """The part-only cut calls the float32 clean-up of the mesh step. The mesh step had finished, so the error is a `PostResourceError`."""
    inner = MeshResourceError("the float32 clean-up (simplify)", "MemoryError: x")
    with pytest.raises(PostResourceError) as got:
        reraise_if_resource(inner, "the part-only cut", missing=PART_ONLY)
    assert got.value.step == "the float32 clean-up (simplify), in the part-only cut" and got.value.cause == "MemoryError: x"
    assert got.value.__cause__ is inner
    with pytest.raises(MeshResourceError) as same:                                      # without `missing` (the mesh step): as it is
        reraise_if_resource(inner, "the SDF mesh step")
    assert same.value is inner


# -- the part-only cut (#447) --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def keyed_run(tmp_path_factory):
    """A finished run of the bar of `test_part_only` with every kind of region in `regions.not_printed`."""
    folder = tmp_path_factory.mktemp("bar")
    out = folder / "keyed"
    assert main(["run", str(write_case(folder, ALL_KINDS, "keyed.yaml")), "--out", str(out), "--quiet"]) == 0
    assert (out / "result" / "part_only.stl").is_file()
    return out


def again(run, tmp_path):
    """A copy of a finished run for one more post step: `part_only.stl` of an older pass is there, `metrics.json` is not."""
    copy = tmp_path / "copy"
    shutil.copytree(run, copy)
    (copy / "result" / "part_only.stl").write_bytes(b"old")
    (copy / "result" / "final.stl").unlink()
    (copy / "result" / "metrics.json").unlink()
    return copy


@BOTH
def test_a_resource_error_in_the_part_only_cut_stops_the_post_step(keyed_run, tmp_path, monkeypatch, exc):
    run_dir = again(keyed_run, tmp_path)
    monkeypatch.setattr(part_only, "cut", raising(exc))
    with pytest.raises(PostResourceError) as got:
        post.finish(run_dir)
    err = got.value
    assert (err.step, err.missing) == ("the part-only cut", PART_ONLY) and err.__cause__ is exc
    st = json.loads((run_dir / "status.json").read_text())
    assert st["state"] == "error" and st["step"] == "postprocess"
    for part in ("the part-only cut", f"{type(exc).__name__}: {exc}", "result/part_only.stl is not", ROUTE):
        assert part in st["message"], part
    assert (run_dir / "result" / "final.stl").is_file()                                   # this pass wrote final.stl
    assert not (run_dir / "result" / "part_only.stl").exists()                            # the older file is gone: it fits an older final.stl
    assert not (run_dir / "result" / "metrics.json").exists()


def test_a_resource_error_in_the_clean_up_inside_the_cut_names_both_steps(keyed_run, tmp_path, monkeypatch):
    run_dir = again(keyed_run, tmp_path)
    box = trimesh.creation.box(extents=(2, 2, 2))

    def cut(*a, **k):    # the real clean-up of the mesh step, as the cut calls it, with a memory error in its manifold
        with monkeypatch.context() as mp:
            mp.setattr(smooth_sdf, "_manifold_of", raising(MEMORY))
            smooth_sdf._cleanup_float32(np.asarray(box.vertices, float), np.asarray(box.faces, np.int64), 1e-4)
        raise AssertionError("the clean-up did not raise")
    monkeypatch.setattr(part_only, "cut", cut)
    with pytest.raises(PostResourceError) as got:
        post.finish(run_dir)
    assert got.value.step == "the float32 clean-up (simplify), in the part-only cut" and "Unable to allocate 14.7 MiB" in got.value.cause
    assert "the float32 clean-up (simplify), in the part-only cut" in json.loads((run_dir / "status.json").read_text())["message"]


def test_another_error_in_the_part_only_cut_keeps_the_record_and_the_run_ends(keyed_run, tmp_path, monkeypatch):
    run_dir = again(keyed_run, tmp_path)
    monkeypatch.setattr(part_only, "cut", raising(ValueError("the cut leaves no material")))
    m = post.finish(run_dir)
    assert m["part_only"]["applied"] is False and m["part_only"]["error"] == "ValueError: the cut leaves no material"
    assert not (run_dir / "result" / "part_only.stl").exists()
    assert json.loads((run_dir / "status.json").read_text())["state"] == "done"


def test_topopt_run_stops_with_exit_2_and_the_route_makes_the_file(tmp_path, monkeypatch):
    """The CLI path: a failed thread start in the part-only cut. Exit 2, the status names the step; the route of the message repeats only
    the post step, and without the error it writes part_only.stl. run_status and run_result report the failure in between."""
    from topopt_mcp import server

    yaml_path = write_case(tmp_path, ALL_KINDS, "keyed.yaml")
    out = tmp_path / "run"
    with monkeypatch.context() as mp:
        mp.setattr(part_only, "cut", raising(THREAD))
        assert main(["run", str(yaml_path), "--out", str(out), "--quiet", "--no-wait"]) == 2
    st = json.loads((out / "status.json").read_text())
    assert st["state"] == "error" and "the part-only cut" in st["message"] and "can't start new thread" in st["message"]
    assert (out / "result" / "final.stl").is_file() and not (out / "result" / "part_only.stl").exists()
    assert server.run_status(str(out))["state"] == "error"
    res = server.run_result(str(out), preview=False)
    assert res["done"] is False and res["message"].startswith("The run failed: ") and "the part-only cut" in res["message"]
    assert "final_stl" not in res
    assert main(_route_args(st["message"])) == 0
    assert (out / "result" / "part_only.stl").is_file()
    assert json.loads((out / "result" / "metrics.json").read_text())["part_only"]["applied"] is True
    assert json.loads((out / "status.json").read_text())["state"] == "done"


# -- the stress report (#449) --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stress_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stress_run")
    out = tmp / "r"
    assert main(["run", str(stress_yaml(tmp, iterations=4)), "--out", str(out), "--quiet"]) == 0
    assert "error" not in json.loads((out / "result" / "metrics.json").read_text())["stress"]
    return out


def solves(*outcomes):
    """A `compute_stress` whose calls give the outcomes in turn: an exception is raised, None runs the real solve."""
    real, todo = rs.compute_stress, list(outcomes)

    def compute(*a, **k):
        what = todo.pop(0)
        if what is not None:
            raise what
        return real(*a, **k)
    return compute


@pytest.mark.parametrize("exc", [MEMORY, THREAD, llvmlite_error()], ids=["MemoryError", "no-thread", "libllvmlite"])
def test_a_resource_error_in_the_stress_report_raises(stress_run, monkeypatch, exc):
    monkeypatch.setattr(rs, "compute_stress", solves(exc))
    with pytest.raises(PostResourceError) as got:
        sr.stress_section(stress_run)
    assert (got.value.step, got.value.missing) == ("the stress report", sr.STRESS_MISSING)
    assert "the stress block of result/metrics.json is not" in str(got.value)
    if exc.__cause__ is not None:      # libllvmlite: the cause names the loader's error, not the outer text
        assert "failed to map segment from shared object" in got.value.cause


def test_a_resource_error_in_the_thresholded_estimate_raises(stress_run, monkeypatch):
    monkeypatch.setattr(rs, "compute_stress", solves(None, MEMORY))
    with pytest.raises(PostResourceError) as got:
        sr.stress_section(stress_run)
    assert got.value.step == "the thresholded stress estimate of the stress report"


def test_other_errors_in_the_stress_report_keep_their_records(stress_run, monkeypatch):
    monkeypatch.setattr(rs, "compute_stress", solves(None, ValueError("singular")))
    s = sr.stress_section(stress_run)
    assert "error" not in s and {c["physical_thresholded"]["error"] for c in s["cases"].values()} == {"ValueError: singular"}
    monkeypatch.setattr(rs, "compute_stress", solves(ValueError("singular")))
    s = sr.stress_section(stress_run)
    assert s == {"error": "ValueError: singular", "note": "the stress report was not made"}


def test_update_metrics_does_not_change_the_file_on_a_resource_error(stress_run, tmp_path, monkeypatch):
    copy = tmp_path / "copy"
    shutil.copytree(stress_run, copy)
    before = (copy / "result" / "metrics.json").read_bytes()
    monkeypatch.setattr(rs, "compute_stress", solves(MEMORY))
    with pytest.raises(PostResourceError):
        sr.update_metrics(copy)
    assert (copy / "result" / "metrics.json").read_bytes() == before


def test_a_resource_error_in_the_stress_report_stops_the_post_step(stress_run, tmp_path, monkeypatch):
    copy = tmp_path / "copy"
    shutil.copytree(stress_run, copy)
    (copy / "result" / "metrics.json").unlink()
    monkeypatch.setattr(rs, "compute_stress", solves(llvmlite_error()))
    with pytest.raises(PostResourceError):
        post.finish(copy)
    st = json.loads((copy / "status.json").read_text())
    assert st["state"] == "error" and "the stress report" in st["message"] and "failed to map segment from shared object" in st["message"]
    assert "the stress block of result/metrics.json is not" in st["message"] and ROUTE in st["message"]
    assert (copy / "result" / "final.stl").is_file() and not (copy / "result" / "metrics.json").exists()


def test_topopt_run_with_a_memory_error_in_the_stress_report_ends_with_exit_2(tmp_path, monkeypatch):
    out = tmp_path / "r"
    monkeypatch.setattr(rs, "compute_stress", solves(MEMORY))
    assert main(["run", str(stress_yaml(tmp_path, iterations=4)), "--out", str(out), "--quiet", "--no-wait"]) == 2
    st = json.loads((out / "status.json").read_text())
    assert st["state"] == "error" and "the stress report" in st["message"] and "Unable to allocate 14.7 MiB" in st["message"]
    assert not (out / "result" / "metrics.json").exists()
