"""`topopt verify` on the recorded hook run (CONTRACTS.md section 16): the command, the exit codes, the files.

The recorded run is `recordings/hook_*` (its `run.json` names its YAML relative to the run folder; the tests work on a
copy in a temporary folder, where that path is not valid, so they give the YAML of this repo with `--yaml`). Tests that mesh need
gmsh (`needs_gmsh`) and scikit-fem.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import fake_factor_availability, needs_gmsh, needs_skfem, recorded_hook_run  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RECORDING = recorded_hook_run()
HOOK_YAML = REPO / "examples" / "hook" / "hook.yaml"

needs_recording = pytest.mark.skipif(not (RECORDING / "result" / "final.stl").is_file(),
                                     reason="needs the recorded hook run recordings/hook_*")


def copy_run(dst: Path) -> Path:
    """A small copy of the recorded run: run.json, final.stl and metrics.json (the check writes into the run folder)."""
    (dst / "result").mkdir(parents=True)
    shutil.copy(RECORDING / "run.json", dst / "run.json")
    for name in ("final.stl", "metrics.json"):
        shutil.copy(RECORDING / "result" / name, dst / "result" / name)
    return dst


@pytest.fixture
def run_dir(tmp_path):
    if not (RECORDING / "result" / "final.stl").is_file():
        pytest.skip("needs the recorded hook run recordings/hook_*")
    return copy_run(tmp_path / "hook")


def cli(*argv):
    from topopt.cli import main
    return main(["verify", *map(str, argv)])


# --------------------------------------------------------------------------- #
# Inputs and the exit code 1
# --------------------------------------------------------------------------- #

def test_run_folder_errors_are_user_errors(tmp_path):
    from topopt.config import UserError
    from topopt.verify.run import read_run
    with pytest.raises(UserError, match="run folder not found"):
        read_run(tmp_path / "none")
    (tmp_path / "empty").mkdir()
    with pytest.raises(UserError, match="no run.json"):
        read_run(tmp_path / "empty")
    (tmp_path / "empty" / "run.json").write_text("{}")
    with pytest.raises(UserError, match="no final mesh"):
        read_run(tmp_path / "empty")


@needs_recording
def test_missing_yaml_names_the_yaml_option(run_dir):
    from topopt.config import UserError
    from topopt.verify.run import read_run
    with pytest.raises(UserError) as e:
        read_run(run_dir)                                      # the relative path of run.json is not valid in the copy
    assert "--yaml" in str(e.value) and "missing" in str(e.value)
    info = read_run(run_dir, yaml_path=HOOK_YAML)
    assert info.voxel_mm == 1.2 and info.cfg.name == "tshirt_hook"


@needs_recording
def test_verify_reads_nominal_stl_when_the_run_has_one(run_dir):
    """A run with machining stock (CONTRACTS.md section 23) has final.stl (as printed) and nominal.stl (the part in service)."""
    from topopt.verify import verify_run
    from topopt.verify.run import read_run
    info = read_run(run_dir, yaml_path=HOOK_YAML)
    assert info.final_stl.name == "final.stl" and info.verified_file == "final.stl"
    shutil.copy(run_dir / "result" / "final.stl", run_dir / "result" / "nominal.stl")
    info = read_run(run_dir, yaml_path=HOOK_YAML)
    assert info.final_stl.name == "nominal.stl" and info.verified_file == "nominal.stl"
    res = verify_run(run_dir, yaml_path=HOOK_YAML, estimate_only=True)
    assert res["run"]["verified_file"] == "nominal.stl" and res["run"]["final_stl"].endswith("nominal.stl")


@needs_recording
def test_exit_code_1_for_bad_input_and_usage(run_dir, capsys):
    assert cli(run_dir, "--yaml", HOOK_YAML, "--mesh-size", "abc") == 1
    assert "must be a number" in capsys.readouterr().err
    assert cli(run_dir, "--yaml", HOOK_YAML, "--mesh-size", "-1") == 1
    assert cli(run_dir, "--yaml", HOOK_YAML, "--mesh-size", "1.5", "--levels", "1.5,1.0") == 1
    assert cli(run_dir, "--yaml", HOOK_YAML, "--percentile", "120") == 1
    assert cli(run_dir, "--yaml", HOOK_YAML, "--levels", "1.5,x") == 1
    assert cli(run_dir, "--yaml", HOOK_YAML, "--solver", "foo") == 1
    assert "unknown solver" in capsys.readouterr().err
    assert cli(run_dir, "--yaml", HOOK_YAML, "--solver", "amg") == 1
    assert cli(run_dir) == 1                                    # no valid YAML in run.json
    assert cli(run_dir / "nope") == 1
    with pytest.raises(SystemExit) as e:
        cli("--bogus")
    assert e.value.code == 1                                    # a usage error is exit code 1, not 2
    with pytest.raises(SystemExit) as e:
        from topopt.cli import main
        main(["verify"])                                        # the run folder is missing
    assert e.value.code == 1


# --------------------------------------------------------------------------- #
# Estimates, the memory refusal and the missing extras (no meshing)
# --------------------------------------------------------------------------- #

@needs_recording
def test_estimate_only_prints_the_estimate_for_each_size_and_meshes_nothing(run_dir, capsys):
    assert cli(run_dir, "--yaml", HOOK_YAML, "--estimate-only", "--levels", "1.5,1.0,0.6") == 0
    out = capsys.readouterr().out
    for size in ("1.5", "1", "0.6"):
        assert f"mesh {size} mm" in out
    assert "memory estimate" in out and "default mesh size" in out
    assert not (run_dir / "verify").exists()


@needs_recording
def test_estimate_only_result_has_the_facts(run_dir):
    from topopt.verify import verify_run
    res = verify_run(run_dir, yaml_path=HOOK_YAML, estimate_only=True, levels=[1.5, 1.0], max_gb=2.0)
    assert res["ok"] and res["estimate_only"] and res["exit_code"] == 0
    assert [e["size_mm"] for e in res["estimates_before_meshing"]] == [1.5, 1.0]        # coarse to fine
    e15, e10 = res["estimates_before_meshing"]
    assert e15["estimated_dofs"] < e10["estimated_dofs"] and not e15["above_limit"] and e10["above_limit"]
    assert res["would_refuse"] == [1.0]
    solver = res["backend"]["solver"]
    if solver == "superlu":                                       # the model of SuperLU (it depends on the compactness)
        from topopt.verify import memory
        assert e15["estimate_bytes"] == pytest.approx(
            memory.estimate_bytes(e15["estimated_dofs"], "superlu", e15["compactness"]), rel=1e-3)
        assert e15["estimate_bytes"] == e15["bytes_by_backend"]["superlu"]
    else:
        assert e15["estimate_bytes"] == pytest.approx(7.5e9 * 1.25 * e15["estimated_dofs"] / 1e6, rel=1e-3)
    assert set(e15["bytes_by_backend"]) == {"cholesky", "pardiso", "cholespy", "superlu"}
    assert res["band_mm"] == pytest.approx(0.6) and res["band_is_default"]               # 0.5 x the voxel size 1.2 mm
    assert res["memory_limit"]["limit_bytes"] == 2_000_000_000
    assert set(res["extras"]) == {"gmsh", "scikit-fem", "cholespy"}
    assert "available" in res["backend"]


# --------------------------------------------------------------------------- #
# The material reference: the message of a run with no allowable (P4b-1, CONTRACTS.md section 25.7)
# --------------------------------------------------------------------------- #

MATERIAL_URI = "topopt://materials/reference"


@needs_recording
def test_a_result_with_no_allowable_names_the_material_reference_and_the_cli_option(run_dir):
    from topopt.verify import verify_run
    res = verify_run(run_dir, yaml_path=HOOK_YAML, estimate_only=True)
    fact = res["no_allowable"]
    assert fact["resource"] == MATERIAL_URI and fact["package_file"] == "topopt_mcp/materials/reference.md"
    note = fact["note"]
    assert MATERIAL_URI in note and "no default allowable" in note and "no limit block" in note
    assert "--allowable-mpa" in note and "allowable_mpa" not in note.replace("--allowable-mpa", "")
    assert "limit" not in res


@needs_recording
def test_the_note_for_the_mcp_tool_names_the_mcp_parameter(run_dir):
    from topopt.verify import names, verify_run
    with names.use_style("mcp"):
        note = verify_run(run_dir, yaml_path=HOOK_YAML, estimate_only=True)["no_allowable"]["note"]
    assert MATERIAL_URI in note and "allowable_mpa" in note and "--allowable-mpa" not in note


@needs_recording
def test_an_allowable_removes_the_note(run_dir):
    from topopt.verify import verify_run
    without = verify_run(run_dir, yaml_path=HOOK_YAML, estimate_only=True)
    given = verify_run(run_dir, yaml_path=HOOK_YAML, estimate_only=True, allowable_mpa=30.0)
    assert "no_allowable" in without and given["ok"] and "no_allowable" not in given
    assert set(without) - {"no_allowable"} == set(given)       # the note is the only difference


@needs_recording
def test_mcp_tool_result_without_an_allowable_names_the_resource_and_the_parameter(run_dir, tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    res = server.verify_run(run_dir=str(run_dir), yaml_path=str(HOOK_YAML))
    assert res["ok"] and MATERIAL_URI in res["no_allowable"]["note"] and "allowable_mpa" in res["no_allowable"]["note"]
    assert "--allowable-mpa" not in res["no_allowable"]["note"]
    given = server.verify_run(run_dir=str(run_dir), yaml_path=str(HOOK_YAML), allowable_mpa=30.0)
    assert given["ok"] and "no_allowable" not in given


def minimal_result(**extra):
    """The keys that `summary_lines` reads, with made-up values (a synthetic result: the real one needs gmsh)."""
    res = dict(run=dict(name="synthetic"), size_mm=1.5, band_mm=0.6, backend=dict(solver="superlu"),
               mesh=dict(tets=10, dofs=30, quality_min=0.5, below_0p05=0, p2_volume_over_stl=1.0),
               timing_s=dict(verify_total_s=1.0), memory=dict(peak_rss_bytes=10**9, estimate_after_meshing_bytes=10**9),
               cases={}, regions=dict(load=[]))
    res.update(extra)
    return res


def test_the_summary_has_one_limit_line_with_the_note_only_when_the_result_has_the_fact():
    from topopt.verify.report import summary_lines
    from topopt.verify.run import no_allowable_fact
    fact = no_allowable_fact()
    assert [x for x in summary_lines(minimal_result(no_allowable=fact)) if x.startswith("limit:")] == ["limit: " + fact["note"]]
    assert not [x for x in summary_lines(minimal_result()) if x.startswith("limit:")]


def fake_extras(gmsh="ok", skfem="ok"):
    g = {"state": gmsh, "version": "4.15.2" if gmsh == "ok" else None, "detail": "", "fix": ""}
    if gmsh == "load_failed":
        g.update(detail="OSError: libGLU.so.1: cannot open shared object file. Missing system libraries: libGLU.so.1",
                 fix="With root on Debian or Ubuntu: apt install libglu1-mesa")
    if gmsh == "missing":
        g.update(detail="gmsh is not installed", fix="Install it with: uv pip install -e '.[tet]'")
    return {"gmsh": g, "scikit-fem": {"state": skfem, "detail": "ModuleNotFoundError: no skfem"},
            "cholespy": {"state": "ok", "version": None}}


@needs_recording
def test_tiny_max_gb_refuses_before_meshing_with_exit_code_2(run_dir, monkeypatch, capsys):
    from topopt.verify import run as vrun
    from topopt.verify.errors import MemoryRefused
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras())          # the refusal needs no gmsh
    monkeypatch.setattr(vrun, "mesh_stl", None, raising=False)
    with pytest.raises(MemoryRefused) as e:
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, max_gb=0.01)
    assert e.value.facts["when"] == "before meshing" and e.value.facts["limit_source"] == "--max-gb"
    assert cli(run_dir, "--yaml", HOOK_YAML, "--max-gb", "0.01") == 2
    err = capsys.readouterr().err
    assert "memory estimate" in err and "--force" in err and "--max-gb" in err
    assert not (run_dir / "verify").exists()


def test_force_lifts_the_refusal():
    from topopt.verify.errors import MemoryRefused
    from topopt.verify.run import _check_estimate
    lim = {"limit_bytes": 1_000_000, "limit_source": "--max-gb"}
    with pytest.raises(MemoryRefused):
        _check_estimate(5_000_000, lim, 1.0, "after meshing", 100_000, False)
    _check_estimate(5_000_000, lim, 1.0, "after meshing", 100_000, True)          # no error
    _check_estimate(500_000, lim, 1.0, "after meshing", 100_000, False)           # below the limit
    _check_estimate(5_000_000, {"limit_bytes": None, "limit_source": None}, 1.0, "after meshing", 1, False)   # RAM not known


@needs_recording
def test_missing_gmsh_gives_exit_code_2_and_names_the_install_step_and_the_libraries(run_dir, monkeypatch, capsys):
    from topopt.verify import run as vrun
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras(gmsh="load_failed"))
    assert cli(run_dir, "--yaml", HOOK_YAML) == 2
    err = capsys.readouterr().err
    assert "gmsh" in err and "libGLU.so.1" in err
    if sys.platform == "linux":
        assert "libglu1-mesa libxft2 libxinerama1 libopengl0" in err
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras(gmsh="missing"))
    assert cli(run_dir, "--yaml", HOOK_YAML) == 2
    assert "uv pip install" in capsys.readouterr().err


def _hook_places(cfg):
    """A place inside the wall plate (a fixed region) and one near the hook tip (a load region), in mm."""
    import trimesh
    plate = trimesh.load(cfg.regions.fixed[0])
    tip = trimesh.load(cfg.regions.load[0].path)
    return plate.bounds.mean(axis=0), tip.bounds.mean(axis=0)


def test_the_meshing_failure_names_the_bodies_at_the_places_where_the_surface_crosses_itself():
    """P4, F2-6: the places come with the nearest bodies of the YAML (the region STLs), so the reader sees which bodies they are."""
    from topopt.config import load_config
    from topopt.verify.run import _crossing_bodies
    cfg = load_config(HOOK_YAML)
    a, b = _hook_places(cfg)
    text = _crossing_bodies(cfg, {"prepared": {"count": 2, "points": [a.tolist(), b.tolist()]}, "input": {"count": 0, "points": []}})
    assert "where the prepared surface crosses itself" in text and "fixed wall_plate.stl inside" in text
    assert "load tip (hook_tip.stl) " in text and "mm away" in text         # the centre of the hook tip box is not in the curved tip
    only_input = _crossing_bodies(cfg, {"prepared": {"count": 0, "points": []}, "input": {"count": 1, "points": [a.tolist()]}})
    assert "where the STL crosses itself" in only_input and "fixed wall_plate.stl inside" in only_input
    assert _crossing_bodies(cfg, None) == "" and _crossing_bodies(cfg, {"prepared": {"count": 0, "points": []}}) == ""
    assert _crossing_bodies(cfg, {"prepared": {"count": 1, "points": "bad"}}) == ""          # a help text never raises


def test_the_bodies_for_an_input_that_crosses_itself_are_the_ones_at_the_places_of_the_input():
    """P4 review, L5: the retries stopped because the STL crosses itself (near the plate). The prepared surface also has a crossing,
    at the hook tip: that place did not cause the stop, so the text must not name the bodies there."""
    from topopt.config import load_config
    from topopt.verify.run import _crossing_bodies
    cfg = load_config(HOOK_YAML)
    plate, tip = _hook_places(cfg)
    facts = {"prepared": {"count": 1, "points": [tip.tolist()], "attempts": [1]},
             "input": {"count": 1, "points": [plate.tolist()]}, "stopped": "input_crosses_itself"}
    text = _crossing_bodies(cfg, facts)
    assert "where the STL crosses itself" in text and "fixed wall_plate.stl inside" in text and "hook_tip.stl" not in text
    facts["stopped"] = "no_progress"                                   # any other stop: the places of the prepared surface
    other = _crossing_bodies(cfg, facts)
    assert "where the prepared surface crosses itself" in other and "hook_tip.stl" in other


@needs_recording
def test_a_meshing_failure_with_crossings_is_a_verify_error_with_the_places_and_the_bodies(run_dir, monkeypatch):
    from topopt import tet
    from topopt.config import load_config
    from topopt.verify import run as vrun
    from topopt.verify.errors import VerifyError
    a, _ = _hook_places(load_config(HOOK_YAML))
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras())

    def fail(*args, **kw):
        raise tet.MeshingFailed("gmsh stopped on final.stl at size 3 mm: the prepared surface crosses itself at 1 place(s)",
                                crossings={"prepared": {"count": 1, "points": [a.tolist()], "attempts": [1]}, "stopped": "no_progress"})
    monkeypatch.setattr(tet, "mesh_stl", fail)
    with pytest.raises(VerifyError) as e:
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, mesh_size=3.0, force=True)
    text = str(e.value)
    assert text.startswith("meshing the final STL at 3 mm failed: gmsh stopped") and "fixed wall_plate.stl inside" in text


@needs_recording
def test_the_sentence_about_the_retry_of_the_surface_reaches_the_verify_error_the_cli_and_the_ratchet_row(run_dir, monkeypatch, capsys):
    """F1-5 review, MJ1: when gmsh fails on a surface that the clean retry made, the error of `mesh_stl` has a sentence that names the
    retry and the first refusal. The sentence must stay in the error of `verify_run`, in the text of `topopt verify` (exit code 2) and
    in `errors.verify` of the ratchet row. `mesh_stl` is the real one; the surface step and gmsh are faked (a sphere with the step)."""
    import trimesh
    from test_tet import _ok_probe
    from topopt import ratchet as rt
    from topopt.tet import mesher, surface as surf
    from topopt.verify import run as vrun
    from topopt.verify.errors import VerifyError
    _ok_probe(monkeypatch)
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras())
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    first = "prepared surface has an edge of 0.0033 mm (limit 0.05 mm)"
    step = dict(step="clean_short_edges", limit_mm=0.05, short_edges_before=9, refused_first=first)
    monkeypatch.setattr(surf, "prepare_surface", lambda path, h, **kw: surf.PreparedSurface(
        np.array(sphere.vertices), np.array(sphere.faces), {}, {"deviation_from_input_mm": {}}, [dict(step)]))

    def fail(V, F, size, **kw):
        raise mesher.MeshingFailed("algorithm 10: HXT 3D mesh failed", [(0.0, 0.0, 10.0)])
    monkeypatch.setattr(mesher, "mesh_surface_isolated", fail)
    sentence = f"The surface that gmsh got in attempt 2 is the prepared surface that was refused ({first}), with its 9 edges under 0.05 mm cleaned (clean retry)"
    with pytest.raises(VerifyError) as e:
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, mesh_size=3.0, force=True)
    assert str(e.value).startswith("meshing the final STL at 3 mm failed: gmsh stopped on final.stl") and sentence in str(e.value)
    assert cli(run_dir, "--yaml", HOOK_YAML, "--mesh-size", 3, "--force") == 2
    assert sentence in capsys.readouterr().err
    ctx = rt.JudgeContext(yaml_path=HOOK_YAML, needs=frozenset({"verify"}), mesh_size_mm=3.0)
    facts = rt.judge_facts(run_dir, None, ctx)
    assert facts["errors"]["verify"]["exit_code"] == 2 and sentence in facts["errors"]["verify"]["message"]


@needs_recording
def test_missing_scikit_fem_gives_exit_code_2_and_names_the_extra(run_dir, monkeypatch, capsys):
    from topopt.verify import run as vrun
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras(skfem="missing"))
    assert cli(run_dir, "--yaml", HOOK_YAML) == 2
    err = capsys.readouterr().err
    assert "scikit-fem" in err and "verify" in err and "uv pip install" in err


def _patch_mesher(monkeypatch, exc):
    """A mesher that fails with `exc`, and the extras and factors that the check needs before it meshes."""
    import topopt.tet
    import topopt.verify.factor as factor
    from topopt.verify import run as vrun
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras())
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("superlu"))

    def fail(*args, **kwargs):
        raise exc
    monkeypatch.setattr(topopt.tet, "mesh_stl", fail)


@needs_recording
def test_a_meshing_failure_names_the_size_that_failed_and_the_mesh_size_option_of_the_caller(run_dir, monkeypatch, capsys):
    """Q1-01: the refusal of the prepared surface says which size failed and that another size can work (no retry is done)."""
    from topopt.verify import names
    from topopt.verify import run as vrun
    from topopt.verify.errors import VerifyError
    from topopt.tet.surface import PreparedSurfaceRejected
    refusal = ("prepared surface has an edge of 0.0458 mm (limit 0.05 mm); the weld retry was tried and did not help: "
               "the welded surface is not a closed manifold")
    _patch_mesher(monkeypatch, PreparedSurfaceRejected(refusal))
    with pytest.raises(VerifyError) as e:
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, mesh_size=1.59788, estimate_only=False)
    text = str(e.value)
    assert text.startswith("meshing the final STL at 1.59788 mm failed: " + refusal)
    assert "another size can work" in text and "--mesh-size to a value other than 1.59788 mm" in text
    assert "--levels" not in text                                                       # Q1 R1 n1: the call gave one size, not levels
    assert cli(run_dir, "--yaml", HOOK_YAML, "--mesh-size", "1.59788") == 2          # the CLI prints the same text, exit code 2
    err = capsys.readouterr().err
    assert "--mesh-size to a value other than 1.59788 mm" in err and "--levels" not in err
    with names.use_style("mcp"), pytest.raises(VerifyError) as m:                      # the MCP tool names its own parameters
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, mesh_size=1.59788, estimate_only=False)
    assert "mesh_size_mm to a value other than 1.59788 mm" in str(m.value) and "levels_mm" not in str(m.value) and "--" not in str(m.value)
    with pytest.raises(VerifyError) as lv:                                              # Q1 R1 n1: a call with levels is told to change them
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, levels=[1.59788], estimate_only=False)
    assert "another size can work" in str(lv.value) and "change that size in --levels" in str(lv.value)
    assert "--mesh-size" not in str(lv.value)                                           # not both: the call has levels
    with names.use_style("mcp"), pytest.raises(VerifyError) as lm:
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, levels=[1.59788], estimate_only=False)
    assert "change that size in levels_mm" in str(lm.value) and "mesh_size_mm" not in str(lm.value)
    with pytest.raises(VerifyError) as dflt:                                            # no size given: the default size failed
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, estimate_only=False)
    assert "--mesh-size to a value other than" in str(dflt.value) and "--levels" not in str(dflt.value)


@needs_recording
def test_a_meshing_failure_that_no_mesh_size_changes_has_no_mesh_size_hint(run_dir, monkeypatch):
    """The input STL itself is refused, or gmsh is missing: another mesh size does not help, so the text does not say so."""
    from topopt.tet import TetUnavailable
    from topopt.tet.surface import SurfaceRejected
    from topopt.verify import run as vrun
    from topopt.verify.errors import VerifyError
    for exc in (SurfaceRejected("input surface is not a closed oriented manifold: open edges 3"), TetUnavailable("no gmsh")):
        _patch_mesher(monkeypatch, exc)
        with pytest.raises(VerifyError) as e:
            vrun.verify_run(run_dir, yaml_path=HOOK_YAML, estimate_only=False)
        assert str(e.value).startswith("meshing the final STL at ") and "failed: " + str(exc) in str(e.value)
        assert "mesh-size" not in str(e.value) and "another size" not in str(e.value)


def _verify_parser(cli):
    import argparse
    sub = next(a for a in cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices["verify"]


def test_the_two_missing_extra_messages_give_the_install_command_for_the_running_python():
    """Q1-12: both texts (gmsh, scikit-fem) name an extra and give a command with `--python <sys.executable>`; no bare `-e '.[verify]'`."""
    import shlex
    from topopt.verify import run as vrun
    from topopt.verify.errors import VerifyUnavailable
    py = f"--python {shlex.quote(sys.executable)}"
    with pytest.raises(VerifyUnavailable) as gm:
        vrun.require_extras(fake_extras(gmsh="missing"))
    gm_text = str(gm.value)
    assert "needs gmsh (the 'tet' extra)" in gm_text and f"uv pip install {py} -e " in gm_text and "[tet]" in gm_text
    assert "the 'verify' extra installs all three" in gm_text and "[verify]" in gm_text      # the next step after gmsh is scikit-fem
    with pytest.raises(VerifyUnavailable) as sk:
        vrun.require_extras(fake_extras(skfem="missing"))
    sk_text = str(sk.value)
    assert "needs scikit-fem" in sk_text and "'verify' extra" in sk_text
    assert f"uv pip install {py} -e " in sk_text and "[verify]" in sk_text and "-e '.[verify]'" not in sk_text
    with pytest.raises(VerifyUnavailable) as lib:                          # a gmsh that is installed and does not load: the fix is a system library
        vrun.require_extras(fake_extras(gmsh="load_failed"))
    assert "libGLU.so.1" in str(lib.value) and "the 'verify' extra installs all three" not in str(lib.value)


def test_the_p2_import_message_for_scikit_fem_gives_the_same_command(monkeypatch):
    import builtins
    import shlex
    from topopt.verify import p2
    from topopt.verify.errors import VerifyUnavailable
    real = builtins.__import__

    def no_skfem(name, *args, **kwargs):
        if name == "skfem":
            raise ImportError("No module named 'skfem'")
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_skfem)
    with pytest.raises(VerifyUnavailable) as e:
        p2.require_skfem()
    assert f"uv pip install --python {shlex.quote(sys.executable)} -e " in str(e.value) and "[verify]" in str(e.value)
    assert "-e '.[verify]'" not in str(e.value)


def _verify_extra_pins() -> list[str]:
    """The pins of the `verify` extra in `pyproject.toml`, as `name==version` (the platform marker cut off)."""
    import tomllib
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    return [req.split(";")[0].strip() for req in cfg["project"]["optional-dependencies"]["verify"]]


def _not_a_source_tree(monkeypatch, tmp_path, *, gmsh_wheel=True):
    """Put `topopt.verify.factor` in a folder with no `pyproject.toml` two levels up (a wheel or a non-editable install)."""
    from topopt.tet import gmsh_probe
    from topopt.verify import factor
    monkeypatch.setattr(factor, "__file__", str(tmp_path / "site-packages" / "topopt" / "verify" / "factor.py"))
    monkeypatch.setattr(gmsh_probe, "has_wheel_platform", lambda: gmsh_wheel)


def test_the_install_command_names_the_whole_verify_extra_in_a_package_that_is_not_a_source_tree(monkeypatch, tmp_path):
    """Q1 R1 m1: with no `pyproject.toml` beside the package (a wheel or a non-editable install) the command installs the three pins of
    the `verify` extra, not cholespy only. The pins are the ones in `pyproject.toml`, so a bump in one place and not the other fails."""
    import shlex
    from topopt.verify import factor
    pins = _verify_extra_pins()
    assert {p.split("==")[0] for p in pins} == {"gmsh", "scikit-fem", "cholespy"}
    _not_a_source_tree(monkeypatch, tmp_path)
    tokens = shlex.split(factor.verify_install_command())
    assert tokens[:5] == ["uv", "pip", "install", "--python", sys.executable]
    assert sorted(tokens[5:]) == sorted(pins)                                         # all three, nothing else, and no `-e`
    _not_a_source_tree(monkeypatch, tmp_path, gmsh_wheel=False)                       # no gmsh wheel here (the marker of the extra)
    assert sorted(shlex.split(factor.verify_install_command())[5:]) == sorted(p for p in pins if not p.startswith("gmsh"))


def test_every_message_with_the_verify_install_command_gives_the_whole_extra_in_a_package_that_is_not_a_source_tree(monkeypatch,
                                                                                                                       tmp_path):
    """Q1 R1 m1: the gmsh text says "the 'verify' extra installs all three"; the command in each of the four texts must do that."""
    import builtins
    from topopt.verify import factor, p2
    from topopt.verify import run as vrun
    from topopt.verify.errors import VerifyUnavailable
    _not_a_source_tree(monkeypatch, tmp_path)
    cmd = factor.verify_install_command()
    texts = {"cholespy": factor.cholespy_note(("missing", "No module named 'cholespy'"))}
    with pytest.raises(VerifyUnavailable) as gm:
        vrun.require_extras(fake_extras(gmsh="missing"))
    texts["gmsh"] = str(gm.value)
    with pytest.raises(VerifyUnavailable) as sk:
        vrun.require_extras(fake_extras(skfem="missing"))
    texts["scikit-fem"] = str(sk.value)
    real = builtins.__import__

    def no_skfem(name, *args, **kwargs):
        if name == "skfem":
            raise ImportError("No module named 'skfem'")
        return real(name, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", no_skfem)
        with pytest.raises(VerifyUnavailable) as p2e:
            p2.require_skfem()
    texts["p2"] = str(p2e.value)
    for name, text in texts.items():
        assert cmd in text, name
        for pin in _verify_extra_pins():
            assert pin in text, (name, pin)


def test_import_isolation_core_modules_import_none_of_the_extras():
    """`import topopt`, `topopt.cli`, `topopt_mcp.server` and the verify modules import no gmsh, skfem or cholespy."""
    import subprocess
    pytest.importorskip("mcp")
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("gmsh", "skfem", "cholespy"):
            raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Block())
import topopt, topopt.cli, topopt_mcp.server
import topopt.verify, topopt.verify.run, topopt.verify.p2, topopt.verify.factor, topopt.verify.stats
import topopt.verify.regions, topopt.verify.bc, topopt.verify.memory, topopt.verify.report, topopt.verify.errors
import topopt.verify.names, topopt.verify.status, topopt.verify.buckling_p2, topopt.verify.motion
import topopt.verify.fatigue, topopt.verify.bore, topopt.verify.pressfit
bad = [m for m in sys.modules if m.split(".")[0] in ("gmsh", "skfem", "cholespy")]
print("CLEAN" if not bad else "LOADED " + str(bad))
# the code paths that need an extra say so, with the install step
from topopt.verify.errors import VerifyUnavailable
from topopt.verify.p2 import P2Model
try:
    P2Model([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[0], [1], [2], [3]])
except VerifyUnavailable as e:
    print("P2 " + str(e)[:60])
from topopt.config import UserError
from topopt.verify.errors import SolverUnavailable
from topopt.verify.factor import CholespyFactor
import scipy.sparse as sp
try:
    CholespyFactor(sp.identity(3, format="csr"))
except SolverUnavailable as e:
    print("CHOLESPY " + str(e)[:60] + " | " + str(isinstance(e, UserError) and not isinstance(e, VerifyUnavailable)))
'''
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": str(REPO), "PATH": "", "HOME": "/nonexistent"})
    lines = out.stdout.splitlines()
    assert lines and lines[0] == "CLEAN", out.stdout + out.stderr
    assert any(x.startswith("P2 topopt verify needs the package scikit-fem") for x in lines)
    # a cholespy that is gone at factor time is `SolverUnavailable` (a UserError, exit code 1), as in `resolve_factor`
    assert any(x.startswith("CHOLESPY solver 'cholespy' needs the package cholespy") and x.endswith("| True") for x in lines)


# --------------------------------------------------------------------------- #
# The hook, end to end (gmsh)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def hook_result(tmp_path_factory):
    """One real run at the default size, through the command. Return (run folder, exit code, stdout, stderr)."""
    pytest.importorskip("skfem")
    if not (RECORDING / "result" / "final.stl").is_file():
        pytest.skip("needs the recorded hook run recordings/hook_*")
    from helpers import tet_probe
    probe = tet_probe()
    if not probe.ok:
        pytest.skip(f"needs gmsh (the 'tet' extra): {probe.state}: {probe.detail}")
    import contextlib
    import io
    from topopt.cli import main
    run = copy_run(tmp_path_factory.mktemp("hook") / "run")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(run), "--yaml", str(HOOK_YAML)])
    return run, code, out.getvalue(), err.getvalue()


@needs_gmsh
@needs_skfem
def test_hook_end_to_end_reports_both_cases(hook_result):
    run, code, out, err = hook_result
    assert code == 0, err
    res = json.loads((run / "verify" / "verify.json").read_text())
    assert res["format"] == 1 and res["exit_code"] == 0 and res["ok"]
    assert set(res["cases"]) == {"hang", "pull_off"}
    assert res["no_allowable"]["resource"] == MATERIAL_URI and "limit" not in res           # no allowable was given
    assert res["run"]["name"] == "tshirt_hook" and res["band_mm"] == pytest.approx(0.6)
    assert res["backend"]["solver"] in ("cholesky", "pardiso", "cholespy", "superlu")
    # the numbers of each case (ranges, not equality: the mesh is not repeatable across machines)
    for name, c in res["cases"].items():
        v = c["von_mises"]
        assert 0 < v["p50_by_volume_MPa"] < v["p95_by_volume_MPa"] <= v["p99_by_volume_MPa"] <= v["p99_9_by_volume_MPa"]
        assert v["p99_9_by_volume_MPa"] <= v["peak_qp_MPa"] <= v["peak_vertex_MPa"] * (1 + 1e-9)
        assert len(v["top_elements"]) == 10 and all("tet_quality" in t for t in v["top_elements"])
        assert "p99_9_by_volume_MPa_excluding_bad_tets" in v and "peak_qp_MPa_excluding_bad_tets" in v
        assert c["equilibrium"]["force_relative_residual"] <= 1e-6
        assert c["equilibrium"]["moment_relative_residual"] <= 1e-6
        assert c["linear_solve"]["relative_residual"] <= 1e-8
        assert c["weight_used"] is False and c["weight_of_the_optimizer"] in (234.0, 1.0)
        for f in c["forces"]:
            assert f["resultant_relative_error"] <= 1e-10                 # the resultant equals the YAML force
            assert f["resultant_N"] == pytest.approx(f["force_N"], rel=1e-9, abs=1e-12)
    # the larger force gives the larger stress and the smaller stiffness
    assert res["cases"]["pull_off"]["von_mises"]["p99_9_by_volume_MPa"] > res["cases"]["hang"]["von_mises"]["p99_9_by_volume_MPa"]
    assert res["cases"]["hang"]["stiffness_N_per_mm"] > 100
    assert 0.2 < res["cases"]["hang"]["stiffness_over_optimizer"] < 2.0
    # regions: one fixed region and one load region, with counts
    fixed, load = res["regions"]["fixed"][0], res["regions"]["load"][0]
    assert fixed["region"] == "wall_plate" and fixed["nodes"] > 1000 and fixed["counts_by_band"]["0"]["nodes"] > 0
    assert load["region"] == "tip" and load["facets"] > 100 and load["area_mm2"] > 100
    assert fixed["counts_by_band"]["0"]["nodes"] < fixed["counts_by_band"]["0.5x"]["nodes"] <= fixed["counts_by_band"]["1x"]["nodes"]
    assert res["regions"]["keep_out_used"] is False
    # the mesh
    m = res["mesh"]
    assert m["dofs"] == 3 * m["p2_nodes"] and m["tets"] > 5000 and m["below_0p05"] == 0
    assert 0.95 < m["p2_volume_over_stl"] < 1.01
    assert m["size"] == pytest.approx(1.5, abs=0.15)                       # the default size of the mesher on the hook
    # memory and time
    mem = res["memory"]
    assert mem["peak_rss_bytes"] > 1e8 and mem["estimate_after_meshing_bytes"] > 1e8
    assert res["estimates_before_meshing"][0]["estimated_dofs"] == pytest.approx(m["dofs"], rel=0.2)
    assert res["timing_s"]["verify_total_s"] > 0 and "limit" not in res and "levels" not in res


@needs_gmsh
@needs_skfem
def test_hook_end_to_end_compliance_is_f_dot_u(hook_result):
    """`compliance_Nmm` is f.u, as the optimizer (review m1): compliance x stiffness = |F|^2."""
    run, code, out, err = hook_result
    res = json.loads((run / "verify" / "verify.json").read_text())
    forces = {"hang": [0.0, 0.0, -1.96], "pull_off": [20.89, 5.21, -20.89]}
    for name, c in res["cases"].items():
        F2 = float(np.dot(forces[name], forces[name]))
        assert c["compliance_Nmm"] * c["stiffness_N_per_mm"] == pytest.approx(F2, rel=1e-9)


@needs_gmsh
@needs_skfem
def test_hook_end_to_end_facts_for_a_reader_who_asks_what_the_p99_9_stands_for(hook_result):
    """Review M2, m4, m7: the load centre, the stress without the region patches, the load band, the normal check."""
    run, code, out, err = hook_result
    res = json.loads((run / "verify" / "verify.json").read_text())
    assert res["yaml_vs_run"]["matches"] and res["yaml_vs_run"]["differences"] == []
    assert res["near_region_mm"] == 2.0 and res["load_sensitivity"] is True
    for name, c in res["cases"].items():
        v = c["von_mises"]
        assert v["peak_qp_over_p99_9"] == pytest.approx(v["peak_qp_MPa"] / v["p99_9_by_volume_MPa"])
        nr = v["near_regions"]
        assert nr["distance_mm"] == 2.0 and set(nr["regions"]) == {"wall_plate", "tip"}
        assert 0.0 < nr["volume_fraction_excluded"] < 1.0
        assert nr["p99_9_by_volume_MPa_excluding_near_regions"] > 0
        sh = nr["top_volume_share_near"]
        assert 0.0 <= sh["wall_plate"] <= 1.0 and 0.0 <= sh["tip"] <= 1.0 and sh["any"] + sh["none"] == pytest.approx(1.0)
        lc = c["forces"][0]["load_centre"]
        assert lc["optimizer_center_mm"] == pytest.approx([67.7493, -0.1, 14.1761])     # from run.json
        assert 1.0 < lc["distance_mm"] < 6.0                                            # the review measured 3.13 mm
        assert 0.8 < lc["moment_ratio"] < 1.3
    # most of the top 0.1 % volume of the pull-off case sits at the tip patch (the review measured 72 to 82 %)
    assert res["cases"]["pull_off"]["von_mises"]["near_regions"]["top_volume_share_near"]["tip"] > 0.5
    # the load-band sensitivity: band 0 and 2 x the band, the factor reused
    bands = {b["label"]: b for b in res["load_band_sensitivity"]["bands"]}
    assert set(bands) == {"0", "2x"} and bands["0"]["band_mm"] == 0.0 and bands["2x"]["band_mm"] == pytest.approx(1.2)
    base_facets = res["regions"]["load"][0]["facets"]
    assert bands["0"]["load_regions"]["tip"]["facets"] < base_facets < bands["2x"]["load_regions"]["tip"]["facets"]
    for b in bands.values():
        c = b["cases"]["pull_off"]
        assert c["p99_9_by_volume_MPa"] > 0 and c["stiffness_N_per_mm"] > 0
        assert abs(c["p99_9_change_vs_base"]) < 0.5                  # the review measured about -7 and -10 %
    # the normal check of the load region
    nc = res["regions"]["load"][0]["normal_check"]
    assert 0.0 < nc["coherence"] <= 1.0 and len(nc["mean_normal"]) == 3
    # memory note: the child number is not a measurement of gmsh
    assert "not a measurement of gmsh" in res["memory"]["note"]
    assert "sensitivity_s" in res["timing_s"]
    assert "load band 0 mm" in out and "without points within 2 mm" in out


@needs_gmsh
@needs_skfem
def test_hook_end_to_end_files(hook_result):
    from topopt.tet import TetMesh
    run, code, out, err = hook_result
    v = run / "verify"
    assert sorted(p.name for p in v.iterdir()) == ["fields.npz", "mesh.npz", "status.json", "verify.json"]
    res = json.loads((v / "verify.json").read_text())
    mesh = TetMesh.load(v / "mesh.npz")
    assert mesh.sha256() == mesh.facts["sha256"] == res["mesh"]["sha256"]               # the file is the mesh that was solved
    with np.load(v / "fields.npz") as z:
        assert {"p", "t", "node_xyz", "vm_elem_max_hang", "vm_elem_max_pull_off"} <= set(z.files)
        assert z["p"].shape == (3, len(mesh.nodes)) and z["t"].shape == (4, len(mesh.tets))
        assert z["vm_elem_max_pull_off"].shape == (len(mesh.tets),) and z["vm_elem_max_pull_off"].dtype == np.float32
        assert float(z["vm_elem_max_pull_off"].max()) == pytest.approx(res["cases"]["pull_off"]["von_mises"]["peak_qp_MPa"], rel=1e-5)
    assert res["files"] == {"verify": "verify.json", "mesh": "mesh.npz", "fields": "fields.npz"}   # names inside the verify folder
    status = json.loads((v / "status.json").read_text())
    assert status["state"] == "done" and status["exit_code"] == 0 and status["estimated_seconds_total"] > 0


@needs_gmsh
@needs_skfem
def test_hook_end_to_end_prints_the_summary(hook_result):
    run, code, out, err = hook_result
    assert "verify: tshirt_hook" in out and "case hang:" in out and "case pull_off:" in out
    assert "p99.9" in out and "equilibrium residual" in out and "wrote" in out
    assert "EXCEEDED" not in out
    assert "memory estimate" in err                                   # the estimate is printed before and after meshing
    assert "limit: no allowable stress was given (--allowable-mpa)" in out and MATERIAL_URI in out


@needs_gmsh
@needs_skfem
def test_hook_exit_code_3_only_with_a_tiny_allowable(hook_result, run_dir, capsys):
    code = cli(run_dir, "--yaml", HOOK_YAML, "--allowable-mpa", "0.001")
    out = capsys.readouterr().out
    assert code == 3 and "EXCEEDED" in out
    res = json.loads((run_dir / "verify" / "verify.json").read_text())
    assert res["limit"]["exceeded"] and res["limit"]["allowable_MPa"] == 0.001 and res["limit"]["percentile"] == 99.9
    assert res["exit_code"] == 3 and set(res["limit"]["cases"]) == {"hang", "pull_off"}
    assert "no_allowable" not in res and "no allowable stress" not in out            # an allowable was given
    assert res["limit"]["cases"]["pull_off"]["margin"] < 1


@needs_gmsh
@needs_skfem
@pytest.mark.slow
def test_hook_levels_block_and_files_per_level(run_dir, capsys):
    # files of an earlier single-level run, of another final.stl, and a file of the user: review M3
    v = run_dir / "verify"
    v.mkdir()
    (v / "mesh.npz").write_bytes(b"stale")
    (v / "fields.npz").write_bytes(b"stale")
    (v / "verify.json").write_text(json.dumps({"run": {"final_stl_sha256": "0" * 64}}))
    (v / "notes.txt").write_text("keep me")
    assert cli(run_dir, "--yaml", HOOK_YAML, "--levels", "2.0,1.5", "--allowable-mpa", "1000", "--no-sensitivity") == 0
    res = json.loads((run_dir / "verify" / "verify.json").read_text())
    assert not (v / "mesh.npz").exists() and not (v / "fields.npz").exists()           # the stale files are gone
    assert (v / "notes.txt").read_text() == "keep me"
    assert res["previous_verify"]["same_final_stl"] is False and "mesh.npz" in res["previous_verify"]["removed"]
    assert "load_band_sensitivity" not in res and res["load_sensitivity"] is False
    # `files` points to the files of the levels, and the reference mesh is the one of verify.json
    assert res["files"]["mesh"] == "h1.5/mesh.npz" and res["files"]["fields"] == "h1.5/fields.npz"
    assert set(res["files"]["levels"]) == {"2", "1.5"} and res["files"]["levels"]["2"]["mesh"] == "h2/mesh.npz"
    from topopt.tet import TetMesh
    assert TetMesh.load(v / res["files"]["mesh"]).sha256() == res["mesh"]["sha256"]
    assert TetMesh.load(v / res["files"]["levels"]["2"]["mesh"]).sha256() != res["mesh"]["sha256"]
    assert res["levels"]["sizes_mm"] == [2.0, 1.5] and res["reference_level_mm"] == 1.5
    ch = res["levels"]["relative_change_first_to_last"]
    assert ch["from_mm"] == 2.0 and ch["to_mm"] == 1.5 and set(ch["cases"]["hang"]) == {
        "p99_9_by_volume_MPa", "peak_qp_MPa", "peak_vertex_MPa"}
    assert abs(ch["cases"]["pull_off"]["p99_9_by_volume_MPa"]) < 0.2        # p99.9 moves little between 2.0 and 1.5 mm
    assert res["limit"]["mesh_size_mm"] == 1.5 and not res["limit"]["exceeded"]
    for sub in ("h2", "h1.5"):
        assert (run_dir / "verify" / sub / "mesh.npz").is_file() and (run_dir / "verify" / sub / "fields.npz").is_file()


def test_the_summary_says_n_a_for_a_load_band_value_that_is_missing():
    """A case that applies no load (U21) has p99.9 0, no stiffness and no change against the base: the values are None."""
    from topopt.verify.report import summary_lines
    cases = {"zero": dict(p99_9_by_volume_MPa=0.0, p99_9_change_vs_base=None, stiffness_N_per_mm=None),
             "down": dict(p99_9_by_volume_MPa=1.5, p99_9_change_vs_base=0.25, stiffness_N_per_mm=335.7)}
    lines = summary_lines(minimal_result(load_band_sensitivity=dict(bands=[dict(band_mm=0.5, cases=cases)])))
    assert [x for x in lines if x.startswith("load band")] == [
        "load band 0.5 mm: zero p99.9 0 MPa (n/a), stiffness n/a; down p99.9 1.5 MPa (+25 %), stiffness 336 N/mm"]


@needs_gmsh
@needs_skfem
def test_verify_of_a_run_with_a_case_that_applies_no_load_ends_with_exit_code_0_and_n_a_in_the_summary(tmp_path, capsys):
    """The run of a zero case and a normal case, then `topopt verify` (exit code 2 with a TypeError before this fix)."""
    from test_repeated_load_cases import write_case
    zero = {"name": "zero", "forces": [{"region": "tip", "force": [0, 0, 0]}]}
    down = {"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}
    yaml_path = write_case(tmp_path / "case", [zero, down], {"voxel_size": 1.0, "iterations": 15})
    run = tmp_path / "run"
    from topopt.cli import main
    assert main(["run", str(yaml_path), "--out", str(run), "--quiet"]) == 0
    capsys.readouterr()
    assert cli(run, "--yaml", yaml_path) == 0
    out, err = capsys.readouterr()
    assert "internal error" not in err and "TypeError" not in err
    band = [x for x in out.splitlines() if x.startswith("load band")]
    assert band and all("zero p99.9 0 MPa (n/a), stiffness n/a; down p99.9 " in x for x in band)
    res = json.loads((run / "verify" / "verify.json").read_text())
    zero_row = res["load_band_sensitivity"]["bands"][0]["cases"]["zero"]
    assert zero_row["p99_9_change_vs_base"] is None and zero_row["stiffness_N_per_mm"] is None
    assert res["exit_code"] == 0 and set(res["cases"]) == {"zero", "down"}


def _level(size_mm, zero_vm, down_vm):
    """A synthetic level for `levels_block`: a case that applies no load (all stresses 0) and a normal case."""
    keys = ("p99_9_by_volume_MPa", "peak_qp_MPa", "peak_vertex_MPa")
    return dict(size_mm=size_mm, mesh=dict(dofs=100, tets=40),
                cases={"zero": dict(von_mises={k: zero_vm for k in keys}), "down": dict(von_mises=dict(zip(keys, down_vm)))})


def test_levels_block_gives_no_change_for_a_case_that_applies_no_load_and_the_summary_says_n_a():
    """A case that applies no load (U21) has 0 MPa at every level: the relative change is None, not a ZeroDivisionError.
    Found during the U21 fix round (`topopt verify --levels` ended with exit 2)."""
    from topopt.verify.report import summary_lines
    from topopt.verify.run import levels_block
    block = levels_block([_level(1.5, 0.0, (1.0, 2.0, 4.0)), _level(1.0, 0.0, (1.1, 2.5, 3.0))])
    zero, down = block["relative_change_first_to_last"]["cases"]["zero"], block["relative_change_first_to_last"]["cases"]["down"]
    assert zero == {"p99_9_by_volume_MPa": None, "peak_qp_MPa": None, "peak_vertex_MPa": None}
    assert down == pytest.approx({"p99_9_by_volume_MPa": 0.1, "peak_qp_MPa": 0.25, "peak_vertex_MPa": -0.25})
    assert block["relative_change"][0]["cases"]["zero"]["p99_9_by_volume_MPa"] is None
    lines = summary_lines(minimal_result(levels=block))
    assert "  zero: p99.9 n/a, peak qp n/a, peak vertex n/a" in lines
    assert "  down: p99.9 +10.0 %, peak qp +25.0 %, peak vertex -25.0 %" in lines


@needs_gmsh
@needs_skfem
def test_verify_levels_of_a_run_with_a_case_that_applies_no_load_ends_with_exit_code_0(tmp_path, capsys):
    """The run of a zero case and a normal case, then `topopt verify --levels` (exit code 2 with a ZeroDivisionError before)."""
    from test_repeated_load_cases import write_case
    zero = {"name": "zero", "forces": [{"region": "tip", "force": [0, 0, 0]}]}
    down = {"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}
    yaml_path = write_case(tmp_path / "case", [zero, down], {"voxel_size": 1.0, "iterations": 15})
    run = tmp_path / "run"
    from topopt.cli import main
    assert main(["run", str(yaml_path), "--out", str(run), "--quiet"]) == 0
    capsys.readouterr()
    assert cli(run, "--yaml", yaml_path, "--levels", "2.0,1.5") == 0
    out, err = capsys.readouterr()
    assert "internal error" not in err and "ZeroDivisionError" not in err
    assert "  zero: p99.9 n/a, peak qp n/a, peak vertex n/a" in out.splitlines()
    res = json.loads((run / "verify" / "verify.json").read_text())
    assert res["levels"]["relative_change_first_to_last"]["cases"]["zero"]["p99_9_by_volume_MPa"] is None
    assert res["levels"]["relative_change_first_to_last"]["cases"]["down"]["p99_9_by_volume_MPa"] is not None


def test_limit_block_compares_the_percentile_with_the_allowable():
    from topopt.verify.run import limit_block
    level = {"size_mm": 1.5, "cases": {
        "a": {"von_mises": {"percentile_by_volume_MPa": 40.0, "peak_qp_MPa": 90.0}},
        "b": {"von_mises": {"percentile_by_volume_MPa": 60.0, "peak_qp_MPa": 70.0}}}}
    lim = limit_block(level, 50.0, 99.9)
    assert lim["exceeded"] and lim["exceeded_cases"] == ["b"]
    assert lim["cases"]["a"]["margin"] == pytest.approx(1.25) and not lim["cases"]["a"]["exceeded"]
    assert lim["cases"]["a"]["peak_qp_margin"] == pytest.approx(50 / 90)         # the peak is information: no exit code
    assert not limit_block(level, 100.0, 99.9)["exceeded"]


def test_levels_block_gives_relative_change():
    from topopt.verify.run import levels_block

    def lv(size, p, q, r):
        return {"size_mm": size, "mesh": {"dofs": 1, "tets": 1},
                "cases": {"a": {"von_mises": {"p99_9_by_volume_MPa": p, "peak_qp_MPa": q, "peak_vertex_MPa": r}}}}
    b = levels_block([lv(2.0, 100.0, 200.0, 300.0), lv(1.5, 102.0, 250.0, 330.0), lv(1.0, 101.0, 260.0, 360.0)])
    assert b["sizes_mm"] == [2.0, 1.5, 1.0]
    assert b["relative_change"][0]["cases"]["a"]["p99_9_by_volume_MPa"] == pytest.approx(0.02)
    assert b["relative_change"][0]["cases"]["a"]["peak_qp_MPa"] == pytest.approx(0.25)
    ft = b["relative_change_first_to_last"]
    assert (ft["from_mm"], ft["to_mm"]) == (2.0, 1.0)
    assert ft["cases"]["a"]["peak_vertex_MPa"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# The MCP tool and the packaging
# --------------------------------------------------------------------------- #

def test_mcp_tool_estimate_is_the_default_and_errors_have_a_shape(run_dir, tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    res = server.verify_run(run_dir=str(run_dir), yaml_path=str(HOOK_YAML))
    assert res["ok"] and res["estimate_only"] and res["exit_code"] == 0
    assert res["estimates_before_meshing"] and not (run_dir / "verify").exists()
    bad = server.verify_run(run_dir=str(run_dir), yaml_path=str(HOOK_YAML), solver="amg")
    assert bad["ok"] is False and bad["exit_code"] == 1 and "amg" in bad["error"].lower()
    gone = server.verify_run(run_dir=str(tmp_path / "none"))
    assert gone["ok"] is False and gone["exit_code"] == 1 and "not found" in gone["error"]


def test_mcp_tool_memory_refusal_carries_the_facts(run_dir, monkeypatch):
    pytest.importorskip("mcp")
    from topopt.verify import run as vrun
    from topopt_mcp import server
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras())
    res = server.verify_run(run_dir=str(run_dir), yaml_path=str(HOOK_YAML), estimate_only=False, max_gb=0.01)
    assert res["ok"] is False and res["exit_code"] == 2 and res["facts"]["when"] == "before meshing"
    assert "--force" in res["error"] or "force" in res["error"]


def test_verify_extra_holds_the_pins_and_the_docs_name_the_section():
    import tomllib
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    extras = cfg["project"]["optional-dependencies"]
    assert extras["tet"][0] in extras["verify"]                               # the same gmsh pin and marker
    assert "scikit-fem==12.0.2" in extras["verify"] and "cholespy==2.2.0" in extras["verify"]
    main_deps = " ".join(cfg["project"]["dependencies"]).lower()
    for name in ("gmsh", "scikit-fem", "cholespy"):
        assert name not in main_deps
    assert not any(n in (REPO / "requirements.lock").read_text().lower() for n in ("gmsh", "scikit-fem", "cholespy"))
    contracts = (REPO / "CONTRACTS.md").read_text()
    assert "## 16. Verify (added with B2)" in contracts
    readme = (REPO / "docs" / "reference" / "verify.md").read_text()
    assert "topopt verify" in readme and "CHOLMOD" in readme and "GPL-2.0-or-later" in readme


def test_the_cholespy_license_line_is_in_every_kept_place():
    """Review M1: the wheel bundles CHOLMOD Supernodal (GPL-2.0-or-later). The same words in every place."""
    line = "GPL-2.0-or-later in effect: the wheel bundles CHOLMOD Supernodal, SuiteSparse 5.4.0"
    for path in ("pyproject.toml", "docs/reference/verify.md", "CONTRACTS.md", "topopt/verify/factor.py"):
        assert line in (REPO / path).read_text(), path
    for path in ("pyproject.toml", "docs/reference/verify.md", "CONTRACTS.md", "topopt/verify/factor.py"):
        assert "cholespy (BSD-3-Clause" not in (REPO / path).read_text() and "(BSD-3-Clause, `verify` extra)" not in (REPO / path).read_text()
    readme = (REPO / "docs" / "reference" / "verify.md").read_text()
    assert "5 to 10 %" not in readme and "776 against 1091" in readme and "+20 %" in readme


# --------------------------------------------------------------------------- #
# Review round 1: the YAML against run.json, stale files, the option names, the time in the estimate
# --------------------------------------------------------------------------- #

def _edit_run_json(run_dir, edit):
    p = run_dir / "run.json"
    data = json.loads(p.read_text())
    edit(data)
    p.write_text(json.dumps(data))


@needs_recording
def test_yaml_matches_run_json_for_the_recorded_run(run_dir):
    from topopt.verify.run import compare_yaml_run, read_run
    out = compare_yaml_run(read_run(run_dir, yaml_path=HOOK_YAML))
    assert out["matches"] and out["differences"] == []


@needs_recording
def test_yaml_differences_are_facts_not_errors(run_dir, capsys):
    from topopt.verify.run import compare_yaml_run, preflight, read_run

    def edit(d):
        d["load_cases"][1]["forces"][0]["force"] = [30.0, 5.21, -20.89]      # another force than the YAML
        d["load_cases"][0]["weight"] = 99.0
        d["regions"]["load"] = {"tip": "other_tip.stl"}
        d["regions"]["fixed"] = ["other_wall.stl"]
    _edit_run_json(run_dir, edit)
    out = compare_yaml_run(read_run(run_dir, yaml_path=HOOK_YAML))
    text = " ".join(out["differences"])
    assert not out["matches"]
    assert "case pull_off, region tip: force" in text and "case hang: weight" in text
    assert "load regions" in text and "fixed region files" in text
    plan = preflight(run_dir, yaml_path=HOOK_YAML, estimate_only=True)         # no error: the report says it
    assert plan.base["yaml_vs_run"]["matches"] is False
    assert cli(run_dir, "--yaml", HOOK_YAML, "--estimate-only") == 0
    assert "the YAML differs from run.json" in capsys.readouterr().out

    def rename(d):
        d["load_cases"][0]["name"] = "renamed"
    _edit_run_json(run_dir, rename)
    assert "load case names" in " ".join(compare_yaml_run(read_run(run_dir, yaml_path=HOOK_YAML))["differences"])


@needs_recording
def test_yaml_check_skips_what_run_json_does_not_hold(run_dir):
    from topopt.verify.run import compare_yaml_run, read_run

    def edit(d):
        d.pop("load_cases")
        d.pop("regions")
        d.pop("name")
    _edit_run_json(run_dir, edit)
    assert compare_yaml_run(read_run(run_dir, yaml_path=HOOK_YAML))["matches"]


def test_clear_outputs_removes_the_files_of_a_run_and_keeps_the_others(tmp_path):
    from topopt.verify.status import clear_outputs
    v = tmp_path / "verify"
    (v / "h1.5").mkdir(parents=True)
    (v / "h0.8").mkdir()
    (v / "other").mkdir()
    for name in ("verify.json", "fields.npz", "mesh.npz", "status.json", "status.json.tmp", "notes.txt"):
        (v / name).write_text("x")
    (v / "h1.5" / "mesh.npz").write_text("x")
    (v / "other" / "keep.txt").write_text("x")
    removed = clear_outputs(v)
    assert set(removed) == {"verify.json", "fields.npz", "mesh.npz", "status.json", "status.json.tmp", "h1.5/", "h0.8/"}
    assert sorted(p.name for p in v.iterdir()) == ["notes.txt", "other"]
    assert clear_outputs(tmp_path / "none") == []


def test_status_file_is_written_whole_and_read_back(tmp_path):
    from topopt.verify.status import StatusWriter, read_status
    assert read_status(tmp_path) is None
    sw = StatusWriter(tmp_path / "verify", sizes_mm=[1.5])
    sw.update("meshing", level=1, message="a")
    st = read_status(tmp_path / "verify")
    assert st["state"] == "meshing" and st["level"] == 1 and st["sizes_mm"] == [1.5] and st["updated_at"] >= st["started_at"]
    sw.update("done", exit_code=0)
    assert read_status(tmp_path / "verify")["level"] == 1                      # the fields stay
    assert not (tmp_path / "verify" / "status.json.tmp").exists()
    with pytest.raises(ValueError):
        sw.update("flying")
    (tmp_path / "verify" / "status.json").write_text("{not json")
    assert read_status(tmp_path / "verify") is None


def test_a_new_optimizer_run_removes_the_verify_folder_of_the_old_run(tmp_path):
    """Review m5: `verify` is in RUN_DIRS, so a new run does not leave a result for another final.stl."""
    from topopt.snapshot import RUN_DIRS, RunWriter
    assert "verify" in RUN_DIRS
    (tmp_path / "verify").mkdir()
    (tmp_path / "verify" / "verify.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("keep")
    RunWriter(tmp_path, quiet=True).clean()
    assert not (tmp_path / "verify").exists() and (tmp_path / "notes.txt").read_text() == "keep"


def test_previous_verify_json_for_another_final_stl_is_reported(tmp_path):
    from topopt.verify.run import _previous_verify
    v = tmp_path / "verify"
    assert _previous_verify(v, "aaa") is None
    v.mkdir()
    (v / "verify.json").write_text(json.dumps({"run": {"final_stl_sha256": "bbb"}}))
    assert _previous_verify(v, "aaa")["same_final_stl"] is False
    assert _previous_verify(v, "bbb")["same_final_stl"] is True
    (v / "verify.json").write_text("not json")
    assert _previous_verify(v, "aaa") is None


@needs_recording
def test_options_in_error_texts_name_the_cli_flag_or_the_mcp_parameter(run_dir, monkeypatch):
    from topopt.verify import names
    from topopt.verify import run as vrun
    from topopt.verify.errors import MemoryRefused
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras())
    with pytest.raises(MemoryRefused) as cli_err:
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, max_gb=0.01)
    assert "--max-gb" in str(cli_err.value) and "--force" in str(cli_err.value) and "--mesh-size" in str(cli_err.value)
    with names.use_style("mcp"), pytest.raises(MemoryRefused) as mcp_err:
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, max_gb=0.01)
    text = str(mcp_err.value)
    assert "max_gb" in text and "force" in text and "mesh_size_mm" in text and "--" not in text
    from topopt.config import UserError
    with names.use_style("mcp"), pytest.raises(UserError, match="mesh_size_mm must be a number above 0"):
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, mesh_size=-1)
    with names.use_style("mcp"), pytest.raises(UserError, match="solver 'amg'|unknown solver"):
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, solver="amg")


@needs_recording
def test_estimate_has_the_seconds_for_each_backend(run_dir, capsys):
    from topopt.verify import memory
    from topopt.verify.run import preflight
    plan = preflight(run_dir, yaml_path=HOOK_YAML, estimate_only=True, levels=[1.5, 1.0])
    for e in plan.estimates:
        by = e["seconds_by_backend"]
        assert set(by) == {"cholesky", "pardiso", "cholespy", "superlu"} and by["cholesky"] is None
        assert by["pardiso"] == pytest.approx(memory.estimate_seconds(e["estimated_dofs"], "pardiso", 2, True))
        assert by["superlu"] == pytest.approx(memory.estimate_seconds(e["estimated_dofs"], "superlu", 2, True, e["compactness"]))
        assert "fit on" in e["seconds_note"]
        if plan.chosen:
            assert e["estimated_seconds"] == by[plan.chosen]
    fine, coarse = plan.estimates[1], plan.estimates[0]
    assert fine["seconds_by_backend"]["pardiso"] > coarse["seconds_by_backend"]["pardiso"]
    assert cli(run_dir, "--yaml", HOOK_YAML, "--estimate-only", "--levels", "1.5,1.0") == 0
    out = capsys.readouterr().out
    assert "time estimate:" in out and "pardiso about" in out and "no measurement for the others" in out
    plan2 = preflight(run_dir, yaml_path=HOOK_YAML, estimate_only=True, load_sensitivity=False)
    assert plan2.estimates[0]["seconds_by_backend"]["pardiso"] < memory.estimate_seconds(plan2.estimates[0]["estimated_dofs"], "pardiso", 2, True)


@needs_recording
def test_new_options_are_checked(run_dir, capsys):
    assert cli(run_dir, "--yaml", HOOK_YAML, "--near-mm", "abc") == 1
    assert "--near-mm must be a number" in capsys.readouterr().err
    assert cli(run_dir, "--yaml", HOOK_YAML, "--near-mm", "-2") == 1
    assert cli(run_dir, "--yaml", HOOK_YAML, "--estimate-only", "--near-mm", "3", "--no-sensitivity") == 0


# --------------------------------------------------------------------------- #
# SuperLU as the last fallback, and no direct factor at all (E3)
# --------------------------------------------------------------------------- #

@needs_recording
def test_no_direct_factor_is_exit_code_1_with_the_facts_and_the_options(run_dir, monkeypatch, capsys):
    import topopt.verify.factor as factor
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability())
    assert cli(run_dir, "--yaml", HOOK_YAML) == 1
    cap = capsys.readouterr()
    err = cap.err
    assert err.count("error:") == 1 and "internal error" not in err and "Traceback" not in err and cap.out == ""
    assert "no direct linear solver can run here" in err and "Options" in err
    for name in ("cholesky", "pardiso", "cholespy", "superlu"):
        assert f"{name}: " in err                                    # the note of every factor
    assert "uv pip install --python" in err and "[verify]" in err and "Check the install of scipy" in err
    assert not (run_dir / "verify").exists()                        # nothing was written


@needs_recording
def test_a_named_factor_that_cannot_run_is_exit_code_1_and_the_options_name_the_others(run_dir, monkeypatch, capsys):
    import topopt.verify.factor as factor
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("superlu"))
    assert cli(run_dir, "--yaml", HOOK_YAML, "--solver", "pardiso") == 1
    err = capsys.readouterr().err
    assert "solver 'pardiso' cannot run here" in err and "pypardiso is not installed" in err
    assert "--solver superlu" in err and "auto" in err and "Traceback" not in err
    assert cli(run_dir, "--yaml", HOOK_YAML, "--solver", "cholespy") == 1
    assert "cholespy is not installed" in capsys.readouterr().err and not (run_dir / "verify").exists()


@needs_recording
def test_estimate_only_with_no_factor_states_the_fact_and_exits_0(run_dir, monkeypatch, capsys):
    import topopt.verify.factor as factor
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability())
    assert cli(run_dir, "--yaml", HOOK_YAML, "--estimate-only") == 0
    out = capsys.readouterr().out
    assert "none can run" in out and "no direct linear solver can run here" in out and "memory estimate" in out
    from topopt.verify import verify_run
    res = verify_run(run_dir, yaml_path=HOOK_YAML, estimate_only=True)
    assert res["backend"]["solver"] is None and "no direct linear solver" in res["backend"]["problem"]
    assert res["estimates_before_meshing"][0]["estimated_seconds"] is None


@needs_recording
def test_auto_names_superlu_and_the_factors_it_skipped_as_facts(run_dir, monkeypatch, capsys):
    import topopt.verify.factor as factor
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("superlu"))
    assert cli(run_dir, "--yaml", HOOK_YAML, "--estimate-only", "--levels", "1.5,1.0") == 0
    out = capsys.readouterr().out
    assert "solver:         superlu  (requested auto)" in out
    for name in ("cholesky", "pardiso", "cholespy"):
        assert f"auto skipped {name}: " in out                      # no silent switch: each skipped factor and its reason
    assert "time estimate:" in out and "superlu about" in out and "(this run)" in out
    assert "SuperLU model" in out and "C = 0.016" in out
    from topopt.verify import memory
    from topopt.verify.run import preflight
    plan = preflight(run_dir, yaml_path=HOOK_YAML, estimate_only=True, levels=[1.5, 1.0])
    assert plan.chosen == "superlu" and list(plan.base["backend"]["skipped"]) == ["cholesky", "pardiso", "cholespy"]
    assert plan.base["backend"]["order"] == ["cholesky", "pardiso", "cholespy", "superlu"]
    for e in plan.estimates:
        assert e["estimate_bytes"] == memory.estimate_bytes(e["estimated_dofs"], "superlu", e["compactness"])
        assert e["estimated_seconds"] == e["seconds_by_backend"]["superlu"] > 0
        assert e["estimate_bytes"] == e["range_bytes"]["central"] == e["ranges_by_backend"]["superlu"]["central"]   # no margin
    plan = preflight(run_dir, yaml_path=HOOK_YAML, estimate_only=True, solver="superlu")
    assert plan.chosen == "superlu" and plan.base["backend"]["skipped"] == {}                 # asked for by name: nothing skipped


@needs_recording
def test_the_memory_refusal_names_the_solver_the_model_and_the_options_of_superlu(run_dir, monkeypatch):
    from topopt.verify import names
    from topopt.verify import run as vrun
    from topopt.verify.errors import MemoryRefused
    import topopt.verify.factor as factor
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("superlu"))
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras())
    with pytest.raises(MemoryRefused) as e:
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, max_gb=0.05)
    text = str(e.value)
    assert "solver superlu: SuperLU model" in text and e.value.facts["solver"] == "superlu"
    assert "x 1.25" not in text and "central value, no margin; range " in text and "from the measured spread" in text
    assert "--mesh-size" in text and "--max-gb" in text and "--force" in text
    assert "pypardiso" in text and "cholespy" in text and "--solver NAME" in text
    with names.use_style("mcp"), pytest.raises(MemoryRefused) as m:
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, max_gb=0.05)
    assert "--" not in str(m.value) and "solver NAME" in str(m.value)                  # the MCP style names the parameters
    with pytest.raises(MemoryRefused) as other:                                         # no SuperLU text for another factor
        monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("pardiso"))
        vrun.verify_run(run_dir, yaml_path=HOOK_YAML, max_gb=0.05)
    assert "SuperLU" not in str(other.value) and "7.5 GB per million dofs" in str(other.value)


@needs_recording
def test_superlu_estimate_is_a_central_value_with_a_range_and_the_guard_refuses_on_the_central_value_only(run_dir, monkeypatch, capsys):
    """The hook at 0.8 mm: central about 7 GB, range 0.69 to 1.67 times that. A limit between them: the run starts, with a note."""
    from topopt.verify import memory
    from topopt.verify import run as vrun
    from topopt.verify.errors import MemoryRefused
    import topopt.verify.factor as factor
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("superlu"))
    monkeypatch.setattr(vrun, "extras_facts", lambda: fake_extras())
    monkeypatch.setattr(memory, "available_ram_bytes", lambda: 1000 * 10 ** 9)           # the limit is `max_gb`, not this host's RAM
    plan = vrun.preflight(run_dir, yaml_path=HOOK_YAML, estimate_only=True, levels=[0.8], max_gb=1000.0)
    (e,) = plan.estimates
    rng = e["range_bytes"]
    assert rng["central"] == e["estimate_bytes"] and rng["low"] < rng["central"] < rng["high"]
    assert rng["low"] == pytest.approx(rng["central"] / 1.45, rel=1e-6) and rng["high"] == pytest.approx(rng["central"] / 0.60, rel=1e-6)
    assert e["ranges_by_backend"]["superlu"] == rng and e["notes"] == [] and not e["above_limit"] and not e["upper_end_above_limit"]
    gb = lambda x: x / 1e9                                                                     # noqa: E731
    mid_limit = (gb(rng["central"]) + gb(rng["high"])) / 2                                     # between central and upper end
    said = []
    plan = vrun.preflight(run_dir, yaml_path=HOOK_YAML, levels=[0.8], max_gb=mid_limit, say=said.append)   # starts: no MemoryRefused
    (e,) = plan.estimates
    assert not e["above_limit"] and e["upper_end_above_limit"] and len(e["notes"]) == 1
    assert "upper end" in e["notes"][0] and "run starts" in e["notes"][0]
    assert any(t.startswith("  note: the upper end of the SuperLU memory range") for t in said)
    assert cli(run_dir, "--yaml", HOOK_YAML, "--estimate-only", "--levels", "0.8", "--max-gb", f"{mid_limit:.2f}") == 0
    out = capsys.readouterr().out
    assert "central value, no margin; range " in out and "note: the upper end of the SuperLU memory range" in out
    assert "ABOVE THE LIMIT" not in out
    with pytest.raises(MemoryRefused) as refused:                                              # the central value is above: refused
        vrun.preflight(run_dir, yaml_path=HOOK_YAML, levels=[0.8], max_gb=gb(rng["central"]) * 0.9)
    assert "central value, no margin; range " in str(refused.value)


def test_the_refusal_text_has_the_32_bit_index_fact_for_a_large_modelled_fill_only():
    from topopt.verify.run import refusal_text
    lim = dict(limit_bytes=int(10e9), limit_source="available RAM")
    big = refusal_text(int(60e9), lim, 0.5, "before meshing", 1_500_000, "superlu", 0.016)     # modelled fill 37 GB
    assert "32-bit indices" in big and "not measured" in big and "central value, no margin; range " in big
    small = refusal_text(int(12e9), lim, 0.8, "before meshing", 600_000, "superlu", 0.016)    # modelled fill 6 GB
    assert "32-bit" not in small and "SuperLU model" in small
    other = refusal_text(int(12e9), lim, 0.8, "before meshing", 1_500_000, "pardiso", 0.016)
    assert "32-bit" not in other and "x 1.25" in other and "range" not in other


@needs_recording
def test_the_line_with_superlu_instead_is_printed_only_when_superlu_can_run(run_dir, monkeypatch, capsys):
    import topopt.verify.factor as factor
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("pardiso", "superlu"))
    assert cli(run_dir, "--yaml", HOOK_YAML, "--estimate-only") == 0
    out = capsys.readouterr().out
    assert "solver:         pardiso" in out and "memory estimate with SuperLU instead" in out and "range " in out
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability())     # no factor can run, SuperLU neither
    assert cli(run_dir, "--yaml", HOOK_YAML, "--estimate-only") == 0
    out = capsys.readouterr().out
    assert "none can run" in out and "memory estimate with SuperLU instead" not in out
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("superlu"))
    assert cli(run_dir, "--yaml", HOOK_YAML, "--estimate-only") == 0
    assert "memory estimate with SuperLU instead" not in capsys.readouterr().out               # SuperLU is the solver: no "instead"


def test_cholespy_extra_state_is_missing_or_load_failed_with_the_real_error(monkeypatch):
    """The `extras` block of the report: `missing` for a package that is not installed, `load_failed` with the error text."""
    from topopt.verify import factor
    from topopt.verify import run as vrun
    monkeypatch.setattr("topopt.tet.gmsh_probe.probe_subprocess",
                        lambda cache=True: type("P", (), {"as_dict": lambda self: {"state": "ok"}})())
    monkeypatch.setattr(factor, "cholespy_problem", lambda: ("missing", "No module named 'cholespy'"))
    assert vrun.extras_facts()["cholespy"] == {"state": "missing", "detail": "No module named 'cholespy'"}
    monkeypatch.setattr(factor, "cholespy_problem", lambda: ("load_failed", "ImportError: libstdc++.so.6: too old"))
    got = vrun.extras_facts()["cholespy"]
    assert got["state"] == "load_failed" and "libstdc++.so.6" in got["detail"]


@needs_gmsh
@needs_skfem
def test_hook_end_to_end_on_superlu_gives_the_numbers_of_the_other_factor(hook_result, tmp_path, capsys):
    """The factor changes the time and the memory, not the answer: the same mesh gives the same stress (refined solves)."""
    run, code, out, err = hook_result
    other = json.loads((run / "verify" / "verify.json").read_text())
    mine = copy_run(tmp_path / "hook")
    assert cli(mine, "--yaml", HOOK_YAML, "--solver", "superlu", "--no-sensitivity") == 0
    cap = capsys.readouterr()
    assert "solver superlu" in cap.out and "solver:         superlu  (requested superlu)" in cap.err
    res = json.loads((mine / "verify" / "verify.json").read_text())
    assert res["backend"]["solver"] == "superlu" and res["mesh"]["dofs"] == other["mesh"]["dofs"]
    assert res["memory"]["estimate_model"].startswith("SuperLU model") and res["memory"]["peak_rss_bytes"] > 0
    for name, c in res["cases"].items():
        assert c["linear_solve"]["relative_residual"] <= 1e-9
        assert c["equilibrium"]["force_relative_residual"] <= 1e-6
        if other["backend"]["solver"] != "superlu":
            v, w = c["von_mises"], other["cases"][name]["von_mises"]
            assert v["p99_9_by_volume_MPa"] == pytest.approx(w["p99_9_by_volume_MPa"], rel=1e-6)
            assert c["stiffness_N_per_mm"] == pytest.approx(other["cases"][name]["stiffness_N_per_mm"], rel=1e-6)
