"""`topopt verify` with the buckling check and the A5 rows, on the recorded hook run (B3, CONTRACTS.md sections 16, 31 and 33).

The recorded run is a hook run of an older build. A real run needs gmsh (`needs_gmsh`) and scikit-fem. The runs here use the mesh
size 3 mm (a few seconds each).
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_gmsh, needs_skfem, tet_probe  # noqa: E402
from test_verify_run import HOOK_YAML, RECORDING, copy_run, needs_recording  # noqa: E402

pytestmark = [needs_gmsh, needs_skfem, needs_recording]

SIZE = 3.0
# The voxel check of the run (A6a, the recorded hook design): the lowest load factors of CONTRACTS.md section 33.
VOXEL_LOWEST = {"hang": 1276.367, "pull_off": 120.907}
ROW = "displacement:tip_pull:pull_off"
ROW_VOXEL = 0.020998216820864963                      # value_end of the voxel row of the recorded A5 design


def voxel_metrics(run: Path, rows=False):
    """Add the key `buckling` (and with `rows` the key `rows`) of a voxel run to the `metrics.json` of the copy."""
    path = run / "result" / "metrics.json"
    m = json.loads(path.read_text())
    m["buckling"] = {"status": "done", "density": "printed", "solver": "pardiso", "cases": [
        {"name": n, "status": "ok", "load_factors": [v, 1.3 * v]} for n, v in VOXEL_LOWEST.items()]}
    if rows:
        m["rows"] = [{"name": ROW, "kind": "displacement", "unit": "mm", "limit": 0.021, "value_start": 1.1,
                      "value_end": ROW_VOXEL, "g_end": -3.5e-05, "met": True}]
    path.write_text(json.dumps(m))


def cli(run, *argv, yaml_path=HOOK_YAML):
    from topopt.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(run), "--yaml", str(yaml_path), "--mesh-size", str(SIZE), *map(str, argv)])
    return code, out.getvalue(), err.getvalue()


def result(run) -> dict:
    return json.loads((Path(run) / "verify" / "verify.json").read_text())


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
    p = tmp_path / "hook_b3.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two real runs of the same recorded run: default (everything off) and `--buckling`."""
    probe = tet_probe()
    if not probe.ok:          # the marker `needs_gmsh` skips the tests, but a module fixture runs before it: skip here without any work
        pytest.skip(f"needs gmsh (the 'tet' extra): {probe.state}: {probe.detail}. {probe.fix}")
    root = tmp_path_factory.mktemp("b3")
    out = {}
    for name in ("off", "on"):
        run = copy_run(root / name)
        voxel_metrics(run)
        code, text, err = cli(run, "--buckling") if name == "on" else cli(run)
        out[name] = (run, code, text, err)
    return out


# --------------------------------------------------------------------------- #
# Off: nothing new
# --------------------------------------------------------------------------- #

# The keys of verify.json at e6a3e70 (before B3), taken from a real run of that revision.
MAIN_TOP_KEYS = {"backend", "band_is_default", "band_mm", "cases", "estimate_only", "estimates_before_meshing", "exit_code", "extras",
                 "files", "format", "kind", "load_band_sensitivity", "load_sensitivity", "material", "memory", "memory_limit", "mesh",
                 "near_region_mm", "ok", "peak_rss_bytes", "platform", "reference_level_mm", "regions", "run", "size_mm", "stl",
                 "timing_s", "yaml_vs_run"}
MAIN_TIMING_KEYS = {"assemble_s", "factor_s", "mesh_s", "model_s", "regions_s", "sensitivity_s", "solve_s", "stress_s", "total_s",
                    "verify_total_s"}
MAIN_MEMORY_KEYS = {"children_peak_rss_bytes", "estimate_after_meshing_bytes", "estimate_before_meshing_bytes", "estimate_model",
                    "estimate_range_after_meshing_bytes", "estimate_seconds_after_meshing", "measured_over_estimate",
                    "measured_over_rule", "note", "peak_rss_after_assembly_bytes", "peak_rss_after_factor_bytes", "peak_rss_bytes",
                    "rule_bytes"}
MAIN_ESTIMATE_KEYS = {"above_limit", "bytes_by_backend", "compactness", "estimate_bytes", "estimate_gb", "estimate_model",
                      "estimated_dofs", "estimated_seconds", "notes", "range_bytes", "ranges_by_backend", "rule_bytes",
                      "seconds_by_backend", "seconds_note", "size_mm", "upper_end_above_limit"}
MAIN_CASE_KEYS = {"compliance_Nmm", "equilibrium", "force_on_fixed_nodes_N", "force_on_fixed_nodes_relative", "forces", "linear_solve",
                  "max_displacement_mm", "optimizer_stiffness_N_per_mm", "stiffness_N_per_mm", "stiffness_over_optimizer", "von_mises",
                  "weight_of_the_optimizer", "weight_used"}


def test_with_both_checks_off_verify_json_has_the_keys_of_the_revision_before_b3(runs):
    run, code, out, err = runs["off"]
    assert code == 0
    res = result(run)
    # P4b-1 adds one key for a run with no allowable (this run has none): `no_allowable`, the note that names the material
    # reference (CONTRACTS.md section 25.7). TC-2 adds `partial` and `coverage` to every real result (section 44). The checks of B3 add no key.
    assert set(res) == MAIN_TOP_KEYS | {"no_allowable", "partial", "coverage"} and res["no_allowable"]["resource"] == "topopt://materials/reference"
    assert res["partial"] is False and res["coverage"]["ran"] == ["stress"]
    assert set(res["timing_s"]) == MAIN_TIMING_KEYS and set(res["memory"]) == MAIN_MEMORY_KEYS
    assert set(res["estimates_before_meshing"][0]) == MAIN_ESTIMATE_KEYS
    assert all(set(c) == MAIN_CASE_KEYS for c in res["cases"].values())
    # The plain text has a scope line (P4b-1, `report.plain_lines`): "It does not cover fatigue, buckling, ...". The line says that the
    # check did not run. It is not output of the check (and the text wraps it), so the test cuts that one sentence and looks at the rest.
    scope = "It does not cover fatigue, buckling, printing defects or print direction."
    flat = " ".join(out.split())
    assert scope in flat
    assert "buckling" not in flat.replace(scope, "") and "motion rows" not in out and "buckling" not in err
    assert not any("buckling" in k or "motion" in k for k in res)


# The numbers of verify.json at e6a3e70 (before B3): `tests/fixtures/verify_checks_off_3mm.json` holds the numeric leaves of the keys
# below, from a real run of that revision with the same command (the recorded hook, mesh size 3 mm, solver auto). The paths, hashes,
# texts, times, memory and the lists of top elements are not pinned. The factor can differ between platforms (pardiso, cholesky, ...),
# and every factor gives the same refined solution to about 1e-9, so the floats are compared to 1e-6 (relative).
CHECKS_OFF_VALUES = Path(__file__).parent / "fixtures" / "verify_checks_off_3mm.json"
PIN_KEYS = ("format", "kind", "ok", "exit_code", "estimate_only", "size_mm", "reference_level_mm", "band_mm", "band_is_default",
            "near_region_mm", "load_sensitivity", "material", "stl", "mesh", "regions", "cases", "load_band_sensitivity", "yaml_vs_run")
NOT_PINNED = {"seconds", "seconds_gmsh", "seconds_surface", "top_elements"}


def pinned_numbers(res: dict) -> dict:
    """`{path: number or bool}` of the pinned keys of a `verify.json`."""
    out = {}

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if k not in NOT_PINNED:
                    walk(v, f"{path}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, (int, float, bool)):
            out[path] = x
    for key in PIN_KEYS:
        if key in res:
            walk(res[key], key)
    return out


def test_with_both_checks_off_the_numbers_are_those_of_the_revision_before_b3(runs):
    ref = json.loads(CHECKS_OFF_VALUES.read_text())
    got = pinned_numbers(result(runs["off"][0]))
    if (got["mesh/tets"], got["mesh/dofs"]) != (ref["mesh/tets"], ref["mesh/dofs"]):
        pytest.skip(f"this gmsh made another mesh ({got['mesh/tets']} tets) than the fixture ({ref['mesh/tets']}); the numbers cannot be compared")
    assert set(got) == set(ref), sorted(set(got) ^ set(ref))[:10]
    wrong = [k for k, v in ref.items()
             if (got[k] != v if isinstance(v, (bool, int)) else got[k] != pytest.approx(v, rel=1e-6, abs=1e-9))]
    assert not wrong, [(k, ref[k], got[k]) for k in wrong[:10]]
    assert len(ref) > 300 and "cases/pull_off/von_mises/p99_9_by_volume_MPa" in ref        # the fixture is not empty


def same(a, b, path=""):
    """Recursive equality; floats to 1e-9 (relative), so that a platform with another BLAS still passes."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


def test_with_the_check_on_every_other_result_is_the_same(runs):
    off, on = result(runs["off"][0]), result(runs["on"][0])
    assert set(on) - set(off) == {"buckling", "buckling_check"} and set(off) <= set(on)
    for key in ("cases", "mesh", "regions", "load_band_sensitivity", "stl", "material", "band_mm", "yaml_vs_run", "size_mm",
                "reference_level_mm", "near_region_mm", "load_sensitivity", "exit_code"):
        same({k: v for k, v in off[key].items() if k not in ("seconds", "seconds_gmsh", "seconds_surface")} if key == "mesh" else off[key],
             {k: v for k, v in on[key].items() if k not in ("seconds", "seconds_gmsh", "seconds_surface")} if key == "mesh" else on[key],
             key)
    assert set(on["timing_s"]) - set(off["timing_s"]) == {"buckling_s"}
    assert set(on["estimates_before_meshing"][0]) - set(off["estimates_before_meshing"][0]) == {
        "buckling_bytes", "estimate_with_buckling_bytes", "above_limit_with_buckling"}


# --------------------------------------------------------------------------- #
# On: the record
# --------------------------------------------------------------------------- #

def test_the_flag_gives_the_record_with_the_voxel_value_and_the_ratio(runs):
    run, code, out, err = runs["on"]
    assert code == 0
    res = result(run)
    assert res["buckling_check"] == {"on": True, "source": "the flag --buckling", "modes": 6, "max_seconds": 120.0}
    bk = res["buckling"]
    assert bk["status"] == "done" and bk["solver"] == res["backend"]["solver"] and bk["modes"] == 6
    assert bk["kind"].startswith("linear buckling of the P2 tet model") and bk["loose_piece"]["applicable"] is False
    assert [c["name"] for c in bk["cases"]] == ["hang", "pull_off"]
    for c in bk["cases"]:
        v = VOXEL_LOWEST[c["name"]]
        assert c["status"] == "ok" and len(c["load_factors"]) == 6 and c["load_factors"] == sorted(c["load_factors"])
        assert c["residual_max"] < 1e-6 and c["near_repeated"] is False and c["modes_converged"] == 6
        assert c["voxel"]["lowest"] == v and c["voxel"]["density"] == "printed"
        assert c["tet_over_voxel"] == pytest.approx(c["load_factors"][0] / v, rel=1e-12)
        assert 0.4 < c["tet_over_voxel"] < 0.9                                   # the tet model is softer (section 33 and the plan)
        assert c["stiffness_over_optimizer"] == pytest.approx(res["cases"][c["name"]]["stiffness_over_optimizer"], rel=1e-12)
        assert len(c["lowest_mode_energy_box_mm"]) == 2
    low = bk["lowest_load_factor"]
    assert low["case"] == "pull_off" and low["value"] == bk["cases"][1]["load_factors"][0]      # the pull-off case is the weak one
    assert bk["facts"] == [] and bk["memory"]["term_bytes"] > 0
    assert bk["peak_rss_bytes"] >= bk["peak_rss_before_bytes"] > 0 and bk["seconds"] > 0
    assert res["timing_s"]["buckling_s"] >= bk["seconds"] * 0.9
    # the lines for the reader
    assert "buckling (P2 tets, solver" in out and "tet over voxel" in out and "difference of two models" in out
    assert "linear buckling check of the tet model" in err and "memory estimate with the check" in err


def test_the_estimate_with_the_check_is_in_the_result_and_in_the_text(runs):
    run, code, out, err = runs["on"]
    e = result(run)["estimates_before_meshing"][0]
    assert e["estimate_with_buckling_bytes"] == e["estimate_bytes"] + e["buckling_bytes"] and e["buckling_bytes"] > 1e8
    assert e["above_limit_with_buckling"] is False
    assert f"with the buckling check: {e['estimate_with_buckling_bytes'] / 1e9:.2f} GB" in err
    # the peak of the process holds the check, so the result compares it also with the estimate that has the term
    mem = result(run)["memory"]
    assert mem["estimate_with_buckling_bytes"] > mem["estimate_after_meshing_bytes"]
    assert mem["measured_over_estimate_with_buckling"] == pytest.approx(mem["peak_rss_bytes"] / mem["estimate_with_buckling_bytes"], rel=1e-12)
    assert f"{mem['estimate_with_buckling_bytes'] / 1e9:.2f} GB with the buckling check)" in out


def test_a_check_that_does_not_fit_is_skipped_with_the_facts_and_the_stress_check_goes_on(tmp_path):
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    est = verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, solver="superlu", buckling=True, estimate_only=True)
    e = est["estimates_before_meshing"][0]
    limit_gb = (e["estimate_bytes"] + 0.45 * e["buckling_bytes"]) / 1e9              # room for the factor, not for the check
    code, out, err = cli(run, "--solver", "superlu", "--buckling", "--max-gb", f"{limit_gb:.4f}")
    assert code == 0, err
    res = result(run)
    bk = res["buckling"]
    assert bk["status"] == "skipped" and bk["reason"] == "memory" and bk["cases"] == [] and bk["lowest_load_factor"] is None
    assert "above the limit" in bk["facts"][0] and "stress check of verify goes on" in bk["facts"][0]
    assert any("--max-gb" in o and "--force" in o for o in bk["options"])
    assert res["ok"] is True and set(res["cases"]) == {"hang", "pull_off"} and "buckling_s" not in res["timing_s"]
    assert "buckling: skipped (memory)" in out and "the check is skipped" in err
    # --force starts the check
    run2 = copy_run(tmp_path / "run2")
    code, out, err = cli(run2, "--solver", "superlu", "--buckling", "--max-gb", f"{limit_gb:.4f}", "--force")
    assert code == 0 and result(run2)["buckling"]["status"] == "done"


# --------------------------------------------------------------------------- #
# The switch: the YAML, the flags, and several levels
# --------------------------------------------------------------------------- #

def test_the_yaml_key_turns_the_check_on_and_the_flags_override_it(tmp_path):
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    on = yaml_with(tmp_path, buckling={"check": True, "modes": 3, "max_seconds": 77})
    res = verify_run(run, yaml_path=on, mesh_size=SIZE, estimate_only=True)
    assert res["buckling_check"] == {"on": True, "source": "buckling.check in the YAML of the run", "modes": 3, "max_seconds": 77.0}
    assert "buckling_bytes" in res["estimates_before_meshing"][0]
    assert "buckling_check" not in verify_run(run, yaml_path=on, mesh_size=SIZE, estimate_only=True, buckling=False)
    off = verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, estimate_only=True)
    assert "buckling_check" not in off and "buckling_bytes" not in off["estimates_before_meshing"][0]
    assert verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, estimate_only=True, buckling=True)["buckling_check"]["modes"] == 6
    # the MCP tool follows the YAML (no argument): the estimate carries the check
    pytest.importorskip("mcp")
    from topopt_mcp import server
    assert server.verify_run(run_dir=str(run), yaml_path=str(on), mesh_size_mm=SIZE)["buckling_check"]["modes"] == 3


def test_the_yaml_switch_runs_the_check_with_its_modes_and_no_flag_is_needed(tmp_path):
    run = copy_run(tmp_path / "run")
    voxel_metrics(run)
    code, out, err = cli(run, "--no-sensitivity", yaml_path=yaml_with(tmp_path, buckling={"check": True, "modes": 3}))
    assert code == 0, err
    res = result(run)
    assert res["buckling_check"]["source"] == "buckling.check in the YAML of the run" and res["buckling"]["modes"] == 3
    assert all(len(c["load_factors"]) == 3 for c in res["buckling"]["cases"])
    run2 = copy_run(tmp_path / "run2")
    code, out, err = cli(run2, "--no-sensitivity", "--no-buckling", yaml_path=yaml_with(tmp_path, buckling={"check": True}))
    assert code == 0 and "buckling" not in result(run2)


def test_several_levels_run_the_check_on_every_level_and_report_the_change(tmp_path):
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    voxel_metrics(run)
    res = verify_run(run, yaml_path=HOOK_YAML, levels=[3.0, 2.5], buckling=True, load_sensitivity=False)
    assert res["buckling"]["status"] == "done" and res["reference_level_mm"] == 2.5
    lv = res["levels"]
    lo = [row["buckling_lowest_load_factor"] for row in lv["rows"]]
    assert [row["size_mm"] for row in lv["rows"]] == [3.0, 2.5] and all(set(x) == {"hang", "pull_off"} for x in lo)
    assert lo[1]["pull_off"] == res["buckling"]["lowest_load_factor"]["value"]
    ch = lv["relative_change"][0]["buckling_lowest_load_factor"]["pull_off"]
    assert ch == pytest.approx((lo[1]["pull_off"] - lo[0]["pull_off"]) / lo[0]["pull_off"], rel=1e-12) and abs(ch) < 0.2
    assert lv["relative_change_first_to_last"]["buckling_lowest_load_factor"]["hang"] is not None
    from topopt.verify.report import summary_lines
    assert any("lowest buckling load factor" in x for x in summary_lines(res))


# --------------------------------------------------------------------------- #
# The A5 rows
# --------------------------------------------------------------------------- #

def test_the_rows_of_the_yaml_are_computed_from_the_tet_displacement(tmp_path):
    run = copy_run(tmp_path / "run")
    voxel_metrics(run, rows=True)
    limits = {"displacement": [{"name": "tip_pull", "region": "tip", "limit_mm": 0.021, "cases": ["pull_off"]}]}
    code, out, err = cli(run, "--no-sensitivity", yaml_path=yaml_with(tmp_path, limits=limits))
    assert code == 0, err
    res = result(run)
    assert "buckling" not in res and "buckling_check" not in res                    # the rows need no flag and no check
    mo = res["motion"]
    assert mo["status"] == "done" and len(mo["rows"]) == 1 and res["timing_s"]["motion_s"] > 0
    (row,) = mo["rows"]
    assert row["name"] == ROW and row["unit"] == "mm" and row["limit"] == 0.021 and row["case"] == "pull_off"
    assert row["voxel_value"] == ROW_VOXEL and row["difference"] == pytest.approx(row["tet_value"] - ROW_VOXEL, rel=1e-12)
    assert row["tet_over_limit"] == pytest.approx(row["tet_value"] / 0.021, rel=1e-12) and "met" not in row
    assert 0.02 < row["tet_value"] < 0.06                                          # 0.034 mm at 1.496 mm, 0.031 mm at 0.8 mm
    tip = mo["regions"]["tip"]
    assert tip["role"] == "load region" and tip["nodes"] > 100 and tip["vertices"] < tip["nodes"] and tip["fixed_nodes"] == 0
    assert "motion rows (A5, P2 tets):" in out and ROW in out and "tet over limit" in out
    # the tet value is the mean over the P2 nodes of the region: the max displacement of the case is larger
    assert row["tet_value"] < res["cases"]["pull_off"]["max_displacement_mm"]


def test_both_checks_together_in_one_run(tmp_path):
    run = copy_run(tmp_path / "run")
    voxel_metrics(run, rows=True)
    limits = {"displacement": [{"name": "tip_pull", "region": "tip", "limit_mm": 0.021, "cases": ["pull_off"]}]}
    code, out, err = cli(run, "--no-sensitivity", "--buckling", yaml_path=yaml_with(tmp_path, limits=limits))
    assert code == 0, err
    res = result(run)
    assert res["buckling"]["status"] == "done" and res["motion"]["status"] == "done"
    assert set(res["timing_s"]) >= {"buckling_s", "motion_s"}


# --------------------------------------------------------------------------- #
# A failure of a check does not stop verify; the check gets the plain live factor
# --------------------------------------------------------------------------- #

def test_a_failure_of_the_whole_check_is_an_error_record_and_the_stress_result_stays(tmp_path, monkeypatch):
    from topopt.verify import buckling_p2

    def boom(*args, **kwargs):
        raise RuntimeError("phase 33 error -4")
    monkeypatch.setattr(buckling_p2, "run_check", boom)
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, "--buckling", "--no-sensitivity")
    assert code == 0, err
    res = result(run)
    bk = res["buckling"]
    assert bk["status"] == "error" and bk["reason"] == "exception" and bk["cases"] == [] and bk["lowest_load_factor"] is None
    assert bk["message"] == "RuntimeError: phase 33 error -4" and "phase 33 error -4" in bk["facts"][0]
    assert any("--max-gb" in o and "--mesh-size" in o and "direct factor" not in o for o in bk["options"])
    assert res["ok"] is True and res["exit_code"] == 0 and set(res["cases"]) == {"hang", "pull_off"}      # the stress result stays
    assert all(c["von_mises"]["p99_9_by_volume_MPa"] > 0 for c in res["cases"].values()) and "buckling_s" in res["timing_s"]
    assert "buckling check stopped: RuntimeError: phase 33 error -4" in err and "buckling: error (exception)" in out


def test_the_check_gets_the_plain_live_factor_and_the_factor_is_freed_once(tmp_path, monkeypatch):
    from topopt import buckling
    from topopt.verify import factor as vfactor
    frees, seen = [], []
    real_make, real_modes = vfactor.make_factor, buckling.buckling_modes

    def make(*args, **kwargs):
        fac = real_make(*args, **kwargs)
        real_free = fac.free

        def counted():
            frees.append(1)
            return real_free()
        fac.free = counted
        return fac

    def spy(Kg, K, fac, **kwargs):
        x = np.linspace(1.0, 2.0, K.shape[0])
        residual = np.linalg.norm(K @ fac.solve(K @ x) - K @ x) / np.linalg.norm(K @ x)
        seen.append((fac, len(frees), residual))
        return real_modes(Kg, K, fac, **kwargs)
    monkeypatch.setattr(vfactor, "make_factor", make)
    monkeypatch.setattr(buckling, "buckling_modes", spy)
    run = copy_run(tmp_path / "run")
    code, out, err = cli(run, "--buckling", "--no-sensitivity")
    assert code == 0, err
    assert len(seen) == 2 and seen[0][0] is seen[1][0]                      # one factor for both cases
    assert not isinstance(seen[0][0], vfactor.RefinedFactor)                # the plain factor, not the refined one
    assert all(n == 0 for _, n, _ in seen) and all(r < 1e-6 for _, _, r in seen)    # alive, and it solves
    assert len(frees) == 1                                                  # freed once, after the check
    assert result(run)["buckling"]["status"] == "done"


def test_a_failure_of_the_rows_is_an_error_record_and_the_stress_result_stays(tmp_path, monkeypatch):
    from topopt.verify import motion
    monkeypatch.setattr(motion, "motion_record", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad measure STL")))
    run = copy_run(tmp_path / "run")
    limits = {"displacement": [{"name": "tip_pull", "region": "tip", "limit_mm": 0.021, "cases": ["pull_off"]}]}
    code, out, err = cli(run, "--no-sensitivity", yaml_path=yaml_with(tmp_path, limits=limits))
    assert code == 0, err
    res = result(run)
    mo = res["motion"]
    assert mo["status"] == "error" and mo["reason"] == "exception" and mo["rows"] == [] and mo["message"] == "ValueError: bad measure STL"
    assert "bad measure STL" in mo["facts"][0] and "Verify goes on without the rows" in mo["options"][0] and "met" not in mo
    assert res["ok"] is True and set(res["cases"]) == {"hang", "pull_off"} and "motion_s" in res["timing_s"]
    assert "motion rows: error (exception): ValueError: bad measure STL" in err and "motion rows: error (exception)" in out


# --------------------------------------------------------------------------- #
# The texts before meshing do not state a skip that is not decided yet
# --------------------------------------------------------------------------- #

def test_the_estimate_lines_say_the_check_can_be_skipped_after_meshing_and_name_force(tmp_path):
    from topopt.verify.report import estimate_lines
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    first = verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, solver="superlu", buckling=True, estimate_only=True)
    e = first["estimates_before_meshing"][0]
    max_gb = (e["estimate_bytes"] + 0.45 * e["buckling_bytes"]) / 1e9              # room for the factor, not for the check
    res = verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, solver="superlu", buckling=True, estimate_only=True, max_gb=max_gb)
    assert res["estimates_before_meshing"][0]["above_limit_with_buckling"] is True
    text = "\n".join(estimate_lines(res))
    assert "the check can be skipped after meshing" in text and "--force starts the check" in text and "check is skipped" not in text
    forced = verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, solver="superlu", buckling=True, estimate_only=True, max_gb=max_gb, force=True)
    assert "skipped" not in "\n".join(estimate_lines(forced))
