"""`topopt verify` with the bore checks (B4a, plan sections 7 and 8, CONTRACTS.md section 40), on the recorded hook run.

The hook has no press-fit bore. The tests give it one bore along the screwdriver channel (`CHANNEL`): the part has a wall there, so the
bore finds wall facets, and the numbers are those of a real tet run. The geometry of the facets is tested on a structured ring in
`test_verify_bore.py`; here the tests are about the switch, the keys, the report and the numbers that must not change.
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
from helpers import needs_gmsh, needs_skfem, tet_probe  # noqa: E402
from test_verify_run import HOOK_YAML, copy_run, needs_recording  # noqa: E402

pytestmark = [needs_gmsh, needs_skfem, needs_recording]

SIZE = 3.0
CHANNEL = {"name": "channel_top", "point": [3.5, 0.0, 30.0], "axis": [1.0, 0.0, 0.0], "radius_mm": 4.5, "length_mm": 76.5}
FAR = {"name": "far", "point": [3.5, 0.0, 300.0], "axis": [1.0, 0.0, 0.0], "radius_mm": 4.5, "length_mm": 20.0}
GOODMAN = {"materials": {"pla": {"su_mpa": 55.0, "sy_mpa": 45.0, "se_mpa": 15.0, "source": "numbers for a test"}}}
PRESS = {"cups": [{"name": "rigid", "k_mpa_per_mm": 1.0e6, "source": "numbers for a test"}]}


def block(bores=(CHANNEL,), goodman=GOODMAN, press_fit=None, **knobs) -> dict:
    b = {"bores": [dict(x) for x in bores], **knobs}
    if goodman:
        b["goodman"] = goodman
    if press_fit:
        b["press_fit"] = press_fit
        for x in b["bores"]:
            x["diametral_interference_mm"] = 0.1
            x["diametral_interference_mm_source"] = "number for a test"
    return b


def yaml_with(tmp_path, bore_checks=None, name="hook_b4.yaml") -> Path:
    """The hook YAML with absolute region paths and the given block `bore_checks`."""
    d = yaml.safe_load(HOOK_YAML.read_text())
    base = HOOK_YAML.parent
    r = d["regions"]
    r["design"] = str(base / r["design"])
    r["fixed"] = [str(base / x) for x in r["fixed"]]
    r["keep_out"] = [str(base / x) for x in r["keep_out"]]
    r["load"] = {k: str(base / v) for k, v in r["load"].items()}
    if bore_checks is not None:
        d["bore_checks"] = bore_checks
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d))
    return p


def cli(run, yaml_path, *argv):
    from topopt.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(run), "--yaml", str(yaml_path), "--mesh-size", str(SIZE), "--no-sensitivity", *map(str, argv)])
    return code, out.getvalue(), err.getvalue()


def result(run) -> dict:
    return json.loads((Path(run) / "verify" / "verify.json").read_text())


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Real runs of the recorded hook run at 3 mm: no block, the block with `--no-bore-checks`, the block (Goodman only), a press-fit block."""
    probe = tet_probe()
    if not probe.ok:          # a module fixture runs before the marker `needs_gmsh` skips: skip here without any work
        pytest.skip(f"needs gmsh (the 'tet' extra): {probe.state}: {probe.detail}. {probe.fix}")
    root = tmp_path_factory.mktemp("b4")
    plan = {"absent": (None, ()), "flag_off": (block(), ("--no-bore-checks",)), "goodman": (block(), ()),
            "press": (block(press_fit=PRESS), ()), "far": (block(bores=(FAR,)), ())}
    out = {}
    for name, (bc, argv) in plan.items():
        run = copy_run(root / name)
        code, text, err = cli(run, yaml_with(root, bc, f"{name}.yaml"), *argv)
        out[name] = (run, code, text, err)
    return out


# --------------------------------------------------------------------------- #
# Off: nothing new
# --------------------------------------------------------------------------- #

def _numbers(x, path=()):
    """The numeric leaves of a JSON value with their paths, without the keys of the clock, the memory of the machine and the files."""
    if isinstance(x, dict):
        for k, v in x.items():
            if not (k in ("timing_s", "memory", "memory_limit", "files", "run", "stl") or k.startswith("seconds") or k.endswith("_bytes")):
                yield from _numbers(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _numbers(v, path + (i,))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, x


def test_the_block_with_the_flag_off_gives_the_result_of_the_run_with_no_block(runs):
    a, b = result(runs["absent"][0]), result(runs["flag_off"][0])
    assert runs["absent"][1] == runs["flag_off"][1] == 0
    assert set(a) == set(b) and not any(k in a for k in ("bore_checks", "bore_facts", "goodman", "pressfit"))
    assert set(a["timing_s"]) == set(b["timing_s"]) and "bore_s" not in a["timing_s"] and "goodman_s" not in a["timing_s"]
    assert dict(_numbers(a)) == dict(_numbers(b))
    for stream in (2, 3):
        assert not any(w in runs["absent"][stream] for w in ("bore", "goodman", "press fit"))


def test_the_block_changes_no_number_of_the_run_and_adds_only_its_keys(runs):
    a, g = result(runs["absent"][0]), result(runs["goodman"][0])
    assert runs["goodman"][1] == 0, runs["goodman"][3]
    assert set(g) - set(a) == {"bore_checks", "bore_facts", "goodman"} and set(a) <= set(g)
    assert set(g["timing_s"]) - set(a["timing_s"]) == {"bore_s", "goodman_s"} and set(a["timing_s"]) <= set(g["timing_s"])
    na, ng = dict(_numbers(a)), dict(_numbers(g))
    assert {k: v for k, v in ng.items() if k[0] not in ("bore_checks", "bore_facts", "goodman")} == na      # bit for bit
    assert g["cases"] == a["cases"]


# --------------------------------------------------------------------------- #
# On: the base key, the bores, the Goodman rows
# --------------------------------------------------------------------------- #

def test_the_base_key_names_what_is_on_the_knobs_and_the_bores(runs):
    bc = result(runs["goodman"][0])["bore_checks"]
    assert bc["on"] is True and bc["source"] == "bore_checks in the YAML of the run" and bc["checks"] == ["goodman"]
    assert bc["statistics"] == ["p99_9", "max"]
    assert {k: (v["value"], v["origin"]) for k, v in bc["knobs"].items()} == {
        "radius_tolerance_mm": (0.35, "default"), "end_margin_mm": (0.35, "default"), "normal_limit": (0.35, "default"),
        "sample_band_mm": (1.2, "default")}
    assert all("default_source" in v and "source" not in v for v in bc["knobs"].values())
    (b,) = bc["bores"]
    assert b["name"] == "channel_top" and b["geometry_from"] == "yaml" and b["axis"] == [1.0, 0.0, 0.0] and "percentile_note" not in bc


def test_a_knob_of_the_yaml_has_the_origin_yaml_and_its_source(tmp_path):
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    bc = block(radius_tolerance_mm=0.5, radius_tolerance_mm_source="a number for a test")
    res = verify_run(run, yaml_path=yaml_with(tmp_path, bc), mesh_size=SIZE, estimate_only=True)
    k = res["bore_checks"]["knobs"]
    assert k["radius_tolerance_mm"] == {"value": 0.5, "origin": "yaml", "source": "a number for a test"}
    assert k["end_margin_mm"]["origin"] == "default"
    assert res["bore_checks"]["bores"][0]["name"] == "channel_top" and "bore_facts" not in res


def test_a_percentile_other_than_the_default_is_noted(tmp_path):
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    res = verify_run(run, yaml_path=yaml_with(tmp_path, block()), mesh_size=SIZE, estimate_only=True, percentile=99.0)
    assert "stay p99.9 and max" in res["bore_checks"]["percentile_note"]


def test_the_bore_has_wall_facets_a_sample_set_and_the_facts_of_the_funnel(runs):
    (e,) = result(runs["goodman"][0])["bore_facts"]
    assert e["name"] == "channel_top" and e["status"] == "ok" and e["geometry_from"] == "yaml"
    f = e["funnel"]
    assert f["boundary"] >= f["radius"] >= f["axial"] >= f["normal"] >= f["side"] == e["facets"] > 0
    assert e["sample_set"]["points"] > 0 and e["sample_set"]["band_mm"] == 1.2 and e["facts"] and e["options"]
    assert e["area_over_analytic"] == pytest.approx(e["area_mm2"] / e["analytic_area_mm2"])


def test_the_goodman_rows_follow_the_rule_with_sm_zero(runs):
    g = result(runs["goodman"][0])["goodman"]
    assert g["status"] == "done" and g["mean_stress"]["from"] == "zero_mean" and g["missing_rows"] == []
    assert [r["id"] for r in g["rows"]] == ["pla/zero_mean/channel_top/p99_9", "pla/zero_mean/channel_top/max"]
    sv = g["service_stress"]
    assert set(sv) == {"hang", "pull_off"}
    for r in g["rows"]:
        key = "vm_" + r["statistic"] + "_MPa"
        by_case = {c["case"]: c for c in r["cases"]}
        for case, c in by_case.items():
            sa = sv[case]["channel_top"][key]
            assert c["sa_MPa"] == sa and c["n_goodman"] == 1.0 / (sa / 15.0 + 0.0 / 55.0) and c["peak_over_yield"] == (0.0 + sa) / 45.0
        worst = min(by_case.values(), key=lambda c: c["n_goodman"])
        assert r["worst_case"] == worst["case"] == "pull_off" and r["n_goodman"] == worst["n_goodman"] and r["sm_MPa"] == 0.0
        assert r["all_n_none"] is False
    assert g["lowest_n"]["value"] == min(r["n_goodman"] for r in g["rows"]) and g["lowest_n"]["id"] in [r["id"] for r in g["rows"]]
    assert g["materials"]["pla"]["su_mpa"] == 55.0 and g["materials"]["pla"]["sources"] == {"su_mpa": "numbers for a test",
                                                                                         "sy_mpa": "numbers for a test",
                                                                                         "se_mpa": "numbers for a test"}


def test_the_report_has_the_lines_of_the_bore_and_the_goodman_check(runs):
    text = runs["goodman"][2]
    assert "bore checks:    goodman (bore_checks in the YAML of the run); 1 bore(s): channel_top; statistics p99.9 and max" in text
    assert "radius_tolerance_mm 0.35 (default), pipeline s9_print/final/pressfit.py:45" in text
    assert "bore channel_top (yaml):" in text and "wall facets, area" in text
    for case, by_bore in result(runs["goodman"][0])["goodman"]["service_stress"].items():          # the median is next to p99.9 and max (review m-2)
        s = by_bore["channel_top"]
        assert (f"  case {case}: von Mises at the sample set: p99.9 {s['vm_p99_9_MPa']:.4g}, max {s['vm_max_MPa']:.4g}, "
                f"median {s['vm_median_MPa']:.4g} MPa") in text
    assert "goodman (sm from zero_mean; statistics p99_9, max):" in text
    assert "pla/zero_mean/channel_top/p99_9: worst case pull_off" in text and "  fact: Rule: n = 1 / (sa/Se + sm/Su)" in text
    assert "  option: Remove the `bore_checks` block from the YAML (or use `--no-bore-checks`)" in text
    assert "pressfit: skipped" not in text                                                       # no press_fit block, no line of it


# --------------------------------------------------------------------------- #
# A press_fit block in this revision, and a bore with no wall
# --------------------------------------------------------------------------- #

def test_with_a_press_fit_block_the_stage_runs_and_its_hoop_stress_is_sm_of_the_goodman_rows(runs):
    """The solve itself is tested in `test_verify_pressfit.py` and `test_verify_pressfit_run.py`; here the Goodman rows of the cup variant follow it."""
    run, code, text, err = runs["press"]
    assert code == 0, err
    res = result(run)
    assert res["bore_checks"]["checks"] == ["goodman", "pressfit"]
    pf = res["pressfit"]
    assert pf["status"] == "done" and pf["reason"] is None and [v["name"] for v in pf["variants"]] == ["rigid"]
    g = res["goodman"]
    assert g["status"] == "done" and g["mean_stress"]["from"] == "press_fit" and g["missing_rows"] == [] and g["lowest_n"] is not None
    assert [r["id"] for r in g["rows"]] == ["pla/rigid/channel_top/p99_9", "pla/rigid/channel_top/max"]
    hoop = pf["variants"][0]["bores"][0]
    for r in g["rows"]:
        assert r["sm_MPa"] == hoop[f"hoop_{r['statistic']}_MPa"] and r["variant"] == "rigid"
    assert "pressfit (1 cup variant(s)" in text and "rigid / channel_top: k 1e+06 MPa/mm, hoop p99.9 " in text


def test_a_bore_with_no_wall_is_a_fact_and_the_run_goes_on(runs):
    run, code, text, err = runs["far"]
    assert code == 0, err
    res = result(run)
    (e,) = res["bore_facts"]
    assert e["name"] == "far" and e["status"] == "empty" and e["facets"] == 0 and e["sample_set"] is None
    assert any("No boundary facet of the mesh passes the four tests of the bore 'far'" in t for t in e["facts"])
    assert res["goodman"]["status"] == "skipped" and res["goodman"]["reason"] == "no_bore" and res["goodman"]["rows"] == []
    assert res["ok"] is True and "bore far (yaml): no wall facet" in text
    assert res["timing_s"]["bore_s"] >= 0


# --------------------------------------------------------------------------- #
# The switch: the YAML, the flags, E13
# --------------------------------------------------------------------------- #

def test_the_flag_needs_the_block_e13(tmp_path):
    from topopt.config import UserError
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    with pytest.raises(UserError, match=r"--bore-checks needs a bore_checks block in the YAML .*Options: add the block \(CONTRACTS.md section 40\)"):
        verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, estimate_only=True, bore_checks=True)
    code, out, err = cli(run, HOOK_YAML, "--bore-checks", "--estimate-only")
    assert code == 1 and "--bore-checks needs a bore_checks block" in err
    assert not (run / "verify").exists()


def test_the_yaml_turns_the_checks_on_and_the_flag_overrides_it(tmp_path):
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    on = yaml_with(tmp_path, block())
    res = verify_run(run, yaml_path=on, mesh_size=SIZE, estimate_only=True)
    assert res["bore_checks"]["source"] == "bore_checks in the YAML of the run"
    assert "bore_checks" not in verify_run(run, yaml_path=on, mesh_size=SIZE, estimate_only=True, bore_checks=False)
    assert "bore_checks" not in verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, estimate_only=True)
    assert verify_run(run, yaml_path=on, mesh_size=SIZE, estimate_only=True, bore_checks=True)["bore_checks"]["source"] == "the flag --bore-checks"
    pytest.importorskip("mcp")
    from topopt_mcp import server
    assert server.verify_run(run_dir=str(run), yaml_path=str(on), mesh_size_mm=SIZE)["bore_checks"]["checks"] == ["goodman"]


def test_a_bad_block_stops_before_any_mesh_with_the_key_in_the_message(tmp_path):
    from topopt.config import UserError
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    bad = block()
    bad["goodman"] = {"materials": {"pla": {"su_mpa": 55.0, "sy_mpa": 45.0, "se_mpa": 15.0}}}       # no source: E7
    with pytest.raises(UserError, match="source"):
        verify_run(run, yaml_path=yaml_with(tmp_path, bad), mesh_size=SIZE, estimate_only=True)
    assert not (run / "verify").exists()


def test_the_card_of_the_tool_names_the_keys_of_the_block_and_the_tool_has_no_argument_for_it():
    """K1-FU-C: the list of result keys moved from the tool text to the card `verify_run`; the tool text points to the card."""
    pytest.importorskip("mcp")
    from topopt import levers
    from topopt_mcp import server
    doc = server.verify_run.__doc__
    card = levers.reference("verify_run")
    assert "bore_checks: bore_checks, bore_facts, goodman, pressfit if press_fit." in card          # pressfit is there only with a press_fit block (review n-8)
    assert "bore_checks" in doc and "topopt://levers/verify_run" in doc                              # the tool text names the block and the card
    assert "bore_checks" not in server.verify_run.__code__.co_varnames[:server.verify_run.__code__.co_argcount]       # the YAML decides (plan 7.1)


def test_several_levels_run_the_bores_on_every_level_and_report_the_change(tmp_path):
    from topopt.verify.report import summary_lines
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    res = verify_run(run, yaml_path=yaml_with(tmp_path, block()), levels=[3.0, 2.5], load_sensitivity=False)
    assert res["reference_level_mm"] == 2.5 and res["goodman"]["status"] == "done"
    lv = res["levels"]
    ids = [r["id"] for r in res["goodman"]["rows"]]
    assert [set(r["goodman_n"]) for r in lv["rows"]] == [set(ids), set(ids)]
    assert lv["rows"][1]["goodman_n"] == {r["id"]: r["n_goodman"] for r in res["goodman"]["rows"]}
    first, last = lv["rows"][0]["goodman_n"], lv["rows"][1]["goodman_n"]
    for i in ids:
        assert lv["relative_change"][0]["goodman_n"][i] == pytest.approx((last[i] - first[i]) / first[i], rel=1e-12)
    assert lv["relative_change_first_to_last"]["goodman_n"].keys() == set(ids)
    assert any(x.startswith("goodman (sm from zero_mean") for x in summary_lines(res))


# --------------------------------------------------------------------------- #
# The record when the check stops
# --------------------------------------------------------------------------- #

def test_an_exception_of_the_goodman_check_is_a_fact_and_the_run_goes_on(monkeypatch):
    from topopt.config import parse_bore_checks_block
    from topopt.verify import fatigue
    from topopt.verify import run as vrun

    class Cfg:
        bore_checks = parse_bore_checks_block(block())
        load_cases = []

    def boom(*a, **k):
        raise RuntimeError("a made up failure")

    monkeypatch.setattr(fatigue, "build_record", boom)
    said, T = [], {}
    facts, good, press = vrun._bore_records(Cfg, [], {}, T, said.append)
    assert facts == [] and press is None and good["status"] == "error" and good["reason"] == "exception"
    assert "RuntimeError: a made up failure" in good["message"] and "goodman_s" in T and any("goodman check stopped" in s for s in said)


# --------------------------------------------------------------------------- #
# A bore that fails (review m-5 of B4a): CONTRACTS 40.4 says that `topopt verify` goes on
# --------------------------------------------------------------------------- #

SECOND = dict(CHANNEL, name="channel_b", length_mm=60.0)


def _two_bores(tmp_path, name):
    run = copy_run(tmp_path / name)
    code, text, err = cli(run, yaml_with(tmp_path, block(bores=(CHANNEL, SECOND)), f"{name}.yaml"))
    return run, code, text, err


def _the_rest_of_the_report_goes_on(runs, run, code, text):
    """The other bore has the numbers of the run with that bore alone, and the cases and the text are there."""
    assert code == 0
    res, one = result(run), result(runs["goodman"][0])
    assert dict(_numbers(res["cases"])) == dict(_numbers(one["cases"]))
    a, b = res["bore_facts"]
    assert a["name"] == "channel_top" and a["status"] == "ok" and a["facets"] == one["bore_facts"][0]["facets"] and b["name"] == "channel_b"
    g = res["goodman"]
    assert g["status"] == "done" and [r["id"] for r in g["rows"]] == [r["id"] for r in one["goodman"]["rows"]]
    assert [r["n_goodman"] for r in g["rows"]] == [r["n_goodman"] for r in one["goodman"]["rows"]]
    assert set(g["service_stress"]["hang"]) == set(g["service_stress"]["pull_off"]) == {"channel_top"}
    assert [(m["variant"], m["bore"], m["reason"]) for m in g["missing_rows"]] == [(None, "channel_b", "error")] and g["missing_rows"][0]["options"]
    assert "bore channel_top (yaml):" in text and "goodman (sm from zero_mean" in text and "wrote " in text
    return b


def test_a_bore_whose_build_raises_is_dropped_with_a_fact_and_verify_goes_on(runs, tmp_path, monkeypatch):
    from topopt.verify import bore as vbore
    real = vbore.build_bore

    def build_bore(model, spec, knobs):
        if spec.name == "channel_b":
            raise RuntimeError("a made up failure of the build")
        return real(model, spec, knobs)

    monkeypatch.setattr(vbore, "build_bore", build_bore)
    run, code, text, err = _two_bores(tmp_path, "build_raises")
    b = _the_rest_of_the_report_goes_on(runs, run, code, text)
    assert b["status"] == "error" and b["stage"] == "build" and b["message"] == "RuntimeError: a made up failure of the build"
    assert "'channel_b'" in b["facts"][0] and "RuntimeError: a made up failure of the build" in b["facts"][0] and b["options"]
    assert "bore channel_b (yaml): stopped at the build: RuntimeError: a made up failure of the build" in text
    assert "channel_b: stopped at the build (see bore_facts)" in text + err and "  option: Remove this bore from the `bore_checks` block" in text


def test_a_bore_whose_statistics_raise_is_dropped_with_a_fact_and_verify_goes_on(runs, tmp_path, monkeypatch):
    """The statistics fail at the second case: the value of the first case is dropped too, so that no row is made from a part of the cases."""
    from topopt.verify import bore as vbore
    real, calls = vbore.build, []

    def build(model, bc):
        bores = [b for b in real(model, bc)]
        for b in bores:
            if b.spec.name == "channel_b":
                ok = b.sample.stats

                def stats(sig, ok=ok):
                    calls.append(1)
                    if len(calls) == 2:
                        raise RuntimeError("a made up failure of the statistics")
                    return ok(sig)
                b.sample.stats = stats
        return bores

    monkeypatch.setattr(vbore, "build", build)
    run, code, text, err = _two_bores(tmp_path, "stats_raise")
    b = _the_rest_of_the_report_goes_on(runs, run, code, text)
    assert len(calls) == 2                                              # the bore is not asked again after it failed
    assert b["status"] == "error" and b["stage"] == "stats" and b["message"] == "RuntimeError: a made up failure of the statistics"
    assert "'channel_b'" in b["facts"][0] and "RuntimeError: a made up failure of the statistics" in b["facts"][0] and b["options"]
    assert "bore channel_b (yaml): stopped at the stats: RuntimeError: a made up failure of the statistics" in text
    assert "bore channel_b stopped: RuntimeError: a made up failure of the statistics" in text + err


def test_the_plan_carries_the_switch(tmp_path):
    """`Plan.bore_checks` is what `run_level` reads: the YAML block, the flag, or off (the ratchet judge passes `bore_checks=False`, plan 7.5)."""
    from topopt.verify.run import preflight
    run = copy_run(tmp_path / "run")
    on = yaml_with(tmp_path, block())
    plan = lambda y, **kw: preflight(run, yaml_path=y, mesh_size=SIZE, estimate_only=True, **kw)      # noqa: E731
    assert plan(on).bore_checks is True and plan(on, bore_checks=True).bore_checks is True
    off = plan(on, bore_checks=False)
    assert off.bore_checks is False and "bore_checks" not in off.base
    assert plan(HOOK_YAML).bore_checks is False and plan(HOOK_YAML, bore_checks=False).bore_checks is False
