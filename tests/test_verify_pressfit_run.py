"""`topopt verify` with the press-fit check (B4b, plan sections 6.4 to 6.6, 7 and 8; CONTRACTS.md section 40), on the recorded hook run.

The hook gets one bore along the screwdriver channel (`CHANNEL` of `test_verify_bore_run.py`). The module level (the Lame stiffness, the assembly, the closed form of a
ring) is tested in `test_verify_pressfit.py`; here the tests are about the wiring: the record, `sm` of the Goodman rows, the memory decision, the one factor at a
time, the switch, the `levels` block, the estimate keys and the text. A real run needs gmsh (`needs_gmsh`) and scikit-fem. The runs use the mesh size 3 mm.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_gmsh, needs_skfem, tet_probe  # noqa: E402
from test_verify_bore_run import PRESS, SIZE, _numbers, block, cli, result, yaml_with  # noqa: E402
from test_verify_run import HOOK_YAML, copy_run, needs_recording  # noqa: E402

pytestmark = [needs_gmsh, needs_skfem, needs_recording]

TWO_CUPS = {"cups": [{"name": "rigid", "k_mpa_per_mm": 1.0e6, "k_mpa_per_mm_source": "numbers for a test"},
                     {"name": "soft", "e_mpa": 3500.0, "nu": 0.35, "wall_mm": 1.5, "source": "numbers for a test"}]}
NEW_KEYS = {"bore_checks", "bore_facts", "goodman", "pressfit"}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Real runs at 3 mm: no block; the press-fit block with the flag off; Goodman with two cups (a spy counts the live factors); the press-fit block alone."""
    probe = tet_probe()
    if not probe.ok:
        pytest.skip(f"needs gmsh (the 'tet' extra): {probe.state}: {probe.detail}. {probe.fix}")
    import topopt.verify.factor as vf
    root = tmp_path_factory.mktemp("b4b")
    plan = {"absent": (None, ()), "flag_off": (block(press_fit=TWO_CUPS), ("--no-bore-checks",)), "two": (block(press_fit=TWO_CUPS), ()),
            "alone": (block(goodman=None, press_fit=PRESS), ())}
    out = {}
    live, peak, made = set(), [0], []
    real = vf.make_factor

    def spy(A_, name):
        fac = real(A_, name)
        made.append(name)
        live.add(id(fac))
        peak[0] = max(peak[0], len(live))
        orig = fac.free

        def free_():
            live.discard(id(fac))
            return orig()
        fac.free = free_
        return fac
    for name, (bc, argv) in plan.items():
        run = copy_run(root / name)
        with pytest.MonkeyPatch.context() as mp:
            if name == "two":
                mp.setattr(vf, "make_factor", spy)
            code, text, err = cli(run, yaml_with(root, bc, f"{name}.yaml"), *argv)
        out[name] = (run, code, text, err)
    out["factors"] = dict(made=list(made), peak=peak[0], live=len(live))
    out["root"] = root
    return out


def test_the_press_fit_record_has_a_variant_for_each_cup_and_the_facts(runs):
    run, code, text, err = runs["two"]
    assert code == 0, err
    pf = result(run)["pressfit"]
    assert pf["status"] == "done" and pf["kind"] == "pressfit" and pf["cups"] == ["rigid", "soft"] and pf["statistics"] == ["p99_9", "max"]
    assert [v["name"] for v in pf["variants"]] == ["rigid", "soft"] and pf["dropped_bores"] == []
    rigid, soft = pf["variants"]
    assert rigid["k_form"] == "k" and rigid["sources"] == {"k_mpa_per_mm": "numbers for a test"}
    assert soft["k_form"] == "lame" and soft["wall_mm"] == 1.5 and soft["e_mpa"] == 3500.0 and soft["sources"] == {k: "numbers for a test" for k in ("e_mpa", "nu", "wall_mm")}
    for v in pf["variants"]:
        (b,) = v["bores"]
        assert b["name"] == "channel_top" and b["radial_interference_mm"] == 0.05 and b["diametral_interference_mm"] == 0.1 and b["facets"] >= 1       # the channel is 4.5 mm in radius: 1 wall facet at 3 mm
        assert b["hoop_max_MPa"] >= b["hoop_p99_9_MPa"] >= b["hoop_median_MPa"] >= b["hoop_min_MPa"] and b["n_qp"] >= 1 and b["vm_p99_9_MPa"] > 0
        assert b["contact_pressure_MPa"]["nodes"] == 3 and b["area_mm2"] > 0
        s = v["solve"]
        assert s["relative_residual"] < 1e-9 and s["backward_error"] < 1e-9 and s["refinement_steps"] == 3
        net, react = np.array(b["net_spring_force_N"]), np.array(s["clamp_reaction_N"])
        assert np.abs(net + react).max() < 1e-6 * max(np.abs(net).max(), np.abs(react).max(), 1.0)        # the spring force is balanced by the clamps
    assert soft["bores"][0]["k_mpa_per_mm"] < rigid["bores"][0]["k_mpa_per_mm"]
    assert soft["bores"][0]["hoop_p99_9_MPa"] < rigid["bores"][0]["hoop_p99_9_MPa"]                  # a softer cup: a lower hoop stress
    text = " ".join(pf["facts"])
    assert "not a stress measure" in text and "solved with no cup spring" in text and "diametral value of the YAML" in text and pf["options"]
    assert pf["model"]["facet_quadrature_order"] == 4 and pf["seconds"] > 0


def test_sm_of_the_goodman_rows_is_the_hoop_stress_of_the_variant_and_there_is_no_row_with_sm_zero(runs):
    from topopt.verify.fatigue import goodman_point
    run, code, text, err = runs["two"]
    res = result(run)
    pf, g = res["pressfit"], res["goodman"]
    assert g["status"] == "done" and g["mean_stress"]["from"] == "press_fit" and g["missing_rows"] == []
    assert [r["id"] for r in g["rows"]] == [f"pla/{v}/channel_top/{s}" for v in ("rigid", "soft") for s in ("p99_9", "max")]
    hoop = {v["name"]: v["bores"][0] for v in pf["variants"]}
    for r in g["rows"]:
        assert r["sm_MPa"] == hoop[r["variant"]][f"hoop_{r['statistic']}_MPa"] != 0.0
        sa = {c["case"]: c["sa_MPa"] for c in r["cases"]}
        assert sa == {c: g["service_stress"][c]["channel_top"][f"vm_{r['statistic']}_MPa"] for c in sa}      # sa is the service stress: solved with no cup spring
        for c in r["cases"]:
            n, y = goodman_point(c["sa_MPa"], r["sm_MPa"], 55.0, 45.0, 15.0)
            assert c["n_goodman"] == n and c["peak_over_yield"] == y
    assert g["lowest_n"]["value"] == min(r["n_goodman"] for r in g["rows"] if r["n_goodman"] is not None)


def test_the_press_fit_stage_changes_no_number_of_the_run_and_adds_only_its_keys(runs):
    a, t = result(runs["absent"][0]), result(runs["two"][0])
    assert set(t) - set(a) == NEW_KEYS and set(a) <= set(t)
    assert set(t["timing_s"]) - set(a["timing_s"]) == {"bore_s", "goodman_s", "pressfit_s"} and set(a["timing_s"]) <= set(t["timing_s"])
    na, nt = dict(_numbers(a)), dict(_numbers(t))
    assert {k: v for k, v in nt.items() if k[0] not in NEW_KEYS and k[:1] + k[2:3] != ("estimates_before_meshing", "pressfit")} == na      # bit for bit
    assert t["cases"] == a["cases"]


def test_with_the_flag_off_the_block_with_a_press_fit_gives_the_result_of_the_run_with_no_block(runs):
    a, b = result(runs["absent"][0]), result(runs["flag_off"][0])
    assert runs["flag_off"][1] == 0 and set(a) == set(b) and not (NEW_KEYS & set(b)) and set(a["timing_s"]) == set(b["timing_s"])
    assert dict(_numbers(a)) == dict(_numbers(b))
    lines = (runs["flag_off"][2] + "\n" + runs["flag_off"][3]).splitlines()      # the line prefixes of the checks, not the word anywhere (the run path can hold "bore")
    assert not any(ln.strip().startswith(("bores:", "bore ", "bore checks:", "press fit", "pressfit", "goodman")) for ln in lines)


def test_the_press_fit_block_alone_has_no_goodman_key(runs):
    run, code, text, err = runs["alone"]
    assert code == 0, err
    res = result(run)
    assert res["bore_checks"]["checks"] == ["pressfit"] and res["pressfit"]["status"] == "done" and "goodman" not in res
    assert set(res) - set(result(runs["absent"][0])) == {"bore_checks", "bore_facts", "pressfit"}
    assert "goodman_s" not in res["timing_s"] and res["timing_s"]["pressfit_s"] > 0
    assert "pressfit (1 cup variant(s)" in text and "goodman" not in text.replace("pressfit", "")


def test_one_factor_is_alive_at_a_time_and_the_static_factor_is_freed_before_the_stage(runs):
    f = runs["factors"]
    assert f["made"] == [f["made"][0]] * 3 and len(f["made"]) == 3          # the static factor and one factor for each cup variant, all the same solver
    assert f["peak"] == 1 and f["live"] == 0


def test_the_memory_record_has_the_numbers_of_the_decision_and_the_peaks(runs):
    res = result(runs["two"][0])
    m = res["pressfit"]["memory"]
    assert m["extra_bytes"] > 0 and m["estimate_without_stage_bytes"] == res["memory"]["estimate_after_meshing_bytes"]
    assert m["estimate_with_stage_bytes"] == m["estimate_without_stage_bytes"] + m["extra_bytes"] and m["limit_bytes"] is not None and m["limit_source"]
    assert m["estimate_with_stage_bytes"] <= m["limit_bytes"] and "nbytes" in m["model"]
    assert m["peak_rss_after_stage_bytes"] >= m["peak_rss_before_stage_bytes"] > 0
    assert res["memory"]["peak_rss_bytes"] >= m["peak_rss_after_stage_bytes"]


def test_the_text_has_a_line_for_each_variant_and_the_progress_lines(runs):
    run, code, text, err = runs["two"]
    assert "pressfit (2 cup variant(s), solver " in text
    assert "  rigid / channel_top: k 1e+06 MPa/mm, hoop p99.9 " in text and "  soft / channel_top: k " in text and "contact pressure area mean " in text
    line = next(ln for ln in text.splitlines() if ln.startswith("  rigid / channel_top: k "))                  # the median next to p99.9 and max
    hoop = result(run)["pressfit"]["variants"][0]["bores"][0]
    assert f"hoop p99.9 {hoop['hoop_p99_9_MPa']:.4g} MPa, max {hoop['hoop_max_MPa']:.4g} MPa, median {hoop['hoop_median_MPa']:.4g} MPa; contact pressure" in line
    assert "  fact: Rule: p = k (d_r - u_r) per unit of area" in text and "  option: A smaller mesh size" in text
    assert "pla/rigid/channel_top/p99_9: worst case " in text and "pla/soft/channel_top/max: worst case " in text and "no row for" not in text
    assert "  press fit: memory estimate with the stage " in err and "  press fit rigid: channel_top: hoop p99.9 " in err and "  press fit: done, 2 variant(s)" in err


def test_the_combined_stress_of_each_case_and_variant_is_in_the_record_and_the_text(runs):
    """Plan 6.8: the stress of a load case plus the stress of the press fit at the sample points; the sum of the maxima is the bound of the maximum of the sum."""
    run, code, text, err = runs["two"]
    res = result(run)
    pf, g = res["pressfit"], res["goodman"]
    cases = list(g["service_stress"])
    assert len(cases) >= 2 and list(pf["combined"]) == ["rigid", "soft"]
    var = {v["name"]: v["bores"][0] for v in pf["variants"]}
    for name, by_bore in pf["combined"].items():
        assert list(by_bore) == ["channel_top"] and list(by_bore["channel_top"]) == cases
        for case, c in by_bore["channel_top"].items():
            assert set(c) == {"vm_p99_9_MPa", "vm_max_MPa", "hoop_p99_9_MPa", "hoop_max_MPa"} and c["vm_max_MPa"] >= c["vm_p99_9_MPa"] > 0
            assert c["vm_max_MPa"] <= g["service_stress"][case]["channel_top"]["vm_max_MPa"] + var[name]["vm_max_MPa"] + 1e-9
            assert c["vm_max_MPa"] != var[name]["vm_max_MPa"]                          # the case adds its stress to the press fit
    assert any(t.startswith("combined is the stress of a load case plus the stress of the press fit") for t in pf["facts"])
    assert f"  combined rigid / channel_top / {cases[0]}: von Mises p99.9 " in text and f"  combined soft / channel_top / {cases[-1]}: von Mises p99.9 " in text
    assert "the sum is not a solve of the assembled problem" in text


def test_the_press_fit_block_alone_still_has_the_combined_stress_of_the_cases(runs):
    """No Goodman key, so no `service` stress of the run; the stress of the cases at the sample points is still collected for `combined`."""
    run, code, text, err = runs["alone"]
    res = result(run)
    cases = list(result(runs["two"][0])["goodman"]["service_stress"])
    assert "goodman" not in res and list(res["pressfit"]["combined"]["rigid"]["channel_top"]) == cases
    two = result(runs["two"][0])["pressfit"]["combined"]["rigid"]["channel_top"]
    alone = res["pressfit"]["combined"]["rigid"]["channel_top"]
    for case in cases:
        assert alone[case]["vm_max_MPa"] == pytest.approx(two[case]["vm_max_MPa"], rel=1e-9)         # the press fit of the variant is the same in both runs


def test_the_combined_stress_is_absent_when_the_stage_is_skipped(tmp_path, monkeypatch):
    from topopt.verify import pressfit
    monkeypatch.setattr(pressfit, "csr_bytes", lambda A: 10 ** 15)
    run = copy_run(tmp_path / "run")
    code, text, err = cli(run, yaml_with(tmp_path, block(press_fit=PRESS)))
    assert code == 0, err
    assert "combined" not in result(run)["pressfit"] and "combined " not in text.replace("combined stress", "")


def test_a_bore_whose_points_raise_is_dropped_and_the_stage_and_the_goodman_check_go_on(tmp_path, monkeypatch):
    """The service points of the press fit are read inside the guarded loop of the bores (review m-5): a bore with an exception is dropped with the reason `error`, it has no
    value for the earlier cases either, and the other bore and `topopt verify` go on."""
    from test_verify_bore_run import CHANNEL
    import topopt.verify.bore as vbore
    real = vbore.build

    def build(model, bc):
        bores = real(model, bc)
        for b in bores:
            if b.ok and b.spec.name == "second":
                def boom(sig):
                    raise FloatingPointError("a made up failure")
                b.sample.points = boom
        return bores
    monkeypatch.setattr(vbore, "build", build)
    second = dict(CHANNEL, name="second")
    run = copy_run(tmp_path / "run")
    code, text, err = cli(run, yaml_with(tmp_path, block(bores=(CHANNEL, second), press_fit=PRESS)))
    assert code == 0, err
    res = result(run)
    pf, g, facts = res["pressfit"], res["goodman"], {e["name"]: e for e in res["bore_facts"]}
    assert facts["second"]["status"] == "error" and facts["second"]["stage"] == "stats" and "a made up failure" in facts["second"]["message"]
    assert pf["status"] == "done" and pf["dropped_bores"] == ["second"] and [b["name"] for b in pf["variants"][0]["bores"]] == ["channel_top"]
    assert list(pf["combined"]["rigid"]) == ["channel_top"] and any("second (error)" in t for t in pf["facts"])
    assert g["status"] == "done" and {r["bore"] for r in g["rows"]} == {"channel_top"} and all("second" not in v for v in g["service_stress"].values())
    assert [(m["bore"], m["reason"]) for m in g["missing_rows"]] == [("second", "error")]
    assert "bore second stopped" in err and res["ok"] is True


def test_the_stage_is_skipped_when_it_does_not_fit_and_the_run_goes_on(tmp_path, monkeypatch):
    """The memory decision of the stage (plan 6.6): above the limit it is skipped with the facts and the options, the Goodman rows of the cup variants are missing."""
    from topopt.verify import pressfit
    monkeypatch.setattr(pressfit, "csr_bytes", lambda A: 10 ** 15)
    run = copy_run(tmp_path / "run")
    code, text, err = cli(run, yaml_with(tmp_path, block(press_fit=PRESS)))
    assert code == 0, err
    res = result(run)
    pf, g = res["pressfit"], res["goodman"]
    assert pf["status"] == "skipped" and pf["reason"] == "memory" and pf["variants"] == [] and pf["memory"]["extra_bytes"] >= 10 ** 15
    assert "above the limit" in pf["facts"][0] and any("--max-gb" in o for o in pf["options"]) and any("--force" in o for o in pf["options"])
    assert g["status"] == "done" and g["rows"] == [] and g["missing_rows"] == [dict(variant="rigid", bore="channel_top", reason="memory", options=pf["options"])]
    assert any("a row with sm = 0 would describe a part with no press fit" in t for t in g["facts"])
    assert "pressfit: skipped (memory)" in text and "no row for rigid / channel_top: memory" in text
    assert "above the limit, the stage is skipped" in err and res["ok"] is True and res["exit_code"] == 0


def test_with_force_the_stage_runs_above_the_limit(tmp_path, monkeypatch):
    from topopt.verify import pressfit
    monkeypatch.setattr(pressfit, "csr_bytes", lambda A: 10 ** 15)
    run = copy_run(tmp_path / "run")
    code, text, err = cli(run, yaml_with(tmp_path, block(press_fit=PRESS)), "--force")
    assert code == 0, err
    pf = result(run)["pressfit"]
    assert pf["status"] == "done" and pf["memory"]["extra_bytes"] >= 10 ** 15 and pf["memory"]["estimate_with_stage_bytes"] > pf["memory"]["limit_bytes"]


def test_a_failure_of_the_stage_is_a_fact_and_the_run_goes_on(tmp_path, monkeypatch):
    from topopt.verify import pressfit

    def boom(*a, **k):
        raise np.linalg.LinAlgError("a made up failure")
    monkeypatch.setattr(pressfit, "run_stage", boom)
    run = copy_run(tmp_path / "run")
    code, text, err = cli(run, yaml_with(tmp_path, block(press_fit=PRESS)))
    assert code == 0, err
    res = result(run)
    pf, g = res["pressfit"], res["goodman"]
    assert pf["status"] == "error" and pf["reason"] == "exception" and "LinAlgError: a made up failure" in pf["message"] and pf["variants"] == []
    assert g["status"] == "done" and g["rows"] == [] and g["missing_rows"][0]["reason"] == "exception" and res["ok"] is True
    assert "press fit stopped: LinAlgError: a made up failure" in err and "pressfit: error (exception)" in text
    assert "pressfit_s" in res["timing_s"]


def test_several_levels_run_the_stage_on_every_level_and_report_the_change(tmp_path):
    from topopt.verify.report import summary_lines
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    res = verify_run(run, yaml_path=yaml_with(tmp_path, block(press_fit=PRESS)), levels=[3.0, 2.5], load_sensitivity=False)
    assert res["reference_level_mm"] == 2.5 and res["pressfit"]["status"] == "done" and res["goodman"]["status"] == "done"
    lv = res["levels"]
    key = "rigid/channel_top"
    hoop = res["pressfit"]["variants"][0]["bores"][0]["hoop_p99_9_MPa"]
    assert [set(r["pressfit_hoop_p99_9_MPa"]) for r in lv["rows"]] == [{key}, {key}] and lv["rows"][1]["pressfit_hoop_p99_9_MPa"][key] == hoop
    first = lv["rows"][0]["pressfit_hoop_p99_9_MPa"][key]
    assert lv["relative_change"][0]["pressfit_hoop_p99_9_MPa"][key] == pytest.approx((hoop - first) / first, rel=1e-12)
    assert lv["relative_change_first_to_last"]["pressfit_hoop_p99_9_MPa"].keys() == {key}
    ids = [r["id"] for r in res["goodman"]["rows"]]
    assert set(lv["rows"][0]["goodman_n"]) == set(ids)
    assert any(x.startswith("pressfit (1 cup variant(s)") for x in summary_lines(res))


def test_the_estimate_has_the_press_fit_keys_when_the_block_is_there(tmp_path):
    from topopt.verify import memory
    from topopt.verify.report import estimate_lines
    from topopt.verify.run import verify_run
    run = copy_run(tmp_path / "run")
    y = yaml_with(tmp_path, block(press_fit=TWO_CUPS))
    res = verify_run(run, yaml_path=y, mesh_size=SIZE, estimate_only=True)
    (e,) = res["estimates_before_meshing"]
    pf = e["pressfit"]
    assert pf["variants"] == pf["factorizations"] == 2 and pf["estimate_bytes"] == e["estimate_bytes"] and pf["note"]
    solver = res["backend"]["solver"]
    fs = memory.factor_seconds(e["estimated_dofs"], solver, e["compactness"])
    want = None if fs is None else 2 * (fs + memory.TIME_CASE_PER_1E5_S * e["estimated_dofs"] / 1e5)
    assert pf["estimated_seconds"] == want
    assert any(x.strip().startswith("with the press-fit check: 2 cup variant(s), one factor for each, one at a time") for x in estimate_lines(res))
    for kw in ({"bore_checks": False},):
        assert "pressfit" not in verify_run(run, yaml_path=y, mesh_size=SIZE, estimate_only=True, **kw)["estimates_before_meshing"][0]
    assert "pressfit" not in verify_run(run, yaml_path=yaml_with(tmp_path, block(), "g.yaml"), mesh_size=SIZE, estimate_only=True)["estimates_before_meshing"][0]
    assert "pressfit" not in verify_run(run, yaml_path=HOOK_YAML, mesh_size=SIZE, estimate_only=True)["estimates_before_meshing"][0]
    assert {k: v for k, v in e.items() if k != "pressfit"} == {k: v for k, v in verify_run(run, yaml_path=yaml_with(tmp_path, block(), "g.yaml"), mesh_size=SIZE,
                                                                                         estimate_only=True)["estimates_before_meshing"][0].items()}
