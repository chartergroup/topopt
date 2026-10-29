"""Tests for the part-to-topopt skill helper."""
import importlib.util
import re
from pathlib import Path

import pytest
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec_ = importlib.util.spec_from_file_location(
    "build_regions", ROOT / ".claude/skills/part-to-topopt/build_regions.py")
br = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(br)

SPEC = {
    "name": "shelf_bracket",
    "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [60, 20, 40]}},
                   {"subtract": {"sphere": {"center": [60, 10, 40], "radius": 5}}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 40]}}]},
        "keep_out": {"hole": [{"cylinder": {"p0": [-5, 10, 30], "p1": [5, 10, 30], "radius": 2}}]},
        "load": {"shelf": [{"capsule": {"p0": [50, 10, 8], "p1": [58, 10, 8], "radius": 3}},
                           {"box": {"min": [48, 4, 0], "max": [60, 16, 6]}}]},
    },
    "load_cases": [{"name": "static", "weight": 1.0,
                    "forces": [{"region": "shelf", "force": [0, 0, -30.0]}]}],
    "settings": {"volume_fraction": 0.3, "voxel_size": 2.0},
}


# The old budget words. The output of the builder also holds absolute paths ("wrote <path>"), so a bare "demo" in it can come from the
# folder of the checkout or of the test (P9-8 review N3): the asserts look for the words of the message.
OLD_BUDGET = re.compile(r"demo( time)?[ -]budget|conservative (time )?estimate", re.IGNORECASE)


def write(tmp_path, spec):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p


def test_build_and_check(tmp_path, capsys):
    out = tmp_path / "out"
    assert br.main([str(write(tmp_path, SPEC)), "--out", str(out)]) == 0
    for f in ("design", "wall", "hole", "shelf"):
        assert trimesh.load(out / f"{f}.stl").is_watertight
    doc = yaml.safe_load((out / "shelf_bracket.yaml").read_text())
    assert doc["regions"]["load"] == {"shelf": "shelf.stl"}
    assert doc["regions"]["keep_out"] == ["hole.stl"]
    assert "RESULT: OK" in capsys.readouterr().out


def test_check_mode_writes_nothing(tmp_path):
    out = tmp_path / "out"
    assert br.main([str(write(tmp_path, SPEC)), "--out", str(out), "--check"]) == 0
    assert not out.exists() or not list(out.iterdir())


def test_auto_voxel(tmp_path):
    s = dict(SPEC, settings={"voxel_size": "auto"})
    out = tmp_path / "out"
    assert br.main([str(write(tmp_path, s)), "--out", str(out)]) == 0
    doc = yaml.safe_load((out / "shelf_bracket.yaml").read_text())
    assert 0.6 <= doc["settings"]["voxel_size"] <= 3.0


def test_disconnected_load_fails(tmp_path, capsys):
    s = yaml.safe_load(yaml.safe_dump(SPEC))
    s["regions"]["load"] = {"shelf": [{"box": {"min": [100, 0, 0], "max": [110, 10, 10]}}]}
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1
    assert "separate pieces" in capsys.readouterr().out


def test_empty_load_region_fails(tmp_path, capsys):
    s = yaml.safe_load(yaml.safe_dump(SPEC))
    s["regions"]["load"] = {"shelf": [{"box": {"min": [50, 10, 8], "max": [50.1, 10.1, 8.1]}}]}
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1
    assert "ERROR" in capsys.readouterr().out


def test_too_many_voxels_fails(tmp_path, capsys):
    s = yaml.safe_load(yaml.safe_dump(SPEC))
    s["settings"]["voxel_size"] = 0.6
    s["regions"]["design"] = [{"box": {"min": [0, 0, 0], "max": [200, 100, 100]}}]
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1


def test_bad_spec(tmp_path, capsys):
    s = yaml.safe_load(yaml.safe_dump(SPEC))
    s["regions"]["design"] = [{"torus": {}}]
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1
    assert "unknown primitive" in capsys.readouterr().out


def test_bowl_and_prism_closed():
    b = br.to_trimesh(br.make_primitive("bowl", {"center": [0, 0, 0], "radius": 20, "wall": 3,
                                                 "open": [1, 0, 0]}))
    assert b.is_watertight and b.volume > 0 and b.bounds[1][0] < 1  # cup opens to +x, shell on -x side
    p = br.to_trimesh(br.make_primitive("prism", {"plane": "xz", "extent": [-5, 5],
                                                  "points": [[0, 0], [10, 0], [0, 10]]}))
    assert p.is_watertight and abs(p.volume - 500) < 1


def clone(**over):
    s = yaml.safe_load(yaml.safe_dump(SPEC))
    for k, v in over.items():
        s[k] = v
    return s


def collect(lines):
    """A `say` function for the builder: it joins the arguments of one report line."""
    return lambda *a: lines.append(" ".join(map(str, a)))


def _counts(text):
    """The voxel counts of the report line `voxels:` and the `active:` line."""
    import re
    n = {k: int(v) for k, v in re.findall(r"(design|fixed|load\+solid)=(\d+)", text)}
    return n["fixed"] + n["load+solid"], n["design"]


def test_volume_fraction_is_free_fraction(tmp_path, capsys):
    """P1 (F3-2): the YAML keeps the volume fraction of the spec, the mean density of the FREE design voxels.

    The builder wrote (frozen + vf * free) / active before. The engine applies the YAML value to the free
    voxels only (`volume_constraint`), so a run got more material than the spec asked for.
    """
    out = tmp_path / "out"
    assert br.main([str(write(tmp_path, SPEC)), "--out", str(out)]) == 0
    yml = out / "shelf_bracket.yaml"
    doc = yaml.safe_load(yml.read_text())
    text = capsys.readouterr().out
    frozen, free = _counts(text)
    old = (frozen + 0.3 * free) / (frozen + free)
    assert frozen > 0 and abs(old - 0.3) > 0.03          # the spec has forced-solid voxels: the old number differs
    assert doc["settings"]["volume_fraction"] == 0.3
    assert f"{old:.4f}" not in text.split("target:")[0] and f"{old:.5f}" not in yml.read_text()
    # The report states the basis once, then the target of the run and the share (information only).
    assert text.count("is the mean density of the") == 1 and f"the {free} free design voxels" in text
    assert f"the {frozen} forced-solid voxels are solid on top (the YAML keeps 0.3)" in text
    assert "engine value" not in text and "written to the YAML" not in text
    h = doc["settings"]["voxel_size"]
    target_mm3 = (frozen + 0.3 * free) * h ** 3
    assert f"= {target_mm3:.0f} mm3, {target_mm3 / 1000 * 1.24:.1f} g at 1.24 g/cm3" in text
    assert f"the target is {old:.4f} of the {frozen + free} active voxels (for information; written nowhere)" in text


def test_report_target_agrees_with_check_and_the_derived_numbers(tmp_path):
    """The target volume of the report, of `topopt check` and of the `derived` numbers is one number."""
    from topopt.cli import check_data
    from topopt.regions_builder import build_spec
    lines = []
    r = build_spec(clone(settings={"volume_fraction": 0.3, "voxel_size": 2.0}), tmp_path / "o", say=collect(lines))
    assert r["ok"]
    vol = check_data(r["yaml"])["derived"]["volume"]
    assert vol["volume_fraction"] == 0.3
    text = "\n".join(lines)
    assert f"= {vol['target_volume_mm3']:.0f} mm3, {vol['target_mass_g']:.1f} g at" in text
    assert f"the {vol['free_design_voxels']} free design voxels" in text


def test_a_volume_fraction_of_1_builds_and_a_larger_one_is_a_spec_error(tmp_path):
    """P1 (F3-2): with the old conversion a fraction of 1 gave the engine value 1.0, and the builder stopped with
    "the engine volume fraction is 1 or more". The engine accepts a fraction up to 1 (a stress study uses it), so the run
    works and it is no error. A fraction above 1 stays an error: `load_config` states the limit."""
    lines = []
    r = build_spec_for(clone(settings={"volume_fraction": 1.0, "voxel_size": 2.0}), tmp_path / "one", lines)
    assert r["ok"] and yaml.safe_load(Path(r["yaml"]).read_text())["settings"]["volume_fraction"] == 1.0
    assert not any(ln.startswith("ERROR") for ln in lines)
    lines = []
    r = build_spec_for(clone(settings={"volume_fraction": 1.5, "voxel_size": 2.0}), tmp_path / "over", lines)
    assert not r["ok"] and any("'settings.volume_fraction' must be <= 1.0, got 1.5" in ln for ln in lines)


def build_spec_for(spec, out, lines):
    from topopt.regions_builder import build_spec
    return build_spec(spec, out, say=collect(lines))


def test_the_ladle_spec_builds_the_pinned_ladle_yaml(tmp_path):
    """P1 review (L4): `examples/ladle/ladle_spec.yaml` has the value of the pinned `examples/ladle/ladle.yaml` (0.10628), and the builder keeps it,
    so the spec builds that YAML (same settings, regions, load cases and material). The old builder turned the spec value 0.06 into 0.10628."""
    from topopt.regions_builder import build_spec
    spec = yaml.safe_load((ROOT / "examples/ladle/ladle_spec.yaml").read_text())
    assert spec["settings"]["volume_fraction"] == 0.10628
    r = build_spec(spec, tmp_path / "o", say=lambda *a: None)
    assert r["ok"]
    built = yaml.safe_load(Path(r["yaml"]).read_text())
    pinned = yaml.safe_load((ROOT / "examples/ladle/ladle.yaml").read_text())
    assert built["settings"] == pinned["settings"] == {"voxel_size": 3.25, "volume_fraction": 0.10628, "iterations": 80}
    assert built == pinned


def test_report_target_volume_is_what_a_short_run_reaches(tmp_path):
    """P1 (F3-2): a short CPU run reaches the target volume of the report within the projection tolerance.

    On d19c7fb the YAML held (frozen + vf * free) / active = 0.3784 at this voxel size (3 mm), and this run reached a design
    mean of 0.37789 for 0.3: a run volume of 24060 mm3 for 20342 mm3 asked, 18.3 percent too high (measured, 15 iterations).
    The tolerance is the one of `test_cantilever_converges`: the design mean is within 1 percent of the fraction.
    """
    import json
    import re

    from topopt.config import load_config
    from topopt.optimize import run
    from topopt.problem import build_problem
    from topopt.regions_builder import build_spec
    from topopt.snapshot import RunWriter
    from topopt.voxelize import DESIGN
    lines = []
    r = build_spec(clone(settings={"volume_fraction": 0.3, "voxel_size": 3.0, "iterations": 15}),
                   tmp_path / "o", say=collect(lines))
    assert r["ok"]
    prob = build_problem(load_config(r["yaml"]))
    n_free = int((prob.labels == DESIGN).sum())
    n_frozen = int(((prob.labels > DESIGN) & (prob.labels < 4)).sum())
    asked = (n_frozen + 0.3 * n_free) * prob.h ** 3          # what the spec asks for, in mm3
    writer = RunWriter(tmp_path / "run", quiet=True)
    writer.clean()
    run(prob, writer)
    last = [json.loads(ln) for ln in (tmp_path / "run" / "history.jsonl").read_text().splitlines()][-1]
    assert last["volume_fraction"] == pytest.approx(0.3, rel=0.01)
    reached = (n_frozen + last["volume_fraction"] * n_free) * prob.h ** 3
    assert reached == pytest.approx(asked, rel=0.01)
    assert float(re.search(r"= (\d+) mm3,", "\n".join(lines)).group(1)) == pytest.approx(reached, rel=0.01)


BRIDGED = {
    "name": "bridged",
    "regions": {
        "design": [{"box": {"min": [4, 0, 0], "max": [24, 10, 10]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 10, 10]}}]},
        "solid": {"bridge": [{"box": {"min": [0, 0, 0], "max": [4, 10, 10]}}]},
        "load": {"tip": [{"box": {"min": [20, 3, 6], "max": [24, 7, 10]}}]},
    },
    "load_cases": [{"name": "down", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -5.0]}]}],
    "settings": {"volume_fraction": 0.3, "voxel_size": 1.0, "iterations": 5},
}


def test_fixed_region_that_reaches_the_design_through_a_solid_body_is_a_warning(tmp_path, capsys):
    """P1 (F2-3): the fixed region shares no face with the design space, but a solid bridge joins them. The run works, so the
    builder gives a WARNING with the fact and the options (it was an ERROR and RESULT: FAILED on d19c7fb), and so does `check`."""
    from topopt.cli import check_data
    out = tmp_path / "out"
    assert br.main([str(write(tmp_path, BRIDGED)), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "ERROR" not in text and "RESULT: OK" in text and "the fixed region does not touch" not in text
    warn = [ln for ln in text.splitlines() if ln.startswith("WARNING: no fixed voxel touches a design voxel")]
    assert len(warn) == 1 and "Options: overlap the design space" in warn[0]
    assert "Every design voxel is in a piece that holds a support" in warn[0]
    assert warn[0][len("WARNING: "):] in check_data(out / "bridged.yaml")["warnings"]          # one rule, one text


def test_fixed_region_far_from_the_rest_is_an_error_only_because_the_run_cannot_work(tmp_path, capsys):
    """The pieces are the error (the design has no support, `topopt run` stops). The touch fact is a warning beside it."""
    s = yaml.safe_load(yaml.safe_dump(BRIDGED))
    s["regions"]["fixed"] = {"wall": [{"box": {"min": [-20, 0, 0], "max": [-16, 10, 10]}}]}
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1
    text = capsys.readouterr().out
    assert "ERROR: the active voxels make 2 separate pieces" in text and "RESULT: FAILED" in text
    assert "ERROR: no fixed voxel touches" not in text and "the fixed region does not touch" not in text
    assert "WARNING: no fixed voxel touches a design voxel" in text and "Some design voxels are in a piece with no support" in text


TWO_WALLS = {
    "name": "two_walls",
    "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [20, 10, 10]}}, {"box": {"min": [0, 30, 0], "max": [20, 40, 10]}}],
        "fixed": {"w1": [{"box": {"min": [-4, 0, 0], "max": [0, 10, 10]}}], "w2": [{"box": {"min": [-4, 30, 0], "max": [0, 40, 10]}}]},
        "load": {"t1": [{"box": {"min": [16, 3, 6], "max": [20, 7, 10]}}], "t2": [{"box": {"min": [16, 33, 6], "max": [20, 37, 10]}}]},
    },
    "load_cases": [{"name": "down", "weight": 1.0,
                    "forces": [{"region": "t1", "force": [0, 0, -5.0]}, {"region": "t2", "force": [0, 0, -5.0]}]}],
    "settings": {"volume_fraction": 0.3, "voxel_size": 2.0, "iterations": 2},
}


def test_separate_pieces_that_each_hold_a_support_are_a_warning_and_the_run_works(tmp_path, capsys):
    """P1 review (M2): two walls, each with its own piece of the design. `topopt run` solves it (each piece holds a support), `topopt check`
    warns, so the builder warns too, with the text of `check`. It was `ERROR: ... separate pieces` and `RESULT: FAILED` before."""
    from topopt import connectivity
    from topopt.cli import check_data
    from topopt.config import load_config
    from topopt.optimize import run
    from topopt.problem import build_problem
    from topopt.snapshot import RunWriter
    out = tmp_path / "o"
    assert br.main([str(write(tmp_path, TWO_WALLS)), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "ERROR" not in text and "RESULT: OK" in text
    [line] = [ln for ln in text.splitlines() if "separate pieces" in ln]
    assert line.startswith("WARNING: the active voxels make 2 separate pieces") and "each holds a fixed voxel" in line
    assert "stops with this error" not in line
    assert line[len("WARNING: "):] in check_data(out / "two_walls.yaml")["warnings"]       # one text in both tools
    prob = build_problem(load_config(out / "two_walls.yaml"))
    assert connectivity.floating_error(prob) is None
    writer = RunWriter(tmp_path / "run", quiet=True)
    writer.clean()
    run(prob, writer)                                                                       # the run works
    assert len((tmp_path / "run" / "history.jsonl").read_text().splitlines()) == TWO_WALLS["settings"]["iterations"] + 1


def test_a_piece_with_no_support_is_the_pieces_error_with_the_end_of_check(tmp_path, capsys):
    """The same two pieces, but the second wall is taken away: the second piece has no support, `topopt run` stops, so it is an ERROR."""
    s = yaml.safe_load(yaml.safe_dump(TWO_WALLS))
    del s["regions"]["fixed"]["w2"]
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1
    text = capsys.readouterr().out
    [line] = [ln for ln in text.splitlines() if "separate pieces" in ln]
    assert line.startswith("ERROR: the active voxels make 2 separate pieces") and "1 of them has no fixed voxel" in line
    assert line.endswith("(`topopt run` stops with this error)") and "RESULT: FAILED" in text


def test_solid_role_writes_regions_solid(tmp_path):
    """A `solid` body goes to `regions.solid` (CONTRACTS.md section 22a), not to `regions.load`."""
    s = clone()
    s["regions"]["solid"] = {"cap": [{"box": {"min": [30, 4, 30], "max": [42, 16, 40]}}]}
    out = tmp_path / "out"
    assert br.main([str(write(tmp_path, s)), "--out", str(out)]) == 0
    doc = yaml.safe_load((out / "shelf_bracket.yaml").read_text())
    assert doc["regions"]["load"] == {"shelf": "shelf.stl"}
    assert doc["regions"]["solid"] == ["cap.stl"]
    assert all(f["region"] == "shelf" for c in doc["load_cases"] for f in c["forces"])
    from topopt.config import load_config
    assert [p.name for p in load_config(out / "shelf_bracket.yaml").regions.solid] == ["cap.stl"]


def test_no_solid_writes_no_solid_key(tmp_path):
    out = tmp_path / "out"
    assert br.main([str(write(tmp_path, SPEC)), "--out", str(out)]) == 0
    assert "solid" not in yaml.safe_load((out / "shelf_bracket.yaml").read_text())["regions"]


def test_a3a_settings_pass_through(tmp_path):
    s = clone()
    s["settings"].update({"min_member_mm": 5.0, "filter_solid_neighbours": "ignore"})
    out = tmp_path / "out"
    assert br.main([str(write(tmp_path, s)), "--out", str(out)]) == 0
    st = yaml.safe_load((out / "shelf_bracket.yaml").read_text())["settings"]
    assert st["min_member_mm"] == 5.0 and st["filter_solid_neighbours"] == "ignore"


def test_small_solid_region_warns(tmp_path, capsys):
    s = clone()
    s["regions"]["solid"] = {"cap": [{"box": {"min": [30, 4, 30], "max": [34, 8, 34]}}]}
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "WARNING: region 'cap' is small or thin" in out and "RESULT: OK" in out


def test_load_region_without_force_fails(tmp_path, capsys):
    s = clone()
    s["regions"]["load"]["extra"] = [{"box": {"min": [30, 4, 30], "max": [42, 16, 40]}}]
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1
    assert "no force" in capsys.readouterr().out


def test_small_load_region_warns(tmp_path, capsys):
    s = clone()
    s["regions"]["load"] = {"shelf": [{"box": {"min": [50, 8, 6], "max": [54, 12, 10]}}]}
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "WARNING: region 'shelf' is small or thin" in out and "voxels" in out
    assert "RESULT: OK" in out


def test_too_much_solid_warns(tmp_path, capsys):
    s = clone()
    s["regions"]["solid"] = {"slab": [{"box": {"min": [0, 0, 0], "max": [60, 20, 38]}}]}
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "WARNING: forced-solid voxels are" in out and "percent" in out


def test_time_over_the_budget_of_the_user_warns_with_voxel_hint(tmp_path, capsys):
    """P9-8: the budget is the user's own, `settings.stop_rules.time_budget_s` (the spec copies it into the YAML). The warning is a fact."""
    s = clone()
    s["settings"]["voxel_size"] = 1.0
    s["settings"]["stop_rules"] = {"time_budget_s": 40}
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "WARNING: the estimate is above your time budget of 40 s (settings.stop_rules.time_budget_s)" in out
    assert "; the stop rules are on in this YAML (a stop_rules mapping turns them on), so the run stops at the first pass that ends after 40 s" in out
    assert "(rule time_budget; the rule converged is on too) and the part is the design of that pass; --no-stop-rules" in out
    assert "does not stop" not in out
    assert "(the model)" in out and "conservative" not in out.lower()       # E1-P3: the printed lines do not call the estimate conservative
    assert "voxel_size of about" in out and "RESULT: OK" in out and "retry with" not in out
    assert yaml.safe_load((tmp_path / "o" / f"{s['name']}.yaml").read_text())["settings"]["stop_rules"] == {"time_budget_s": 40}
    assert not OLD_BUDGET.search(out) and "conservative" not in out


def test_time_over_the_budget_with_the_stop_rules_off_says_that_the_run_does_not_stop(tmp_path, capsys):
    """P9-8 review MJ1: `enabled: false` keeps the budget in the YAML and the rules off, so the run does not stop at it."""
    s = clone()
    s["settings"]["voxel_size"] = 1.0
    s["settings"]["stop_rules"] = {"enabled": False, "time_budget_s": 40}
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert ("WARNING: the estimate is above your time budget of 40 s (settings.stop_rules.time_budget_s); the stop rules are off in this "
            "YAML (enabled: false), so the run does not stop at this budget; --stop-rules (MCP run_topopt(stop_rules=true)) turns them on "
            "for one run. A voxel_size of about ") in out
    assert "the run stops" not in out and "rule time_budget" not in out and "RESULT: OK" in out


def test_without_a_budget_of_the_user_the_time_is_a_fact_and_not_a_warning(tmp_path, capsys):
    """P9-8 (D5): no fixed budget. A run of any size gets the estimate in plain words and no budget warning."""
    s = clone()
    s["settings"]["voxel_size"] = 1.0                        # the model estimate here is above the old fixed 240 s
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith("time estimate:")]
    assert len(line) == 1 and "(the model); scaled by the fill " in line[0]
    assert "your time budget" not in out and not OLD_BUDGET.search(out) and "conservative" not in out and "240 s" not in out
    assert int(line[0].split("about ")[1].split(" s")[0]) > 240      # above the old fixed budget, and no warning
    assert "WARNING: the estimate" not in out and "RESULT: OK" in out


def test_target_active_override_is_honored(tmp_path):
    from topopt import regions_builder as rb
    shapes = rb.build_all(clone())
    h_default = rb.auto_voxel(shapes, 80)
    h_big = rb.auto_voxel(shapes, 80, target=rb.CAP_ACTIVE * 4)   # above the default cap
    assert h_big < h_default
    s = clone(settings={"volume_fraction": 0.3, "voxel_size": "auto", "target_active": 3000})
    out = tmp_path / "out"
    assert br.main([str(write(tmp_path, s)), "--out", str(out)]) == 0
    doc = yaml.safe_load((out / "shelf_bracket.yaml").read_text())
    assert doc["settings"]["voxel_size"] > h_default


def _ram(monkeypatch, nbytes):
    monkeypatch.setattr("topopt.cli.physical_ram_bytes", lambda: nbytes)


def test_ram_hard_block_and_override(tmp_path, capsys, monkeypatch):
    _ram(monkeypatch, 10_000_000)                    # 10 MB: the estimate is above it
    s = clone()
    s["settings"]["voxel_size"] = 1.0
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1
    out = capsys.readouterr().out
    assert "ERROR" in out and "RAM" in out
    assert "allow_large_grid" in out
    s["settings"]["allow_large_grid"] = True         # the documented override
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o2")]) == 0
    out = capsys.readouterr().out
    assert "RESULT: OK" in out


def test_ram_unknown_gives_no_block(tmp_path, capsys, monkeypatch):
    _ram(monkeypatch, None)
    s = clone()
    s["settings"]["voxel_size"] = 1.0
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 0
    assert "memory estimate: not available" in capsys.readouterr().out


def test_engine_guard_message_names_override(tmp_path, capsys):
    s = clone()
    s["settings"]["voxel_size"] = 0.2
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1
    assert "max_grid_voxels" in capsys.readouterr().out


def test_no_retry_for_separate_pieces(tmp_path, capsys):
    s = clone(settings={"volume_fraction": 0.3, "voxel_size": "auto"})
    s["regions"]["load"] = {"shelf": [{"box": {"min": [100, 0, 0], "max": [120, 20, 20]}}]}
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1
    out = capsys.readouterr().out
    assert "separate pieces" in out and "retry with" not in out


def test_time_calibration():
    assert br.time_factor(0.05) == pytest.approx(0.27)
    assert br.time_factor(0.40) == pytest.approx(1.0)
    assert br.target_active(0.05, 80) == br.CAP_ACTIVE          # slender: the cap binds
    assert 30000 < br.target_active(0.37, 80) < 40000           # compact: the time goal binds


def test_memory_model_matches_bench_rss():
    from topopt.cli import estimate_memory_bytes, gib
    bench = {65_067: 0.96, 97_344: 1.48, 129_360: 2.05, 159_300: 2.62, 190_464: 3.22, 253_575: 4.12}
    for dofs, gb in bench.items():
        est = estimate_memory_bytes(dofs // 3, dofs) / 1e9
        assert 0.85 * gb <= est <= 1.2 * gb, (dofs, est, gb)
    assert gib(16 * 2**30) == "16.0 GiB"


def test_allow_large_grid_must_be_boolean(tmp_path, capsys):
    s = clone()
    s["settings"]["allow_large_grid"] = "false"
    assert br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")]) == 1
    assert "allow_large_grid must be true or false" in capsys.readouterr().out


@pytest.mark.parametrize("solver", ["amg", "cholesky", "pardiso"])
def test_time_over_budget_is_a_warning_and_amg_adds_the_pardiso_estimate(tmp_path, capsys, monkeypatch, solver):
    """The budget of the user is a fact for every solver, not a stop. With amg the note gives the pardiso estimate."""
    from topopt import solver_choice   # the code of the check is in the package
    monkeypatch.setattr(solver_choice, "resolve_solver", lambda setting: solver)
    s = clone(settings={"volume_fraction": 0.3, "voxel_size": 1.5, "stop_rules": {"time_budget_s": 1}})   # the estimate is above 1 s for every solver
    code = br.main([str(write(tmp_path, s)), "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert code == 0 and "RESULT: OK" in out and "retry with" not in out
    assert "WARNING: the estimate is above your time budget of 1 s (settings.stop_rules.time_budget_s)" in out
    assert not OLD_BUDGET.search(out)
    if solver == "amg":
        assert "with amg" in out and "with pardiso" in out
    else:
        assert "with pardiso" not in out


# ---- I7 (review M1): a spec with `settings.target_mass_g` ----------------------------------------------------------------------
def test_a_spec_with_target_mass_g_writes_that_key_and_no_volume_fraction(tmp_path):
    """The builder kept only `volume_fraction` (default 0.25), so a mass that the user asked for was lost with no message."""
    from topopt.cli import check_data
    lines = []
    r = build_spec_for(clone(settings={"target_mass_g": 30.0, "voxel_size": 2.0}), tmp_path / "o", lines)
    assert r["ok"]
    st = yaml.safe_load(Path(r["yaml"]).read_text())["settings"]
    assert st["target_mass_g"] == 30.0 and "volume_fraction" not in st
    text = "\n".join(lines)
    assert "volume_fraction: " in text and "is derived from settings.target_mass_g 30 g" in text
    vol = check_data(r["yaml"])["derived"]["volume"]
    assert vol["target_mass_asked_g"] == 30.0 and vol["target_mass_g"] == pytest.approx(30.0, abs=0.01)


def test_a_spec_with_both_volume_keys_is_a_spec_error_with_the_text_of_the_loader(tmp_path):
    from topopt.regions_builder import SpecError, build_in_workspace, build_spec
    s = clone(settings={"target_mass_g": 30.0, "volume_fraction": 0.3, "voxel_size": 2.0})
    with pytest.raises(SpecError) as exc:
        build_spec(s, tmp_path / "o")
    assert "has both volume_fraction and target_mass_g" in str(exc.value) and "Options:" in str(exc.value)
    assert "The words of the user say which one they asked for" in str(exc.value)
    assert not (tmp_path / "o").exists()          # nothing was written
    r = build_in_workspace(s, tmp_path / "ws")
    assert r["ok"] is False and "has both volume_fraction and target_mass_g" in r["error"] and "Options:" in r["error"]


def test_the_loader_and_the_builder_say_the_same_text_for_both_keys(tmp_path):
    """One text (`topopt.config.BOTH_VOLUME_KEYS_TEXT`): a YAML with both keys gets the words that the builder gives for a spec with both keys."""
    from topopt.config import UserError, load_config
    from topopt.regions_builder import SpecError, build_spec
    with pytest.raises(SpecError) as spec_err:
        build_spec(clone(settings={"target_mass_g": 30.0, "volume_fraction": 0.3, "voxel_size": 2.0}), tmp_path / "s")
    r = build_spec_for(clone(settings={"volume_fraction": 0.3, "voxel_size": 2.0}), tmp_path / "o", [])
    doc = yaml.safe_load(Path(r["yaml"]).read_text())
    doc["settings"]["target_mass_g"] = 30.0
    Path(r["yaml"]).write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(UserError) as exc:
        load_config(r["yaml"])
    from topopt.config import BOTH_VOLUME_KEYS_TEXT
    assert str(spec_err.value) in str(exc.value) and str(spec_err.value) == BOTH_VOLUME_KEYS_TEXT


def test_the_time_model_takes_0_25_only_for_a_spec_with_no_fraction_and_the_report_says_so(tmp_path, monkeypatch):
    """With `target_mass_g` the fraction does not exist before the grid does: the time model of the voxel aim takes 0.25, with a note."""
    from topopt import regions_builder
    seen = []

    def fake_model(spec, user_vf):
        seen.append(user_vf)
        return "cholesky", object()           # a gpu-mg time model is not on a CPU host: any object stands for it

    monkeypatch.setattr(regions_builder, "spec_time_model", fake_model)
    monkeypatch.setattr(regions_builder, "auto_voxel", lambda *a, **k: 2.0)
    lines = []
    r = build_spec_for(clone(settings={"target_mass_g": 30.0, "voxel_size": "auto"}), tmp_path / "m", lines)
    assert r["ok"] and seen == [0.25]
    assert any(ln.startswith("note: the voxel aim took volume_fraction 0.25 for the gpu-mg time model") and "settings.target_mass_g" in ln
               for ln in lines)
    lines = []
    r = build_spec_for(clone(settings={"volume_fraction": 0.3, "voxel_size": "auto"}), tmp_path / "f", lines)
    assert r["ok"] and seen == [0.25, 0.3] and not any("voxel aim took" in ln for ln in lines)
