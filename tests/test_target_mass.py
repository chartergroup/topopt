"""Tests of `settings.target_mass_g` (item I7, CONTRACTS.md section 3, "The mass in grams").

A YAML gives `volume_fraction` OR `target_mass_g`. The loader checks the keys and the sign. `build_problem` derives the fraction from the voxel
grid with the rule of `volume_basis.volume_target` read backward, so that `derived.volume.target_mass_g` is the asked mass again. A target at or
below the forced-solid mass, a target above the mass of all active voxels, and `objective: mass` with a target are refused with options.
Every path that reads the YAML again gets the same fraction: the run, `--resume`, `--continue-from`, the refine rebuild, the YAML check of
the verify step and of the hints tool.
"""

import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from test_optimize import make_case
from topopt.cli import check_data, main
from topopt.config import UserError, load_config
from topopt.problem import build_problem

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "examples" / "hook"


def edit(path, fn):
    d = yaml.safe_load(Path(path).read_text())
    fn(d)
    Path(path).write_text(yaml.safe_dump(d))
    return path


def with_target(path, grams, **settings):
    """Replace `volume_fraction` of the YAML by `target_mass_g`."""
    def fn(d):
        d["settings"].pop("volume_fraction", None)
        d["settings"]["target_mass_g"] = grams
        d["settings"].update(settings)
    return edit(path, fn)


@pytest.fixture(scope="module")
def cantilever(tmp_path_factory):
    """The cantilever of `test_optimize.make_case` at 2 mm: `(yaml, check facts of the YAML with volume_fraction)`."""
    root = tmp_path_factory.mktemp("target_mass_cantilever")
    p = make_case(root, h=2.0, iterations=6)
    return p, check_data(p)


def grams(facts):
    """`(forced-solid mass, mass of all active voxels)` in g, from the numbers that `topopt check` reports."""
    d = facts["derived"]
    return d["volume"]["forced_solid_mm3"] / 1000.0 * d["density_g_cm3"], d["mass_active_block_g"]


# --- the loader: keys and sign -------------------------------------------------------------------------------------------------------


def test_a_yaml_takes_target_mass_g_in_place_of_volume_fraction(tmp_path):
    cfg = load_config(with_target(make_case(tmp_path, h=2.0), 1.5))
    assert cfg.settings.target_mass_g == 1.5
    assert cfg.settings.volume_fraction is None, "the loader has no voxel grid: the problem build derives the fraction"


def test_a_yaml_with_volume_fraction_is_as_before(tmp_path):
    cfg = load_config(make_case(tmp_path, h=2.0))
    assert cfg.settings.volume_fraction == 0.3 and cfg.settings.target_mass_g is None


def test_both_keys_are_refused_with_the_options(tmp_path):
    p = edit(make_case(tmp_path, h=2.0), lambda d: d["settings"].update(target_mass_g=1.5))
    with pytest.raises(UserError) as err:
        load_config(p)
    msg = str(err.value)
    assert "both volume_fraction and target_mass_g" in msg and "Options:" in msg and "keep settings.target_mass_g" in msg
    assert "keep settings.volume_fraction" in msg and "words of the user say which one" in msg and f"(file: {p.resolve()})" in msg
    assert "\n" not in msg


def test_neither_key_is_refused_with_the_options(tmp_path):
    p = edit(make_case(tmp_path, h=2.0), lambda d: d["settings"].pop("volume_fraction"))
    with pytest.raises(UserError) as err:
        load_config(p)
    msg = str(err.value)
    assert "missing key 'volume_fraction' or 'target_mass_g' in 'settings'" in msg and "Options: settings.volume_fraction" in msg
    assert "settings.target_mass_g" in msg and "\n" not in msg


@pytest.mark.parametrize("bad", [0, -2.5, 0.0, True, "heavy", None, [3]])
def test_a_target_that_is_not_a_number_above_zero_is_refused(tmp_path, bad):
    p = with_target(make_case(tmp_path, h=2.0), bad)
    with pytest.raises(UserError) as err:
        load_config(p)
    msg = str(err.value)
    assert "settings.target_mass_g" in msg and "Options: write a number above 0 in grams" in msg and "(file:" in msg and "\n" not in msg


def test_a_target_written_in_the_yaml_11_exponent_form_is_a_number(tmp_path):
    p = make_case(tmp_path, h=2.0)
    text = p.read_text().replace("volume_fraction: 0.3", "target_mass_g: 3e1")   # YAML 1.1 reads 3e1 as text
    p.write_text(text)
    assert load_config(p).settings.target_mass_g == 30.0


def test_objective_mass_with_a_target_is_refused_with_the_options(tmp_path):
    p = with_target(make_case(tmp_path, h=2.0), 1.5, objective="mass")
    with pytest.raises(UserError) as err:
        load_config(p)
    msg = str(err.value)
    assert "does not go with settings.objective 'mass'" in msg and "minimizes the mass" in msg and "start density" in msg and "Options:" in msg
    assert "settings.objective compliance with settings.target_mass_g" in msg and "\n" not in msg


def test_the_key_lists_of_the_loader_name_the_new_key():
    from topopt import config
    assert "volume_fraction" not in config.SETTINGS_REQUIRED and "target_mass_g" not in config.SETTINGS_REQUIRED
    assert "volume_fraction" in config.SETTINGS_OPTIONAL and "target_mass_g" in config.SETTINGS_OPTIONAL
    assert "settings.target_mass_g" in config.accepted_key_paths()


def test_no_text_says_that_the_asked_mass_is_the_mass_of_the_finished_part(tmp_path):
    """Review M3: the final mesh is a result (it weighs 5 % to 9 % less on the hook at 2 mm and 1.2 mm; with `regions.not_printed` more).
    The refusals, the card, the schema comment, CONTRACTS section 3 and the README call the key the mass that the user asks for."""
    from topopt import levers_cards
    for name in ("n", "b", "o"):
        (tmp_path / name).mkdir()
    cases = (edit(make_case(tmp_path / "n", h=2.0), lambda d: d["settings"].pop("volume_fraction")),   # neither key
             with_target(make_case(tmp_path / "b", h=2.0), 0),                                          # a bad value
             with_target(make_case(tmp_path / "o", h=2.0), 1.5, objective="mass"))                      # the objective mass
    messages = []
    for p in cases:
        with pytest.raises(UserError) as err:
            load_config(p)
        messages.append(str(err.value))
    assert len(messages) == 3 and all("asked mass" in m or "that mass" in m for m in messages), messages
    card = next(c for c in levers_cards.CARDS if c.key == "settings.target_mass_g")
    contracts = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    section = contracts.split("### The mass in grams", 1)[1].split("\n### ", 1)[0]
    schema = [ln for ln in contracts.splitlines() if ln.strip().startswith("# target_mass_g:")]
    readme = [ln for ln in (ROOT / "docs" / "reference" / "problem-and-run.md").read_text(encoding="utf-8").splitlines() if "`target_mass_g` (the mass" in ln]
    comment = [ln for ln in (ROOT / "topopt" / "config.py").read_text(encoding="utf-8").splitlines() if ln.strip().startswith("target_mass_g: float")]
    assert len(schema) == len(readme) == len(comment) == 1
    for text in (*messages, card.effect, card.values, section, *schema, *readme, *comment):
        assert "finished part" not in text and "part of that mass" not in text, text[:160]
    # Review m1: the asked mass holds the forced-solid voxels of a `regions.not_printed` body; the card and section 3 say so and name `part_only.stl`.
    for text in (card.values, section):
        assert "regions.not_printed" in text and "forced solid" in text and "in the asked mass" in text and "part_only.stl" in text, text[:160]
    assert "that the user asks for" in schema[0] and "that the user asks for" in readme[0] and "that the user asks for" in section.split("\n", 3)[2]


# --- the derivation: one rule in both directions -----------------------------------------------------------------------------------


def test_the_problem_build_derives_the_fraction_and_the_round_trip_is_exact(tmp_path, cantilever):
    from topopt.volume_basis import fraction_for_mass, volume_target
    _, facts = cantilever
    forced_g, block_g = grams(facts)
    n_active, n_free, h = facts["active_voxels"], facts["free_design_voxels"], facts["grid"]["voxel_mm"]
    dens = facts["derived"]["density_g_cm3"]
    for t in (forced_g + 0.05 * (block_g - forced_g), 0.5 * (forced_g + block_g), block_g - 0.001, 1.2345):
        t = max(t, forced_g + 1e-3)
        vf = fraction_for_mass(t, h, dens, n_active, n_free)
        assert 0.0 < vf <= 1.0
        assert volume_target(vf, h, dens, n_active, n_free)["target_mass_g"] == pytest.approx(t, abs=5.1e-4)   # rounded to 3 decimals
        # the exact inverse, without the rounding of the report
        vox = h ** 3
        assert ((n_active - n_free) + vf * n_free) * vox / 1000.0 * dens == pytest.approx(t, rel=1e-12)
    p = with_target(make_case(tmp_path, h=2.0), 1.5)
    prob = build_problem(load_config(p))
    st = prob.config.settings
    assert st.target_mass_g == 1.5 and st.volume_fraction == pytest.approx(fraction_for_mass(1.5, h, dens, n_active, n_free), rel=1e-12)


def test_check_reports_the_asked_mass_with_its_origin_and_the_derived_fraction(tmp_path):
    p = with_target(make_case(tmp_path, h=2.0), 1.5)
    facts = check_data(p)
    d, vol = facts["derived"], facts["derived"]["volume"]
    assert d["target_mass_asked_g"] == 1.5 and d["target_mass_origin"] == "settings.target_mass_g"
    assert vol["target_mass_asked_g"] == 1.5 and vol["target_mass_origin"] == "settings.target_mass_g"
    assert vol["target_mass_g"] == pytest.approx(1.5, abs=5.1e-4)                       # the engine's target, from the derived fraction
    assert d["volume_fraction"] == vol["volume_fraction"] == facts["volume_fraction"] and 0.0 < d["volume_fraction"] < 1.0
    assert d["volume_fraction_origin"].startswith("derived from settings.target_mass_g")
    # the same YAML with the derived fraction written down gives the same volume numbers, and has none of the new keys
    q = tmp_path / "plain.yaml"
    shutil.copy(p, q)
    edit(q, lambda dd: (dd["settings"].pop("target_mass_g"), dd["settings"].update(volume_fraction=d["volume_fraction"])))
    plain = check_data(q)["derived"]
    assert {k: v for k, v in plain["volume"].items()} == {k: v for k, v in vol.items() if k not in ("target_mass_asked_g", "target_mass_origin")}
    assert "target_mass_asked_g" not in plain and "volume_fraction_origin" not in plain


def test_the_derived_numbers_of_a_config_with_no_grid_say_that_the_fraction_comes_later(tmp_path):
    from topopt.cli import derived_numbers
    d = derived_numbers(load_config(with_target(make_case(tmp_path, h=2.0), 1.5)))
    assert d["volume_fraction"] is None and d["target_mass_asked_g"] == 1.5 and "needs the voxel grid" in d["volume_fraction_origin"]


def test_topopt_check_prints_one_line_for_the_target(tmp_path, capsys):
    p = with_target(make_case(tmp_path, h=2.0), 1.5)
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("target mass:")]
    assert len(lines) == 1 and "1.5 g asked (settings.target_mass_g)" in lines[0] and "is derived from it" in lines[0]
    assert "applies to the free part; target " in out and "= 1.5 g at 1.24 g/cm3" in out


def test_the_mcp_check_problem_carries_the_facts(tmp_path):
    from topopt_mcp import server
    res = server.check_problem(str(with_target(make_case(tmp_path, h=2.0), 1.5)))
    assert res["ok"] and res["derived"]["volume"]["target_mass_asked_g"] == 1.5 and res["derived"]["volume"]["target_mass_origin"] == "settings.target_mass_g"


# --- the range refusals ------------------------------------------------------------------------------------------------------------------


def test_a_target_below_the_forced_solid_mass_is_refused_with_the_options(tmp_path, cantilever):
    _, facts = cantilever
    forced_g, _ = grams(facts)
    assert forced_g > 0.1
    p = with_target(make_case(tmp_path, h=2.0), round(forced_g * 0.5, 3))
    with pytest.raises(UserError) as err:
        build_problem(load_config(p))
    msg = str(err.value)
    assert "is not above the mass of the forced-solid voxels" in msg and f"{forced_g:.3f} g" in msg
    assert "Options: a target above" in msg and "smaller fixed, load and solid regions" in msg and "\n" not in msg
    assert "and union interfaces)" in msg and "regions or union interfaces" in msg     # review n1: a boss or a tube is forced solid too
    # the same text reaches the user through `topopt check` as one error line with the file
    assert main(["check", str(p)]) == 1


def test_the_guard_for_a_design_space_with_no_free_voxel_stays_true_alone(tmp_path):
    """Review n2: `voxelize` refuses a design space with no design voxel first, so a normal run never reaches the guard of `resolve_volume`."""
    import numpy as np
    from topopt.problem import resolve_volume
    from topopt.voxelize import FIXED
    cfg = load_config(with_target(make_case(tmp_path, h=2.0), 1.5))
    labels = np.full((3, 3, 3), FIXED, np.uint8)
    with pytest.raises(UserError, match="the design space has no free design voxel") as err:
        resolve_volume(cfg, labels, 2.0)
    assert "and union interfaces)" in str(err.value) and "regions or union interfaces" in str(err.value)


def test_a_target_just_above_the_forced_solid_mass_builds(tmp_path, cantilever):
    _, facts = cantilever
    forced_g, _ = grams(facts)
    prob = build_problem(load_config(with_target(make_case(tmp_path, h=2.0), forced_g + 0.01)))
    assert 0.0 < prob.config.settings.volume_fraction < 0.1


def test_a_target_above_the_mass_of_all_active_voxels_is_refused_with_the_numbers_of_check(tmp_path, cantilever, capsys):
    _, facts = cantilever
    _, block_g = grams(facts)
    p = with_target(make_case(tmp_path, h=2.0), round(block_g * 1.5, 3))
    with pytest.raises(UserError) as err:
        build_problem(load_config(p))
    msg = str(err.value)
    assert "is above the mass of all active voxels solid" in msg and f"{facts['derived']['mass_active_block_g']:.3f} g" in msg
    assert "derived.mass_active_block_g" in msg and "Options: a target at or below" in msg and "a larger design space" in msg and "\n" not in msg
    assert "(fixed, load and solid regions, and union interfaces)" in msg                  # review n1
    capsys.readouterr()
    assert main(["check", str(p)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "is above the mass of all active voxels solid" in err and "(file:" in err and len(err.strip().splitlines()) == 1


def test_a_target_at_the_mass_of_the_whole_block_less_a_little_builds_at_a_fraction_below_one(tmp_path, cantilever):
    _, facts = cantilever
    _, block_g = grams(facts)
    prob = build_problem(load_config(with_target(make_case(tmp_path, h=2.0), block_g - 0.01)))
    assert 0.9 < prob.config.settings.volume_fraction < 1.0


# --- the run: the engine uses the derived fraction exactly -------------------------------------------------------------------------


def run(yaml_path, out, *flags):
    return main(["run", str(yaml_path), "--out", str(out), "--quiet", *flags])


def read(path):
    return json.loads(Path(path).read_text())


def test_a_run_with_a_target_is_the_run_of_the_derived_fraction(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    pa = with_target(make_case(a, h=2.0, iterations=6), 1.5)
    assert run(pa, a / "run") == 0
    ra = read(a / "run" / "run.json")
    pb = make_case(b, h=2.0, iterations=6)
    edit(pb, lambda d: d["settings"].update(volume_fraction=ra["volume_fraction"]))
    assert run(pb, b / "run") == 0
    rb = read(b / "run" / "run.json")
    assert rb["volume_fraction"] == ra["volume_fraction"] and 0.0 < ra["volume_fraction"] < 1.0
    vol = ra["derived"]["volume"]
    assert vol["target_mass_asked_g"] == 1.5 and vol["target_mass_origin"] == "settings.target_mass_g"
    assert vol["target_mass_g"] == pytest.approx(1.5, abs=5.1e-4) and rb["derived"]["volume"]["target_mass_g"] == vol["target_mass_g"]
    assert ra["constraints"] == rb["constraints"]                                       # the volume row has the same limit
    ha = [json.loads(ln) for ln in (a / "run" / "history.jsonl").read_text().splitlines()]
    hb = [json.loads(ln) for ln in (b / "run" / "history.jsonl").read_text().splitlines()]
    assert [r["volume_fraction"] for r in ha] == pytest.approx([r["volume_fraction"] for r in hb], rel=1e-9)
    log = (a / "run" / "run.log").read_text().splitlines()
    lines = [ln for ln in log if "target mass:" in ln]
    assert len(lines) == 1 and "1.5 g asked (settings.target_mass_g)" in lines[0] and f"{ra['volume_fraction']:.6g}" in lines[0]
    assert not any("target mass:" in ln for ln in (b / "run" / "run.log").read_text().splitlines())
    m = read(a / "run" / "result" / "metrics.json")
    assert "mass_final_g" in m and not [k for k in m if "target_mass" in k], "no new verdict test and no tolerance: the card says how to compare"
    assert not [k for k in m["verdict"] if "mass" in k and k != "text"]
    # Review M2: the end fact of a run with a target is in the log; the run of the same fraction (a YAML with volume_fraction) has none.
    assert m["mass_asked_g"] == 1.5
    facts = [ln for ln in log if "final mesh" in ln]
    assert facts == [f"fact: final mesh {m['mass_final_g']:.2f} g against 1.5 g asked (settings.target_mass_g)"]
    mb = read(b / "run" / "result" / "metrics.json")
    assert "mass_asked_g" not in mb and not any("final mesh" in ln for ln in (b / "run" / "run.log").read_text().splitlines())


@pytest.fixture(scope="module")
def hook(tmp_path_factory):
    """A short CPU run of `examples/hook/hook_short.yaml` with `target_mass_g` (12 passes): `(folder, yaml, out, run.json)`."""
    root = tmp_path_factory.mktemp("target_mass_hook")
    for f in HOOK.glob("*.stl"):
        shutil.copy(f, root / f.name)
    p = root / "hook_short.yaml"
    shutil.copy(HOOK / "hook_short.yaml", p)
    edit(p, lambda d: (d["settings"].pop("volume_fraction"),
                       d["settings"].update(target_mass_g=25.0, iterations=12, beta_schedule=[[0, 1], [4, 2], [8, 4], [10, 8]])))
    out = root / "run"
    assert run(p, out) == 0
    return root, p, out, read(out / "run.json")


def test_the_short_hook_run_has_the_asked_mass_and_the_derived_fraction(hook):
    from topopt.volume_basis import volume_target
    _, p, out, rj = hook
    vol = rj["derived"]["volume"]
    assert vol["target_mass_asked_g"] == 25.0 and vol["target_mass_origin"] == "settings.target_mass_g"
    assert vol["target_mass_g"] == pytest.approx(25.0, abs=5.1e-4) and 0.1 < rj["volume_fraction"] < 0.2
    again = volume_target(rj["volume_fraction"], rj["h"], rj["derived"]["density_g_cm3"], rj["n_active"], vol["free_design_voxels"])
    assert again["target_mass_g"] == vol["target_mass_g"] and again["target_volume_mm3"] == vol["target_volume_mm3"]
    assert sum("target mass:" in ln for ln in (out / "run.log").read_text().splitlines()) == 1
    m = read(out / "result" / "metrics.json")
    assert 0.5 * 25.0 < m["mass_final_g"] < 1.5 * 25.0                                      # the final mass is in metrics; the test sets no tolerance of its own


def test_the_end_fact_is_in_metrics_the_log_the_status_and_no_lever_flag(hook):
    """Review M2: `metrics.mass_asked_g`, one log line, the words of `run_status`; a fact, so `run_levers` flags nothing for it."""
    from topopt.levers_run import run_levers
    from topopt_mcp import server
    _, _, out, _ = hook
    m = read(out / "result" / "metrics.json")
    assert m["mass_asked_g"] == 25.0 and list(m).index("mass_asked_g") == list(m).index("mass_final_g") + 1
    want = f"fact: final mesh {m['mass_final_g']:.2f} g against 25 g asked (settings.target_mass_g)"
    assert [ln for ln in (out / "run.log").read_text().splitlines() if "final mesh" in ln] == [want]
    summary = server.run_status(str(out))["summary"]
    assert summary.startswith("done: ") and f"; final mesh {m['mass_final_g']:.2f} g against 25 g asked (settings.target_mass_g)" in summary
    assert "fact: final mesh" not in summary
    assert server.run_result(str(out), preview=False)["metrics"]["mass_asked_g"] == 25.0       # `run_result` returns metrics.json: the key reaches the agent
    assert not [e for e in run_levers(out)["entries"] if "mass_asked" in e["fact"]]
    assert "part_only" not in m and "part_only.stl" not in want


def test_a_run_with_a_not_printed_body_adds_the_mass_of_part_only_stl_to_the_end_fact(hook, tmp_path):
    """Review m1: the asked mass holds the forced-solid voxels of a `regions.not_printed` body; the line gives `part_only.stl` too."""
    root, p, _, _ = hook
    folder = tmp_path / "np"
    shutil.copytree(root, folder, ignore=shutil.ignore_patterns("run", "*.yaml"))
    q = folder / "hook_short.yaml"
    shutil.copy(p, q)
    edit(q, lambda d: d["regions"].update(not_printed=["hook_tip.stl"]))
    assert run(q, folder / "run") == 0
    m = read(folder / "run" / "result" / "metrics.json")
    po = m["part_only"]
    assert po["applied"] and m["mass_asked_g"] == 25.0 and 0.0 < po["mass_g"] < m["mass_final_g"]
    want = (f"fact: final mesh {m['mass_final_g']:.2f} g against 25 g asked (settings.target_mass_g); "
            f"part_only.stl {po['mass_g']:.2f} g (without regions.not_printed)")
    assert [ln for ln in (folder / "run" / "run.log").read_text().splitlines() if "final mesh" in ln] == [want]


def test_the_end_fact_text_is_one_function_and_a_fact_only():
    from topopt.volume_basis import mass_asked_line
    assert mass_asked_line({"mass_final_g": 27.314, "mass_asked_g": 30.0}) == "fact: final mesh 27.31 g against 30 g asked (settings.target_mass_g)"
    assert mass_asked_line({"mass_final_g": 27.3, "mass_asked_g": 22.5}) == "fact: final mesh 27.30 g against 22.5 g asked (settings.target_mass_g)"
    assert mass_asked_line({"mass_final_g": 27.3}) is None and mass_asked_line({"mass_final_g": 27.3, "mass_asked_g": None}) is None
    assert mass_asked_line({"mass_final_g": 27.3, "mass_asked_g": True}) is None and mass_asked_line(None) is None
    both = mass_asked_line({"mass_final_g": 27.3, "mass_asked_g": 30.0, "part_only": {"applied": True, "mass_g": 24.9}})
    assert both == ("fact: final mesh 27.30 g against 30 g asked (settings.target_mass_g); "
                    "part_only.stl 24.90 g (without regions.not_printed)")
    failed = mass_asked_line({"mass_final_g": 27.3, "mass_asked_g": 30.0, "part_only": {"applied": False, "error": "x"}})
    assert failed == "fact: final mesh 27.30 g against 30 g asked (settings.target_mass_g)"
    assert "%" not in both and "percent" not in both and "toleran" not in both          # a fact: no percent, no tolerance, no verdict word


def test_a_resume_of_the_finished_run_sees_no_change(hook, tmp_path):
    _, p, out, rj = hook
    copy = tmp_path / "run"
    shutil.copytree(out, copy)
    for name in ("job.DONE", "job.guard", "job.json"):
        (copy / name).unlink(missing_ok=True)
    assert run(p, copy, "--resume") == 0
    log = (copy / "run.log").read_text()
    assert "the post step runs again" in log.lower() and "inputs changed" not in log
    assert read(copy / "run.json")["volume_fraction"] == rj["volume_fraction"]


def test_a_continuation_with_the_same_yaml_has_the_same_fraction_and_needs_no_override(hook, tmp_path):
    _, p, out, rj = hook
    child = tmp_path / "child"
    assert run(p, child, "--continue-from", str(out), "--more-iterations", "3") == 0, "no spurious change of settings.volume_fraction"
    rc = read(child / "run.json")
    assert rc["volume_fraction"] == rj["volume_fraction"] and rc["derived"]["volume"] == rj["derived"]["volume"]
    assert rc["continued_from"]["iterations"] == 15 and rc["derived"]["volume"]["target_mass_asked_g"] == 25.0
    assert sum("target mass:" in ln for ln in (child / "run.log").read_text().splitlines()) == 1


def test_a_continuation_with_another_target_is_a_named_change_and_uses_the_new_fraction(hook, tmp_path, capsys):
    root, p, out, rj = hook
    work = tmp_path / "case"
    shutil.copytree(root, work, ignore=shutil.ignore_patterns("run"))
    q = work / "hook_short.yaml"
    edit(q, lambda d: d["settings"].update(target_mass_g=20.0))
    child = tmp_path / "child"
    capsys.readouterr()
    assert run(q, child, "--continue-from", str(out), "--more-iterations", "3") == 1
    err = capsys.readouterr().err
    assert "settings.target_mass_g: 25.0 -> 20.0" in err and "settings.volume_fraction" not in err
    assert run(q, child, "--continue-from", str(out), "--more-iterations", "3", "--allow-changed-inputs") == 0
    rc = read(child / "run.json")
    assert rc["derived"]["volume"]["target_mass_asked_g"] == 20.0 and rc["derived"]["volume"]["target_mass_g"] == pytest.approx(20.0, abs=5.1e-4)
    assert rc["volume_fraction"] < rj["volume_fraction"]
    assert read(out / "run.json")["volume_fraction"] == rj["volume_fraction"], "the parent is only read"


def test_the_refine_rebuild_and_the_yaml_checks_see_the_same_fraction(hook):
    from topopt.feedback import _yaml_vs_run
    from topopt.refine.build import problem_from_run
    from topopt.verify.run import compare_yaml_problem
    _, p, out, rj = hook
    problem = problem_from_run(out, rj)                                                     # builds the YAML again and compares the labels with the run
    assert problem.config.settings.volume_fraction == rj["volume_fraction"] and problem.config.settings.target_mass_g == 25.0
    res = compare_yaml_problem(rj, problem)                                                 # the check of `topopt verify` and `topopt refine`
    assert res["matches"] and res["differences"] == []
    diffs, compared, not_compared = _yaml_vs_run(rj, yaml.safe_load(p.read_text()))
    assert diffs == [] and "settings.target_mass_g" in compared and "settings.volume_fraction" not in compared
    assert [x["key"] for x in not_compared if x["key"] == "settings.volume_fraction"] == ["settings.volume_fraction"]
    other = yaml.safe_load(p.read_text())
    other["settings"]["target_mass_g"] = 20.0
    diffs, _, _ = _yaml_vs_run(rj, other)
    assert [(x["key"], x["yaml"], x["run_json"]) for x in diffs] == [("settings.target_mass_g", 20.0, 25.0)]


def test_the_ratchet_and_the_part_builder_read_a_target_yaml(hook):
    from topopt.ratchet import inputs_digest
    from topopt.regions_builder import check_yaml
    _, p, _, rj = hook
    assert inputs_digest(p)["error"] is None
    said = []
    check_yaml(p, lambda *a: said.append(" ".join(str(x) for x in a)))
    text = "\n".join(said)
    assert not text.startswith("ERROR"), text
    assert any(ln.startswith("volume_fraction: ") and f"{rj['volume_fraction']:.6g}" in ln and "derived from settings.target_mass_g 25 g" in ln
               for ln in said), text
    assert re.search(r"target: .* 25\.0 g at 1\.24 g/cm3", text), text


def test_an_old_checkpoint_resumes_without_the_new_key(tmp_path):
    """`effective_settings` leaves `target_mass_g` out when it is unset, so a checkpoint made before I7 still resumes."""
    from topopt import checkpoint
    p = make_case(tmp_path, h=2.0)
    eff = checkpoint.effective_settings(load_config(p))
    assert "target_mass_g" not in eff["settings"] and eff["settings"]["volume_fraction"] == 0.3
    with_target(p, 1.5)
    tgt = checkpoint.effective_settings(load_config(p))
    assert tgt["settings"]["target_mass_g"] == 1.5 and "volume_fraction" not in tgt["settings"]
    built = build_problem(load_config(p)).config
    assert checkpoint.effective_settings(built) == tgt, "the config of load_config and the config of build_problem give one dict"
