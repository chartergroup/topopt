"""A region STL that looks like the wrong unit (TC-3, finding #162; `topopt/unit_fact.py`, CONTRACTS.md section 3, "The unit of the region STLs").

An STL has no unit. The loader reads the numbers as mm, so a part drawn in inches loads 25.4 times too small, and a part drawn in metres 1000
times too small. The tools state it as a fact with options when `L x 25.4` still fits the longer side of the print bed (L is the largest
bounding-box size of the design space STL). The fact is in `optimize.problem_facts` (so in `topopt check`, `topopt check --json` and the start of
`run.log`) and in the `warnings` of `check_data` (so `check_problem` and `build_regions`). It never stops a run. A part so small that the
loader finds no voxel stops with the loader error, and that error carries the same hint.
"""
import json
import math
import re
from pathlib import Path

import pytest
import trimesh
import yaml

from test_optimize import box
from topopt import unit_fact
from topopt.cli import check_data, main
from topopt.config import UserError, load_config
from topopt.optimize import problem_facts
from topopt.problem import build_problem

REPO = Path(__file__).resolve().parent.parent
START = "fact: the largest size of the design space"
EXAMPLES = {"hook": REPO / "examples/hook/hook.yaml", "bracket": REPO / "examples/interfaces/bracket.yaml",
            "ladle": REPO / "examples/ladle/ladle.yaml"}
IN, M = 1 / 25.4, 0.001      # the scale of a part drawn in inches, and of a part drawn in metres


def unit_lines(lines):
    return [x for x in lines if x.startswith(START)]


def scaled_copy(src, dst, factor, voxel_factor=1.0):
    """A copy of an example YAML in `dst` whose region STLs (design, fixed, load, keep_out, solid: all of them) are scaled by `factor`.

    The voxel size is multiplied by `voxel_factor` (1.0 keeps `voxel_size` as it is). The STL files keep their names. Returns the YAML path."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    doc = yaml.safe_load(src.read_text())

    def scale(name):
        m = trimesh.load(src.parent / name, file_type="stl", force="mesh")
        m.apply_scale(factor)
        m.export(dst / Path(name).name)
        return Path(name).name

    reg = doc["regions"]
    for key, val in list(reg.items()):
        if isinstance(val, str):
            reg[key] = scale(val)
        elif isinstance(val, list):
            reg[key] = [scale(v) for v in val]
        elif isinstance(val, dict):
            reg[key] = {k: scale(v) for k, v in val.items()}
    doc["settings"]["voxel_size"] = float(doc["settings"]["voxel_size"]) * voxel_factor
    out = dst / src.name
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out


def small_part(tmp_path, size=10.0, print_block=None, h=1.0, iterations=2):
    """A cantilever whose design space is `size` mm long (a part of the size of a real small part, in mm)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    box(tmp_path / "design.stl", (0, 0, 0), (size, 6, 6))
    box(tmp_path / "plate.stl", (-1, 0, 0), (0, 6, 6))
    box(tmp_path / "tip.stl", (size - 1, 2, 0), (size, 4, 2))
    d = {"name": "small", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.4, "iterations": iterations, "beta_schedule": [[0, 1]]}}
    if print_block is not None:
        d["print"] = print_block
    p = tmp_path / "small.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


# --- the examples as they are: no fact -------------------------------------------------------------------------------------------


def test_the_design_spaces_of_the_examples_are_70_40_and_154_mm():
    """The sizes (trimesh extents) are far above 220 / 25.4 = 8.66 mm, so the rule has a wide margin on the examples."""
    from topopt import unit_fact
    sizes = {k: unit_fact.design_size(load_config(p)) for k, p in EXAMPLES.items()}
    assert {k: round(v, 3) for k, v in sizes.items()} == {"hook": 70.0, "bracket": 40.0, "ladle": 154.0}


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_the_examples_give_no_unit_fact_in_problem_facts(name):
    assert unit_lines(problem_facts(load_config(EXAMPLES[name]))) == []


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_the_examples_give_no_unit_fact_in_check_data(name):
    assert unit_lines(check_data(EXAMPLES[name])["warnings"]) == []


# --- a scaled copy: the fact, with the sizes ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("factor, expect", [
    (IN, ["is 2.756 mm, read as mm", "70 mm after x 25.4 (inches)", "2756 mm after x 1000 (metres)", "(design.stl)"]),
    (M, ["is 0.07 mm, read as mm", "1.778 mm after x 25.4 (inches)", "70 mm after x 1000 (metres)", "(design.stl)"]),
], ids=["inches", "metres"])
def test_a_scaled_hook_gives_the_fact_in_problem_facts_check_data_and_topopt_check(tmp_path, capsys, factor, expect):
    p = scaled_copy(EXAMPLES["hook"], tmp_path, factor, voxel_factor=factor)      # the voxel size is scaled too: the check goes on to the facts
    [fact] = unit_lines(problem_facts(load_config(p)))
    for text in expect:
        assert text in fact, (text, fact)
    assert "\n" not in fact and "220 mm: the default, print.bed_mm is not set" in fact              # one line; it says what it rests on
    assert ("options: scale every STL file of the YAML (regions, measure regions, interface solids) by 25.4 or by 1000 (or by the factor of another "
            "unit, for example 10 for cm), and also the lengths in the YAML that use the unit of the files") in fact
    assert "voxel_size and the bed are read as mm); or keep the files as they are." in fact
    assert "A part that is really this small can ignore this fact." in fact
    d = check_data(p)                                                                                  # no gate: the check goes on
    assert unit_lines(d["warnings"]) == [fact]                                                         # the warnings (so the MCP tools) have the same text
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.count(fact) == 1 and out.count(START) == 1                                              # the text output prints it once
    assert main(["check", str(p), "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert unit_lines(j["facts"]) == [fact] and unit_lines(j["warnings"]) == [fact]


def test_the_sizes_are_those_of_the_stl_files(tmp_path):
    """The numbers are read from the file: scale the hook by 0.123 and the fact follows (8.61 mm: 8.61 x 25.4 = 218.7 mm fits 220 mm)."""
    p = scaled_copy(EXAMPLES["hook"], tmp_path, 0.123, voxel_factor=0.123)
    [fact] = unit_lines(problem_facts(load_config(p)))
    assert "is 8.61 mm, read as mm" in fact and "218.7 mm after x 25.4" in fact and "8610 mm after x 1000" in fact
    p = scaled_copy(EXAMPLES["hook"], tmp_path / "b", 0.1245, voxel_factor=0.1245)               # 8.715 mm: x 25.4 = 221.4 mm does not fit 220 mm
    assert unit_lines(problem_facts(load_config(p))) == []


# --- the bed moves the threshold -----------------------------------------------------------------------------------------------------


def test_a_bed_from_print_bed_mm_moves_the_threshold(tmp_path):
    """A design space of 10 mm: 10 x 25.4 = 254 mm. It does not fit the default bed (220 mm), and it fits a bed of 350 mm."""
    default = small_part(tmp_path / "default", 10.0)
    assert unit_lines(problem_facts(load_config(default))) == []
    assert unit_lines(check_data(default)["warnings"]) == []
    wide = small_part(tmp_path / "wide", 10.0, {"bed_mm": 350})
    [fact] = unit_lines(problem_facts(load_config(wide)))
    assert "is 10 mm, read as mm" in fact and "254 mm after x 25.4" in fact and "10000 mm after x 1000" in fact
    assert "(350 mm: print.bed_mm of the YAML)" in fact
    assert unit_lines(check_data(wide)["warnings"]) == [fact]
    long_side = small_part(tmp_path / "long", 10.0, {"bed_mm": [100, 350]})             # the longer side counts: a part may be turned on the bed
    assert "(350 mm: print.bed_mm of the YAML)" in unit_lines(problem_facts(load_config(long_side)))[0]
    short = small_part(tmp_path / "short", 10.0, {"bed_mm": [250, 200]})                # 254 mm does not fit 250 mm
    assert unit_lines(problem_facts(load_config(short))) == []


def test_the_bed_of_a_printer_profile_is_the_bed_and_the_text_names_the_printer(tmp_path):
    p = small_part(tmp_path, 10.0, {"printer_model": "Bambu P1S"})                       # a bed of 256 mm: 254 mm fits
    cfg = load_config(p)
    assert cfg.print_settings.bed_mm == (256.0, 256.0) and cfg.print_settings.derived[0] == "bed_mm"
    [fact] = unit_lines(problem_facts(cfg))
    assert "(256 mm: the bed of the printer bambu-lab-p1s, print.printer_model)" in fact
    given = small_part(tmp_path / "g", 10.0, {"printer_model": "Bambu P1S", "bed_mm": 200})   # the YAML wins over the profile
    assert unit_lines(problem_facts(load_config(given))) == []


def test_a_printer_profile_without_a_bed_is_named_in_the_text(tmp_path):
    """The catalog entry `voron-2-4` states no bed (`bed_mm: null`): the default bed applies, and the text names the printer (review of #253, N2)."""
    p = small_part(tmp_path, 5.0, {"printer_model": "voron-2-4"})                       # largest size 6 mm (the 6 mm sides): 152.4 mm fits 220 mm
    cfg = load_config(p)
    assert cfg.print_settings.printer_model == "voron-2-4" and "bed_mm" not in cfg.print_settings.derived
    [fact] = unit_lines(problem_facts(cfg))
    assert "(220 mm: the default: the profile of the printer voron-2-4 states no bed, and print.bed_mm is not set)" in fact


def test_a_design_stl_that_cannot_be_read_gives_no_fact_and_no_error(tmp_path):
    """`problem_facts` runs before the loader at the start of a run: a broken file gives no unit line and no error (the loader names it)."""
    cfg = load_config(small_part(tmp_path, 5.0))
    assert len(unit_lines(problem_facts(cfg))) == 1
    (tmp_path / "design.stl").write_bytes(b"not an stl file")
    assert unit_fact.unit_fact_lines(cfg) == [] and unit_lines(problem_facts(cfg)) == []


def test_the_kept_size_follows_a_change_of_the_file(tmp_path):
    """The size is kept for one file version (path, modification time, byte size): a new file at the same path is read again."""
    cfg = load_config(small_part(tmp_path, 5.0))
    assert unit_fact.design_size(cfg) == pytest.approx(6.0)                              # 5 x 6 x 6 mm
    box(tmp_path / "design.stl", (0, 0, 0), (7.5, 6, 6.5))
    assert unit_fact.design_size(cfg) == pytest.approx(7.5)


@pytest.mark.parametrize("args", [[], ["--json"]], ids=["text", "json"])
def test_topopt_check_reads_the_design_stl_once_as_on_main(tmp_path, capsys, monkeypatch, args):
    """The loader keeps the size of the design mesh, so `topopt check` reads the design STL once, as before TC-3 (review of #253, F1)."""
    p = scaled_copy(EXAMPLES["hook"], tmp_path, IN, voxel_factor=IN)
    design = (tmp_path / "design.stl").resolve()
    reads = []
    real_load = trimesh.load

    def counting_load(obj, *a, **k):
        if Path(str(obj)).resolve() == design:
            reads.append(k.get("process", True))
        return real_load(obj, *a, **k)

    monkeypatch.setattr(trimesh, "load", counting_load)
    assert main(["check", str(p), *args]) == 0
    assert START in capsys.readouterr().out                                              # the fact is there
    assert len(reads) == 1, reads


def test_the_rule_is_x_25_4_at_most_the_longer_side_of_the_bed():
    from topopt import unit_fact
    bed = 10.0 * unit_fact.INCH_MM
    assert unit_fact.looks_wrong(10.0, bed) is True                                      # at the bed: it fits
    assert unit_fact.looks_wrong(10.0, math.nextafter(bed, 0.0)) is False                # a hair over the bed: it does not
    for bad in (None, 0.0, -1.0, math.nan, math.inf):
        assert unit_fact.looks_wrong(bad, 1e9) is False
    assert (unit_fact.INCH_MM, unit_fact.METRE_MM) == (25.4, 1000.0)                      # unit conversions, not limits


# --- no gate: a run is not stopped ------------------------------------------------------------------------------------------------------


def test_the_run_goes_on_and_run_log_has_the_fact_before_the_first_pass(tmp_path):
    p = small_part(tmp_path, 10.0, {"bed_mm": 350})
    [fact] = unit_lines(problem_facts(load_config(p)))
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0                       # no gate
    log = (out / "run.log").read_text()
    assert log.count(fact) == 1 and log.index(fact) < log.index("iter")
    assert (out / "result" / "final.stl").is_file()


def test_no_fact_in_run_log_for_a_part_that_does_not_look_small(tmp_path):
    p = small_part(tmp_path, 10.0)                                                         # default bed: 254 mm does not fit 220 mm
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert START not in (out / "run.log").read_text()


# --- the loader error carries the hint ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("factor, region, size_text", [
    (IN, "region fixed[0] (wall_plate.stl) has no voxel", "is 2.756 mm read as mm, and it is 70 mm after x 25.4 (inches) or 2756 mm after x 1000 (metres)"),
    (M, "region design has no voxel", "is 0.07 mm read as mm, and it is 1.778 mm after x 25.4 (inches) or 70 mm after x 1000 (metres)"),
], ids=["inches", "metres"])
def test_a_scaled_hook_with_the_voxel_size_kept_stops_with_the_hint(tmp_path, capsys, factor, region, size_text):
    """Step 1 of TC-3: on main this stops with `... make the region larger` and no word on the unit. Now the same sentence has the sizes."""
    p = scaled_copy(EXAMPLES["hook"], tmp_path, factor)                                   # voxel_size 1.2 as it is
    with pytest.raises(UserError) as e:
        build_problem(load_config(p))
    msg = str(e.value)
    assert region in msg and "Use a smaller voxel_size or make the region larger. Check the unit of the STL files: the largest size of the design space (design.stl) " in msg
    assert size_text in msg and msg.endswith("(metres).")
    with pytest.raises(UserError) as e2:
        check_data(p)
    assert size_text in str(e2.value)
    assert main(["check", str(p)]) == 1
    err = capsys.readouterr().err
    assert region in err and size_text in err and err.count("error:") == 1                 # the one message the user gets


@pytest.mark.parametrize("voxel, guard", [(0.0003, r"grid would have \d+ voxels at voxel_size"),
                                          (0.0006, r"model has \d+ active voxels, the limit is 100000")], ids=["grid guard", "active voxel guard"])
def test_the_two_voxel_guards_carry_the_hint_too(tmp_path, voxel, guard):
    """A part in metres with a voxel size that is fine for a part in metres drawn at a larger scale stops at a guard, and the advice of the guard
    (a larger voxel_size) hides the cause. Found by trying voxel sizes between the two cases of step 1."""
    p = scaled_copy(EXAMPLES["hook"], tmp_path, M, voxel_factor=voxel / 1.2)
    with pytest.raises(UserError) as e:
        build_problem(load_config(p))
    msg = str(e.value)
    assert re.search(guard, msg), msg
    assert ". Check the unit of the STL files: the largest size of the design space (design.stl) is 0.07 mm read as mm, and it is 1.778 mm after x 25.4" in msg
    assert msg.endswith("70 mm after x 1000 (metres).")


def test_a_guard_of_a_part_that_looks_large_has_no_hint(tmp_path):
    p = scaled_copy(EXAMPLES["hook"], tmp_path, 1.0, voxel_factor=0.04 / 1.2)               # voxel 0.04 mm on a 70 mm part: the grid guard
    with pytest.raises(UserError) as e:
        build_problem(load_config(p))
    assert re.search(r"grid would have \d+ voxels", str(e.value)) and "unit" not in str(e.value)


def test_the_active_voxel_guard_of_a_part_that_looks_large_has_no_hint(tmp_path):
    """The hook as it is (70 mm) at voxel 0.6 mm: about 320,000 active voxels stop at the active-voxel guard, and the text has no unit hint."""
    p = scaled_copy(EXAMPLES["hook"], tmp_path, 1.0, voxel_factor=0.6 / 1.2)
    with pytest.raises(UserError) as e:
        build_problem(load_config(p))
    assert re.search(r"model has \d+ active voxels", str(e.value)) and "unit" not in str(e.value)


def test_the_loader_error_of_a_part_that_looks_large_has_no_hint(tmp_path):
    p = scaled_copy(EXAMPLES["hook"], tmp_path, 1.0, voxel_factor=100.0)                   # voxel 120 mm: a 70 mm part has no voxel centre inside
    with pytest.raises(UserError) as e:
        build_problem(load_config(p))
    assert str(e.value).endswith("Use a smaller voxel_size or make the region larger") and "unit" not in str(e.value)


# --- the tools of the MCP server and the part builder ---------------------------------------------------------------------------------------


def test_the_mcp_tool_check_problem_has_the_fact_in_its_warnings(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    p = scaled_copy(EXAMPLES["hook"], tmp_path / "w", IN, voxel_factor=IN)
    res = server.check_problem(str(p))
    assert res["ok"] and len(unit_lines(res["warnings"])) == 1 and "70 mm after x 25.4 (inches)" in unit_lines(res["warnings"])[0]
    bad = server.check_problem(str(scaled_copy(EXAMPLES["hook"], tmp_path / "e", IN)))     # voxel size kept: the one error has the hint
    assert not bad["ok"] and "70 mm after x 25.4 (inches)" in bad["error"]


def test_build_regions_check_has_the_fact_for_a_small_spec(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    spec = {"name": "tiny", "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [6, 4, 4]}}],
        "fixed": {"wall": [{"box": {"min": [-1, 0, 0], "max": [0, 4, 4]}}]},
        "load": {"tip": [{"box": {"min": [5, 1, 1], "max": [6, 3, 3]}}]}},
        "load_cases": [{"name": "a", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -1.0]}]}],
        "settings": {"voxel_size": 0.5, "iterations": 2}}
    r = server.build_regions(spec)
    assert r["ok"], r
    [fact] = unit_lines(r["warnings"])
    assert "is 6 mm, read as mm" in fact and "152.4 mm after x 25.4" in fact and "6000 mm after x 1000" in fact
