"""F4-41: a value of the wrong type in a part spec or in run.json is a user error that names the key.

The part builder gave `SpecError: TypeError: ...` or `SpecError: AttributeError: ...` with no key. `Interface.from_dict` (reads the
`interfaces` of run.json) gave `KeyError` for an unknown `kind` and `TypeError` for a list. Both are user errors now.
"""
import copy
import json
import re

import pytest
import yaml

from test_post import make_hole_run
from topopt import interfaces, post
from topopt import regions_builder as rb
from topopt.config import INTERFACE_KINDS, Interface, UserError

ALLOWED = ", ".join(INTERFACE_KINDS)
BORE = {"name": "b", "kind": "bore", "op": "cut", "stock_mm": 0.0, "point": [5, 5, 12], "axis": [0, 0, -1], "radius_mm": 2.0,
        "length_mm": 8.0}


# ---- Interface.from_dict (run.json) ------------------------------------------------------------------------------------------
KINDS = [pytest.param("sphere", id="unknown"), pytest.param(["bore"], id="list"), pytest.param({"a": 1}, id="mapping"),
         pytest.param(5, id="number"), pytest.param(None, id="null")]


@pytest.mark.parametrize("kind", KINDS)
def test_interface_kind_that_is_not_a_kind_is_a_user_error_with_the_key_and_the_kinds(kind):
    with pytest.raises(UserError) as e:
        Interface.from_dict(dict(BORE, kind=kind))
    assert f"'interface.kind' must be one of {ALLOWED}, got {kind!r}" in str(e.value)
    assert "run.json" in str(e.value)                                     # the options name the file


def test_interface_without_a_kind_is_a_user_error():
    d = {k: v for k, v in BORE.items() if k != "kind"}
    with pytest.raises(UserError, match=r"'interface.kind' must be one of .*, got None"):
        Interface.from_dict(d)


@pytest.mark.parametrize("entry", [["bore"], 5, None, "bore"], ids=["list", "number", "null", "string"])
def test_interface_entry_that_is_not_a_mapping_is_a_user_error(entry):
    with pytest.raises(UserError, match=r"'interface' must be a mapping, got "):
        Interface.from_dict(entry)


def test_a_valid_interface_reads_as_before():
    spec = Interface.from_dict(BORE)
    assert (spec.name, spec.kind, spec.op, spec.axis, spec.radius_mm, spec.length_mm) == ("b", "bore", "cut", (0.0, 0.0, -1.0), 2.0, 8.0)
    assert Interface.from_dict(spec.to_dict()) == spec
    stl = Interface.from_dict({"name": "s", "kind": "stl", "op": "union", "file": "s.stl"})
    assert stl.kind == "stl" and stl.op == "union"


@pytest.mark.parametrize("kind", ["sphere", ["bore"]], ids=["unknown", "list"])
def test_specs_from_run_names_the_index_of_the_entry(kind):
    run = {"yaml_path": "x.yaml", "interfaces": [BORE, dict(BORE, name="c", kind=kind)]}
    with pytest.raises(UserError) as e:
        interfaces.specs_from_run(run)
    assert f"'interfaces[1].kind' must be one of {ALLOWED}, got {kind!r}" in str(e.value)


def test_the_post_step_gives_the_user_error_and_writes_it_to_the_status(tmp_path):
    """post.finish is the caller of `specs_from_run`. A `UserError` goes up to `cmd_run` (exit 1, `error: ...`); the status keeps the text."""
    make_hole_run(tmp_path)
    raw = json.loads((tmp_path / "run.json").read_text())
    raw["interfaces"] = [dict(BORE, kind="sphere")]
    (tmp_path / "run.json").write_text(json.dumps(raw))
    with pytest.raises(UserError, match=r"interfaces\[0\]\.kind"):
        post.finish(tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "error" and "interfaces[0].kind" in status["message"]


def test_a_bad_kind_in_run_json_ends_the_cli_run_with_exit_1_and_one_line(tmp_path, monkeypatch, capsys):
    """The whole path: the run writes run.json, the post step reads it. `to_dict` is patched so that the stored kind is a list, as after a hand edit."""
    from test_interfaces import BORE as IF_BORE, _config_with
    from topopt.cli import main
    p = _config_with(tmp_path, [dict(IF_BORE)])
    real = Interface.to_dict
    monkeypatch.setattr(Interface, "to_dict", lambda self, yaml_dir=None: dict(real(self, yaml_dir), kind=["bore"]))
    out = tmp_path / "out"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 1
    err = capsys.readouterr().err
    assert f"error: 'interfaces[0].kind' must be one of {ALLOWED}, got ['bore']" in err and "internal error" not in err and "Traceback" not in err
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "error" and "interfaces[0].kind" in status["message"]


# ---- the part builder spec ---------------------------------------------------------------------------------------------------
SPEC = {
    "name": "shelf_bracket",
    "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [60, 20, 40]}},
                   {"subtract": {"sphere": {"center": [60, 10, 40], "radius": 5}}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 40]}}]},
        "keep_out": {"hole": [{"cylinder": {"p0": [-5, 10, 30], "p1": [5, 10, 30], "radius": 2}}]},
        "load": {"shelf": [{"capsule": {"p0": [50, 10, 8], "p1": [58, 10, 8], "radius": 3}},
                           {"box": {"min": [48, 4, 0], "max": [60, 16, 6]}}], "pad": [{"sphere": {"center": [30, 10, 40], "radius": 3}}]},
        "solid": {"s": [{"prism": {"plane": "xz", "extent": [-5, 5], "points": [[0, 0], [10, 0], [0, 10]]}}]},
    },
    "load_cases": [{"name": "static", "weight": 1.0,
                    "forces": [{"region": "shelf", "force": [0, 0, -30.0]}],
                    "pressures": [{"region": "pad", "value_mpa": 0.1}]}],
    "fixed_axes": {"wall": ["y"]},
    "springs": [{"region": "pad", "stiffness": [1, 1, 1]}],
    "assumptions": ["a"],
    "material": {"E": 3500, "nu": 0.35, "density": 1.24},
    "settings": {"volume_fraction": 0.3, "voxel_size": 2.0, "iterations": 50, "penal": 3.0, "target_active": 10000,
                 "allow_large_grid": False},
}
LIST, MAP, NULL, NUM, BOOL, TEXT = [1], {"a": 1}, None, 5, True, "abc"
A_NUMBER = "a number in mm, for example 3.0"
A_LIST_OF_PAIRS = "a list of [a, b] pairs in mm, for example [[0, 0], [10, 0], [0, 10]]"
FORCE = "{region: tip, force: [0, 0, -2]}"
CASE = "{name: hang, forces: [" + FORCE + "]}"
PRESSURE = "{region: pad, value_mpa: 0.1}"
SPRING = "{region: pad, stiffness: [100, 100, 100]}"
JOINT = "{region: pad, to: [30, 10, 60], section: {kind: round, d: 2}}"
SETTINGS = "a mapping, for example {volume_fraction: 0.25, voxel_size: auto, iterations: 80}"

# (the place in the spec, the bad values, the key path of the text, what the text expects)
SHAPES = [
    (("regions", "design", 1, "subtract", "sphere", "radius"), [LIST, MAP, NULL, BOOL, TEXT], "regions.design[1].subtract.sphere.radius", A_NUMBER),
    (("regions", "keep_out", "hole", 0, "cylinder", "radius"), [LIST, MAP, NULL, BOOL], "regions.keep_out.hole[0].cylinder.radius", A_NUMBER),
    (("regions", "load", "shelf", 0, "capsule", "radius"), [LIST, MAP, NULL, BOOL], "regions.load.shelf[0].capsule.radius", A_NUMBER),
    (("regions", "load", "pad", 0, "sphere", "radius"), [LIST, MAP, NULL, BOOL], "regions.load.pad[0].sphere.radius", A_NUMBER),
    (("regions", "solid", "s", 0, "prism", "points"), [NULL, NUM, BOOL, MAP, TEXT], "regions.solid.s[0].prism.points", A_LIST_OF_PAIRS),
    (("load_cases",), [NUM, BOOL, MAP, TEXT], "load_cases", f"a list of load cases, for example [{CASE}]"),
    (("load_cases", 0), [LIST, NULL, NUM, BOOL, TEXT], "load_cases[0]", f"a mapping, for example {CASE}"),
    (("load_cases", 0, "forces"), [MAP, NUM, BOOL, TEXT], "load_cases[0].forces", f"a list of mappings, for example [{FORCE}]"),
    (("load_cases", 0, "forces", 0), [LIST, NULL, NUM, BOOL, TEXT], "load_cases[0].forces[0]", f"a mapping, for example {FORCE}"),
    (("load_cases", 0, "pressures"), [MAP, NUM, BOOL, TEXT], "load_cases[0].pressures", f"a list of mappings, for example [{PRESSURE}]"),
    (("load_cases", 0, "pressures", 0), [LIST, NULL, NUM, BOOL], "load_cases[0].pressures[0]", f"a mapping, for example {PRESSURE}"),
    (("springs",), [MAP, NUM, BOOL, TEXT], "springs", f"a list of mappings, for example [{SPRING}]"),
    (("springs", 0), [LIST, NULL, NUM, BOOL], "springs[0]", f"a mapping, for example {SPRING}"),
    (("joints",), [MAP, NUM, BOOL, TEXT], "joints", f"a list of mappings, for example [{JOINT}]"),
    (("settings",), [LIST, NUM, BOOL, TEXT], "settings", SETTINGS),
    (("settings", "volume_fraction"), [LIST, MAP, NULL, BOOL, TEXT], "settings.volume_fraction", "a number, for example 0.25"),
    (("settings", "penal"), [LIST, MAP, NULL, BOOL], "settings.penal", "a number, for example 3.0"),
    (("settings", "voxel_size"), [LIST, MAP, NULL, BOOL, TEXT], "settings.voxel_size", "a number in mm, for example 2.0, or auto"),
    (("settings", "iterations"), [LIST, MAP, NULL, BOOL, TEXT], "settings.iterations", "a whole number, for example 80"),
    (("settings", "target_active"), [LIST, MAP, BOOL, TEXT], "settings.target_active", "a whole number of voxels, for example 40000"),
    (("material",), [LIST, NUM, BOOL, TEXT], "material", "a mapping, for example {E: 3500, nu: 0.35, density: 1.24}"),
    (("material", "E"), [LIST, MAP, NULL, BOOL, TEXT], "material.E", "a number in MPa, for example 3500"),
    (("material", "nu"), [LIST, NULL, BOOL], "material.nu", "a number, for example 0.35"),
    (("material", "density"), [MAP, NULL, BOOL], "material.density", "a number in g/cm3, for example 1.24"),
]
CASES = [pytest.param(path, value, key, expect, id=f"{key}={value!r}") for path, values, key, expect in SHAPES for value in values]


def spec_with(path, value):
    spec = copy.deepcopy(SPEC)
    where = spec
    for k in path[:-1]:
        where = where[k]
    where[path[-1]] = copy.deepcopy(value)
    return spec


def got(value):
    return "null" if value is None else f"{type(value).__name__} {value!r}"


@pytest.mark.parametrize("path,value,key,expect", CASES)
def test_a_spec_value_of_the_wrong_shape_is_a_spec_error_that_names_the_key(tmp_path, path, value, key, expect):
    with pytest.raises(rb.SpecError) as e:
        rb.build_spec(spec_with(path, value), tmp_path / "out", say=lambda *a: None)
    assert str(e.value) == f"{key} must be {expect}, got {got(value)}"
    assert not re.match(r"\w*(Error|Exception)\b", str(e.value))                  # not the text of a Python exception
    assert not (tmp_path / "out").exists()                                         # the check is before any work


def test_the_cli_prints_the_key_and_exits_with_1(tmp_path, capsys):
    from topopt.regions_builder import main
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(spec_with(("load_cases", 0, "forces", 0), 5)))
    assert main([str(spec), "--out", str(tmp_path / "out")]) == 1
    assert f"SPEC ERROR: load_cases[0].forces[0] must be a mapping, for example {FORCE}, got int 5" in capsys.readouterr().out
    spec.write_text(yaml.safe_dump(spec_with(("regions", "load", "shelf", 0, "capsule", "radius"), None)))
    assert main([str(spec), "--out", str(tmp_path / "out")]) == 1
    assert "SPEC ERROR: regions.load.shelf[0].capsule.radius must be a number in mm, for example 3.0, got null" in capsys.readouterr().out


def test_the_mcp_entry_gives_one_line_with_the_key(tmp_path):
    reply = rb.build_in_workspace(spec_with(("settings", "volume_fraction"), [0.3]), tmp_path)
    assert reply == {"ok": False, "error": "settings.volume_fraction must be a number, for example 0.25, got list [0.3]"}
    reply = rb.build_in_workspace(yaml.safe_dump(spec_with(("springs", 0), None)), tmp_path)
    assert reply["ok"] is False and reply["error"].startswith("springs[0] must be a mapping, for example {region: pad")


def test_a_long_value_is_cut_in_the_text(tmp_path):
    with pytest.raises(rb.SpecError) as e:
        rb.build_spec(spec_with(("settings", "volume_fraction"), list(range(500))), tmp_path / "out", say=lambda *a: None)
    assert len(str(e.value)) < 200 and "..." in str(e.value)


def test_a_yaml_anchor_that_holds_itself_is_a_spec_error(tmp_path):
    spec = yaml.safe_load("""
name: loop
load_cases: [{name: a, forces: [{region: t, force: [0, 0, -1]}]}]
regions:
  design: &d [{box: {min: [0, 0, 0], max: [1, 1, 1]}}, *d]
""")
    with pytest.raises(rb.SpecError, match=r"regions\.design\[1\] holds itself"):
        rb.build_spec(spec, tmp_path / "out", say=lambda *a: None)


def test_no_value_of_the_wrong_shape_gives_a_keyless_error_or_a_traceback(tmp_path, monkeypatch):
    """The probe of F4-41 as a guard: every value of SPEC replaced by a list, a mapping, null, a number, a bool and a text.

    The check of the written YAML is replaced (the loader has its own tests); the build of the shapes and the STL files is real.
    """
    monkeypatch.setattr(rb, "check_yaml", lambda path, say=print, allow_large_grid=False: (True, "ok"))

    def places(node, path=()):
        yield path
        if isinstance(node, dict):
            for k, v in node.items():
                yield from places(v, path + (k,))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from places(v, path + (i,))

    built = 0
    for path in list(places(SPEC))[1:]:
        for value in (LIST, MAP, NULL, NUM, BOOL, TEXT):
            try:
                rb.build_spec(spec_with(path, value), tmp_path / "out", say=lambda *a: None)
                built += 1
            except rb.SpecError as e:      # any other exception fails the test
                assert not re.match(r"\w*(Error|Exception)\b", str(e)), (path, value, str(e))
    assert built > 100                      # the values that are fine (a name, a vector item, an unused key) still build


def test_a_valid_spec_builds_as_before(tmp_path):
    """Numbers as text (YAML 1.1 reads 5e-1 as text), a null for an optional list and a second force build as they did."""
    spec = copy.deepcopy(SPEC)
    spec["settings"].update({"volume_fraction": "3e-1", "iterations": "50", "target_active": None})
    spec["regions"]["load"]["shelf"][0]["capsule"]["radius"] = "3"
    spec["load_cases"][0]["forces"].append({"region": "pad", "force": [0, 0, -1.0]})
    spec["material"]["nu"] = "0.35"
    spec["joints"] = None
    reply = rb.build_in_workspace(spec, tmp_path)
    assert reply["ok"] is True and reply["builder_ok"] is True and reply["check"] is not None
    doc = yaml.safe_load((tmp_path / reply["yaml_rel"]).read_text())
    assert doc["settings"]["voxel_size"] == 2.0 and doc["settings"]["iterations"] == 50
    assert doc["settings"]["volume_fraction"] == pytest.approx(0.3)             # the text "3e-1", read by float() as before
    assert doc["material"] == {"E": 3500.0, "nu": 0.35, "density": 1.24}
    assert doc["regions"]["load"] == {"shelf": "shelf.stl", "pad": "pad.stl"}
    assert doc["regions"]["fixed"] == [{"file": "wall.stl", "axes": ["y"]}] and doc["springs"] == SPEC["springs"]
    assert doc["assumptions"] == ["a"]
    plain = rb.build_in_workspace(SPEC, tmp_path, "plain")
    assert plain["ok"] is True and plain["name"] == "plain"


def test_the_shipped_examples_pass_the_shape_check():
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "part-to-topopt" / "examples"
    specs = sorted(folder.glob("*.yaml"))
    assert specs
    for f in specs:
        rb.check_spec_shape(yaml.safe_load(f.read_text()))
