"""F4-40: a choice key with a list, a mapping, a number or null is a user error that names the key.

A raw YAML value was tested with `in` against a dict. A list or a mapping cannot be a dict key, so Python raised
`TypeError: unhashable type` and the CLI printed `internal error`. A string that is not a choice already gave a user error.
Every case here uses a public entry (`load_config`, `parse_print_block`, `PrintSettings.from_dict`, the part builder).
"""
import pytest

from test_input import make_case
from topopt.config import PRINTER_TYPES, PrintSettings, UserError, load_config, parse_print_block

ALLOWED = ", ".join(PRINTER_TYPES)
VALUES = [pytest.param(["fdm"], id="list"), pytest.param({"a": 1}, id="mapping"), pytest.param(5, id="number"),
          pytest.param(None, id="null")]


def problem(tmp_path, **blocks):
    return make_case(tmp_path, h=2.0, mutate=lambda d: d.update(blocks))


@pytest.mark.parametrize("value", VALUES)
def test_print_printer_type_that_is_not_a_string_is_a_user_error(tmp_path, value):
    with pytest.raises(UserError) as e:
        load_config(problem(tmp_path, print={"printer_type": value}))
    assert f"'print.printer_type' must be one of {ALLOWED}, got {value!r}" in str(e.value)


@pytest.mark.parametrize("value", VALUES)
def test_the_print_block_of_an_option_is_checked_the_same_way(value):
    """`topopt printcheck --set` and the MCP `print_settings` call `parse_print_block` with a dict."""
    with pytest.raises(UserError, match="'print.printer_type' must be one of"):
        parse_print_block({"printer_type": value}, "<options>")


@pytest.mark.parametrize("value", VALUES)
def test_joint_section_kind_that_is_not_a_string_is_a_user_error(tmp_path, value):
    section = {"kind": value, "d": 2}
    joints = [{"region": "tip", "to": [30, 5, 5], "section": section}]
    with pytest.raises(UserError) as e:
        load_config(problem(tmp_path, joints=joints))
    assert "'joints[0].section' must be a mapping with a kind: {kind: round, d}" in str(e.value)
    assert f"'kind': {value!r}" in str(e.value)


@pytest.mark.parametrize("value", VALUES)
def test_print_settings_from_run_json_falls_back_to_the_default(value):
    """`run.json` is a file that the program wrote: a bad value gives the default, as an unknown string does."""
    assert PrintSettings.from_dict({"printer_type": value}) == PrintSettings()
    assert PrintSettings.from_dict({"printer_type": "?"}) == PrintSettings()


@pytest.mark.parametrize("value", [pytest.param(["fdm"], id="list"), pytest.param({"a": 1}, id="mapping")])
def test_cli_check_gives_the_user_error_and_not_an_internal_error(tmp_path, capsys, value):
    from topopt.cli import main
    code = main(["check", str(problem(tmp_path, print={"printer_type": value}))])
    out = capsys.readouterr()
    text = out.out + out.err
    assert "internal error" not in text and "unhashable" not in text
    assert f"'print.printer_type' must be one of {ALLOWED}, got {value!r}" in text
    assert code == 1



# --- the part builder spec (topopt/regions_builder.py) ---------------------------------------------------------------


def spec(prism=None, region="tip"):
    prism = {"extent": [0, 10], "points": [[0, 0], [10, 0], [0, 10]], **(prism or {})}
    return {"name": "bar",
            "regions": {"design": [{"box": {"min": [0, 0, 0], "max": [20, 10, 10]}}, {"prism": prism}],
                        "fixed": {"wall": [{"box": {"min": [-2, 0, 0], "max": [0, 10, 10]}}]},
                        "load": {"tip": [{"box": {"min": [18, 3, 0], "max": [20, 7, 3]}}]}},
            "load_cases": [{"name": "down", "forces": [{"region": region, "force": [0, 0, -2.0]}]}],
            "settings": {"voxel_size": 2.0, "iterations": 5}}


@pytest.mark.parametrize("value", VALUES)
def test_prism_plane_that_is_not_a_string_is_a_spec_error(value):
    pytest.importorskip("manifold3d")
    from topopt.regions_builder import SpecError, build_all
    with pytest.raises(SpecError, match="prism.plane must be xy, xz or yz"):
        build_all(spec(prism={"plane": value}))


@pytest.mark.parametrize("value", [pytest.param(["tip"], id="list"), pytest.param({"a": 1}, id="mapping")])
def test_a_force_region_that_is_not_a_name_is_a_spec_error_with_the_same_text_as_an_unknown_name(tmp_path, value):
    pytest.importorskip("manifold3d")
    from topopt.regions_builder import SpecError, build_spec
    with pytest.raises(SpecError) as e:
        build_spec(spec(region=value), tmp_path / "out", say=lambda *a, **k: None)
    assert f"load_cases use region '{value!r}', but regions.load has no such name" in str(e.value)
    assert "unhashable" not in str(e.value)
