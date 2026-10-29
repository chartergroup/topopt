"""The preferences file of the work folder (topopt/preferences.py, CONTRACTS.md section 10.2) and the tool `save_preferences`.

Covered: the file (round trip, each section, merge, the checks of the problem YAML, a file that a person broke, keys that are kept), the
key `preferences` of `environment`, the tool, the real server over stdio, the rule that a preference does not change a run by itself
(`run_topopt`, `check_problem` and the loader), `build_regions` (only a block that the spec does not have), the fact of `check_problem`,
and the texts (the tool description, the two copies of the skills).

Nothing here uses a GPU. The tests run one tiny problem (a few voxels) through the real `topopt run`; run pytest with a CPU OpenCL device
(pocl) if the machine has a GPU that you do not want to use.
"""
import asyncio
import copy
import inspect
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import preferences, runinfo, workfolder  # noqa: E402
from topopt.config import UserError, load_config, parse_material_block, parse_print_block, parse_voxel_size  # noqa: E402
from topopt_mcp import server  # noqa: E402

NOT_ROOT = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write in any folder")
FILE = preferences.FILE_NAME

ALL = {"print": {"printer_type": "fdm", "bed_mm": [256, 256], "build_direction": "+z", "self_supporting_angle_deg": 40},
       "material": {"E": 2100, "nu": 0.35, "density": 1.27, "name": "PETG"},
       "quality": {"voxel_size": 1.5},
       "outputs": ["renders", "step"]}


def load(folder):
    return yaml.safe_load((Path(folder) / FILE).read_text())


# --- the file ----------------------------------------------------------------------------------------------------------------

def test_round_trip_all_sections(tmp_path):
    r = preferences.save(tmp_path, **ALL)
    assert r["ok"] and r["created"] and r["changed"] == ["print", "material", "quality", "outputs"]
    assert Path(r["path"]) == tmp_path / FILE
    on_disk = load(tmp_path)
    assert on_disk["format"] == 1
    assert on_disk["print"] == {"printer_type": "fdm", "bed_mm": [256.0, 256.0], "build_direction": "+z", "self_supporting_angle_deg": 40.0}
    assert on_disk["material"] == {"E": 2100.0, "nu": 0.35, "density": 1.27, "name": "PETG"}
    assert on_disk["quality"] == {"voxel_size": 1.5} and on_disk["outputs"] == ["renders", "step"]
    read = preferences.read(tmp_path)
    assert read["exists"] and read["values"] == r["values"] == {k: v for k, v in on_disk.items() if k != "format"}
    assert "problems" not in read and "kept_not_used" not in read
    assert not list(tmp_path.glob("*.tmp"))        # written with runinfo.write_atomic: no part file is left


@pytest.mark.parametrize("section", ["print", "material", "quality", "outputs"])
def test_each_section_alone(tmp_path, section):
    r = preferences.save(tmp_path, **{section: ALL[section]})
    assert r["ok"] and r["changed"] == [section]
    assert set(load(tmp_path)) == {"format", section}
    assert set(preferences.read(tmp_path)["values"]) == {section}


def test_a_printer_type_and_a_scalar_bed_are_checked_and_stored_in_one_form(tmp_path):
    r = preferences.save(tmp_path, print={"printer_type": " SLA ", "bed_mm": 150, "build_direction": [0, 0, 2]})
    assert r["values"]["print"] == {"printer_type": "sla", "bed_mm": [150.0, 150.0], "build_direction": [0.0, 0.0, 2.0]}


def test_merge_keeps_what_is_not_given_and_replaces_key_by_key(tmp_path):
    preferences.save(tmp_path, **ALL)
    r = preferences.save(tmp_path, print={"bed_mm": [300, 200]})                # only the bed: the other print keys stay
    assert r["changed"] == ["print"]
    assert r["values"]["print"] == {"printer_type": "fdm", "bed_mm": [300.0, 200.0], "build_direction": "+z", "self_supporting_angle_deg": 40.0}
    assert r["values"]["material"] == preferences.read(tmp_path)["values"]["material"] and r["values"]["quality"] == {"voxel_size": 1.5}
    r = preferences.save(tmp_path, print={"self_supporting_angle_deg": None})  # null removes one key
    assert "self_supporting_angle_deg" not in r["values"]["print"] and r["changed"] == ["print"]
    r = preferences.save(tmp_path, quality={})                                  # {} removes a section
    assert "quality" not in r["values"] and "quality" not in load(tmp_path) and r["changed"] == ["quality"]
    r = preferences.save(tmp_path, outputs=[])                                  # [] is a value: none of the three
    assert r["values"]["outputs"] == [] and load(tmp_path)["outputs"] == [] and r["changed"] == ["outputs"]
    r = preferences.save(tmp_path, outputs={})                                  # {} removes the list
    assert "outputs" not in r["values"] and "outputs" not in load(tmp_path)


def test_a_call_that_changes_nothing_does_not_write(tmp_path):
    preferences.save(tmp_path, **ALL)
    before = (tmp_path / FILE).read_bytes()
    stamp = (tmp_path / FILE).stat().st_mtime_ns
    r = preferences.save(tmp_path, quality={"voxel_size": 1.5}, print={"printer_type": "fdm"})
    assert r["ok"] and r["changed"] == [] and not r["created"] and "Nothing changed" in r["note"]
    assert (tmp_path / FILE).read_bytes() == before and (tmp_path / FILE).stat().st_mtime_ns == stamp


def test_no_section_makes_an_empty_file_once(tmp_path):
    r = preferences.save(tmp_path)
    assert r["ok"] and r["created"] and r["changed"] == [] and r["values"] == {}
    assert "defaults" in r["note"]
    assert load(tmp_path) == {"format": 1}
    read = preferences.read(tmp_path)
    assert read["exists"] is True and read["values"] == {} and "problems" not in read
    assert preferences.save(tmp_path)["created"] is False


def test_the_folder_is_made_when_it_is_not_there(tmp_path):
    folder = tmp_path / "new" / "work"
    assert preferences.save(folder, quality={"voxel_size": 2})["ok"] and (folder / FILE).is_file()


# --- the checks: the same as the problem YAML ----------------------------------------------------------------------------------

BAD = [
    ("print", "printer_type", "laser"), ("print", "printer_type", 3), ("print", "bed_mm", "wide"), ("print", "bed_mm", 0), ("print", "bed_mm", -5),
    ("print", "bed_mm", [1, 2, 3]), ("print", "bed_mm", [100, 0]), ("print", "bed_mm", True), ("print", "build_direction", "q"),
    ("print", "build_direction", [0, 0, 0]), ("print", "build_direction", [1, 2]), ("print", "build_direction", 5),
    ("print", "self_supporting_angle_deg", 91), ("print", "self_supporting_angle_deg", -1), ("print", "self_supporting_angle_deg", "steep"),
    ("material", "E", 0), ("material", "E", -1), ("material", "E", "stiff"), ("material", "nu", 0.5), ("material", "nu", -1),
    ("material", "nu", "x"), ("material", "density", 0), ("material", "density", True), ("quality", "voxel_size", 0),
    ("quality", "voxel_size", -1), ("quality", "voxel_size", "fine"), ("quality", "voxel_size", True),
]


@pytest.mark.parametrize("section, key, value", BAD)
def test_a_bad_value_is_refused_with_the_key_the_value_and_the_rule(tmp_path, section, key, value):
    r = preferences.save(tmp_path, **{section: {key: value}})
    assert r["ok"] is False and "wrote nothing" in r["error"]
    assert f"{section}.{key}" in r["error"] and not (tmp_path / FILE).exists()
    shown = [value] + (value if isinstance(value, list) else [])                  # a list: the whole list, or the one item that is wrong
    assert any(repr(v) in r["error"] for v in shown) or value == [0, 0, 0], r["error"]
    # the problem YAML says no for the same value
    with pytest.raises(UserError):
        if section == "print":
            parse_print_block({key: value})
        elif section == "material":
            parse_material_block({key: value})
        else:
            parse_voxel_size(value)


@pytest.mark.parametrize("section, key, value", [
    ("print", "printer_type", "sls"), ("print", "bed_mm", [250, 210]), ("print", "bed_mm", 180), ("print", "build_direction", "-x"),
    ("print", "build_direction", [1, 1, 0]), ("print", "self_supporting_angle_deg", 0), ("print", "self_supporting_angle_deg", 90),
    ("material", "E", 1.5), ("material", "nu", 0.49), ("material", "nu", -0.5), ("material", "density", 7.8), ("quality", "voxel_size", 0.2)])
def test_a_good_value_is_accepted_when_the_problem_yaml_accepts_it(tmp_path, section, key, value):
    assert preferences.save(tmp_path, **{section: {key: value}})["ok"]
    if section == "print":
        parse_print_block({key: value})
    elif section == "material":
        parse_material_block({key: value})
    else:
        parse_voxel_size(value)


def test_the_material_name_is_free_text_but_not_empty(tmp_path):
    assert preferences.save(tmp_path, material={"name": "  PLA+ "})["values"]["material"] == {"name": "PLA+"}
    for bad in ("", "   ", 5, ["PLA"]):
        r = preferences.save(tmp_path, material={"name": bad})
        assert r["ok"] is False and "material.name" in r["error"]


def test_an_unknown_key_in_the_input_is_refused_and_names_the_allowed_keys(tmp_path):
    r = preferences.save(tmp_path, print={"nozzle_mm": 0.4}, material={"colour": "red"}, quality={"fast": True})
    assert r["ok"] is False
    for text in ("print.nozzle_mm", "material.colour", "quality.fast", "printer_type, bed_mm, build_direction, self_supporting_angle_deg",
                 "E, nu, density, filament, name", "Allowed keys: voxel_size"):
        assert text in r["error"], text
    assert not (tmp_path / FILE).exists()


@pytest.mark.parametrize("key", ["min_wall_mm", "min_wall_area_pct", "max_closed_void_mm3", "max_bodies", "max_overhang_area_mm2",
                                 "max_support_volume_mm3", "min_plate_contact_mm2"])
def test_a_print_limit_belongs_to_one_part_and_is_refused(tmp_path, key):
    r = preferences.save(tmp_path, print={key: 1})
    assert r["ok"] is False and f"print.{key}" in r["error"] and "belongs to one part" in r["error"]


def test_an_unknown_output_name_is_refused_and_the_allowed_names_are_listed(tmp_path):
    r = preferences.save(tmp_path, outputs=["renders", "pdf"])
    assert r["ok"] is False and "'pdf'" in r["error"]
    assert "renders (render_part)" in r["error"] and "step (export_step)" in r["error"] and "verify (verify_run)" in r["error"]
    assert preferences.save(tmp_path, outputs="renders")["ok"] is False                      # a text is not a list
    assert preferences.save(tmp_path, outputs=[" Renders ", "renders", "VERIFY"])["values"]["outputs"] == ["renders", "verify"]
    assert not (tmp_path / "nothing").exists()


def test_a_section_of_the_wrong_type_is_refused(tmp_path):
    r = preferences.save(tmp_path, print="fdm")
    assert r["ok"] is False and "print" in r["error"] and "'fdm'" in r["error"]


def test_one_wrong_value_stops_the_whole_call(tmp_path):
    preferences.save(tmp_path, quality={"voxel_size": 2})
    before = (tmp_path / FILE).read_bytes()
    assert preferences.save(tmp_path, quality={"voxel_size": 1}, material={"E": -1})["ok"] is False
    assert (tmp_path / FILE).read_bytes() == before


# --- a file that a person broke, and keys that are kept ------------------------------------------------------------------------

def put(folder, text):
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / FILE).write_text(text)


@pytest.mark.parametrize("text, expect", [
    ("print: [unclosed\n  - x: : :\n", "not valid YAML"),
    ("- just\n- a list\n", "must hold a mapping"),
    ("42\n", "must hold a mapping"),
    ("format: 2\nquality: {voxel_size: 1}\n", "format"),
    ("format: true\n", "format"),
    ("print: fdm\n", "print: must be a mapping"),
    ("print: {bed_mm: wide}\n", "print.bed_mm"),
    ("print: {printer_type: [a, b]}\n", "print.printer_type"),
    ("material: {E: stiff, nu: 0.9}\n", "material.E"),
    ("quality: {voxel_size: -2}\n", "quality.voxel_size"),
    ("quality: 3\n", "quality: must be a mapping"),
    ("outputs: renders\n", "outputs: must be a list"),
    ("outputs: [renders, 5]\n", "5 is not an output name"),
])
def test_a_broken_file_is_a_fact_in_problems_and_never_a_crash(tmp_path, text, expect):
    put(tmp_path, text)
    read = preferences.read(tmp_path)
    assert read["exists"] is True and "values" not in read
    assert any(expect in p for p in read["problems"]), read["problems"]
    assert "problems" in read["note"]


def test_all_problems_of_a_file_are_listed(tmp_path):
    put(tmp_path, "print: {bed_mm: wide, printer_type: laser}\nmaterial: {E: -1}\noutputs: [5]\n")
    problems = preferences.read(tmp_path)["problems"]
    assert len(problems) == 4
    for key in ("print.bed_mm", "print.printer_type", "material.E", "outputs"):
        assert any(key in p for p in problems), key


def test_a_file_that_cannot_be_read_is_a_fact(tmp_path):
    (tmp_path / FILE).mkdir()                                                   # a folder with the name of the file
    read = preferences.read(tmp_path)
    assert read["exists"] is True and "cannot read" in read["problems"][0]
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / FILE).write_bytes(b"\xff\xfe\x00bad")                   # not text
    assert "problems" in preferences.read(tmp_path / "bin")


def fake_unreadable(monkeypatch, platform):
    """`read_text` of the preferences file raises the error that macOS gives when Files and Folders blocks the read."""
    real = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.name == FILE:
            raise PermissionError(1, "Operation not permitted")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(preferences.sys, "platform", platform)


def test_a_read_that_macos_blocks_names_the_privacy_setting_and_is_not_called_a_broken_file(tmp_path, monkeypatch):
    """R1 m2: the file may be good; the server cannot see it. No advice to remove it, and the same setting as the write path."""
    (tmp_path / FILE).write_text("format: 1\nquality:\n  voxel_size: 2\n")
    fake_unreadable(monkeypatch, "darwin")
    read = preferences.read(tmp_path)
    assert read["exists"] is True and "values" not in read
    assert "cannot read the file: Operation not permitted" in read["problems"][0] and workfolder.MAC_PRIVACY in read["problems"][0]
    assert "cannot read the file" in read["note"] and "Fix the file" not in read["note"] and "valid YAML" not in read["note"]
    r = preferences.save(tmp_path, quality={"voxel_size": 3})
    assert r["ok"] is False and "did not change" in r["error"] and workfolder.MAC_PRIVACY in r["error"]
    assert "remove" not in r["error"].lower() and "Fix the file" not in r["error"] and "valid YAML" not in r["error"]
    monkeypatch.undo()
    assert preferences.read(tmp_path)["values"] == {"quality": {"voxel_size": 2.0}}      # the file was not changed
    fake_unreadable(monkeypatch, "linux")                                                # on another system there is no macOS text
    assert workfolder.MAC_PRIVACY not in preferences.read(tmp_path)["problems"][0]
    assert workfolder.MAC_PRIVACY not in preferences.save(tmp_path, quality={"voxel_size": 3})["error"]


def test_a_name_that_is_a_folder_and_a_file_that_is_not_text_are_broken_files_that_a_person_can_remove(tmp_path):
    (tmp_path / FILE).mkdir()
    r = preferences.save(tmp_path, quality={"voxel_size": 3})
    assert r["ok"] is False and "cannot read the file" in r["error"] and "remove it" in r["error"]
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / FILE).write_bytes(b"\xff\xfe\x00bad")
    r = preferences.save(tmp_path / "bin", quality={"voxel_size": 3})
    assert r["ok"] is False and "not valid UTF-8" in r["error"] and "remove it" in r["error"]


def test_a_folder_that_is_a_file_has_no_preferences(tmp_path):
    (tmp_path / "work").write_text("x")
    assert preferences.read(tmp_path / "work")["exists"] is False


@pytest.mark.parametrize("text", ["print: [unclosed\n  - x: : :\n", "- a\n- list\n", "format: 7\n", "\x00\x01 not yaml: [\n"])
def test_save_does_not_overwrite_a_file_that_it_cannot_read(tmp_path, text):
    put(tmp_path, text)
    before = (tmp_path / FILE).read_bytes()
    r = preferences.save(tmp_path, quality={"voxel_size": 2})
    assert r["ok"] is False and str(tmp_path / FILE) in r["error"] and "did not change" in r["error"]
    assert (tmp_path / FILE).read_bytes() == before and not list(tmp_path.glob("*.tmp"))


def test_the_error_of_a_file_that_is_not_yaml_is_named(tmp_path):
    put(tmp_path, "print: [unclosed\n")
    r = preferences.save(tmp_path, quality={"voxel_size": 2})
    assert "not valid YAML" in r["error"] and "expected" in r["error"]


def test_a_file_with_a_wrong_value_can_be_repaired_with_save(tmp_path):
    put(tmp_path, "print: {bed_mm: wide, printer_type: sla}\nquality: {voxel_size: 2}\n")
    r = preferences.save(tmp_path, print={"bed_mm": 200})
    assert r["ok"] and "problems" not in r and r["values"]["print"] == {"printer_type": "sla", "bed_mm": [200.0, 200.0]}
    assert r["values"]["quality"] == {"voxel_size": 2.0}


@pytest.mark.parametrize("fmt", ["1.0", "true", "'1'", "2", "[1]"])
def test_the_format_must_be_the_integer_one(tmp_path, fmt):
    """R1 n6: `format: 1.0` (or true, or the text '1') is not format 1."""
    put(tmp_path, f"format: {fmt}\nquality: {{voxel_size: 2}}\n")
    read = preferences.read(tmp_path)
    assert "values" not in read and read["problems"][0].startswith("format: is ")
    before = (tmp_path / FILE).read_bytes()
    r = preferences.save(tmp_path, quality={"voxel_size": 3})
    assert r["ok"] is False and "format is" in r["error"] and (tmp_path / FILE).read_bytes() == before
    put(tmp_path, "format: 1\nquality: {voxel_size: 2}\n")                       # the integer 1 is read and written
    assert preferences.read(tmp_path)["values"] == {"quality": {"voxel_size": 2.0}}
    assert preferences.save(tmp_path, quality={"voxel_size": 3})["ok"]


def test_save_reports_what_is_still_wrong(tmp_path):
    put(tmp_path, "print: {bed_mm: wide}\nquality: {voxel_size: 2}\n")
    r = preferences.save(tmp_path, quality={"voxel_size": 3})
    assert r["ok"] and r["changed"] == ["quality"] and "print.bed_mm" in r["problems"][0]
    assert load(tmp_path)["print"] == {"bed_mm": "wide"}                        # the key of the person stays as it is


def test_an_unknown_output_name_in_the_file_is_kept_not_used_and_the_known_names_are_used(tmp_path):
    """R1 m4: a name of a later format must not make the whole file unused (the rule for keys of a later format)."""
    put(tmp_path, "format: 1\nprint: {bed_mm: 200}\noutputs: [Renders, pdf, step, pdf]\n")
    read = preferences.read(tmp_path)
    assert "problems" not in read and read["kept_not_used"] == ["outputs.pdf"]
    assert read["values"] == {"print": {"bed_mm": [200.0, 200.0]}, "outputs": ["renders", "step"]}
    assert "Kept, not used: outputs.pdf" in read["note"]
    _, values, problems = preferences.usable(tmp_path)                           # a tool may use the file: the bed and the known names
    assert problems == [] and values == read["values"]
    r = preferences.save(tmp_path, quality={"voxel_size": 2})                     # a save that does not give outputs leaves the list as it is
    assert r["ok"] and r["kept_not_used"] == ["outputs.pdf"] and "problems" not in r
    assert load(tmp_path)["outputs"] == ["Renders", "pdf", "step", "pdf"]
    # the same name in the INPUT of save_preferences is still a refusal; a wrong type in the file is still a problem
    assert preferences.save(tmp_path, outputs=["pdf"])["ok"] is False
    put(tmp_path, "outputs: [renders, 5]\n")
    assert "not an output name" in preferences.read(tmp_path)["problems"][0]
    put(tmp_path, "outputs: [pdf]\n")                                            # only a name that this server does not know: none of the three
    assert preferences.read(tmp_path)["values"] == {"outputs": []}


def test_a_save_that_leaves_problems_does_not_say_the_values_are_live_and_names_the_keys(tmp_path):
    """R1 m3: the server uses no value from a file with problems, so the note of the save must not say "live now"."""
    put(tmp_path, "print: {bed_mm: wide, printer_type: foo}\nquality: {voxel_size: 2}\noutputs: [5]\n")
    r = preferences.save(tmp_path, quality={"voxel_size": 3})
    note = r["note"]
    assert r["ok"] and r["changed"] == ["quality"] and "live now" not in note
    assert "still has problems with print.bed_mm, print.printer_type, outputs" in note
    assert "uses no value from the file until they are fixed" in note
    assert preferences.read(tmp_path).get("values") is None                      # what the note says is what environment shows
    again = preferences.save(tmp_path, quality={"voxel_size": 3})                # no change: the problems are still said
    assert again["note"].startswith("Nothing changed") and "still has problems with print.bed_mm" in again["note"]
    preferences.save(tmp_path, print={"bed_mm": 200, "printer_type": "fdm"}, outputs=["renders"])    # repaired: the old note
    fixed = preferences.save(tmp_path, quality={"voxel_size": 2})
    assert "problems" not in fixed and fixed["note"] == "Saved. The server reads the file at each call, so the values are live now."


def test_unknown_keys_of_the_file_are_kept_and_named_and_do_not_break_it(tmp_path):
    put(tmp_path, "format: 1\nnotes: for the printer in the garage\nprint:\n  bed_mm: 200\n  nozzle_mm: 0.4\nfuture: {a: 1}\n")
    read = preferences.read(tmp_path)
    assert "problems" not in read and read["values"] == {"print": {"bed_mm": [200.0, 200.0]}}
    assert set(read["kept_not_used"]) == {"notes", "future", "print.nozzle_mm"}
    assert "Kept, not used" in read["note"] and "notes" in read["note"]
    r = preferences.save(tmp_path, quality={"voxel_size": 2}, print={"printer_type": "sls"})
    assert r["ok"] and set(r["kept_not_used"]) == {"notes", "future", "print.nozzle_mm"} and "problems" not in r
    kept = load(tmp_path)
    assert kept["notes"] == "for the printer in the garage" and kept["future"] == {"a": 1}
    assert kept["print"] == {"printer_type": "sls", "bed_mm": 200, "nozzle_mm": 0.4}      # a key that is not given stays as the person wrote it
    preferences.save(tmp_path, print={})                                         # {} removes the known keys; the key of the person stays
    assert load(tmp_path)["print"] == {"nozzle_mm": 0.4}
    assert load(tmp_path)["notes"] == "for the printer in the garage"


def test_an_empty_or_blank_file_is_valid(tmp_path):
    put(tmp_path, "")
    assert preferences.read(tmp_path)["values"] == {}
    put(tmp_path, "# only a comment\nprint:\n")
    assert preferences.read(tmp_path)["values"] == {}
    assert preferences.save(tmp_path, quality={"voxel_size": 2})["ok"]


def test_a_write_that_fails_is_a_refusal_and_leaves_no_part_file(tmp_path, monkeypatch):
    def boom(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(runinfo, "write_atomic", boom)
    r = preferences.save(tmp_path, quality={"voxel_size": 2})
    assert r["ok"] is False and "No space left on device" in r["error"] and not list(tmp_path.iterdir())


def test_a_save_replaces_a_link_at_the_file_name_and_does_not_write_what_it_pointed_to(tmp_path):
    """R1 n4 (the README and CONTRACTS 10.2 say it): the link is read through, a save makes a plain file."""
    target = tmp_path / "elsewhere.yaml"
    target.write_text("format: 1\nquality: {voxel_size: 4}\n")
    work = tmp_path / "work"
    work.mkdir()
    try:
        (work / FILE).symlink_to(target)
    except OSError:
        pytest.skip("this system makes no symlink")
    assert preferences.read(work)["values"] == {"quality": {"voxel_size": 4.0}}          # read through the link
    before = target.read_bytes()
    assert preferences.save(work, quality={"voxel_size": 2})["ok"]
    assert not (work / FILE).is_symlink() and load(work)["quality"] == {"voxel_size": 2.0}
    assert target.read_bytes() == before


def test_a_write_that_macos_blocks_names_the_privacy_setting(tmp_path, monkeypatch):
    def blocked(path, data):
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(runinfo, "write_atomic", blocked)
    monkeypatch.setattr(preferences.sys, "platform", "darwin")
    r = preferences.save(tmp_path, quality={"voxel_size": 2})
    assert r["ok"] is False and "Operation not permitted" in r["error"] and workfolder.MAC_PRIVACY in r["error"]
    monkeypatch.setattr(preferences.sys, "platform", "linux")
    assert workfolder.MAC_PRIVACY not in preferences.save(tmp_path, quality={"voxel_size": 2})["error"]


def test_the_file_is_written_with_write_atomic(tmp_path, monkeypatch):
    seen = []
    real = runinfo.write_atomic
    monkeypatch.setattr(runinfo, "write_atomic", lambda p, d: (seen.append(Path(p)), real(p, d)))
    preferences.save(tmp_path, quality={"voxel_size": 2})
    assert seen == [tmp_path / FILE]


# --- environment.preferences and the tool ----------------------------------------------------------------------------------------

@pytest.fixture
def work(tmp_path, monkeypatch):
    """The server with a work folder of this test. The environment probe (OpenCL and more) is replaced by a small dict."""
    folder = tmp_path / "work"
    folder.mkdir()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(folder))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)
    monkeypatch.chdir(tmp_path)
    import topopt.environment as env_module
    monkeypatch.setattr(env_module, "environment_data", lambda yaml=None: {"python": {"executable": sys.executable}, "ok": None})
    return folder


def test_environment_without_a_file_says_so_in_one_sentence(work):
    p = server.environment()["preferences"]
    assert p["path"] == str(work / FILE) and p["exists"] is False and "values" not in p and "problems" not in p
    assert p["note"].count(".") == 1 and "No preferences are saved" in p["note"] and "save_preferences keeps the answers of the user" in p["note"]


def test_environment_with_a_valid_file(work):
    preferences.save(work, **ALL)
    p = server.environment()["preferences"]
    assert p["exists"] is True and p["values"]["print"]["bed_mm"] == [256.0, 256.0] and p["values"]["outputs"] == ["renders", "step"]
    assert "problems" not in p and "reads this file at each call" in p["note"]


def test_environment_with_a_broken_file_gives_problems_and_no_crash(work):
    put(work, "print: [unclosed\n")
    reply = server.environment()
    assert reply["ok"] is True and reply["preferences"]["exists"] is True and "values" not in reply["preferences"]
    assert "not valid YAML" in reply["preferences"]["problems"][0]
    put(work, "material: {E: -3}\n")
    assert "material.E" in server.environment()["preferences"]["problems"][0]


def test_the_descriptions_of_environment_build_regions_and_check_problem_are_not_changed_by_this_item():
    """The reply keys `preferences` and `preferences_facts` carry the text, and `save_preferences` explains the rest. Other PRs add text to
    these three descriptions (P9-14 to environment, P9-8 to build_regions, E1-P3 to check_problem); with ours too they would pass 4,096."""
    tools = {t.name: t.description for t in asyncio.run(server.mcp.list_tools())}
    for name in ("environment", "build_regions", "check_problem"):
        assert len(tools[name]) <= 4096 and "preferences" not in tools[name], name


def test_the_tool_saves_the_file_in_the_work_folder_and_nowhere_else(work):
    r = server.save_preferences(print={"printer_type": "fdm", "bed_mm": [256, 256]}, material={"E": 2100, "name": "PETG"},
                                quality={"voxel_size": 1.5}, outputs=["renders", "step"])
    assert r["ok"] and Path(r["path"]) == work / FILE and r["changed"] == ["print", "material", "quality", "outputs"]
    assert sorted(p.name for p in work.iterdir()) == [FILE]                      # no other file, no part file, no folder
    assert sorted(p.name for p in work.parent.iterdir()) == ["work"]
    assert server.environment()["preferences"]["values"] == r["values"]          # the next call reads it: no restart


def test_the_tool_has_no_path_argument():
    tool = {t.name: t for t in asyncio.run(server.mcp.list_tools())}["save_preferences"]
    assert list(tool.input_schema["properties"]) == ["print", "material", "quality", "outputs", "filaments"]
    assert not tool.input_schema.get("required")


def test_the_tool_refuses_a_wrong_value_and_a_bad_file_without_a_crash(work):
    r = server.save_preferences(quality={"voxel_size": 0})
    assert r["ok"] is False and "quality.voxel_size" in r["error"] and not (work / FILE).exists()
    put(work, "- not a mapping\n")
    r = server.save_preferences(quality={"voxel_size": 2})
    assert r["ok"] is False and "did not change" in r["error"] and (work / FILE).read_text() == "- not a mapping\n"


def test_the_tool_refuses_a_work_folder_that_has_a_fix_needed_text(tmp_path, monkeypatch):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("TOPOPT_WORKSPACE", "${FOO}/topopt")                      # a variable that nobody replaced
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    text = workfolder.resolve().fix_needed[0]
    r = server.save_preferences(quality={"voxel_size": 2})
    assert r["ok"] is False and r["error"] == text and workfolder.FIX_OTHER in r["error"] and '"${FOO}/topopt"' in r["error"]
    assert "work_folder_note" not in r                                           # the error holds the text: it is not said twice
    assert not list(cwd.iterdir()) and not list(home.rglob(FILE))                # the file is nowhere
    monkeypatch.setenv("TOPOPT_DEFAULT_WORKSPACE", "${HOME}/Documents/topopt")   # the Desktop extension
    r = server.save_preferences(quality={"voxel_size": 2})
    assert r["ok"] is False and workfolder.FIX_DESKTOP in r["error"] and not list(home.rglob(FILE)) and not list(cwd.iterdir())


@NOT_ROOT
def test_the_tool_refuses_a_work_folder_that_cannot_be_written(tmp_path, monkeypatch):
    locked = tmp_path / "locked"
    locked.mkdir()
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(locked))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    locked.chmod(0o500)
    try:
        r = server.save_preferences(quality={"voxel_size": 2})
        assert r["ok"] is False and "cannot" in r["error"] and str(locked) in r["error"] and not list(locked.iterdir())
    finally:
        locked.chmod(0o700)


# --- the real server over stdio ------------------------------------------------------------------------------------------------

def payload(res):
    data = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if data:
        return data.get("result", data) if set(data) == {"result"} else data
    return json.loads(res.content[0].text)


def tiny_spec(**settings):
    return {"name": "tiny", "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [20, 10, 10]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 10, 10]}}]},
        "load": {"tip": [{"box": {"min": [16, 2, 0], "max": [20, 8, 6]}}]}},
        "load_cases": [{"name": "a", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
        "settings": {"iterations": 3, **settings}}


def test_stdio_server_saves_the_file_and_reads_it_at_the_next_call_with_no_restart(tmp_path):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    work = tmp_path / "work"
    env = dict(os.environ, TOPOPT_WORKSPACE=str(work), PYTHONPATH=str(REPO))
    env.pop("TOPOPT_DEFAULT_WORKSPACE", None)
    env.pop("TOPOPT_PYTHON", None)
    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], env=env, cwd=str(tmp_path))
    errfile = tmp_path / "server.stderr"
    out = {}

    async def go():
        with open(errfile, "w") as errlog:
            async with stdio_client(params, errlog=errlog) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    tools = {t.name for t in (await session.list_tools()).tools}
                    out["tool_listed"] = "save_preferences" in tools
                    out["env_before"] = payload(await session.call_tool("environment", {}))["preferences"]
                    out["saved"] = payload(await session.call_tool("save_preferences", {
                        "print": {"printer_type": "fdm", "bed_mm": [256, 256]},
                        "material": {"E": 2100, "nu": 0.35, "density": 1.27, "name": "PETG"},
                        "quality": {"voxel_size": 2.0}, "outputs": ["renders", "step"]}))
                    out["file_after_save"] = (work / FILE).read_text() if (work / FILE).is_file() else None
                    out["env_after"] = payload(await session.call_tool("environment", {}))["preferences"]
                    out["built"] = payload(await session.call_tool("build_regions", {"spec": tiny_spec()}))
                    # a person edits the file between two calls: the server has no restart
                    doc = yaml.safe_load(out["file_after_save"])
                    doc["print"]["bed_mm"] = [300.0, 200.0]
                    (work / FILE).write_text(yaml.safe_dump(doc))
                    yaml_rel = out["built"]["yaml_rel"]
                    out["checked"] = payload(await session.call_tool("check_problem", {"yaml_path": yaml_rel}))
                    out["bad"] = payload(await session.call_tool("save_preferences", {"print": {"bed_mm": "wide"}}))

    asyncio.run(asyncio.wait_for(go(), 600))
    assert out["tool_listed"]
    assert out["env_before"]["exists"] is False and "save_preferences" in out["env_before"]["note"]
    assert out["saved"]["ok"] and Path(out["saved"]["path"]) == work / FILE and out["file_after_save"].startswith("# topopt preferences")
    assert sorted(p.name for p in work.iterdir() if p.is_file()) == [FILE]
    assert out["env_after"]["exists"] and out["env_after"]["values"] == out["saved"]["values"]
    built = out["built"]
    assert built["ok"] and len(built["preferences_facts"]) == 3 and built["voxel_size"] == 2.0
    doc = yaml.safe_load((work / built["yaml_rel"]).read_text())
    assert doc["print"]["bed_mm"] == [256.0, 256.0] and doc["material"]["E"] == 2100.0
    facts = out["checked"]["preferences_facts"]                                  # the second call read the new bed: 300 x 200
    assert len(facts) == 1 and "256 x 256 mm in the YAML, 300 x 200 mm in the preferences" in facts[0]
    assert out["bad"]["ok"] is False and "print.bed_mm" in out["bad"]["error"]


def test_the_tool_schema_forbids_extra_arguments_and_every_tool_does():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert tools["save_preferences"].input_schema["additionalProperties"] is False
    assert [name for name, t in tools.items() if t.input_schema.get("additionalProperties") is not False] == []


def test_stdio_server_refuses_an_argument_name_that_the_tool_does_not_have_and_makes_no_file(tmp_path):
    """B1 of the review: `{"preferences": {...}}` or `{"printer": {...}}` was dropped by the SDK, and a fresh folder got the "keep the
    defaults" file with no value. Each of these calls must be refused with the tool's own reply, name the key, and make no file."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, TOPOPT_WORKSPACE=str(work), PYTHONPATH=str(REPO))
    env.pop("TOPOPT_DEFAULT_WORKSPACE", None)
    env.pop("TOPOPT_PYTHON", None)
    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], env=env, cwd=str(tmp_path))
    elsewhere = tmp_path / "elsewhere.yaml"
    calls = {
        "preferences": {"preferences": {"print": {"printer_type": "fdm", "bed_mm": [256, 256]}, "material": {"name": "PETG"}}},
        "printer": {"printer": {"bed_mm": 300}},
        "path": {"path": str(elsewhere), "quality": {"voxel_size": 2.0}},          # a good section next to a wrong name: still refused
    }
    out = {}

    async def go():
        with open(tmp_path / "server.stderr", "w") as errlog:
            async with stdio_client(params, errlog=errlog) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    out["schema"] = {t.name: t.input_schema for t in (await session.list_tools()).tools}["save_preferences"]
                    for key, args in calls.items():
                        res = await session.call_tool("save_preferences", args)
                        out[key] = payload(res), bool(getattr(res, "is_error", False)), sorted(p.name for p in tmp_path.rglob("*") if p.is_file()
                                                                                         and p.name != "server.stderr")
                    out["good"] = payload(await session.call_tool("save_preferences", {"quality": {"voxel_size": 2.0}}))

    asyncio.run(asyncio.wait_for(go(), 600))
    assert out["schema"]["additionalProperties"] is False
    for key in calls:
        reply, is_error, files = out[key]
        assert reply["ok"] is False and is_error is False, key                      # the tool's own refusal, not an SDK error
        assert f"'{key}'" in reply["error"] and "print, material, quality, outputs" in reply["error"], reply
        assert files == [], (key, files)                                            # no preferences file, no elsewhere.yaml
    assert out["good"]["ok"] is True and out["good"]["created"] is True            # the same server still saves a good call
    assert (work / FILE).is_file() and not elsewhere.exists()

# --- a preference does not change a run by itself -------------------------------------------------------------------------------

PREFS_FAR = {"print": {"printer_type": "sla", "bed_mm": [999, 999], "build_direction": "+x", "self_supporting_angle_deg": 10},
             "material": {"E": 123456, "nu": 0.3, "density": 9.9, "name": "other"}, "quality": {"voxel_size": 0.7}, "outputs": ["verify"]}


@pytest.fixture(scope="module")
def tiny_yaml(tmp_path_factory):
    """A problem YAML with no print block and the default material, built in a work folder that has no preferences file."""
    folder = tmp_path_factory.mktemp("guard_build")
    old = {k: os.environ.get(k) for k in ("TOPOPT_WORKSPACE", "TOPOPT_DEFAULT_WORKSPACE")}
    os.environ["TOPOPT_WORKSPACE"] = str(folder)
    os.environ.pop("TOPOPT_DEFAULT_WORKSPACE", None)
    try:
        built = server.build_regions(tiny_spec(voxel_size=2.0))
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    assert built["ok"] and "preferences_facts" not in built, built
    return folder / built["yaml_rel"]


def test_the_loader_and_the_preferences_use_the_same_voxel_size_rule(tiny_yaml, monkeypatch):
    """R1 n1: one helper for the rule, no copy. A recorder in its place is called by the loader and by the preferences check."""
    from topopt import config
    calls = []
    real = config._voxel_size

    def recorder(ctx, value, where="settings.voxel_size"):
        calls.append(where)
        return real(ctx, value, where)

    monkeypatch.setattr(config, "_voxel_size", recorder)
    load_config(tiny_yaml)
    assert calls == ["settings.voxel_size"]
    assert preferences.parse_mapping({"quality": {"voxel_size": 2}}, "x.yaml").values == {"quality": {"voxel_size": 2.0}}
    assert calls == ["settings.voxel_size", "quality.voxel_size"]


def test_only_the_builder_and_the_server_read_the_preferences_file():
    """A guard on the source: no loader, optimizer, run or CLI module imports `topopt.preferences`."""
    importers = set()
    for path in sorted((REPO / "topopt").rglob("*.py")):
        text = path.read_text()
        if re.search(r"^\s*(?:from\s+(?:\.+|topopt\.)preferences\b|from\s+(?:\.+|topopt)\s+import\s+[^#\n]*\bpreferences\b"
                     r"|import\s+topopt\.preferences\b)", text, re.M):
            importers.add(path.relative_to(REPO).as_posix())
    assert importers == {"topopt/regions_builder.py"}, importers
    server_text = (REPO / "topopt_mcp" / "server.py").read_text()
    assert set(re.findall(r"preferences\.(\w+)\(", server_text)) <= {"read", "save", "facts_for_yaml", "path_of"}
    assert "fill_spec" not in server_text              # the server does not fill a YAML; only the builder does, and it says so


def _calls_by_function(names: set[str]) -> set[tuple[str, str, str]]:
    """`(file, outer function, called name)` for each call of one of `names` in `topopt/` and `topopt_mcp/` (an `ast` scan: a multi-line
    call, an attribute call and a call inside a nested function are found; a comment or a docstring is not a call)."""
    import ast
    found = set()
    for folder in ("topopt", "topopt_mcp"):
        for path in sorted((REPO / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            rel = path.relative_to(REPO).as_posix()

            def visit(node, outer):
                for child in ast.iter_child_nodes(node):
                    inner = outer or (child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else None)
                    if isinstance(child, ast.Call):
                        func = child.func
                        called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                        if called in names:
                            found.add((rel, outer or "<module>", called))
                    visit(child, inner)

            visit(tree, None)
    return found


def test_the_source_has_one_caller_of_the_builder_and_one_of_the_fill_and_no_other_way_to_the_preferences():
    """R1 n3: an `ast` scan (the regex guard above reads import lines only, so a multi-line import or a transitive caller passed it)."""
    import ast
    assert _calls_by_function({"build_in_workspace"}) == {("topopt_mcp/server.py", "build_regions", "build_in_workspace")}
    assert _calls_by_function({"fill_spec"}) == {("topopt/regions_builder.py", "build_in_workspace", "fill_spec")}
    importers, indirect = set(), set()
    for folder in ("topopt", "topopt_mcp"):
        for path in sorted((REPO / folder).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                if isinstance(node, ast.ImportFrom):
                    names = {a.name for a in node.names}
                    if (node.module or "").split(".")[-1] == "preferences" or ("preferences" in names and (node.module in (None, "topopt"))):
                        importers.add(rel)
                elif isinstance(node, ast.Import) and any(a.name.split(".")[-1] == "preferences" for a in node.names):
                    importers.add(rel)
                elif isinstance(node, ast.Call):        # importlib.import_module("topopt.preferences"), __import__(...), getattr(m, "preferences")
                    func = node.func
                    called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                    if called in ("import_module", "__import__", "getattr") and any(
                            isinstance(a, ast.Constant) and isinstance(a.value, str) and "preferences" in a.value for a in node.args):
                        indirect.add(rel)
    # the builder and the server (the tools `environment` and `save_preferences`, `check_problem`) are the only modules that import it
    assert importers == {"topopt/regions_builder.py", "topopt_mcp/server.py"}, importers
    assert indirect == set(), indirect


def run_dir_facts(run_json: dict) -> dict:
    """The keys of run.json that describe the problem: all of them except the start time."""
    return {k: v for k, v in run_json.items() if k != "started_at"}


def test_a_preferences_file_changes_nothing_about_the_loaded_problem_the_check_or_the_run(tiny_yaml, tmp_path, monkeypatch):
    """The same YAML with and without a preferences file (with values that differ from the YAML) gives the same loaded problem, the same
    check result, the same reply of run_topopt and the same run.json. The only difference is the fact of check_problem."""
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)
    monkeypatch.delenv("TOPOPT_MCP_CMD", raising=False)
    plain, with_file = tmp_path / "plain", tmp_path / "with"
    plain.mkdir()
    with_file.mkdir()
    assert preferences.save(with_file, **PREFS_FAR)["ok"]
    results = {}
    for label, folder in (("plain", plain), ("file", with_file)):
        monkeypatch.setenv("TOPOPT_WORKSPACE", str(folder))
        monkeypatch.chdir(folder)
        cfg = load_config(tiny_yaml)
        from topopt.cli import check_data
        checked = server.check_problem(str(tiny_yaml))
        server._current.clear()
        started = server.run_topopt(str(tiny_yaml), str(folder / "out"))
        assert "pid" in started, started
        proc = server._current["proc"]
        run_json = None
        end = time.time() + 120
        while time.time() < end:
            status = server.run_status(str(folder / "out"))
            if status["state"] in ("done", "error"):
                break
            time.sleep(0.3)
        assert status["state"] == "done", status
        run_json = json.loads((folder / "out" / "run.json").read_text())
        try:
            os.killpg(proc.pid, signal.SIGKILL)               # the child that this test started stays alive for the dashboard
        except OSError:
            pass
        proc.wait()
        server._current.clear()
        results[label] = {"cfg": cfg, "data": json.loads(json.dumps(check_data(tiny_yaml), default=str)), "checked": checked,
                          "started": {k: started[k] for k in ("assumptions", "derived")}, "run_json": run_dir_facts(run_json)}
    a, b = results["plain"], results["file"]
    assert a["cfg"] == b["cfg"]                                                # the loaded problem: print, material, settings
    assert a["cfg"].settings.voxel_size == 2.0 and a["cfg"].material.E == 3500.0 and a["cfg"].print_settings.bed_mm == (220.0, 220.0)
    assert a["data"] == b["data"]                                              # the check result of the loader
    facts = b["checked"].pop("preferences_facts")
    assert "preferences_facts" not in a["checked"] and a["checked"] == b["checked"]
    assert len(facts) == 1 and "the YAML has no print block" in facts[0] and "bed 999 x 999 mm" in facts[0] and "not changed" in facts[0]
    assert a["started"] == b["started"]                                        # assumptions and derived of run_topopt
    assert a["run_json"] == b["run_json"]                                      # the problem as the run saw it
    assert b["run_json"]["print"]["bed_mm"] == [220.0, 220.0] and b["run_json"]["print"]["given"] == []
    assert b["run_json"]["derived"]["density_g_cm3"] == 1.24 and b["run_json"]["derived"]["voxel_size_mm"] == 2.0


def test_the_cli_run_and_check_do_not_read_the_file(tiny_yaml, tmp_path, monkeypatch):
    """`topopt check` and `topopt run` in a folder that holds a preferences file: the same text, the same run.json."""
    import subprocess
    plain, with_file = tmp_path / "plain", tmp_path / "with"
    plain.mkdir()
    with_file.mkdir()
    preferences.save(with_file, **PREFS_FAR)
    env = dict(os.environ, PYTHONPATH=str(REPO))
    env.pop("TOPOPT_WORKSPACE", None)
    env.pop("TOPOPT_DEFAULT_WORKSPACE", None)
    texts, runs = {}, {}
    for label, folder in (("plain", plain), ("file", with_file)):
        env["TOPOPT_WORKSPACE"] = str(folder)
        check = subprocess.run([sys.executable, "-m", "topopt", "check", str(tiny_yaml)], cwd=folder, env=env, capture_output=True,
                               text=True, timeout=300)
        assert check.returncode == 0, check.stderr
        texts[label] = check.stdout
        run = subprocess.run([sys.executable, "-m", "topopt", "run", str(tiny_yaml), "--out", str(folder / "out")], cwd=folder, env=env,
                             capture_output=True, text=True, timeout=300)
        assert run.returncode == 0, run.stderr
        runs[label] = run_dir_facts(json.loads((folder / "out" / "run.json").read_text()))
    assert texts["plain"] == texts["file"] and runs["plain"] == runs["file"]
    assert "999" not in texts["file"] and runs["file"]["print"]["bed_mm"] == [220.0, 220.0]


# --- build_regions: only a block that the spec does not have --------------------------------------------------------------------

@pytest.fixture
def prefs_work(work):
    assert preferences.save(work, **{**ALL, "quality": {"voxel_size": 2.0}})["ok"]
    return work


def built_yaml(work, reply):
    return yaml.safe_load((work / reply["yaml_rel"]).read_text())


def test_build_regions_fills_the_missing_blocks_and_says_so(prefs_work):
    spec = tiny_spec()
    spec_before = copy.deepcopy(spec)
    r = server.build_regions(spec)
    assert r["ok"], r
    assert spec == spec_before                                                   # the spec of the caller is not changed
    doc = built_yaml(prefs_work, r)
    assert doc["material"] == {"E": 2100.0, "nu": 0.35, "density": 1.27}
    assert doc["print"] == {"printer_type": "fdm", "bed_mm": [256.0, 256.0], "build_direction": "+z", "self_supporting_angle_deg": 40.0}
    assert doc["settings"]["voxel_size"] == 2.0 and r["voxel_size"] == 2.0 and r["voxel_auto"] is False
    facts = r["preferences_facts"]
    assert len(facts) == 3 and all(str(prefs_work / FILE) in f and "from your preferences file" in f for f in facts)
    assert "PETG, E 2100 MPa, nu 0.35, density 1.27 g/cm3" in facts[0] and "bed 256 x 256 mm" in facts[1] and "2 mm" in facts[2]
    cfg = load_config(prefs_work / r["yaml_rel"])
    assert cfg.print_settings.bed_mm == (256.0, 256.0) and "bed_mm" in cfg.print_settings.given and cfg.material.E == 2100.0


def test_the_spec_always_wins_and_a_partial_block_is_not_mixed_with_the_preferences(prefs_work):
    spec = tiny_spec(voxel_size=3.0)
    spec["material"] = {"E": 1000}                    # a part of a block: nu and density are NOT taken from the preferences
    spec["print"] = {"bed_mm": 150}                   # printer_type and the others are NOT taken either
    r = server.build_regions(spec)
    assert r["ok"], r
    doc = built_yaml(prefs_work, r)
    assert doc["material"] == {"E": 1000.0, "nu": 0.35, "density": 1.24}         # the defaults of the problem YAML
    assert doc["print"] == {"bed_mm": 150}
    assert doc["settings"]["voxel_size"] == 3.0
    assert "preferences_facts" not in r and "from your preferences" not in json.dumps(r)


def test_a_voxel_size_auto_in_the_spec_is_a_value_of_the_spec(prefs_work):
    spec = tiny_spec(voxel_size="auto")
    spec["material"] = {"E": 1000.0, "nu": 0.3, "density": 1.0}
    spec["print"] = {"printer_type": "sls"}
    r = server.build_regions(spec)
    assert r["ok"] and r["voxel_auto"] is True
    assert built_yaml(prefs_work, r)["settings"]["voxel_size"] == r["voxel_size"]      # the size that the builder chose, not the 2.0 of the file
    assert "from your preferences" not in json.dumps(r) and "preferences_facts" not in r


def test_one_missing_block_is_filled_and_the_others_stay_as_the_spec_has_them(prefs_work):
    spec = tiny_spec(voxel_size=3.0)
    spec["material"] = {"E": 1000.0}
    r = server.build_regions(spec)                    # the print block is missing: only it comes from the preferences
    doc = built_yaml(prefs_work, r)
    assert doc["material"] == {"E": 1000.0, "nu": 0.35, "density": 1.24}         # a partial block: the defaults, not the 1.27 of the file
    assert doc["print"]["bed_mm"] == [256.0, 256.0] and doc["settings"]["voxel_size"] == 3.0
    assert len(r["preferences_facts"]) == 1 and r["preferences_facts"][0].startswith("print settings from your preferences file")


def test_a_partial_block_is_not_completed_from_the_preferences_while_another_block_is_filled(prefs_work):
    spec = tiny_spec(voxel_size=3.0)
    spec["print"] = {"bed_mm": 150}                   # a part of a block: no printer type, no build direction from the file
    r = server.build_regions(spec)                    # the material is missing: it comes from the file
    doc = built_yaml(prefs_work, r)
    assert doc["print"] == {"bed_mm": 150} and doc["material"] == {"E": 2100.0, "nu": 0.35, "density": 1.27}
    assert len(r["preferences_facts"]) == 1 and r["preferences_facts"][0].startswith("material from your preferences file")
    spec = tiny_spec()                                # no voxel size in the spec: it comes from the file; the partial print stays
    spec["print"], spec["material"] = {"printer_type": "sla"}, {"density": 1.1}
    r = server.build_regions(spec)
    doc = built_yaml(prefs_work, r)
    assert doc["print"] == {"printer_type": "sla"} and doc["material"] == {"E": 3500.0, "nu": 0.35, "density": 1.1}
    assert doc["settings"]["voxel_size"] == 2.0 and len(r["preferences_facts"]) == 1 and r["preferences_facts"][0].startswith("voxel_size from")


def test_an_empty_block_gives_no_value_so_it_counts_as_missing(prefs_work):
    spec = tiny_spec()
    spec["material"], spec["print"] = {}, None
    r = server.build_regions(spec)
    assert len(r["preferences_facts"]) == 3 and built_yaml(prefs_work, r)["material"]["E"] == 2100.0


def test_a_preference_that_the_file_does_not_hold_stays_the_default_and_the_fact_says_so(work):
    preferences.save(work, material={"E": 2100, "name": "PETG"})
    r = server.build_regions(tiny_spec(voxel_size=2.0))
    doc = built_yaml(work, r)
    assert doc["material"] == {"E": 2100.0, "nu": 0.35, "density": 1.24} and "print" not in doc
    assert len(r["preferences_facts"]) == 1
    assert "no nu, density" in r["preferences_facts"][0] and "nu 0.35, density 1.24" in r["preferences_facts"][0]


def test_without_a_file_or_with_an_empty_file_the_yaml_is_the_old_yaml(work):
    spec = tiny_spec(voxel_size=2.0)
    plain = server.build_regions(spec)
    assert "preferences_facts" not in plain
    doc_plain = built_yaml(work, plain)
    assert "print" not in doc_plain and doc_plain["material"] == {"E": 3500.0, "nu": 0.35, "density": 1.24}
    preferences.save(work)                            # the user kept the defaults: an empty file
    again = server.build_regions(spec)
    assert "preferences_facts" not in again and built_yaml(work, again) == doc_plain


def test_a_file_with_a_problem_is_not_used_and_the_fact_says_so(work):
    put(work, "material: {E: -1}\nprint: {bed_mm: 300}\n")
    r = server.build_regions(tiny_spec(voxel_size=2.0))
    assert r["ok"]
    doc = built_yaml(work, r)
    assert "print" not in doc and doc["material"]["E"] == 3500.0
    assert len(r["preferences_facts"]) == 1 and "did not use it" in r["preferences_facts"][0] and "material.E" in r["preferences_facts"][0]
    put(work, "- list\n")
    assert "must hold a mapping" in server.build_regions(tiny_spec(voxel_size=2.0))["preferences_facts"][0]
    put(work, "print: [unclosed\n")
    assert "not valid YAML" in server.build_regions(tiny_spec(voxel_size=2.0))["preferences_facts"][0]


def test_the_cli_wrapper_does_not_read_the_file(prefs_work, tmp_path):
    from topopt.regions_builder import build_spec
    out = tmp_path / "cli"
    r = build_spec(tiny_spec(voxel_size=2.0), out, say=lambda *a: None)
    assert r["ok"]
    doc = yaml.safe_load((out / "tiny.yaml").read_text())
    assert "print" not in doc and doc["material"]["E"] == 3500.0


# --- check_problem: one fact for a difference --------------------------------------------------------------------------------------

def edited_yaml(src: Path, dst: Path, change) -> Path:
    doc = yaml.safe_load(src.read_text())
    change(doc)
    dst.write_text(yaml.safe_dump(doc, sort_keys=False))
    return dst


@pytest.fixture
def case_yaml(work, tiny_yaml):
    """A copy of the tiny problem in the work folder (the STL files beside it)."""
    import shutil
    dst = work / "runs" / "case"
    shutil.copytree(tiny_yaml.parent, dst)
    return dst / tiny_yaml.name


def test_check_problem_names_a_difference_of_the_bed_the_printer_and_the_material(prefs_work, case_yaml):
    yml = edited_yaml(case_yaml, case_yaml.with_name("other.yaml"), lambda d: d.update(
        print={"printer_type": "sla", "bed_mm": [200, 200]}, material={"E": 1800, "nu": 0.35, "density": 1.27}))
    r = server.check_problem(str(yml))
    assert r["ok"]
    facts = r["preferences_facts"]
    assert len(facts) == 1 and facts[0].startswith(f"The preferences file {prefs_work / FILE} differs from this YAML")
    for text in ("printer: sla in the YAML, fdm in the preferences", "bed: 200 x 200 mm in the YAML, 256 x 256 mm in the preferences",
                 "material E: 1800 MPa in the YAML, 2100 MPa (PETG) in the preferences", "is not changed"):
        assert text in facts[0], text
    assert "material density" not in facts[0] and "material nu" not in facts[0]  # these agree
    assert not any("preferences" in w for w in r["warnings"])                    # a fact, not a warning, not a stop
    # R1 m5: each material key alone is compared (a mutant that drops the nu comparison survived 119 tests)
    base = {"print": {"printer_type": "fdm", "bed_mm": 256, "build_direction": [0, 0, 1], "self_supporting_angle_deg": 40},
            "material": {"E": 2100, "nu": 0.35, "density": 1.27}}
    for key, other, text in (("nu", 0.3, "material nu: 0.3 in the YAML, 0.35 (PETG) in the preferences"),
                             ("density", 1.1, "material density: 1.1 g/cm3 in the YAML, 1.27 g/cm3 (PETG) in the preferences"),
                             ("E", 1900, "material E: 1900 MPa in the YAML, 2100 MPa (PETG) in the preferences")):
        one = edited_yaml(case_yaml, case_yaml.with_name(f"only-{key}.yaml"), lambda d: d.update(
            print=base["print"], material={**base["material"], key: other}))
        facts = server.check_problem(str(one))["preferences_facts"]
        assert len(facts) == 1 and text in facts[0], (key, facts)
        assert sum("material " in part for part in facts[0].split("; ")) == 1, facts[0]       # only that key is named


def test_check_problem_says_that_a_material_value_is_the_default_when_the_yaml_does_not_give_it(prefs_work, case_yaml):
    """R1 n2: a YAML with no material block (or with a block that lacks the key) gets the default of the loader, and the fact says so."""
    none = edited_yaml(case_yaml, case_yaml.with_name("nomat.yaml"), lambda d: d.pop("material"))
    fact = server.check_problem(str(none))["preferences_facts"][0]
    assert "material E: the default 3500 MPa (the YAML has no material block), 2100 MPa (PETG) in the preferences" in fact
    assert "material density: the default 1.24 g/cm3 (the YAML has no material block), 1.27 g/cm3 (PETG) in the preferences" in fact
    assert "MPa in the YAML" not in fact and "material nu" not in fact              # nu: the default 0.35 equals the preference
    part = edited_yaml(case_yaml, case_yaml.with_name("partmat.yaml"), lambda d: d.update(material={"density": 1.27}))
    fact = server.check_problem(str(part))["preferences_facts"][0]
    assert "material E: the default 3500 MPa (the YAML does not give it), 2100 MPa (PETG) in the preferences" in fact
    assert "material density" not in fact                                            # given and equal
    given = edited_yaml(case_yaml, case_yaml.with_name("givenmat.yaml"), lambda d: d.update(material={"E": 1800, "nu": 0.35, "density": 1.27}))
    assert "material E: 1800 MPa in the YAML, 2100 MPa (PETG) in the preferences" in server.check_problem(str(given))["preferences_facts"][0]


def test_check_problem_says_when_the_yaml_has_no_print_block_and_the_preferences_have_one(prefs_work, case_yaml):
    r = server.check_problem(str(case_yaml))
    facts = r["preferences_facts"]
    assert len(facts) == 1 and "the YAML has no print block, so the print report assumes" in facts[0]
    assert "printer fdm, bed 220 x 220 mm" in facts[0] and "bed 256 x 256 mm" in facts[0]
    assert "material E: 3500 MPa in the YAML, 2100 MPa (PETG) in the preferences" in facts[0]    # build_regions wrote the block with E


def test_check_problem_has_no_fact_when_the_yaml_and_the_preferences_agree(prefs_work, case_yaml):
    agree = edited_yaml(case_yaml, case_yaml.with_name("agree.yaml"), lambda d: d.update(
        print={"printer_type": "fdm", "bed_mm": 256, "build_direction": [0, 0, 1], "self_supporting_angle_deg": 40},
        material={"E": 2100, "nu": 0.35, "density": 1.27}))
    r = server.check_problem(str(agree))
    assert r["ok"] and "preferences_facts" not in r
    # a file that holds fewer keys compares only those keys
    preferences.save(prefs_work, material={}, print={"build_direction": None, "self_supporting_angle_deg": None})
    only = edited_yaml(case_yaml, case_yaml.with_name("only.yaml"), lambda d: d.update(print={"printer_type": "fdm", "bed_mm": [256, 256]}))
    assert "preferences_facts" not in server.check_problem(str(only))


def test_check_problem_names_a_print_value_that_the_yaml_does_not_give(prefs_work, case_yaml):
    part = edited_yaml(case_yaml, case_yaml.with_name("part.yaml"), lambda d: d.update(print={"printer_type": "fdm"}, material={
        "E": 2100, "nu": 0.35, "density": 1.27}))
    fact = server.check_problem(str(part))["preferences_facts"][0]
    assert "bed: 220 x 220 mm in the YAML (assumed: the YAML does not give it), 256 x 256 mm in the preferences" in fact
    assert "self-supporting angle: 45 deg in the YAML (assumed: the YAML does not give it), 40 deg in the preferences" in fact
    assert "printer:" not in fact and "build direction" not in fact                  # these agree (+z is the assumed direction)


def test_check_problem_has_no_fact_without_a_file_or_with_a_broken_file(work, case_yaml):
    assert "preferences_facts" not in server.check_problem(str(case_yaml))
    put(work, "print: [unclosed\n")
    r = server.check_problem(str(case_yaml))
    assert r["ok"] and "preferences_facts" not in r


def test_check_problem_result_is_the_same_apart_from_the_fact(prefs_work, case_yaml):
    with_file = server.check_problem(str(case_yaml))
    with_file.pop("preferences_facts")
    (prefs_work / FILE).unlink()
    plain = server.check_problem(str(case_yaml))
    assert with_file == plain


# --- the texts ---------------------------------------------------------------------------------------------------------------------

def tool_text():
    return {t.name: t.description for t in asyncio.run(server.mcp.list_tools())}["save_preferences"]


def test_the_tool_text_is_short_has_the_facts_and_has_no_must_never_always():
    text = tool_text()
    assert len(text) <= 2048                          # the key text rule of test_plugin_manifest.py asks for an entry only above 2,048
    assert len(text) <= 4096
    assert not re.search(r"\b(must|never|always)\b", text, re.I)
    for fact in ("topopt-preferences.yaml", "work folder", "reads the file at each call", "does not change a run by itself",
                 "build_regions", "preferences_facts", "settings.voxel_size", "stays as it is", "not valid YAML", "There is no CLI command",
                 "renders (render_part)", "step (export_step)", "verify (verify_run)", "unknown key"):
        assert fact in text or fact == "unknown key" and "An unknown key" in text, fact
    assert tool_text() == inspect.cleandoc(server.save_preferences.__wrapped__.__doc__)      # #388: cleaned the same way on every Python


def test_the_tool_text_needs_no_key_text_entry():
    from test_plugin_manifest import KEY_TEXT, OLD_CLIENT_CUT
    assert len(tool_text()) <= OLD_CLIENT_CUT or "save_preferences" in KEY_TEXT


def skill_section(path: Path, heading: str) -> str:
    text = path.read_text()
    start = text.index(heading)
    nxt = re.search(r"^## ", text[start + len(heading):], re.M)
    return text[start: start + len(heading) + (nxt.start() if nxt else len(text))]


def test_the_two_copies_of_each_skill_are_equal():
    for source, copy_ in (("skills/start-here", ".agents/skills/start-here"),
                          (".claude/skills/part-to-topopt", ".agents/skills/part-to-topopt")):
        assert (REPO / source / "SKILL.md").read_bytes() == (REPO / copy_ / "SKILL.md").read_bytes(), source


def test_start_here_asks_once_and_offers_the_answers():
    section = skill_section(REPO / "skills" / "start-here" / "SKILL.md", "## Step 1b. Preferences")
    for word in ("environment", "`preferences`", "`exists`", "false", "once", "one message", "print bed", "printer type", "material", "quality",
                 "voxel size", "slower", "outputs", "keep the defaults", "save_preferences", "do not ask again", "change them at any time"):
        assert word in section, word
    assert not re.search(r"\b(must|never|always)\b", section, re.I)


def test_part_to_topopt_takes_the_values_from_the_preferences_and_offers_the_outputs():
    section = skill_section(REPO / ".claude" / "skills" / "part-to-topopt" / "SKILL.md", "## Preferences")
    for word in ("`preferences`", "`print`", "`material`", "voxel size", "`build_regions`", "tell the user", "from there",
                 "`outputs`", "offer", "`render_part`", "`export_step`", "`verify_run`", "spec", "partial"):
        assert word in section, word
    assert not re.search(r"\b(must|never|always)\b", section, re.I)
