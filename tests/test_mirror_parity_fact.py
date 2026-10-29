"""Tests for F4-33: a mirror plane with a given `align` (face or centre) says when the other parity fits better.

CONTRACTS.md section 26.1, "Parity choice" and "Facts". The rule under test:

- `align: face|centre`: the given parity is the one that the run uses (the grid does not change). `resolve` also builds the
  other parity and the facts of the plane get `other_parity` (`parity`, `label_mismatches`, `orphans`). When the other parity
  has fewer label mismatches, `summary_lines` has one more line with both counts and the options.
- `align: auto`: facts and lines are the ones of the commit before F4-33 (pinned below as literals, taken from that commit).

The counts are exact (they are counts of voxels). The numbers of the synthetic part were measured at the start commit cd83de9
(the part of `test_mirror.write_yaml`, voxel 2 mm).
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from test_mirror import write_yaml
from topopt import symmetry
from topopt.cli import check_data, main
from topopt.config import load_config
from topopt.problem import build_problem

ROOT = Path(__file__).resolve().parents[1]
HOOK_SHORT = ROOT / "examples" / "hook" / "hook_short.yaml"
FEWER = "has fewer label mismatches"   # the words of the added line


def plane(align, at_mm=0.0, normal="y"):
    return {"normal": normal, "at_mm": at_mm, "align": align}


def planes_of(tmp_path, mirror, name="m.yaml"):
    prob = build_problem(load_config(write_yaml(tmp_path, mirror, name=name)))
    return prob, prob.mirror["planes"]


def hook_short_yaml(tmp_path, align):
    """`examples/hook/hook_short.yaml` (voxel 2.0 mm) with one plane y = 0 and the given align; the paths are absolute."""
    d = yaml.safe_load(HOOK_SHORT.read_text())
    r = d["regions"]
    r["design"] = str(HOOK_SHORT.parent / r["design"])
    r["fixed"] = [str(HOOK_SHORT.parent / f) for f in r["fixed"]]
    r["keep_out"] = [str(HOOK_SHORT.parent / f) for f in r["keep_out"]]
    r["load"] = {k: str(HOOK_SHORT.parent / v) for k, v in r["load"].items()}
    d["mirror"] = [{"normal": [0, 1, 0], "at_mm": 0.0, "align": align}]
    p = tmp_path / f"hook_short_{align}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


# --- the card case: hook_short at 2.0 mm, plane y = 0 --------------------------------------------------------------


def test_card_case_centre_says_that_face_has_50_mismatches_and_12_orphans(tmp_path):
    prob = build_problem(load_config(hook_short_yaml(tmp_path, "centre")))
    pl = prob.mirror["planes"][0]
    assert pl["parity"] == "centre" and pl["label_mismatches"] == {"face": None, "centre": 1140}   # the forced parity is used
    assert pl["orphans"]["count"] == 570
    assert pl["other_parity"] == {"parity": "face", "label_mismatches": 50, "orphans": 12}
    [line] = [ln for ln in symmetry.summary_lines(prob.mirror) if FEWER in ln]
    assert line == (
        "plane y = 0 mm: parity face has fewer label mismatches than parity centre, which align centre sets "
        "(face: 50 label mismatches and 12 orphans; centre: 1140 label mismatches and 570 orphans). "
        "The run keeps centre. Options: set align: auto, or set align: face")


def test_card_case_auto_and_face_have_no_sentence_and_no_fact_for_auto(tmp_path):
    auto = build_problem(load_config(hook_short_yaml(tmp_path, "auto"))).mirror
    pl = auto["planes"][0]
    assert pl["parity"] == "face" and pl["label_mismatches"] == {"face": 50, "centre": 1140} and pl["orphans"]["count"] == 12
    assert "other_parity" not in pl and not [ln for ln in symmetry.summary_lines(auto) if FEWER in ln]
    face = build_problem(load_config(hook_short_yaml(tmp_path, "face"))).mirror
    pl = face["planes"][0]
    assert pl["other_parity"] == {"parity": "centre", "label_mismatches": 1140, "orphans": 570}   # a fact, and no sentence
    assert not [ln for ln in symmetry.summary_lines(face) if FEWER in ln]


# --- the synthetic part: plane y = 0 is symmetric on the face parity (0 mismatches), the centre parity has 76 -----------


def test_a_given_centre_gets_the_counts_of_face_and_one_line_with_both_counts(tmp_path):
    prob, [pl] = planes_of(tmp_path, [plane("centre")])
    assert pl["parity"] == "centre" and pl["align"] == "centre" and pl["k"] == 5
    assert pl["label_mismatches"] == {"face": None, "centre": 76} and pl["orphans"]["count"] == 34
    assert pl["other_parity"] == {"parity": "face", "label_mismatches": 0, "orphans": 0}
    lines = symmetry.summary_lines(prob.mirror)
    [line] = [ln for ln in lines if FEWER in ln]
    assert line == (
        "plane y = 0 mm: parity face has fewer label mismatches than parity centre, which align centre sets "
        "(face: 0 label mismatches and 0 orphans; centre: 76 label mismatches and 34 orphans). "
        "The run keeps centre. Options: set align: auto, or set align: face")
    assert len(lines) == 2 + 1 + 3   # the two lines of the plane, the new line, three load cases
    assert lines.index(line) == 2    # right after the orphan line of its plane


def test_the_other_parity_agrees_with_the_facts_of_the_run_that_uses_it(tmp_path):
    """The numbers in `other_parity` are the ones that `align: face` itself measures on its own grid."""
    _, [given] = planes_of(tmp_path, [plane("centre")], name="c.yaml")
    sub = tmp_path / "f"
    sub.mkdir()
    _, [used] = planes_of(sub, [plane("face")])
    assert given["other_parity"]["label_mismatches"] == used["label_mismatches"]["face"]
    assert given["other_parity"]["orphans"] == used["orphans"]["count"]
    back = used["other_parity"]   # and the same the other way round
    assert back["parity"] == "centre" and back["label_mismatches"] == given["label_mismatches"]["centre"]
    assert back["orphans"] == given["orphans"]["count"]


def test_a_given_face_that_fits_better_gets_the_fact_and_no_sentence(tmp_path):
    prob, [pl] = planes_of(tmp_path, [plane("face")])
    assert pl["parity"] == "face" and pl["label_mismatches"] == {"face": 0, "centre": None}
    assert pl["other_parity"]["parity"] == "centre" and pl["other_parity"]["label_mismatches"] == 76
    assert FEWER not in "\n".join(symmetry.summary_lines(prob.mirror))
    without = {**prob.mirror, "planes": [{k: v for k, v in pl.items() if k != "other_parity"}]}
    assert symmetry.summary_lines(prob.mirror) == symmetry.summary_lines(without)   # the fact adds no line here


def test_a_given_face_whose_other_parity_fits_better_gets_the_line_with_the_counts(tmp_path):
    """At y = 1 the centre parity has 40 label mismatches and the face parity 112 (measured at cd83de9)."""
    prob, [pl] = planes_of(tmp_path, [plane("face", 1.0)])
    assert pl["label_mismatches"] == {"face": 112, "centre": None} and pl["orphans"]["count"] == 64
    assert pl["other_parity"] == {"parity": "centre", "label_mismatches": 40, "orphans": 34}
    [line] = [ln for ln in symmetry.summary_lines(prob.mirror) if FEWER in ln]
    assert "parity centre has fewer label mismatches than parity face, which align face sets" in line
    assert "(centre: 40 label mismatches and 34 orphans; face: 112 label mismatches and 64 orphans)" in line
    assert line.endswith("The run keeps face. Options: set align: auto, or set align: centre")
    prob, [pl] = planes_of(tmp_path, [plane("centre", 1.0)], name="c.yaml")   # the parity that fits better is given: no line
    assert pl["label_mismatches"]["centre"] == 40 and pl["other_parity"]["label_mismatches"] == 112
    assert FEWER not in "\n".join(symmetry.summary_lines(prob.mirror))


def test_the_line_needs_fewer_mismatches_and_not_a_threshold(tmp_path):
    """Any lower count of the other parity fires the line (1 against 2), and an equal count does not."""
    base = {"axis": "y", "at_mm": 0.0, "align": "centre", "parity": "centre", "k": 5, "plane_index": 2,
            "label_mismatches": {"face": None, "centre": 2}, "grid_range_mm": [-5.0, 5.0],
            "orphans": {"count": 0, "share": 0.0, "outside_grid": 0, "empty": 0, "keep_out": 0, "fixed": 0, "load": 0,
                        "other": 0}}

    def lines(other):
        mirror = {"planes": [{**base, **({"other_parity": other} if other else {})}], "design_voxels": 1000,
                  "load_asymmetry": []}
        return [ln for ln in symmetry.summary_lines(mirror) if FEWER in ln]

    assert len(lines({"parity": "face", "label_mismatches": 1, "orphans": 0})) == 1
    assert lines({"parity": "face", "label_mismatches": 2, "orphans": 0}) == []   # a tie: the given parity is not worse
    assert lines({"parity": "face", "label_mismatches": 3, "orphans": 0}) == []
    assert lines(None) == []   # no fact (facts of an older run, or the other grid could not be built)
    [line] = lines({"parity": "face", "label_mismatches": 1})   # no orphan count: only the mismatches
    assert "(face: 1 label mismatches; centre: 2 label mismatches and 0 orphans)" in line


# --- auto: the facts and the lines are the ones of the start commit --------------------------------------------------


AUTO_PLANE = {
    "align": "auto", "at_mm": 0.0, "axis": "y", "grid_range_mm": [-4.0, 4.0], "k": 4,
    "label_mismatches": {"centre": 76, "face": 0},
    "orphans": {"count": 0, "empty": 0, "fixed": 0, "keep_out": 0, "load": 0, "other": 0, "outside_grid": 0, "share": 0.0},
    "parity": "face", "plane_index": 2}


def test_auto_facts_and_lines_are_the_ones_of_the_start_commit(tmp_path):
    """Pinned from cd83de9 (`resolve` and `summary_lines` before F4-33). `auto` has no `other_parity` and no new line."""
    prob, [pl] = planes_of(tmp_path, [plane("auto")])
    assert pl == AUTO_PLANE
    assert json.dumps(pl, sort_keys=True) == json.dumps(AUTO_PLANE, sort_keys=True)
    lines = symmetry.summary_lines(prob.mirror)
    assert lines[:2] == [
        "plane y = 0 mm: parity face, the plane goes on a voxel face (align auto). Label mismatches under the mirror: "
        "face 0 (used), centre 76",
        "plane y = 0 mm: 0 of 124 design voxels have no design partner (0.00 percent). They stay free variables"]
    assert len(lines) == 5 and not [ln for ln in lines if FEWER in ln]
    assert (prob.shape, prob.origin.tolist()) == ((9, 4, 4), [-2.0, -4.0, -4.0])


def test_the_default_align_is_auto_and_has_no_new_fact(tmp_path):
    _, [pl] = planes_of(tmp_path, [{"normal": "y", "at_mm": 0.0}])
    assert pl == AUTO_PLANE   # no `align` key in the YAML is `auto`


# --- the given parity is the one that the run uses ----------------------------------------------------------------


@pytest.mark.parametrize("align, shape, origin_y, k", [("centre", (9, 5, 4), -5.0, 5), ("face", (9, 4, 4), -4.0, 4)])
def test_the_given_parity_and_its_grid_are_the_ones_of_the_start_commit(tmp_path, align, shape, origin_y, k):
    prob, [pl] = planes_of(tmp_path, [plane(align)])
    assert prob.shape == shape and prob.origin.tolist() == [-2.0, origin_y, -4.0]   # pinned from cd83de9
    assert pl["parity"] == align and pl["k"] == k
    cfg, info = symmetry.resolve(load_config(write_yaml(tmp_path, [plane(align)], name="again.yaml")))
    assert [p.align for p in cfg.mirror] == [align]   # the config that builds the grid keeps the given align
    assert info[0]["parity"] == align and info[0]["label_mismatches"][align] == pl["label_mismatches"][align]
    assert info[0]["label_mismatches"]["face" if align == "centre" else "centre"] is None   # the old key stays as it was


def test_a_given_plane_does_not_change_the_decision_of_a_later_auto_plane(tmp_path):
    """Plane 1 is y with `centre` (the worse parity), plane 2 is z with `auto`: plane 2 gets what it got at cd83de9."""
    prob, [py, pz] = planes_of(tmp_path, [plane("centre"), plane("auto", normal="z")])
    assert prob.shape == (9, 5, 4)
    assert py["parity"] == "centre" and py["other_parity"] == {"parity": "face", "label_mismatches": 0, "orphans": 0}
    assert pz["parity"] == "face" and pz["label_mismatches"] == {"face": 0, "centre": 76} and "other_parity" not in pz
    assert pz["orphans"]["count"] == 0 and pz["k"] == 4 and pz["grid_range_mm"] == [-4.0, 4.0]
    text = symmetry.summary_lines(prob.mirror)
    assert len([ln for ln in text if FEWER in ln]) == 1 and "plane z = 0 mm: parity face has fewer" not in "\n".join(text)


def test_a_grid_that_cannot_be_built_for_the_other_parity_gives_no_fact_and_no_line(tmp_path, monkeypatch):
    """The other grid can fail (the grid guard raises `UserError`). The given parity still runs; a fact never blocks a run."""
    from topopt import voxelize as voxelize_module
    from topopt.config import UserError
    real = voxelize_module.voxelize

    def refuse_face(config, *args, **kwargs):   # `resolve` calls this name; `build_problem` keeps its own binding
        if any(p.align == "face" for p in config.mirror):
            raise UserError("grid is too big")
        return real(config, *args, **kwargs)

    monkeypatch.setattr(voxelize_module, "voxelize", refuse_face)
    prob, [pl] = planes_of(tmp_path, [plane("centre")])
    assert pl["parity"] == "centre" and "other_parity" not in pl
    assert FEWER not in "\n".join(symmetry.summary_lines(prob.mirror))


# --- check, the JSON, the MCP tool and the run log ------------------------------------------------------------------


def test_check_prints_the_line_and_the_json_and_check_problem_have_the_fact(tmp_path, capsys):
    p = write_yaml(tmp_path, [plane("centre")])
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^mirror:\s+plane y = 0 mm: parity face has fewer label mismatches than parity centre", out, re.M)
    want = {"parity": "face", "label_mismatches": 0, "orphans": 0}
    assert check_data(p)["mirror"]["planes"][0]["other_parity"] == want
    assert json.loads(json.dumps(check_data(p)["mirror"]))["planes"][0]["other_parity"] == want   # plain JSON values
    server = pytest.importorskip("topopt_mcp.server")
    res = server.check_problem(str(p))
    assert res["ok"] is True and res["mirror"]["planes"][0]["other_parity"] == want


def test_the_run_log_and_run_json_carry_the_line_and_the_fact(tmp_path):
    p = write_yaml(tmp_path, [plane("centre")])
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    log = (out / "run.log").read_text()
    assert re.search(r"^.*mirror: plane y = 0 mm: parity face has fewer label mismatches than parity centre", log, re.M)
    run = json.loads((out / "run.json").read_text())
    assert run["mirror"]["planes"][0]["other_parity"] == {"parity": "face", "label_mismatches": 0, "orphans": 0}
    assert run["mirror"]["planes"][0]["parity"] == "centre"   # the run used the given parity
    assert "end" in run["mirror"]


def test_the_lever_card_of_align_names_the_new_fact():
    from topopt.levers_cards import CARDS
    card = next(c for c in CARDS if c.key == "mirror[].align")
    assert "other_parity" in card.values and "keeps its parity" in card.values
    [symptom] = card.symptoms
    assert "other_parity" in symptom.detect   # named in the text only: the short run of test_levers has no given align to prove a facts path
    assert "auto` costs one more voxelization" not in card.values   # a given align costs it too: the old words were wrong
