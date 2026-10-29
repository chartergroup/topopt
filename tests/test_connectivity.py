"""A piece of active voxels that no support holds (topopt/connectivity.py, CONTRACTS.md section 3).

The base case is the cantilever of `test_optimize.make_case`: design x 0..20, y 0..10, z 0..10; clamp plate x -2..0;
a tip box. At voxel size 1 the grid origin is (-2, 0, 0). An island is a load region that no load case uses (a
forced-solid body with no force and no clamp). The test builds it at a corner, at an edge and at a face of the design
box, and checks the facts, the `UserError` of `topopt run` and the warning of `topopt check`.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_optimize import box, make_case
from topopt import connectivity
from topopt.cli import check_data, main
from topopt.config import load_config
from topopt.problem import build_problem

REPO = Path(__file__).resolve().parents[1]


def case_with(tmp_path, islands, mutate=None):
    """The cantilever plus load regions `islands` (name -> (lo, hi)) that no load case uses. Return the YAML path."""
    p = make_case(tmp_path, h=1.0, iterations=2)
    d = yaml.safe_load(p.read_text())
    for name, (lo, hi) in islands.items():
        box(tmp_path / f"{name}.stl", lo, hi)
        d["regions"]["load"][name] = f"{name}.stl"
    if mutate:
        mutate(d)
    p.write_text(yaml.safe_dump(d))
    return p


def facts_of(tmp_path, islands, mutate=None):
    return connectivity.unsupported_pieces(build_problem(load_config(case_with(tmp_path, islands, mutate))))


# A cube of 2 mm (8 voxels) that touches the design box (x 0..20, y 0..10, z 0..10) at one corner node, at an edge, at a face.
CORNER = {"island": ((20, 10, 10), (22, 12, 12))}
EDGE = {"island": ((20, 10, 4), (22, 12, 6))}
FACE = {"island": ((20, 4, 4), (22, 6, 6))}


def test_the_plain_case_has_one_piece_and_no_fact(tmp_path):
    prob = build_problem(load_config(make_case(tmp_path, h=1.0, iterations=2)))
    f = connectivity.unsupported_pieces(prob)
    assert f == {"pieces": 1, "unsupported_pieces": 0, "unsupported_voxels": 0, "springs": False, "listed": []}
    assert connectivity.message(f) is None and connectivity.floating_error(prob) is None


def test_a_cube_that_touches_at_a_corner_is_a_piece_with_no_support(tmp_path):
    f = facts_of(tmp_path, CORNER)
    assert f["pieces"] == 2 and f["unsupported_pieces"] == 1 and f["unsupported_voxels"] == 8
    (piece,) = f["listed"]
    assert piece["voxels"] == 8 and piece["labels"] == {"design": 0, "fixed": 0, "load": 8, "solid": 0}
    assert piece["box_mm"] == [[20.0, 10.0, 10.0], [22.0, 12.0, 12.0]]
    first = piece["first"][0]                                    # the first voxel in the order (i, j, k)
    assert first == {"ijk": [22, 10, 10], "mm": [20.5, 10.5, 10.5], "label": "load"} and len(piece["first"]) == 3


def test_a_cube_that_touches_at_an_edge_is_a_piece_with_no_support(tmp_path):
    f = facts_of(tmp_path, EDGE)
    assert f["pieces"] == 2 and f["unsupported_pieces"] == 1 and f["unsupported_voxels"] == 8
    assert f["listed"][0]["box_mm"] == [[20.0, 10.0, 4.0], [22.0, 12.0, 6.0]]
    assert f["listed"][0]["first"][0]["mm"] == [20.5, 10.5, 4.5]


def test_a_cube_that_shares_a_face_is_one_piece(tmp_path):
    f = facts_of(tmp_path, FACE)
    assert f["pieces"] == 1 and f["unsupported_pieces"] == 0


@pytest.mark.parametrize("islands", [CORNER, EDGE], ids=["corner", "edge"])
def test_run_stops_before_the_solve_with_the_count_the_positions_and_the_options(tmp_path, capsys, islands):
    p = case_with(tmp_path, islands)
    out = tmp_path / "out"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 1
    err = capsys.readouterr().err
    assert "the active voxels make 2 separate pieces" in err and "1 of them has no fixed voxel, 8 voxels in all" in err
    assert "piece 1: 8 voxels (8 load)" in err and "first voxels (i, j, k): (22, 10, " in err and "at (20.5, 10.5, " in err
    assert "stiffness matrix is singular" in err and "zero pivot" in err and "Options: move or enlarge a region" in err and "add a fixed region" in err
    assert "file:" in err                                     # `cli` adds the YAML path to every UserError
    assert not (out / "result" / "final.stl").exists() and not any((out / "snapshots").glob("*"))      # no pass ran
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "error" and "has no fixed voxel" in status["message"]


@pytest.mark.parametrize("islands", [CORNER, EDGE], ids=["corner", "edge"])
def test_check_warns_and_says_that_run_stops(tmp_path, capsys, islands):
    p = case_with(tmp_path, islands)
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert "warning: the active voxels make 2 separate pieces" in out and "(`topopt run` stops with this error)" in out
    warnings = check_data(p)["warnings"]
    assert sum("has no fixed voxel" in w for w in warnings) == 1                    # one text, not two
    assert not any("each holds a fixed voxel" in w for w in warnings)


def test_a_second_piece_with_its_own_fixed_voxel_is_held(tmp_path, capsys):
    """Two pieces are allowed when each holds a support: `check` states the fact, `run` does not stop for it."""
    box(tmp_path / "plate2.stl", (20, 10, 10), (22, 12, 12))
    p = case_with(tmp_path, {}, mutate=lambda d: d["regions"]["fixed"].append("plate2.stl"))
    f = connectivity.unsupported_pieces(build_problem(load_config(p)))
    assert f["pieces"] == 2 and f["unsupported_pieces"] == 0
    warnings = check_data(p)["warnings"]
    assert any("2 separate pieces" in w and "each holds a fixed voxel" in w for w in warnings)
    assert not any("stops with this error" in w for w in warnings)


def test_a_spring_region_holds_a_piece(tmp_path):
    """With springs, a voxel of a spring region is a support (`springs.support_mask`): the pad holds the design, the tip is a free body next to it."""
    def spring(d):
        d["regions"]["fixed"] = []
        d["springs"] = [{"region": "pad", "stiffness": [100.0, 100.0, 100.0]}]

    f = facts_of(tmp_path, {"pad": ((20, 0, 0), (22, 10, 10))}, mutate=spring)            # the pad shares a face with the design
    assert f["pieces"] == 1 and f["unsupported_pieces"] == 0 and f["springs"] is True
    sub = tmp_path / "far"
    sub.mkdir()
    f = facts_of(sub, {"pad": ((30, 0, 0), (32, 10, 10))}, mutate=spring)                 # the pad is 10 mm away: the design is loose
    assert f["pieces"] == 2 and f["unsupported_pieces"] == 1 and f["listed"][0]["labels"]["design"] > 0
    text = connectivity.message(f)
    assert "no fixed voxel and no spring region" in text and "add a fixed region or a spring on the piece" in text


def test_the_text_without_springs_does_not_name_springs(tmp_path):
    text = connectivity.message(facts_of(tmp_path, CORNER))
    assert "spring" not in text


class WithJoints:
    """`Problem` on main has no `joints`: A7b adds it (a joint set, or None; `springs.support_mask` of A7b reads its `region_ids`). This wrapper gives
    the attribute that the problem of A7b has, and a joint set with no region, so the test holds on main and with A7b."""

    def __init__(self, problem, joints=SimpleNamespace(region_ids=[])):
        self._problem, self.joints = problem, joints

    def __getattr__(self, name):
        return getattr(self._problem, name)


def test_the_facts_and_the_text_name_joints_only_when_the_problem_has_joints(tmp_path):
    """Integration with A7b (a joint region holds a piece, as a spring region does): with `Problem.joints` set the facts have `joints` and the
    text and the options name joint regions. A problem with none (main: no attribute; A7b: None) gets the same facts and the same text as before."""
    prob = build_problem(load_config(case_with(tmp_path, CORNER)))
    plain = connectivity.unsupported_pieces(prob)
    assert "joints" not in plain and connectivity.unsupported_pieces(WithJoints(prob, joints=None)) == plain
    held = connectivity.unsupported_pieces(WithJoints(prob))
    assert held == {**plain, "joints": True}
    text, joint_text = connectivity.message(plain), connectivity.message(held)
    assert "joint" not in text and "1 of them has no fixed voxel, 8 voxels in all" in text and "add a fixed region on the piece" in text
    assert "1 of them has no fixed voxel and no joint region, 8 voxels in all" in joint_text and "add a fixed region or a joint on the piece" in joint_text
    assert joint_text.replace(" and no joint region", "").replace("a fixed region or a joint", "a fixed region") == text      # nothing else differs


def test_the_text_names_springs_and_joints_together(tmp_path):
    def spring(d):
        d["regions"]["fixed"] = []
        d["springs"] = [{"region": "pad", "stiffness": [100.0, 100.0, 100.0]}]

    prob = build_problem(load_config(case_with(tmp_path, {"pad": ((30, 0, 0), (32, 10, 10))}, mutate=spring)))      # the pad is 10 mm away: the design is loose
    facts = connectivity.unsupported_pieces(WithJoints(prob))
    assert facts["springs"] is True and facts["joints"] is True and facts["unsupported_pieces"] == 1
    text = connectivity.message(facts)
    assert "no fixed voxel, no spring region and no joint region" in text and "add a fixed region, a spring or a joint on the piece" in text
    assert "no fixed voxel and no spring region" in connectivity.message(connectivity.unsupported_pieces(prob))        # springs only: as before


def test_the_mirror_key_does_not_change_the_pieces(tmp_path, capsys):
    """A mirror plane moves the grid origin (less than one voxel) and ties the design variables of partner voxels. It adds no support."""
    plain = facts_of(tmp_path, CORNER)
    sub = tmp_path / "mirror"
    sub.mkdir()
    mirrored = facts_of(sub, CORNER, mutate=lambda d: d.update(mirror=[{"normal": "y", "at_mm": 5.0}]))
    assert mirrored["pieces"] == plain["pieces"] == 2
    assert mirrored["unsupported_voxels"] == plain["unsupported_voxels"] == 8
    assert mirrored["listed"][0]["box_mm"] == plain["listed"][0]["box_mm"]
    sub2 = tmp_path / "mirror_run"
    sub2.mkdir()
    p = case_with(sub2, CORNER, mutate=lambda d: d.update(mirror=[{"normal": "y", "at_mm": 5.0}]))
    assert main(["run", str(p), "--out", str(sub2 / "out"), "--quiet"]) == 1
    assert "has no fixed voxel" in capsys.readouterr().err


def test_only_the_first_pieces_are_named_and_the_counts_cover_all(tmp_path):
    """Seven separate cubes: the facts list the first `SHOW_PIECES`, the largest first; the counts are for all seven."""
    islands = {f"c{n}": ((24 + 3 * n, 0, 0), (24 + 3 * n + (1 + n % 2), 1 + n % 3, 1)) for n in range(7)}
    f = facts_of(tmp_path, islands)
    assert f["pieces"] == 8 and f["unsupported_pieces"] == 7
    assert len(f["listed"]) == connectivity.SHOW_PIECES
    sizes = [p["voxels"] for p in f["listed"]]
    assert sizes == sorted(sizes, reverse=True) and f["unsupported_voxels"] >= sum(sizes)
    text = connectivity.message(f)
    assert "7 of them have no fixed voxel" in text and "piece 5:" in text and "piece 6:" not in text and "and 2 more" in text


def test_a_loose_design_voxel_is_named_as_design(tmp_path):
    """A piece of design voxels: a design box in two bodies. The island body is design with the label `design`."""
    import trimesh
    a = trimesh.creation.box(extents=(20, 10, 10))
    a.apply_translation((10, 5, 5))
    b = trimesh.creation.box(extents=(3, 3, 3))
    b.apply_translation((24.5, 5, 5))                                  # a second body of the design STL, 4 mm from the first
    trimesh.util.concatenate([a, b]).export(tmp_path / "two.stl")
    p = make_case(tmp_path, h=1.0, iterations=2)
    d = yaml.safe_load(p.read_text())
    d["regions"]["design"] = "two.stl"
    p.write_text(yaml.safe_dump(d))
    f = connectivity.unsupported_pieces(build_problem(load_config(p)))
    assert f["pieces"] == 2 and f["unsupported_pieces"] == 1
    assert f["listed"][0]["labels"] == {"design": 27, "fixed": 0, "load": 0, "solid": 0}
    assert f["listed"][0]["first"][0]["label"] == "design"


def test_the_examples_have_no_piece_without_a_support():
    """The hook and the other examples run as before: every piece holds a support (hook, hook_mirror, hook_short, interfaces, ladle)."""
    names = ["hook/hook.yaml", "hook/hook_mirror.yaml", "hook/hook_short.yaml", "hook/hook_stress.yaml",
             "interfaces/bracket.yaml", "interfaces/bracket_nostock.yaml", "ladle/ladle.yaml"]
    for name in names:
        prob = build_problem(load_config(REPO / "examples" / name))
        f = connectivity.unsupported_pieces(prob)
        assert f["unsupported_pieces"] == 0, (name, f)


def test_the_numpy_free_import_of_the_cli():
    """`topopt.cli` must not load numpy at import. The new module is imported inside the functions that use it."""
    import subprocess
    import sys
    code = "import sys, topopt.cli; sys.exit(1 if ('numpy' in sys.modules or 'topopt.connectivity' in sys.modules) else 0)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def test_a_refused_perimeter_bound_does_not_hide_a_piece_with_no_support(tmp_path, capsys):
    """Integration of F2a (#53) and A4c (#57): the loose piece comes first in `topopt run`, and both facts show in `topopt check`."""
    from topopt.config import UserError

    p = case_with(tmp_path, CORNER, mutate=lambda d: d["settings"].update(perimeter_max_mm2=1.0))
    with pytest.raises(UserError) as e:
        check_data(p)
    text = str(e.value)
    assert text.index("has no fixed voxel") < text.index("settings.perimeter_max_mm2 is 1 mm2")
    assert main(["check", str(p)]) == 1
    err = capsys.readouterr().err
    assert "has no fixed voxel" in err and "settings.perimeter_max_mm2 is 1 mm2" in err
    assert main(["run", str(p), "--out", str(tmp_path / "out"), "--quiet"]) == 1
    err = capsys.readouterr().err
    assert "has no fixed voxel" in err and "perimeter_max_mm2" not in err      # the loose piece stops the run first
    (tmp_path / "plain").mkdir()
    p2 = case_with(tmp_path / "plain", {}, mutate=lambda d: d["settings"].update(perimeter_max_mm2=1.0))
    with pytest.raises(UserError) as e2:                                           # no loose piece: the refusal alone
        check_data(p2)
    assert "has no fixed voxel" not in str(e2.value) and "settings.perimeter_max_mm2 is 1 mm2" in str(e2.value)


# P1 (F2-3): "the fixed region does not touch the design space" is one rule in `connectivity.support_touch`, a fact with one
# severity (a warning) in `topopt check` and in the part builder. An error stays only where the run cannot work (a piece
# with no support).
def bridged(tmp_path):
    """The cantilever with its design pulled back to x 4..20 and a solid bridge x 0..4: the plate (x -2..0) shares no face with the design."""
    def pull_back(d):
        box(tmp_path / "design.stl", (4, 0, 0), (20, 10, 10))
    return case_with(tmp_path, {"bridge": ((0, 0, 0), (4, 10, 10))}, mutate=pull_back)


def test_a_fixed_region_that_reaches_the_design_through_a_solid_body_is_a_fact_and_the_run_works(tmp_path):
    from topopt.optimize import run
    from topopt.snapshot import RunWriter
    p = bridged(tmp_path)
    prob = build_problem(load_config(p))
    f = connectivity.support_touch(prob)
    assert f["touches"] is False and f["held_by"] == [] and f["design_held"] is True
    assert f["nearest_voxels"] == 5.0 and f["nearest_mm"] == 5.0         # plate centre x -0.5, design centre x 4.5
    assert connectivity.unsupported_pieces(prob)["unsupported_pieces"] == 0 and connectivity.floating_error(prob) is None
    text = connectivity.touch_message(f)
    assert text.startswith("no fixed voxel touches a design voxel: the nearest design voxel is 5 voxels (5 mm) from a fixed voxel")
    assert "Every design voxel is in a piece that holds a support" in text and "Options: overlap the design space" in text
    d = check_data(p)
    assert [w for w in d["warnings"] if "touches a design voxel" in w] == [text]            # a warning, the same text
    assert d["connectivity"]["fixed_touches_design"] is False and not any("stops with this error" in w for w in d["warnings"])
    writer = RunWriter(tmp_path / "run", quiet=True)
    writer.clean()
    run(prob, writer)                                                                       # the run works
    assert len((tmp_path / "run" / "history.jsonl").read_text().splitlines()) == 3


def test_a_fixed_region_that_touches_the_design_has_no_touch_fact(tmp_path):
    prob = build_problem(load_config(make_case(tmp_path, h=1.0, iterations=2)))
    assert connectivity.support_touch(prob) == {"touches": True, "held_by": []}
    assert connectivity.touch_message(connectivity.support_touch(prob)) is None


def test_a_fixed_region_that_holds_nothing_is_the_case_where_the_run_cannot_work(tmp_path):
    """The plate is far from the rest: the design and the tip are a piece with no support. `check` says so, with the touch fact."""
    def far_plate(d):
        box(tmp_path / "plate.stl", (-12, 0, 0), (-10, 10, 10))
    p = case_with(tmp_path, {}, mutate=far_plate)
    prob = build_problem(load_config(p))
    f = connectivity.support_touch(prob)
    assert f["touches"] is False and f["design_held"] is False
    assert "Some design voxels are in a piece with no support" in connectivity.touch_message(f)
    assert connectivity.floating_error(prob) is not None                                    # `topopt run` stops
    warnings = check_data(p)["warnings"]
    assert any(w.endswith("(`topopt run` stops with this error)") for w in warnings)
    assert any("touches a design voxel" in w for w in warnings)
