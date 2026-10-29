"""The cleanup list through the ratchet (CONTRACTS.md sections 29.3 and 36.7): the `marks:` source applies each mark with
`apply_marks` and the YAML rules judge it. These tests need gmsh, scikit-fem and one linear solver (the two-body test needs
printcheck only). A run takes 10 to 60 s; deselect with -m "not slow".

Done 4: every applied candidate of a small case passes `topopt verify` meshing (a closed manifold, sewn).
Done 5: one search candidate is accepted and one is rejected, each with its reason in `ratchet.jsonl`.
Done 6: facts for each body; the body with no fixed and no load voxel is a candidate; the ratchet judges it; the list of the
        accepted parent goes on in the same root.

The ratchet outcome of a mark depends on the lattice phase: the cut of a mark can reach into the parent (the detected set has a
collar under the feature), and the surface that `topopt verify` meshes can have an edge below its 0.05 mm limit. So the ratchet
tests use the fixtures at the phase that the list tests (tests/test_cleanup.py) show as the reference, and they do not claim the same
outcome at other phases. The list tests assert the list at three phases.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import needs_gmsh, needs_skfem  # noqa: E402
from test_cleanup import PLATE, BAR, THIN, body_part, cube, cyl_z, posts_part, rods_part, run_at, web_part  # noqa: E402
from test_feedback import tree_hash  # noqa: E402

from topopt import cleanup as cl  # noqa: E402
from topopt import ratchet as rt  # noqa: E402

pytestmark = [pytest.mark.slow]
VERIFY = [needs_gmsh, needs_skfem]


@pytest.fixture
def need_a_solver():
    from topopt.verify.factor import factor_availability
    if not any(v["available"] for v in factor_availability().values()):
        pytest.skip("no linear solver can run here")
    from topopt.verify.memory import available_ram_bytes
    free = available_ram_bytes()
    if free is not None and free < 1.5e9:
        pytest.skip(f"only {free / 1e9:.2f} GB of RAM is free for this process; verify needs about 0.5 GB for a 1 mm mesh")


def with_rules(ypath: Path, rules, mesh_size_mm=1.0) -> Path:
    doc = yaml.safe_load(ypath.read_text())
    doc["acceptance"] = {"mesh_size_mm": mesh_size_mm, "rules": rules}
    ypath.write_text(yaml.safe_dump(doc, sort_keys=False))
    return ypath


def judge(run, ypath, marks_file, root, capsys):
    code = rt.main(["run", str(run), "--source", f"marks:{marks_file}", "--root", str(root), "--yaml", str(ypath)])
    capsys.readouterr()
    assert code == 0
    return rt.read_log(root)[0]


def check(row, rule):
    return [c for c in row["checks"] if c["rule"] == rule]


# --- Done 5 ---------------------------------------------------------------------------------------------------------------

@pytest.mark.usefixtures("need_a_solver")
@needs_gmsh
@needs_skfem
def test_done5_one_search_candidate_is_accepted_and_one_rejected_with_their_reasons(tmp_path, capsys):
    """The thin bar with a tall post near the root (x 8) and a short post near the tip (x 16). The list has both as spurs, the
    taller one first. The stiffness rule has a limit of 5 %: the post near the root carries load (removing it costs about 7 %);
    the post near the tip does not (about 3 %). The list is not edited by hand."""
    run, y = run_at(tmp_path / "case", posts_part())
    with_rules(y, [{"rule": "watertight"}, {"rule": "bodies"},
                   {"rule": "stiffness", "max_drop_pct": 5, "source": "test: between the cost of the two posts"},
                   {"rule": "mass", "must_fall": True}])
    rep = cl.cleanup_list(run, out_dir=tmp_path / "list", yaml_path=y)
    assert rep["ok"] and rep["counts"]["listed"] == 2
    before = tree_hash(run)
    base, rejected, accepted = judge(run, y, tmp_path / "list" / "marks.json", tmp_path / "root", capsys)
    assert tree_hash(run) == before and base["decision"] == "BASELINE" and base["errors"] is None
    assert (rejected["id"], rejected["decision"], rejected["failed"]) == ("c001", "REJECT", ["stiffness"])
    assert (accepted["id"], accepted["decision"], accepted["failed"], accepted["promoted"]) == ("c002", "ACCEPT", [], "acc_01")
    for row, ref in ((rejected, "marks.json#1"), (accepted, "marks.json#2")):
        assert row["source"]["ref"] == ref and row["source"]["facts"]["change_mm3"] < -100 and row["errors"] is None
        assert row["source"]["facts"]["marks_status"] == "applied" and row["judge"]["tools"] == {"printcheck": "exit 0", "verify": "exit 0"}
        assert [c["state"] for c in check(row, "mass")] == ["pass"] and [c["state"] for c in check(row, "watertight")] == ["pass"]
    (bad,), (good,) = check(rejected, "stiffness"), check(accepted, "stiffness")
    assert bad["state"] == "fail" and bad["delta_pct"] < -5.0 and "the limit is 5 %" in bad["reason"]      # the reason is recorded
    assert good["state"] == "pass" and -5.0 < good["delta_pct"] and "the limit is 5 %" in good["reason"]
    # the margins of this fixture, a fact for the reader: both are about 2 percentage points from the limit at the 1 mm mesh (drops
    # 7.401 % and 2.990 %, the same in 7 of 7 runs). There is no scatter between runs. A shift of the whole case changes the drops by
    # 6 to 21 percentage points (CONTRACTS 29.3 limit 5), so this test holds the position.
    assert (-bad["delta_pct"] - 5.0) > 0.5 and (5.0 + good["delta_pct"]) > 0.5
    assert (tmp_path / "root" / "acc_01" / "labels.bin").is_file() and not (tmp_path / "root" / "acc_02").exists()


# --- C3c: a sub-voxel shift of the case ---------------------------------------------------------------------------------------

SUBVOXEL = (0.31, 0.0, 0.19)     # the phase of the C3b review: apply_marks leaves clusters of vertices 1e-5 mm apart (CONTRACTS 14)


def posts_at_subvoxel_shift(tmp_path):
    run, y = run_at(tmp_path / "case", posts_part(), shift=SUBVOXEL)
    with_rules(y, [{"rule": "watertight"}, {"rule": "bodies"},
                   {"rule": "stiffness", "max_drop_pct": 5, "source": "test: between the cost of the two posts"},
                   {"rule": "mass", "must_fall": True}])
    rep = cl.cleanup_list(run, out_dir=tmp_path / "list", yaml_path=y)
    assert rep["ok"] and rep["counts"]["listed"] == 2, rep.get("error") or rep["counts"]
    return run, y


@pytest.mark.usefixtures("need_a_solver")
@needs_gmsh
@needs_skfem
def test_c3c_both_candidates_at_a_subvoxel_shift_are_judged_on_a_rule_after_the_weld_retry(tmp_path, capsys):
    """The posts case at the shift (0.31, 0, 0.19). The edited.stl of each mark has edges of 1e-5 mm (marching cubes on a flat
    face on a lattice node). Before C3c verify refused both (`no_input`, "prepared surface has an edge of 0.0000 mm") and a
    good edit was lost. Now verify welds the input once, after that refusal, and both candidates are judged on a rule."""
    run, y = posts_at_subvoxel_shift(tmp_path)
    before = tree_hash(run)
    base, first, second = judge(run, y, tmp_path / "list" / "marks.json", tmp_path / "root", capsys)
    assert tree_hash(run) == before and base["decision"] == "BASELINE" and base["errors"] is None
    for row in (first, second):
        assert row["errors"] is None and row["judge"]["tools"] == {"printcheck": "exit 0", "verify": "exit 0"}, (row["errors"], row["checks"])
        assert all(c["state"] in ("pass", "fail") for c in row["checks"]), row["checks"]      # no `no_input`
        (stiff,) = check(row, "stiffness")
        assert "the limit is 5 %" in stiff["reason"] and stiff["delta_pct"] is not None
    assert (first["id"], first["decision"], first["failed"]) == ("c001", "REJECT", ["stiffness"])
    assert (second["id"], second["decision"], second["failed"], second["promoted"]) == ("c002", "ACCEPT", [], "acc_01")
    # the retry is a fact in the verify record of each candidate (the folder of the accepted one is the promoted folder), a line of
    # the verify text and a note of the ratchet row
    from topopt.verify.report import summary_lines, weld_line
    root = tmp_path / "root"
    for row in (first, second):
        vj = json.loads((root / (row["promoted"] or row["shadow_dir"]) / "verify" / "verify.json").read_text())
        weld = vj["mesh"]["surface_weld_retry"]
        assert weld["step"] == "weld_short_edges" and weld["limit_mm"] == 0.05 and weld["short_edges_in_input"] > 0
        assert weld["edges_welded"] > weld["short_edges_in_input"]                  # a second pass welded the edges that the first made
        assert weld["bodies_before"] == weld["bodies_after"] == 1 and weld["vertices_dropped"] > 0
        assert weld["vertices_before"] == weld["vertices_after"] + weld["vertices_merged"] + weld["vertices_dropped"]
        assert 0 < weld["max_vertex_move_mm"] < weld["limit_mm"]                    # measured 0.034 and 0.038 mm against 0.05 mm
        assert (weld["genus_before"], weld["genus_after"], weld["euler_before"], weld["euler_after"]) == ([2], [0], [-2], [2])
        assert abs(weld["volume_change_mm3"]) < 0.01 and weld["refused_first"].startswith("prepared surface has an edge of")
        assert weld_line(weld) in summary_lines(vj) and row["facts"]["notes"] == [weld_line(weld)]
    base_vj = json.loads((root / "acc_00" / "verify" / "verify.json").read_text())
    assert "surface_weld_retry" not in base_vj["mesh"] and "notes" not in base["facts"]     # the start part passes at the first try
    assert not [x for x in summary_lines(base_vj) if x.startswith("surface: the prepared surface was refused")]


@pytest.mark.usefixtures("need_a_solver")
@needs_gmsh
@needs_skfem
def test_c3c_with_the_weld_retry_off_verify_refuses_both_candidates_as_before(tmp_path, capsys, monkeypatch):
    """The companion of the test above: the same case with the retry switched off. It is the state before C3c: both candidates are
    `no_input` on every rule that needs verify, with the refusal text in the error. The test shows that the retry is what changes
    the outcome. The clean retry of F1-5 (`clean_short_edges`) welds the edges that stay with the same `weld_short_edges`, so it is
    switched off too (review MN5): else only its collapse would run and the old state would hold by chance."""
    from topopt.tet import surface as ts

    def off(*a, **k):
        raise ts._WeldFailed("the weld is switched off in this test")

    def clean_off(*a, **k):
        raise ts._CleanFailed("the clean retry is switched off in this test")
    monkeypatch.setattr(ts, "weld_short_edges", off)
    monkeypatch.setattr(ts, "clean_short_edges", clean_off, raising=False)           # the test runs on the code before F1-5 too
    run, y = posts_at_subvoxel_shift(tmp_path)
    base, first, second = judge(run, y, tmp_path / "list" / "marks.json", tmp_path / "root", capsys)
    assert base["decision"] == "BASELINE" and base["errors"] is None
    for row in (first, second):
        (stiff,) = check(row, "stiffness")
        assert stiff["state"] == "no_input" and "prepared surface has an edge of" in json.dumps(row), row
        assert row["decision"] == "REJECT" and "stiffness" in row["failed"]


# --- Done 4 ---------------------------------------------------------------------------------------------------------------

KINDS = [
    pytest.param(web_part, {"member_min_cv": 0.12}, {"spur", "member"}, id="spur-and-fair-member"),
    pytest.param(web_part, {"member_min_cv": 0.12, "web_flatness": 9.0}, {"spur", "member"}, id="spur-and-smooth-tube"),
    pytest.param(rods_part, {}, None, id="round-rods"),
]


@pytest.mark.usefixtures("need_a_solver")
@needs_gmsh
@needs_skfem
@pytest.mark.parametrize("make,options,detectors", KINDS)
def test_done4_every_applied_candidate_is_a_closed_manifold_that_verify_meshes(tmp_path, capsys, make, options, detectors):
    run, y = run_at(tmp_path / "case", make())
    with_rules(y, [{"rule": "watertight"}, {"rule": "bodies"}, {"rule": "stiffness", "max_drop_pct": 99, "source": "test: any drop"}])
    rep = cl.cleanup_list(run, out_dir=tmp_path / "list", yaml_path=y, options=options)
    assert rep["ok"] and rep["counts"]["listed"] == 2, rep.get("error") or rep["counts"]
    if detectors is not None:
        assert {m["cleanup"]["detector"] for m in rep["marks"]} == detectors
    rows = judge(run, y, tmp_path / "list" / "marks.json", tmp_path / "root", capsys)
    # one row for each MARK: a candidate whose voxel set has two 26-components is two marks (CONTRACTS 29.3 part 4; the spur of
    # the round rods has a collar on the bar apart from the rod)
    assert rows[0]["decision"] == "BASELINE" and len(rows) == 1 + len(rep["marks"]) and len(rep["marks"]) >= rep["counts"]["listed"]
    for row in rows[1:]:
        assert row["errors"] is None and row["judge"]["tools"] == {"printcheck": "exit 0", "verify": "exit 0"}, row["errors"]
        assert (row["decision"], row["failed"]) == ("ACCEPT", []), row["checks"]
        rep_a = json.loads((tmp_path / "root" / f"cand_{row['id'][1:]}" / "apply_marks.json").read_text())
        mk = rep_a["marks"][0]
        assert mk["status"] == "applied" and rep_a["totals"]["closed_after"] and rep_a["totals"]["bodies_after"] == 1
        assert rep_a["totals"]["reload_from_disk"]["open_edges"] == 0 and rep_a["totals"]["reload_from_disk"]["non_manifold_edges"] == 0
        assert rep_a["totals"]["surface_quality"]["after"]["closed_manifold"] is True
        assert all(c["state"] == "pass" for c in check(row, "stiffness")), "verify solved the candidate: it has a stiffness"


# --- Done 6 ---------------------------------------------------------------------------------------------------------------

def test_done6_a_loose_body_is_judged_by_the_ratchet_and_the_list_of_the_accepted_parent_goes_on_in_the_same_root(tmp_path, capsys):
    """Printcheck rules only (no gmsh): a two-body part (the bar with a loose cube) and a post on the bar. The list holds the
    body; the ratchet accepts the removal (bodies 2 -> 1); the list of acc_01 holds the post; the same root judges it."""
    part = body_part() + cyl_z(16, 0, 8, 16, 2.5)
    run, y = run_at(tmp_path / "case", part)
    with_rules(y, [{"rule": "watertight"}, {"rule": "bodies"}, {"rule": "mass", "must_fall": True}])
    rep = cl.cleanup_list(run, out_dir=tmp_path / "list1", yaml_path=y)
    b = rep["bodies"]                                                           # the facts for each body
    assert (b["count"], b["holding_fixed_or_load"]) == (2, 1) and sorted(x["removable"] for x in b["list"]) == [False, True]
    assert all({"volume_mm3", "bbox_mm", "holds_fixed_or_load", "cavity", "triangles"} <= set(x) for x in b["list"])
    assert [m["cleanup"]["detector"] for m in rep["marks"]] == ["body"] and rep["counts"]["not_listed"] == 1
    root = tmp_path / "root"
    base, body = judge(run, y, tmp_path / "list1" / "marks.json", root, capsys)
    assert base["facts"]["bodies"] == 2 and body["decision"] == "ACCEPT" and body["promoted"] == "acc_01"
    assert body["facts"]["bodies"] == 1 and [c["state"] for c in check(body, "bodies")] == ["pass"]
    assert body["facts"]["mass_g"] < base["facts"]["mass_g"] and body["judge"]["tools"] == {"printcheck": "exit 0"}
    # the features of the accepted parent: the out folder is beside the root, so the root stays its own
    files_before = sorted(p.name for p in root.iterdir())
    rep2 = cl.cleanup_list(root / "acc_01", yaml_path=y, root=root)
    assert Path(rep2["out_dir"]) == tmp_path / "root-cleanup-acc_01" and sorted(p.name for p in root.iterdir()) == files_before
    assert [m["cleanup"]["detector"] for m in rep2["marks"]] == ["spur"] and rep2["bodies"]["count"] == 1
    # the estimate (review M1): the root holds the body candidate and its baseline, and the list has 1 mark: 1 candidate, 1 call (not 0)
    je = rep2["estimate"]["judge"]
    assert (je["candidates"], je["judge_calls"], je["baseline_included"]) == (len(rep2["marks"]), 1, False) and je["seconds_total_model"] > 0
    assert je["measured_from_root"]["candidates_in_log"] == 1 and je["measured_from_root"]["seconds_total"] > 0
    # the same list without the root: its baseline is judged too (2 calls)
    alone = cl.cleanup_list(root / "acc_01", out_dir=tmp_path / "list2-no-root", yaml_path=y)["estimate"]["judge"]
    assert (alone["candidates"], alone["judge_calls"], alone["baseline_included"]) == (1, 2, True) and "measured_from_root" not in alone
    rows = judge(run, y, Path(rep2["out_dir"]) / "marks.json", root, capsys)       # the same root goes on
    assert [r["id"] for r in rows] == ["c000", "c001", "c002"]
    spur = rows[2]
    assert (spur["decision"], spur["failed"], spur["promoted"]) == ("ACCEPT", [], "acc_02") and spur["source"]["ref"] == "marks.json#1"
    assert spur["facts"]["mass_g"] < body["facts"]["mass_g"] and (root / "acc_02" / "result" / "final.stl").is_file()
