"""The seam of `apply_marks` (C3a-fix, CONTRACTS.md section 29): the new surface is sewn to the old one, and the edited part
is a surface that `topopt verify` can mesh.

The defect (D3 research, RESEARCH.md 5.2): the boolean `(part - zone) + (new surface ^ zone)` joined two triangulations of
one surface along the wall of the zone. They differ by 0 to 65 um there and cross many times, so the boolean left a ribbon of
sliver triangles with fold edges and pinch vertices. printcheck and the apply_marks report said "closed, one body", but
`topopt verify` exited 2 on every applied candidate of the hook and the ladle.

The parts of these tests are curved (spheres and cylinders with many facets, like an optimized part and unlike a box): a part
of flat faces has no seam to speak of. The same cases fail with the old boolean (see `test_the_boolean_seam_...`). Tests that
mesh need gmsh (`needs_gmsh`); the others need numpy, scipy and numba only.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_gmsh, needs_skfem, on_the_platform_of_the_measurement, recorded_hook_run  # noqa: E402
from test_apply_marks import BAR, FAST, M, PLATE, am, cube, cyl_x, doc_for, make_run  # noqa: E402
from test_feedback import mark  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RECORDING = recorded_hook_run()
HOOK_DIR = REPO / "examples" / "hook"
needs_recording = pytest.mark.skipif(not (RECORDING / "result" / "final.stl").is_file(),
                                     reason="needs the recorded hook run recordings/hook_*")

PARTS = {
    "rod": lambda: PLATE() + cube((0, -4, 0), (24, 4, 8)) + cyl_x(0, 8, 0, 24, 2.0, 64),      # a rod on the bar: half of it sticks out
    "ball": lambda: PLATE() + BAR() + M.sphere(3.0, 64).translate((12, 0, 9.5)),
    "bowl": lambda: PLATE() + (M.sphere(7.0, 96) - M.sphere(5.5, 96)).translate((14, 0, 5)) + cube((0, -2, 3), (8, 2, 6)),
    "bump": lambda: PLATE() + BAR() + M.sphere(2.5, 64).translate((1.5, 0, 8.5)),
}
# (part, mark type, points, radius, amount): every case is applied. The old boolean gave 47 to 297 fold edges on each, and
# `prepare_surface` of verify refused each. max_folds: the strip between two loops that cross has a twist; the sewn seam of the
# other cases has none.
CASES = {
    "rod_smooth": ("rod", "smooth", [[12, 0, 9.8]], 2.5, None, 0),
    "ball_smooth": ("ball", "smooth", [[12, 0.5, 11.6]], 2.5, None, 0),
    "bowl_smooth": ("bowl", "smooth", [[14, 0, 12.0]], 3.0, None, 0),
    "bowl_remove": ("bowl", "remove", [[14, 7, 5]], 2.0, 0.5, 2),
    # the wall of the zone meets the flat top of the bar at a small angle: the two cut loops there are 0.7 mm apart (2 nodes)
    "bump_remove": ("bump", "remove", [[1.5, 0, 10.5]], 3.0, None, 2),
}
SIZE = 1.2          # a mesh size in mm for the small parts (the voxel of the test cases is 1.2 mm)


def run_case(root, case, sewn=True):
    """(report, run folder) of one case. With sewn=False the seam is the plain boolean (the code before the fix)."""
    part, typ, pts, radius, amount, _max = CASES[case]
    run, _ = make_run(root / case / ("sewn" if sewn else "boolean"), PARTS[part]())
    mk = mark(1, typ, pts, radius=radius)
    if amount is not None:
        mk["amount_mm"] = amount
    with pytest.MonkeyPatch.context() as mp:
        if not sewn:
            mp.setattr(am, "_graft", lambda *a, **k: (None, None, "the sewn seam is switched off in this test"))
        rep = am.apply_marks(run, doc_for(run, mk), options=FAST, out_dir=Path(root) / case / ("o_sewn" if sewn else "o_boolean"))
    return rep, run


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    """Each case once, sewn (the default) and with the boolean; the other tests read the results."""
    root = tmp_path_factory.mktemp("seam")
    out = {}

    def get(case, sewn=True):
        if (case, sewn) not in out:
            out[(case, sewn)] = run_case(root, case, sewn)
        return out[(case, sewn)]
    return get


def surface(path):
    from topopt.tet import surface as ts
    V, F = ts.load_stl_surface(str(path))
    return ts, V, F


@pytest.mark.parametrize("case", list(CASES))
def test_the_sewn_seam_is_a_closed_manifold_that_verify_accepts(cases, case):
    rep, _run = cases(case)
    e = rep["marks"][0]
    assert e["status"] == "applied", (e["status"], e.get("reason"))
    seam = e["steps"][0]["seam"]
    assert seam["method"] == "sewn" and seam["loops"] >= 1 and seam["wall_triangles_dropped"] > 0, seam
    assert abs(seam["volume_vs_boolean_mm3"]) < 0.5, seam                 # the sewn part is the part of the boolean, to a sliver
    edited = Path(rep["out_dir"]) / "edited.stl"
    ts, V, F = surface(edited)
    r = ts.surface_report(V, F)
    assert r["closed_manifold"] and r["nonmanifold_vertices"] == 0 and r["bodies"] == 1, r
    assert r["fold_edges"] <= CASES[case][5], r["fold_edges"]
    ts.prepare_surface(str(edited), SIZE, verbose=False)         # the surface step of `topopt verify`: it raises when it refuses
    sq = rep["totals"]["surface_quality"]["after"]
    assert sq["closed_manifold"] and sq["fold_edges"] == r["fold_edges"] and not rep["problems"]


# The route of the surface step for each plain-boolean case at SIZE (measured on Linux x86_64 with manifold3d 3.5.4, review MN4):
#   rod_smooth                   prepared at the first try (48 fold edges, the preparation repairs them)
#   ball_smooth, bowl_remove     refused: "prepared surface is not a closed manifold" (no retry answers that refusal)
#   bowl_smooth, bump_remove     refused for an edge under the limit (the weld of the input moves a vertex by 0.0819 and 0.136 mm, more
#                                than the radius 0.05 mm); the clean retry prepares them (F1-5)
ROUTES = {
    "rod_smooth": "first try",
    "ball_smooth": "not a closed manifold",
    "bowl_smooth": "clean retry",
    "bowl_remove": "not a closed manifold",
    "bump_remove": "clean retry",
}


def route_of(ts, edited, clean_off=False):
    """How the surface step of `topopt verify` ends for a file: 'first try', 'weld retry', 'clean retry', 'not a closed manifold' or
    'short edge' (refused for an edge under the limit)."""
    with pytest.MonkeyPatch.context() as mp:
        if clean_off:
            def off(*_a, **_k):
                raise ts._CleanFailed("the clean retry is switched off in this test")
            mp.setattr(ts, "clean_short_edges", off, raising=False)           # the test runs on the code before F1-5 too
        try:
            prep = ts.prepare_surface(str(edited), SIZE, verbose=False)
        except ts.SurfaceRejected as e:
            return "not a closed manifold" if str(e).startswith("prepared surface is not a closed manifold") else "short edge"
    steps = [st["step"] for st in prep.steps]
    return ("weld retry" if "weld_short_edges" in steps else "clean retry" if "clean_short_edges" in steps else "first try")


def test_the_boolean_seam_of_the_same_cases_has_fold_edges_and_the_surface_step_ends_as_measured(cases):
    """The cause, as a test: with the sewn seam switched off the same edits leave fold edges at the seam. `prepare_surface` refused
    4 of the 5 files (the D3 research; the rod, with 48 fold edges, it accepts at this size: folds are a danger for the mesher, not a
    sure refusal, so the report gives the count). The route of each case is asserted, with the clean retry (F1-5, CONTRACTS section 14)
    switched off, which is the state of the D3 research, and with it on. The clean retry answers the refusal for an edge under the
    limit only: `bowl_smooth` and `bump_remove` get through the surface step, `ball_smooth` and `bowl_remove` stay refused (their
    prepared surface is not a closed manifold). What happens next is tested below: `bump_remove` meshes (`topopt verify` exits 0),
    `bowl_smooth` stops in gmsh with an error that names the clean retry."""
    from topopt.tet import surface as ts
    for case in CASES:
        rep, _run = cases(case, sewn=False)
        e = rep["marks"][0]
        assert e["status"] == "applied" and e["steps"][0]["seam"]["method"] == "boolean"
        assert e["steps"][0]["seam"]["why_not_sewn"] == "the sewn seam is switched off in this test"
        q = rep["totals"]["surface_quality"]["after"]
        assert q["fold_edges"] > 0, (case, q)                                        # the fold edges are there in all cases
    if not on_the_platform_of_the_measurement():
        # The routes below were measured on Linux x86_64. The first full run on an Apple silicon Mac ended `bump_remove` at the first try
        # where Linux refuses it for an edge under the limit: the boolean (manifold3d) depends on the floating-point order.
        pytest.skip("the route of the surface step for the plain boolean cases was measured on Linux x86_64 with manifold3d 3.5.4")
    edited = {case: Path(cases(case, sewn=False)[0]["out_dir"]) / "edited.stl" for case in CASES}
    without_clean = {case: route_of(ts, edited[case], clean_off=True) for case in CASES}
    assert without_clean == {"rod_smooth": "first try", "ball_smooth": "not a closed manifold", "bowl_smooth": "short edge",
                             "bowl_remove": "not a closed manifold", "bump_remove": "short edge"}        # 4 of the 5 refused
    assert {case: route_of(ts, edited[case]) for case in CASES} == ROUTES


@needs_gmsh
def test_the_plain_boolean_cases_that_the_clean_retry_prepares_are_meshed_or_fail_with_the_retry_in_the_error(cases):
    """What the user sees after the surface step. `bump_remove` meshes (two tets under quality 0.05 at this size; `topopt verify`
    exits 0) and the facts name the clean retry. `bowl_smooth` fails in gmsh (the plain boolean leaves a near-fold that no check sees,
    review MJ1): the error must name the clean retry and the first refusal, the retries must end when the refinement does not change
    the surface, and the facts must hold the step. If gmsh meshes it on another platform, the facts name the retry."""
    from topopt.tet import MeshingFailed, mesh_stl
    mesh = mesh_stl(Path(cases("bump_remove", sewn=False)[0]["out_dir"]) / "edited.stl", size=SIZE)
    assert mesh.facts["volume_ratio_stl"] == pytest.approx(1.0, rel=0.02)
    if on_the_platform_of_the_measurement():       # on another platform the first try may mesh it: no clean step runs, the key is not set
        assert mesh.facts["surface_clean_retry"]["step"] == "clean_short_edges"
    edited = Path(cases("bowl_smooth", sewn=False)[0]["out_dir"]) / "edited.stl"
    try:
        mesh = mesh_stl(edited, size=SIZE)
    except MeshingFailed as e:
        text = str(e)
        assert "(clean retry)" in text and "prepared surface that was refused (prepared surface has an edge of 0.0" in text, text
        assert e.crossings["surface_retry"]["step"] == "clean_short_edges"
        assert e.crossings["stopped"] == "no_progress" and e.crossings["same_surface_as_attempt"] >= 1       # not 10 identical attempts
    else:   # the Mac run also ended `bowl_smooth` at "short edge", so a mesh from gmsh here comes after the clean step
        assert mesh.facts["surface_clean_retry"]["step"] == "clean_short_edges"


def test_gmsh_failing_on_the_cleaned_bowl_smooth_gives_the_retry_and_stops_at_the_same_surface(cases, monkeypatch):
    """The review MJ1 on the real surface of `bowl_smooth` (the plain boolean, SIZE): gmsh is faked (it names the place of the review),
    the surface step and the retries of `mesh_stl` are real. Before the fix the error had the gmsh text and 10 attempts of the same
    surface; the text did not say that the surface was the refused one with its short edges cleaned."""
    from test_tet import _ok_probe
    from topopt.tet import MeshingFailed, mesh_stl, mesher
    _ok_probe(monkeypatch)
    calls = []

    def fail(V, F, size, **kw):
        calls.append(len(F))
        raise MeshingFailed("algorithm 10: HXT 3D mesh failed; algorithm 1: PLC Error:  A segment and a facet intersect at point",
                            [(17.3449, -1.5664, 10.9362)])
    monkeypatch.setattr(mesher, "mesh_surface", fail)
    edited = Path(cases("bowl_smooth", sewn=False)[0]["out_dir"]) / "edited.stl"
    with pytest.raises(MeshingFailed) as e:
        mesh_stl(edited, isolate=False, size=SIZE, max_attempts=10)
    text = str(e.value)
    assert len(calls) <= 4 and e.value.crossings["stopped"] == "no_progress"          # measured: 3 attempts, the third equals the second
    assert e.value.crossings["same_surface_as_attempt"] == len(calls) - 1
    assert "so the next attempt would repeat this one" in text and "(clean retry)" in text
    assert "prepared surface that was refused (prepared surface has an edge of 0.0" in text
    assert e.value.crossings["surface_retry"]["step"] == "clean_short_edges" and "every surface attempt" not in text


def test_the_report_gives_the_surface_facts_before_and_after(cases):
    rep, run = cases("rod_smooth")
    sq = rep["totals"]["surface_quality"]
    ts, V0, F0 = surface(Path(run) / "result" / "final.stl")
    ts, V1, F1 = surface(Path(rep["out_dir"]) / "edited.stl")
    for key, (V, F) in (("before", (V0, F0)), ("after", (V1, F1))):
        r = ts.surface_report(V, F)
        got = sq[key]
        assert got["triangles"] == r["faces"] and got["closed_manifold"] is r["closed_manifold"]
        assert got["fold_edges"] == r["fold_edges"] and got["edge_min_mm"] == pytest.approx(r["edge_min_mm"], abs=1e-5)
        assert got["non_manifold_vertices"] == r["nonmanifold_vertices"] and got["bodies"] == r["bodies"]
        eu, ev, _c = ts.edge_table(F, len(V))
        assert got["edges_under_verify_limit"] == int((np.linalg.norm(V[eu] - V[ev], axis=1) < ts.MIN_EDGE_REJECT_MM).sum())
    assert sq["verify_edge_limit_mm"] == ts.MIN_EDGE_REJECT_MM
    assert "facts, not a refusal" in sq["note"]
    small = am.short_report(rep)["totals"]["surface_quality"]
    assert small["after"]["closed_manifold"] is True and small["before"]["fold_edges"] == 0
    assert small["verify_edge_limit_mm"] == ts.MIN_EDGE_REJECT_MM
    on_disk = json.loads((Path(rep["out_dir"]) / "apply_marks.json").read_text())
    assert on_disk["totals"]["surface_quality"]["after"]["fold_edges"] == sq["after"]["fold_edges"]


def test_a_boolean_fallback_with_new_fold_edges_is_named_in_the_problems_and_the_short_reply(cases):
    """Review m1: the facts are in `surface_quality`, but the caller must not have to compare fold counts. A step that fell back to
    the plain boolean and an edited.stl with more fold edges than final.stl give a line in `problems`; the short reply of the
    mark carries the seam methods."""
    rep, _run = cases("rod_smooth", sewn=False)
    q = rep["totals"]["surface_quality"]
    assert q["after"]["fold_edges"] > q["before"]["fold_edges"]
    lines = [p for p in rep["problems"] if "plain boolean" in p]
    assert len(lines) == 1 and f"{q['after']['fold_edges']} fold edges" in lines[0] and "switched off in this test" in lines[0]
    assert am.short_report(rep)["marks"][0]["seam_methods"] == {"boolean": 1}
    sewn, _run = cases("rod_smooth")
    assert not [p for p in sewn["problems"] if "plain boolean" in p]
    assert am.short_report(sewn)["marks"][0]["seam_methods"] == {"sewn": 1}


def test_the_followers_of_a_loop_never_include_a_vertex_of_the_old_part():
    """Review m3: after a first loop pair is sewn, FB holds vertices of the old part; a second loop must not move them."""
    FA = np.array([[0, 1, 2], [2, 1, 5]])
    FB = np.array([[5, 6, 7], [7, 6, 8], [8, 9, 5]])        # 5 is an old vertex (it is in FA) that a Q vertex became
    assert am._followers(FA, FB, [8, 9]).tolist() == [6, 7]


def test_the_area_floor_follows_the_float32_spacing_of_the_coordinates():
    """Review m4: the floor under which a triangle has no normal is pitch x the float32 spacing at the largest coordinate."""
    near = np.array([[0.0, 0.0, 0.0], [80.0, 1.0, 1.0]])
    far = np.array([[0.0, 0.0, 0.0], [300.0, 1.0, 1.0]])
    assert am._area_floor(near, 0.35) == pytest.approx(0.35 * np.spacing(np.float32(80.0)))
    assert am._area_floor(far, 0.35) == pytest.approx(0.35 * np.spacing(np.float32(300.0)))
    assert am._area_floor(far, 0.35) > 3.5 * am._area_floor(near, 0.35)
    assert am._area_floor(np.zeros((2, 3)), 0.35) > 0                                  # a part at the origin has a floor too


def test_the_strip_tier_reports_the_old_vertices_that_the_weld_removes(tmp_path):
    """Review m2: in tier 2 a vertex of the old loop can be removed (it moves onto a neighbour). The facts give the count and the
    largest move. Tier 1 alone does not move the old part, and has no such fact."""
    rep, _run = run_case(tmp_path, "bump_remove")                  # two loops by strip
    seam = rep["marks"][0]["steps"][0]["seam"]
    assert seam["loops_joined_by_strip"] > 0 and seam["old_vertices_removed"] > 0
    assert 0.0 < seam["old_removed_move_max_mm"] < 0.5
    rep, _run = run_case(tmp_path / "t1", "bowl_smooth")           # the loop is joined edge to edge
    seam = rep["marks"][0]["steps"][0]["seam"]
    assert seam["method"] == "sewn" and seam["loops_joined_by_strip"] == 0
    assert "old_vertices_removed" not in seam and "old_removed_move_max_mm" not in seam


def test_the_weld_says_when_the_pass_guard_stopped_it(tmp_path, monkeypatch):
    """Review n1: SEAM_WELD_PASSES only guards the loop. When a pass still collapsed edges at the end, the facts say so."""
    monkeypatch.setattr(am, "SEAM_WELD_PASSES", 1)
    rep, _run = run_case(tmp_path, "bump_remove")
    seam = rep["marks"][0]["steps"][0]["seam"]
    assert seam["method"] == "sewn" and seam["strip_edges_welded"] > 0 and seam["weld_stopped_by_guard"] is True


def test_the_simplify_try_does_not_sew_the_seam_again(tmp_path, monkeypatch):
    """Review n5: when the plain splice does not reload closed and no vertices share a float32 position, the simplify try takes the
    plain mesh. The sewing runs once for the step."""
    calls, real = [], am._reload_facts

    def reload_facts(mesh):
        facts, back = real(mesh)
        if not calls:
            calls.append("first")
            return {**facts, "closed": False}, back
        return facts, back
    count = []
    real_graft = am._graft
    monkeypatch.setattr(am, "_reload_facts", reload_facts)
    monkeypatch.setattr(am, "_separate_pinched", lambda mesh: (None, 0, 0))
    monkeypatch.setattr(am, "_graft", lambda *a, **k: (count.append(1), real_graft(*a, **k))[1])
    rep, _run = run_case(tmp_path, "rod_smooth")
    step = rep["marks"][0]["steps"][0]
    assert rep["marks"][0]["status"] == "applied" and "simplify" in step["float32_repair"]
    assert len(count) == 1, count


def test_a_run_that_changes_nothing_has_no_surface_facts(tmp_path):
    run, _ = make_run(tmp_path, PARTS["rod"]())
    rep = am.apply_marks(run, doc_for(run, mark(1, "smooth", [[12, 0, 30]], radius=2.0)), options=FAST)
    assert rep["marks"][0]["status"] != "applied" and "surface_quality" not in rep["totals"]


def test_surface_quality_reads_open_and_unreadable_files(tmp_path):
    tri = trimesh.Trimesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), np.array([[0, 1, 2]]), process=False)
    tri.export(str(tmp_path / "one.stl"))
    q = am._surface_quality(tmp_path / "one.stl")
    assert q["closed_manifold"] is False and q["open_edges"] == 3 and q["triangles"] == 1
    (tmp_path / "empty.stl").write_bytes(b"\0" * 84)
    q = am._surface_quality(tmp_path / "empty.stl")
    assert list(q) == ["error"] and "no triangles" in q["error"]
    assert list(am._surface_quality(tmp_path / "none.stl")) == ["error"]


def test_a_part_that_verify_would_refuse_is_named_in_the_problems(tmp_path, monkeypatch):
    """The file is written in each case: the facts are in the report, and `problems` says that verify would refuse it."""
    real = am._surface_quality
    monkeypatch.setattr(am, "_surface_quality", lambda p: {**real(p), "closed_manifold": False} if Path(p).name == "edited.stl" else real(p))
    rep, _run = run_case(tmp_path, "rod_smooth")
    assert rep["marks"][0]["status"] == "applied" and (Path(rep["out_dir"]) / "edited.stl").is_file()
    assert any("not a closed manifold" in p and "totals.surface_quality.after" in p for p in rep["problems"])


def test_the_triangles_outside_the_zone_are_the_triangles_of_the_part(cases):
    """The seam changes the triangles at the wall of the zone, and no triangle farther out. Bit for bit: the same float32 corners."""
    rep, run = cases("ball_smooth")
    ring = am.ZONE_NODES * rep["grid"]["pitch_mm"]
    far = rep["totals"]["edit_reach_mm"] + ring + 1.0
    centre = np.array([12, 0.5, 11.6])

    def triangles(path):
        m = trimesh.load(str(path), file_type="stl", force="mesh", process=False)
        t = np.asarray(m.triangles, np.float32)
        return t, np.asarray(m.triangles_center)

    before, c0 = triangles(Path(run) / "result" / "final.stl")
    after, c1 = triangles(Path(rep["out_dir"]) / "edited.stl")

    def keys(t):
        return {tuple(sorted(map(tuple, tri.tolist()))) for tri in t}

    kept0 = keys(before[np.linalg.norm(c0 - centre, axis=1) > far])
    kept1 = keys(after[np.linalg.norm(c1 - centre, axis=1) > far])
    assert len(kept0) > 100 and kept0 == kept1
    assert rep["totals"]["triangles_replaced"]["triangles_kept_bit_equal"] > 0


def test_the_frozen_regions_stay_exact_with_the_sewn_seam(tmp_path):
    """A smooth on a curved bump next to the fixed plate: the weight of the edit falls to 0 near the region, and the exact check of
    the report (the regions of the YAML) finds no change. The undo of a change inside a region, with a part surface that crosses
    the wall of the region, is tested below (`test_the_frozen_undo_...`)."""
    run, _ = make_run(tmp_path, PARTS["bump"]())
    rep = am.apply_marks(run, doc_for(run, mark(1, "smooth", [[1.5, 0, 10.5]], radius=3.0)), options=FAST)
    assert rep["marks"][0]["status"] == "applied"
    fe = rep["totals"]["frozen_exact"]
    assert fe["done"] and fe["unchanged"], fe
    ts, V, F = surface(Path(rep["out_dir"]) / "edited.stl")
    assert ts.surface_report(V, F)["closed_manifold"]


# ---- the undo of a change inside a frozen region (review B1 of C3a-fix) -------------------------------------
def rod_into_tip():
    """The bar (x 0..14) and a rod (axis x, y 0, z 2, r 1.5) that ends at x = 23.5, inside the load region `tip` (x 21..24, y -2..2,
    z 0..4) of the test case. The surface of the part crosses the wall x = 21 of the region, so the cut of the old part and the cut of
    the new surface both have a loop there."""
    return PLATE() + cube((0, -4, 0), (14, 4, 8)) + cyl_x(0, 2, 12, 23.5, 1.5, 96)


def frozen_change(run, edited):
    """(added, removed) mm3 inside the frozen regions of the YAML between final.stl and edited.stl (exact, manifold3d)."""
    run = Path(run)
    F, _facts = am._load_frozen(run, json.loads((run / "run.json").read_text()), None)
    m0 = am._manifold(trimesh.load(str(run / "result" / "final.stl"), force="mesh"))
    m1 = am._manifold(trimesh.load(str(edited), force="mesh"))
    return float(((m1 - m0) ^ F).volume()), float(((m0 - m1) ^ F).volume())


def test_a_sewn_undo_that_moves_the_frozen_side_falls_back_to_the_boolean():
    """The cut of the edited part is the edit itself, not a second triangulation of one surface: the loops are 0.2 mm apart, and
    the sewing moves the vertices of the frozen side. Without `exact_inside` the sewn undo leaves 0.2 mm3 in the region (tolerance
    0.01); with it the result is measured and the boolean runs. Fails on 88bed35 (no `exact_inside`, no measure)."""
    old = rod_into_tip()
    new = old - M.sphere(3.0, 96).translate((21.0, 0.0, 3.5 + 3.0 - 0.2))          # a dimple across the wall x = 21
    tip = cube((21, -2, 0), (24, 2, 4))
    edit = float(((old - new) ^ tip).volume())
    assert edit > 0.05
    _m, sewn, facts, _why = am._swap_inside(new, old, tip, 0.35)
    assert facts["method"] == "sewn" and am._region_residual(sewn, old, tip) > 3 * am.FROZEN_TOL_MM3        # why the check is needed
    mesh, kept, facts, why = am._swap_inside(new, old, tip, 0.35, exact_inside=True)
    assert why is None and mesh is not None
    assert facts["method"] == "boolean" and "moved the frozen side" in facts["why_not_sewn"]
    assert am._region_residual(kept, old, tip) < 1e-6


# (mark type, points, radius, amount, options): operators with the documented options frozen_keep_mm / frozen_full_mm; the edit
# reaches the region, the step undoes it. The first case is the one that the review measured at 1.34 mm3 removed from the region.
FROZEN_UNDO_CASES = {
    "smooth_at_the_wall": ("smooth", [[21.0, 0, 3.5]], 3.0, None, {"frozen_keep_mm": 0.0, "frozen_full_mm": 0.5}),
    "smooth_near_the_wall": ("smooth", [[19.5, 1.0, 3.2]], 4.0, None, {"frozen_keep_mm": 0.5, "frozen_full_mm": 1.0}),
    "shave_near_the_wall": ("remove", [[20.5, 0, 3.5]], 2.0, 0.4, {"remove_mode": "shave", "frozen_keep_mm": 0.0, "frozen_full_mm": 0.5}),
}


@pytest.mark.parametrize("case", list(FROZEN_UNDO_CASES))
def test_the_frozen_undo_leaves_the_regions_exact_when_the_part_surface_crosses_them(tmp_path, case):
    """Fails on 88bed35: there the sewn undo removed 0.10 to 1.34 mm3 from the region `tip`, and the report said `restored` and
    "undone" (the base code d6cbff0 keeps the region exact to 1e-9 mm3)."""
    typ, pts, radius, amount, opts = FROZEN_UNDO_CASES[case]
    run, _ = make_run(tmp_path, rod_into_tip())
    mk = mark(1, typ, pts, radius=radius)
    if amount is not None:
        mk["amount_mm"] = amount
    rep = am.apply_marks(run, doc_for(run, mk), options={"pitch_mm": 0.35, **opts}, out_dir=tmp_path / "o")
    e = rep["marks"][0]
    assert e["status"] == "applied", (e["status"], e.get("reason"))
    step = e["steps"][0]
    assert (step.get("frozen_restored_mm3") or 0) > 0.01, step.get("frozen_restored_mm3")      # the undo ran: the test is not empty
    added, removed = frozen_change(run, Path(rep["out_dir"]) / "edited.stl")
    assert added + removed < am.FROZEN_TOL_MM3, (added, removed)
    fe = rep["totals"]["frozen_exact"]
    assert fe["unchanged"] and fe["added_in_frozen_mm3"] + fe["removed_from_frozen_mm3"] < am.FROZEN_TOL_MM3
    undo = step["seam"]["frozen_undo"]
    assert undo["method"] in ("sewn", "boolean")
    if undo["method"] == "sewn":
        assert undo["region_residual_mm3"] < am.FROZEN_TOL_MM3                       # the sewn undo was measured


def test_the_end_of_call_undo_measures_again_before_it_says_undone(monkeypatch):
    """`_frozen_exact` sets `restored` only from a new measurement. An undo that returns a part with a changed region is a failure
    (the steps are rolled back) and the note gives the measured volume."""
    old = rod_into_tip()
    new = old - M.sphere(3.0, 96).translate((21.0, 0.0, 6.3))
    tip = cube((21, -2, 0), (24, 2, 4))
    st = type("St", (), {"frozen_solid": tip, "frozen_facts": {"done": True}, "entries": [], "bodies": 1, "pitch": 0.35})()
    facts, restored, failed = am._frozen_exact(st, old, new)
    assert facts["restored"] is True and not failed and restored is not None
    assert facts["added_after_undo_mm3"] + facts["removed_after_undo_mm3"] < am.FROZEN_TOL_MM3
    # an undo that does not undo: the real swap is replaced by one that returns the edited part as it is
    monkeypatch.setattr(am, "_swap_inside", lambda base, patch, region, pitch=None, exact_inside=False:
                        (am._mesh_of(base), base, {"method": "boolean"}, None))
    facts, restored, failed = am._frozen_exact(st, old, new)
    assert facts["restored"] is False and failed and restored is None
    assert "left a change of" in facts["note"] and " mm3" in facts["note"]


def test_the_surface_module_of_verify_is_imported_without_gmsh(tmp_path):
    """The core imports `topopt.tet.surface` for the seam and the report. It must not load gmsh (GPL-2.0-or-later, an optional
    extra). Checked in a fresh Python."""
    tri = trimesh.creation.box((2, 2, 2))
    tri.export(str(tmp_path / "box.stl"))
    code = ("import sys, json\nimport topopt.apply_marks as am\n"
            f"q = am._surface_quality({str(tmp_path / 'box.stl')!r})\n"
            "print(json.dumps({'gmsh': 'gmsh' in sys.modules, 'closed': q['closed_manifold'], 'limit': am._tet_surface_limit()}))\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True, cwd=str(REPO))
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["gmsh"] is False and got["closed"] is True and got["limit"] == 0.05


@needs_gmsh
@pytest.mark.parametrize("case", list(CASES))
def test_the_edited_part_meshes_with_gmsh(cases, case):
    """The ladle-like cases (a bowl and a handle): the mesher of `topopt verify` takes the edited.stl."""
    from topopt.tet import mesh_stl
    rep, _run = cases(case)
    mesh = mesh_stl(str(Path(rep["out_dir"]) / "edited.stl"), SIZE)
    assert len(mesh.tets) > 100


# ---- the recorded hook: the candidates of the D3 research ---------------------------------------------------
def hook_marks(final_stl):
    """Three marks like the D3 research: smooth r 4 mm at the vertex farthest from the regions, remove r 3 mm at a vertex in the
    top 2 percent of that distance, remove r 6 mm at the vertex nearest the mass centre."""
    from scipy.spatial import cKDTree
    mesh = trimesh.load(str(final_stl), force="mesh")
    regions = [trimesh.load(str(p), force="mesh") for p in sorted(HOOK_DIR.glob("*.stl")) if p.name != "design.stl"]
    dist = cKDTree(np.vstack([r.vertices for r in regions])).query(mesh.vertices)[0]
    order = np.argsort(-dist)
    centre = int(np.argmin(np.linalg.norm(mesh.vertices - mesh.center_mass, axis=1)))

    def mk(typ, v, radius):
        return mark(1, typ, [mesh.vertices[v].tolist()], radius=radius)       # smooth: strength 1.5; remove: amount 1 mm
    return {"smooth_far": mk("smooth", int(order[0]), 4.0), "remove_far": mk("remove", int(order[len(order) // 50]), 3.0),
            "remove_core": mk("remove", centre, 6.0)}


@pytest.mark.slow                                    # 25 to 30 s each: apply_marks, then the meshing and the FE run of verify
@needs_gmsh
@needs_skfem
@needs_recording
@pytest.mark.parametrize("case", ["smooth_far", "remove_far", "remove_core"])
def test_verify_meshes_the_edited_hook_at_the_default_size(tmp_path, case):
    """RESEARCH.md 5.2 on the recorded hook: `apply_marks` writes edited.stl, a shadow run folder holds it as final.stl, and
    `topopt verify` exits 0 at its default mesh size (before the fix: exit 2, "non-manifold edges", "an edge of 0.0002 mm")."""
    from topopt.cli import main
    mk = hook_marks(RECORDING / "result" / "final.stl")[case]
    rep = am.apply_marks(RECORDING, doc_for(RECORDING, mk), out_dir=tmp_path / "out",       # reads the recording, writes elsewhere
                         yaml_path=HOOK_DIR / "hook.yaml")                                  # run.json names a YAML of the author's machine
    e = rep["marks"][0]
    assert e["status"] == "applied", (e["status"], e.get("reason"))
    assert rep["totals"]["frozen_exact"]["unchanged"], rep["totals"]["frozen_exact"]
    assert rep["totals"]["surface_quality"]["after"]["closed_manifold"] and not rep["problems"]
    shadow = tmp_path / "shadow"
    (shadow / "result").mkdir(parents=True)
    shutil.copy(RECORDING / "run.json", shadow / "run.json")
    shutil.copy(RECORDING / "result" / "metrics.json", shadow / "result" / "metrics.json")
    shutil.copy(tmp_path / "out" / "edited.stl", shadow / "result" / "final.stl")
    assert main(["verify", str(shadow), "--yaml", str(HOOK_DIR / "hook.yaml"), "--no-sensitivity"]) == 0
