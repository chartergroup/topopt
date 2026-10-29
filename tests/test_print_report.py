"""Tests for the minimal print report (CONTRACTS.md section 18): the `print` block, overhang and
support numbers, bed fit, the thinnest-member estimate, and the hooks in `check`, `post` and the tools.

The "comb" fixture (`tests/fixtures/comb.stl`) is the union of three boxes with whole-number sizes
and no face at exactly 45 degrees (x, y, z in mm; x0 x1 y0 y1 z0 z1):
wall (0 4 0 10 0 10), lower arm (0 20 0 10 0 2), upper arm (0 20 0 10 6 8). Volume 1040 mm3, area 1128 mm2.
The hand calculations are in the comments of the tests.
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from topopt import print_report as pr
from topopt.config import PRINTER_TYPES, PrintSettings, UserError, load_config

from test_input import box, make_case

ROOT = Path(__file__).resolve().parent.parent
COMB = ROOT / "tests" / "fixtures" / "comb.stl"


@pytest.fixture(scope="module")
def comb():
    return trimesh.load(COMB, force="mesh")


def by_label(report):
    return {o["label"]: o for o in report["options"]}


# --- the fixture itself -------------------------------------------------------


def test_comb_fixture_is_the_union_of_three_boxes(comb):
    parts = []
    for (x0, x1, y0, y1, z0, z1) in [(0, 4, 0, 10, 0, 10), (0, 20, 0, 10, 0, 2), (0, 20, 0, 10, 6, 8)]:
        b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
        b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        parts.append(b)
    union = trimesh.boolean.union(parts, engine="manifold")
    assert comb.is_watertight
    assert comb.volume == pytest.approx(1040.0) and union.volume == pytest.approx(1040.0)
    assert comb.area == pytest.approx(1128.0) and union.area == pytest.approx(1128.0)
    assert np.allclose(comb.bounds, [[0, 0, 0], [20, 10, 10]])


# --- equivalence with the pipeline numbers -----------------------------------


@pytest.mark.equivalence
def test_comb_matches_the_pipeline_measure(comb):
    """Build direction +z, 45 degrees, plate at z = 0.

    Overhang: the underside of the upper arm, x 4..20, y 0..10, so 16 x 10 = 160 mm2 (n.b = -1, strict test).
    Plate contact: the bottom of the lower arm and the wall, x 0..20 = 20 x 10 = 200 mm2.
    Support (gap to plate or part): the arm underside is at z = 6, the lower arm top at z = 2,
    so the gap is 4 mm and the volume is 160 x 4 = 640 mm3. The pipeline numbers: overhang 160,
    contact 200 (Study.metrics), support 640 mm3 (printability.py, gap definition).
    """
    r = pr.build_report(comb, PrintSettings())
    c = r["configured"]
    assert c["label"] == "+z"
    assert c["overhang_area_mm2"] == pytest.approx(160.0)
    assert c["plate_contact_area_mm2"] == pytest.approx(200.0)
    assert c["support_volume_mm3"] == pytest.approx(640.0)
    assert c["overhang_fraction"] == pytest.approx(160.0 / 1128.0, abs=1e-5)
    assert r["total_area_mm2"] == pytest.approx(1128.0)
    assert c["extents_mm"] == [20.0, 10.0, 10.0] and c["height_mm"] == 10.0
    assert c["bed_fit"] is True


def test_comb_all_six_axis_options(comb):
    """Hand calculation for each axis direction (the part is 20 x 10 x 10 mm).

    +x up (plate at x = 0): the whole x = 0 face is the plate, 10 x 10 = 100. No overhang.
    -x up (plate at x = 20): the arm ends (2 x 10 x 2 = 40) touch the plate. The wall face x = 4
      shows to +x, so it is an overhang: z 2..6 is 4 x 10 = 40 and z 8..10 is 2 x 10 = 20, total 60.
      Both are 16 mm above the plate with nothing in the way: 40 x 16 + 20 x 16 = 960.
    +y or -y up: the y face is the plate (wall 4 x 10 + two arms 16 x 2 each = 104). No overhang.
    -z up (plate at z = 10): the wall top is the plate, 4 x 10 = 40. The top of the upper arm
      (160 mm2) is 2 mm above the plate, gap 2 mm: 320. The top of the lower arm (160 mm2) is 8 mm
      above the plate and the upper arm is in the way at 4 mm: 160 x 4 = 640. Overhang 320, support 960.
    """
    r = pr.build_report(comb, PrintSettings())
    o = by_label(r)
    assert list(o) == ["+x", "-x", "+y", "-y", "+z", "-z"]
    want = {  # overhang, contact, support
        "+x": (0.0, 100.0, 0.0), "-x": (60.0, 40.0, 960.0),
        "+y": (0.0, 104.0, 0.0), "-y": (0.0, 104.0, 0.0),
        "+z": (160.0, 200.0, 640.0), "-z": (320.0, 40.0, 960.0),
    }
    for label, (ov, contact, vol) in want.items():
        assert o[label]["overhang_area_mm2"] == pytest.approx(ov), label
        assert o[label]["plate_contact_area_mm2"] == pytest.approx(contact), label
        assert o[label]["support_volume_mm3"] == pytest.approx(vol), label
        assert o[label]["bed_fit"] is True
    # extents in the build frame: the two other axes in the order x, y, z, then the height
    assert o["+x"]["extents_mm"] == [10.0, 10.0, 20.0]
    assert o["+y"]["extents_mm"] == [20.0, 10.0, 10.0]
    assert o["-z"]["extents_mm"] == [20.0, 10.0, 10.0]


def test_build_direction_changes_the_configured_numbers(comb):
    """The configured entry follows `print.build_direction`, and equals the option of the same axis."""
    r = pr.build_report(comb, PrintSettings(build_direction=(0.0, 0.0, -1.0), given=("build_direction",)))
    c = r["configured"]
    assert c["label"] == "-z"
    assert (c["overhang_area_mm2"], c["plate_contact_area_mm2"], c["support_volume_mm3"]) == \
        pytest.approx((320.0, 40.0, 960.0))
    assert c == by_label(r)["-z"]
    assert r["settings"]["build_direction_label"] == "-z"
    assert "build_direction" not in r["settings"]["assumed"]


def test_tilted_direction_box():
    """A 10 mm cube, build direction 30 degrees from z toward x: b = (0.5, 0, cos 30).

    Faces: -z has n.b = -cos 30 = -0.866 < -0.7071, so it is an overhang (100 mm2). -x has n.b = -0.5: no.
    The lowest vertex is the origin, so the -z face centre (5, 5, 0) is 5 x 0.5 = 2.5 mm above the plate.
    Nothing is under it, so the gap is 2.5 and the volume is 2.5 x 0.866 x 100 = 216.5 mm3. No face rests on the plate.
    The height along b is 10 (sin 30 + cos 30) = 13.66 mm; the in-plane extents are 13.66 and 10.
    """
    cube = trimesh.creation.box(extents=[10, 10, 10])
    cube.apply_translation([5, 5, 5])
    b = (math.sin(math.radians(30)), 0.0, math.cos(math.radians(30)))
    r = pr.build_report(cube, PrintSettings(build_direction=b, given=("build_direction",)))
    c = r["configured"]
    assert c["overhang_area_mm2"] == pytest.approx(100.0)
    assert c["plate_contact_area_mm2"] == 0.0
    assert c["support_volume_mm3"] == pytest.approx(2.5 * math.cos(math.radians(30)) * 100.0, abs=0.1)
    assert c["height_mm"] == pytest.approx(10 * (0.5 + math.cos(math.radians(30))), abs=1e-3)
    assert sorted(c["extents_mm"][:2]) == pytest.approx([10.0, 10 * (0.5 + math.cos(math.radians(30)))], abs=1e-3)
    assert c["bed_fit"] is True   # the bed is 220 mm


def test_overhang_test_is_strict(comb):
    """A flat underside has n.b = -1 exactly. With the angle 0 the limit is -cos(0) = -1: not strictly below."""
    zero = pr.build_report(comb, PrintSettings(self_supporting_angle_deg=0.0, given=("self_supporting_angle_deg",)))
    assert zero["configured"]["overhang_area_mm2"] == 0.0
    small = pr.build_report(comb, PrintSettings(self_supporting_angle_deg=0.001))
    assert small["configured"]["overhang_area_mm2"] == pytest.approx(160.0)
    # The plate contact does not depend on the angle.
    assert zero["configured"]["plate_contact_area_mm2"] == pytest.approx(200.0)


def wedge():
    """Profile in x-z: (0,0) (10,0) (16,10) (-15,10), extruded 10 mm in y (the pipeline test mesh).

    Left face: 10 x sqrt(15^2 + 10^2) = 180.28 mm2, 33.7 degrees from the plate.
    Right face: 10 x sqrt(6^2 + 10^2) = 116.62 mm2, 59.0 degrees from the plate.
    Bottom: 10 x 10 = 100 mm2 on the plate.
    """
    pts = []
    for y in (0.0, 10.0):
        pts += [[0, y, 0], [10, y, 0], [16, y, 10], [-15, y, 10]]
    return trimesh.convex.convex_hull(np.array(pts, float))


def test_explicit_angle_wins_over_printer_default():
    m = wedge()
    default = pr.build_report(m, PrintSettings(printer_type="fdm"))
    assert default["settings"]["self_supporting_angle_deg"] == 45.0
    assert default["settings"]["angle_source"].startswith("default of printer_type")
    assert default["configured"]["overhang_area_mm2"] == pytest.approx(180.2776, abs=1e-3)
    assert default["configured"]["plate_contact_area_mm2"] == pytest.approx(100.0)
    # the pipeline value of the support volume of this mesh: 180.2776 x 0.83205 x 5 = 750 mm3
    assert default["configured"]["support_volume_mm3"] == pytest.approx(750.0, abs=0.5)
    # an explicit 60 degrees also counts the 59 degree face; the explicit value wins over the printer type
    explicit = pr.build_report(m, PrintSettings(printer_type="sls", self_supporting_angle_deg=60.0,
                                                given=("printer_type", "self_supporting_angle_deg")))
    assert explicit["settings"]["self_supporting_angle_deg"] == 60.0
    assert explicit["settings"]["angle_source"] == "yaml"
    assert explicit["configured"]["overhang_area_mm2"] == pytest.approx(180.2776 + 116.6190, abs=1e-3)
    below = pr.build_report(m, PrintSettings(self_supporting_angle_deg=58.0))
    assert below["configured"]["overhang_area_mm2"] == pytest.approx(180.2776, abs=1e-3)


def test_sls_needs_no_support_but_gives_the_overhang_area(comb):
    r = pr.build_report(comb, PrintSettings(printer_type="sls", given=("printer_type",)))
    assert r["support_needed"] is False
    assert "no support needed" in r["support_note"]
    assert r["configured"]["overhang_area_mm2"] == pytest.approx(160.0)
    assert r["configured"]["support_volume_mm3"] is None and r["configured"]["support_rays"] == 0
    assert all(o["support_volume_mm3"] is None for o in r["options"])
    fdm = pr.build_report(comb, PrintSettings())
    assert fdm["support_needed"] is True and "support_note" not in fdm
    # the same overhang and contact areas as fdm (same default angle), for all 6 directions
    for a, b in zip(r["options"], fdm["options"]):
        assert (a["overhang_area_mm2"], a["plate_contact_area_mm2"]) == (b["overhang_area_mm2"], b["plate_contact_area_mm2"])


def test_inside_out_mesh_is_flipped(comb):
    flipped = comb.copy()
    flipped.invert()
    r = pr.build_report(flipped, PrintSettings())
    assert r["configured"]["overhang_area_mm2"] == pytest.approx(160.0)
    assert "negative volume" in r["warning"]


def test_support_rays_are_a_fixed_sample(monkeypatch, comb):
    """Above the cap the report samples overhang faces (fixed seed): the result repeats."""
    monkeypatch.setattr(pr, "MAX_SUPPORT_RAYS", 1)
    a = pr.build_report(comb, PrintSettings())["configured"]
    b = pr.build_report(comb, PrintSettings())["configured"]
    assert a == b and a["support_rays"] == 1
    # one of the two triangles of the arm underside has the same gap as the whole face
    assert a["support_volume_mm3"] == pytest.approx(640.0)


# --- bed fit ------------------------------------------------------------------


def test_fits_bed_sorts_both_sides():
    assert pr.fits_bed([20, 10], (10, 20)) and pr.fits_bed([10, 20], (20, 10))
    assert not pr.fits_bed([20, 10], (15, 15))
    assert pr.fits_bed([15, 15], (15, 15))   # equal fits


def test_bed_fit_option_values(comb):
    r = pr.build_report(comb, PrintSettings(bed_mm=(15.0, 15.0), given=("bed_mm",)))
    o = by_label(r)
    # footprint: +x/-x (10 x 10) fits; +y/-y/+z/-z (20 x 10) does not fit a 15 x 15 bed
    assert [o[k]["bed_fit"] for k in ("+x", "-x", "+y", "-y", "+z", "-z")] == [True, True, False, False, False, False]
    assert r["configured"]["bed_fit"] is False
    assert "bed_mm" not in r["settings"]["assumed"]
    narrow = pr.build_report(comb, PrintSettings(bed_mm=(25.0, 12.0), given=("bed_mm",)))
    assert narrow["configured"]["bed_fit"] is True   # 20 x 10 in 25 x 12; a 90 degree turn is allowed


# --- thinnest member ----------------------------------------------------------

# The bias was measured on these meshes with 30,000 samples: a box
# gives the exact value, a 64-sided rod gives -0.12 percent (the polygon is smaller than the circle).
# The tolerance is 1 percent.
THICK_TOL = 0.01


def thin(mesh, **kw):
    return pr.thickness_estimate(pr._Geometry(mesh), **kw)


def test_plate_thickness():
    for size in ([20, 20, 4], [60, 60, 4], [20, 20, 1.2]):
        t = thin(trimesh.creation.box(extents=size))
        assert t["thinnest_member_mm"] == pytest.approx(min(size), rel=THICK_TOL), size
        assert t["min_mm"] == pytest.approx(min(size), rel=THICK_TOL)


def test_plate_thickness_does_not_depend_on_the_turn():
    """The chord follows the surface normal, so a turned plate gives the same value (no voxel bias)."""
    m = trimesh.creation.box(extents=[20, 20, 4])
    m.apply_transform(trimesh.transformations.rotation_matrix(0.6, [1, 2, 3]))
    assert thin(m)["thinnest_member_mm"] == pytest.approx(4.0, rel=THICK_TOL)
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        n = trimesh.creation.box(extents=[20, 20, 4])
        n.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, axis))
        assert thin(n)["thinnest_member_mm"] == pytest.approx(4.0, rel=THICK_TOL)


def test_rod_thickness():
    for d in (1.0, 3.0, 6.0):
        rod = trimesh.creation.cylinder(radius=d / 2, height=30, sections=64)
        t = thin(rod)
        assert t["thinnest_member_mm"] == pytest.approx(d, rel=THICK_TOL), d
        assert t["thinnest_member_mm"] <= d   # a polygon rod is never thicker than the circle


def test_comb_thinnest_member_is_the_arm(comb):
    t = pr.build_report(comb, PrintSettings())["thinnest_member"]
    assert t["thinnest_member_mm"] == pytest.approx(2.0, rel=THICK_TOL)
    assert t["estimate"] is True and "chord" in t["method"] and t["percentile"] == 1.0
    assert t["samples"] == pr.THICKNESS_SAMPLES and t["seed"] == pr.THICKNESS_SEED
    assert set(t["percentiles_mm"]) == {"p0.1", "p1", "p5", "p50"}


def test_thickness_is_repeatable(comb):
    assert thin(comb, samples=2000) == thin(comb, samples=2000)


def test_chords_agree_with_the_trimesh_ray_engine(comb):
    """The short-piece search finds the same first wall as a plain ray cast with all hits."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=6.0)
    g = pr._Geometry(m)
    pts, fi = trimesh.sample.sample_surface(m, 300, seed=3)
    n = g.N[fi]
    dist, near = pr._first_hits(g, pts, -n, 12.0)
    loc, ray, _ = m.ray.intersects_location(pts - 0.02 * n, -n, multiple_hits=True)
    for i in range(len(pts)):
        d = np.linalg.norm(loc[ray == i] - pts[i], axis=1)
        d = d[(d > 0.03) & (d <= 12.0)]
        assert (dist[i] == pytest.approx(d.min(), abs=1e-6)) if len(d) else not np.isfinite(dist[i])


def test_chord_search_without_the_vectorised_box_query():
    """An rtree without `intersection_v` (an old libspatialindex) gives the same chords through the loop."""
    import types

    m = trimesh.creation.icosphere(subdivisions=3, radius=6.0)
    g = pr._Geometry(m)
    pts, fi = trimesh.sample.sample_surface(m, 200, seed=4)
    want = pr._first_hits(g, pts, -g.N[fi], 12.0)
    tree = m.triangles_tree

    class Plain:   # only `intersection`
        def intersection(self, box):
            return tree.intersection(box)

    g.mesh = types.SimpleNamespace(triangles_tree=Plain())
    got = pr._first_hits(g, pts, -g.N[fi], 12.0)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_a_wall_thicker_than_the_cap_reads_the_cap():
    t = thin(trimesh.creation.box(extents=[40, 40, 40]))
    assert t["thinnest_member_mm"] == pytest.approx(pr.THICKNESS_MAX_MM)


# --- ray batches: bounded memory, same numbers ------------------------------------


def test_chord_batches_do_not_change_the_chords():
    m = trimesh.creation.icosphere(subdivisions=3, radius=6.0)
    g = pr._Geometry(m)
    pts, fi = trimesh.sample.sample_surface(m, 500, seed=5)
    want = pr._first_hits(g, pts, -g.N[fi], 12.0, chunk=10 ** 6)
    for chunk in (1, 7, 499, 500, 501):
        got = pr._first_hits(g, pts, -g.N[fi], 12.0, chunk=chunk)
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1]), chunk


def test_thinnest_member_does_not_depend_on_the_batch_size(monkeypatch, comb):
    want = pr.thickness_estimate(pr._Geometry(comb))
    monkeypatch.setattr(pr, "RAY_CHUNK", 333)
    assert pr.thickness_estimate(pr._Geometry(comb)) == want


@pytest.mark.parametrize("direction", [(0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 2.0, 3.0)])
def test_support_batches_are_small_and_do_not_change_the_volume(monkeypatch, direction):
    """No ray query gets more than RAY_CHUNK rays, and the volume is the same for any batch size."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=6.0)
    b = np.array(direction) / np.linalg.norm(direction)
    want = pr.measure_direction(pr._Geometry(m), b, 45.0, (220.0, 220.0))
    assert want["support_rays"] > 100
    sizes = []
    real = trimesh.ray.ray_triangle.RayMeshIntersector.intersects_location

    def spy(self, origins, directions, multiple_hits=True):
        sizes.append(len(origins))
        return real(self, origins, directions, multiple_hits=multiple_hits)

    monkeypatch.setattr(trimesh.ray.ray_triangle.RayMeshIntersector, "intersects_location", spy)
    monkeypatch.setattr(pr, "RAY_CHUNK", 64)
    got = pr.measure_direction(pr._Geometry(m), b, 45.0, (220.0, 220.0))
    assert got == want
    assert len(sizes) > 1 and max(sizes) <= 64 and sum(sizes) == want["support_rays"]


# --- review round 1 (REVIEW.md): thin features, pairs per pass, turn, text ----------------


def rect_box(lo, hi):
    m = trimesh.creation.box(extents=np.subtract(hi, lo))
    m.apply_translation(np.add(lo, hi) / 2)
    return m


def block_with_rib(rib_l, rib_h):
    """A 40 x 40 x 12 mm block with a 1.0 mm thick rib on top (the rib has two side faces of l x h)."""
    rib = rect_box([10, 19.5, 12], [10 + rib_l, 20.5, 12 + rib_h])
    return trimesh.boolean.union([rect_box([0, 0, 0], [40, 40, 12]), rib], engine="manifold")


@pytest.mark.parametrize("rib_l,rib_h", [(30, 20), (10, 6), (6, 4), (4, 3)])
def test_area_below_sees_a_thin_rib_that_the_percentile_misses(rib_l, rib_h):
    """A 1.0 mm rib on a thick block: its side faces are the only surface with a chord below 1.2 mm.

    `area_below_mm2["1.2"]` is the rib side area (2 l h) within the sampling error: the share f of the
    samples has the standard deviation sqrt(f (1 - f) / N); the tolerance is 4 of them. For the small
    ribs the percentile `thinnest_member_mm` stays at the thick value (REVIEW.md M1); `min_mm` and the
    area show the rib.
    """
    m = block_with_rib(rib_l, rib_h)
    t = pr.thickness_estimate(pr._Geometry(m))
    rib_area = 2.0 * rib_l * rib_h
    f = rib_area / m.area
    tol = 4.0 * math.sqrt(f * (1 - f) / t["samples"]) * m.area
    assert t["area_below_mm2"]["1.2"] == pytest.approx(rib_area, abs=tol)
    assert t["area_below_mm2"]["0.8"] == 0.0                      # the rib is 1.0 mm
    assert t["area_below_mm2"]["2.0"] == pytest.approx(t["area_below_mm2"]["1.2"])
    assert t["area_below_pct"]["1.2"] == pytest.approx(100.0 * t["area_below_mm2"]["1.2"] / m.area, rel=1e-3)
    assert t["min_mm"] == pytest.approx(1.0, abs=1e-6)
    if f < 0.005:                                                 # well below the 1 percent of the percentile
        assert t["thinnest_member_mm"] > 5.0


def test_area_below_is_zero_for_a_plain_block():
    t = pr.thickness_estimate(pr._Geometry(rect_box([0, 0, 0], [40, 40, 12])))
    assert t["area_below_mm2"] == {"0.8": 0.0, "1.2": 0.0, "2.0": 0.0}
    assert t["thinnest_member_mm"] == pytest.approx(12.0)


def test_area_below_in_the_report_and_the_definitions(comb):
    r = pr.build_report(comb, PrintSettings())
    t = r["thinnest_member"]
    assert list(t["area_below_mm2"]) == ["0.8", "1.2", "2.0"] and list(t["area_below_pct"]) == ["0.8", "1.2", "2.0"]
    assert t["area_below_mm2"]["2.0"] == 0.0                       # the thinnest wall of the comb is 2.0 mm (not below)
    d = r["definitions"]["thinnest_member"]
    for word in ("percent of the sampled surface area", "area_below_mm2", "sharp edges", "max_chord_mm"):
        assert word in d
    assert "1e-15" in r["definitions"]["overhang"] and "area_below_mm2" in r["definitions"]


def fan_disk(sections):
    """A disk r 50 x 2 mm whose caps are triangle fans: every fan triangle has a box over a big part of the disk."""
    return trimesh.creation.cylinder(radius=50, height=2, sections=sections)


def test_pairs_per_pass_stay_below_the_budget(monkeypatch):
    """REVIEW.md M2: the (ray, triangle) pairs of one pass are bounded, and the chords do not change."""
    m = fan_disk(1500)
    g = pr._Geometry(m)
    pts, fi = trimesh.sample.sample_surface(m, 3000, seed=2)
    free = {}
    want = pr._first_hits(g, pts, -g.N[fi], 12.0, stats=free)       # the default budget: no slicing here
    budget = 20000
    assert free["max_pairs"] > budget                               # the mesh does make big passes
    monkeypatch.setattr(pr, "PAIR_BUDGET", budget)
    stats = {}
    got = pr._first_hits(g, pts, -g.N[fi], 12.0, stats=stats)
    assert stats["max_pairs"] <= budget and stats["passes"] > free["passes"]
    assert stats["max_rays"] <= pr.RAY_CHUNK
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_a_lone_ray_above_the_budget_runs_alone(monkeypatch):
    m = fan_disk(300)
    g = pr._Geometry(m)
    pts, fi = trimesh.sample.sample_surface(m, 200, seed=3)
    want = pr._first_hits(g, pts, -g.N[fi], 12.0)
    monkeypatch.setattr(pr, "PAIR_BUDGET", 1)                       # every ray has more pairs than this
    stats = {}
    got = pr._first_hits(g, pts, -g.N[fi], 12.0, stats=stats)
    assert stats["max_rays"] == 1
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_support_rays_have_a_pair_budget_too(monkeypatch):
    m = trimesh.creation.icosphere(subdivisions=3, radius=6.0)
    b = np.array([0.0, 0.0, 1.0])
    want = pr.measure_direction(pr._Geometry(m), b, 45.0, (220.0, 220.0))
    monkeypatch.setattr(pr, "PAIR_BUDGET", 100)
    stats = {}
    got = pr.measure_direction(pr._Geometry(m), b, 45.0, (220.0, 220.0), stats=stats)
    assert got == want
    assert stats["max_pairs"] <= 100 and stats["passes"] > 3


def test_plate_faces_are_left_out_of_the_support_volume():
    """REVIEW.md Q8 gap 2: a face 0.2 mm above the lowest point is plate contact (it counts for the contact area).

    Its gap is 0.2 mm, so counting it as an overhang would add 100 x 0.2 = 20 mm3 of support.
    """
    m = trimesh.util.concatenate([rect_box([0, 0, 0], [10, 10, 5]), rect_box([20, 0, 0.2], [30, 10, 5.2])])
    c = pr.build_report(m, PrintSettings())["configured"]
    assert c["plate_contact_area_mm2"] == pytest.approx(200.0)
    assert c["overhang_area_mm2"] == 0.0
    assert c["support_volume_mm3"] == 0.0 and c["support_rays"] == 0
    higher = trimesh.util.concatenate([rect_box([0, 0, 0], [10, 10, 5]), rect_box([20, 0, 0.5], [30, 10, 5.5])])
    h = pr.build_report(higher, PrintSettings())["configured"]
    assert h["overhang_area_mm2"] == pytest.approx(100.0)          # 0.5 mm up: no plate face, a real overhang
    assert h["support_volume_mm3"] == pytest.approx(100.0 * 0.5)


def tilted_prism(phi_deg, length=10.0):
    """Profile in x-z: A (0,0), B (L cos, L sin), then 10 mm up; extruded 10 mm in y. Face AB tilts phi from the plate."""
    c, s = math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))
    pts = []
    for y in (0.0, 10.0):
        pts += [[0, y, 0], [length * c, y, length * s], [length * c, y, length * s + 10], [0, y, 10]]
    return trimesh.convex.convex_hull(np.array(pts, float))


def test_faces_just_inside_and_just_outside_the_angle():
    """REVIEW.md m1: a face 0.1 degree below 45 degrees is an overhang, a face 0.1 degree above is not.

    A face exactly at the limit can fall on either side (floating point); this test does not use it.
    """
    inside = pr.build_report(tilted_prism(44.9), PrintSettings())["configured"]
    outside = pr.build_report(tilted_prism(45.1), PrintSettings())["configured"]
    assert inside["overhang_area_mm2"] == pytest.approx(100.0, rel=1e-6)
    assert outside["overhang_area_mm2"] == 0.0


def test_frame_is_right_handed_and_the_extents_keep_their_order():
    """REVIEW.md n2: u x v = b for the six axes; the in-plane extents stay in the order of the other axes."""
    m = rect_box([0, 0, 0], [1, 2, 3])
    want = {"+x": [2, 3, 1], "-x": [2, 3, 1], "+y": [1, 3, 2], "-y": [1, 3, 2], "+z": [1, 2, 3], "-z": [1, 2, 3]}
    for label, vec in pr.AXIS_OPTIONS:
        f = pr.build_frame(vec)
        assert np.allclose(np.cross(f[0], f[1]), f[2]) and np.isclose(np.linalg.det(f), 1.0), label
        assert pr.extents_in_frame(m.vertices, vec).tolist() == pytest.approx(want[label]), label
    for vec in ([1, 2, 3], [0.3, 0, 1], [-1, 1, -1]):
        f = pr.build_frame(vec)
        assert np.isclose(np.linalg.det(f), 1.0) and np.allclose(f @ f.T, np.eye(3))


def bar():
    return rect_box([0, 0, 0], [230, 10, 10])


def bar_footprint(deg, length=230.0, width=10.0):
    c, s = abs(math.cos(math.radians(deg))), abs(math.sin(math.radians(deg)))
    return length * c + width * s, length * s + width * c


def bar_margin(deg, bed=220.0, angle0=0.0, length=230.0):
    """Margin of a `length` x 10 bar on a square bed at a turn, from the formula (not from the code under test)."""
    return bed - max(bar_footprint(angle0 - deg, length=length))


def test_a_long_bar_fits_the_bed_turned_with_the_largest_margin():
    """REVIEW.md m2 and R1b: 230 x 10 x 10 mm on a 220 mm bed. The turn is the one with the largest margin: 45 degrees."""
    want = max(range(90), key=lambda d: (bar_margin(d), -d))
    assert want == 45
    r = pr.build_report(bar(), PrintSettings(bed_mm=(220.0, 220.0)))
    c = r["configured"]
    assert c["extents_mm"] == [230.0, 10.0, 10.0]                    # unchanged
    assert c["bed_fit"] is True and c["bed_turn_deg"] == 45
    assert c["bed_footprint_mm"] == pytest.approx(list(bar_footprint(45)), abs=1e-3)
    assert c["bed_margin_mm"] == pytest.approx(220.0 - 240.0 / math.sqrt(2.0), abs=1e-3)      # 50.294
    # the smallest turn that fits (20 degrees) leaves 0.5 mm: it is not the one that is reported
    assert bar_margin(20) < 1.0 < c["bed_margin_mm"]
    o = by_label(r)
    assert o["+x"]["bed_fit"] and o["+x"]["bed_turn_deg"] == 0       # standing on its end: 10 x 10 is best as it is
    assert o["+x"]["bed_margin_mm"] == pytest.approx(210.0)
    assert o["+y"]["bed_turn_deg"] == 45 and o["+y"]["bed_margin_mm"] == c["bed_margin_mm"]   # the bar lies along x
    # a part that fits as it is and has the largest margin as it is: turn 0 and the axis footprint
    ok = pr.build_report(bar(), PrintSettings(bed_mm=(300.0, 40.0)))["configured"]
    assert ok["bed_fit"] and ok["bed_turn_deg"] == 0 and ok["bed_footprint_mm"] == [230.0, 10.0]
    assert ok["bed_margin_mm"] == pytest.approx(30.0)                # min(300 - 230, 40 - 10)
    # a part that fits as it is but gains at least 10 mm of margin turned (60 mm): the turn is reported
    roomy = pr.build_report(bar(), PrintSettings(bed_mm=(300.0, 300.0)))["configured"]
    assert roomy["bed_fit"] and roomy["bed_turn_deg"] == 45 and roomy["bed_margin_mm"] == pytest.approx(300.0 - 240.0 / math.sqrt(2.0), abs=1e-3)
    # a compact part: turn 0 is best
    cube = pr.build_report(rect_box([0, 0, 0], [100, 100, 20]), PrintSettings(bed_mm=(220.0, 220.0)))["configured"]
    assert cube["bed_turn_deg"] == 0 and cube["bed_margin_mm"] == pytest.approx(120.0)


def test_no_turn_fits_gives_the_best_turn_and_a_negative_margin():
    """R1b: the bed is shorter than the footprint at every turn (the best, 45 degrees, needs 169.7 mm)."""
    no = pr.build_report(bar(), PrintSettings(bed_mm=(100.0, 100.0)))["configured"]
    assert no["bed_fit"] is False and no["bed_turn_deg"] == 45
    assert no["bed_margin_mm"] == pytest.approx(100.0 - 240.0 / math.sqrt(2.0), abs=1e-3)      # -69.706
    assert no["bed_footprint_mm"] == pytest.approx(list(bar_footprint(45)), abs=1e-3)
    assert pr.fit_phrase(no) == "does not fit at any turn (best: 45 degrees, 69.71 mm over)"
    # a bed that is too narrow on one side only: the margin is the worse of the two spare lengths
    narrow = pr.build_report(bar(), PrintSettings(bed_mm=(300.0, 5.0)))["configured"]
    assert narrow["bed_fit"] is False and narrow["bed_margin_mm"] < 0
    # a margin of zero fits (equal fits)
    exact = pr.build_report(rect_box([0, 0, 0], [20, 10, 5]), PrintSettings(bed_mm=(10.0, 20.0)))["configured"]
    assert exact["bed_fit"] is True and exact["bed_margin_mm"] == 0.0 and exact["bed_turn_deg"] == 0


def test_a_turn_is_reported_only_if_it_gains_enough_margin_when_turn_0_fits(monkeypatch):
    """R1c: when turn 0 fits, keep turn 0 unless a turn gains at least BED_TURN_MIN_GAIN_MM of margin over turn 0."""
    assert pr.BED_TURN_MIN_GAIN_MM == 10.0
    settings = PrintSettings(bed_mm=(220.0, 220.0))
    low = 55.0           # a 55 x 10 bar: 45 degrees gains 55 - 65 / sqrt 2 = 9.04 mm, below 10 mm
    high = 62.0          # a 62 x 10 bar: 45 degrees gains 11.1 mm
    gain_low = bar_margin(45, length=low) - bar_margin(0, length=low)
    gain_high = bar_margin(45, length=high) - bar_margin(0, length=high)
    assert 9.0 < gain_low < 10.0 < gain_high < 11.5
    assert max(range(90), key=lambda d: (round(bar_margin(d, length=low), 6), -d)) == 45          # 45 is the best turn
    c = pr.build_report(rect_box([0, 0, 0], [low, 10, 10]), settings)["configured"]
    assert c["bed_fit"] is True and c["bed_turn_deg"] == 0
    assert c["bed_footprint_mm"] == [low, 10.0] and c["bed_margin_mm"] == pytest.approx(220.0 - low)
    assert pr.fit_phrase(c) == "fits"
    c = pr.build_report(rect_box([0, 0, 0], [high, 10, 10]), settings)["configured"]
    assert c["bed_fit"] is True and c["bed_turn_deg"] == 45
    assert c["bed_margin_mm"] == pytest.approx(220.0 - (high + 10.0) / math.sqrt(2.0), abs=1e-3)
    assert pr.fit_phrase(c) == "fits turned 45 degrees (margin 169.1 mm)"
    # the constant is the knob: with 0 the 55 mm bar is reported turned
    monkeypatch.setattr(pr, "BED_TURN_MIN_GAIN_MM", 0.0)
    c = pr.build_report(rect_box([0, 0, 0], [low, 10, 10]), settings)["configured"]
    assert c["bed_turn_deg"] == 45
    # the rule is in the texts
    assert "at least 10 mm" in pr.DEFINITIONS["bed_fit"]


def test_the_hook_footprint_keeps_turn_0():
    """R1c: a hook-sized footprint (76.1 x 35.2 mm) on 220 x 220 mm: 45 degrees gains 4.3 mm, so the entry says turn 0."""
    c = pr.build_report(rect_box([0, 0, 0], [76.142, 35.206, 66.458]), PrintSettings())["configured"]
    assert c["bed_turn_deg"] == 0 and c["bed_margin_mm"] == pytest.approx(220.0 - 76.142, abs=1e-3)      # 143.858
    assert pr.fit_phrase(c) == "fits"


def test_a_turned_bar_fits_the_same_way():
    """The footprint is searched in the build frame: a bar that is turned in the world has the same margin."""
    angle = 0.7
    m = bar()
    m.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
    c = pr.build_report(m, PrintSettings(bed_mm=(220.0, 220.0)))["configured"]
    deg0 = math.degrees(angle)
    want = max(range(90), key=lambda d: (round(bar_margin(d, angle0=deg0), 6), -d))     # brute force, from the formula
    assert bar_margin(want, angle0=deg0) - bar_margin(0, angle0=deg0) >= pr.BED_TURN_MIN_GAIN_MM      # a gain of 12 mm: the turn is reported
    assert c["bed_fit"] is True and c["bed_turn_deg"] == want
    assert c["bed_margin_mm"] == pytest.approx(bar_margin(want, angle0=deg0), abs=1e-3)
    assert c["bed_margin_mm"] > 49.0


def corners(x, y, z):
    return np.array([[i, j, k] for i in (0, x) for j in (0, y) for k in (0, z)], float)


def test_check_lines_say_fits_turned_and_use_one_number_format():
    facts = pr.check_facts(corners(230, 10, 10), PrintSettings(bed_mm=(220.0, 220.0), given=("bed_mm",)))
    lines = pr.check_lines(facts)
    assert "it fits turned 45 degrees (margin 50.29 mm); footprint 169.7 x 169.7 mm, height 10 mm" in lines[1], lines[1]
    none = pr.check_lines(pr.check_facts(corners(230, 10, 10), PrintSettings(bed_mm=(100.0, 100.0), given=("bed_mm",))))
    assert "it does not fit at any turn (best: 45 degrees, 69.71 mm over); footprint 169.7 x 169.7 mm" in none[1], none[1]
    tilt = pr.check_facts(corners(71.2763, 102.8791, 9.87654321), PrintSettings(build_direction=(1.0, 2.0, 3.0)))
    text = " ".join(pr.check_lines(tilt))
    assert re.search(r"footprint \d+\.?\d{0,2} x \d+\.?\d{0,2} mm, height \d+\.?\d{0,2} mm", text), text   # n4: 4 digits


def test_axis_and_tilted_footprints_use_the_same_format():
    axis = pr.check_lines(pr.check_facts(corners(75.6, 34.8, 66), PrintSettings()))[1]
    assert "footprint 75.6 x 34.8 mm" in axis and "height 66 mm" in axis


# --- tools/check_part.py: golden output, one bed rule, one error line ------------------------

CP_DIR = ROOT / "tests" / "fixtures" / "check_part"


def cp_copy(tmp_path, block=None, stl=None):
    """A copy of the fixture folder (a 30 x 20 x 10 mm part); optional `print` block and part mesh."""
    import shutil
    d = tmp_path / "cp"
    shutil.copytree(CP_DIR, d)
    if block is not None:
        doc = yaml.safe_load((d / "problem.yaml").read_text())
        doc["print"] = block
        (d / "problem.yaml").write_text(yaml.safe_dump(doc))
    if stl is not None:
        stl.export(d / "run" / "result" / "final.stl")
    return d


def run_check_part(d, *extra):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "check_part.py"), str(d / "run"), str(d / "problem.yaml"), *extra],
                          capture_output=True, text=True, cwd=d)


def bed_line(r):
    lines = [l for l in r.stdout.splitlines() if "mm bed" in l]
    assert len(lines) == 1, r.stdout + r.stderr
    return lines[0]


def test_check_part_output_without_print_keys_is_byte_equal_to_main(tmp_path):
    """REVIEW.md Q8 gap 7. The golden file is the output of tools/check_part.py at 7555b19 on the same inputs."""
    d = cp_copy(tmp_path)
    r = run_check_part(d)
    assert r.returncode == 0 and r.stderr == ""
    assert r.stdout.encode() == (CP_DIR / "golden_stdout.txt").read_bytes()
    # `--bed` alone keeps the old strict rule (the YAML has no print keys)
    strict = run_check_part(d, "--bed", "20")
    assert strict.returncode == 1 and "[FAIL] fits 20x20 mm bed" in bed_line(strict) and "any 2 axes < bed" in bed_line(strict)


def test_check_part_uses_one_bed_rule_when_the_yaml_has_print_keys(tmp_path):
    """REVIEW.md m4: with print keys the line uses `<=` (as fits_bed): an extent equal to the bed side fits."""
    r = run_check_part(cp_copy(tmp_path, {"bed_mm": 20}))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[PASS] fits 20x20 mm bed" in bed_line(r) and "any 2 axes <= bed" in bed_line(r)


def test_check_part_turned_fit_in_the_line(tmp_path):
    line = bed_line(run_check_part(cp_copy(tmp_path, {"bed_mm": [40, 20], "build_direction": "+z"})))
    assert line.startswith("[PASS] fits 40x20 mm bed") and "build direction +z" in line and "turned" not in line
    d = cp_copy(tmp_path / "bar", {"bed_mm": 220, "build_direction": "+z"}, stl=bar())    # 230 x 10 x 10 mm
    line = bed_line(run_check_part(d))
    assert line.startswith("[PASS] fits 220x220 mm bed") and "fits turned 45 degrees (margin 50.29 mm); footprint 169.706 x 169.706 mm" in line
    line = bed_line(run_check_part(cp_copy(tmp_path / "nofit", {"bed_mm": 100, "build_direction": "+z"}, stl=bar())))
    assert line.startswith("[FAIL] fits 100x100 mm bed") and "does not fit at any turn (best: 45 degrees, 69.71 mm over)" in line


def test_check_part_prints_one_error_line_for_a_wrong_print_block(tmp_path):
    """REVIEW.md m5: the same form and exit code as tools/print_section.py."""
    r = run_check_part(cp_copy(tmp_path, {"bed": 220}))
    assert r.returncode == 1 and r.stdout == ""
    assert r.stderr.startswith("error: unknown key 'bed' in 'print'") and r.stderr.count("\n") == 1
    assert "Traceback" not in r.stderr


# --- the print block in the YAML ----------------------------------------------


def with_print(tmp_path, block, **kw):
    def mutate(d):
        d["print"] = block
    return make_case(tmp_path, h=2.0, mutate=mutate, **kw)


def test_no_print_block_gives_the_defaults(tmp_path):
    cfg = load_config(make_case(tmp_path, h=2.0))
    ps = cfg.print_settings
    assert ps.build_direction == (0.0, 0.0, 1.0) and ps.bed_mm == (220.0, 220.0)
    assert ps.printer_type == "fdm" and ps.angle_deg == 45.0 and ps.given == ()


def test_print_block_values(tmp_path):
    cfg = load_config(with_print(tmp_path, {"build_direction": [0, 3, 4], "bed_mm": [180, 150],
                                            "printer_type": "SLA", "self_supporting_angle_deg": 50}))
    ps = cfg.print_settings
    assert ps.build_direction == pytest.approx((0.0, 0.6, 0.8))
    assert ps.bed_mm == (180.0, 150.0) and ps.printer_type == "sla" and ps.angle_deg == 50.0
    assert ps.given == ("build_direction", "bed_mm", "printer_type", "self_supporting_angle_deg")


@pytest.mark.parametrize("name,vec", [("+x", (1, 0, 0)), ("-x", (-1, 0, 0)), ("y", (0, 1, 0)),
                                      ("-Y", (0, -1, 0)), ("z", (0, 0, 1)), ("-z", (0, 0, -1))])
def test_build_direction_axis_names(tmp_path, name, vec):
    ps = load_config(with_print(tmp_path, {"build_direction": name})).print_settings
    assert ps.build_direction == vec
    assert "-0.0" not in repr(ps.build_direction)


def test_bed_can_be_one_number_and_the_angle_default_follows_the_printer(tmp_path):
    ps = load_config(with_print(tmp_path, {"bed_mm": 150, "printer_type": "sls"})).print_settings
    assert ps.bed_mm == (150.0, 150.0) and ps.angle_deg == PRINTER_TYPES["sls"][0]
    assert ps.support_needed is False
    assert load_config(with_print(tmp_path, None)).print_settings.given == ()   # "print:" with nothing


@pytest.mark.parametrize("block,text", [
    ({"bed": 220}, "unknown key 'bed' in 'print'"),
    ({"build_direction": [0, 0, 0]}, "zero vector"),
    ({"build_direction": [0, 1]}, "print.build_direction"),
    ({"build_direction": "up"}, "print.build_direction"),
    ({"build_direction": "+-x"}, "print.build_direction"),
    ({"build_direction": "--z"}, "print.build_direction"),
    ({"build_direction": "++z"}, "print.build_direction"),
    ({"build_direction": "xy"}, "print.build_direction"),
    ({"bed_mm": 0}, "print.bed_mm"),
    ({"bed_mm": [100]}, "print.bed_mm"),
    ({"bed_mm": "big"}, "print.bed_mm"),
    ({"printer_type": "laser"}, "print.printer_type"),
    ({"self_supporting_angle_deg": 91}, "print.self_supporting_angle_deg"),
    ({"self_supporting_angle_deg": -1}, "print.self_supporting_angle_deg"),
    ({"self_supporting_angle_deg": True}, "print.self_supporting_angle_deg"),
    ("fdm", "'print' must be a mapping"),
])
def test_print_block_errors(tmp_path, block, text):
    with pytest.raises(UserError) as e:
        load_config(with_print(tmp_path, block))
    msg = str(e.value)
    assert text in msg and "\n" not in msg and "(file:" in msg


def test_print_settings_round_trip():
    ps = PrintSettings((0.0, 1.0, 0.0), (180.0, 150.0), "lpbf", 40.0, ("bed_mm",))
    assert PrintSettings.from_dict(json.loads(json.dumps(ps.to_dict()))) == ps
    assert PrintSettings.from_dict(None) == PrintSettings()
    assert PrintSettings.from_dict({"bed_mm": "x", "printer_type": "?"}) == PrintSettings()


# --- the hooks ------------------------------------------------------------------


def run_case(tmp_path, block):
    from topopt.cli import main
    p = make_case(tmp_path, h=2.0, mutate=(lambda d: d.update({"print": block})) if block is not None else None)
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    return out


def test_run_writes_the_print_section(tmp_path):
    out = run_case(tmp_path, {"build_direction": "+x", "bed_mm": 100, "printer_type": "fdm"})
    run = json.loads((out / "run.json").read_text())
    assert run["print"]["build_direction"] == [1.0, 0.0, 0.0] and run["print"]["bed_mm"] == [100.0, 100.0]
    assert run["print"]["given"] == ["build_direction", "bed_mm", "printer_type"]
    m = json.loads((out / "result" / "metrics.json").read_text())
    assert m["print"]["ok"] is True
    assert m["print"]["configured"]["label"] == "+x"
    assert m["print"]["settings"]["assumed"] == []
    assert len(m["print"]["options"]) == 6
    assert m["print"]["thinnest_member"]["estimate"] is True
    assert m["print"]["elapsed_s"] >= 0


def test_run_without_a_print_block_labels_the_defaults_assumed(tmp_path):
    out = run_case(tmp_path, None)
    m = json.loads((out / "result" / "metrics.json").read_text())
    s = m["print"]["settings"]
    assert s["bed_mm"] == [220.0, 220.0] and s["printer_type"] == "fdm" and s["build_direction_label"] == "+z"
    assert s["assumed"] == ["build_direction", "bed_mm", "printer_type"]
    old_keys = ["mass_design_space_g", "mass_final_g", "mass_saved_pct", "volume_final_mm3",
                "compliance_initial", "compliance_final", "stiffness_cases", "run_time_s",
                "iterations", "solver", "constraints", "watertight", "triangles"]
    keys = list(m)
    assert [k for k in keys if k in old_keys] == old_keys     # the keys from before V2 keep their order
    assert keys.index("print") > keys.index("triangles")      # `print` is added after them (other PRs add keys too)


def test_a_failing_report_does_not_stop_the_run(tmp_path, monkeypatch):
    def boom(mesh, settings=None, **kw):
        raise RuntimeError("test fault")
    monkeypatch.setattr(pr, "build_report", boom)
    out = run_case(tmp_path, None)
    m = json.loads((out / "result" / "metrics.json").read_text())
    assert {k: v for k, v in m["print"].items() if k not in ("partial", "coverage")} == {"ok": False, "error": "RuntimeError: test fault"}
    assert m["print"]["partial"] is True      # TC-2: the section says what failed (tests/test_print_coverage.py)
    assert m["watertight"] is True and (out / "result" / "final.stl").is_file()
    assert json.loads((out / "status.json").read_text())["state"] == "done"


def test_post_of_an_old_run_uses_the_defaults(tmp_path):
    """A run.json without the `print` key (a recording) gives the defaults, all assumed."""
    sys.path.insert(0, str(ROOT / "tools"))
    import fake_run
    from topopt import post
    fake_run.run(tmp_path, spi=0.0, n_it=3)
    run = json.loads((tmp_path / "run.json").read_text())
    run.pop("print", None)
    (tmp_path / "run.json").write_text(json.dumps(run))
    m = post.finish(tmp_path)
    assert m["print"]["settings"]["assumed"] == ["build_direction", "bed_mm", "printer_type"]


def test_check_gives_the_bed_fit_as_a_fact(tmp_path, capsys):
    """The cantilever case: the design-space box is 22 x 10 x 10 mm (plate -2..0, design 0..20)."""
    from topopt.cli import check_data, main
    p = with_print(tmp_path, {"bed_mm": 15, "build_direction": "+z"})
    d = check_data(p)["print"]
    assert d["box_extents_mm"] == [22.0, 10.0, 10.0]
    assert d["configured"]["bed_fit"] is False and d["configured"]["label"] == "+z"
    fits = {o["label"]: o["bed_fit"] for o in d["options"]}
    assert fits == {"+x": True, "-x": True, "+y": False, "-y": False, "+z": False, "-z": False}
    assert d["assumed"] == ["printer_type"]
    assert main(["check", str(p)]) == 0     # a misfit is a fact, not a refusal
    out = capsys.readouterr().out
    assert "print bed fit:" in out and "does not fit" in out and "fits with +x -x up" in out
    assert main(["check", str(p), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["print"]["configured"]["bed_fit"] is False


def test_check_without_a_print_block_names_the_defaults(tmp_path, capsys):
    from topopt.cli import main
    assert main(["check", str(make_case(tmp_path, h=2.0))]) == 0
    out = capsys.readouterr().out
    assert "bed 220 x 220 mm" in out and "assumed: build_direction, bed_mm, printer_type" in out
    assert "fits with +x -x +y -y +z -z up" in out


def test_spec_print_block_goes_to_the_yaml(tmp_path):
    from topopt.regions_builder import build_spec
    spec = {"name": "bar", "regions": {"design": [{"box": {"min": [0, 0, 0], "max": [20, 10, 10]}}],
                                       "fixed": {"wall": [{"box": {"min": [-2, 0, 0], "max": [0, 10, 10]}}]},
                                       "load": {"tip": [{"box": {"min": [18, 3, 0], "max": [20, 7, 3]}}]}},
            "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -2.0]}]}],
            "settings": {"voxel_size": 2.0, "iterations": 5},
            "print": {"bed_mm": 120, "printer_type": "sla"}}
    res = build_spec(spec, tmp_path / "out", say=lambda *a, **k: None)
    assert res["ok"]
    doc = yaml.safe_load(Path(res["yaml"]).read_text())
    assert doc["print"] == {"bed_mm": 120, "printer_type": "sla"}
    assert load_config(res["yaml"]).print_settings.printer_type == "sla"
    bad = dict(spec, print={"bed": 1})
    res = build_spec(bad, tmp_path / "out2", say=lambda *a, **k: None)
    assert not res["ok"]


# --- tools/check_part.py ----------------------------------------------------------


def load_check_part():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_part", ROOT / "tools" / "check_part.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_part_reads_the_print_keys(tmp_path):
    cp = load_check_part()
    assert cp.print_settings({"name": "x"}, tmp_path / "a.yaml") is None   # no keys: nothing changes
    ps = cp.print_settings({"print": {"bed_mm": [100, 80], "build_direction": "+x"}}, tmp_path / "a.yaml")
    assert ps.bed_mm == (100.0, 80.0) and ps.given == ("build_direction", "bed_mm")
    with pytest.raises(UserError):
        cp.print_settings({"print": {"bed": 1}}, tmp_path / "a.yaml")


def test_check_part_bed_line(tmp_path):
    """Run the tool on a small finished run. Without keys the line is as before; with keys it uses them."""
    out = run_case(tmp_path, None)
    yml = tmp_path / "c.yaml"
    script = str(ROOT / "tools" / "check_part.py")

    def bed_line(extra=()):
        r = subprocess.run([sys.executable, script, str(out), str(yml), *extra], capture_output=True, text=True, cwd=tmp_path)
        lines = [l for l in r.stdout.splitlines() if "mm bed" in l]
        assert len(lines) == 1, r.stdout + r.stderr
        return lines[0]

    assert re.match(r"\[PASS\] fits 220x220 mm bed: extents=.* \(any 2 axes < bed; flat side down\)", bed_line())
    assert "fits 90x90 mm bed" in bed_line(["--bed", "90"])
    d = yaml.safe_load(yml.read_text())
    d["print"] = {"bed_mm": [30, 30], "build_direction": "+x"}
    yml.write_text(yaml.safe_dump(d))
    line = bed_line()
    assert "fits 30x30 mm bed" in line and "build frame" in line and "build direction +x" in line


# --- the text ------------------------------------------------------------------------


def test_skill_reads_the_bed_from_print():
    text = (ROOT / ".claude" / "skills" / "part-to-topopt" / "SKILL.md").read_text()
    assert "print.bed_mm" in text and "220" in text
    assert "must fit a 220 mm" not in text
    assert "smaller than 220 mm" not in text


def test_contracts_section_18_names_every_key():
    text = (ROOT / "CONTRACTS.md").read_text()
    m = re.search(r"^## 18\..*?(?=^## \d+\.|\Z)", text, re.S | re.M)
    assert m, "CONTRACTS.md has no section 18"
    sec = m.group(0)
    for key in ("build_direction", "bed_mm", "printer_type", "self_supporting_angle_deg", "overhang_area_mm2",
                "plate_contact_area_mm2", "support_volume_mm3", "bed_fit", "extents_mm", "thinnest_member_mm",
                "percentile", "min_mm", "samples", "method", "assumed", "support_needed",
                "area_below_mm2", "area_below_pct", "bed_turn_deg", "bed_footprint_mm", "bed_margin_mm", "PAIR_BUDGET"):
        assert key in sec, key
    for printer in PRINTER_TYPES:
        assert f"`{printer}`" in sec
    assert "n.b < -cos" in sec and "strict" in sec
