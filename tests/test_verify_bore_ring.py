"""`topopt verify` with the bore checks on a part with a real bore (B4a, plan section 10.6): a thick steel ring under an axial force.

The ring is a closed STL (360 sections), fixed at the bottom and pulled at the top. Away from the ends the stress is uniform and uniaxial:
sigma = F / (pi (b^2 - a^2)), so the von Mises stress of the case at the wall of a bore in the middle third is sigma. The ring has a wall of 3.26 mm,
and the run uses the mesh size 3 mm: the P2 stress is not smoothed, so single points scatter around sigma (the maximum is a few percent
above, and the p99.9 is close to the maximum). The scatter is a property of the P2 stress and of the mesh realization, not of the surface and not of the
mesh size alone (the review of B4a measured p99.9 +4.8 %, +18.2 % and +8.3 % at 3, 2 and 1.5 mm on this ring, and +3.0 % at 2 mm on a ring of 120 mm;
CONTRACTS.md section 40.7). The test therefore compares the median of the sample set with sigma, and the p99.9 and the maximum only
with the median and with each other. The load cases are linear: a case with half the force has half the stress.
A real run needs gmsh (`needs_gmsh`) and scikit-fem.
"""
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from helpers import needs_gmsh, needs_skfem, tet_probe  # noqa: E402

pytestmark = [needs_gmsh, needs_skfem]

A, B, L, FORCE = 21.9875, 25.25, 60.0, 1000.0
SIGMA = FORCE / (math.pi * (B * B - A * A))                 # 2.065 MPa
SE = 350.0
SIZE = 3.0


def build_ring(work: Path):
    import manifold3d as m3
    import trimesh
    outer = m3.Manifold.cylinder(L, B, B, 360)
    inner = m3.Manifold.cylinder(L + 2, A, A, 360).translate([0, 0, -1])
    m = (outer - inner).to_mesh()
    ring = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)
    assert ring.is_watertight and ring.volume > 0
    (work / "run" / "result").mkdir(parents=True)
    ring.export(work / "run" / "result" / "final.stl")
    ring.export(work / "design.stl")

    def slab(z0, z1):
        box = trimesh.creation.box(extents=[2 * (B + 5), 2 * (B + 5), z1 - z0])
        box.apply_translation([0, 0, (z0 + z1) / 2])
        return box
    slab(-2.0, 1.0).export(work / "base.stl")
    slab(L - 1.0, L + 2.0).export(work / "top.stl")


def case_yaml(work: Path, bore_checks=None) -> Path:
    d = {"name": "ring", "regions": {"design": str(work / "design.stl"), "fixed": [str(work / "base.stl")], "load": {"top": str(work / "top.stl")}},
         "load_cases": [{"name": "pull", "weight": 1.0, "forces": [{"region": "top", "force": [0.0, 0.0, FORCE]}]},
                        {"name": "half", "weight": 1.0, "forces": [{"region": "top", "force": [0.0, 0.0, FORCE / 2]}]}],
         "material": {"E": 197000.0, "nu": 0.28, "density": 7.8}, "settings": {"voxel_size": SIZE, "volume_fraction": 0.5, "iterations": 1}}
    if bore_checks is not None:
        d["bore_checks"] = bore_checks
    p = work / "ring.yaml"
    p.write_text(yaml.safe_dump(d))
    (work / "run" / "run.json").write_text(json.dumps({"name": "ring", "yaml_path": str(p), "h": SIZE}))
    return p


MID = {"bores": [{"name": "mid", "point": [0.0, 0.0, L / 3], "axis": [0.0, 0.0, 1.0], "radius_mm": A, "length_mm": L / 3}],
       "goodman": {"materials": {"steel": {"su_mpa": 1100.0, "sy_mpa": 1000.0, "se_mpa": SE, "source": "numbers for a test"}}}}


def verify(work, *argv):
    from topopt.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["verify", str(work / "run"), "--mesh-size", str(SIZE), "--yaml", str(work / "ring.yaml"), "--no-sensitivity", *argv])
    return code, out.getvalue(), err.getvalue(), json.loads((work / "run" / "verify" / "verify.json").read_text())


@pytest.fixture(scope="module")
def ring_run(tmp_path_factory):
    probe = tet_probe()
    if not probe.ok:          # a module fixture runs before the marker `needs_gmsh` skips: skip here without any work
        pytest.skip(f"needs gmsh (the 'tet' extra): {probe.state}: {probe.detail}. {probe.fix}")
    pytest.importorskip("manifold3d")
    pytest.importorskip("trimesh")
    work = tmp_path_factory.mktemp("ring")
    build_ring(work)
    case_yaml(work, MID)
    return work, verify(work)


def test_the_median_stress_at_the_bore_is_the_axial_stress_of_the_ring(ring_run):
    work, (code, out, err, res) = ring_run
    assert code == 0, err
    st = res["goodman"]["service_stress"]
    for case, scale in (("pull", 1.0), ("half", 0.5)):
        s = st[case]["mid"]
        assert s["vm_median_MPa"] == pytest.approx(SIGMA * scale, rel=0.03)               # measured +0.4 % at 3 mm
        assert s["vm_min_MPa"] <= s["vm_median_MPa"] <= s["vm_p99_9_MPa"] <= s["vm_max_MPa"] and s["n_qp"] > 1000
        assert abs(s["hoop_median_MPa"]) < 0.1 * SIGMA * scale                                  # a uniform axial stress has no hoop part
    # a linear problem: half the force, half the stress, at every statistic
    for key, v in st["pull"]["mid"].items():
        if key.startswith("vm_"):
            assert st["half"]["mid"][key] == pytest.approx(0.5 * v, rel=1e-9)


def test_the_bore_facts_describe_a_closed_wall_in_the_middle_third(ring_run):
    work, (code, out, err, res) = ring_run
    (e,) = res["bore_facts"]
    f = e["funnel"]
    assert e["status"] == "ok" and f["boundary"] > f["radius"] > f["axial"] >= f["normal"] >= f["side"] == e["facets"] > 100
    assert 0.9 < e["area_over_analytic"] <= 1.001                                         # the facets cover the wall of the contact range
    assert e["closes_cylinder"] is True and e["near_miss"]["facets"] == 0
    assert e["largest_vertex_deviation_mm"] < 0.35 and e["vertex_s_range_mm"][0] >= -2 * SIZE and e["vertex_s_range_mm"][1] <= L / 3 + 2 * SIZE      # s runs from the point of the bore
    assert 0 < e["sample_set"]["tets"] <= e["facets"] and e["facets"] < e["sample_set"]["points"] <= 14 * e["sample_set"]["tets"]


def test_the_goodman_rows_are_the_formula_of_their_own_stress(ring_run):
    work, (code, out, err, res) = ring_run
    g = res["goodman"]
    assert g["status"] == "done" and g["mean_stress"]["from"] == "zero_mean" and [r["id"] for r in g["rows"]] == ["steel/zero_mean/mid/p99_9", "steel/zero_mean/mid/max"]
    for r in g["rows"]:
        sa = g["service_stress"]["pull"]["mid"][f"vm_{r['statistic']}_MPa"]
        assert r["worst_case"] == "pull" and r["sa_MPa"] == sa and r["sm_MPa"] == 0.0
        assert r["n_goodman"] == 1.0 / (sa / SE + 0.0 / 1100.0) and r["peak_over_yield"] == (0.0 + sa) / 1000.0
        assert [c["case"] for c in r["cases"]] == ["pull", "half"] and r["cases"][1]["n_goodman"] == pytest.approx(2.0 * r["n_goodman"], rel=1e-9)
    assert g["lowest_n"]["value"] == min(r["n_goodman"] for r in g["rows"]) and g["facts"] and g["options"]


def test_the_text_has_the_lines_of_the_bore_and_the_rows(ring_run):
    work, (code, out, err, res) = ring_run
    assert "bore checks:    goodman (bore_checks in the YAML of the run); 1 bore(s): mid; statistics p99.9 and max" in out
    assert "bore mid (yaml): " in out and " wall facets, area " in out and " sample points (band 1.2 mm)" in out
    assert "steel/zero_mean/mid/p99_9: worst case pull" in out and "steel/zero_mean/mid/max: worst case pull" in out
    assert "  bores: mid: " in err and "  goodman: 2 rows, lowest n " in err                           # the progress lines


def test_the_flags_and_the_yaml_switch_the_checks(ring_run, tmp_path):
    work, _ = ring_run
    code, out, err, res = verify(work, "--no-bore-checks")
    assert code == 0 and not any(k in res for k in ("bore_checks", "bore_facts", "goodman", "pressfit")) and "bore" not in out
    code, out, err, res = verify(work, "--bore-checks")
    assert code == 0 and res["bore_checks"]["source"] == "the flag --bore-checks" and res["goodman"]["status"] == "done"
    case_yaml(work, None)                                          # the YAML with no block: the flag needs it (E13)
    from topopt.config import UserError
    from topopt.verify.run import verify_run
    with pytest.raises(UserError, match="--bore-checks needs a bore_checks block"):
        verify_run(work / "run", yaml_path=work / "ring.yaml", mesh_size=SIZE, estimate_only=True, bore_checks=True)
    case_yaml(work, MID)                                           # the module fixture is shared: put the block back
