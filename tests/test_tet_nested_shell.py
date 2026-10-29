"""F4-45: `topopt verify` could not mesh a result whose STL has a shell inside the material of another shell.

The case behind the tests is the bracket at voxel 0.7 mm with the mass objective: `nominal.stl` has two closed shells. The second is a
thin sheet (14.7 mm3, 1.8e-3 of the volume) that lies in the material of the first. gmsh (HXT) stops on it ("duplicate volume"), and
the Delaunay algorithm makes the solid of the first shell, which differs from the volume of the surface by the sheet: the volume of
the surface counts that material twice. Here a cube with a smaller cube inside has the same defect (its tet volume is 6 % below the
volume of the surface). The tests build the STL; no large file is in the repository.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from helpers import needs_gmsh
from topopt.tet import MeshingFailed, mesh_stl, mesher


def box(lo, hi, inward=False):
    """(V, F) of an axis-aligned box. With `inward` the normals point into the box: the shell of a cavity."""
    m = trimesh.creation.box(extents=np.array(hi, float) - np.array(lo, float))
    m.apply_translation((np.array(hi, float) + np.array(lo, float)) / 2)
    F = np.asarray(m.faces)[:, [0, 2, 1]] if inward else np.asarray(m.faces)
    return np.asarray(m.vertices, float), F


def join(*parts):
    V = np.concatenate([p[0] for p in parts])
    offsets = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    return V, np.concatenate([p[1] + o for p, o in zip(parts, offsets)])


OUTER = ((0, 0, 0), (10, 10, 10))
INNER = ((3, 3, 3), (7, 7, 7))              # 64 mm3, in the material of OUTER (1000 mm3)
APART = ((15, 0, 0), (20, 10, 10))          # 500 mm3, next to OUTER


def nested_surface():
    return join(box(*OUTER), box(*INNER))


def write_stl(path, surface):
    V, F = surface
    trimesh.Trimesh(V, F, process=False).export(str(path))
    return path


# --------------------------------------------------------------------------- where the shells lie (no gmsh)

def test_the_winding_number_is_one_inside_a_body_zero_outside_and_minus_one_inside_a_cavity():
    V, F = box(*OUTER)
    assert mesher.winding_number(V, F, [5, 5, 5]) == pytest.approx(1.0, abs=1e-9)
    assert mesher.winding_number(V, F, [25, 5, 5]) == pytest.approx(0.0, abs=1e-9)
    V, F = box(*OUTER, inward=True)
    assert mesher.winding_number(V, F, [5, 5, 5]) == pytest.approx(-1.0, abs=1e-9)


def test_shell_relations_names_a_nested_shell_a_cavity_and_a_body_that_is_apart():
    assert mesher.shell_relations(*box(*OUTER)) == []                               # one shell: nothing to say
    got = mesher.shell_relations(*nested_surface())
    assert [(s["kind"], s["others"], s["volume_mm3"]) for s in got] == [("outer", 0, 1000.0), ("nested", 1, 64.0)]
    assert got[1]["bbox_min_mm"] == [3.0, 3.0, 3.0] and got[1]["bbox_max_mm"] == [7.0, 7.0, 7.0] and got[1]["faces_count"] == 12
    got = mesher.shell_relations(*join(box(*OUTER), box(*INNER, inward=True)))
    assert [(s["kind"], s["others"], s["volume_mm3"]) for s in got] == [("outer", 0, 1000.0), ("cavity", 1, -64.0)]
    got = mesher.shell_relations(*join(box(*OUTER), box(*APART)))
    assert [s["kind"] for s in got] == ["outer", "outer"]
    # a body in the void of a cavity is apart; a shell in the material of that body is nested (the winding numbers add up)
    got = mesher.shell_relations(*join(box(*OUTER), box((2, 2, 2), (8, 8, 8), inward=True), box((4, 4, 4), (6, 6, 6)),
                                       box((4.5, 4.5, 4.5), (5.5, 5.5, 5.5))))
    assert [s["kind"] for s in got] == ["outer", "cavity", "outer", "nested"]


def test_the_sentence_about_the_shells_names_the_cause_and_the_options():
    text = mesher.shells_sentence(mesher.shell_relations(*nested_surface()))
    assert text.startswith("the surface has 2 closed shells (volumes 1000, 64 mm3): shell 2 (64 mm3, x 3 to 7, y 3 to 7, z 3 to 7 mm) ")
    assert "lies in the material of another shell" in text and "counts that material twice" in text
    assert "Options: repair the STL" in text and "cavity" not in text
    text = mesher.shells_sentence(mesher.shell_relations(*join(box(*OUTER), box(*INNER, inward=True))))
    assert "is a cavity (an enclosed void)" in text and "fill the void in the STL" in text and "boolean union" not in text


# --------------------------------------------------------------------------- the mesh (gmsh)

@needs_gmsh
def test_a_nested_shell_is_left_out_and_the_rest_is_meshed_with_the_first_algorithm():
    V, F = nested_surface()
    with pytest.raises(MeshingFailed, match="differs from the surface volume by -6.0"):
        mesher.mesh_surface(V, F, 3.0, shell_retry=False)                           # what main did: the cause is not named
    p, t, info = mesher.mesh_surface(V, F, 3.0)
    assert info["algorithm"] == 10 and info["volume_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert info["volume_surface_mm3"] == pytest.approx(1000.0)                      # the volume without the nested shell
    retry = info["shell_retry"]
    assert retry["step"] == "drop_nested_shells" and retry["shells_in_surface"] == 2
    assert retry["volume_before_mm3"] == pytest.approx(1064.0) and retry["volume_after_mm3"] == pytest.approx(1000.0)
    assert [(s["shell"], s["volume_mm3"], s["faces_count"]) for s in retry["left_out"]] == [(2, 64.0, 12)]
    assert retry["whole_surface_errors"][0].startswith("algorithm 10: ") and "-6.0" in retry["whole_surface_errors"][1]
    json.dumps(info)                                                                # the child process sends it as JSON
    vol = np.abs(np.linalg.det((p[:, t[1:]] - p[:, t[:1]]).transpose(2, 1, 0))).sum() / 6.0
    assert vol == pytest.approx(1000.0, rel=1e-9)


@needs_gmsh
@pytest.mark.parametrize("isolate", [True, False])
def test_mesh_stl_gives_the_fact_of_the_shell_retry(tmp_path, isolate):
    stl = write_stl(tmp_path / "nested.stl", nested_surface())
    mesh = mesh_stl(stl, size=3.0, isolate=isolate)
    f = mesh.facts
    assert f["algorithm"] == 10 and f["volume_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert f["volume_ratio_stl"] == pytest.approx(1000.0 / 1064.0, abs=2e-3)          # the STL volume counts the nested shell too
    retry = f["surface_shell_retry"]
    assert retry["left_out"][0]["volume_mm3"] == pytest.approx(64.0, rel=0.02)
    assert mesh.volumes().sum() == pytest.approx(f["volume_mm3"])
    json.dumps(f)


@needs_gmsh
def test_a_surface_that_one_try_meshes_has_no_shell_fact_and_the_same_mesh(tmp_path):
    one = mesh_stl(write_stl(tmp_path / "one.stl", box(*OUTER)), size=3.0, isolate=False)
    assert "surface_shell_retry" not in one.facts and one.facts["algorithm"] == 10
    apart = mesh_stl(write_stl(tmp_path / "apart.stl", join(box(*OUTER), box(*APART))), size=3.0, isolate=False)
    assert "surface_shell_retry" not in apart.facts and apart.facts["algorithm"] == 1       # HXT stops on two bodies, Delaunay meshes both
    assert apart.facts["volume_ratio_stl"] == pytest.approx(1.0, abs=1e-9)
    assert apart.volumes().sum() == pytest.approx(1500.0, rel=1e-6)


@needs_gmsh
def test_a_cavity_stops_with_a_text_that_names_it(tmp_path):
    V, F = join(box(*OUTER), box(*INNER, inward=True))
    with pytest.raises(MeshingFailed) as err:
        mesher.mesh_surface(V, F, 3.0)
    text = str(err.value)
    assert "differs from the surface volume by +6.8" in text                        # gmsh fills the void
    assert "shell 2 (-64 mm3, x 3 to 7, y 3 to 7, z 3 to 7 mm) is a cavity (an enclosed void)" in text
    assert "Options: fill the void in the STL" in text
    with pytest.raises(MeshingFailed, match="is a cavity"):
        mesh_stl(write_stl(tmp_path / "cavity.stl", (V, F)), size=3.0, isolate=False)


@needs_gmsh
def test_a_nested_shell_and_a_cavity_together_say_that_the_retry_failed_too():
    V, F = join(box(*OUTER), box((2, 2, 2), (8, 8, 8), inward=True), box((4, 4, 4), (6, 6, 6)), box((4.5, 4.5, 4.5), (5.5, 5.5, 5.5)))
    with pytest.raises(MeshingFailed) as err:
        mesher.mesh_surface(V, F, 3.0)
    text = str(err.value)
    assert "after the nested shell(s) were left out: algorithm 10: " in text
    assert "shell 2 (-216 mm3" in text and "is a cavity" in text and "shell 4 (1 mm3" in text and "lies in the material of another shell" in text
    assert "the nested shell(s) were left out and the surface was meshed again, and that failed too" in text
    assert "repair the STL" in text and "fill the void" in text


@needs_gmsh
def test_a_crossing_that_gmsh_names_is_not_a_nested_shell():
    """Two bodies that cross each other: gmsh names the crossing, so the shells are not looked at and the error is the one of gmsh."""
    V, F = join(box(*OUTER), box((8, 3, 3), (14, 7, 7)))
    with pytest.raises(MeshingFailed) as err:
        mesher.mesh_surface(V, F, 3.0)
    assert err.value.points and "closed shells" not in str(err.value)


def test_a_failure_of_the_shell_search_does_not_replace_the_error_of_gmsh(monkeypatch):
    """The look at the shells helps the error text; if it fails, the error of gmsh stays as it was."""
    def fail_everywhere(V, F, size, threads, factor, optimize, algorithms, log, workdir, errors, points):
        errors.append("algorithm 10: HXT 3D mesh failed")
        return None
    monkeypatch.setattr(mesher, "_try_algorithms", fail_everywhere)
    monkeypatch.setattr(mesher, "shell_relations", lambda V, F: (_ for _ in ()).throw(MemoryError("no room")))
    with pytest.raises(MeshingFailed) as err:
        mesher.mesh_surface(*nested_surface(), 3.0)
    assert str(err.value) == "algorithm 10: HXT 3D mesh failed"


def test_the_box_test_of_the_search_gives_the_same_kinds_for_shells_that_are_far_apart():
    far = [box((i * 100, 0, 0), (i * 100 + 10, 10, 10)) for i in range(4)]
    got = mesher.shell_relations(*join(*far, box((303, 3, 3), (307, 7, 7))))
    assert [s["kind"] for s in got] == ["outer", "outer", "outer", "outer", "nested"]
    assert [s["others"] for s in got] == [0, 0, 0, 0, 1]


# --------------------------------------------------------------------------- what verify says

def a_shell_step():
    return dict(step="drop_nested_shells", shells_in_surface=2, volume_before_mm3=8062.343311653135, volume_after_mm3=8049.800673030724,
                left_out=[dict(shell=2, volume_mm3=12.542638622429138, faces_count=292, bbox_min_mm=[4.0, 4.53, 2.225],
                               bbox_max_mm=[4.417, 17.49, 13.57])],
                whole_surface_errors=["algorithm 10: HXT 3D mesh failed",
                                      "algorithm 1: tet volume differs from the surface volume by -1.56e-03"])


def test_the_shell_line_says_what_was_left_out_and_why():
    from topopt.verify.report import shell_line
    line = shell_line(a_shell_step())
    assert line.startswith("surface: gmsh could not mesh the prepared surface (algorithm 10: HXT 3D mesh failed; algorithm 1: "
                           "tet volume differs from the surface volume by -1.56e-03). It has 2 closed shells, and this shell lies in "
                           "the material of another shell: shell 2 (12.54 mm3, 292 faces, x 4 to 4.417, y 4.53 to 17.49, z 2.225 to 13.57 mm).")
    assert "counted that material twice (8062.34 mm3)" in line and "left out and the surface was meshed again (8049.8 mm3)" in line
    assert line.endswith("If the shell is a thin void in the part, repair the STL and verify again")
    two = a_shell_step()
    two["left_out"] = two["left_out"] * 2
    assert "these shells lie in the material" in shell_line(two) and "They were left out" in shell_line(two)


def test_the_summary_of_a_verify_run_has_the_shell_line_and_only_then():
    from topopt.verify.report import shell_line, summary_lines
    step = a_shell_step()
    res = {"run": {"name": "x"}, "size_mm": 1.0, "backend": {"solver": "none"}, "band_mm": 0.5, "cases": {}, "regions": {"load": []},
           "timing_s": {"verify_total_s": 1.0}, "memory": {"peak_rss_bytes": 1, "estimate_after_meshing_bytes": 1},
           "mesh": {"tets": 1, "dofs": 3, "quality_min": 0.3, "below_0p05": 0, "p2_volume_over_stl": 1.0}}
    assert shell_line(step) not in summary_lines(res)
    res["mesh"]["surface_shell_retry"] = step
    lines = summary_lines(res)
    assert lines.count(shell_line(step)) == 1 and lines[2] == shell_line(step)


def test_verify_json_keeps_the_shell_fact_in_the_compact_mesh_facts():
    from topopt.verify.run import _compact_mesh_facts
    step = a_shell_step()
    assert json.loads(json.dumps(step)) == step
    assert _compact_mesh_facts({"tets": 5, "surface_shell_retry": step, "other": 1}) == {"tets": 5, "surface_shell_retry": step}


def test_the_ratchet_row_has_a_note_for_the_shell_retry_and_no_other_fact_changes():
    from topopt import ratchet as rt
    from topopt.verify.report import shell_line
    case = {"stiffness_N_per_mm": 10.0, "compliance_Nmm": 1.0, "max_displacement_mm": 0.1,
            "von_mises": {"percentile": 99.9, "percentile_by_volume_MPa": 5.0, "peak_qp_MPa": 8.0}}
    report = {"size_mm": 1.0, "cases": {"a": case}, "mesh": {"tets": 10}}
    plain = rt.verify_facts(report)
    step = a_shell_step()
    got = rt.verify_facts(dict(report, mesh={"tets": 10, "surface_shell_retry": step}))
    assert got["notes"] == [shell_line(step)] and got["sources"]["notes"] == "verify.json:mesh.surface_shell_retry"
    assert {k: v for k, v in got.items() if k not in ("notes", "sources")} == {k: v for k, v in plain.items() if k != "sources"}
