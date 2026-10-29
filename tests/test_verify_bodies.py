"""Bodies of the mesh and a singular matrix in `topopt verify` (CONTRACTS.md section 16, "Bodies and a singular matrix").

A part that the tet model cannot solve (a body with no support, a load region that no body touches, a singular matrix)
gives a user error with the facts and the options: exit code 1, no traceback, no "internal error". A mesh of two boxes
made with `verify_mesh.box_mesh` and a fake `mesh_stl` stand in for gmsh, so these tests need no gmsh.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import fake_factor_availability, needs_skfem  # noqa: E402
from verify_mesh import box_mesh  # noqa: E402

pytestmark = needs_skfem

GAP = 10.0                       # the second box starts this far (mm) along x
LENGTHS, CELLS = (4.0, 2.0, 2.0), (4, 2, 2)


def two_boxes():
    """`(p, t)` of two boxes that share no vertex: box 1 at x 0 to 4, box 2 at x 10 to 14."""
    p, t = box_mesh(LENGTHS, CELLS)
    p2 = p.copy()
    p2[0] += GAP
    return np.hstack([p, p2]), np.hstack([t, t + p.shape[1]])


def box_stl(path, lo, hi):
    import trimesh
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    box.export(str(path))
    return path


def selections(model, tmp_path, fixed_boxes, load_box, band=0.3):
    """`(fixed_sel, load_sel, bfacets)` for region boxes given as `(lo, hi)`."""
    from topopt.verify.regions import boundary_facets, select_region
    bf = boundary_facets(model.t.T)
    fixed = [select_region(model, bf, box_stl(tmp_path / f"fixed{i}.stl", *b), f"fixed{i}", "fixed", band)
             for i, b in enumerate(fixed_boxes)]
    load = {}
    if load_box is not None:
        load["tip"] = select_region(model, bf, box_stl(tmp_path / "tip.stl", *load_box), "tip", "load", band)
    return fixed, load, bf


FIXED_1 = ((-1, -1, -1), (0.5, 3, 3))                   # the x = 0 face of box 1
FIXED_2 = ((GAP - 1, -1, -1), (GAP + 0.5, 3, 3))        # the x = 10 face of box 2
LOAD_2 = ((GAP + 3.5, -1, -1), (GAP + 5, 3, 3))         # the x = 14 face of box 2


# --------------------------------------------------------------------------- #
# The bodies of a mesh
# --------------------------------------------------------------------------- #

def test_two_boxes_are_two_bodies_with_the_facts_of_each():
    from topopt.verify.bodies import find_bodies, one_line
    from topopt.verify.p2 import P2Model
    p, t = two_boxes()
    model = P2Model(p, t)
    b = find_bodies(model)
    assert b.count == 2 and len(b.facts) == 2
    for f in b.facts:
        assert f["volume_mm3"] == pytest.approx(16.0) and f["tets"] == 96            # 16 cubes of 6 tets
    assert sorted(f["body"] for f in b.facts) == [1, 2]
    # every vertex and every edge midpoint has a body, and the two boxes have separate nodes
    assert b.vertex_body.shape == (model.nv,) and b.node_body.shape == (model.nn,)
    x = model.node_xyz[0]
    assert set(b.node_body[x < 5]) != set(b.node_body[x > 5]) and len(set(b.node_body)) == 2
    assert "bodies: 2" in one_line(b) and "volumes" in one_line(b)


def test_one_box_is_one_body_and_the_larger_body_comes_first():
    from topopt.verify.bodies import find_bodies, one_line
    from topopt.verify.p2 import P2Model
    p, t = box_mesh(LENGTHS, CELLS)
    b = find_bodies(P2Model(p, t))
    assert b.count == 1 and set(b.node_body) == {0} and "bodies: 1" in one_line(b)
    # a small box and a big box: the big one is body 1
    ps, ts = box_mesh((1.0, 1.0, 1.0), (1, 1, 1))
    pb, tb = box_mesh((4.0, 4.0, 4.0), (2, 2, 2))
    pb = pb + np.array([[20.0], [0.0], [0.0]])
    model = P2Model(np.hstack([ps, pb]), np.hstack([ts, tb + ps.shape[1]]))
    b = find_bodies(model)
    assert b.count == 2 and b.facts[0]["volume_mm3"] == pytest.approx(64.0) and b.facts[1]["volume_mm3"] == pytest.approx(1.0)
    assert b.facts[0]["bbox_min_mm"][0] == pytest.approx(20.0)


def test_the_fixed_nodes_and_load_facets_are_counted_for_each_body(tmp_path):
    from topopt.verify.bodies import add_regions, floating, find_bodies
    from topopt.verify.p2 import P2Model
    model = P2Model(*two_boxes())
    b = find_bodies(model)
    fixed, load, _ = selections(model, tmp_path, [FIXED_1], LOAD_2)
    add_regions(b, fixed, load)
    one, two = b.facts
    assert one["fixed_nodes"] == fixed[0].facts["nodes"] > 0 and one["load_facets"] == {}
    assert two["fixed_nodes"] == 0 and two["load_facets"] == {"tip": load["tip"].facts["facets"]}
    assert floating(b) == [1]


# --------------------------------------------------------------------------- #
# The errors before the solve
# --------------------------------------------------------------------------- #

def test_a_body_with_no_fixed_node_is_a_user_error_with_facts_and_options(tmp_path):
    from topopt.config import UserError
    from topopt.verify.bodies import add_regions, check_supported, find_bodies
    from topopt.verify.errors import UnsolvableMesh, VerifyError
    from topopt.verify.p2 import P2Model
    model = P2Model(*two_boxes())
    b = find_bodies(model)
    fixed, load, _ = selections(model, tmp_path, [FIXED_1], LOAD_2)
    add_regions(b, fixed, load)
    with pytest.raises(UnsolvableMesh) as e:
        check_supported(b, fixed, 0.3, reach=0.6)
    exc = e.value
    assert isinstance(exc, UserError) and not isinstance(exc, VerifyError)           # exit code 1
    text = str(exc)
    assert "2 bodies" in text and "body 2" in text and "no fixed node" in text and "singular" in text
    assert "tip" in text                                                              # the load region sits on the floating body
    assert "not a part that carries the load" in text
    assert "--region-band-mm" in text and "Options" in text and "Likely causes" in text
    assert "closest approach" in text or "no node within" in text
    assert exc.facts["floating_bodies"] == [2] and exc.facts["bodies"] == 2 and len(exc.facts["body_facts"]) == 2


def test_a_mesh_of_two_supported_bodies_is_not_an_error(tmp_path):
    from topopt.verify.bodies import add_regions, check_supported, find_bodies
    from topopt.verify.p2 import P2Model
    model = P2Model(*two_boxes())
    b = find_bodies(model)
    fixed, load, _ = selections(model, tmp_path, [FIXED_1, FIXED_2], LOAD_2)
    add_regions(b, fixed, load)
    check_supported(b, fixed, 0.3)                                                    # no error: each body has a support
    assert [f["fixed_nodes"] > 0 for f in b.facts] == [True, True]


def _held_selection(model, tmp_path, box):
    """The selection of a joint or spring region (a load-region selection): the facets of `box`."""
    from topopt.verify.regions import boundary_facets, select_region
    return select_region(model, boundary_facets(model.t.T), box_stl(tmp_path / "pad.stl", *box), "pad", "load", 0.3)


def test_a_body_held_by_a_joint_or_spring_facet_next_to_a_floating_body(tmp_path):
    """K1-A: `fixed: []` and a joint or spring region on body 1. Body 1 is held, body 2 floats: the error names the supports (REVIEW-135 L1, L5)."""
    from topopt.verify.bodies import add_regions, check_supported, find_bodies, floating
    from topopt.verify.errors import UnsolvableMesh
    from topopt.verify.p2 import P2Model
    model = P2Model(*two_boxes())
    b = find_bodies(model)
    pad = _held_selection(model, tmp_path, FIXED_1)
    add_regions(b, [], {}, held=[pad])
    one, two = b.facts
    assert one["support_facets"] == pad.facts["facets"] > 0 and two["support_facets"] == 0 and one["fixed_nodes"] == 0
    assert floating(b) == [1]
    with pytest.raises(UnsolvableMesh) as e:
        check_supported(b, [], 0.3, reach=0.6, support_sel=[pad])
    text = str(e.value)
    assert "no fixed node and no facet of a joint or spring" in text and "body 2" in text and "body 1" not in text.split("(")[0]
    assert "touch a fixed, joint or spring region" in text and "the fixed, joint or spring region does not reach this body" in text
    assert "closest approach to a fixed, joint or spring region" in text or "no node within 0.6 mm of a fixed, joint or spring region" in text
    assert "joint or spring facets" in text                                              # the body line says how many facets hold a body
    assert e.value.facts["floating_bodies"] == [2]
    add_regions(b, [], {}, held=[pad, _held_selection(model, tmp_path, FIXED_2)])
    check_supported(b, [], 0.3)                                                           # both bodies held: no error


def test_a_plain_part_has_no_support_key_and_keeps_the_old_error_text(tmp_path):
    from topopt.verify.bodies import add_regions, check_supported, find_bodies
    from topopt.verify.errors import UnsolvableMesh
    from topopt.verify.p2 import P2Model
    model = P2Model(*two_boxes())
    b = find_bodies(model)
    fixed, load, _ = selections(model, tmp_path, [FIXED_1], LOAD_2)
    add_regions(b, fixed, load)
    assert all("support_facets" not in f for f in b.facts)
    with pytest.raises(UnsolvableMesh) as e:
        check_supported(b, fixed, 0.3, reach=0.6)
    assert "has no fixed node, so it can move" in str(e.value) and "joint" not in str(e.value) and "spring" not in str(e.value)


def test_the_error_names_the_option_of_the_mcp_tool_in_the_mcp_style(tmp_path):
    from topopt.verify import names
    from topopt.verify.bodies import add_regions, check_supported, find_bodies
    from topopt.verify.errors import UnsolvableMesh
    from topopt.verify.p2 import P2Model
    model = P2Model(*two_boxes())
    b = find_bodies(model)
    fixed, load, _ = selections(model, tmp_path, [FIXED_1], LOAD_2)
    add_regions(b, fixed, load)
    with names.use_style("mcp"), pytest.raises(UnsolvableMesh) as e:
        check_supported(b, fixed, 0.3)
    assert "region_band_mm" in str(e.value) and "--region-band-mm" not in str(e.value)


def test_a_load_region_that_no_body_touches_is_a_user_error(tmp_path):
    from topopt.verify.bodies import add_regions, empty_load_error, find_bodies
    from topopt.verify.errors import EmptyRegion, UnsolvableMesh
    from topopt.verify.p2 import P2Model
    from topopt.verify.regions import boundary_facets, select_region
    model = P2Model(*two_boxes())
    b = find_bodies(model)
    path = box_stl(tmp_path / "far.stl", (30, -1, -1), (32, 3, 3))
    with pytest.raises(EmptyRegion) as e:
        select_region(model, boundary_facets(model.t.T), path, "tip", "load", 0.3)
    fixed, _, _ = selections(model, tmp_path, [FIXED_1, FIXED_2], None)
    add_regions(b, fixed, {})
    err = empty_load_error(e.value, b)
    assert isinstance(err, UnsolvableMesh)
    text = str(err)
    assert "load region 'tip' touches no body" in text and "2 bodies" in text and "not a part that carries the load" in text
    assert "--region-band-mm" in text and err.facts["reason"] == "load_region_touches_no_body"
    assert err.facts["nearest_distance_mm"] == pytest.approx(16.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# A singular matrix from each solver path
# --------------------------------------------------------------------------- #

class _FakeLinearSolver:
    """Stands for `topopt.solver.LinearSolver`: the factor raises the error that the real PARDISO raises."""
    message = ("PARDISO factor failed: The Pardiso solver failed with error code -4. See Pardiso documentation for details. "
               "(zero pivot: the matrix is not positive definite (a singular or unsupported system?))")

    def __init__(self, method="auto", **kw):
        self.method = method

    def factor(self, K, nullspace=None):
        raise RuntimeError(self.message)

    def solve(self, B, warm=True):
        raise RuntimeError(self.message)

    def free(self):
        pass


def small_spd(n=12):
    import scipy.sparse as sp
    return sp.diags([np.full(n, 4.0), np.full(n - 1, -1.0), np.full(n - 1, -1.0)], [0, 1, -1], format="csr")


@pytest.mark.parametrize("name", ["pardiso", "cholesky"])
def test_a_zero_pivot_of_the_linear_solver_is_a_user_error_with_facts_causes_and_options(monkeypatch, name):
    import topopt.solver
    from topopt.config import UserError
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import make_factor
    monkeypatch.setattr(topopt.solver, "LinearSolver", _FakeLinearSolver)
    with pytest.raises(SingularSystem) as e:
        make_factor(small_spd(), name)
    exc = e.value
    assert isinstance(exc, UserError)
    text = str(exc)
    assert name in text and "singular" in text and "zero pivot" in text and "error code -4" in text
    assert "Likely causes" in text and "floating body" in text and "no contact" in text
    assert "Options" in text and "--region-band-mm" in text and "--mesh-size" in text
    assert exc.facts["solver"] == name and exc.facts["reason"] == "singular_matrix" and exc.facts["dofs"] == 12


def test_the_accelerate_status_of_a_singular_matrix_is_a_user_error(monkeypatch):
    import topopt.solver
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import make_factor

    class Fake(_FakeLinearSolver):
        def factor(self, K, nullspace=None):
            raise RuntimeError("Accelerate SparseFactor failed, status -2")

    monkeypatch.setattr(topopt.solver, "LinearSolver", Fake)
    with pytest.raises(SingularSystem):
        make_factor(small_spd(), "cholesky")


def test_a_singular_solve_after_a_good_factor_is_a_user_error_too(monkeypatch):
    import topopt.solver
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import make_factor

    class Fake(_FakeLinearSolver):
        def factor(self, K, nullspace=None):
            pass

    monkeypatch.setattr(topopt.solver, "LinearSolver", Fake)
    fac = make_factor(small_spd(), "pardiso")
    with pytest.raises(SingularSystem):
        fac.solve(np.ones(12))


def test_another_error_of_the_solver_is_not_called_singular(monkeypatch):
    """An out-of-memory error or a bug is not turned into a user error: it goes through."""
    import topopt.solver
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import make_factor

    class Fake(_FakeLinearSolver):
        def factor(self, K, nullspace=None):
            raise RuntimeError("PARDISO factor failed: not enough memory (error code -2)")

    monkeypatch.setattr(topopt.solver, "LinearSolver", Fake)
    with pytest.raises(RuntimeError, match="not enough memory") as e:
        make_factor(small_spd(), "pardiso")
    assert not isinstance(e.value, SingularSystem)


def test_cholespy_on_a_matrix_that_is_not_positive_definite_is_a_user_error():
    """The real cholespy raises ValueError "Matrix is not positive definite!" (CHOLMOD prints a warning). This matrix is indefinite
    (the pivot of row 2 is 1 - 4 = -3), so the error does not depend on the roundoff of an assembled matrix."""
    pytest.importorskip("cholespy")
    import scipy.sparse as sp
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import make_factor
    K = sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    with pytest.raises(SingularSystem) as e:
        make_factor(K, "cholespy")
    assert "cholespy" in str(e.value) and "not positive definite" in str(e.value) and "Options" in str(e.value)


def test_cholespy_on_a_floating_body_is_a_user_error(tmp_path):
    """The floating body (box 1 fixed) is a `SingularSystem` on every numpy pin. The route depends on the roundoff of the assembly.
    With numpy 2.5.3 CHOLMOD raises ValueError "Matrix is not positive definite!" (and prints a warning). With numpy 2.3.5 (the
    Intel Mac lock) the entries differ in the last digits (19,127 of 77,949 entries, at most 1.8e-12), CHOLMOD factors the matrix with
    no error, and the solution has entries of 1e11 to 1e12: the test solve after the factor (a relative residual of 0.8, E3b) stops it.
    This body is one example: CHOLMOD is often silent on a singular matrix on both pins (see the cube tests at the end of this file)."""
    pytest.importorskip("cholespy")
    from topopt.verify import bc
    from topopt.verify.bodies import find_bodies
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import make_factor
    from topopt.verify.p2 import P2Model
    model = P2Model(*two_boxes())
    fixed = bc.fixed_dofs(np.flatnonzero(model.node_xyz[0] < 0.01))          # box 1 only
    free = np.setdiff1d(np.arange(model.N), fixed)
    K = model.stiffness(3500.0, 0.35)[free][:, free]
    assert find_bodies(model).count == 2
    with pytest.raises(SingularSystem) as e:
        make_factor(K, "cholespy")
    assert "cholespy" in str(e.value) and "Options" in str(e.value)
    detail = e.value.facts["detail"]                                         # the error of the solver, or of the test solve
    assert "not positive definite" in detail or "test solve with a relative residual of" in detail


def test_superlu_on_a_floating_body_is_a_user_error(tmp_path):
    """SuperLU does not raise on this matrix (no pivoting): the factor finds it with a test solve."""
    from topopt.verify import bc
    from topopt.verify.bodies import find_bodies
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import make_factor
    from topopt.verify.p2 import P2Model
    model = P2Model(*two_boxes())
    fixed = bc.fixed_dofs(np.flatnonzero(model.node_xyz[0] < 0.01))          # box 1 only
    free = np.setdiff1d(np.arange(model.N), fixed)
    K = model.stiffness(3500.0, 0.35)[free][:, free]
    assert find_bodies(model).count == 2
    with pytest.raises(SingularSystem) as e:
        make_factor(K, "superlu")
    assert "superlu" in str(e.value) and "Options" in str(e.value)
    assert "or a solve with a large residual" in str(e.value)               # the test solve is named in the first clause (review n1)


def test_singular_reason_reads_the_texts_of_the_solvers():
    from topopt.verify.factor import singular_reason
    assert singular_reason(ValueError("Matrix is not positive definite!"))
    assert singular_reason(np.linalg.LinAlgError("Singular matrix"))
    assert singular_reason(RuntimeError("Factor is exactly singular"))
    assert singular_reason(RuntimeError("PARDISO factor failed: error code -7 (diagonal matrix is singular)"))
    assert singular_reason(ValueError("non-positive diagonal in the stiffness matrix"))
    assert singular_reason(RuntimeError("out of memory")) is None
    assert singular_reason(KeyError("x")) is None


# --------------------------------------------------------------------------- #
# The command: a fake mesher, a run folder of two boxes
# --------------------------------------------------------------------------- #

def make_run(tmp_path, fixed_boxes, load_box):
    """A run folder, its YAML and the region STL files. `final.stl` is the two boxes (one closed STL file)."""
    import trimesh
    d = tmp_path / "files"
    d.mkdir()
    box_stl(d / "design.stl", (-1, -1, -1), (GAP + 5, 3, 3))
    for i, b in enumerate(fixed_boxes):
        box_stl(d / f"fixed{i}.stl", *b)
    box_stl(d / "tip.stl", *load_box)
    fixed = "\n".join(f"    - fixed{i}.stl" for i in range(len(fixed_boxes)))
    (d / "problem.yaml").write_text(
        f"name: two_boxes\nregions:\n  design: design.stl\n  fixed:\n{fixed}\n  load:\n    tip: tip.stl\n"
        "load_cases:\n  - name: pull\n    forces:\n      - region: tip\n        force: [0.0, -5.0, 0.0]\n"
        "material:\n  E: 3500.0\n  nu: 0.35\n  density: 1.24\nsettings:\n  voxel_size: 0.6\n  volume_fraction: 0.3\n"
        "  iterations: 5\n")
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(dict(name="two_boxes", h=0.6, yaml_path=str(d / "problem.yaml"))))
    a = trimesh.creation.box(extents=LENGTHS)
    a.apply_translation(np.array(LENGTHS) / 2)
    b = a.copy()
    b.apply_translation((GAP, 0, 0))
    trimesh.util.concatenate([a, b]).export(str(run / "result" / "final.stl"))
    return run


def fake_mesher(monkeypatch, p, t):
    """Replace the gmsh mesher with the mesh of the two boxes. Pretend that PARDISO runs here."""
    import topopt.tet
    from topopt.tet import TetMesh
    import topopt.verify.factor as factor

    def mesh_stl(path, size=None, threads=1, optimize=None, log=None, **kw):
        nodes, tets = np.ascontiguousarray(p.T), np.ascontiguousarray(t.T)
        m = TetMesh(nodes, tets, {})
        q = m.quality()
        m.facts.update(size=size, tets=len(tets), nodes=len(nodes), quality_min=float(q.min()),
                       below_0p05=int((q < 0.05).sum()), volume_stl_mm3=float(np.abs(m.volumes()).sum()))
        return m

    monkeypatch.setattr(topopt.tet, "mesh_stl", mesh_stl)
    avail = {"cholesky": {"available": False, "note": "fake"}, "cholespy": {"available": False, "note": "fake"},
             "pardiso": {"available": True, "note": "fake"}}
    monkeypatch.setattr(factor, "factor_availability", lambda: avail)
    from topopt.verify import run as vrun
    ok = {"gmsh": {"state": "ok", "version": "4.15.2", "detail": "", "fix": ""},
          "scikit-fem": {"state": "ok", "version": None}, "cholespy": {"state": "ok", "version": None}}
    monkeypatch.setattr(vrun, "extras_facts", lambda: ok)


def cli(*argv):
    from topopt.cli import main
    return main(["verify", *map(str, argv)])


def test_the_command_stops_before_the_solve_for_a_floating_body_with_exit_code_1(tmp_path, monkeypatch, capsys):
    import topopt.solver
    run = make_run(tmp_path, [FIXED_1], LOAD_2)
    fake_mesher(monkeypatch, *two_boxes())

    class Never(_FakeLinearSolver):
        def factor(self, K, nullspace=None):
            raise AssertionError("the check must stop before the factor")

    monkeypatch.setattr(topopt.solver, "LinearSolver", Never)
    assert cli(run, "--solver", "pardiso", "--no-sensitivity") == 1
    cap = capsys.readouterr()
    assert "bodies: 2" in cap.err                                      # the count is stated before the solve
    assert cap.err.count("error:") == 1 and "internal error" not in cap.err and "Traceback" not in cap.err
    assert "no fixed node" in cap.err and "body 2" in cap.err and "--region-band-mm" in cap.err
    status = json.loads((run / "verify" / "status.json").read_text())
    assert status["state"] == "error" and status["exit_code"] == 1


def test_the_command_reports_a_load_region_that_no_body_touches_with_exit_code_1(tmp_path, monkeypatch, capsys):
    run = make_run(tmp_path, [FIXED_1, FIXED_2], ((GAP + 20, -1, -1), (GAP + 22, 3, 3)))
    fake_mesher(monkeypatch, *two_boxes())
    assert cli(run, "--solver", "pardiso", "--no-sensitivity") == 1
    err = capsys.readouterr().err
    assert "load region 'tip' touches no body" in err and "internal error" not in err and "Traceback" not in err


def test_the_command_reports_a_zero_pivot_with_exit_code_1_and_the_facts_of_the_mesh(tmp_path, monkeypatch, capsys):
    import topopt.solver
    run = make_run(tmp_path, [FIXED_1, FIXED_2], LOAD_2)
    fake_mesher(monkeypatch, *two_boxes())
    monkeypatch.setattr(topopt.solver, "LinearSolver", _FakeLinearSolver)          # PARDISO says: zero pivot
    assert cli(run, "--solver", "pardiso", "--no-sensitivity") == 1
    err = capsys.readouterr().err
    assert "bodies: 2" in err and "internal error" not in err and "Traceback" not in err
    assert "singular" in err and "zero pivot" in err and "The mesh has 2 bodies (each has a fixed node)" in err
    assert "Likely causes" in err and "Options" in err


def test_the_json_of_a_good_run_has_the_body_count(tmp_path, monkeypatch, capsys):
    """Two supported bodies solve. The report states the count and the facts of each body."""
    pytest.importorskip("cholespy")
    import topopt.verify.factor as factor
    run = make_run(tmp_path, [FIXED_1, FIXED_2], LOAD_2)
    fake_mesher(monkeypatch, *two_boxes())
    avail = {"cholesky": {"available": False, "note": "fake"}, "pardiso": {"available": False, "note": "fake"},
             "cholespy": {"available": True, "note": "real"}}
    monkeypatch.setattr(factor, "factor_availability", lambda: avail)
    assert cli(run, "--solver", "cholespy", "--no-sensitivity") == 0
    cap = capsys.readouterr()
    assert "2 bodies" in cap.out and "bodies: 2" in cap.err
    res = json.loads((run / "verify" / "verify.json").read_text())
    bodies = res["mesh"]["bodies"]
    assert bodies["count"] == 2 and [b["fixed_nodes"] > 0 for b in bodies["bodies"]] == [True, True]
    assert bodies["bodies"][0]["load_facets"] == {} and bodies["bodies"][1]["load_facets"]["tip"] > 0
    assert res["cases"]["pull"]["equilibrium"]["force_relative_residual"] <= 1e-6


def test_the_superlu_estimate_uses_the_compactness_of_the_part_before_and_after_meshing(tmp_path, monkeypatch, capsys):
    """If C were lost (the hook value used for every part), the numbers below would differ: the parts of the tests are not hooks."""
    import topopt.verify.factor as factor
    from topopt.verify import memory
    from topopt.verify.run import level_estimate, stl_geometry
    run = make_run(tmp_path, [FIXED_1, FIXED_2], LOAD_2)
    fake_mesher(monkeypatch, *two_boxes())
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("superlu"))
    assert cli(run, "--no-sensitivity") == 0
    res = json.loads((run / "verify" / "verify.json").read_text())
    g = stl_geometry(run / "result" / "final.stl")
    c = memory.compactness(g["volume_mm3"], g["area_mm2"])
    assert abs(c - memory.HOOK_COMPACTNESS) > 0.01                                 # a compact part, not a hook
    (e,) = res["estimates_before_meshing"]
    assert e["compactness"] == pytest.approx(c)
    assert e["estimate_bytes"] == memory.estimate_bytes(e["estimated_dofs"], "superlu", c)             # before meshing
    assert e["estimate_bytes"] != memory.estimate_bytes(e["estimated_dofs"], "superlu")                 # not the hook value
    n = res["mesh"]["dofs"]
    assert res["memory"]["estimate_after_meshing_bytes"] == memory.estimate_bytes(n, "superlu", c)     # after meshing
    assert res["memory"]["estimate_after_meshing_bytes"] != memory.estimate_bytes(n, "superlu")
    rng = res["memory"]["estimate_range_after_meshing_bytes"]
    assert rng["central"] == pytest.approx(res["memory"]["estimate_after_meshing_bytes"], abs=1) and rng["low"] < rng["high"]
    # the same at a size where the fill matters: a cube of 30 mm at 0.5 mm, C = 0.068 (the hook value would give 18 times less fill)
    cube = dict(volume_mm3=27000.0, area_mm2=5400.0, default_size_mm=1.0)
    big = level_estimate(cube, 0.5, {"limit_bytes": None}, 2, "superlu")
    assert big["compactness"] == pytest.approx(0.068, abs=1e-3)
    assert big["estimate_bytes"] == memory.estimate_bytes(big["estimated_dofs"], "superlu", big["compactness"])
    assert big["estimate_bytes"] > 5 * memory.estimate_bytes(big["estimated_dofs"], "superlu")


def test_auto_runs_on_superlu_when_it_is_the_only_factor_and_the_report_says_so(tmp_path, monkeypatch, capsys):
    """No pypardiso, no cholespy (and no Accelerate): `auto` takes scipy SuperLU, and the run states it and why."""
    import topopt.verify.factor as factor
    run = make_run(tmp_path, [FIXED_1, FIXED_2], LOAD_2)
    fake_mesher(monkeypatch, *two_boxes())
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("superlu"))
    assert cli(run, "--no-sensitivity") == 0
    cap = capsys.readouterr()
    assert "solver superlu" in cap.out and "internal error" not in cap.err and "Traceback" not in cap.err
    assert "solver:         superlu  (requested auto)" in cap.err                       # the fact, before the work starts
    assert all(f"auto skipped {n}: " in cap.err for n in ("cholesky", "pardiso", "cholespy"))
    assert "factor (superlu):" in cap.err and "time estimate about" in cap.err
    res = json.loads((run / "verify" / "verify.json").read_text())
    b = res["backend"]
    assert b["solver"] == "superlu" and b["requested"] == "auto" and list(b["skipped"]) == ["cholesky", "pardiso", "cholespy"]
    assert res["memory"]["estimate_model"].startswith("SuperLU model")
    assert res["memory"]["estimate_seconds_after_meshing"] > 0 and res["memory"]["peak_rss_bytes"] > 0
    assert res["cases"]["pull"]["linear_solve"]["relative_residual"] <= 1e-9
    assert res["cases"]["pull"]["equilibrium"]["force_relative_residual"] <= 1e-6
    superlu_p99 = res["cases"]["pull"]["von_mises"]["p99_9_by_volume_MPa"]
    pytest.importorskip("cholespy")                                                       # the same answer from another factor
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability("cholespy"))
    assert cli(run, "--no-sensitivity") == 0
    other = json.loads((run / "verify" / "verify.json").read_text())
    assert other["backend"]["solver"] == "cholespy"
    assert other["cases"]["pull"]["von_mises"]["p99_9_by_volume_MPa"] == pytest.approx(superlu_p99, rel=1e-6)


# --------------------------------------------------------------------------- #
# A part held along one line, and a load case with no force (E3b, review M1)
# --------------------------------------------------------------------------- #

CUBE_MM = 6.0


def cube_run(tmp_path, held, cases=("pull",)):
    """A run folder of a 6 mm cube (the mesh of `box_mesh((6, 6, 6), (6, 6, 6))`, made by `cube_mesh`).

    `held` "line": the edge x 0, y 0 is the fixed region. The part is held at nodes on one line, so it can turn about that line:
    the matrix is singular, and the body has fixed nodes, so `check_supported` passes. `held` "face": the face x 0 (valid).
    `cases`: the load cases in order, "pull" (5 N along -y on the tip) or "idle" (a force of zero).
    """
    L = CUBE_MM
    d = tmp_path / "files"
    d.mkdir()
    box_stl(d / "design.stl", (-1, -1, -1), (L + 1, L + 1, L + 1))
    box_stl(d / "rod.stl", *(((-0.05, -0.05, -1), (0.05, 0.05, L + 1)) if held == "line" else ((-1, -1, -1), (0.05, L + 1, L + 1))))
    box_stl(d / "tip.stl", (L - 0.3, -1, -1), (L + 1, L + 1, L + 1))
    force = {"pull": "[0.0, -5.0, 0.0]", "idle": "[0.0, 0.0, 0.0]"}
    load_cases = "".join(f"  - name: {c}\n    forces:\n      - region: tip\n        force: {force[c]}\n" for c in cases)
    (d / "problem.yaml").write_text(
        "name: cube\nregions:\n  design: design.stl\n  fixed:\n    - rod.stl\n  load:\n    tip: tip.stl\n"
        f"load_cases:\n{load_cases}"
        "material:\n  E: 3500.0\n  nu: 0.35\n  density: 1.24\nsettings:\n  voxel_size: 1.0\n  volume_fraction: 0.3\n  iterations: 5\n")
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(dict(name="cube", h=1.0, yaml_path=str(d / "problem.yaml"))))
    box_stl(run / "result" / "final.stl", (0, 0, 0), (L, L, L))
    return run


def cube_setup(tmp_path, monkeypatch, held, solver, cases=("pull",), blind=False):
    """A cube run, the fake mesher, and a host where only `solver` can run. `blind`: the test solve of the factor finds nothing (its
    fixed vector misses the null space: the measured case of the E3b review, M1). Only the residual of the real load can stop the run."""
    import topopt.verify.factor as factor
    if solver == "cholespy":
        pytest.importorskip("cholespy")
    run = cube_run(tmp_path, held, cases)
    fake_mesher(monkeypatch, *box_mesh((CUBE_MM,) * 3, (6, 6, 6)))
    monkeypatch.setattr(factor, "factor_availability", lambda: fake_factor_availability(solver))
    if blind:
        monkeypatch.setattr(factor, "_test_solve_residual", lambda *a, **k: 0.0)
    return run


CUBE_ARGS = ("--region-band-mm", "0.01", "--no-sensitivity", "--force")        # --force: no dependence on the free RAM of the host


@pytest.mark.parametrize("solver", ["cholespy", "superlu"])
def test_the_residual_of_a_load_case_stops_a_part_held_along_one_line_when_the_test_solve_misses_it(tmp_path, monkeypatch, capsys, solver):
    """The cube held at the edge x 0, y 0 can turn about it. With one fixed test vector the factor can miss this singular matrix
    (measured: 7.7e-4 for cholespy and 4.2e-4 for SuperLU on numpy 2.5.3, below 1e-3), and the run then gave a displacement of 1e10 mm
    and exit code 0. Here the test solve is switched off, so the residual of the real load case (0.2 to 0.4) is what stops it: exit
    code 1, with the load case, the value and the threshold in the text and in the facts."""
    from topopt.verify.errors import SingularSystem
    from topopt.verify.factor import FACTOR_TEST_RESIDUAL
    from topopt.verify.run import verify_run
    run = cube_setup(tmp_path, monkeypatch, "line", solver, blind=True)
    assert cli(run, "--solver", solver, *CUBE_ARGS) == 1
    err = capsys.readouterr().err
    assert "singular" in err and "refined solve of load case 'pull' has a relative residual of" in err and "threshold 1e-03" in err
    assert "stopped on a zero pivot, a matrix that is not positive definite, or a solve with a large residual" in err     # review n1
    assert "Likely causes" in err and "Options" in err and "internal error" not in err and "Traceback" not in err
    status = json.loads((run / "verify" / "status.json").read_text())
    assert status["state"] == "error" and status["exit_code"] == 1
    with pytest.raises(SingularSystem) as e:
        verify_run(run, solver=solver, region_band_mm=0.01, load_sensitivity=False, force=True)
    f = e.value.facts
    assert f["reason"] == "singular_matrix" and f["solver"] == solver and f["load_case"] == "pull"
    assert f["threshold"] == FACTOR_TEST_RESIDUAL and f["relative_residual"] >= FACTOR_TEST_RESIDUAL
    assert f"{f['relative_residual']:.1e}" in f["detail"] and f["bodies"] == 1


@pytest.mark.parametrize("solver", ["cholespy", "superlu"])
def test_the_command_exits_with_1_on_a_part_held_along_one_line_with_the_real_factors(tmp_path, monkeypatch, capsys, solver):
    """The same cube, with the test solve on. Whichever of the two stops it (the test solve of the factor, or the residual of the
    load case, which depends on the roundoff of the platform), the exit code is 1 and the text names a relative residual."""
    run = cube_setup(tmp_path, monkeypatch, "line", solver)
    assert cli(run, "--solver", solver, *CUBE_ARGS) == 1
    err = capsys.readouterr().err
    assert "singular" in err and "relative residual of" in err and "threshold 1e-03" in err
    assert "internal error" not in err and "Traceback" not in err


@pytest.mark.parametrize("solver", ["cholespy", "superlu"])
def test_a_valid_part_keeps_exit_code_0_with_the_stop_on_the_residual(tmp_path, monkeypatch, capsys, solver):
    """The same cube held at the face x 0: a valid matrix, a residual far below the threshold, exit code 0."""
    run = cube_setup(tmp_path, monkeypatch, "face", solver)
    assert cli(run, "--solver", solver, *CUBE_ARGS) == 0
    res = json.loads((run / "verify" / "verify.json").read_text())
    assert res["cases"]["pull"]["linear_solve"]["relative_residual"] < 1e-10
    assert res["cases"]["pull"]["max_displacement_mm"] < 1.0


def test_a_load_case_with_no_force_does_not_stop_the_run_and_has_no_residual(tmp_path, monkeypatch, capsys):
    """|b| = 0: the relative residual is not defined. The run does not stop and does not divide by zero (a RuntimeWarning is an
    error here). The report has `None` and a note (not NaN: NaN is not valid JSON); the case that has a force has its residual."""
    import warnings
    run = cube_setup(tmp_path, monkeypatch, "face", "superlu", cases=("pull", "idle"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert cli(run, "--solver", "superlu", *CUBE_ARGS) == 0
    cap = capsys.readouterr()
    assert "case idle:" in cap.out and "solve residual not defined (the load is zero)" in cap.out
    text = (run / "verify" / "verify.json").read_text()
    res = json.loads(text, parse_constant=lambda c: pytest.fail(f"{c} in verify.json"))               # no NaN, no Infinity
    idle, pull = res["cases"]["idle"]["linear_solve"], res["cases"]["pull"]["linear_solve"]
    assert idle["relative_residual"] is None and idle["backward_error"] is None and "not defined" in idle["note"]
    assert pull["relative_residual"] < 1e-10 and pull["backward_error"] < 1e-10 and "note" not in pull
    assert res["cases"]["idle"]["max_displacement_mm"] == 0.0


def test_a_zero_load_case_does_not_hide_a_singular_part_and_is_not_the_one_that_stops_it(tmp_path, monkeypatch, capsys):
    """Two cases on the cube held along one line: `idle` (no force) comes first and is skipped; `pull` stops the run."""
    run = cube_setup(tmp_path, monkeypatch, "line", "superlu", cases=("idle", "pull"), blind=True)
    assert cli(run, "--solver", "superlu", *CUBE_ARGS) == 1
    err = capsys.readouterr().err
    assert "load case 'pull'" in err and "load case 'idle'" not in err and "residual of nan" not in err
