"""Tests for A3a (CONTRACTS.md section 22a): solid regions, the filter rule at frozen voxels,
and the minimum member size.

The solid tests use boxes and the voxel centres, so the expected counts come from the geometry
and not from the code under test.
"""

import filecmp
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import yaml

from test_optimize import box, make_case
from topopt import members
from topopt.cli import check_data, main
from topopt.config import UserError, load_config
from topopt.filter import build_filter, heaviside_project
from topopt.optimize import Model, run
from topopt.problem import build_problem
from topopt.snapshot import RunWriter
from topopt.voxelize import DESIGN, FIXED, KEEP_OUT, LOAD, solid_lines

# Every test of this file compares the demo with a record of the old pipeline: pytest -m equivalence.
pytestmark = pytest.mark.equivalence

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "data" / "filter_pipeline_fixture.json"

# The cantilever of `make_case`: design x 0..20, y 0..10, z 0..10 (mm); plate x -2..0; load tip at x 18..20.
CAP = ((8, 0, 0), (12, 10, 4))          # a solid cap inside the design space
RIB = ((16, 2, 0), (20, 8, 6))          # a solid box that overlaps the load region `tip`
STUB = ((-4, 0, 0), (2, 10, 4))         # a solid box that overlaps the fixed plate and sticks out of the design
# All faces are on even mm and the voxel size is 2 mm: no voxel centre lies on a face.
HOLE = ((8, 4, 0), (12, 6, 4))          # a keep-out region that cuts the cap


def write_case(tmp_path, solids=(), keep_out=(), as_load=False, h=2.0, iterations=4, **settings):
    """The cantilever with the given solid boxes. `as_load`: the old form, a load region with no force.

    The old form lists the solid boxes after `tip`, so `tip` wins where they overlap (the rule that
    `regions.solid` has by priority).
    """
    p = make_case(tmp_path, h=h, iterations=iterations)
    d = yaml.safe_load(p.read_text())
    names = []
    for i, (lo, hi) in enumerate(solids):
        box(tmp_path / f"solid{i}.stl", lo, hi)
        names.append(f"solid{i}.stl")
    for i, (lo, hi) in enumerate(keep_out):
        box(tmp_path / f"hole{i}.stl", lo, hi)
    if keep_out:
        d["regions"]["keep_out"] = [f"hole{i}.stl" for i in range(len(keep_out))]
    if names and as_load:
        d["regions"]["load"].update({n[:-4]: n for n in names})
    elif names:
        d["regions"]["solid"] = names
    d["settings"].update(settings)
    p.write_text(yaml.safe_dump(d, sort_keys=False))   # the order of `regions.load` is the priority
    return p


def sub(tmp_path, name):
    """A new sub-folder of `tmp_path` (a second case in one test)."""
    d = tmp_path / name
    d.mkdir()
    return d


def centres(prob):
    """Voxel centres (nx, ny, nz, 3) of a problem."""
    ijk = np.indices(prob.shape).transpose(1, 2, 3, 0)
    return prob.origin + prob.h * (ijk + 0.5)


def inside(prob, lo, hi):
    c = centres(prob)
    return np.all((c > np.asarray(lo, float)) & (c < np.asarray(hi, float)), axis=-1)


# --- solid regions -----------------------------------------------------------


def test_config_reads_solid_list(tmp_path):
    cfg = load_config(write_case(tmp_path, solids=[CAP, RIB]))
    assert [p.name for p in cfg.regions.solid] == ["solid0.stl", "solid1.stl"]


def test_config_without_solid_is_the_old_config(tmp_path):
    cfg = load_config(make_case(tmp_path))
    assert cfg.regions.solid == ()
    s = cfg.settings
    assert (s.filter_solid_neighbours, s.min_member_mm, s.filter_radius_source) == ("density_one", None, "default")
    assert s.filter_radius == 1.5 * s.voxel_size


@pytest.mark.parametrize("value, text", [("cap.stl", "must be a list of STL files"),
                                         (["missing.stl"], "STL file not found")])
def test_config_solid_errors(tmp_path, value, text):
    p = make_case(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["regions"]["solid"] = value
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match=text):
        load_config(p)


def test_solid_voxels_are_label_load_without_load_id(tmp_path):
    prob = build_problem(load_config(write_case(tmp_path, solids=[CAP])))
    cap = inside(prob, *CAP)
    assert cap.sum() == 20
    assert (prob.labels[cap] == LOAD).all()
    assert (prob.load_region_ids[cap] == -1).all()
    assert prob.region_facts["solid"]["voxels"] == 20
    # Fixed, free design and the load region are as without the cap.
    base = build_problem(load_config(make_case(sub(tmp_path, "base"), h=2.0)))  # same cantilever, no solid
    assert (prob.labels == FIXED).sum() == (base.labels == FIXED).sum()
    assert (prob.labels == DESIGN).sum() == (base.labels == DESIGN).sum() - 20


def test_priority_keep_out_fixed_load_solid_design(tmp_path):
    """keep_out > fixed > load > solid > design, and the counts are what the geometry gives."""
    p = write_case(tmp_path, solids=[CAP, RIB, STUB], keep_out=[HOLE])
    prob = build_problem(load_config(p))
    lab = prob.labels
    tip = inside(prob, (18, 4, 0), (20, 6, 3))
    plate = inside(prob, (-2, 0, 0), (0, 10, 10))
    hole = inside(prob, *HOLE)
    cap, rib, stub = (inside(prob, *b) for b in (CAP, RIB, STUB))
    design = inside(prob, (0, 0, 0), (20, 10, 10))
    assert (lab[hole] == KEEP_OUT).all()                      # keep_out wins over a solid
    assert (lab[plate & stub & ~hole] == FIXED).all()          # fixed wins over a solid
    assert (lab[tip & rib & ~hole] == LOAD).all()              # load wins over a solid ...
    assert (prob.load_region_ids[tip & rib] == 0).all()        # ... and keeps its load id
    only = (cap | rib | stub) & ~hole & ~plate & ~tip
    assert (lab[only] == LOAD).all() and (prob.load_region_ids[only] == -1).all()
    assert (lab[design & ~(cap | rib | stub | hole | tip)] == DESIGN).all()
    f = prob.region_facts["solid"]
    union = cap | rib | stub
    assert f["stl_voxels"] == int(union.sum())
    assert f["voxels"] == int(only.sum())
    assert f["in_keep_out"] == int((union & hole).sum()) > 0
    assert f["in_fixed"] == int((union & plate & ~hole).sum()) > 0       # a voxel counts for the region that wins
    assert f["in_load"] == int((union & tip & ~hole & ~plate).sum()) > 0
    assert f["stl_voxels"] == f["voxels"] + f["in_fixed"] + f["in_load"] + f["in_keep_out"]   # no overlap
    assert f["in_two_solid_stls"] == int(((cap.astype(int) + rib + stub) > 1).sum())
    assert f["outside_design_stl"] == int((only & ~design).sum()) > 0   # the stub sticks out at x < -2
    assert f["inside_design_stl"] + f["outside_design_stl"] == f["voxels"]
    assert sum(r["voxels"] for r in f["regions"]) >= f["voxels"] - f["in_two_solid_stls"]
    text = " ".join(solid_lines(f))
    assert f"{f['voxels']} voxels stay solid" in text and "keep-out region (they are removed)" in text
    assert "outside the design STL" in text


def test_solid_outside_design_adds_active_voxels(tmp_path):
    prob = build_problem(load_config(write_case(tmp_path, solids=[STUB])))
    f = prob.region_facts["solid"]
    active = int(((prob.labels >= DESIGN) & (prob.labels <= LOAD)).sum())
    assert f["outside_design_stl"] > 0
    assert active == int(((inside(prob, (0, 0, 0), (20, 10, 10)) | inside(prob, (-2, 0, 0), (0, 10, 10))
                           | inside(prob, *STUB))).sum())


def test_no_voxel_left_is_a_fact_not_an_error(tmp_path):
    """A solid STL that a fixed region covers completely has no voxel left. That is a count of 0."""
    p = write_case(tmp_path, solids=[((-1.6, 1, 1), (-0.4, 9, 9))])   # inside the plate x -2..0
    prob = build_problem(load_config(p))
    f = prob.region_facts["solid"]
    assert f["stl_voxels"] > 0 and f["voxels"] == 0 and f["in_fixed"] == f["stl_voxels"]


def test_old_form_gives_identical_labels_filter_and_run(tmp_path):
    """`regions.solid` equals the old form (a load region with no force): labels, filter and run."""
    kw = dict(solids=[CAP, RIB, STUB], keep_out=[HOLE])
    new_dir, old_dir = tmp_path / "new", tmp_path / "old"
    new_dir.mkdir(), old_dir.mkdir()
    new = build_problem(load_config(write_case(new_dir, **kw)))
    old = build_problem(load_config(write_case(old_dir, as_load=True, **kw)))
    assert new.shape == old.shape and (new.origin == old.origin).all()
    assert np.array_equal(new.labels, old.labels)
    assert np.array_equal(new.forces, old.forces)
    assert np.array_equal(new.fixed_dofs, old.fixed_dofs)
    assert np.array_equal(new.load_region_ids == 0, old.load_region_ids == 0)   # the real load region
    solid_only = (new.labels == LOAD) & (new.load_region_ids < 0)
    assert solid_only.any() and (old.load_region_ids[solid_only] > 0).all()
    for rule in ("density_one", "ignore"):
        Fn, _, bn = build_filter(new.labels, new.h, 3.0, rule)
        Fo, _, bo = build_filter(old.labels, old.h, 3.0, rule)
        assert (Fn != Fo).nnz == 0 and np.array_equal(bn, bo)


def one_thread_run(yaml_path, out):
    """`topopt run` in a new process with every thread cap at 1 (the runs are then repeatable bit for bit)."""
    env = {k: v for k, v in os.environ.items()}
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")})
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-m", "topopt", "run", str(yaml_path), "--out", str(out), "--quiet"],
                       capture_output=True, text=True, env=env, cwd=out.parent)
    assert r.returncode == 0, r.stderr
    return out


@pytest.mark.skipif(__import__("topopt.solver_choice", fromlist=["x"]).resolve_solver("auto") == "amg",
                    reason="bit-equal runs need a direct solver (pardiso or cholesky): with AMG the two forms differ at 1e-11")
def test_old_form_run_is_bit_equal_at_one_thread(tmp_path):
    """`regions.solid` and the old form give the same run, bit for bit, at 1 thread: all snapshots, every
    compliance value, `final.stl`, `preview.bin`. (With more threads the sums may differ in the last digit.)"""
    kw = dict(solids=[CAP, RIB, STUB], keep_out=[HOLE])
    new_dir, old_dir = sub(tmp_path, "new"), sub(tmp_path, "old")
    for rule in ("density_one", "ignore"):
        new = one_thread_run(write_case(new_dir, filter_solid_neighbours=rule, **kw), new_dir / f"run_{rule}")
        old = one_thread_run(write_case(old_dir, as_load=True, filter_solid_neighbours=rule, **kw),
                             old_dir / f"run_{rule}")
        snaps_new, snaps_old = sorted((new / "snapshots").iterdir()), sorted((old / "snapshots").iterdir())
        assert len(snaps_new) == len(snaps_old) == 5
        for a, b in zip(snaps_new, snaps_old):
            assert filecmp.cmp(a, b, shallow=False), f"{rule}: {a.name} differs"
        hist = lambda d: [json.loads(line) for line in (d / "history.jsonl").read_text().splitlines()]  # noqa: E731
        assert [h["compliance"] for h in hist(new)] == [h["compliance"] for h in hist(old)]
        for name in ("final.stl", "preview.bin"):
            assert filecmp.cmp(new / "result" / name, old / "result" / name, shallow=False), f"{rule}: {name}"


def test_every_frozen_consumer_sees_solid_as_load(tmp_path):
    """The model counts solid voxels as full stiffness, density 255 and no design variable."""
    prob = build_problem(load_config(write_case(tmp_path, solids=[CAP])))
    m = Model(prob)
    assert m.n_design == int((prob.labels == DESIGN).sum())
    assert m.solid_pos.size == int(((prob.labels == FIXED) | (prob.labels == LOAD)).sum())
    u8 = m.density_bytes(np.zeros(m.n_design))
    assert (u8[prob.labels.reshape(-1) == LOAD] == 255).all()
    m.solver.free()


def test_check_data_reports_solid(tmp_path, capsys):
    p = write_case(tmp_path, solids=[CAP, STUB], keep_out=[HOLE])
    d = check_data(p)
    assert d["solid_regions"]["voxels"] > 0 and d["solid_regions"]["in_keep_out"] > 0
    assert d["voxel_counts"]["load"] > d["solid_regions"]["voxels"] - 1     # the label LOAD counts solid voxels
    assert d["forced_solid_voxels"] == d["voxel_counts"]["fixed"] + d["voxel_counts"]["load"]
    assert main(["check", str(p)]) == 0
    assert "solid regions:" in capsys.readouterr().out
    # No solid: no key content, no line.
    plain = make_case(sub(tmp_path, "plain"), h=2.0)
    assert check_data(plain)["solid_regions"] is None


def test_small_solid_region_warns_in_check(tmp_path):
    d = check_data(write_case(tmp_path, solids=[((8, 4, 0), (10, 6, 2))]))
    assert any("solid region 'solid0' is small or thin" in w for w in d["warnings"])


def test_run_json_lists_solid_and_the_render_draws_it(tmp_path):
    p = write_case(tmp_path, solids=[CAP], iterations=2)
    assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet"]) == 0
    run_json = json.loads((tmp_path / "r" / "run.json").read_text())
    assert run_json["regions"]["solid"] == ["solid0.stl"]
    assert "load" in run_json["regions"] and "solid0.stl" not in run_json["regions"]["load"].values()
    from topopt import render_input
    items, missing, _ = render_input._run_inputs(tmp_path / "r")
    assert ("load", "solid0.stl") in [(role, Path(path).name) for path, role in items] and not missing
    assert "solid regions:" in (tmp_path / "r" / "run.log").read_text()


# --- the filter rule ---------------------------------------------------------


def gap_labels():
    """Two fixed plates of 1 voxel with a design gap of 1 voxel between them (x direction), 7 x 7 across.

    This is the case of the KB page pitfalls/filter-solid-bias (grid 3 x 7 x 7).
    """
    lab = np.full((3, 7, 7), DESIGN, dtype=np.uint8)
    lab[0] = FIXED
    lab[2] = FIXED
    return lab


def test_default_rule_is_the_filter_of_7555b19_bit_for_bit(tmp_path):
    """The default filter gives the same bytes as the filter before A3a (origin/main 7555b19).

    `tests/data/filter_default_reference.json` holds sha256 hashes of F.data, F.indices, F.indptr, FT and b,
    made with the code of 7555b19 by `tools/make_filter_reference.py` (FIXED, LOAD, KEEP_OUT and EMPTY voxels,
    radius 1.5 and 3.75 voxels). The default argument and "density_one" must both reproduce them.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_filter_reference", REPO / "tools" / "make_filter_reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    want = json.loads((Path(__file__).parent / "data" / "filter_default_reference.json").read_text())["cases"]
    assert [c["radius_voxels"] for c in want] == list(ref.R_VOXELS) == [1.5, 3.75]
    assert ref.digest(build_filter) == want
    assert ref.digest(lambda lab, h, r: build_filter(lab, h, r, "density_one")) == want
    assert want[0]["nnz"] > 0 and want[1]["nnz"] > want[0]["nnz"]
    # "ignore" is a different filter: the same grid gives other bytes.
    assert ref.digest(lambda lab, h, r: build_filter(lab, h, r, "ignore")) != want
    with pytest.raises(ValueError, match="density_one"):
        build_filter(ref.case_labels(), ref.H, 3.0, "zero")


def test_filter_equals_pipeline_fixture():
    """The demo filter equals the pipeline filter on a uniform grid with no solid neighbour (rel 1e-12).

    The fixture is made by tools/make_filter_fixture.py with the real `build_filter` of the pipeline
    (earlier pipeline, revision d43bcf3). Both rules give the same matrix here, because no solid voxel is near.
    """
    fx = json.loads(FIXTURE.read_text())
    assert len(fx["cases"]) >= 3
    for c in fx["cases"]:
        ref = sp.csr_matrix((c["values"], (c["rows"], c["cols"])), shape=(c["n"], c["n"]))
        labels = np.full(c["shape"], DESIGN, dtype=np.uint8)
        for rule in ("density_one", "ignore"):
            F, FT, b = build_filter(labels, c["h"], c["radius"], rule)
            assert F.shape == ref.shape and F.nnz == ref.nnz                    # the same stencil
            assert (F.indices == ref.indices).all() and (F.indptr == ref.indptr).all()
            np.testing.assert_allclose(F.data, ref.data, rtol=1e-12, atol=0)
            assert not b.any() and (FT != F.T).nnz == 0


def test_ignore_rule_b_is_zero_and_rows_sum_to_one():
    lab = gap_labels()
    F, FT, b = build_filter(lab, 1.0, 3.0, "ignore")
    assert not b.any()
    np.testing.assert_allclose(np.asarray(F.sum(axis=1)).ravel(), 1.0, rtol=0, atol=1e-12)
    F1, _, b1 = build_filter(lab, 1.0, 3.0, "density_one")
    assert b1.max() > 0.2                                                     # the bias that `ignore` removes
    np.testing.assert_allclose(np.asarray(F1.sum(axis=1)).ravel() + b1, 1.0, rtol=0, atol=1e-12)
    # A design voxel next to the plate: with `ignore` its row holds design voxels only.
    flat = lab.reshape(-1)
    design = np.flatnonzero(flat == DESIGN)
    next_to_plate = np.flatnonzero((np.indices(lab.shape)[0].reshape(-1)[design] == 4))
    assert (b1[next_to_plate] > 0).all() and (b[next_to_plate] == 0).all()


def test_ignore_rule_keeps_a_gap_open_and_density_one_fills_it():
    """KB pitfall filter-solid-bias: a 1-voxel gap with radius 3 voxels has a filtered density of 0.597 and a
    projected density of 0.826 at beta 8 (grid 3 x 7 x 7, plates of 1 voxel), although the design variable is 0."""
    lab = gap_labels()
    x = np.zeros((lab == DESIGN).sum())
    ijk = np.argwhere(lab == DESIGN)
    mid = np.flatnonzero((ijk[:, 1] == 3) & (ijk[:, 2] == 3))   # the voxel in the centre
    F, _, b = build_filter(lab, 1.0, 3.0, "density_one")
    rho_tilde = F @ x + b
    rho = heaviside_project(rho_tilde, 8.0, 0.5)
    assert rho_tilde[mid[0]] == pytest.approx(0.597, abs=0.002)      # the numbers of the KB page reproduce
    assert rho[mid[0]] == pytest.approx(0.826, abs=0.002)
    F, _, b = build_filter(lab, 1.0, 3.0, "ignore")
    assert (F @ x + b).max() == 0.0 and heaviside_project(F @ x + b, 8.0, 0.5).max() < 1e-9   # no design density


@pytest.mark.parametrize("rule", ["density_one", "ignore"])
def test_gradients_match_finite_differences_at_two_betas(tmp_path, rule):
    """FD check of the whole chain (filter, projection, SIMP, solve) in both rules, at beta 2 and 8.

    The error of a central difference falls with the step (truncation) and then rises (noise of the
    solves). The best step of a design variable gives the error. Measured on Linux with PARDISO and
    with AMG at rtol 1e-13: the worst best-step error is 4e-7 (beta 2) and 2e-7 (beta 8), the noise at
    the step 1e-6 is up to 7e-6. The limit 5e-6 is above the best-step error by a factor 10 or more.
    This test guards the chain rule: the gradient matches the model that the code evaluates. It does NOT
    guard the filter definition. A mutation probe (review of A3a) found that a wrong `b` (worst best-step
    error 8e-8) or rows that sum to 0.8 (2.5e-7) still pass, because the model and its gradient use the
    same wrong filter. Only an `FT` that does not match `F` fails (0.25). The filter definition is guarded
    by `test_ignore_rule_b_is_zero_and_rows_sum_to_one`, the pipeline fixture and the 7555b19 reference.
    """
    p = write_case(tmp_path, solids=[CAP], filter_solid_neighbours=rule, filter_radius=3.0)
    m = Model(build_problem(load_config(p)))
    if m.solver_name == "amg":      # AMG-CG is not exact at its default tolerance: tighten it for the check
        m.solver.rtol = 1e-13
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 0.7, m.n_design)
    sample = rng.choice(m.n_design, size=12, replace=False)
    for beta in (2.0, 8.0):
        _c, dc, _v, dv = m.evaluate(x, beta)
        best = np.full((2, sample.size), np.inf)
        for eps in (1e-3, 1e-4, 1e-5):
            for k, e in enumerate(sample):
                xp, xm = x.copy(), x.copy()
                xp[e] += eps
                xm[e] -= eps
                cp, _, vp, _ = m.evaluate(xp, beta)
                cm, _, vm, _ = m.evaluate(xm, beta)
                err = [abs((cp - cm) / (2 * eps) - dc[e]) / abs(dc[e]),
                       abs((vp - vm) / (2 * eps) - dv[e]) / abs(dv[e])]
                best[:, k] = np.minimum(best[:, k], err)
        assert best.max() < 5e-6, f"beta {beta}: best-step error {best.max():.2e}"
    m.solver.free()


def test_ignore_rule_changes_the_model_filter_and_only_that(tmp_path):
    p = write_case(tmp_path, solids=[CAP], filter_solid_neighbours="ignore", filter_radius=3.0)
    m = Model(build_problem(load_config(p)))
    assert not m.b.any() and np.allclose(m.F @ np.ones(m.n_design), 1.0, atol=1e-12)      # each row sums to 1
    rho_tilde, _ = m.density(np.full(m.n_design, 0.3), 2.0)
    assert np.allclose(rho_tilde, 0.3, atol=1e-12)        # a uniform design stays uniform: no solid pull
    m.solver.free()
    m2 = Model(build_problem(load_config(write_case(sub(tmp_path, "d"), solids=[CAP], filter_radius=3.0))))
    rho_tilde2, _ = m2.density(np.full(m2.n_design, 0.3), 2.0)
    assert rho_tilde2.max() > 0.3 + 0.05                  # density_one pulls voxels next to the cap upward
    m2.solver.free()


def test_config_rejects_a_bad_rule(tmp_path):
    p = make_case(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["settings"]["filter_solid_neighbours"] = "zero"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError, match="filter_solid_neighbours.*density_one or ignore"):
        load_config(p)


# --- the minimum member size -------------------------------------------------


def with_settings(tmp_path, **settings):
    p = make_case(tmp_path, h=2.0, iterations=3)
    d = yaml.safe_load(p.read_text())
    d["settings"].update(settings)
    p.write_text(yaml.safe_dump(d))
    return p


def test_min_member_derives_the_filter_radius(tmp_path):
    s = load_config(with_settings(tmp_path, min_member_mm=5.0)).settings
    assert s.min_member_mm == 5.0 and s.filter_radius_source == "min_member_mm"
    assert s.filter_radius == members.radius_from_min_member(5.0) == members.RADIUS_PER_MEMBER * 5.0


def test_explicit_filter_radius_wins_and_the_conflict_is_a_fact(tmp_path):
    p = with_settings(tmp_path, min_member_mm=5.0, filter_radius=3.0)
    s = load_config(p).settings
    assert s.filter_radius == 3.0 and s.filter_radius_source == "settings" and s.min_member_mm == 5.0
    d = check_data(p)
    fact = [w for w in d["warnings"] if "settings.filter_radius" in w and "is set, so it is used" in w]
    assert len(fact) == 1 and "Options:" in fact[0] and f"{members.radius_from_min_member(5.0):.3g}" in fact[0]
    assert d["derived"]["filter"]["radius_mm"] == 3.0 and d["derived"]["filter"]["min_member_radius_mm"] == 5.0 * members.RADIUS_PER_MEMBER
    # The same radius from both keys: no conflict.
    same = with_settings(sub(tmp_path, "s"), min_member_mm=5.0, filter_radius=members.radius_from_min_member(5.0))
    assert not [w for w in check_data(same)["warnings"] if "is set, so it is used" in w]


def test_min_member_below_two_voxels_is_a_fact_with_options_not_an_error(tmp_path):
    p = with_settings(tmp_path, min_member_mm=3.0)             # 1.5 voxels at 2 mm
    d = check_data(p)                                          # no exception
    fact = [w for w in d["warnings"] if w.startswith("min_member_mm 3 mm is 1.5 voxels")]
    assert len(fact) == 1
    assert "Options:" in fact[0] and "voxel_size 1 mm or less" in fact[0]
    ok = check_data(with_settings(sub(tmp_path, "ok"), min_member_mm=4.0))   # exactly 2 voxels: no fact
    assert not [w for w in ok["warnings"] if w.startswith("min_member_mm")]


@pytest.mark.parametrize("bad", [0, -1.0, "four", True, float("inf")])
def test_min_member_must_be_a_positive_number(tmp_path, bad):
    with pytest.raises(UserError, match="min_member_mm"):
        load_config(with_settings(tmp_path, min_member_mm=bad))


def test_check_prints_request_derived_radius_and_the_estimate_label(tmp_path, capsys):
    assert main(["check", str(with_settings(tmp_path, min_member_mm=6.0))]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("filter:"))
    assert "min_member_mm 6 mm" in line and f"{members.radius_from_min_member(6.0):.3g} mm" in line
    assert "estimate, not a guarantee" in line and members.RULE_TEXT in line
    assert "min feature:" in out


def test_run_output_and_metrics_agree(tmp_path):
    """The requested value, the derived radius and `min_feature_mm` are the same in the log, run.json and metrics.json."""
    p = with_settings(tmp_path, min_member_mm=6.0)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    run_json = json.loads((out / "run.json").read_text())
    metrics = json.loads((out / "result" / "metrics.json").read_text())
    f = metrics["filter"]
    assert f == run_json["derived"]["filter"]
    assert f["min_member_mm"] == 6.0
    assert f["min_member_radius_mm"] == f["radius_mm"] == members.radius_from_min_member(6.0)
    assert f["min_feature_mm"] == run_json["derived"]["min_feature_mm"] == max(2.0, f["radius_mm"])
    assert f["min_member_note"].startswith("estimate, not a guarantee")
    log = (out / "run.log").read_text()
    assert "min_member_mm 6 mm: derived filter radius" in log and "estimate, not a guarantee" in log


def test_metrics_without_min_member_still_show_the_filter_facts(tmp_path):
    out = tmp_path / "r"
    assert main(["run", str(make_case(tmp_path, h=2.0, iterations=2)), "--out", str(out), "--quiet"]) == 0
    f = json.loads((out / "result" / "metrics.json").read_text())["filter"]
    assert f["min_member_mm"] is None and f["radius_source"] == "default" and f["radius_mm"] == 3.0
    assert f["solid_neighbours"] == "density_one" and f["min_feature_mm"] == 3.0


def test_filter_size_estimate_and_the_ram_warning(tmp_path, monkeypatch):
    """The memory of the filter is a fact with an option. It does not stop `check`."""
    sz = members.filter_size(6.0, 2.0, 1000, (40, 20, 10))
    assert sz["entries_per_row"] == members.stencil_count(3.0) == 93        # KB methods/density-filter: R = 3
    assert sz["method"] == "fft" and sz["design_voxels"] == 1000 and "entries" not in sz
    p = with_settings(tmp_path, min_member_mm=6.0)
    from topopt import cli
    assert not [w for w in check_data(p)["warnings"] if w.startswith("the filter needs")]
    monkeypatch.setattr(cli, "physical_ram_bytes", lambda: 1000)
    d = check_data(p)
    warn = [w for w in d["warnings"] if w.startswith("the filter needs")]
    assert len(warn) == 1 and "Option: a larger voxel_size" in warn[0] and "padded grid" in warn[0]


# --- the standalone comparison script ------------------------------------------


def test_solid_vs_load_script_reports_identical(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "solid_vs_load", Path(__file__).resolve().parents[1] / "tools" / "solid_vs_load.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    p = write_case(tmp_path, solids=[CAP, RIB, STUB], keep_out=[HOLE], filter_radius=3.0)
    assert script.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "solid regions:" in out and "identical, labels: True" in out and "identical, filter ignore: True" in out
    assert "False" not in out
    assert script.main([str(tmp_path / "missing.yaml")]) == 1


# --- review fixes (round 1): min feature meaning, filter memory, solid pull -----------------


def test_min_feature_text_says_what_the_request_gets(tmp_path, capsys):
    """M2: the `min feature` fact and the `min_member_mm` rule say the same thing. With a request of 4 mm a wall
    of 4 mm survives (0.5328 R) and a rod needs 1.76 x 4 mm. The old text said that members under R are not resolved."""
    p = with_settings(tmp_path, min_member_mm=4.0)
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("min feature:"))
    assert "thinner members are not resolved" not in out
    assert "walls of about 4 mm survive" in line and "rods need about 7.03 mm" in line and "1.76 x min_member_mm" in line
    assert "free face" in line and "about 7.508 mm" in line                       # the number stays for old readers
    # The meaning agrees with the numbers: a wall of the request is at the survival limit of the radius.
    R = load_config(p).settings.filter_radius
    assert members.WALL_CRITICAL * R == pytest.approx(4.0, rel=0.005)
    assert members.ROD_CRITICAL * R == pytest.approx(1.76 * 4.0, rel=0.005)
    # check --json, run.json, run.log and metrics.json carry the same text.
    d = check_data(p)
    note = d["derived"]["filter"]["min_feature_note"]
    assert d["derived"]["min_feature_note"] == note and "walls of about 4 mm survive" in note
    o = tmp_path / "r"
    assert main(["run", str(p), "--out", str(o), "--quiet"]) == 0
    assert json.loads((o / "result" / "metrics.json").read_text())["filter"]["min_feature_note"] == note
    assert json.loads((o / "run.json").read_text())["derived"]["min_feature_note"] == note
    log = (o / "run.log").read_text()
    assert "min feature about 7.508 mm" in log and note in log


def test_min_feature_text_without_a_request_or_with_an_explicit_radius(tmp_path):
    """Without a request, and when an explicit radius wins, the text gives the sizes that follow from R (no claim
    about the request). The default radius is 3 mm here."""
    default = check_data(with_settings(tmp_path))["derived"]["filter"]
    assert "walls survive from about 1.6 mm and rods from about 2.81 mm" in default["min_feature_note"]
    wins = check_data(with_settings(sub(tmp_path, "w"), min_member_mm=4.0, filter_radius=3.0))["derived"]["filter"]
    assert "walls of about 4 mm survive" not in wins["min_feature_note"]
    assert "walls survive from about 1.6 mm" in wins["min_feature_note"]
    assert "free face" in default["min_feature_note"]


def test_estimate_note_names_the_free_face_skins():
    assert "skin at a free face" in members.ESTIMATE_NOTE and "estimate, not a guarantee" in members.ESTIMATE_NOTE


def test_memory_estimate_has_the_filter(tmp_path):
    """M4, #333: the estimate is grid + factor + the FFT filter. The filter follows its padded grid (56 B a cell) and its design voxels
    (24 B each), not the radius cubed."""
    from topopt.cli import BASE_BYTES, GRID_BYTES_PER_VOXEL, FACTOR_BYTES_PER_DOF43, estimate_memory_bytes
    from topopt.filter import fft_shape
    g, f = 1000, 3000
    old = int(BASE_BYTES + GRID_BYTES_PER_VOXEL * g + FACTOR_BYTES_PER_DOF43 * f ** (4 / 3))
    assert estimate_memory_bytes(g, f) == old                       # without a filter: the old number
    assert estimate_memory_bytes(g, f, 123_456) == old + 123_456     # the filter next to the factor
    sz = members.filter_size(6.0, 2.0, 1000, (40, 20, 10))
    assert sz["fft_shape"] == list(fft_shape((40, 20, 10), 3)) and all(s >= n + 6 for s, n in zip(sz["fft_shape"], (40, 20, 10)))
    cells = sz["fft_shape"][0] * sz["fft_shape"][1] * sz["fft_shape"][2]
    assert sz["fft_cells"] == cells
    assert sz["held_bytes_estimate"] == sz["build_peak_bytes_estimate"] == int(cells * members.FILTER_BYTES_PER_PADDED_CELL
                                                                               + 1000 * members.FILTER_BYTES_PER_DESIGN_VOXEL)
    # The arrays of design size come on top of the padded grid: a grid that is almost all design measured above 56 B a padded cell.
    more = members.filter_size(6.0, 2.0, 2000, (40, 20, 10))
    assert more["fft_cells"] == cells and more["held_bytes_estimate"] - sz["held_bytes_estimate"] == 1000 * 24
    # A radius twice as large grows the matrix 8 times; the FFT filter only gets a wider pad.
    small, big = (members.filter_size(r, 2.0, 1000, (200, 100, 50)) for r in (6.0, 12.0))
    assert big["entries_per_row"] > 7 * small["entries_per_row"] and big["held_bytes_estimate"] < 1.3 * small["held_bytes_estimate"]
    assert members.size_text(37.2 * 2 ** 20) == "37.2 MiB" and members.size_text(2.5 * 2 ** 30) == "2.5 GiB"


def memory_case(tmp_path, monkeypatch, margin, **settings):
    """check_data of a case on a machine whose RAM is `margin` bytes above the estimate without the filter."""
    from topopt import cli
    p = with_settings(tmp_path, **settings)
    d = check_data(p)
    nf = cli.estimate_memory_bytes(d["grid"]["voxels"], d["free_dofs"])
    monkeypatch.setattr(cli, "physical_ram_bytes", lambda: nf + margin)
    return p, nf


def test_filter_memory_is_in_check_and_in_the_text(tmp_path, monkeypatch, capsys):
    """M4: a machine that holds the grid and the factor but not the filter gets the warning in JSON and in the text."""
    p, nf = memory_case(tmp_path, monkeypatch, 1, min_member_mm=6.0)
    d = check_data(p)
    assert d["memory_estimate"]["bytes"] > nf and d["memory_estimate"]["filter_held_bytes"] > 0
    assert d["memory_estimate"]["filter_build_peak_bytes"] == d["memory_estimate"]["filter_held_bytes"]    # one model (#333)
    assert d["memory_estimate"]["bytes"] == nf + d["memory_estimate"]["filter_held_bytes"]
    memory = [w for w in d["warnings"] if w.startswith("the memory estimate")]
    filt = [w for w in d["warnings"] if w.startswith("the filter needs")]
    assert len(memory) == 1 and len(filt) == 1 and "Option: a larger voxel_size" in filt[0] and "an FFT on a padded grid" in filt[0]
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert "warning: the filter needs about" in out and "warning: the memory estimate is" in out
    assert "the filter is part of it" in out                              # the memory line names it
    assert "filter matrix" not in out


def test_small_filter_does_not_warn_and_a_big_machine_does_not_warn(tmp_path, monkeypatch):
    p, nf = memory_case(tmp_path, monkeypatch, 50_000_000)              # default radius: the filter is small
    assert not [w for w in check_data(p)["warnings"] if w.startswith(("the memory estimate", "the filter needs"))]
    p2, _ = memory_case(sub(tmp_path, "b"), monkeypatch, 50_000_000, min_member_mm=6.0)   # large radius, ample RAM
    assert not [w for w in check_data(p2)["warnings"] if w.startswith(("the memory estimate", "the filter needs"))]


def test_run_log_shows_the_filter_memory_and_the_run_goes_on(tmp_path, monkeypatch):
    """M4: `topopt run` logs the memory estimate and the filter warning at its start. It is a fact: the run finishes."""
    p, _ = memory_case(tmp_path, monkeypatch, 1, min_member_mm=6.0)
    o = tmp_path / "r"
    assert main(["run", str(p), "--out", str(o), "--quiet"]) == 0
    log = (o / "run.log").read_text()
    assert "memory estimate: about" in log and "the filter is part of it" in log
    assert "warning: the filter needs about" in log and "warning: the memory estimate is" in log
    assert (o / "result" / "final.stl").exists()


def test_builder_guard_has_the_filter(tmp_path, monkeypatch):
    """M4: the memory error of the part builder (`check_yaml`) includes the filter too."""
    from topopt.regions_builder import check_yaml
    p, nf = memory_case(tmp_path, monkeypatch, 1, min_member_mm=6.0)
    lines = []
    say = lambda *a, **k: lines.append(" ".join(map(str, a)))  # noqa: E731
    ok, kind = check_yaml(p, say)
    text = "\n".join(lines)
    assert not ok and kind == "memory" and "ERROR: the memory estimate" in text and "with the filter)" in text
    p2, _ = memory_case(sub(tmp_path, "d"), monkeypatch, 50_000_000)          # default radius: no error
    ok2, kind2 = check_yaml(p2, lambda *a, **k: None)
    assert ok2 or kind2 != "memory"


def test_solid_pull_matches_the_filter_matrix(tmp_path):
    """Minor 8: the count of design voxels with b > 0.1 and the largest b equal what `build_filter` gives."""
    from topopt.filter import solid_pull, solid_pull_line
    prob = build_problem(load_config(write_case(tmp_path, solids=[CAP, RIB], keep_out=[HOLE], filter_radius=5.0)))
    for R in (3.0, 5.0, 7.0):
        _, _, b = build_filter(prob.labels, prob.h, R, "density_one")
        sp_ = solid_pull(prob.labels, prob.h, R, "density_one")
        assert sp_["design_voxels"] == b.size and sp_["voxels_above"] == int((b > 0.1).sum()) > 0
        assert sp_["b_max"] == pytest.approx(float(b.max()), abs=1e-4)
        assert sp_["share_above"] == pytest.approx((b > 0.1).mean(), abs=1e-4)
    ign = solid_pull(prob.labels, prob.h, 5.0, "ignore")
    assert ign["voxels_above"] == 0 and ign["b_max"] == 0.0
    assert "none (rule ignore" in solid_pull_line(ign)
    assert solid_pull_line(solid_pull(np.full((4, 4, 4), DESIGN, np.uint8), 1.0, 2.0)) is None   # no solid voxel
    line = solid_pull_line(solid_pull(prob.labels, prob.h, 5.0))
    assert "design voxels" in line and "have a filtered density above 0.1" in line and "ignore removes this pull" in line


def test_solid_pull_line_is_in_check_and_run_log(tmp_path, capsys):
    p = write_case(tmp_path, solids=[CAP], filter_radius=5.0)
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("solid pull:"))
    assert check_data(p)["solid_pull"]["voxels_above"] > 0 and "of" in line
    o = tmp_path / "r"
    assert main(["run", str(p), "--out", str(o), "--quiet"]) == 0
    assert line in (o / "run.log").read_text()
    ignore = write_case(sub(tmp_path, "i"), solids=[CAP], filter_radius=5.0, filter_solid_neighbours="ignore")
    assert main(["check", str(ignore)]) == 0
    assert "solid pull: none (rule ignore" in capsys.readouterr().out


def test_solid_facts_count_the_cuts_that_only_the_labels_hold(tmp_path):
    """A cut with no region mask (an interface cutter of a later merge) removes solid voxels: the facts count it."""
    from topopt.voxelize import solid_facts

    cfg = load_config(write_case(tmp_path, solids=[CAP]))
    prob = build_problem(cfg)
    cap = inside(prob, *CAP)
    labels, ids = prob.labels.copy(), prob.load_region_ids.copy()
    masks = {"design": inside(prob, (0, 0, 0), (20, 10, 10)), "solid[0] (solid0.stl)": cap}
    base = solid_facts(cfg, masks, labels, ids)
    assert base["voxels"] == int(cap.sum()) > 4 and base["in_keep_out"] == 0
    cut = cap & (np.indices(cap.shape)[0] == np.argwhere(cap)[:, 0].min())     # one x-layer of the cap
    labels[cut], ids[cut] = KEEP_OUT, -1
    after = solid_facts(cfg, masks, labels, ids)
    assert after["in_keep_out"] == int(cut.sum()) > 0
    assert after["voxels"] == base["voxels"] - int(cut.sum()) and after["stl_voxels"] == base["stl_voxels"]
    assert after["stl_voxels"] == after["voxels"] + after["in_fixed"] + after["in_load"] + after["in_keep_out"]


def test_solid_facts_use_the_singular_for_one_voxel():
    s = {"voxels": 1, "stl_voxels": 3, "in_fixed": 1, "in_load": 0, "in_keep_out": 2, "in_two_solid_stls": 0,
         "outside_design_stl": 1}
    text = solid_lines(s)[0]
    assert "1 voxel stays solid" in text and "1 voxel lies in a fixed region (it is FIXED)" in text
    assert "2 voxels lie in a keep-out region (they are removed)" in text
    assert "1 voxel lies outside the design STL (it adds material to the part)" in text and "1 voxels" not in text
