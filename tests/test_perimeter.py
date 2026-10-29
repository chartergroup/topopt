"""Tests for the perimeter bound `settings.perimeter_max_mm2` (A4c, CONTRACTS.md section 26.3).

Tolerances (each one is stated at the test that uses it; the values are measured):
- Exact face counts (6 h^2, 10 h^2, the border, the solid rule): EXACT, round-off of a few sums of 1.
- The gradient of P against central differences of P (P has no solve, so there is no noise floor): FD_RHO on rho,
  FD_CHAIN through the filter, the projection, the robust mode and the mirror average (best of three steps, relative to
  the larger of |gradient| and 1e-3 times the largest |gradient|: the error scale of A1).
- The minimum cut against the brute-force minimum over all binary designs: EXACT.
"""

import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from helpers import needs_accelerate, needs_pardiso
from test_checkpoint import needs_direct
from test_mirror import two_planes, write_yaml
from test_optimize import make_case
from topopt import perimeter as pm
from topopt import symmetry
from topopt.config import UserError, load_config
from topopt.formulation import NOT_MET
from topopt.optimize import Model, formulation_from_config
from topopt.problem import build_problem
from topopt.voxelize import DESIGN, EMPTY, FIXED, KEEP_OUT, LOAD

EXACT = 1e-12
ROOT_EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "hook"
FD_RHO = 1e-6       # P(rho): central difference with step 1e-6; measured worst 5.5e-8 (4 random label arrays, 40 components)
FD_CHAIN = 1e-5     # dP/dx through mirror, filter and projection: the KB bar at the best step; measured worst 1.3e-7


def grid(shape, **cells):
    """A label array of EMPTY voxels with the given cells: `grid((5, 5, 5), design=[(2, 2, 2)], fixed=[(3, 2, 2)])`."""
    lab = np.full(shape, EMPTY, dtype=np.uint8)
    for kind, label in (("design", DESIGN), ("fixed", FIXED), ("load", LOAD), ("keep_out", KEEP_OUT)):
        for c in cells.get(kind, []):
            lab[c] = label
    return lab


def dense_reference(labels, rho, h, eps):
    """P from the full density array and `np.diff` on the padded grid: no face list, an independent code path."""
    lab = np.asarray(labels)
    full = np.zeros(lab.shape)
    full[(lab == FIXED) | (lab == LOAD)] = 1.0
    full[lab == DESIGN] = rho            # C order: the order of x
    padded = np.pad(full, 1)             # outside the grid: density 0
    norm = np.sqrt(1.0 + eps ** 2) - eps
    total = 0.0
    for ax in range(3):
        d = np.diff(padded, axis=ax)
        total += float(((np.sqrt(d * d + eps ** 2) - eps) / norm).sum())
    return h * h * total


# --- P of small label arrays ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("h", [0.5, 2.0])
def test_one_design_voxel_in_void_is_6h2(h):
    f = pm.FaceList(grid((5, 5, 5), design=[(2, 2, 2)]), h)
    assert f.value(np.array([1.0])) == pytest.approx(6 * h * h, rel=EXACT)
    assert f.value(np.array([0.0])) == 0.0
    assert f.fixed_part_mm2 == 0.0


def test_two_neighbouring_design_voxels_are_10h2():
    f = pm.FaceList(grid((5, 5, 5), design=[(2, 2, 2), (3, 2, 2)]), 2.0)
    assert f.value(np.ones(2)) == pytest.approx(10 * 4.0, rel=EXACT)
    assert f.value(np.array([1.0, 0.0])) == pytest.approx(6 * 4.0, rel=EXACT)        # the shared face is a jump again


def test_faces_to_the_border_keep_out_and_empty_voxels_count_as_void():
    h = 1.5
    one = pm.FaceList(grid((1, 1, 1), design=[(0, 0, 0)]), h)                      # six faces to the outside of the grid
    assert one.value(np.array([1.0])) == pytest.approx(6 * h * h, rel=EXACT)
    around = pm.FaceList(grid((3, 3, 3), design=[(1, 1, 1)], keep_out=[(0, 1, 1), (1, 0, 1), (1, 1, 0)]), h)
    assert around.value(np.array([1.0])) == pytest.approx(6 * h * h, rel=EXACT)    # KEEP_OUT and EMPTY: both void


@pytest.mark.parametrize("kind", ["fixed", "load"])
def test_a_face_to_a_solid_voxel_has_the_other_side_at_density_one(kind):
    h = 2.0
    f = pm.FaceList(grid((5, 5, 5), design=[(2, 2, 2)], **{kind: [(3, 2, 2)]}), h)
    assert f.fixed_part_mm2 == pytest.approx(5 * h * h, rel=EXACT)                  # the solid voxel: 5 faces to void
    assert f.value(np.array([1.0])) == pytest.approx(5 * h * h + 5 * h * h, rel=EXACT)   # design 1: 5 to void, no jump to the solid
    assert f.value(np.array([0.0])) == pytest.approx(5 * h * h + h * h, rel=EXACT)       # design 0: the face to the solid is a jump
    assert f.value(np.array([0.5])) < f.value(np.array([1.0]))
    assert f.n_design_solid_faces == 1 and f.n_design_void_faces == 5


def test_a_solid_voxel_at_the_grid_border_counts_its_outer_face():
    h = 2.0
    f = pm.FaceList(grid((3, 3, 3), fixed=[(0, 1, 1)], design=[(1, 1, 1)]), h)
    assert f.n_const_faces == 5                                                     # 4 to void + 1 to the outside
    assert f.fixed_part_mm2 == pytest.approx(5 * h * h, rel=EXACT)


def test_the_normaliser_makes_a_binary_jump_exactly_one_face_for_any_eps():
    for eps in (1e-3, 1e-2, 1e-1):
        f = pm.FaceList(grid((3, 3, 3), design=[(1, 1, 1)]), 1.0, eps=eps)
        assert f.value(np.array([1.0])) == pytest.approx(6.0, rel=EXACT)


@pytest.mark.parametrize("seed", range(4))
def test_value_and_gradient_match_the_dense_reference(seed):
    """Random labels of all five kinds: the face list agrees with the padded-grid code, also at the border."""
    rng = np.random.default_rng(seed)
    lab = rng.choice([EMPTY, DESIGN, FIXED, LOAD, KEEP_OUT], size=(6, 5, 4), p=[0.25, 0.4, 0.15, 0.1, 0.1]).astype(np.uint8)
    h, eps = 1.7, pm.EPS
    f = pm.FaceList(lab, h, eps)
    rho = rng.uniform(0, 1, f.n_design)
    P, g = f.value_grad(rho)
    assert P == pytest.approx(dense_reference(lab, rho, h, eps), rel=EXACT)
    assert f.value(rho) == pytest.approx(P, rel=EXACT)
    step = 1e-6
    for e in rng.choice(f.n_design, size=min(10, f.n_design), replace=False):
        rp, rm = rho.copy(), rho.copy()
        rp[e] += step
        rm[e] -= step
        fd = (dense_reference(lab, rp, h, eps) - dense_reference(lab, rm, h, eps)) / (2 * step)
        assert abs(fd - g[e]) <= FD_RHO * max(abs(g[e]), 1.0)


def test_the_gradient_is_zero_where_all_neighbours_are_equal():
    """The term is smooth at d = 0: an interior voxel with the density of all its neighbours has the slope 0."""
    lab = np.full((5, 5, 5), DESIGN, dtype=np.uint8)
    f = pm.FaceList(lab, 1.0)
    _P, g = f.value_grad(np.full(f.n_design, 0.4))
    centre = np.flatnonzero(lab.reshape(-1) == DESIGN).tolist().index((2 * 5 + 2) * 5 + 2)
    assert g[centre] == 0.0


def test_a_gray_design_has_no_more_perimeter_than_its_exact_total_variation():
    rng = np.random.default_rng(5)
    lab = rng.choice([EMPTY, DESIGN, FIXED], size=(6, 6, 5), p=[0.2, 0.7, 0.1]).astype(np.uint8)
    f = pm.FaceList(lab, 1.0)
    rho = rng.uniform(0, 1, f.n_design)
    assert f.value(rho) <= dense_reference(lab, rho, 1.0, 1e-12) + 1e-9               # eps -> 0 is the exact total variation


# --- the minimum cut --------------------------------------------------------------------------------------------


def brute_force_minimum(f):
    return min(f.value(np.array(bits, dtype=float)) for bits in itertools.product((0.0, 1.0), repeat=f.n_design))


@pytest.mark.parametrize("seed", range(6))
def test_p_min_is_the_smallest_p_of_all_binary_designs(seed):
    rng = np.random.default_rng(100 + seed)
    while True:
        lab = rng.choice([EMPTY, DESIGN, FIXED, LOAD, KEEP_OUT], size=(4, 3, 3), p=[0.25, 0.3, 0.2, 0.1, 0.15]).astype(np.uint8)
        n = int((lab == DESIGN).sum())
        if 2 <= n <= 11:
            break
    f = pm.FaceList(lab, 1.3)
    assert f.p_min_mm2() == pytest.approx(brute_force_minimum(f), rel=EXACT)


def test_p_min_is_below_the_all_void_design_when_solid_voxels_surround_a_design_voxel():
    """A design voxel with 5 solid faces and 1 void face: the minimum fills it (1 face) and not empties it (5 faces)."""
    h = 2.0
    cells = [(1, 2, 2), (3, 2, 2), (2, 1, 2), (2, 3, 2), (2, 2, 1)]               # five solid neighbours; (2, 2, 3) stays empty
    f = pm.FaceList(grid((5, 5, 5), design=[(2, 2, 2)], fixed=cells), h)
    all_void = f.value(np.array([0.0]))
    assert f.p_min_mm2() < all_void
    assert f.p_min_mm2() == pytest.approx(f.value(np.array([1.0])), rel=EXACT)
    assert f.min_cut_faces() == 1


def test_p_min_without_a_solid_voxel_is_zero_and_without_a_void_face_is_the_fixed_part():
    lone = pm.FaceList(grid((5, 5, 5), design=[(2, 2, 2)]), 1.0)
    assert lone.p_min_mm2() == 0.0                                                  # the empty design
    full = np.full((3, 3, 3), FIXED, dtype=np.uint8)
    full[1, 1, 1] = DESIGN
    boxed = pm.FaceList(full, 1.0)
    assert boxed.min_cut_faces() == 0                                               # the voxel is inside the solid: fill it
    assert boxed.p_min_mm2() == pytest.approx(boxed.fixed_part_mm2, rel=EXACT) == pytest.approx(54.0, rel=EXACT)


def test_the_hook_short_example_has_the_numbers_of_the_research():
    """R9 of the A4c research: fixed part 4640.0 mm2 and p_min 6776.0 mm2 at 2.0 mm (equal to the all-void design)."""
    from pathlib import Path
    from topopt.config import load_config
    from topopt.problem import build_problem
    yaml = Path(__file__).resolve().parent.parent / "examples" / "hook" / "hook_short.yaml"
    prob = build_problem(load_config(yaml))
    f = pm.FaceList(prob.labels, prob.h)
    assert f.fixed_part_mm2 == pytest.approx(4640.0, rel=EXACT)
    assert f.p_min_mm2() == pytest.approx(6776.0, rel=EXACT)
    assert f.p_min_mm2() == pytest.approx(f.value(np.zeros(f.n_design)), rel=EXACT)


# --- the order of the faces is the design order of the model ------------------------------------------------------


def p_of_model_grid(model, rho, eps=pm.EPS):
    """P from the full-grid density that the model's own map gives (`Model.density_bytes` as float, not as bytes)."""
    full = np.zeros(model.problem.labels.size)
    full[model.active_index[model.solid_pos]] = 1.0
    full[model.active_index[model.design_pos]] = rho
    padded = np.pad(full.reshape(model.problem.labels.shape), 1)
    norm = np.sqrt(1.0 + eps ** 2) - eps
    total = sum(float(((np.sqrt(np.diff(padded, axis=ax) ** 2 + eps ** 2) - eps) / norm).sum()) for ax in range(3))
    return model.problem.h ** 2 * total, full


@pytest.mark.parametrize("case", ["cantilever", "hook_short.yaml", "hook_mirror.yaml"])
def test_the_face_list_order_is_the_design_order_of_the_model(tmp_path, case):
    """`FaceList` takes `np.flatnonzero(labels == DESIGN)`; `Model` takes `active_index[design_pos]`. A finite-difference test cannot
    see a permutation of the two, so this test compares the values for one random density (review n1)."""
    path = cantilever(tmp_path) if case == "cantilever" else ROOT_EXAMPLES / case
    m = Model(build_problem(load_config(path)))
    try:
        rho = np.random.default_rng(11).uniform(0.0, 1.0, m.n_design)
        faces = pm.FaceList(m.problem.labels, m.problem.h)
        p_model, full = p_of_model_grid(m, rho)
        assert faces.value(rho) == pytest.approx(p_model, rel=EXACT)
        assert abs(faces.value(rho[::-1]) - p_model) > 1e-6 * p_model                  # the check can see a permutation
        assert np.abs(m.density_bytes(rho).astype(float) / 255 - full).max() <= 0.5 / 255 + 1e-12   # the map is the one of the bytes
    finally:
        m.solver.free()


# --- the gradient through the density chain ---------------------------------------------------------------------


def chain_error(m, beta, seed=3, samples=16, steps=(1e-4, 1e-5, 1e-6)):
    """Worst relative error of dP/dx (`Model.gradient_wrt_x` of the row gradient) against central differences of P(rho(x))."""
    faces = pm.FaceList(m.problem.labels, m.problem.h)
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9, m.n_design)                                           # an asymmetric x
    rho_tilde, rho = m.density(x, beta)
    _P, g = faces.value_grad(rho)
    D = m.gradient_wrt_x([g], ["rho"], rho_tilde, beta)[0]
    scale = 1e-3 * np.abs(D).max()
    best = np.inf
    for step in steps:
        worst = 0.0
        for e in rng.choice(m.n_design, size=samples, replace=False):
            xp, xm = x.copy(), x.copy()
            xp[e] += step
            xm[e] -= step
            fd = (faces.value(m.density(xp, beta)[1]) - faces.value(m.density(xm, beta)[1])) / (2 * step)
            worst = max(worst, abs(fd - D[e]) / max(abs(D[e]), scale))
        best = min(best, worst)
    return best


def set_settings(**kw):
    def edit(path):
        d = yaml.safe_load(path.read_text())
        d["settings"].update(kw)
        path.write_text(yaml.safe_dump(d))
        return path
    return edit


CHAIN_CASES = {
    "filter and projection": lambda tmp: make_case(tmp, h=2.0),
    "robust eroded": lambda tmp: set_settings(robust="eroded", eta_eroded=0.6)(make_case(tmp, h=2.0)),
    "one mirror plane": lambda tmp: write_yaml(tmp, [{"normal": "y", "at_mm": 0.0}]),
    "two mirror planes": lambda tmp: write_yaml(tmp, two_planes()),
}


@pytest.mark.parametrize("beta", [1.0, 2.0, 8.0])
@pytest.mark.parametrize("case", list(CHAIN_CASES))
def test_the_gradient_through_the_chain_matches_finite_differences(tmp_path, case, beta):
    m = Model(build_problem(load_config(CHAIN_CASES[case](tmp_path))))
    assert (m.robust is not None) == (case == "robust eroded")
    assert bool(symmetry.perms_of(m)) == ("mirror" in case)
    assert chain_error(m, beta) < FD_CHAIN
    m.solver.free()


def test_a_symmetric_x_gives_a_symmetric_gradient_with_a_mirror_key(tmp_path):
    m = Model(build_problem(load_config(CHAIN_CASES["two mirror planes"](tmp_path))))
    faces = pm.FaceList(m.problem.labels, m.problem.h)
    x = np.random.default_rng(2).uniform(0.1, 0.9, m.n_design)
    rho_tilde, rho = m.density(x, 2.0)
    D = m.gradient_wrt_x([faces.value_grad(rho)[1]], ["rho"], rho_tilde, 2.0)[0]
    for perm in symmetry.perms_of(m):
        assert np.abs(D - D[perm]).max() <= 1e-12 * np.abs(D).max()                 # the gradient lives on the averaged design
    assert faces.value(rho) == pytest.approx(faces.value(m.density(symmetry.average(m, x), 2.0)[1]), rel=EXACT)
    m.solver.free()


# --- the key ------------------------------------------------------------------------------------------------------

ITER = 40           # passes of the small runs (the cantilever of test_optimize at 2 mm: 249 design voxels, about 1 s)
FEAS_TOL = NOT_MET  # a row that is met ends with g at or below the 0.5 % of the run log


def cantilever(tmp_path, **settings):
    """The cantilever of test_optimize at 2 mm with the given settings added (or replaced)."""
    p = make_case(tmp_path, h=2.0, iterations=ITER)
    d = yaml.safe_load(p.read_text())
    d["settings"].update(settings)
    p.write_text(yaml.safe_dump(d))
    return p


def sub(tmp_path, name):
    folder = tmp_path / name
    folder.mkdir()
    return folder


def edit_yaml(path, fn):
    d = yaml.safe_load(Path(path).read_text())
    fn(d)
    Path(path).write_text(yaml.safe_dump(d))
    return path


@pytest.mark.parametrize("value, expected", [(12000, 12000.0), (825.5, 825.5), ("1.2e4", 12000.0), ("1e4", 10000.0)])
def test_the_key_is_a_number_in_mm2(tmp_path, value, expected):
    """YAML 1.1 loads 1.2e4 and 1e4 as text (they have no dot or no sign); `_num_text` reads them as numbers."""
    p = cantilever(tmp_path)
    p.write_text(p.read_text().replace("settings:\n", f"settings:\n  perimeter_max_mm2: {value}\n"))
    assert load_config(p).settings.perimeter_max_mm2 == expected


def test_without_the_key_there_is_no_bound(tmp_path):
    assert load_config(cantilever(tmp_path)).settings.perimeter_max_mm2 is None


@pytest.mark.parametrize("value", [0, -3.5, "abc", True, None, [5], float("nan"), float("inf")])
def test_an_invalid_value_is_refused_with_the_key_and_the_value(tmp_path, value):
    p = cantilever(tmp_path, perimeter_max_mm2=value)
    with pytest.raises(UserError) as e:
        load_config(p)
    text = str(e.value)
    assert "'settings.perimeter_max_mm2' must be" in text and "(file:" in text and "\n" not in text
    assert f"got {value!r}" in text


# --- the row in the formulation ---------------------------------------------------------------------------------


def names(config, n=10):
    return [c.name for c in formulation_from_config(config, n).constraints]


def test_a_config_without_the_key_has_no_perimeter_row(tmp_path):
    cfg = load_config(cantilever(tmp_path))
    form = formulation_from_config(cfg, 10)
    assert [c.name for c in form.constraints] == ["volume"] and not form.perimeter.active


def test_the_row_comes_last_and_the_volume_ceiling_stays(tmp_path):
    """The volume fallback of a mass run with no other row is made BEFORE the row (R0 item 2 of the research)."""
    assert names(load_config(cantilever(sub(tmp_path, "a"), perimeter_max_mm2=900.0))) == ["volume", "perimeter"]
    lim = cantilever(sub(tmp_path, "b"), perimeter_max_mm2=900.0)
    edit_yaml(lim, lambda d: d["load_cases"][0].update(compliance_limit=0.3))
    assert names(load_config(lim)) == ["volume", "compliance:down", "perimeter"]
    mass = formulation_from_config(load_config(cantilever(sub(tmp_path, "c"), perimeter_max_mm2=900.0, objective="mass")), 10)
    assert [c.name for c in mass.constraints] == ["volume", "perimeter"] and mass.n_rows == 2    # the ceiling is kept
    both = formulation_from_config(load_config(cantilever(sub(tmp_path, "d"), perimeter_max_mm2=900.0, case_combine="max")), 10)
    assert both.n_bound == 2 and both.n_rows == 4


def test_the_row_is_linear_with_the_price_of_the_volume_row(tmp_path):
    from topopt.mma import C_PENALTY
    row = formulation_from_config(load_config(cantilever(tmp_path, perimeter_max_mm2=900.0)), 10).constraints[-1]
    assert (row.name, row.limit, row.scale, row.form, row.price) == ("perimeter", 900.0, 900.0, "linear", C_PENALTY)
    assert row.hint and row.cause and "perimeter" in row.cause


def test_a_mass_run_with_the_key_has_one_fact_that_the_row_holds_no_material(tmp_path):
    from topopt.optimize import problem_facts
    mass = load_config(cantilever(sub(tmp_path, "m"), perimeter_max_mm2=900.0, objective="mass"))
    lines = [t for t in problem_facts(mass) if "perimeter_max_mm2" in t]
    assert len(lines) == 1 and "does not hold material" in lines[0] and lines[0].startswith("fact: ")
    assert "price 1000" in lines[0] and "compliance limit has price 100" in lines[0]            # review m2 (a): the price fact
    assert "a tight bound can leave a compliance limit not met" in lines[0]
    assert not [t for t in problem_facts(load_config(cantilever(sub(tmp_path, "c"), perimeter_max_mm2=900.0))) if "perimeter" in t]
    assert not [t for t in problem_facts(load_config(cantilever(sub(tmp_path, "n"), objective="mass"))) if "perimeter" in t]


# --- the summary line of the run log when a compliance limit is not met (review m2) ---------------------------------

OLD_SUMMARY = ("  the other rows are met, so this limit does not fit at this volume; options: raise settings.volume_fraction, "
               "relax the limit, or use the mass objective (A2)")


def compliance_case(tmp_path, **settings):
    """The cantilever with a compliance limit on the case `down`."""
    p = cantilever(tmp_path, **settings)
    edit_yaml(p, lambda d: d["load_cases"][0].update(compliance_limit=0.22))
    return p


def summary_of(form, p_value, c_value):
    """The summary line of `form.unmet` for compliance `down` not met (value `c_value`) and the perimeter row met."""
    perim = [{"name": "perimeter", "value": p_value, "limit": form.perimeter.bound, "g": p_value / form.perimeter.bound - 1,
              "violation": 0.0}] if form.perimeter.active else []
    comp = {"name": "compliance:down", "value": c_value, "limit": 0.22, "g": c_value / 0.22 - 1, "violation": c_value / 0.22 - 1}
    lines = form.unmet([comp] + perim)
    assert lines[0].startswith("constraint not met: compliance:down")
    return lines, [t for t in lines if t.startswith("  ")]


def test_a_mass_run_names_the_perimeter_row_and_its_option_and_not_the_volume(tmp_path):
    """The case of the review: mass objective, compliance:down not met, perimeter met. No "at this volume", no "mass objective"."""
    cfg = load_config(compliance_case(tmp_path, objective="mass", perimeter_max_mm2=900.0))
    m = Model(build_problem(cfg))
    form = formulation_from_config(cfg, m.n_design, m)
    assert [c.name for c in form.constraints] == ["compliance:down", "perimeter"]
    lines, (summary,) = summary_of(form, 900.0, 0.25)
    m.solver.free()
    assert any(t.startswith("fact: perimeter: P of the start design") and "(met)" in t for t in lines)
    assert summary == ("  the other rows are met, and this limit is not met at the end of the run; the perimeter row has price "
                       "1000 and a compliance limit has price 100, so a tight perimeter bound can be a cause; "
                       "options: raise settings.perimeter_max_mm2, or relax the limit")
    for word in ("at this volume", "mass objective", "volume_fraction", "(A2)"):
        assert word not in summary


def test_a_mass_run_without_the_key_has_a_summary_with_no_volume_and_no_perimeter(tmp_path):
    form = formulation_from_config(load_config(compliance_case(tmp_path, objective="mass")), 10)
    _lines, (summary,) = summary_of(form, 0.0, 0.25)
    assert summary == "  the other rows are met, and this limit is not met at the end of the run; options: relax the limit"


@pytest.mark.parametrize("key", [False, True])
def test_a_compliance_run_keeps_the_text_of_the_volume_for_a_limit_not_met(tmp_path, key):
    """The compliance objective: the text before A4c, with or without the perimeter row (the default run is not touched)."""
    settings = {"perimeter_max_mm2": 900.0} if key else {}
    cfg = load_config(compliance_case(tmp_path, **settings))
    m = Model(build_problem(cfg))
    form = formulation_from_config(cfg, m.n_design, m)
    _lines, (summary,) = summary_of(form, 900.0, 0.25)
    m.solver.free()
    assert summary == OLD_SUMMARY


def test_a_mass_run_with_a_perimeter_row_that_is_not_met_too_keeps_the_rows_own_lines(tmp_path):
    """Both rows not met: the two rows have two hints, so there is no summary line (as before), and no false claim."""
    cfg = load_config(compliance_case(tmp_path, objective="mass", perimeter_max_mm2=900.0))
    m = Model(build_problem(cfg))
    form = formulation_from_config(cfg, m.n_design, m)
    both = [{"name": "compliance:down", "value": 0.25, "limit": 0.22, "g": 0.136, "violation": 0.136},
            {"name": "perimeter", "value": 1000.0, "limit": 900.0, "g": 0.111, "violation": 0.111}]
    lines = form.unmet(both)
    m.solver.free()
    assert [t for t in lines if t.startswith("constraint not met")] and not [t for t in lines if t.startswith("  ")]


# --- the refusal of a bound below the smallest possible P, and the facts of `check` -------------------------------


def p_numbers(path):
    """`(fixed part, p_min)` of a YAML, from its own voxelization."""
    prob = build_problem(load_config(path))
    f = pm.FaceList(prob.labels, prob.h)
    return f.fixed_part_mm2, f.p_min_mm2()


def test_a_bound_below_p_min_is_refused_with_the_facts_and_the_options(tmp_path):
    from topopt.cli import check_data
    fixed, p_min = p_numbers(cantilever(sub(tmp_path, "a")))
    assert p_min > fixed > 0
    p = cantilever(sub(tmp_path, "b"), perimeter_max_mm2=0.5 * p_min)
    with pytest.raises(UserError) as e:
        check_data(p)
    text = str(e.value)
    assert "\n" not in text and f"{0.5 * p_min:g} mm2" in text and f"{pm.shown_p_min(p_min)} mm2" in text and f"{fixed:.1f} mm2" in text
    assert "minimum cut" in text and "Options:" in text and "raise settings.perimeter_max_mm2" in text
    assert "reduce the solid regions" in text and "make the design space bigger" in text and "(file:" in text
    assert "voxel face area" in text                                                  # the unit


def test_a_bound_between_the_fixed_part_and_p_min_is_refused_too(tmp_path):
    """The check of the fixed part alone (the first answer) lets this bound pass; no design can meet it."""
    from topopt.cli import check_data
    fixed, p_min = p_numbers(cantilever(sub(tmp_path, "a")))
    with pytest.raises(UserError):
        check_data(cantilever(sub(tmp_path, "b"), perimeter_max_mm2=0.5 * (fixed + p_min)))


def test_a_bound_at_or_above_p_min_passes_the_check(tmp_path):
    from topopt.cli import check_data
    _fixed, p_min = p_numbers(cantilever(sub(tmp_path, "a")))
    for k, bound in enumerate((p_min, p_min * (1 + 1e-6), 2 * p_min)):
        d = check_data(cantilever(sub(tmp_path, f"b{k}"), perimeter_max_mm2=bound))
        assert d["perimeter"]["max_mm2"] == bound
    with pytest.raises(UserError):
        check_data(cantilever(sub(tmp_path, "c"), perimeter_max_mm2=p_min * (1 - 1e-6)))


# --- the number that the texts tell the user to enter is a bound that the check accepts ----------------------------

ADVICE = re.compile(r"raise settings\.perimeter_max_mm2 to (\d+\.\d) mm2 or more")
FLOOR = re.compile(r"can have a perimeter below (\d+\.\d) mm2")
SMALLEST = re.compile(r"smallest possible (\d+\.\d) mm2")


def bind_bound(problem, bound):
    """Bind a row with this bound to a problem: a `UserError` for a refused bound, else the numbers."""
    return pm.PerimeterRow(SimpleNamespace(settings=SimpleNamespace(perimeter_max_mm2=bound))).bind(problem)


def refused_text(problem, bound=1.0):
    with pytest.raises(UserError) as e:
        bind_bound(problem, bound)
    return str(e.value)


def advised(text):
    """The number of the advice of a refusal text, and the number of its statement of the floor (the same number)."""
    a, f = ADVICE.search(text), FLOOR.search(text)
    assert a and f and a.group(1) == f.group(1), text
    return float(a.group(1))


def test_the_advised_bound_of_the_refusal_is_a_bound_that_the_check_accepts_on_the_hook_example_at_1_2_mm():
    """Round 0 printed 6779.5 for 6779.5199999999995; a bound of 6779.5 is below p_min, so the advice was refused again."""
    prob = build_problem(load_config(Path(__file__).resolve().parent.parent / "examples" / "hook" / "hook.yaml"))
    p_min = pm.FaceList(prob.labels, prob.h).p_min_mm2()
    assert prob.h == 1.2 and p_min == 6779.5199999999995
    assert float(f"{p_min:.1f}") < p_min                                    # the nearest value at 0.1 is below p_min (the old text)
    with pytest.raises(UserError):
        bind_bound(prob, 6779.5)
    text = refused_text(prob, 6779.5)
    assert advised(text) == 6779.6
    assert bind_bound(prob, advised(text))["max_mm2"] == 6779.6
    assert bind_bound(prob, p_min)["p_min_mm2"] == p_min                       # the exact value is accepted too


@pytest.mark.parametrize("voxel_size, nearest_is", [(1.3, "below"), (1.7, "above"), (2.0, "equal")])
def test_the_advised_bound_passes_where_the_nearest_value_is_below_above_or_equal(tmp_path, voxel_size, nearest_is):
    """A problem for each way that `.1f` can round: 365.04 (down), 323.68 (up), 304.0 (exact)."""
    prob = build_problem(load_config(cantilever(sub(tmp_path, "a"), voxel_size=voxel_size)))
    p_min = pm.FaceList(prob.labels, prob.h).p_min_mm2()
    nearest = float(f"{p_min:.1f}")
    assert {"below": nearest < p_min, "above": nearest > p_min, "equal": nearest == pytest.approx(p_min, rel=1e-9)}[nearest_is]
    value = advised(refused_text(prob))
    assert 0 <= value - p_min < 0.1 + 1e-9                                  # the smallest value at 0.1 that is not below p_min
    assert bind_bound(prob, value)["p_min_mm2"] == p_min


def test_shown_p_min_is_never_below_p_min_and_never_0_1_above_it():
    rng = np.random.default_rng(7)
    for p_min in np.concatenate([rng.uniform(0.0, 20000.0, 400), 1.44 * np.arange(1, 400), 0.01 * np.arange(1, 400), [0.0, 6776.0]]):
        shown = float(pm.shown_p_min(p_min))
        assert not pm.below_floor(shown, p_min), p_min
        assert shown - p_min <= 0.1 + 1e-9, p_min
    assert pm.shown_p_min(6779.5199999999995) == "6779.6" and pm.shown_p_min(6780.96) == "6781.0"
    assert pm.shown_p_min(1.2 ** 2 * 4710) == pm.shown_p_min(6782.4) == "6782.4"        # round-off above an exact value: not 6782.5


@pytest.mark.parametrize("voxel_size", [1.3, 1.7])
def test_check_and_the_mcp_tool_show_the_same_number_and_it_is_accepted(tmp_path, capsys, voxel_size):
    from topopt.cli import main
    prob = build_problem(load_config(cantilever(sub(tmp_path, "a"), voxel_size=voxel_size)))
    p_min = pm.FaceList(prob.labels, prob.h).p_min_mm2()
    ok = cantilever(sub(tmp_path, "b"), voxel_size=voxel_size, perimeter_max_mm2=3.0 * p_min)
    assert main(["check", str(ok)]) == 0
    line = next(t for t in capsys.readouterr().out.splitlines() if t.startswith("perimeter:"))
    assert float(SMALLEST.search(line).group(1)) == float(pm.shown_p_min(p_min))
    assert bind_bound(prob, float(SMALLEST.search(line).group(1)))["p_min_mm2"] == p_min       # the fact line: a usable bound
    bad = cantilever(sub(tmp_path, "c"), voxel_size=voxel_size, perimeter_max_mm2=0.5 * p_min)
    assert main(["check", str(bad)]) == 1
    cli_text = capsys.readouterr().err
    assert bind_bound(prob, advised(cli_text))["max_mm2"] == advised(cli_text)                  # the check text: usable
    server = pytest.importorskip("topopt_mcp.server")
    r = server.check_problem(str(bad))
    assert r["ok"] is False and advised(r["error"]) == advised(cli_text)                        # the MCP text: the same number
    assert bind_bound(prob, advised(r["error"]))["max_mm2"] == advised(r["error"])


def test_the_run_log_facts_show_a_p_min_that_is_a_usable_bound(tmp_path):
    """`detail` and `facts` of the row (the `not met` summary and the end fact of `run.log`) use the same form."""
    prob = build_problem(load_config(cantilever(sub(tmp_path, "a"), voxel_size=1.3)))
    row = pm.PerimeterRow(SimpleNamespace(settings=SimpleNamespace(perimeter_max_mm2=1000.0)))
    row.bind(prob)
    row.p_base = 400.0
    con = row.constraint()
    shown = float(pm.shown_p_min(row.p_min))
    assert shown > row.p_min
    assert f"({shown:.1f} mm2)" in con.detail({})
    fact = con.facts({"g": 0.1, "value": 1100.0, "limit": 1000.0})[0]
    assert f"smallest possible {shown:.1f} mm2" in fact
    assert not pm.below_floor(shown, row.p_min)


def test_a_problem_with_p_min_zero_gives_plain_texts_and_no_crash():
    """No solid voxel: the empty design has P = 0. The ratio is not defined; no text divides by zero."""
    lab = grid((5, 5, 5), design=[(2, 2, 2)])
    prob = SimpleNamespace(labels=lab, h=1.0)
    row = pm.PerimeterRow(SimpleNamespace(settings=SimpleNamespace(perimeter_max_mm2=100.0)))
    numbers = row.bind(prob)
    assert numbers["p_min_mm2"] == 0.0 and numbers["bound_over_p_min"] is None
    (line,) = pm.PerimeterRow.lines(numbers)
    assert "bound / smallest = not defined (the smallest possible P is 0)" in line and "smallest possible 0.0 mm2" in line
    row.p_base = 6.0
    con = row.constraint()
    assert con.detail({}) == "the smallest possible P is 0 mm2"
    assert "smallest possible 0.0 mm2" in con.facts({"g": -0.9, "value": 6.0, "limit": 100.0})[0]
    assert pm.shown_p_min(0.0) == "0.0" and not pm.below_floor(100.0, 0.0)


def test_check_gives_the_facts_and_the_ratio_and_no_threshold(tmp_path, capsys):
    from topopt.cli import check_data, main
    fixed, p_min = p_numbers(cantilever(sub(tmp_path, "a")))
    bound = 3.0 * p_min
    p = cantilever(sub(tmp_path, "b"), perimeter_max_mm2=bound)
    d = check_data(p)["perimeter"]
    assert d == {"max_mm2": bound, "eps": pm.EPS, "fixed_part_mm2": fixed, "p_min_mm2": p_min, "p_base_mm2": None,
                 "bound_over_p_min": pytest.approx(3.0)}
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    line = next(t for t in out.splitlines() if t.startswith("perimeter:"))
    assert f"bound {bound:g} mm2" in line and f"fixed part {fixed:.1f} mm2" in line and f"smallest possible {pm.shown_p_min(p_min)} mm2" in line
    assert "bound / smallest = 3.000" in line and "voxel face area" in line and "staircase" in line
    assert main(["check", str(p), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["perimeter"]["p_min_mm2"] == p_min
    assert check_data(cantilever(sub(tmp_path, "c")))["perimeter"] is None            # no key: no fact


def test_check_stops_a_refused_bound_with_exit_1_and_the_text_on_stderr(tmp_path, capsys):
    from topopt.cli import main
    _fixed, p_min = p_numbers(cantilever(sub(tmp_path, "a")))
    p = cantilever(sub(tmp_path, "b"), perimeter_max_mm2=0.9 * p_min)
    assert main(["check", str(p)]) == 1
    assert "no design with density 0 or 1 can have a perimeter below" in capsys.readouterr().err


def test_run_stops_a_refused_bound_before_any_pass(tmp_path, capsys):
    from topopt.cli import main
    _fixed, p_min = p_numbers(cantilever(sub(tmp_path, "a")))
    p = cantilever(sub(tmp_path, "b"), perimeter_max_mm2=0.9 * p_min)
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 1
    assert "no design with density 0 or 1 can have a perimeter below" in capsys.readouterr().err
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "error" and "perimeter" in status["message"]
    assert not (out / "history.jsonl").exists() and not list((out / "snapshots").glob("*.bin"))   # no pass ran
    assert "model:" not in (out / "run.log").read_text()                               # the model was not built


def test_a_script_that_builds_the_formulation_is_refused_too(tmp_path):
    """`run()` and `formulation_from_config` with a model bind at once; without a model the row binds at the first pass."""
    _fixed, p_min = p_numbers(cantilever(sub(tmp_path, "a")))
    p = cantilever(sub(tmp_path, "b"), perimeter_max_mm2=0.9 * p_min)
    m = Model(build_problem(load_config(p)))
    with pytest.raises(UserError, match="perimeter"):
        formulation_from_config(m.problem.config, m.n_design, m)
    from topopt.optimize import minimize
    with pytest.raises(UserError, match="perimeter"):
        minimize(m, formulation_from_config(m.problem.config, m.n_design), iterations=2)
    m.solver.free()


# --- runs: compliance, mass and the worst case; the report ---------------------------------------------------------


def run_cli(path, out):
    from topopt.cli import main
    assert main(["run", str(path), "--out", str(out), "--quiet"]) == 0
    return out


def history_of(out):
    return [json.loads(x) for x in (Path(out) / "history.jsonl").read_text().splitlines()]


def perimeter_value(record):
    return next(c for c in record["constraints"] if c["name"] == "perimeter")


@pytest.fixture(scope="module")
def free(tmp_path_factory):
    """The cantilever with no bound, and with a bound that is far above the design (the row is slack: P of the free design)."""
    tmp = tmp_path_factory.mktemp("free")
    plain = run_cli(cantilever(sub(tmp, "plain")), tmp / "plain-run")
    slack = run_cli(cantilever(sub(tmp, "slack"), perimeter_max_mm2=1e6), tmp / "slack-run")
    return {"compliance": history_of(plain)[-1]["compliance"],
            "p_free": json.loads((slack / "run.json").read_text())["perimeter"]["p_final_mm2"]}


def check_report(out, bound):
    """The report of a run with the key: run.json, history, metrics rows, run.log. Return the run.json block."""
    out = Path(out)
    block = json.loads((out / "run.json").read_text())["perimeter"]
    hist = history_of(out)
    assert sorted(block) == ["eps", "fixed_part_mm2", "max_mm2", "p_base_mm2", "p_final_mm2", "p_min_mm2"]
    assert block["max_mm2"] == bound and block["eps"] == pm.EPS
    assert block["p_base_mm2"] == pytest.approx(perimeter_value(hist[0])["value"], rel=1e-9)        # P of the start design = pass 0
    assert block["p_final_mm2"] == pytest.approx(perimeter_value(hist[-1])["value"], rel=1e-12)
    assert block["fixed_part_mm2"] < block["p_min_mm2"] < block["p_final_mm2"]
    rows = json.loads((out / "result" / "metrics.json").read_text())["rows"]
    row = next(r for r in rows if r["name"] == "perimeter")
    assert (row["kind"], row["unit"], row["limit"]) == ("perimeter", "mm2", bound)
    assert row["value_start"] == block["p_base_mm2"] and row["value_end"] == block["p_final_mm2"]
    assert row["met"] == (block["p_final_mm2"] / bound - 1 <= FEAS_TOL)
    text = (out / "run.log").read_text()
    assert f"fact: perimeter: bound {bound:g} mm2" in text and "P of the start design" in text
    assert re.search(r"fact: perimeter: P of the start design [\d.]+ mm2, P of the last design [\d.]+ mm2, bound [\d.e+]+ mm2 "
                     r"\((met|not met)\); smallest possible [\d.]+ mm2", text)
    return block


def test_a_compliance_run_with_the_bound_meets_it_and_reports_it(tmp_path, free):
    bound = round(0.8 * free["p_free"], 1)
    out = run_cli(cantilever(tmp_path, perimeter_max_mm2=bound), tmp_path / "run")
    block = check_report(out, bound)
    assert block["p_final_mm2"] / bound - 1 <= FEAS_TOL and block["p_final_mm2"] < 0.95 * free["p_free"]   # the row is active
    hist = history_of(out)
    assert hist[-1]["compliance"] > free["compliance"]                                  # the bound costs stiffness (a fact)
    assert hist[-1]["volume_fraction"] == pytest.approx(0.3, rel=0.02)                  # the volume row still holds the budget
    assert "(met)" in (out / "run.log").read_text()


def test_a_worst_case_run_with_the_bound_meets_it(tmp_path, free):
    bound = round(0.85 * free["p_free"], 1)
    out = run_cli(cantilever(tmp_path, perimeter_max_mm2=bound, case_combine="max"), tmp_path / "run")
    block = check_report(out, bound)
    assert block["p_final_mm2"] / bound - 1 <= FEAS_TOL


def test_a_mass_run_with_the_bound_holds_it_and_adds_material(tmp_path):
    """Mass objective with two compliance limits: the bound is slack, then 0.97 of the P of that design is binding."""
    def mass(**kw):
        def fn(d):
            d["settings"].update(objective="mass", **kw)
            d["load_cases"][0]["compliance_limit"] = 0.22
            d["load_cases"][1]["compliance_limit"] = 0.05
        return fn
    base = cantilever(sub(tmp_path, "a"))
    edit_yaml(base, mass(perimeter_max_mm2=1e6))
    slack = run_cli(base, tmp_path / "slack")
    p_mass = json.loads((slack / "run.json").read_text())["perimeter"]["p_final_mm2"]
    vol_mass = history_of(slack)[-1]["volume_fraction"]
    bound = round(0.97 * p_mass, 1)
    tight = cantilever(sub(tmp_path, "b"))
    edit_yaml(tight, mass(perimeter_max_mm2=bound))
    out = run_cli(tight, tmp_path / "tight")
    block = check_report(out, bound)
    last = history_of(out)[-1]
    assert block["p_final_mm2"] / bound - 1 <= FEAS_TOL and block["p_final_mm2"] < p_mass
    assert all(c["g"] <= FEAS_TOL for c in last["constraints"])                         # the compliance rows are met too
    assert last["volume_fraction"] > vol_mass                                           # the perimeter row asks for more material
    assert any("does not hold material" in t for t in (out / "run.log").read_text().splitlines() if "perimeter_max_mm2" in t)


def test_a_run_without_the_key_has_no_perimeter_block_and_no_perimeter_row(tmp_path):
    out = run_cli(cantilever(tmp_path), tmp_path / "run")
    run_json = json.loads((out / "run.json").read_text())
    assert "perimeter" not in run_json and [c["name"] for c in run_json["constraints"]] == ["volume"]
    assert all(c["name"] == "volume" for r in history_of(out) for c in r["constraints"])
    assert "perimeter" not in (out / "run.log").read_text()
    assert "rows" in json.loads((out / "result" / "metrics.json").read_text())
    assert json.loads((out / "result" / "metrics.json").read_text())["rows"] == []


def test_run_json_lists_the_row_before_the_run_goes_on(tmp_path):
    """`write_run_json` calls `formulation_from_config` with no model: the row is in `constraints` with its limit."""
    out = run_cli(cantilever(tmp_path, perimeter_max_mm2=2000.0), tmp_path / "run")
    assert {"name": "perimeter", "limit": 2000.0} in json.loads((out / "run.json").read_text())["constraints"]


# --- P of the start design ------------------------------------------------------------------------------------------


def test_start_density_is_the_density_of_pass_zero_and_touches_no_mirror_record(tmp_path):
    for case in ("filter and projection", "two mirror planes"):
        m = Model(build_problem(load_config(CHAIN_CASES[case](sub(tmp_path, case.replace(" ", "_"))))))
        st = m.problem.config.settings
        before = json.dumps(m.problem.mirror, default=str)
        rho0 = pm.start_density(m)
        assert json.dumps(m.problem.mirror, default=str) == before                       # no "end" record from this call
        x0 = np.full(m.n_design, st.volume_fraction)
        assert np.array_equal(rho0, m.density(x0, float(st.beta_schedule[0][1]))[1])    # bit for bit the density of pass 0
        m.solver.free()


# --- checkpoints: the effective settings and the resume ---------------------------------------------------------


FROZEN_EFFECTIVE = Path(__file__).resolve().parent / "data" / "effective_settings_e6a3e70.json"


def test_the_effective_settings_of_a_run_without_the_key_are_those_of_the_code_before_the_key_existed(tmp_path):
    """A frozen fixture: the effective settings of this case, computed once with the code of base e6a3e70 (before A4c).

    The key is left out when it is None, so the dict and its hash are the same as before, and a checkpoint of the old code
    resumes with no `--allow-changed-inputs`. A later `Settings` field that forgets the same pop fails here (review m3).
    The file is not changed for later fields. The same rule holds for them: `gpu_tol` and `gpu_backward_error` (G2) and `objective_scale` (P3a)
    are left out at their defaults. The one allowed addition is a value that is not the default (the next test).
    """
    from topopt import checkpoint as C
    frozen = json.loads(FROZEN_EFFECTIVE.read_text())
    assert frozen["base_commit"].startswith("e6a3e70")
    now = C.input_hashes(load_config(cantilever(tmp_path)))
    added = sorted(set(now["effective"]["settings"]) - set(frozen["effective"]["settings"]))
    assert not added, (f"{added} is new in the effective settings: pop it in `checkpoint.effective_settings` when it is unset, "
                       f"or change the fixture on purpose (older checkpoints then need --allow-changed-inputs)")
    assert now["effective"] == frozen["effective"]
    assert now["effective_sha256"] == frozen["effective_sha256"]
    old = dict(now, effective=frozen["effective"], effective_sha256=frozen["effective_sha256"])   # what the old checkpoint holds
    assert C.diff_inputs(old, now) == ([], [])


def test_a_setting_that_is_not_its_default_is_the_only_addition_to_the_frozen_effective_settings(tmp_path):
    """P3a, F4: `settings.objective_scale: first` is the old behavior and not the default, so it is in the effective settings (a change from or to it is a change of the inputs).

    The file is the one of e6a3e70, and the key is the only difference: without it the dict is the frozen one. Each gpu-mg number at a value that is not
    its default is in the same way (`settings.gpu_tol`).
    """
    from topopt import checkpoint as C
    frozen = json.loads(FROZEN_EFFECTIVE.read_text())
    assert frozen["base_commit"].startswith("e6a3e70") and "objective_scale" not in frozen["effective"]["settings"]
    now = C.input_hashes(load_config(cantilever(tmp_path, objective_scale="first")))
    assert sorted(set(now["effective"]["settings"]) - set(frozen["effective"]["settings"])) == ["objective_scale"]
    assert now["effective"]["settings"]["objective_scale"] == "first" and now["effective_sha256"] != frozen["effective_sha256"]
    rest = {**now["effective"], "settings": {k: v for k, v in now["effective"]["settings"].items() if k != "objective_scale"}}
    assert rest == frozen["effective"]
    other = C.input_hashes(load_config(cantilever(sub(tmp_path, "g"), gpu_tol=1e-8)))
    assert sorted(set(other["effective"]["settings"]) - set(frozen["effective"]["settings"])) == ["gpu_tol"]


def test_the_key_in_the_yaml_is_a_real_change_of_the_inputs(tmp_path):
    from topopt import checkpoint as C
    p = cantilever(tmp_path)
    old = C.input_hashes(load_config(p))
    edit_yaml(p, lambda d: d["settings"].update(perimeter_max_mm2=900.0))
    new = C.input_hashes(load_config(p))
    changes, _ = C.diff_inputs(old, new)
    assert changes == ["settings.perimeter_max_mm2: <not set> -> 900.0"]
    assert new["effective"]["settings"]["perimeter_max_mm2"] == 900.0 and old["effective_sha256"] != new["effective_sha256"]
    edit_yaml(p, lambda d: d["settings"].update(perimeter_max_mm2=950.0))
    assert C.diff_inputs(new, C.input_hashes(load_config(p)))[0] == ["settings.perimeter_max_mm2: 900.0 -> 950.0"]


@needs_direct
def test_a_killed_run_with_the_bound_resumes_to_the_same_bytes_and_the_same_perimeter_block(tmp_path):
    """SIGKILL at pass 12, then `--resume`: history, snapshots and run.json (with the block and p_final) equal the run that was not stopped."""
    from test_checkpoint import finish, killed, resume, same_run
    p = cantilever(sub(tmp_path, "case"), perimeter_max_mm2=900.0, iterations=24)
    base, out = tmp_path / "base", tmp_path / "r"
    finish(p, base)
    killed(p, out, "after_checkpoint:12")
    assert "p_final_mm2" not in json.loads((out / "run.json").read_text())["perimeter"]      # the killed run has the start numbers only
    resume(p, out)
    same_run(base, out)
    a, b = (json.loads((d / "run.json").read_text())["perimeter"] for d in (base, out))
    assert a == b and "p_final_mm2" in b
    assert b["p_base_mm2"] == pytest.approx(perimeter_value(history_of(out)[0])["value"], rel=1e-9)   # from pass 0 of the first run


# --- the spec route and the metrics block ----------------------------------------------------------------------


def hook_like_spec(bound):
    return {"name": "wall", "regions": {
        "design": [{"box": {"min": [0, 0, 0], "max": [40, 20, 20]}}],
        "fixed": {"wall": [{"box": {"min": [-4, 0, 0], "max": [0, 20, 20]}}]},
        "load": {"tip": [{"box": {"min": [32, 5, 0], "max": [40, 15, 8]}}]}},
        "load_cases": [{"name": "a", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
        "settings": {"voxel_size": 2.0, "perimeter_max_mm2": bound}}


def test_the_spec_route_passes_the_key_to_the_yaml_and_check_reports_it(tmp_path, monkeypatch):
    """Without the pass-through in `regions_builder` the key was dropped without a message."""
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    r = server.build_regions(hook_like_spec(2.5e4))
    assert r["ok"], r
    doc = yaml.safe_load((tmp_path / "ws" / r["yaml_rel"]).read_text())
    assert doc["settings"]["perimeter_max_mm2"] == 2.5e4
    assert r["check"]["perimeter"]["max_mm2"] == 2.5e4 and r["check"]["perimeter"]["p_min_mm2"] > 0
    no_key = hook_like_spec(1.0)
    del no_key["settings"]["perimeter_max_mm2"]
    r2 = server.build_regions(no_key, name="plain")
    assert "perimeter_max_mm2" not in yaml.safe_load((tmp_path / "ws" / r2["yaml_rel"]).read_text())["settings"]


def test_the_spec_route_reports_a_refused_bound_as_a_check_error(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    r = server.build_regions(hook_like_spec(1.0))
    assert r["ok"] and "perimeter" in (r.get("check_error") or "") and "minimum cut" in r["check_error"]


def test_the_metrics_rows_block_reads_the_row_with_the_unit_and_an_old_history_without_it():
    from topopt import rows_motion as rm
    rec = lambda v, g: {"constraints": [{"name": "volume", "value": 0.3, "limit": 0.3, "g": 0.0, "violation": 0.0},
                                        {"name": "perimeter", "value": v, "limit": 1000.0, "g": g, "violation": max(g, 0.0)}]}
    block = rm.rows_block([rec(600.0, -0.4), rec(900.0, -0.1), rec(1004.0, 0.004)])
    assert block == [{"name": "perimeter", "kind": "perimeter", "unit": "mm2", "limit": 1000.0, "value_start": 600.0,
                      "value_end": 1004.0, "g_end": 0.004, "met": True}]
    assert rm.rows_block([rec(600.0, -0.4), rec(1100.0, 0.1)])[0]["met"] is False
    assert rm.rows_block([{"iteration": 0}]) == []


# --- every solver path: the row reads the density and the labels only ----------------------------------------------


@pytest.mark.parametrize("solver", ["amg", pytest.param("pardiso", marks=needs_pardiso),
                                    pytest.param("cholesky", marks=needs_accelerate)])
def test_the_row_runs_with_each_solver(tmp_path, solver):
    """The row calls no solver, so each path gives the same row. AMG runs on every platform; PARDISO on Linux x86_64 and the
    Accelerate Cholesky on a Mac with Apple silicon run where they exist (the CI legs)."""
    from topopt.optimize import minimize
    p = cantilever(tmp_path, solver=solver, perimeter_max_mm2=2000.0, iterations=12)
    m = Model(build_problem(load_config(p)))
    assert m.solver_name == solver
    res = minimize(m, formulation_from_config(m.problem.config, m.n_design, m))
    row = perimeter_value(res.history[-1])
    assert row["limit"] == 2000.0 and row["g"] < 0 and perimeter_value(res.history[0])["value"] != row["value"]
    m.solver.free()


def test_the_gpu_mg_model_inherits_what_the_row_reads_and_the_row_runs_from_the_shared_loop():
    """No OpenCL is loaded here: the file is parsed, not imported. `GpuModel` does not define the density chain, `design_point` or
    the labels, so the row (`point.rho`, `point.model.problem.labels`, `Model.gradient_wrt_x`) works as on `Model`. The run loop
    is the shared `optimize.run`, which calls `formulation_from_config` with the model. The gpu-mg run itself is not tested here."""
    import ast
    src = (Path(pm.__file__).parent / "gpu_mg" / "model.py").read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "GpuModel")
    defined = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert cls.bases and ast.unparse(cls.bases[0]) == "Model"
    assert not defined & {"density", "stiffness_density", "gradient_wrt_x", "design_point", "chain_rule"}


# --- the texts: README, CONTRACTS, SKILL.md (both copies), the MCP tool docstrings --------------------------------

ROOT = Path(__file__).resolve().parent.parent
HABITUAL_CAPS = re.compile(r"\b(must|never|always)\b")


def text_section(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i + len(start))]


def test_skill_copies_are_equal_and_name_the_key_the_unit_and_the_options():
    claude = (ROOT / ".claude/skills/part-to-topopt/SKILL.md").read_text()
    agents = (ROOT / ".agents/skills/part-to-topopt/SKILL.md").read_text()
    assert claude == agents
    part = text_section(claude, "### The `perimeter_max_mm2` setting", "\n## ")
    for word in ("settings.perimeter_max_mm2", "mm2", "smallest possible", "minimum cut", "raise the bound",
                 "reduce the solid regions", "make the design space bigger", "1.24"):
        assert word in part, word
    assert not HABITUAL_CAPS.search(part.lower())      # facts and options, no habitual caps
    assert "perimeter_max_mm2 is ... no design with density 0 or 1 can have a perimeter below" in claude   # the message table


def test_server_docstrings_name_the_key_and_the_check_data_key():
    server = (ROOT / "topopt_mcp/server.py").read_text()
    builder_doc = text_section(server, "def build_regions", '"""\n    try:')
    check_doc = text_section(server, "def check_problem", '"""\n    yaml = ')
    from topopt_mcp import replies
    check_doc += "\n" + replies.guide("check_problem")      # P9-11: the lines that explain the keys of the reply moved to the guide topopt://replies/check_problem
    assert "perimeter_max_mm2" in builder_doc and "mm2" in builder_doc
    assert "Key perimeter:" in check_doc and "p_min_mm2" in check_doc and "mm2" in check_doc
    for text in (builder_doc, check_doc):
        lines = [ln for ln in text.splitlines() if "perimeter" in ln.lower()]
        assert lines and not HABITUAL_CAPS.search(" ".join(lines).lower()), lines


def test_readme_and_contracts_name_the_key_and_the_unit():
    readme = (ROOT / "docs" / "reference" / "problem-and-run.md").read_text()
    part = text_section(readme, "## A bound on the surface area", "\n## ")
    for word in ("perimeter_max_mm2", "mm2", "CONTRACTS.md section 26.3", "smallest `P`"):
        assert word in part, word
    assert not HABITUAL_CAPS.search(part.lower())
    assert "result/run.json" not in part and "`run.json` (folder root" in part      # run.json is in the run folder, not in result/ (review m4)
    assert "`hook_short` example, 40 passes" in part                              # the problem of the cost numbers (review n3)
    contracts = (ROOT / "CONTRACTS.md").read_text()
    assert "## 26. " in contracts and "### 26.3" in contracts
    part = text_section(contracts, "### 26.3", "\n## 27.")
    for word in ("perimeter_max_mm2", "mm2", "p_min_mm2", "p_base_mm2", "p_final_mm2", "effective_settings"):
        assert word in part, word
    assert not HABITUAL_CAPS.search(part.lower())
