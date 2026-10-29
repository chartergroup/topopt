"""Tests for the mirror key (A4a, CONTRACTS.md section 26.1): the YAML, the grid plane, the index map, the
averaging in the model, the facts, and the texts.

Tolerances (each one is stated at the test that uses it):
- The involution, the maps and the counts are exact.
- `F P = P F`: rounding of sums in another order, so 1e-12 (measured 6e-17 on a 12 x 6 x 5 grid).
- FD gradients: central difference with the limits of `test_optimize` (1e-5 with a direct solver).
"""

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from test_optimize import box, make_case
from topopt import symmetry
from topopt.cli import check_data, main
from topopt.config import MirrorPlane, UserError, load_config
from topopt.filter import build_filter, heaviside_deriv
from topopt.optimize import ETA, Model
from topopt.problem import build_problem
from topopt.snapshot import RunWriter
from topopt.voxelize import DESIGN, EMPTY, FIXED, KEEP_OUT, LOAD

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "examples" / "hook" / "hook.yaml"


# --- helpers -----------------------------------------------------------------


def write_yaml(tmp_path, mirror=None, name="m.yaml", edit=None, **kw):
    """A cantilever that is symmetric about the planes y = 0 and z = 0 (and about no other plane).

    Design x 0..16, y -4..4, z -4..4. Plate x -2..0 and tip x 14..16, y -2..2, z -2..2, in the same frame.
    Case `down` has a force along z (not symmetric about z = 0), case `side` a force along x (symmetric about
    both planes). Case `mixed` has a symmetric and an antisymmetric part about each plane: the sensitivity of the
    compliance is then not symmetric, and only the transposed average gives the gradient of the averaged design.
    `kw` goes to the settings.
    """
    box(tmp_path / "design.stl", (0, -4, -4), (16, 4, 4))
    box(tmp_path / "plate.stl", (-2, -4, -4), (0, 4, 4))
    box(tmp_path / "tip.stl", (14, -2, -2), (16, 2, 2))
    settings = {"voxel_size": 2.0, "volume_fraction": 0.4, "iterations": 4,
                "beta_schedule": [[0, 1], [2, 2]]}
    settings.update(kw)
    d = {"name": "sym", "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "down", "weight": 1.0, "forces": [{"region": "tip", "force": [0, 0, -2.0]}]},
                        {"name": "side", "weight": 1.0, "forces": [{"region": "tip", "force": [1.5, 0, 0]}]},
                        {"name": "mixed", "weight": 1.0, "forces": [{"region": "tip", "force": [1.0, 0.7, -1.5]}]}],
         "settings": settings}
    if mirror is not None:
        d["mirror"] = mirror
    if edit:
        edit(d)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d))
    return p


def two_planes():
    return [{"normal": "y", "at_mm": 0.0}, {"normal": "z", "at_mm": 0.0}]


def keep_out_corner(tmp_path):
    """A keep-out box in one octant: the part is not symmetric about y = 0 or z = 0 any more."""
    box(tmp_path / "ko.stl", (6, 0, -4), (10, 4, 0))
    return lambda d: d["regions"].__setitem__("keep_out", ["ko.stl"])


def make_model(path):
    m = Model(build_problem(load_config(path)))
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13   # AMG is not exact at the run tolerance: tighten it for FD tests
    return m


def unit_label_grid(parity, pad=0, seed=1):
    """A label grid, symmetric about the plane along axis 1, with solids, keep-out and empty cells.

    `parity` "face": 2 m layers, plane on a voxel face. "centre": 2 m + 1 layers, plane through the middle layer.
    `pad` EMPTY layers before the symmetric part move the plane inside a grid that is not symmetric.
    Return `(labels, k)`.
    """
    rng = np.random.default_rng(seed)
    nx, m, nz = 6, 5, 4
    half = rng.choice([EMPTY, DESIGN, DESIGN, DESIGN, FIXED, LOAD, KEEP_OUT], size=(nx, m, nz)).astype(np.uint8)
    mid = rng.choice([DESIGN, DESIGN, FIXED, KEEP_OUT], size=(nx, 1, nz)).astype(np.uint8)
    parts = [half, half[:, ::-1]] if parity == "face" else [half, mid, half[:, ::-1]]
    sym = np.concatenate(parts, axis=1)
    padding = np.zeros((nx, pad, nz), dtype=np.uint8)
    labels = np.concatenate([padding, sym], axis=1)
    n_sym = sym.shape[1]
    return labels, n_sym + 2 * pad   # k = n_sym + 2 pad: the plane is n_sym / 2 layers from the symmetric part start


# --- the YAML ------------------------------------------------------------------------


def test_no_mirror_key_gives_no_planes(tmp_path):
    cfg = load_config(write_yaml(tmp_path))
    assert cfg.mirror == ()
    assert build_problem(cfg).mirror is None


def test_planes_are_read(tmp_path):
    p = write_yaml(tmp_path, [{"normal": "Y", "at_mm": 1}, {"normal": [0, 0, -2], "at_mm": 3.5, "align": "face"}])
    assert load_config(p).mirror == (MirrorPlane("y", 1.0, "auto"), MirrorPlane("z", 3.5, "face"))


def test_at_mm_reads_a_number_that_yaml_loads_as_text(tmp_path):
    """YAML 1.1 loads `1e-3` as the string '1e-3' (it needs 1.0e-3). The `at_mm` of a plane reads it like the other numbers."""
    assert yaml.safe_load("at_mm: 1e-3") == {"at_mm": "1e-3"}
    p = write_yaml(tmp_path, [{"normal": "y", "at_mm": "1e-3"}])
    assert load_config(p).mirror == (MirrorPlane("y", 1e-3, "auto"),)


def test_empty_list_is_no_mirror(tmp_path):
    assert load_config(write_yaml(tmp_path, [])).mirror == ()


REFUSALS = [
    ([{"normal": [0, 1, 1], "at_mm": 0}], "not axis-aligned"),
    ([{"normal": [1, 0.2, 0], "at_mm": 0}], "not axis-aligned"),
    ([{"normal": "w", "at_mm": 0}], "not x, y or z"),
    ([{"normal": [0, 0, 0], "at_mm": 0}], "zero vector"),
    ([{"normal": "y", "at_mm": 0}, {"normal": "y", "at_mm": 5}], "each axis can have one plane"),
    ([{"normal": a, "at_mm": 0} for a in ("x", "y", "z", "x")], "at most 3"),
    ({"normal": "y", "at_mm": 0}, "must be a list"),
    (["y"], "must be a mapping"),
    ([{"normal": "y"}], "missing key 'at_mm'"),
    ([{"at_mm": 0}], "missing key 'normal'"),
    ([{"normal": "y", "at_mm": 0, "where": 1}], "unknown key 'where'"),
    ([{"normal": "y", "at_mm": "zero"}], "at_mm"),
    ([{"normal": "y", "at_mm": True}], "at_mm"),
    ([{"normal": "y", "at_mm": float("inf")}], "at_mm"),
    ([{"normal": "y", "at_mm": 0, "align": "middle"}], "align"),
    ([{"normal": 7, "at_mm": 0}], "x, y, z or a vector"),
]


@pytest.mark.parametrize("mirror, fact", REFUSALS)
def test_refusal_is_one_line_with_the_supported_form_and_the_option(tmp_path, mirror, fact):
    with pytest.raises(UserError) as e:
        load_config(write_yaml(tmp_path, mirror))
    text = str(e.value)
    assert "\n" not in text and fact in text
    assert "up to 3 planes with distinct axes" in text and "re-orient the part" in text
    assert "(file: " in text


# --- the grid plane ---------------------------------------------------------------------


def test_shift_lo_keeps_every_region_and_puts_the_plane_on_the_grid():
    lo = np.array([-1.0, -17.5, -30.0])
    h = 1.2
    face, centre, auto = (MirrorPlane("y", 0.0, a) for a in ("face", "centre", "auto"))
    for plane, k in ((face, 30), (centre, 31), (auto, 30)):
        new = symmetry.shift_lo((plane,), lo, h)
        assert new[1] == pytest.approx(0.0 - k * h / 2) and new[1] <= lo[1] < new[1] + h
        assert new[0] == lo[0] and new[2] == lo[2]   # only the plane axis moves
        assert symmetry.half_steps_of(new, "y", 0.0, h) == k
    assert symmetry.shift_lo((), lo, h) is lo   # no plane: nothing changes, not even the object


@pytest.mark.parametrize("at", [-3.0, -0.31, 0.0, 0.6, 2.5, 17.5, 40.0])
@pytest.mark.parametrize("align", ["auto", "face", "centre"])
def test_shift_lo_is_below_one_voxel_and_has_the_parity(at, align):
    lo, h = np.array([0.0, -2.2, 0.0]), 0.7
    new = symmetry.shift_lo((MirrorPlane("y", at, align),), lo, h)
    # GRID_TOL h: the origin may be above the region by the rounding of float32 STL vertices
    assert -symmetry.GRID_TOL * h - 1e-12 <= lo[1] - new[1] < (h / 2 if align == "auto" else h) + 1e-9
    k = symmetry.half_steps_of(new, "y", at, h)
    assert align == "auto" or symmetry.parity_of(k) == align


def test_a_plane_on_a_voxel_face_does_not_move_the_grid_by_rounding():
    lo = np.array([0.0, -18.0, 0.0])   # the plane y = 0 is 30 half voxels (h = 1.2) above lo
    new = symmetry.shift_lo((MirrorPlane("y", 0.0, "auto"),), lo, 1.2)
    assert abs(new[1] - lo[1]) < 1e-6


# --- the parity choice on the hook -------------------------------------------------------


def hook_yaml(tmp_path, mirror):
    d = yaml.safe_load(HOOK.read_text())
    r = d["regions"]
    r["design"] = str(HOOK.parent / r["design"])
    r["fixed"] = [str(HOOK.parent / f) for f in r["fixed"]]
    r["keep_out"] = [str(HOOK.parent / f) for f in r["keep_out"]]
    r["load"] = {k: str(HOOK.parent / v) for k, v in r["load"].items()}
    if mirror is not None:
        d["mirror"] = mirror
    p = tmp_path / "hook.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def test_hook_plane_y0_has_no_mismatch_and_no_orphan_on_the_centre_parity(tmp_path):
    """Measured 2026-10-20 at 1.2 mm: face parity 134 label mismatches (24 orphans), centre parity 0 (0 orphans)."""
    prob = build_problem(load_config(hook_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}])))
    pl = prob.mirror["planes"][0]
    assert pl["parity"] == "centre" and pl["k"] == 31 and pl["plane_index"] == 15
    assert pl["label_mismatches"] == {"face": 134, "centre": 0}
    assert pl["orphans"]["count"] == 0 and pl["orphans"]["share"] == 0.0
    assert prob.shape == (68, 31, 55) and prob.origin[1] == pytest.approx(-18.6)


def test_hook_face_parity_has_mismatches_and_orphans_in_the_load_region(tmp_path):
    prob = build_problem(load_config(hook_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0, "align": "face"}])))
    pl = prob.mirror["planes"][0]
    assert pl["parity"] == "face" and pl["align"] == "face" and prob.shape[1] == 30
    assert pl["label_mismatches"] == {"face": 134, "centre": None}   # the other parity is not tried
    assert pl["orphans"]["count"] == 24 and pl["orphans"]["load"] == 24
    assert pl["orphans"]["outside_grid"] + pl["orphans"]["empty"] + pl["orphans"]["keep_out"] == 0


def test_hook_without_mirror_has_the_grid_of_the_base_run(tmp_path):
    """The grid of the base run: origin [-1, -17.5, -30], shape [68, 30, 55]."""
    prob = build_problem(load_config(hook_yaml(tmp_path, None)))
    assert prob.mirror is None and prob.shape == (68, 30, 55)
    np.testing.assert_array_equal(prob.origin, [-1.0, -17.5, -30.0])


def test_hook_load_asymmetry_is_a_fact_per_case(tmp_path):
    prob = build_problem(load_config(hook_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}])))
    hang, pull = prob.mirror["load_asymmetry"]
    assert (hang["case"], pull["case"]) == ("hang", "pull_off")
    assert hang["planes"][0] <= symmetry.SYMMETRIC_TOL
    # Fy = 5.21 of |F| = 30.0 (distributed on the nodes): |f - R f| / |f| = 2 * 5.21 / 30.0
    assert pull["planes"][0] == pytest.approx(2 * 5.21 / np.linalg.norm([20.89, 5.21, -20.89]), rel=1e-6)
    lines = "\n".join(symmetry.summary_lines(prob.mirror))
    assert "'pull_off'" in lines and "not symmetric" in lines and "'hang'" in lines


def test_parity_choice_is_the_fewer_mismatches_and_a_tie_keeps_the_smaller_shift(tmp_path):
    p = write_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}])
    pl = build_problem(load_config(p)).mirror["planes"][0]
    # y -4..4, h = 2: the plane is on a voxel face at the smaller shift (0). On the centre parity the design faces
    # at y = -4 and y = 4 lie on voxel centres, and the ray decides them with one sign (the jitter): mismatches.
    assert pl["parity"] == "face" and pl["label_mismatches"]["face"] == 0
    assert pl["label_mismatches"]["centre"] is not None and pl["label_mismatches"]["centre"] > 0
    # a plane at y = 1 on this part: no parity is symmetric, the smaller count wins
    p = write_yaml(tmp_path, [{"normal": "y", "at_mm": 1.0}], name="m1.yaml")
    pl = build_problem(load_config(p)).mirror["planes"][0]
    counts = pl["label_mismatches"]
    assert counts[pl["parity"]] == min(counts["face"], counts["centre"])


# --- the index map -----------------------------------------------------------------------


@pytest.mark.parametrize("parity", ["face", "centre"])
@pytest.mark.parametrize("pad", [0, 3])
def test_map_is_an_involution_and_a_reflection(parity, pad):
    labels, k = unit_label_grid(parity, pad)
    perm, causes = symmetry.design_map(labels, 1, k)
    design = np.flatnonzero(labels.reshape(-1) == DESIGN)
    assert np.array_equal(perm[perm], np.arange(design.size))   # involution
    assert sum(causes.values()) == 0                              # a symmetric grid has no orphan
    ijk = np.stack(np.unravel_index(design, labels.shape), axis=1)
    centre = ijk + 0.5
    mirrored = centre[perm]
    np.testing.assert_array_equal(centre[:, [0, 2]], mirrored[:, [0, 2]])   # the other axes do not move
    np.testing.assert_array_equal(centre[:, 1] + mirrored[:, 1], k)         # the centres add up to 2 * plane (k half voxels)


@pytest.mark.parametrize("parity", ["face", "centre"])
@pytest.mark.parametrize("pad", [0, 3])
@pytest.mark.parametrize("rule", [None, "density_one", "ignore"])
def test_filter_commutes_with_the_map_on_a_symmetric_label_grid(parity, pad, rule):
    """F P = P F and b symmetric, for a symmetric label field. The rule ignore exists with A3a only."""
    kw = {}
    if rule is not None:
        if "solid_neighbours" not in inspect.signature(build_filter).parameters:
            pytest.skip("build_filter has no solid_neighbours rule in this branch (A3a)")
        kw = {"solid_neighbours": rule}
    labels, k = unit_label_grid(parity, pad, seed=4)
    perm, _ = symmetry.design_map(labels, 1, k)
    F, FT, b = build_filter(labels, 1.0, 2.2, **kw)
    P = np.eye(perm.size)[perm]   # (P x)[p] = x[perm[p]]
    Fd = F.toarray()
    assert np.abs(Fd @ P - P @ Fd).max() < 1e-12
    assert np.abs(b - b[perm]).max() < 1e-12
    assert np.abs(Fd - Fd.T).max() > 1e-3   # F is not a symmetric matrix: the commutation is the property that counts


def test_filter_does_not_commute_where_a_voxel_has_no_partner():
    labels, k = unit_label_grid("face", 0, seed=4)
    labels[2, 1, 1], labels[2, -2, 1] = DESIGN, KEEP_OUT   # a design voxel, a keep-out as its image
    perm, causes = symmetry.design_map(labels, 1, k)
    assert causes["keep_out"] >= 1
    F, _, _ = build_filter(labels, 1.0, 2.2)
    Fd, P = F.toarray(), np.eye(perm.size)[perm]
    assert np.abs(Fd @ P - P @ Fd).max() > 1e-3


def test_orphans_by_cause():
    """Plane k = 6 along y: voxel i goes to 5 - i. Row x = 0 holds five causes, row x = 1 holds one more."""
    D, E, F_, L, K = DESIGN, EMPTY, FIXED, LOAD, KEEP_OUT
    labels = np.zeros((2, 12, 1), dtype=np.uint8)
    labels[0, :, 0] = [D, D, D, K, L, F_, D, E, D, E, E, E]
    labels[1, 0, 0] = D
    perm, causes = symmetry.design_map(labels, 1, 6)
    assert causes == {"outside_grid": 2, "empty": 1, "keep_out": 1, "fixed": 1, "load": 1, "other": 0}
    design = np.flatnonzero(labels.reshape(-1) == DESIGN)
    orphans = np.flatnonzero(perm == np.arange(design.size))
    assert orphans.size == 6   # every orphan maps to itself, and nothing else does
    assert np.array_equal(perm[perm], np.arange(design.size))


def test_a_voxel_on_the_plane_is_its_own_partner_and_not_an_orphan():
    labels = np.zeros((1, 5, 1), dtype=np.uint8) + DESIGN   # centre parity: k = 5, voxel 2 is on the plane
    perm, causes = symmetry.design_map(labels, 1, 5)
    assert perm.tolist() == [4, 3, 2, 1, 0] and sum(causes.values()) == 0


def test_label_mismatches_count_voxels_not_pairs_and_compare_outside_with_empty():
    labels = np.zeros((1, 6, 1), dtype=np.uint8)
    labels[0, :, 0] = [0, DESIGN, DESIGN, DESIGN, 0, 0]   # plane k = 4: i -> 3 - i; 4 and 5 map outside
    assert symmetry.label_mismatches(labels, 1, 4) == 2   # 1 <-> 2 are equal; 0 <-> 3 differ (two voxels, one pair)
    labels[0, 5, 0] = DESIGN   # a design voxel with its image outside the grid
    assert symmetry.label_mismatches(labels, 1, 4) == 3


# --- the model -----------------------------------------------------------------------------------


@pytest.mark.parametrize("planes", [[{"normal": "y", "at_mm": 0.0}], two_planes()])
def test_density_is_the_filter_of_the_averaged_x(tmp_path, planes):
    m = make_model(write_yaml(tmp_path, planes))
    perms = symmetry.perms_of(m)
    assert len(perms) == len(planes)
    x = np.random.default_rng(2).uniform(0.2, 0.9, m.n_design)
    ref = x.copy()
    for perm in perms:
        ref = 0.5 * (ref + ref[perm])
    rho_tilde, rho = m.density(x, 2.0)
    np.testing.assert_allclose(rho_tilde, m.F @ ref + m.b, rtol=0, atol=1e-15)
    for perm in perms:   # symmetric labels: the physical density is symmetric to rounding
        assert np.abs(rho_tilde - rho_tilde[perm]).max() < 1e-12
        assert np.abs(rho - rho[perm]).max() < 1e-12
    m.solver.free()


def test_a_symmetric_x_gives_the_density_of_the_model_without_mirror(tmp_path):
    """The grid does not move (the planes are on grid planes at the smaller shift), so the two models agree."""
    m1 = make_model(write_yaml(tmp_path, two_planes()))
    sub = tmp_path / "plain"
    sub.mkdir()
    m0 = make_model(write_yaml(sub, None))
    assert m0.n_design == m1.n_design
    p1, p2 = symmetry.perms_of(m1)
    index = np.arange(m1.n_design)
    orbit = np.minimum.reduce([index, p1, p2, p1[p2]])   # one value for each orbit of the two maps: x is symmetric to the bit
    x = np.random.default_rng(3).uniform(0.2, 0.9, m1.n_design)[orbit]
    assert np.array_equal(x, x[p1]) and np.array_equal(x, x[p2])
    a, b = m1.density(x, 2.0), m0.density(x, 2.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])   # 0.5 (x + x) = x exactly
    m0.solver.free(), m1.solver.free()


def test_without_a_mirror_key_the_average_is_the_identity_object(tmp_path):
    m = make_model(write_yaml(tmp_path))
    x = np.arange(m.n_design, dtype=float)
    assert symmetry.perms_of(m) == () and symmetry.average(m, x) is x
    assert symmetry.average(m, x, transpose=True) is x
    rho_tilde, _ = m.density(np.full(m.n_design, 0.4), 1.0)
    np.testing.assert_array_equal(rho_tilde, m.F @ np.full(m.n_design, 0.4) + m.b)   # the line of main
    assert m.problem.mirror is None   # density records nothing
    m.solver.free()


def test_the_average_has_the_transpose_of_the_forward_step_with_orphans(tmp_path):
    """<S x, y> = <x, S^T y>. Two planes and orphans: the maps do not commute, so the reverse order is needed."""
    m = make_model(write_yaml(tmp_path, two_planes(), edit=keep_out_corner(tmp_path)))
    n_orphans = [int((p == np.arange(p.size)).sum()) for p in symmetry.perms_of(m)]
    assert min(n_orphans) > 0
    p1, p2 = symmetry.perms_of(m)
    assert not np.array_equal(p1[p2], p2[p1])   # the two maps do not commute
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=m.n_design), rng.normal(size=m.n_design)
    lhs = symmetry.average(m, x) @ y
    assert lhs == pytest.approx(x @ symmetry.average(m, y, transpose=True), rel=1e-12)
    wrong = x @ symmetry.average(m, y)   # the forward order is not the transpose
    assert abs(lhs - wrong) > 1e-6 * abs(lhs)
    m.solver.free()


def test_a_voxel_with_no_partner_stays_a_free_variable(tmp_path):
    m = make_model(write_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}], edit=keep_out_corner(tmp_path)))
    (perm,) = symmetry.perms_of(m)
    orphans = np.flatnonzero(perm == np.arange(perm.size))
    assert orphans.size > 0
    x = np.random.default_rng(1).uniform(size=m.n_design)
    out = symmetry.average(m, x)
    assert np.array_equal(out[orphans], x[orphans])   # exactly: 0.5 (a + a) = a
    pair = np.flatnonzero(perm != np.arange(perm.size))
    np.testing.assert_allclose(out[pair], 0.5 * (x[pair] + x[perm[pair]]), rtol=0, atol=0)
    prob = m.problem.mirror["planes"][0]["orphans"]
    assert prob["count"] == orphans.size and prob["keep_out"] == orphans.size
    assert prob["share"] == pytest.approx(orphans.size / m.n_design)
    m.solver.free()


# --- gradients -------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["one_plane", "two_planes", "two_planes_with_orphans"])
@pytest.mark.parametrize("beta", [1.0, 4.0])
def test_fd_of_the_chain_without_the_solve_at_an_asymmetric_x(tmp_path, variant, beta):
    """r(rho) = sum w rho and r(rho_tilde) = sum w rho_tilde, through the average, the filter and the projection."""
    planes = [{"normal": "y", "at_mm": 0.0}] if variant == "one_plane" else two_planes()
    edit = keep_out_corner(tmp_path) if variant.endswith("orphans") else None
    m = make_model(write_yaml(tmp_path, planes, edit=edit))
    rng = np.random.default_rng(7)
    x = rng.uniform(0.2, 0.8, m.n_design)   # not symmetric
    w = rng.uniform(0.5, 1.5, m.n_design)
    rho_tilde, rho = m.density(x, beta)
    got = m.gradient_wrt_x([w, w], ["rho", "rho_tilde"], rho_tilde, beta)
    eps = 1e-6
    for e in rng.choice(m.n_design, size=12, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[e] += eps
        xm[e] -= eps
        (tp, rp), (tm, rm) = m.density(xp, beta), m.density(xm, beta)
        fd = [(w @ (rp - rm)) / (2 * eps), (w @ (tp - tm)) / (2 * eps)]
        for row in range(2):
            assert abs(fd[row] - got[row, e]) <= 1e-6 * max(abs(got[row, e]), 1.0), (variant, beta, e, row)
    m.solver.free()


@pytest.mark.parametrize("variant", ["one_plane", "two_planes_with_orphans"])
def test_compliance_gradient_matches_finite_differences_at_an_asymmetric_x(tmp_path, variant):
    """The case `mixed` is not symmetric about the planes: the gradient of the compliance is not symmetric,
    and only the transposed average gives the gradient of the averaged design."""
    planes = [{"normal": "z", "at_mm": 0.0}] if variant == "one_plane" else two_planes()
    edit = keep_out_corner(tmp_path) if variant.endswith("orphans") else None
    m = make_model(write_yaml(tmp_path, planes, edit=edit))
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 0.7, m.n_design)
    beta = 2.0
    c0, dc, v0, dv = m.evaluate(x, beta)
    eps, tol = (1e-3, 1e-4) if m.solver_name == "amg" else (1e-5, 1e-5)
    for e in rng.choice(m.n_design, size=12, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[e] += eps
        xm[e] -= eps
        cp, _, vp, _ = m.evaluate(xp, beta)
        cm, _, vm, _ = m.evaluate(xm, beta)
        # The scale of a small slope is a share of the largest one: the noise floor of a central difference
        # (about 3e-14 / eps here) does not shrink with the slope (KB pitfalls/fd-check-noise-floor).
        assert abs((cp - cm) / (2 * eps) - dc[e]) / max(abs(dc[e]), 1e-2 * np.abs(dc).max()) < tol
        assert abs((vp - vm) / (2 * eps) - dv[e]) / max(abs(dv[e]), 1e-2 * np.abs(dv).max()) < tol
    m.solver.free()


def test_gradient_is_symmetric_for_every_wrt_kind(tmp_path):
    m = make_model(write_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}]))
    (perm,) = symmetry.perms_of(m)
    rng = np.random.default_rng(9)
    x = rng.uniform(0.2, 0.8, m.n_design)
    rho_tilde, rho = m.density(x, 2.0)
    g = rng.normal(size=m.n_design)
    out = m.gradient_wrt_x([g, g, g], ["rho", "rho_tilde", "x"], rho_tilde, 2.0)
    for row in out:
        assert np.array_equal(row, row[perm])   # exactly: a + b = b + a
    np.testing.assert_allclose(out[2], 0.5 * (g + g[perm]), rtol=0, atol=1e-15)   # the kind "x": the average only
    dproj = heaviside_deriv(rho_tilde, 2.0, ETA)
    dc, dv = m.chain_rule(g, rho_tilde, 2.0)
    assert np.array_equal(dc, dc[perm]) and np.array_equal(dv, dv[perm])
    ref = m.FT @ (g * dproj)
    np.testing.assert_allclose(dc, 0.5 * (ref + ref[perm]), rtol=0, atol=1e-14)
    m.solver.free()


@pytest.mark.parametrize("variant", ["one_plane", "two_planes_with_orphans"])
def test_mirror_with_the_robust_erosion_form(tmp_path, variant):
    """Order of operations with `settings.robust: eroded` (A3b): average, filter, then both projections.

    The eroded density is the projection of the same filtered field `rho_tilde`, so it is symmetric when
    `rho_tilde` is. The gradient with respect to `rho_stiff` goes through the filter transpose, and then the average
    gives the transpose of the first step. This test runs when A3b is in the code base.
    """
    robust = pytest.importorskip("topopt.robust")
    planes = [{"normal": "y", "at_mm": 0.0}] if variant == "one_plane" else two_planes()
    edit = keep_out_corner(tmp_path) if variant.endswith("orphans") else None
    m = make_model(write_yaml(tmp_path, planes, edit=edit, robust="eroded"))
    assert m.robust is not None
    rng = np.random.default_rng(11)
    x = rng.uniform(0.3, 0.7, m.n_design)
    beta = 4.0
    rho_tilde, rho = m.density(x, beta)
    np.testing.assert_array_equal(rho_tilde, m.F @ symmetry.average(m, x) + m.b)
    np.testing.assert_array_equal(m.stiffness_density(rho_tilde, rho, beta), m.robust.eroded(rho_tilde, beta))
    if variant == "one_plane":   # symmetric labels: the eroded density is symmetric to rounding
        (perm,) = symmetry.perms_of(m)
        assert np.abs(m.robust.eroded(rho_tilde, beta) - m.robust.eroded(rho_tilde, beta)[perm]).max() < 1e-12
    w = rng.uniform(0.5, 1.5, m.n_design)
    out = m.gradient_wrt_x([w, w], ["rho_stiff", "rho"], rho_tilde, beta)
    ref = m.FT @ (w * m.robust.eroded_slope(rho_tilde, beta))
    np.testing.assert_allclose(out[0], symmetry.average(m, ref, transpose=True), rtol=0, atol=1e-14)
    eps, tol = (1e-3, 1e-4) if m.solver_name == "amg" else (1e-5, 1e-5)
    c0, dc, v0, dv = m.evaluate(x, beta)   # the eroded compliance, and the nominal volume
    for e in rng.choice(m.n_design, size=12, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[e] += eps
        xm[e] -= eps
        cp, _, vp, _ = m.evaluate(xp, beta)
        cm, _, vm, _ = m.evaluate(xm, beta)
        # The scale of a small slope is a share of the largest one: the noise floor of a central difference
        # (about 3e-14 / eps here) does not shrink with the slope (KB pitfalls/fd-check-noise-floor).
        assert abs((cp - cm) / (2 * eps) - dc[e]) / max(abs(dc[e]), 1e-2 * np.abs(dc).max()) < tol
        assert abs((vp - vm) / (2 * eps) - dv[e]) / max(abs(dv[e]), 1e-2 * np.abs(dv).max()) < tol
    m.solver.free()


# --- a run ---------------------------------------------------------------------------------------------


def test_run_keeps_x_symmetric_and_reports_the_deviation(tmp_path, capsys):
    """Through the CLI, 4 iterations at 2 mm: x is symmetric to the bit, rho to rounding. A load case is not symmetric."""
    p = write_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}, {"normal": "z", "at_mm": 0.0}])
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    run = json.loads((out / "run.json").read_text())
    mirror = run["mirror"]
    assert [pl["axis"] for pl in mirror["planes"]] == ["y", "z"]
    assert all(pl["orphans"]["count"] == 0 for pl in mirror["planes"])
    # `x_deviation` is the deviation of the averaged design: exactly 0 for the last plane, always (a + b = b + a). With
    # no orphans the maps commute, and it is 0 for the other plane too.
    assert mirror["end"]["x_deviation"] == [0.0, 0.0]
    assert max(mirror["end"]["rho_deviation"]) < 1e-12
    down, side, mixed = mirror["load_asymmetry"]
    assert down["planes"][0] <= symmetry.SYMMETRIC_TOL and down["planes"][1] == pytest.approx(2.0)   # Fz flips about z = 0
    assert side["planes"] == [0.0, 0.0]
    assert mixed["planes"] == pytest.approx([2 * 0.7 / np.linalg.norm([1.0, 0.7, -1.5]),
                                             2 * 1.5 / np.linalg.norm([1.0, 0.7, -1.5])], rel=1e-9)
    log = (out / "run.log").read_text()
    assert "mirror: plane y = 0 mm: parity" in log and "end of run max|x - x[perm]| = 0" in log
    assert "x is the averaged design that the filter reads" in log
    assert "load case 'down', plane z = 0 mm" in log and "not symmetric" in log
    snap = np.fromfile(out / "snapshots" / "it_0004.bin", dtype=np.uint8).reshape(run["shape"]).astype(int)
    labels = np.fromfile(out / "labels.bin", dtype=np.uint8).reshape(run["shape"])
    for axis, pl in enumerate(mirror["planes"], start=1):
        image = np.flip(snap, axis=axis)   # the grid is symmetric: k = n along both axes
        assert pl["k"] == run["shape"][axis] and np.abs(snap - image).max() <= 1
        assert np.array_equal(labels, np.flip(labels, axis=axis))


def test_a_run_without_mirror_writes_no_mirror_key_and_no_mirror_line(tmp_path):
    p = write_yaml(tmp_path)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert "mirror" not in json.loads((out / "run.json").read_text())
    assert "mirror: " not in (out / "run.log").read_text()   # (the folder name of the run can hold the word)


def test_check_prints_the_facts_and_the_json_has_them(tmp_path, capsys):
    p = write_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}], edit=keep_out_corner(tmp_path))
    assert main(["check", str(p)]) == 0
    text = capsys.readouterr().out
    assert re.search(r"^mirror:\s+plane y = 0 mm: parity", text, re.M)
    assert "design voxels have no design partner" in text and "keep_out" in text
    data = check_data(p)
    assert data["mirror"]["planes"][0]["orphans"]["keep_out"] > 0
    assert check_data(write_yaml(tmp_path, None, name="plain.yaml"))["mirror"] is None


def test_write_run_json_has_the_mirror_key_only_with_a_mirror(tmp_path):
    for mirror in (None, [{"normal": "y", "at_mm": 0.0}]):
        sub = tmp_path / ("a" if mirror is None else "b")
        sub.mkdir()
        prob = build_problem(load_config(write_yaml(sub, mirror)))
        (sub / "r").mkdir()
        w = RunWriter(sub / "r", quiet=True)
        w.write_run_json(prob, 1, 1, "cholesky")
        assert ("mirror" in json.loads((sub / "r" / "run.json").read_text())) == (mirror is not None)


# --- the other solver paths and the platform -------------------------------------------------------------


def test_the_map_is_plain_numpy_with_no_platform_branch():
    src = Path(symmetry.__file__).read_text()
    for word in ("platform", "darwin", "sys.", "pyopencl", "pardiso", "accelerate"):
        assert word not in src.lower(), word


def test_symmetry_module_loads_without_numpy():
    """`topopt.cli` imports `config`, and `config` imports `symmetry`: numpy must not load for it."""
    import subprocess
    import sys
    code = "import sys, topopt.cli, topopt.symmetry; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == "False"


# --- the texts -----------------------------------------------------------------------------------------------


def section(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i + len(start))]


def test_skill_and_server_texts_name_the_key_as_facts_and_options():
    skill = (ROOT / ".claude/skills/part-to-topopt/SKILL.md").read_text()
    part = section(skill, "### The `mirror` key", "\n### ")
    server = (ROOT / "topopt_mcp/server.py").read_text()
    builder_doc = section(server, "def build_regions", '"""\n    try:')
    check_doc = section(server, "def check_problem", '"""\n    yaml = ')
    from topopt_mcp import replies
    check_doc += "\n" + replies.guide("check_problem")      # P9-11: the lines that explain the keys of the reply moved to the guide topopt://replies/check_problem
    for text in (part, builder_doc, check_doc):
        lines = [ln for ln in text.splitlines() if "mirror" in ln.lower()]
        assert lines
        assert not re.search(r"\b(must|never|always)\b", " ".join(lines).lower()), lines   # facts and options, no habitual caps
    assert not re.search(r"\b(must|never|always)\b", part.lower())
    for word in ("at_mm", "align", "orphan", "re-orient", "pull_off"):
        assert word in part, word
    assert "at_mm" in builder_doc and "mirror" in check_doc


def test_builder_passes_the_mirror_key_through_and_prints_the_facts(tmp_path, capsys):
    import importlib.util
    spec_ = importlib.util.spec_from_file_location(
        "build_regions", ROOT / ".claude/skills/part-to-topopt/build_regions.py")
    br = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(br)
    spec = {"name": "bar", "regions": {
                "design": [{"box": {"min": [0, -10, 0], "max": [40, 10, 10]}}],
                "fixed": {"wall": [{"box": {"min": [-4, -10, 0], "max": [0, 10, 10]}}]},
                "load": {"tip": [{"box": {"min": [36, -3, 0], "max": [40, 3, 10]}}]}},
            "load_cases": [{"name": "down", "forces": [{"region": "tip", "force": [0, 0, -10.0]}]}],
            "mirror": [{"normal": "y", "at_mm": 0.0}],
            "settings": {"volume_fraction": 0.3, "voxel_size": 2.0}}
    sp = tmp_path / "spec.yaml"
    sp.write_text(yaml.safe_dump(spec))
    assert br.main([str(sp), "--out", str(tmp_path / "o")]) == 0
    doc = yaml.safe_load((tmp_path / "o" / "bar.yaml").read_text())
    assert doc["mirror"] == [{"normal": "y", "at_mm": 0.0}]
    out = capsys.readouterr().out
    assert re.search(r"^mirror:\s+plane y = 0 mm: parity", out, re.M) and "RESULT: OK" in out
    spec["mirror"] = [{"normal": [1, 1, 0], "at_mm": 0.0}]
    sp.write_text(yaml.safe_dump(spec))
    assert br.main([str(sp), "--out", str(tmp_path / "o2")]) == 1
    assert "not axis-aligned" in capsys.readouterr().out


def test_standalone_check_prints_the_facts(tmp_path, capsys):
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("mirror_check", ROOT / "tools" / "mirror_check.py")
    mc = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mc)
    p = write_yaml(tmp_path, two_planes(), edit=keep_out_corner(tmp_path))
    assert mc.main([str(p), "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "involution: ok" in out and "|F P x - P F x|" in out and "orphans" in out and "parity" in out


# --- fixes of the first review: the kind x, the MMA rows, orphans with two planes, the grid plane check --------------


def mma_rows(tmp_path, planes, edit=None, **kw):
    """A model, its formulation with extra rows, and `rows(x)`: the values and the gradients that `Mma.step` gets.

    The rows are the objective, the rows of the YAML, a row on `rho_tilde`, and a row of kind "x" that reads `point.x`.
    """
    from topopt.formulation import Constraint, Formulation, Response
    from topopt.mma import Mma
    from topopt.optimize import formulation_from_config
    cfg = load_config(write_yaml(tmp_path, planes, edit=edit, **kw))
    m = Model(build_problem(cfg))
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13
    n = m.n_design
    w = np.random.default_rng(3).uniform(0.5, 1.5, n)
    base = formulation_from_config(cfg, n)
    extra = [Constraint("rho_tilde_row", lambda p: Response(float(w @ p.rho_tilde) / n, w, wrt="rho_tilde", divisor=n),
                        limit=0.5),
             Constraint("x_row", lambda p: Response(float(w @ p.x) / n, w, wrt="x", divisor=n), limit=0.5)]
    form = Formulation(base.objective, base.constraints + extra, xmin=base.xmin)
    state = Mma(n, form.n_rows, 0.2, xmin=form.xmin).state
    names = ["f0"] + [f"g{i}" for i in range(form.n_rows)]

    def rows(x, beta=2.0):
        rho_tilde, rho = m.density(x, beta)
        point = m.design_point(x, beta, rho_tilde, rho, m.solve(m.stiffness(rho)))
        ev = form.chain_rule(point, form.respond(point, state))
        return np.concatenate([[ev.f0], ev.g]), np.vstack([ev.df0[None, :], ev.dg])
    return m, rows, names


@pytest.mark.parametrize("variant", ["one_plane", "two_planes_with_orphans"])
def test_fd_through_the_mma_rows_at_an_asymmetric_x(tmp_path, variant):
    """The rows that `Mma.step` gets (objective, constraint rows, a `rho_tilde` row, a row of kind "x") against central
    differences of their values, at an x that is not symmetric. The kind "x" row reads `point.x`: the averaged design.
    Without that, its gradient (averaged by the transpose) is the gradient of another function (relative error 0.4 to 0.5).
    """
    if variant == "one_plane":
        planes, edit, kw = [{"normal": "y", "at_mm": 0.0}], None, {}
    else:   # two planes, orphans (the maps do not commute), the max form with a compliance limit on `mixed`
        ko = keep_out_corner(tmp_path)

        def edit(d):
            ko(d)
            d["load_cases"][2]["compliance_limit"] = 50.0
        planes, kw = two_planes(), {"case_combine": "max"}
    m, rows, names = mma_rows(tmp_path, planes, edit, **kw)
    rng = np.random.default_rng(5)
    x = rng.uniform(0.25, 0.75, m.n_design)
    _, D = rows(x)
    eps, tol = (1e-3, 1e-4) if m.solver_name == "amg" else (1e-5, 1e-4)
    worst = np.zeros(len(names))
    for e in rng.choice(m.n_design, size=10, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[e] += eps
        xm[e] -= eps
        fd = (rows(xp)[0] - rows(xm)[0]) / (2 * eps)
        for r in range(len(names)):
            scale = max(abs(D[r, e]), 1e-2 * np.abs(D[r]).max(), 1e-300)   # KB pitfalls/fd-check-noise-floor
            worst[r] = max(worst[r], abs(fd[r] - D[r, e]) / scale)
    assert worst.max() < tol, dict(zip(names, worst))
    m.solver.free()


def test_design_point_x_is_the_averaged_design(tmp_path):
    m = make_model(write_yaml(tmp_path, two_planes(), edit=keep_out_corner(tmp_path)))
    x = np.random.default_rng(2).uniform(0.2, 0.8, m.n_design)
    rho_tilde, rho = m.density(x, 2.0)
    point = m.design_point(x, 2.0, rho_tilde, rho, m.solve(m.stiffness(rho)))
    assert np.array_equal(point.x, symmetry.average(m, x)) and not np.array_equal(point.x, x)
    m.solver.free()
    plain = make_model(write_yaml(tmp_path, None, name="plain.yaml"))
    x = np.random.default_rng(2).uniform(0.2, 0.8, plain.n_design)
    rho_tilde, rho = plain.density(x, 2.0)
    assert plain.design_point(x, 2.0, rho_tilde, rho, plain.solve(plain.stiffness(rho))).x is x   # no key: the same object
    plain.solver.free()


def test_end_values_with_two_planes_and_orphans_are_about_the_averaged_design(tmp_path):
    """The maps of the two planes do not commute near the keep-out corner. The optimizer variable x is symmetric about
    the first plane only (its deviation about the second plane is large). The averaged design is exactly symmetric about
    the last plane, and its deviation about the first plane is a fact of the design (about the density near the orphans)."""
    p = write_yaml(tmp_path, two_planes(), edit=keep_out_corner(tmp_path), iterations=8,
                   beta_schedule=[[0, 1], [4, 2]])
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    mirror = json.loads((out / "run.json").read_text())["mirror"]
    assert [pl["orphans"]["count"] > 0 for pl in mirror["planes"]] == [True, True]
    end = mirror["end"]
    assert end["x_deviation"][-1] == 0.0   # a + b = b + a: exact for the last plane, whatever x is
    assert max(end["rho_deviation"]) > 1e-6   # the density is not symmetric near the orphans: the fact is reported
    # the raw optimizer variable of the same run: symmetric about the first plane only
    from types import SimpleNamespace
    from topopt.optimize import run
    problem = build_problem(load_config(p))
    perms = symmetry.perms_of(SimpleNamespace(problem=problem))
    writer = RunWriter(tmp_path / "r2", quiet=True)
    writer.clean()
    x = run(problem, writer)
    # Bit equality of x about the first plane holds because each per-variable operation of `topopt.mma_sub` is element-wise
    # (a variable and its image have the same data) and the sums over the variables are scalars that all variables share
    # (`test_two_variables_with_equal_data_get_equal_bits` in test_mma_sub.py holds for m = 1, 2 and 4). A matrix product over the
    # rows would depend on the BLAS kernel for two or more rows (review n8); `mma_sub` sums over the rows in a loop. This problem
    # has one row; keep 0.0 pinned for it.
    assert np.abs(x - x[perms[0]]).max() == 0.0
    assert np.abs(x - x[perms[1]]).max() > 0.1    # the raw x is not the design: this is why the end value is of A x


def test_far_plane_text_says_the_mirror_has_no_effect_and_gives_the_options(tmp_path):
    prob = build_problem(load_config(write_yaml(tmp_path, [{"normal": "y", "at_mm": 100.0}])))
    pl = prob.mirror["planes"][0]
    assert pl["orphans"]["count"] == prob.mirror["design_voxels"] and pl["orphans"]["outside_grid"] == pl["orphans"]["count"]
    lo, hi = pl["grid_range_mm"]
    assert lo == pytest.approx(prob.origin[1]) and hi == pytest.approx(prob.origin[1] + prob.shape[1] * prob.h)
    text = "\n".join(symmetry.summary_lines(prob.mirror))
    assert "The mirror has no effect on this plane" in text and "Options: check at_mm, move the plane" in text
    assert "The grid spans y = " in text
    assert "the design is symmetric" not in text and "best symmetric design" not in text
    assert "(no design voxel has a partner)" in text   # the load verdict of a case that is not symmetric
    assert "100.00 percent" in text
    prob.mirror["end"] = {"x_deviation": [0.0], "rho_deviation": [0.0]}   # all voxels map to themselves: 0 says nothing
    assert "say nothing about symmetry" in "\n".join(symmetry.end_lines(prob.mirror))


def test_most_orphans_get_the_words_little_effect_and_a_few_do_not(tmp_path):
    near = build_problem(load_config(write_yaml(tmp_path, [{"normal": "y", "at_mm": 3.0}], name="near.yaml")))
    share = near.mirror["planes"][0]["orphans"]["share"]
    assert symmetry.ORPHAN_MOST <= share < 1.0
    text = "\n".join(symmetry.summary_lines(near.mirror))
    assert "the mirror has little effect on this plane" in text and "Options:" in text
    assert "no effect on this plane" not in text
    assert "symmetric on the paired voxels" in text and "the design is symmetric," not in text
    ok = build_problem(load_config(write_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}], name="ok.yaml")))
    assert ok.mirror["planes"][0]["orphans"]["count"] == 0
    assert "Options:" not in "\n".join(symmetry.summary_lines(ok.mirror))
    some = build_problem(load_config(write_yaml(tmp_path, two_planes(), edit=keep_out_corner(tmp_path), name="s.yaml")))
    assert 0 < some.mirror["planes"][0]["orphans"]["share"] < symmetry.ORPHAN_MOST
    t = "\n".join(symmetry.summary_lines(some.mirror))
    assert "Options:" not in t and "symmetric on the paired voxels" in t


def test_load_text_names_the_load_voxels_that_have_no_partner():
    plane = {"axis": "y", "at_mm": 0.0, "align": "auto", "parity": "centre", "k": 5, "plane_index": 2,
             "label_mismatches": {"face": None, "centre": 0}, "grid_range_mm": [-5.0, 5.0],
             "orphans": {"count": 12, "share": 0.01, "outside_grid": 0, "empty": 0, "keep_out": 0, "fixed": 0,
                         "load": 12, "other": 0}}
    mirror = {"planes": [plane], "design_voxels": 1200,
              "load_asymmetry": [{"case": "hang", "planes": [0.447]}, {"case": "up", "planes": [0.0]}]}
    lines = symmetry.summary_lines(mirror)
    hang = next(l for l in lines if "'hang'" in l)
    assert "12 design voxels have a load voxel as their image (cause load)" in hang and "not symmetric" in hang
    assert "(0 is symmetric, 2 is antisymmetric)" in hang
    assert "cause load" not in next(l for l in lines if "'up'" in l)   # a symmetric value needs no note


def test_orphan_causes_add_up_to_the_orphan_count_with_a_label_code_that_has_no_name():
    labels = np.zeros((1, 6, 1), dtype=np.uint8)
    labels[0, :, 0] = [DESIGN, DESIGN, 0, 7, DESIGN, DESIGN]   # plane k = 6: i -> 5 - i. Voxel 1 has the image 4 (design)
    labels[0, 0, 0], labels[0, 5, 0] = DESIGN, 7               # voxel 0 has an image with the code 7
    perm, causes = symmetry.design_map(labels, 1, 6)
    assert causes["other"] == 1
    n_orphans = int((perm == np.arange(perm.size)).sum())
    assert sum(causes.values()) == n_orphans


def test_a_plane_that_is_not_a_grid_plane_stops_with_the_numbers(tmp_path):
    """A wrong origin (for example a merge that loses the shift in `voxelize`) must not give a map about another plane."""
    from topopt.loads import build_forces
    from topopt.voxelize import voxelize
    cfg, info = symmetry.resolve(load_config(write_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}])))
    origin, h, shape, labels, ids = voxelize(cfg)
    forces = build_forces(cfg, ids)
    names = [c.name for c in cfg.load_cases]
    assert symmetry.problem_facts(info, origin, h, labels, forces, names)["planes"][0]["k"] > 0   # the right origin
    wrong = origin + np.array([0.0, 0.25 * h, 0.0])
    with pytest.raises(RuntimeError) as e:      # an internal error: the CLI exit code is 2 (CONTRACTS section 8)
        symmetry.problem_facts(info, wrong, h, labels, forces, names)
    assert not isinstance(e.value, UserError)
    text = str(e.value)
    assert "\n" not in text and text.startswith("the mirror plane y = 0 mm is not a grid plane")
    assert "half voxels above the origin" in text and "origin" in text and "nearest" in text


@pytest.mark.parametrize("h, half_y", [(1.0, 4.5), (1.0, 5.0), (0.7, 4.37)])
def test_a_plane_far_from_the_origin_gets_the_parity_and_the_shape_of_the_same_part_at_zero(tmp_path, h, half_y):
    """STL vertices are float32: at y = -123.456 mm the rounding is about 8e-6 mm, above 1e-6 half voxels. The grid plane
    tolerance (GRID_TOL h) covers it. Without it, a region face on a voxel face would add one EMPTY layer (k = 11 for 9)."""
    import trimesh

    def make(c):
        sub = tmp_path / f"c{c}"
        sub.mkdir()
        for name, lo, hi in (("d", (0, c - half_y, -4), (16, c + half_y, 4)), ("p", (-2, c - half_y, -4), (0, c + half_y, 4)),
                             ("t", (14, c - 1.7, -2), (16, c + 1.7, 2)), ("k", (6, c - 1.3, 1), (9, c + 1.3, 4))):
            mesh = trimesh.creation.box(extents=np.asarray(hi, float) - lo)
            mesh.apply_translation((np.asarray(lo, float) + hi) / 2)
            mesh.vertices = mesh.vertices.astype(np.float32).astype(np.float64)   # the float32 rounding of an STL file
            mesh.export(sub / f"{name}.stl")
        d = {"name": "o", "regions": {"design": "d.stl", "fixed": ["p.stl"], "keep_out": ["k.stl"], "load": {"tip": "t.stl"}},
             "load_cases": [{"name": "a", "forces": [{"region": "tip", "force": [0.5, 0.0, -2.0]}]}],
             "settings": {"voxel_size": h, "volume_fraction": 0.4, "iterations": 2}, "mirror": [{"normal": "y", "at_mm": c}]}
        (sub / "y.yaml").write_text(yaml.safe_dump(d))
        return build_problem(load_config(sub / "y.yaml"))

    ref = make(0.0)
    for c in (7.3, -123.456):
        prob = make(c)
        a, b = ref.mirror["planes"][0], prob.mirror["planes"][0]
        assert (b["parity"], b["k"], b["label_mismatches"], b["orphans"]["count"]) == \
               (a["parity"], a["k"], a["label_mismatches"], a["orphans"]["count"]), c
        assert prob.shape == ref.shape and prob.mirror["design_voxels"] == ref.mirror["design_voxels"]
