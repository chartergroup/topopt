"""Tests for `settings.robust: wall` (CONTRACTS.md sections 22a and 22b): the void-padded filter, the
length L(eta_e) of the wall rule, the radius rule, the YAML checks, the records and one short run.

The filter tests compare `build_filter` with a dense reference that is written from the rule in the
text, voxel by voxel, so the expected numbers do not come from the code under test.
"""

import hashlib

import numpy as np
import pytest

from topopt.filter import build_filter, solid_pull
from topopt.voxelize import DESIGN, FIXED, KEEP_OUT, LOAD

# --- the padded filter -------------------------------------------------------------------------------------


def _labels():
    """A small grid with every kind of neighbour: a fixed face, a load voxel, a keep-out block, design voxels, the grid border."""
    lab = np.full((11, 7, 6), DESIGN, dtype=np.int8)
    lab[0:2] = FIXED
    lab[9:11, 3:5, 2:4] = LOAD
    lab[5:7, 2:5, 0:3] = KEEP_OUT
    lab[3, 5, 4] = KEEP_OUT
    return lab


def _reference(lab, h, radius, rule, pad):
    """Dense filter by the rule of the text. Return (F, b) with F of shape (n, n)."""
    R = radius / h
    m = int(np.ceil(R))
    design = np.argwhere(lab == DESIGN)
    index = {tuple(p): k for k, p in enumerate(design)}
    n = len(design)
    F, b = np.zeros((n, n)), np.zeros(n)
    for k, p in enumerate(design):
        total = 0.0
        for d in np.ndindex(2 * m + 1, 2 * m + 1, 2 * m + 1):
            off = np.array(d) - m
            dist = np.sqrt((off ** 2).sum())
            if dist >= R:
                continue
            w = R - dist
            q = tuple(p + off)
            inside = all(0 <= q[a] < lab.shape[a] for a in range(3))
            kind = lab[q] if inside else None
            if kind == DESIGN:
                F[k, index[q]] += w
                total += w
            elif kind in (FIXED, LOAD):
                if rule == "density_one":   # density 1: in the sum and in the weights. ignore: in neither
                    b[k] += w
                    total += w
            elif pad:                # void: keep-out, outside the design space, outside the grid. It counts in the weights with pad only
                total += w
        F[k] /= total
        b[k] /= total
    return F, b


@pytest.mark.parametrize("rule", ["density_one", "ignore"])
@pytest.mark.parametrize("pad", [True, False])
def test_the_filter_matches_the_dense_reference(rule, pad):
    """The pad rule (void for keep-out, outside, grid border; solid by its rule) and the old rule, both against a dense reference."""
    lab = _labels()
    F, FT, b = build_filter(lab, 1.0, 2.6, rule, pad_void=pad)
    Fr, br = _reference(lab, 1.0, 2.6, rule, pad)
    np.testing.assert_allclose(F.toarray(), Fr, rtol=0, atol=1e-13)
    np.testing.assert_allclose(b, br, rtol=0, atol=1e-13)
    assert np.array_equal(FT.toarray(), F.toarray().T)


def test_a_wall_on_a_face_has_the_filtered_value_of_an_inner_wall():
    """(a) With the padding a wall of n voxels at the face x = 0 has the profile of the same wall inside the design space."""
    lab = np.full((40, 13, 13), DESIGN, dtype=np.int8)
    F, _, b = build_filter(lab, 1.0, 4.5, "density_one", pad_void=True)
    F0, _, _ = build_filter(lab, 1.0, 4.5, "density_one", pad_void=False)
    for n in (2, 3, 5):
        def profile(M, start):
            x = np.zeros(lab.shape)
            x[start:start + n] = 1.0
            return (M @ x.reshape(-1) + b).reshape(lab.shape)[:, 6, 6]
        inner, face, face_old = profile(F, 20), profile(F, 0), profile(F0, 0)
        np.testing.assert_allclose(face[:12], inner[20:32], rtol=0, atol=1e-12)    # from the wall outward
        assert face_old[0] > face[0] + 0.05                                        # the old rule: the face acts as a mirror


def test_fixed_and_load_voxels_stay_solid_with_the_padding():
    """(b) density_one: a design voxel next to a fixed or a load voxel is pulled to 1 through b (not to void). ignore: no pull, and no weight."""
    lab = _labels()
    F1, _, b1 = build_filter(lab, 1.0, 2.6, "density_one", pad_void=True)
    F2, _, b2 = build_filter(lab, 1.0, 2.6, "ignore", pad_void=True)
    assert b1.max() > 0.2 and np.all(b2 == 0.0)
    design = np.argwhere(lab == DESIGN)
    k = next(i for i, p in enumerate(design) if tuple(p) == (2, 3, 3))   # next to the fixed block
    assert b1[k] > 0.15
    # density 1 everywhere gives 1 next to the solid with density_one (solid pulls up), and less than 1 with ignore (the weight is not renormalised)
    ones = np.ones(len(design))
    assert (F1 @ ones + b1)[k] > (F2 @ ones + b2)[k]
    assert (F2 @ ones + b2)[k] < 1.0   # void at the grid border and at the keep-out counts in the weights


def test_a_keep_out_neighbour_counts_as_void_with_the_padding():
    """(c) One design voxel beside a keep-out block: with density 1 on all design voxels its value is below 1 by the weight share of the block."""
    lab = np.full((9, 9, 9), DESIGN, dtype=np.int8)
    lab[5:9, :, :] = KEEP_OUT
    F, _, b = build_filter(lab, 1.0, 2.5, "density_one", pad_void=True)
    Fo, _, _ = build_filter(lab, 1.0, 2.5, "density_one", pad_void=False)
    n = F.shape[0]
    design = np.argwhere(lab == DESIGN)
    k = next(i for i, p in enumerate(design) if tuple(p) == (4, 4, 4))
    assert (Fo @ np.ones(n))[k] == pytest.approx(1.0)       # old rule: the keep-out block is ignored, the row sums to 1
    assert (F @ np.ones(n))[k] < 0.8                        # padded: the block is void, the row sums to less than 1


def test_the_old_filter_is_unchanged_with_the_padding_off():
    """(d) The default call and pad_void=False give the arrays that the code of main b274485 gave (sha256 of the arrays)."""
    lab = _labels()
    for rule, digest in (("density_one", OLD_DENSITY_ONE), ("ignore", OLD_IGNORE)):
        F, FT, b = build_filter(lab, 1.0, 2.6, rule)
        F2, FT2, b2 = build_filter(lab, 1.0, 2.6, rule, pad_void=False)
        assert _digest(F, FT, b) == _digest(F2, FT2, b2) == digest


def _digest(F, FT, b):
    h = hashlib.sha256()
    for a in (F.indptr, F.indices, F.data, FT.indptr, FT.indices, FT.data, b):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


OLD_DENSITY_ONE = "d1aaefaa3ed41123cc3ba0c3e394f677470283a189ba1d683432070736e99715"   # made by the code of main b274485
OLD_IGNORE = "e5d3a5a3f6dd593dc62fb36c21500e35d350e5a34438df3a1d2e13057e89da6e"


def test_solid_pull_follows_the_padded_filter():
    """The check line `solid pull` reads the same b as the filter that runs, also with the padding."""
    lab = _labels()
    for pad in (False, True):
        _, _, b = build_filter(lab, 1.0, 2.6, "density_one", pad_void=pad)
        pull = solid_pull(lab, 1.0, 2.6, "density_one", pad_void=pad)
        assert pull["b_max"] == pytest.approx(round(float(b.max()), 4), abs=1e-4)
        assert pull["voxels_above"] == int((b > 0.1).sum())


# --- L(eta_e) and the radius rule ---------------------------------------------------------------------------

from topopt import members  # noqa: E402

# The values of the step 1 theory script (ls_theory_ref.py, column "wall: nominal L/R"), 3 digits.
L_REFERENCE = {0.55: 0.406, 0.6: 0.566, 0.65: 0.687, 0.7: 0.789, 0.75: 0.882, 0.8: 0.976}


@pytest.mark.parametrize("eta_e", sorted(L_REFERENCE))
def test_the_nominal_width_per_radius_matches_the_reference(eta_e):
    assert members.nominal_width_per_radius(eta_e) == pytest.approx(L_REFERENCE[eta_e], abs=6e-4)


@pytest.mark.parametrize("eta_e", [0.55, 0.7, 0.8])
def test_the_nominal_width_per_radius_matches_a_numeric_plate(eta_e):
    """An independent check: filter a thin plate with the 1D profile of the 3D hat kernel (numeric convolution) and measure it."""
    s = np.linspace(-3.0, 3.0, 24001)
    ds = s[1] - s[0]
    a_abs = np.abs(s)
    k = np.where(a_abs < 1, 1 - 3 * a_abs ** 2 + 2 * a_abs ** 3, 0.0)
    k /= k.sum() * ds
    lo, hi = 0.0, 2.0
    for _ in range(40):
        a = 0.5 * (lo + hi)
        lo, hi = (a, hi) if k[a_abs <= a / 2].sum() * ds < eta_e else (lo, a)
    design = (a_abs <= 0.25 * (lo + hi)).astype(float)   # half of the thickness a
    filtered = np.convolve(design, k, mode="same") * ds
    inside = np.flatnonzero(filtered >= 0.5)
    width = s[inside[-1]] - s[inside[0]] + ds
    assert members.nominal_width_per_radius(eta_e) == pytest.approx(width, abs=2e-3)


def test_the_radius_of_the_wall_rule():
    """w 1.2 mm at the default eta_e 0.7: R = 1.2 / 0.7885 = 1.522 mm; the floor of 1.5 voxels wins when it is larger."""
    r = members.wall_radius(1.2, 0.7, 0.5)
    assert r["radius_mm"] == pytest.approx(1.5219, abs=1e-3) and not r["floor_wins"]
    assert r["nominal_width_per_radius"] == pytest.approx(0.7885, abs=1e-3)
    r = members.wall_radius(1.2, 0.7, 1.2)           # 1.5 voxels = 1.8 mm
    assert r["radius_mm"] == pytest.approx(1.8) and r["floor_wins"] and r["radius_from_wall_mm"] == pytest.approx(1.5219, abs=1e-3)


# --- the YAML: the mode, the errors, the derived radius and eta_e ---------------------------------------------

import json  # noqa: E402

import yaml  # noqa: E402

from test_solid_filter import write_case  # noqa: E402
from topopt import robust  # noqa: E402
from topopt.cli import check_data, main  # noqa: E402
from topopt.config import UserError, load_config  # noqa: E402


def wall_case(tmp_path, wall=1.2, h=0.5, **settings):
    """The cantilever of the solid-filter tests with `print.min_wall_mm` (when `wall` is not None) and the given settings."""
    p = write_case(tmp_path, h=h, **settings)
    if wall is not None:
        d = yaml.safe_load(p.read_text())
        d["print"] = {"min_wall_mm": wall}
        p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def test_robust_wall_without_min_wall_mm_is_an_error_with_options(tmp_path):
    p = wall_case(tmp_path, wall=None, robust="wall")
    with pytest.raises(UserError) as e:
        load_config(p)
    text = str(e.value)
    assert "'settings.robust' is wall, and 'print.min_wall_mm' is not set" in text
    assert "set print.min_wall_mm" in text and "set settings.robust to eroded" in text


def test_the_radius_at_wall_1_2_and_the_default_eta_is_1_522_mm(tmp_path):
    """w 1.2 mm, eta_e 0.7 (the default of the mode): R = 1.2 / L(0.7) = 1.5219 mm; the source is `wall`."""
    st = load_config(wall_case(tmp_path, robust="wall")).settings
    assert (st.robust, st.filter_radius_source, st.wall_mm) == ("wall", "wall", 1.2)
    assert st.filter_radius == pytest.approx(1.2 / members.nominal_width_per_radius(0.7), rel=1e-12)
    assert st.filter_radius == pytest.approx(1.5219, abs=1e-3)
    assert robust.eta_eroded_of(st) == (0.7, "wall_rule")
    assert st.eta_eroded is None            # the default is applied where it is read, so the YAML shows what the author wrote


def test_the_voxel_floor_of_1_5_voxels_wins_when_it_is_larger(tmp_path):
    st = load_config(wall_case(tmp_path, h=1.2, robust="wall")).settings     # 1.5 voxels = 1.8 mm > 1.522 mm
    assert (st.filter_radius_source, st.filter_radius) == ("wall", pytest.approx(1.8))
    facts = check_data(wall_case(sub_dir(tmp_path, "f"), h=1.2, robust="wall"))["derived"]["filter"]["facts"]
    assert any("below the default radius of 1.5 voxels" in f for f in facts)


def sub_dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def test_eta_eroded_and_filter_radius_override_the_wall_rule(tmp_path):
    st = load_config(wall_case(tmp_path, robust="wall", eta_eroded=0.6)).settings
    assert robust.eta_eroded_of(st) == (0.6, "settings")
    assert st.filter_radius == pytest.approx(1.2 / members.nominal_width_per_radius(0.6), rel=1e-12)
    p = wall_case(sub_dir(tmp_path, "r"), robust="wall", filter_radius=2.0)
    st = load_config(p).settings
    assert (st.filter_radius, st.filter_radius_source) == (2.0, "settings")     # the explicit radius wins
    d = check_data(p)["derived"]
    assert d["filter"]["wall_rule"]["radius_rule_mm"] == pytest.approx(1.5219, abs=1e-3)
    assert any("settings.filter_radius 2 mm is set, so it is used and the wall rule does not set the radius" in f for f in d["filter"]["facts"])
    assert d["filter"]["wall_rule"]["padded"] is True      # the padding stays on


def test_min_member_mm_is_only_reported_in_the_wall_mode(tmp_path):
    p = wall_case(tmp_path, robust="wall", min_member_mm=3.0)
    st = load_config(p).settings
    assert (st.filter_radius_source, st.min_member_mm) == ("wall", 3.0)
    assert st.filter_radius == pytest.approx(1.5219, abs=1e-3)               # not 1.877 * 3
    d = check_data(p)["derived"]["filter"]
    assert d["min_member_mm"] == 3.0 and any("min_member_mm is only reported" in f for f in d["facts"])
    assert "min_member_note" not in d     # its note (a skin at a free face) does not hold with the padding


def test_robust_eroded_does_not_change(tmp_path):
    """eroded keeps eta_e 0.55 and the radius rules, with or without print.min_wall_mm, and its filter is not padded."""
    st = load_config(wall_case(tmp_path, robust="eroded", min_member_mm=3.0)).settings
    assert robust.eta_eroded_of(st) == (0.55, "default")
    assert (st.filter_radius_source, st.wall_mm) == ("min_member_mm", None)
    assert st.filter_radius == pytest.approx(1.877 * 3.0)
    assert robust.from_settings(st).mode == "eroded"
    d = check_data(wall_case(sub_dir(tmp_path, "d"), robust="eroded"))["derived"]
    assert "wall_rule" not in d["filter"] and "padded_filter" not in d["robust"]


@pytest.mark.parametrize("mode", ["off", "eroded"])
def test_a_fact_when_min_wall_mm_is_set_and_the_mode_is_not_wall(tmp_path, mode):
    d = check_data(wall_case(tmp_path, robust=mode))
    facts = [f for f in d["derived"]["robust"]["facts"] if "does not aim at this wall" in f]
    assert len(facts) == 1 and facts[0] in d["warnings"]
    assert "settings.robust: wall would aim at it" in facts[0] and "1.522 mm" in facts[0] and "4.145 to 0.000 percent" in facts[0]
    assert "Options: set settings.robust to wall" in facts[0]
    assert d["derived"]["robust"]["on"] is (mode == "eroded")      # a fact: nothing stops


def test_no_such_fact_without_min_wall_mm_or_in_the_wall_mode(tmp_path):
    d = check_data(wall_case(tmp_path, wall=None, robust="eroded"))
    assert not [w for w in d["warnings"] if "does not aim at this wall" in w]
    d = check_data(wall_case(sub_dir(tmp_path, "w"), robust="wall"))
    assert not [w for w in d["warnings"] if "does not aim at this wall" in w]


def test_the_wall_mode_check_text(tmp_path, capsys):
    """The wall rule line says where R came from; the 'half as thin' claim is not printed (with padding it is false)."""
    assert main(["check", str(wall_case(tmp_path, robust="wall"))]) == 0
    out = capsys.readouterr().out
    assert "half as thin" not in out and "can be about 1 voxel" not in out
    assert "filter radius 1.5219" in out and "(the wall rule: filter_radius = max(w / L(eta_e), 1.5 voxels)" in out and "w 1.2 mm / L 0.7885" in out and "eta_e 0.7" in out
    assert "the filter is padded with void: a wall at a face of the design space is as thick as an inner wall" in out
    assert "filter size:" in out                                 # the memory estimate of the filter is still printed
    assert "robust:         wall rule (the eroded form), eta_n 0.5, eta_e 0.7 (default of the wall rule)" in out


def test_the_stress_and_buckling_stops_name_the_wall_mode(tmp_path):
    st = load_config(wall_case(tmp_path, robust="wall")).settings
    with pytest.raises(UserError, match="settings.robust is wall, and 1 load case"):
        robust.from_settings(st, 1)
    with pytest.raises(UserError, match="settings.robust is wall, and buckling.min_factor is set"):
        robust.from_settings(st, 0, buckling_row=True)


# --- the run ----------------------------------------------------------------------------------------------------


def test_a_short_wall_run_ends_done_with_the_records(tmp_path):
    """h 2 mm, w 4 mm: R = 4 / 0.7885 = 5.073 mm = 2.5 voxels (above the floor of 3 mm). The model's filter is the padded one."""
    p = wall_case(tmp_path, wall=4.0, h=2.0, robust="wall", iterations=4)
    out = tmp_path / "run"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert (out / "job.DONE").exists()
    run = json.loads((out / "run.json").read_text())
    f = run["derived"]["filter"]
    assert f["radius_source"] == "wall" and f["radius_mm"] == pytest.approx(4.0 / members.nominal_width_per_radius(0.7))
    w = f["wall_rule"]
    assert (w["wall_mm"], w["wall_source"], w["eta_e"], w["padded"], w["floor_wins"]) == (4.0, "print.min_wall_mm", 0.7, True, False)
    assert w["nominal_width_per_radius"] == pytest.approx(0.7885, abs=1e-3)
    r = run["derived"]["robust"]
    assert (r["mode"], r["on"], r["eta_e"], r["eta_e_source"], r["padded_filter"]) == ("wall", True, 0.7, "wall_rule", True)
    m = json.loads((out / "result" / "metrics.json").read_text())
    assert m["filter"]["radius_source"] == "wall" and m["filter"]["wall_rule"]["padded"] is True
    assert (m["robust"]["mode"], m["robust"]["eta_e"], m["robust"]["padded_filter"], m["robust"]["complete"]) == ("wall", 0.7, True, True)
    assert json.loads((out / "robust.json").read_text())["mode"] == "wall"
    log = (out / "run.log").read_text()
    assert "robust: wall rule (the eroded form)" in log and "half as thin" not in log


def test_the_model_of_a_wall_run_has_the_padded_filter_and_the_other_modes_do_not(tmp_path):
    from topopt.optimize import Model
    from topopt.problem import build_problem
    for mode, pad in (("wall", True), ("eroded", False), ("off", False)):
        cfg = load_config(wall_case(sub_dir(tmp_path, mode), wall=4.0, h=2.0, robust=mode))
        prob = build_problem(cfg)
        m = Model(prob)
        st = cfg.settings
        F, FT, b = build_filter(prob.labels, prob.h, st.filter_radius, st.filter_solid_neighbours, pad_void=pad)
        x = np.random.default_rng(1).random(m.n_design)          # the run applies the FFT filter (#333): the matrix numbers to rounding
        assert np.max(np.abs(m.F @ x - F @ x)) < 1e-12 and np.max(np.abs(m.FT @ x - FT @ x)) < 1e-12 and np.max(np.abs(m.b - b)) < 1e-12
        other = build_filter(prob.labels, prob.h, st.filter_radius, st.filter_solid_neighbours, pad_void=not pad)[0]
        assert np.max(np.abs(m.F @ x - other @ x)) > 1e-6       # the other padding gives other numbers: the check tells them apart
        m.solver.free()


def test_a_fine_pass_computes_the_wall_radius_again_for_its_voxel(tmp_path):
    """Round 1, finding 1: R = max(w / L, 1.5 voxels) follows the grid of the run. Where the floor won on the parent (h 2 mm: R = 3 mm),
    the fine voxel gives 1.522 mm at 0.5 mm and 1.875 mm (1.5 voxels) at 1.25 mm. Where w / L wins (w 4 mm: 5.073 mm) the radius stays."""
    from topopt import fine_pass as F
    rule = 1.2 / members.nominal_width_per_radius(0.7)
    cfg = load_config(wall_case(tmp_path, wall=1.2, h=2.0, robust="wall"))
    assert cfg.settings.filter_radius == pytest.approx(3.0)                       # the floor won on the parent
    for voxel, expected in ((0.5, rule), (1.25, 1.875), (1.0, rule)):
        fine = F.with_fine_settings(cfg, voxel, 5, 0).settings
        assert fine.filter_radius == pytest.approx(expected) and fine.filter_radius_source == "wall"
        assert (fine.robust, fine.wall_mm) == ("wall", 1.2)
    big = load_config(wall_case(sub_dir(tmp_path, "b"), wall=4.0, h=2.0, robust="wall"))
    assert F.with_fine_settings(big, 1.25, 5, 0).settings.filter_radius == pytest.approx(big.settings.filter_radius)   # w / L wins: unchanged in mm
    explicit = load_config(wall_case(sub_dir(tmp_path, "e"), wall=1.2, h=2.0, robust="wall", filter_radius=2.5))
    assert F.with_fine_settings(explicit, 0.5, 5, 0).settings.filter_radius == 2.5   # an explicit radius stays in mm
    default = load_config(wall_case(sub_dir(tmp_path, "d"), wall=4.0, h=2.0))      # the default radius follows the voxel, as before
    assert F.with_fine_settings(default, 0.7, 5, 0).settings.filter_radius == pytest.approx(1.05)


def test_the_fine_estimate_uses_the_radius_of_the_fine_grid(tmp_path):
    import dataclasses
    from topopt import fine_pass as F
    p = wall_case(tmp_path, wall=1.2, h=2.0, robust="wall")
    cfg, check = load_config(p), check_data(p)
    est = F.fine_estimates(check, cfg, 0.5, 5)
    rule = 1.2 / members.nominal_width_per_radius(0.7)
    same = dataclasses.replace(cfg, settings=dataclasses.replace(cfg.settings, filter_radius=rule, filter_radius_source="settings"))
    assert est["memory_bytes"] == F.fine_estimates(check, same, 0.5, 5)["memory_bytes"]       # the entries of a radius of 1.522 mm
    old = dataclasses.replace(cfg, settings=dataclasses.replace(cfg.settings, filter_radius_source="settings"))   # the coarse 3 mm
    assert est["memory_bytes"] < F.fine_estimates(check, old, 0.5, 5)["memory_bytes"]


def wall_parent(tmp_path, passes=4):
    p = wall_case(tmp_path / "case" if (tmp_path / "case").is_dir() else _mk(tmp_path / "case"), wall=1.2, h=2.0, robust="wall", iterations=passes)
    out = tmp_path / "parent"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    return p, out


def _mk(d):
    d.mkdir()
    return d


def _record(out):
    f = json.loads((out / "run.json").read_text())["derived"]["filter"]
    return f["radius_mm"], f["wall_rule"]["radius_rule_mm"], f["wall_rule"]["floor_wins"], f


def test_a_fine_run_of_a_wall_run_states_the_radius_it_uses_and_its_continuation_and_resume_keep_it(tmp_path):
    """Round 1, findings 1 and 7: parent at 2 mm (R 3 mm, floor won), fine pass at 1.25 mm (R 1.875 mm = 1.5 voxels).
    The record and the fact agree with the radius in use; a continuation of the fine run and a resume keep R, eta_e and the padding."""
    p, parent = wall_parent(tmp_path)
    R, rule, floor, _ = _record(parent)
    assert (R, rule, floor) == (pytest.approx(3.0), pytest.approx(3.0), True)
    assert main(["check", str(p), "--fine-from", str(parent), "--fine-voxel", "1.25"]) == 0       # no refusal: the radius is a soft change
    fine = tmp_path / "fine"
    assert main(["run", str(p), "--out", str(fine), "--quiet", "--fine-from", str(parent), "--fine-voxel", "1.25", "--fine-passes", "2"]) == 0
    R, rule, floor, f = _record(fine)
    assert R == pytest.approx(1.875) and rule == pytest.approx(R) and floor is True                  # record and rule agree
    assert any("nominal wall that survives the erosion is about 1.48 mm" in x for x in f["facts"])   # the fact states the radius in use: 0.7885 * 1.875
    log = (fine / "run.log").read_text()
    assert "filter radius 1.875 mm" in log
    assert "fine pass: the wall rule gives the filter radius 1.875 mm on the fine grid" in log and "the parent run used 3 mm" in log
    cont = tmp_path / "cont"
    assert main(["run", str(p), "--out", str(cont), "--quiet", "--continue-from", str(fine), "--more-iterations", "2"]) == 0
    assert _record(cont)[0] == pytest.approx(1.875)                                                  # the continuation of a fine pass is not refused
    assert main(["run", str(p), "--out", str(fine), "--quiet", "--resume"]) == 0
    assert _record(fine)[0] == pytest.approx(1.875)


def test_a_continuation_and_a_resume_of_a_wall_run_keep_the_radius_eta_and_padding(tmp_path):
    from topopt import checkpoint as C
    p, parent = wall_parent(tmp_path)
    before = json.loads((parent / "run.json").read_text())["derived"]
    cont = tmp_path / "cont"
    assert main(["run", str(p), "--out", str(cont), "--quiet", "--continue-from", str(parent), "--more-iterations", "2"]) == 0
    assert main(["run", str(p), "--out", str(parent), "--quiet", "--resume"]) == 0
    for out in (cont, parent):
        d = json.loads((out / "run.json").read_text())["derived"]
        assert d["filter"]["radius_mm"] == before["filter"]["radius_mm"] and d["filter"]["radius_source"] == "wall"
        assert (d["robust"]["mode"], d["robust"]["eta_e"], d["robust"]["padded_filter"]) == ("wall", 0.7, True)
    # a change of print.min_wall_mm changes the radius, and so the inputs: the resume refuses it without --allow-changed-inputs
    a = C.input_hashes(load_config(wall_case(sub_dir(tmp_path, "a"), wall=4.0, h=2.0, robust="wall")))
    b = C.input_hashes(load_config(wall_case(sub_dir(tmp_path, "b"), wall=5.0, h=2.0, robust="wall")))
    changes, _notes = C.diff_inputs(a, b)
    assert any("print.min_wall_mm" in c for c in changes)                      # refused without --allow-changed-inputs
    assert a["effective"]["settings"]["filter_radius"] != b["effective"]["settings"]["filter_radius"]
    assert a["effective"]["settings"]["robust"] == "wall"


def test_the_min_member_fact_with_filter_radius_in_the_wall_mode_gives_the_wall_option(tmp_path):
    """Round 1, finding 3: removing filter_radius gives the radius of the wall rule, not that of min_member_mm."""
    d = check_data(wall_case(tmp_path, wall=1.2, h=2.0, robust="wall", filter_radius=4.0, min_member_mm=3.0))["derived"]["filter"]["facts"]
    assert not any("would give a filter radius of 5.63" in f for f in d)
    one = [f for f in d if "min_member_mm is only reported" in f and "Removing settings.filter_radius gives the radius of the wall rule (3 mm)" in f]
    assert len(one) == 1 and "remove settings.filter_radius to use 3 mm" in one[0]


def test_the_wall_fact_with_filter_radius_says_the_mode_keeps_it(tmp_path):
    """Round 1, note 5: with settings.filter_radius set, the mode would keep that radius; the rule radius is 'without settings.filter_radius'."""
    d = check_data(wall_case(tmp_path, wall=1.2, h=2.0, robust="off", filter_radius=4.0))["derived"]["robust"]["facts"]
    fact = next(f for f in d if "does not aim at this wall" in f)
    assert "keeps the filter radius settings.filter_radius 4 mm" in fact and "without settings.filter_radius the rule derives 3 mm" in fact
    other = check_data(wall_case(sub_dir(tmp_path, "e"), wall=1.2, h=2.0, robust="eroded", eta_eroded=0.6))["derived"]["robust"]["facts"]
    assert any("settings.eta_eroded 0.6 is used" in f for f in other)


def test_the_stale_texts_name_the_wall_mode():
    from topopt import levers_cards as lc
    eta = next(c for c in lc.CARDS if c.key == "settings.eta_eroded")
    assert "0.7 with `robust: wall`" in eta.values and "Only with `robust: eroded` or `wall`" in eta.values
    assert "`settings.robust: eroded` or `wall`" in next(c for c in lc.CARDS if c.key == "buckling.min_factor").values
    card = next(c for c in lc.CARDS if c.key == "settings.robust")
    assert "31.0 % (default) to 24.9 % (wall rule)" in card.values
    row = next(e for e in card.evidence if "robust: wall" in e.change and "min_wall_mm" in e.change and "prototype" not in e.change)
    assert "gpu-mg" in lc.PART_CLASSES[lc.WALL_HOOK] and "CPU" not in lc.PART_CLASSES[lc.WALL_HOOK]
    assert any("5.51 -> 4.86 GiB" in e.price for e in card.evidence) and any("1.3524" in e.price for e in card.evidence)


def test_the_effective_settings_leave_out_the_wall_record(tmp_path):
    """wall_mm is a record: the effective settings (the hash of a checkpoint) hold robust, eta_eroded and the radius that the rule gave."""
    from topopt import checkpoint as C
    eff = C.effective_settings(load_config(wall_case(tmp_path, wall=4.0, h=2.0, robust="wall")))["settings"]
    assert "wall_mm" not in eff and eff["robust"] == "wall" and eff["filter_radius"] == pytest.approx(4.0 / members.nominal_width_per_radius(0.7))


def test_the_lever_card_of_the_robust_key_names_the_wall_mode_and_its_evidence():
    from topopt import levers_cards as lc
    card = next(c for c in lc.CARDS if c.key == "settings.robust")
    assert "off | eroded | wall" in card.values and "print.min_wall_mm" in card.values and "L(0.7) = 0.7885" in card.values
    row = next(e for e in card.evidence if "robust: wall" in e.change)
    assert "4.145 % -> 0.000 %" in row.effect and "+7.9 %" in row.price and "x1.07" in row.price


def test_the_robust_facts_of_the_wall_mode_do_not_point_at_min_member_mm(tmp_path):
    """min_member_mm is only reported with the wall rule, so no robust fact tells the author to set it."""
    d = check_data(wall_case(tmp_path, wall=1.2, h=2.0, robust="wall", min_member_mm=0.5))
    facts = d["derived"]["robust"]["facts"]
    assert facts and not any("set settings.min_member_mm" in f or "robust mode with min_member_mm" in f for f in facts)
    assert any("keeps a wall of" in f and "the wall rule sets the radius from print.min_wall_mm" in f for f in facts)
