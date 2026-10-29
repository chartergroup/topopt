"""F4-55: the sharp field grows to the design-space surface before the smoothing (CONTRACTS.md section 19, "Envelope surface").

A face of the design space that is slanted to the grid axes gives a staircase in the density field. The tests use a bar whose two side faces are
3 degrees off the x axis.
"""
import dataclasses
import logging

import numpy as np
import pytest
import trimesh

from topopt import post, smooth_sdf as ss
from topopt.config import ENVELOPE_FILL_DEFAULT

H = 1.0
NX, NY, NZ = 48, 14, 10
SLOPE = np.tan(np.radians(3.0))
Y0, Y1 = 2.3, 11.3          # the side faces at x = 0; they move by SLOPE * x along x (up to 2.5 mm over the bar)


def slanted_case(tmp_path):
    """A bar between a fixed foot and a load block; the design space is a prism with two side faces 3 degrees off the x axis."""
    x0, x1 = -3.0, NX * H + 3
    pts = []
    for x in (x0, x1):
        for y in (Y0, Y1):
            for z in (-3.0, NZ * H + 3):
                pts.append([x, y + SLOPE * x, z])
    design = trimesh.convex.convex_hull(np.array(pts))
    design.export(tmp_path / "design.stl")
    labels = np.full((NX, NY, NZ), post.EMPTY, np.uint8)
    rho = np.zeros((NX, NY, NZ), np.float32)
    xc = (np.arange(NX) + 0.5) * H
    yc = (np.arange(NY) + 0.5) * H
    inside = (yc[None, :] >= Y0 + SLOPE * xc[:, None]) & (yc[None, :] <= Y1 + SLOPE * xc[:, None])
    labels[inside[:, :, None] & np.ones((1, 1, NZ), bool)] = post.DESIGN
    rho[labels == post.DESIGN] = 1.0
    labels[:4, 4:10, 1:9] = post.FIXED
    labels[44:, 5:11, 1:9] = post.LOAD
    trimesh.creation.box(bounds=[[-1, 4, 1], [4, 10, 9]]).export(tmp_path / "foot.stl")
    trimesh.creation.box(bounds=[[44, 5, 1], [49, 11, 9]]).export(tmp_path / "tip.stl")
    run = {"yaml_path": str(tmp_path / "x.yaml"),
           "regions": {"design": "design.stl", "fixed": ["foot.stl"], "keep_out": [], "solid": [], "load": {"tip": "tip.stl"}}}
    return (rho, labels, np.zeros(3), H), run, design


def mesh_step(tmp_path, params=None):
    args, run, design = slanted_case(tmp_path)
    logging.disable(logging.WARNING)
    try:
        if params is None:
            v, f, info, rec = post._mesh_method({**run, "envelope_fill": "auto"}, *args)   # the tests that need the fill ask for it
        else:
            rho, labels, org, h = args
            v, f, info = ss.make_mesh_sdf(rho, labels, org, h, up=post.UPSAMPLE, pad=post.PAD, fixed_stls=[tmp_path / "foot.stl"],
                                          load_stls=[tmp_path / "tip.stl"], design_stl=tmp_path / "design.stl", params=params)
            rec = info
    finally:
        logging.disable(logging.NOTSET)
    return v, f, info, rec, design


def face_gap(v, f):
    """The distance of surface points on the lower side face (mid-bar, mid-height) to the design space plane.

    The points are sampled over the area: a flat face has few vertices.
    """
    n = np.array([SLOPE, -1.0, 0.0])
    n /= np.linalg.norm(n)
    off = n @ np.array([0.0, Y0, 0.0])
    pts, _ = trimesh.sample.sample_surface(trimesh.Trimesh(v, f, process=False), 20000, seed=1)
    dist = np.abs(pts @ n - off)
    sel = (pts[:, 0] > 8) & (pts[:, 0] < 40) & (pts[:, 2] > 2.5) & (pts[:, 2] < 7.5) & (dist < 0.6)
    return dist[sel]


def test_the_side_face_is_flat_and_the_default_without_the_step_has_steps(tmp_path):
    v1, f1, _i, rec1, design = mesh_step(tmp_path)
    off = dataclasses.replace(ss.SdfParams(envelope_fill="auto"), envelope_fill="off")
    v0, f0, *_ = mesh_step(tmp_path, off)
    g1, g0 = face_gap(v1, f1), face_gap(v0, f0)
    pitch = H / post.UPSAMPLE
    assert len(g1) > 20 and len(g0) > 20
    assert np.percentile(g1, 95) < 0.1 * pitch
    assert np.percentile(g0, 95) > 3 * np.percentile(g1, 95) + 0.02 * pitch
    assert rec1["envelope_fill"]["ran"] is True


def test_the_mass_target_is_the_volume_before_the_growth(tmp_path):
    v, f, _i, rec, _d = mesh_step(tmp_path)
    es = rec["envelope_fill"]
    assert es["grown_inside_mm3"] > 0 and es["grown_overshoot_mm3"] >= 0
    assert es["target_mm3"] == pytest.approx(rec["volume_thresholded_mm3"])
    assert abs(ss._signed_volume(v, f) - es["target_mm3"]) < 5e-3 * es["target_mm3"]
    assert abs(rec["calibration"][-1]["error"]) < 5e-3


def test_the_calibration_takes_up_to_8_rounds_with_the_fill_and_without_it_when_3_rounds_miss(tmp_path, sdf_cut_only):
    """With the fill the round limit is `calibrate_steps_fill` (8). Without it the plain limit is `calibrate_steps` (3), and a run that still
    misses then gets more rounds up to the same limit (#348). This case misses by 0.23 % after 3 rounds without the fill (the field not
    limited to the envelope: the cut takes part of each level change)."""
    p = ss.SdfParams(envelope_fill="auto")
    assert p.calibrate_steps_fill == 8 and p.calibrate_steps == 3
    _v, _f, _i, rec_on, _d = mesh_step(tmp_path)
    off = dataclasses.replace(p, envelope_fill="off")
    _v, _f, _i, rec_off, _d = mesh_step(tmp_path, off)
    _v, _f, _i, rec_plain, _d = mesh_step(tmp_path, dataclasses.replace(off, calibrate_steps_fill=3))   # the round limit before #348
    assert len(rec_plain["calibration"]) == 4 and abs(rec_plain["calibration"][-1]["error"]) >= 5e-4
    assert 4 < len(rec_off["calibration"]) <= 9 and abs(rec_off["calibration"][-1]["error"]) < 5e-4
    assert len(rec_on["calibration"]) <= 9


def test_the_slope_update_needs_fewer_rounds_and_ends_nearer_than_the_linear_one(tmp_path, sdf_cut_only):
    """With the fill, half of each level change is cut away, so the linear update of the target halves the miss in each round (9 builds, error -0.06 %).
    The slope update (`fill_slope_min` = `fill_slope_max` = 1 is the linear one) reaches the stop rule of the loop in fewer builds."""
    _v, _f, _i, slope, _d = mesh_step(tmp_path)
    lin = dataclasses.replace(ss.SdfParams(envelope_fill="auto"), fill_slope_min=1.0, fill_slope_max=1.0)
    _v, _f, _i, linear, _d = mesh_step(tmp_path, lin)
    n_s, n_l = len(slope["calibration"]), len(linear["calibration"])
    e_s, e_l = abs(slope["calibration"][-1]["error"]), abs(linear["calibration"][-1]["error"])
    assert n_l == 9 and n_s <= 6 and n_s < n_l
    assert e_s < e_l and e_s < 5e-4


def test_a_part_that_meets_the_target_in_few_rounds_takes_few_rounds(tmp_path):
    """The loop ends when the error is under 5e-4: with a cap of 30 rounds, the slanted case still takes 6 builds or fewer."""
    big = dataclasses.replace(ss.SdfParams(envelope_fill="auto"), calibrate_steps_fill=30)
    _v, _f, _i, rec, _d = mesh_step(tmp_path, big)
    assert len(rec["calibration"]) <= 6 and abs(rec["calibration"][-1]["error"]) < 5e-4


def test_the_step_is_off_without_the_flag_or_without_an_envelope(tmp_path):
    off = dataclasses.replace(ss.SdfParams(envelope_fill="auto"), envelope_fill="off")
    _v, _f, _i, rec, _d = mesh_step(tmp_path, off)
    assert rec["envelope_fill"]["ran"] is False and "off" in rec["envelope_fill"]["reason"]
    args, run, _ = slanted_case(tmp_path)
    run = {**run, "regions": {**run["regions"], "design": None}}
    logging.disable(logging.WARNING)
    try:
        v1, f1, _i1, rec1 = post._mesh_method({**run, "envelope_fill": "auto"}, *args)
        v2, f2, _i2, rec2 = post._mesh_method({**run, "envelope_fill": "off"}, *args)   # without a design space the switch changes nothing
    finally:
        logging.disable(logging.NOTSET)
    assert rec1["envelope_fill"]["ran"] is False and "design space" in rec1["envelope_fill"]["reason"]
    assert np.array_equal(v1, v2) and np.array_equal(f1, f2)
    assert rec1["envelope_fill"]["target_mm3"] == pytest.approx(rec1["volume_thresholded_mm3"])


def test_a_no_envelope_run_has_the_bytes_of_the_step_switched_off(tmp_path):
    args, run, _ = slanted_case(tmp_path)
    run = {**run, "regions": {**run["regions"], "design": None}}
    rho, labels, org, h = args
    kw = dict(up=post.UPSAMPLE, pad=post.PAD, fixed_stls=[tmp_path / "foot.stl"], load_stls=[tmp_path / "tip.stl"])
    logging.disable(logging.WARNING)
    try:
        va, fa, _ = ss.make_mesh_sdf(rho, labels, org, h, **kw)
        vb, fb, _ = ss.make_mesh_sdf(rho, labels, org, h, params=dataclasses.replace(ss.SdfParams(envelope_fill="auto"), envelope_fill="off"), **kw)
    finally:
        logging.disable(logging.NOTSET)
    assert np.array_equal(va, vb) and np.array_equal(fa, fb)


def thin_area_share(v, f, limit=1.2):
    """The share of the surface area that lies over a wall thinner than `limit` (ray to the opposite face along the inward normal)."""
    m = trimesh.Trimesh(v, f, process=False)
    c = m.triangles_center
    n = -m.face_normals
    o = c + n * 1e-3
    loc, ray, _tri = m.ray.intersects_location(o, n, multiple_hits=False)
    dist = np.full(len(c), np.inf)
    dist[ray] = np.linalg.norm(loc - o[ray], axis=1)
    a = m.area_faces
    return float(a[dist < limit].sum() / a.sum())


def test_thin_walls_do_not_grow_by_more_than_one_point(tmp_path):
    v1, f1, *_ = mesh_step(tmp_path)
    off = dataclasses.replace(ss.SdfParams(envelope_fill="auto"), envelope_fill="off")
    v0, f0, *_ = mesh_step(tmp_path, off)
    t1, t0 = thin_area_share(v1, f1), thin_area_share(v0, f0)
    assert t1 <= t0 + 0.01 and t1 <= 0.0594


def window_case():
    """A solid slab that ends 1 fine voxel under the design surface, with a window that is open at the surface (x 11 to 15, y 15 to 23: 5 fine voxels wide and deep).

    Return (field, grown field, the facts, the mask that a rule of "within h of the solid in any direction" would grow).
    """
    from scipy import ndimage as ndi
    pitch, h = 0.5, 1.0
    S = np.zeros((40, 40, 20), bool)
    S[:, 5:24, :] = True                  # the solid up to y = 12 mm; the design surface is at y = 12.5 mm
    S[11:16, 15:24, :] = False            # the window: no solid on the inward normal within 3 fine voxels, but solid at its sides
    design = trimesh.creation.box(bounds=[[-10, 0, -10], [30, 25 * pitch, 20]])
    out, facts = ss.grow_to_surface(S, np.zeros(S.shape, bool), design, np.zeros(3), pitch, h, ss.SdfParams(envelope_fill="auto"))
    iso = (ndi.distance_transform_edt(~S) * pitch <= h + 1e-6) & ~S
    iso[:, 25:, :] = False                # the same candidate layer: up to 1 fine voxel past the surface
    return S, out, facts, iso


def test_the_growth_adds_voxels_above_the_solid_and_none_sideways_into_a_window():
    """The rule of "S3 normal": a candidate grows only when the solid lies on the inward normal below it. The rule of "within h of the solid in any
    direction" (the control) fills the window from its sides; the normal rule does not."""
    S, out, facts, iso = window_case()
    grown = out & ~S
    window = np.zeros(S.shape, bool)
    window[12:15, 20:26, :] = True            # the inside of the window, at most 2 voxels away from its walls and 5 or more above its floor
    assert facts["ran"] is True and facts["grown"] > 0
    assert grown[:10, 24:26, :].sum() > 0                  # above the solid: the valley layer and the overshoot
    assert (grown & window).sum() == 0
    assert (iso & window).sum() > 0                        # the control: the isotropic rule grows into the window



# -- the switch `settings.envelope_fill` --
def _cfg(tmp_path, raw=None, **settings):
    """The cantilever YAML. `raw`: text of an `envelope_fill:` line to write as it is (to test the unquoted forms of YAML 1.1)."""
    import yaml

    from test_optimize import make_case
    path = make_case(tmp_path, h=2.0, iterations=2)
    d = yaml.safe_load(path.read_text())
    d["settings"].update(settings)
    text = yaml.safe_dump(d)
    if raw is not None:
        text = text.replace("settings:\n", f"settings:\n  envelope_fill: {raw}\n", 1)
    path.write_text(text)
    return path


def test_the_key_has_the_default_of_the_constant_and_the_yaml_forms_are_read(tmp_path):
    from topopt.config import UserError, load_config
    fill = lambda **kw: load_config(_cfg(tmp_path, **kw)).settings.envelope_fill      # noqa: E731
    assert fill() == ENVELOPE_FILL_DEFAULT
    for value in ("auto", "on", "off"):                       # quoted strings
        assert fill(envelope_fill=value) == value
    for raw, want in (("on", "on"), ("off", "off"), ("auto", "auto"), ("true", "on"), ("false", "off"), ("'on'", "on"), ("'off'", "off")):   # as written in a file
        assert load_config(_cfg(tmp_path, raw=raw)).settings.envelope_fill == want, raw
    for bad in ("maybe", 1, None, "yes"):
        with pytest.raises(UserError, match="envelope_fill.*auto, on or off"):
            fill(envelope_fill=bad)


def test_the_checkpoint_leaves_out_the_value_of_the_default_and_keeps_the_other_two(tmp_path):
    from topopt import checkpoint
    from topopt.config import load_config
    auto, on, off = (load_config(_cfg(tmp_path, envelope_fill=v)) for v in ("auto", "on", "off"))
    default = load_config(_cfg(tmp_path))
    by_value = {"auto": auto, "on": on, "off": off}
    assert "envelope_fill" not in checkpoint.effective_settings(default)["settings"]
    assert checkpoint.effective_settings(by_value[ENVELOPE_FILL_DEFAULT]) == checkpoint.effective_settings(default)   # the value of the constant is left out
    for value, c in by_value.items():
        key = checkpoint.effective_settings(c)["settings"].get("envelope_fill")
        assert key == (None if value == ENVELOPE_FILL_DEFAULT else value)                                              # the other two stay
    assert len({str(checkpoint.effective_settings(c)) for c in (auto, on, off)}) == 3


def test_the_run_record_names_the_value_and_the_flag_wins(tmp_path):
    import json

    from topopt.cli import main
    out = tmp_path / "run"
    assert main(["run", str(_cfg(tmp_path, envelope_fill="off")), "--out", str(out), "--quiet"]) == 0
    assert json.loads((out / "run.json").read_text())["envelope_fill"] == "off"
    m = json.loads((out / "result" / "metrics.json").read_text())["mesh"]
    assert m["envelope_fill"]["ran"] is False and "off" in m["envelope_fill"]["reason"]
    out2 = tmp_path / "run2"
    assert main(["run", str(_cfg(tmp_path, envelope_fill="off")), "--out", str(out2), "--quiet", "--envelope-fill", "auto"]) == 0   # the flag wins
    assert json.loads((out2 / "run.json").read_text())["envelope_fill"] == "auto"


def test_an_older_record_with_a_boolean_is_read():
    assert [post.fill_mode(v) for v in (True, False, "auto", "on", "off", None)] == ["on", "off", "auto", "on", "off", ENVELOPE_FILL_DEFAULT]


def test_an_unknown_value_in_run_json_logs_one_warning_and_uses_the_default(caplog):
    with caplog.at_level(logging.WARNING, logger="topopt.post"):
        assert post.fill_mode("of") == ENVELOPE_FILL_DEFAULT
        assert post.fill_mode(None) == ENVELOPE_FILL_DEFAULT
    lines = [r.getMessage() for r in caplog.records if "envelope_fill" in r.getMessage()]
    assert len(lines) == 1 and "'of'" in lines[0] and "auto, on or off" in lines[0] and lines[0].endswith(f"uses {ENVELOPE_FILL_DEFAULT}")


def test_with_the_switch_off_the_mesh_is_the_one_without_the_fill(tmp_path, sdf_cut_only):
    """The bar case of `test_sdf_envelope_cut` with the switch off: the numbers of the step without the fill. At origin/main 43d931c (before
    F4-55) they were 1991.8595 mm3 after 4 rounds (-1.15 %); #348 gives a run that still misses by 0.1 % or more after 3 rounds more rounds:
    6 rounds, -0.03 %."""
    from test_sdf_envelope_cut import bar_case
    args, run = bar_case(tmp_path)
    logging.disable(logging.WARNING)
    try:
        v, f, _info, rec = post._mesh_method({**run, "envelope_fill": "off"}, *args)
    finally:
        logging.disable(logging.NOTSET)
    ec = rec["envelope_cut"]
    assert ss._signed_volume(v, f) == pytest.approx(2014.3352, abs=2e-3)
    assert len(rec["calibration"]) == 6 and ec["passes"] == 2 and ec["vertices_beyond"] == 2232
    assert ec["volume_removed_mm3"] == pytest.approx(230.9384, abs=2e-3) and ec["largest_excess_mm"] == pytest.approx(0.58174, abs=1e-4)


def _coarse_cantilever_with_bore(tmp_path):
    """The 2 mm cantilever of the optimizer tests (12 passes) with a bore. The fill would add 36 % of the volume target inside the design space."""
    import yaml

    from test_optimize import make_case
    path = make_case(tmp_path, h=2.0, iterations=12)
    d = yaml.safe_load(path.read_text())
    d["interfaces"] = [{"name": "pin", "kind": "bore", "point": [10, 5, 0], "axis": [0, 0, 1], "radius_mm": 1.5, "length_mm": 10, "stock_mm": 0.2}]
    path.write_text(yaml.safe_dump(d))
    return path


def test_the_fill_skips_itself_on_a_coarse_part_and_the_mesh_is_the_one_with_the_fill_off(tmp_path):
    import hashlib
    import json

    from topopt.cli import main
    path = _coarse_cantilever_with_bore(tmp_path)
    for tag, flag in (("on", ["--envelope-fill", "auto"]), ("off", ["--envelope-fill", "off"])):
        assert main(["run", str(path), "--out", str(tmp_path / tag), "--quiet", *flag]) == 0
    get = lambda t, n: (tmp_path / t / "result" / n).read_bytes()      # noqa: E731
    m_on = json.loads(get("on", "metrics.json"))["mesh"]
    ef = m_on["envelope_fill"]
    cap = ss.SdfParams(envelope_fill="auto").fill_max_share
    assert cap == 0.05 and ef["ran"] is False and ef["grown_share_pct"] > 100 * cap
    assert f"{ef['grown_share_pct']:.1f} %" in ef["reason"] and f"{100 * cap:.1f} %" in ef["reason"] and "fill_max_share" in ef["reason"]
    assert ef["grown_inside_mm3"] > 0 and ef["target_mm3"] == pytest.approx(m_on["volume_thresholded_mm3"])
    assert m_on["stl_roundtrip"]["bodies"] == 1                 # the bodies that the mesh step removes are those of the fill-off mesh too (same bytes below)
    assert hashlib.sha256(get("on", "final.stl")).digest() == hashlib.sha256(get("off", "final.stl")).digest()
    assert len(m_on["calibration"]) == len(json.loads(get("off", "metrics.json"))["mesh"]["calibration"])


def test_a_small_share_runs_the_fill_and_the_fact_has_the_share(tmp_path):
    _v, _f, _i, rec, _d = mesh_step(tmp_path)
    ef = rec["envelope_fill"]
    assert ef["ran"] is True and 0 < ef["grown_share_pct"] < 100 * ss.SdfParams(envelope_fill="auto").fill_max_share
    assert ef["grown_share_pct"] == pytest.approx(100 * ef["grown_inside_mm3"] / ef["target_mm3"])


def test_the_skip_logs_one_line_and_a_higher_cap_lets_the_fill_run(tmp_path, caplog):
    args, run, _d = slanted_case(tmp_path)
    rho, labels, org, h = args
    kw = dict(up=post.UPSAMPLE, pad=post.PAD, fixed_stls=[tmp_path / "foot.stl"], load_stls=[tmp_path / "tip.stl"], design_stl=tmp_path / "design.stl")
    with caplog.at_level(logging.WARNING, logger="topopt.smooth_sdf"):
        _v, _f, info = ss.make_mesh_sdf(rho, labels, org, h, params=dataclasses.replace(ss.SdfParams(envelope_fill="auto"), fill_max_share=0.001), **kw)
    lines = [r.getMessage() for r in caplog.records if "envelope fill skipped" in r.getMessage()]
    assert len(lines) == 1 and info["envelope_fill"]["ran"] is False and "0.1 %" in info["envelope_fill"]["reason"]
    _v, _f, info = ss.make_mesh_sdf(rho, labels, org, h, params=dataclasses.replace(ss.SdfParams(envelope_fill="auto"), fill_max_share=1.0), **kw)
    assert info["envelope_fill"]["ran"] is True


def test_on_runs_the_fill_above_the_cap_and_the_fact_says_so_and_auto_skips(tmp_path):
    import json

    from topopt.cli import main
    path = _coarse_cantilever_with_bore(tmp_path)
    for tag in ("auto", "on"):
        assert main(["run", str(path), "--out", str(tmp_path / tag), "--quiet", "--envelope-fill", tag]) == 0
    ef = {t: json.loads((tmp_path / t / "result" / "metrics.json").read_text())["mesh"]["envelope_fill"] for t in ("auto", "on")}
    assert ef["auto"]["ran"] is False and "cap_overridden" not in ef["auto"] and "settings.envelope_fill: on" in ef["auto"]["reason"]
    assert ef["on"]["ran"] is True and ef["on"]["cap_overridden"] is True and ef["on"]["grown_share_pct"] > 100 * ss.SdfParams(envelope_fill="auto").fill_max_share
    assert ef["on"]["grown_share_pct"] == pytest.approx(ef["auto"]["grown_share_pct"])


def test_auto_below_the_cap_runs_and_has_no_override_key(tmp_path):
    _v, _f, _i, rec, _d = mesh_step(tmp_path)
    assert rec["envelope_fill"]["ran"] is True and "cap_overridden" not in rec["envelope_fill"]


def test_the_grey_mass_part_has_one_body_with_the_fill_on(tmp_path):
    """The part of `test_mass_run_facts` (a mass run, 2 bodies at the cut 0.5): the fill grows 12 overshoot voxels and nothing inside. The mesh step must still take
    the loose body out, so that final.stl has one body (a calibration that started with the cuts left a second shell of 12 mm3 here)."""
    import json

    from test_mass_run_facts import run_case
    _, out = run_case(tmp_path, "mass", edit=lambda d: d["settings"].update(envelope_fill="auto"))
    m = json.loads((out / "result" / "metrics.json").read_text())["mesh"]
    assert m["envelope_fill"]["ran"] is True and m["envelope_fill"]["grown_inside_mm3"] == 0.0
    assert m["stl_roundtrip"]["bodies"] == 1 and m["extra_bodies_removed"] == 1


def test_the_default_of_the_key_is_off_in_the_constant_the_help_and_the_card():
    """The one test that pins the decided default. A change of the default changes this line, the constant and the docs only (CONTRACTS.md section 19)."""
    from topopt import cli, config, levers_cards
    assert config.ENVELOPE_FILL_DEFAULT == "off"
    assert config.Settings.__dataclass_fields__["envelope_fill"].default == config.ENVELOPE_FILL_DEFAULT == ss.SdfParams().envelope_fill
    parser = cli.build_parser()
    helptext = " ".join(next(a for a in parser._subparsers._group_actions[0].choices["run"]._actions if "--envelope-fill" in a.option_strings).help.split())
    assert f"the default is {config.ENVELOPE_FILL_DEFAULT}" in helptext
    card = next(c for c in levers_cards.CARDS if c.key == "settings.envelope_fill")
    assert f"default {config.ENVELOPE_FILL_DEFAULT}" in card.values


def test_the_note_and_one_warning_line_appear_when_the_fill_ran_and_not_when_it_did_not(tmp_path, caplog):
    args, run, _d = slanted_case(tmp_path)
    rho, labels, org, h = args
    kw = dict(up=post.UPSAMPLE, pad=post.PAD, fixed_stls=[tmp_path / "foot.stl"], load_stls=[tmp_path / "tip.stl"], design_stl=tmp_path / "design.stl")
    p = ss.SdfParams(envelope_fill="auto")

    def run_with(params, **more):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="topopt.smooth_sdf"):
            _v, _f, info = ss.make_mesh_sdf(rho, labels, org, h, params=params, **{**kw, **more})
        return info["envelope_fill"], [r.getMessage() for r in caplog.records if "envelope fill" in r.getMessage()]

    ef, lines = run_with(p)                                          # ran
    assert ef["ran"] is True and ef["note"] == ss.FILL_COST_NOTE and "stiffness can drop" in ef["note"] and "settings.envelope_fill: off" in ef["note"]
    assert len(lines) == 1 and lines[0].startswith("envelope fill ran") and f"{ef['grown_share_pct']:.1f} %" in lines[0] and ef["note"] in lines[0]
    ef, lines = run_with(dataclasses.replace(p, envelope_fill="off"))   # off: no note, no line
    assert "note" not in ef and lines == []
    ef, lines = run_with(p, design_stl=None)                         # no envelope: no note, no line
    assert ef["ran"] is False and "note" not in ef and lines == []
    ef, lines = run_with(dataclasses.replace(p, fill_max_share=0.001))   # skipped by the cap: the skip line only, no note
    assert ef["ran"] is False and "note" not in ef and len(lines) == 1 and lines[0].startswith("envelope fill skipped")
    ef, lines = run_with(dataclasses.replace(p, envelope_fill="on", fill_max_share=0.001))   # on above the cap: one line with the limit and the cost
    assert ef["ran"] is True and ef["cap_overridden"] is True and ef["note"] == ss.FILL_COST_NOTE
    assert len(lines) == 1 and lines[0].startswith("envelope fill runs above the limit") and ss.FILL_COST_NOTE in lines[0]
