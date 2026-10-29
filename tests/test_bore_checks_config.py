"""The `bore_checks` block of the problem YAML (CONTRACTS.md section 40).

The block is read by `load_config`. Every key of B4 (the Goodman check, the press-fit check) is parsed and checked here, so that the
press-fit solve (B4b) adds no config change. The errors E1 to E12 of the plan each have a case.
"""
import copy
import json
import math

import pytest
import yaml

from test_optimize import make_case
from topopt import config as C
from topopt.config import UserError, load_config

SRC = "test source"


def good_block():
    return {
        "bores": [{"name": "b1", "point": [5, 5, 12], "axis": [0, 0, -1], "radius_mm": 2.0, "length_mm": 8.0,
                   "radius_mm_source": SRC, "length_mm_source": SRC}],
        "goodman": {"materials": {"M": {"su_mpa": 1000, "sy_mpa": 900, "se_mpa": 300, "source": SRC}}},
    }


def press_block():
    b = good_block()
    b["bores"][0].update(diametral_interference_mm=0.1, diametral_interference_mm_source=SRC)
    b["press_fit"] = {"cups": [{"name": "rigid", "k_mpa_per_mm": "1.0e6", "source": SRC},
                               {"name": "al", "e_mpa": 70000, "nu": 0.33, "wall_mm": 1.5, "source": SRC}]}
    return b


BORE_IFACE = {"name": "hole", "kind": "bore", "point": [5, 5, 12], "axis": [0, 0, -2], "radius_mm": 2.5, "length_mm": 9.0}


def write(tmp_path, block, interfaces=None):
    p = make_case(tmp_path, h=2.0, iterations=2)
    d = yaml.safe_load(p.read_text())
    if block is not None:
        d["bore_checks"] = block
    if interfaces is not None:
        d["interfaces"] = interfaces
    p.write_text(yaml.safe_dump(d))
    return p


def load(tmp_path, block, interfaces=None):
    return load_config(write(tmp_path, block, interfaces))


def error(tmp_path, block, interfaces=None) -> str:
    with pytest.raises(UserError) as exc:
        load(tmp_path, block, interfaces)
    return str(exc.value)


# --- the block and its defaults ---------------------------------------------------------------------------------

def test_no_block_gives_none(tmp_path):
    assert load(tmp_path, None).bore_checks is None


def test_the_four_knob_defaults_are_the_pipeline_values():
    # pipeline/s9_print/final/pressfit.py:45 and :50 (TOL, the normal limit), :127 (the band)
    assert (C.DEFAULT_RADIUS_TOLERANCE_MM, C.DEFAULT_END_MARGIN_MM, C.DEFAULT_NORMAL_LIMIT, C.DEFAULT_SAMPLE_BAND_MM) == (0.35, 0.35, 0.35, 1.2)
    assert C.BORE_KNOB_KEYS == ("radius_tolerance_mm", "end_margin_mm", "normal_limit", "sample_band_mm")


def test_goodman_only_block(tmp_path):
    bc = load(tmp_path, good_block()).bore_checks
    assert bc.goodman and not bc.press_fit and bc.checks() == ("goodman",)
    b = bc.bores[0]
    assert (b.name, b.point, b.axis, b.radius_mm, b.length_mm, b.geometry_from) == ("b1", (5.0, 5.0, 12.0), (0.0, 0.0, -1.0), 2.0, 8.0, "yaml")
    assert b.diametral_interference_mm is None and b.mean_stress_mpa is None
    m = bc.materials[0]
    assert (m.label, m.su_mpa, m.sy_mpa, m.se_mpa) == ("M", 1000.0, 900.0, 300.0)
    assert bc.cups == ()


def test_knobs_default_and_yaml_origin(tmp_path):
    k = load(tmp_path, good_block()).bore_checks.knobs
    assert [getattr(k, n) for n in C.BORE_KNOB_KEYS] == [0.35, 0.35, 0.35, 1.2]
    assert [k.origin(n) for n in C.BORE_KNOB_KEYS] == ["default"] * 4 and k.sources == ()
    blk = good_block()
    blk.update(radius_tolerance_mm=0.5, sample_band_mm=2, sample_band_mm_source="why")
    k = load(tmp_path, blk).bore_checks.knobs
    assert (k.radius_tolerance_mm, k.sample_band_mm, k.end_margin_mm) == (0.5, 2.0, 0.35)
    assert [k.origin(n) for n in C.BORE_KNOB_KEYS] == ["yaml", "default", "default", "yaml"]
    assert k.source_of("sample_band_mm") == "why" and k.source_of("radius_tolerance_mm") is None


def test_press_fit_block_is_parsed_in_full(tmp_path):
    bc = load(tmp_path, press_block()).bore_checks
    assert bc.goodman and bc.press_fit and bc.checks() == ("goodman", "pressfit")
    assert bc.bores[0].diametral_interference_mm == 0.1
    rigid, al = bc.cups
    assert (rigid.name, rigid.k_mpa_per_mm, rigid.form) == ("rigid", 1.0e6, "k")         # the text "1.0e6" is a number (YAML 1.1)
    assert (al.name, al.e_mpa, al.nu, al.wall_mm, al.form, al.k_mpa_per_mm) == ("al", 70000.0, 0.33, 1.5, "lame", None)
    assert rigid.source_of("k_mpa_per_mm") == SRC and al.source_of("wall_mm") == SRC


def test_press_fit_alone_without_goodman(tmp_path):
    blk = press_block()
    del blk["goodman"]
    bc = load(tmp_path, blk).bore_checks
    assert bc.press_fit and not bc.goodman and bc.materials == ()


def test_mean_stress_without_press_fit(tmp_path):
    blk = good_block()
    blk["bores"][0].update(mean_stress_mpa=-50, mean_stress_mpa_source=SRC)
    assert load(tmp_path, blk).bore_checks.bores[0].mean_stress_mpa == -50.0


def test_the_block_is_not_in_the_effective_settings(tmp_path):
    """The block is read by verify only: the optimizer settings and the effective settings do not change."""
    from topopt.checkpoint import effective_settings
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = load(tmp_path / "a", None), load(tmp_path / "b", press_block())
    assert effective_settings(a) == effective_settings(b)
    assert a.settings == b.settings and a.load_cases == b.load_cases and a.material == b.material and a.regions.design.name == b.regions.design.name


# --- the C2 shorthand (F2, E4, E5) ------------------------------------------------------------------------------

def test_interface_shorthand_copies_the_c2_geometry(tmp_path):
    blk = good_block()
    blk["bores"] = [{"name": "viaiface", "interface": "hole"}]
    bc = load(tmp_path, blk, [BORE_IFACE]).bore_checks
    b = bc.bores[0]
    assert (b.point, b.radius_mm, b.length_mm, b.geometry_from) == ((5.0, 5.0, 12.0), 2.5, 9.0, "interface:hole")
    assert b.axis == (0.0, 0.0, -1.0) and math.isclose(math.hypot(*b.axis), 1.0, abs_tol=1e-15)   # the C2 axis is a unit vector
    assert b.sources == ()      # the numbers come from the C2 entry: no source of this entry


@pytest.mark.parametrize("name,interfaces,words", [
    ("nothere", [BORE_IFACE], ["'bore_checks.bores[0].interface'", "'nothere'", "hole (bore)", "Options"]),
    ("boss", [{"name": "boss", "kind": "boss", "point": [5, 5, -1], "axis": [0, 0, 1], "radius_mm": 3.0, "length_mm": 6.0}],
     ["boss (boss)", "not a bore entry", "Options"]),
    ("hole", None, ["no 'interfaces'", "Options"]),
])
def test_E5_interface_names_no_bore(tmp_path, name, interfaces, words):
    blk = good_block()
    blk["bores"] = [{"name": "x", "interface": name}]
    msg = error(tmp_path, blk, interfaces)
    for w in words:
        assert w in msg, msg
    assert msg.endswith(f"(file: {tmp_path / 'c.yaml'})")


# --- the errors E1 to E12 -----------------------------------------------------------------------------------------

def _mut(fn):
    def apply(b):
        fn(b)
        return b
    return apply


# id, how to break the block, the words that the message holds
CASES = [
    ("E1 unknown key", _mut(lambda b: b.update(extra=1)), ["unknown key 'extra' in 'bore_checks'", "Known keys", "Options: use one of the known keys, or remove 'extra'", "section 40.2"]),
    ("E1 unknown key in a bore", _mut(lambda b: b["bores"][0].update(colour="red")), ["unknown key 'colour' in 'bore_checks.bores[0]'", "Options: use one of the known keys, or remove 'colour'"]),
    ("E1 unknown key in goodman", _mut(lambda b: b["goodman"].update(factor=1)), ["unknown key 'factor' in 'bore_checks.goodman'", "Options: use one of the known keys, or remove 'factor'"]),
    ("E1 not a mapping", lambda b: [1, 2], ["'bore_checks' must be a mapping", "Options: write 'bore_checks' as lines of 'key: value'"]),
    ("E2 no check", _mut(lambda b: b.pop("goodman")), ["neither 'goodman' nor 'press_fit'", "Options"]),
    ("E3 no bores", _mut(lambda b: b.pop("bores")), ["missing key 'bores' in 'bore_checks'", "Options: add the key 'bores'; see CONTRACTS.md section 40.2"]),
    ("E3 empty bores", _mut(lambda b: b.update(bores=[])), ["'bore_checks.bores' must be a list with one bore or more", "Options"]),
    ("E3 same name", _mut(lambda b: b["bores"].append(dict(b["bores"][0]))), ["duplicate bore name 'b1'", "Options"]),
    ("E4 neither", _mut(lambda b: b["bores"].__setitem__(0, {"name": "x"})), ["needs 'interface' or the four keys", "missing: point, axis, radius_mm, length_mm", "Options"]),
    ("E4 part of the four", _mut(lambda b: b["bores"][0].pop("axis")), ["missing: axis", "Options"]),
    ("E4 both", _mut(lambda b: b["bores"][0].update(interface="hole")), ["'interface' and also point, axis, radius_mm, length_mm", "Options"]),
    ("E6 radius", _mut(lambda b: b["bores"][0].update(radius_mm=0)), ["'bore_checks.bores[0].radius_mm' must be > 0", "Options: give a number above 0"]),
    ("E6 zero axis", _mut(lambda b: b["bores"][0].update(axis=[0, 0, 0])), ["'bore_checks.bores[0].axis' must not be the zero vector", "Options: give three numbers [x, y, z] that are not all 0"]),
    ("E6 empty contact range", _mut(lambda b: b["bores"][0].update(length_mm=0.7)), ["contact range is empty", "2 x end_margin_mm", "Options"]),
    ("E6 empty range by a margin", _mut(lambda b: b.update(end_margin_mm=4.0)), ["contact range is empty", "Options"]),
    ("E6 knob zero", _mut(lambda b: b.update(radius_tolerance_mm=0)), ["'bore_checks.radius_tolerance_mm' must be > 0", "Options: give a number above 0"]),
    ("E6 knob normal", _mut(lambda b: b.update(normal_limit=1.5)), ["'bore_checks.normal_limit' must be <= 1.0", "Options: give a number above 0 and at most 1"]),
    ("E6 knob band", _mut(lambda b: b.update(sample_band_mm=-1)), ["'bore_checks.sample_band_mm' must be > 0", "Options: give a number above 0"]),
    ("E6 negative margin", _mut(lambda b: b.update(end_margin_mm=-0.1)), ["'bore_checks.end_margin_mm' must be >= 0.0", "Options: give a number at least 0"]),
    ("E7 su", _mut(lambda b: b["goodman"]["materials"]["M"].pop("source")), ["'bore_checks.goodman.materials.M.su_mpa' has no source", "decision D7", "'su_mpa_source: <text>'", "one 'source: <text>'", "Options"]),
    ("E8 missing strength", _mut(lambda b: b["goodman"]["materials"]["M"].pop("se_mpa")), ["needs all three strengths", "missing: se_mpa", "Options"]),
    ("E8 no material", _mut(lambda b: b["goodman"].update(materials={})), ["'bore_checks.goodman.materials' must be a mapping", "one entry or more", "Options: give one entry for each material"]),
    ("E8 strength value", _mut(lambda b: b["goodman"]["materials"]["M"].update(sy_mpa=0)), ["'bore_checks.goodman.materials.M.sy_mpa' must be > 0", "Options: give a number above 0"]),
    ("E12 source of a number that is absent", _mut(lambda b: b["bores"][0].update(mean_stress_mpa_source="x")), ["'bore_checks.bores[0].mean_stress_mpa_source' names the number 'mean_stress_mpa'", "does not give it", "Options"]),
    ("E12 knob source without the knob", _mut(lambda b: b.update(end_margin_mm_source="x")), ["'bore_checks.end_margin_mm_source' names the knob 'end_margin_mm'", "Options"]),
]

PRESS_CASES = [
    ("E7 interference", _mut(lambda b: b["bores"][0].pop("diametral_interference_mm_source") and b["bores"][0].pop("source", None)),
     ["'bore_checks.bores[0].diametral_interference_mm' has no source", "Options"]),
    ("E9 both forms", _mut(lambda b: b["press_fit"]["cups"][0].update(e_mpa=1.0)), ["gives k_mpa_per_mm and also e_mpa", "one form", "Options"]),
    ("E9 neither form", _mut(lambda b: b["press_fit"]["cups"].__setitem__(0, {"name": "n"})), ["needs k_mpa_per_mm, or all three of e_mpa, nu and wall_mm", "none of them", "Options"]),
    ("E9 part of the Lame form", _mut(lambda b: b["press_fit"]["cups"][1].pop("nu")), ["it gives e_mpa, wall_mm, missing: nu", "Options"]),
    ("E9 wall above the radius", _mut(lambda b: b["press_fit"]["cups"][1].update(wall_mm=2.5)), ["'bore_checks.press_fit.cups[1].wall_mm' is 2.5", "bore 'b1' has radius_mm 2.0", "Options"]),
    ("E9 nu", _mut(lambda b: b["press_fit"]["cups"][1].update(nu=0.5)), ["'bore_checks.press_fit.cups[1].nu' must be < 0.5", "Options: give a number at least 0 and below 0.5"]),
    ("E9 same cup name", _mut(lambda b: b["press_fit"]["cups"][1].update(name="rigid")), ["duplicate cup name 'rigid'", "Options: one name for each cup variant"]),
    ("E9 no cups", _mut(lambda b: b["press_fit"].update(cups=[])), ["'bore_checks.press_fit.cups' must be a list of cup variants", "Options: give one entry for each cup variant"]),
    ("E10 no interference", _mut(lambda b: [b["bores"][0].pop(k) for k in ("diametral_interference_mm", "diametral_interference_mm_source")]),
     ["entry 'b1' has no diametral_interference_mm", "has 'press_fit'", "Options"]),
    ("E11 mean stress with press_fit", _mut(lambda b: b["bores"][0].update(mean_stress_mpa=10, mean_stress_mpa_source=SRC)), ["gives mean_stress_mpa", "two sources", "Options"]),
    ("E12 cup source of the other form", _mut(lambda b: b["press_fit"]["cups"][0].update(e_mpa_source="x")), ["'bore_checks.press_fit.cups[0].e_mpa_source' names the number 'e_mpa'", "Options"]),
]


@pytest.mark.parametrize("name,mutate,words", CASES, ids=[c[0] for c in CASES])
def test_errors(tmp_path, name, mutate, words):
    msg = error(tmp_path, mutate(good_block()))
    for w in words:
        assert w in msg, f"{w!r} not in {msg!r}"
    assert "Options" in msg and "\n" not in msg, msg                 # every text of the block has an option (review m-4 of B4a)
    assert msg.endswith(f"(file: {tmp_path / 'c.yaml'})")


@pytest.mark.parametrize("name,mutate,words", PRESS_CASES, ids=[c[0] for c in PRESS_CASES])
def test_press_fit_errors(tmp_path, name, mutate, words):
    msg = error(tmp_path, mutate(press_block()))
    for w in words:
        assert w in msg, f"{w!r} not in {msg!r}"
    assert "Options" in msg and "\n" not in msg, msg
    assert msg.endswith(f"(file: {tmp_path / 'c.yaml'})")


def test_E10_interference_without_press_fit(tmp_path):
    blk = good_block()
    blk["bores"][0].update(diametral_interference_mm=0.1, diametral_interference_mm_source=SRC)
    msg = error(tmp_path, blk)
    assert "gives diametral_interference_mm, and the block has no 'press_fit'" in msg and "Options" in msg


def test_E12_message_for_the_interface_geometry_source(tmp_path):
    """A bore with `interface` has no radius or length of its own: a source for them is E12."""
    blk = good_block()
    blk["bores"] = [{"name": "x", "interface": "hole", "radius_mm_source": "s"}]
    msg = error(tmp_path, blk, [BORE_IFACE])
    assert "'bore_checks.bores[0].radius_mm_source' names the number 'radius_mm'" in msg


# --- the `_source` rule (D3 pattern; D7) --------------------------------------------------------------------------

def sourced_block(mean_stress=False):
    """A block where every physical number has its own `K_source` and no entry has a `source` key."""
    b = {"bores": [{"name": "b1", "point": [5, 5, 12], "axis": [0, 0, -1], "radius_mm": 2.0, "length_mm": 8.0}],
         "goodman": {"materials": {"M": {"su_mpa": 1000, "sy_mpa": 900, "se_mpa": 300, "su_mpa_source": SRC, "sy_mpa_source": SRC,
                                         "se_mpa_source": SRC}}}}
    if mean_stress:
        b["bores"][0].update(mean_stress_mpa=5.0, mean_stress_mpa_source=SRC)
    else:
        b["bores"][0].update(diametral_interference_mm=0.1, diametral_interference_mm_source=SRC)
        b["press_fit"] = {"cups": [{"name": "k", "k_mpa_per_mm": 1000, "k_mpa_per_mm_source": SRC},
                                   {"name": "lame", "e_mpa": 70000, "nu": 0.3, "wall_mm": 1.0, "e_mpa_source": SRC, "nu_source": SRC,
                                    "wall_mm_source": SRC}]}
    return b


def _entry(block, kind):
    return {"goodman": lambda: block["goodman"]["materials"]["M"], "bore": lambda: block["bores"][0],
            "cup_k": lambda: block["press_fit"]["cups"][0], "cup_lame": lambda: block["press_fit"]["cups"][1]}[kind]()


NUMBER_KEYS = [
    ("goodman", "su_mpa"), ("goodman", "sy_mpa"), ("goodman", "se_mpa"),
    ("bore", "diametral_interference_mm"), ("bore", "mean_stress_mpa"),
    ("cup_k", "k_mpa_per_mm"), ("cup_lame", "e_mpa"), ("cup_lame", "nu"), ("cup_lame", "wall_mm"),
]


@pytest.mark.parametrize("kind,key", NUMBER_KEYS, ids=[f"{k}:{n}" for k, n in NUMBER_KEYS])
def test_E7_every_physical_number_needs_a_source(tmp_path, kind, key):
    blk = sourced_block(mean_stress=key == "mean_stress_mpa")
    assert load(tmp_path, copy.deepcopy(blk)).bore_checks is not None      # complete: it loads
    del _entry(blk, kind)[f"{key}_source"]
    msg = error(tmp_path, blk)
    assert f".{key}' has no source" in msg and f"'{key}_source: <text>'" in msg and "decision D7" in msg, msg


def test_source_rule_entry_source_covers_and_own_source_wins(tmp_path):
    blk = press_block()
    mat = blk["goodman"]["materials"]["M"]
    mat["se_mpa_source"] = "own text"
    cup = blk["press_fit"]["cups"][1]
    cup["wall_mm_source"] = "own wall"
    bc = load(tmp_path, blk).bore_checks
    m = bc.materials[0]
    assert (m.source_of("se_mpa"), m.source_of("su_mpa"), m.source_of("sy_mpa")) == ("own text", SRC, SRC)
    c = bc.cups[1]
    assert (c.source_of("wall_mm"), c.source_of("e_mpa"), c.source_of("nu")) == ("own wall", SRC, SRC)


def test_geometry_numbers_may_have_no_source(tmp_path):
    blk = good_block()
    blk["bores"][0] = {k: v for k, v in blk["bores"][0].items() if not k.endswith("_source")}
    b = load(tmp_path, blk).bore_checks.bores[0]
    assert b.sources == ()


def test_errors_are_user_errors_of_the_loader_not_a_traceback(tmp_path):
    """A wrong block stops `topopt check` and `topopt run` early with one line (the parse is in load_config)."""
    from topopt.cli import main
    blk = good_block()
    blk["goodman"]["materials"]["M"].pop("source")
    p = write(tmp_path, blk)
    assert main(["check", str(p)]) != 0


def _contracts_example_block():
    """The first YAML block of CONTRACTS.md section 40.2, as written."""
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "CONTRACTS.md").read_text()
    sec = text[text.index("### 40.2 The YAML block"):]
    start = sec.index("```yaml") + len("```yaml")
    return yaml.safe_load(sec[start:sec.index("```", start)])["bore_checks"]


def test_the_yaml_example_of_contracts_40_2_loads(tmp_path):
    """The example of the contract must load as written (review m-1 of B4a: the `lower` bore had no interference while `press_fit` was there)."""
    blk = _contracts_example_block()
    iface = {"name": "lower_bore", "kind": "bore", "point": [5, 5, 12], "axis": [0, 0, -1], "radius_mm": 2.5, "length_mm": 9.0}
    bc = load(tmp_path, blk, [iface]).bore_checks
    assert [b.name for b in bc.bores] == ["upper", "lower"]
    assert bc.goodman and bc.press_fit and bc.checks() == ("goodman", "pressfit")
    upper, lower = bc.bores
    assert upper.geometry_from == "yaml" and lower.geometry_from == "interface:lower_bore"
    assert (upper.diametral_interference_mm, lower.diametral_interference_mm) == (0.10, 0.05)
    assert [c.name for c in bc.cups] == ["rigid", "cup_al"]


def _fuzz(fn):
    b = copy.deepcopy(good_block())
    b["bores"][0] = {"name": "b1", "point": [0, 0, 0], "axis": [0, 0, 1], "radius_mm": 2.0, "length_mm": 10.0}
    r = fn(b)
    return b if r is None else r


def _bore(b):
    return b["bores"][0]


def _press(b, cups):
    """A `press_fit` block with the given `cups` and the interference that it needs (so that the error is the one of the cups, not E10)."""
    _bore(b).update(diametral_interference_mm=0.1, diametral_interference_mm_source="s")
    b["press_fit"] = {"cups": cups}


# The malformed blocks of the review of B4a (`scripts/q3_fuzz.py`, Q3): wrong types and wrong values in each place. Each gives a one-line UserError with an
# option, and none gives another exception. Two are valid input (a text knob such as "1e-1", and a negative mean stress).
FUZZ = {
    "point str": lambda b: _bore(b).update(point="abc"), "point len2": lambda b: _bore(b).update(point=[1, 2]),
    "point nested": lambda b: _bore(b).update(point=[[1], 2, 3]), "point None item": lambda b: _bore(b).update(point=[None, 0, 0]),
    "axis nan": lambda b: _bore(b).update(axis=[float("nan"), 0, 1]), "axis inf": lambda b: _bore(b).update(axis=[float("inf"), 0, 1]),
    "axis tiny": lambda b: _bore(b).update(axis=[1e-300, 0, 0]), "axis bool": lambda b: _bore(b).update(axis=[True, 0, 0]),
    "bores mapping": lambda b: b.update(bores={"a": 1}), "bore not mapping": lambda b: b.update(bores=["upper"]),
    "name int": lambda b: _bore(b).update(name=5), "name empty": lambda b: _bore(b).update(name=""),
    "radius str": lambda b: _bore(b).update(radius_mm="x"), "radius huge int": lambda b: _bore(b).update(radius_mm=10 ** 400),
    "materials list": lambda b: b["goodman"].update(materials=[1]), "material not mapping": lambda b: b["goodman"]["materials"].update(M=5),
    "material label int": lambda b: b["goodman"].update(materials={7: {"su_mpa": 1, "sy_mpa": 1, "se_mpa": 1, "source": "s"}}),
    "source int": lambda b: b["goodman"]["materials"]["M"].update(source=5), "source empty": lambda b: b["goodman"]["materials"]["M"].update(source=""),
    "goodman None": lambda b: b.update(goodman=None), "goodman list": lambda b: b.update(goodman=[1]), "press_fit None": lambda b: b.update(press_fit=None),
    "knob text bad": lambda b: b.update(sample_band_mm="wide"), "mean_stress nan": lambda b: _bore(b).update(mean_stress_mpa=float("nan"), mean_stress_mpa_source="s"),
    "se nan": lambda b: b["goodman"]["materials"]["M"].update(se_mpa=float("nan")), "interface no interfaces": lambda b: b.update(bores=[{"name": "b", "interface": "x"}]),
    "cups mapping": lambda b: _press(b, {"a": 1}), "cup name int": lambda b: _press(b, [{"name": 3, "k_mpa_per_mm": 1.0, "source": "s"}]),
    "cup k str": lambda b: _press(b, [{"name": "c", "k_mpa_per_mm": "soft", "source": "s"}]),
    "interference str": lambda b: _bore(b).update(diametral_interference_mm="a", diametral_interference_mm_source="s"),
    "knob source int": lambda b: b.update(sample_band_mm=1.0, sample_band_mm_source=5), "bore source int": lambda b: _bore(b).update(radius_mm_source=5),
}
VALID_FUZZ = {"knob str": lambda b: b.update(sample_band_mm="1e-1"),
              "mean_stress negative": lambda b: _bore(b).update(mean_stress_mpa=-50, mean_stress_mpa_source="s")}


@pytest.mark.parametrize("name", list(FUZZ))
def test_a_malformed_block_gives_a_one_line_user_error_with_an_option(name):
    """Review m-4 of B4a: no text of the block lacks an option (23 of 26 had none), and no wrong input gives a traceback."""
    with pytest.raises(UserError) as exc:
        C.parse_bore_checks_block(_fuzz(FUZZ[name]), (), "fuzz.yaml")
    msg = str(exc.value)
    assert "Options: " in msg and "\n" not in msg and msg.endswith("(file: fuzz.yaml)"), msg


@pytest.mark.parametrize("name", list(VALID_FUZZ))
def test_two_odd_inputs_are_valid(name):
    assert C.parse_bore_checks_block(_fuzz(VALID_FUZZ[name]), (), "fuzz.yaml").bores[0].name == "b1"


def test_the_shared_checks_and_the_other_blocks_keep_their_texts(tmp_path):
    """The options are added in the block `bore_checks` only: the same wrong value in another block has the text it had (review m-4: the other blocks do not change)."""
    d = yaml.safe_load(write(tmp_path, None).read_text())
    d["settings"]["voxel_size"] = "x"
    p = tmp_path / "other.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError) as exc:
        load_config(p)
    assert str(exc.value) == f"'settings.voxel_size' must be a number, got 'x' (file: {p})"
    d["settings"]["voxel_size"] = 2.0
    d["springs"] = [{"region": 5}]
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(UserError) as exc:
        load_config(p)
    assert str(exc.value) == f"'springs[0].region' must be a non-empty string, got 5 (file: {p})"        # the shared `_str` has no option outside the block


def test_parse_function_for_tools():
    bc = C.parse_bore_checks_block(copy.deepcopy(good_block()), (), "x.yaml")
    assert bc.bores[0].name == "b1"
    with pytest.raises(UserError, match=r"\(file: .*x\.yaml\)"):
        C.parse_bore_checks_block({"bores": []}, (), "x.yaml")


# ---- the default path: a YAML without the block loads as at the base revision (plan 10.2) -----------------------------------------------

EXAMPLES = ["examples/hook/hook.yaml", "examples/hook/hook_mirror.yaml", "examples/hook/hook_short.yaml", "examples/hook/hook_stress.yaml",
            "examples/ladle/ladle.yaml", "examples/interfaces/bracket.yaml", "examples/interfaces/bracket_nostock.yaml"]


def _same_where_stored(stored, now, where):
    """Each value of `stored` is in `now` with the same value. A key that `now` has more (a field of another branch) is not compared."""
    if isinstance(stored, dict):
        assert isinstance(now, dict), where
        for k, v in stored.items():
            assert k in now, f"{where}.{k}"
            _same_where_stored(v, now[k], f"{where}.{k}")
    elif isinstance(stored, list):
        assert isinstance(now, list) and len(now) == len(stored), where
        for i, (a, b) in enumerate(zip(stored, now)):
            _same_where_stored(a, b, f"{where}[{i}]")
    else:
        assert now == stored, where


def test_the_examples_load_with_no_block_and_every_other_field_as_at_be8bef6():
    """`tests/fixtures/config_examples_be8bef6.json` is `dataclasses.asdict(load_config(...))` of each example, made at be8bef6 (the field `bore_checks` did not exist).

    Every field of be8bef6 keeps its value. A field that another branch adds (for example `acceptance`, `joints`) is not compared."""
    import dataclasses
    from pathlib import Path
    from topopt.config import load_config
    root = Path(__file__).resolve().parent.parent
    stored = json.loads((Path(__file__).parent / "fixtures" / "config_examples_be8bef6.json").read_text())
    assert sorted(stored) == sorted(EXAMPLES)
    for rel in EXAMPLES:
        cfg = load_config(root / rel)
        assert cfg.bore_checks is None
        d = dataclasses.asdict(cfg)
        d.pop("bore_checks")
        _same_where_stored(stored[rel], json.loads(json.dumps(d, default=str).replace(str(root), "<repo>")), rel)


def test_the_switch_of_verify_has_the_default_none_and_the_cli_flag_is_three_valued():
    import inspect
    from topopt.cli import build_parser
    from topopt.verify.run import preflight, verify_run
    for fn in (verify_run, preflight):
        assert inspect.signature(fn).parameters["bore_checks"].default is None
    ns = build_parser().parse_args
    assert ns(["verify", "r"]).bore_checks is None and ns(["verify", "r", "--bore-checks"]).bore_checks is True
    assert ns(["verify", "r", "--no-bore-checks"]).bore_checks is False
