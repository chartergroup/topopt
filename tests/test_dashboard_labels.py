"""The places of the load case labels of the dashboard (#296). Load cases with their arrows at the same place share one label; a label
starts past the tail of its arrow and moves away from the tip when it would cover an earlier one, so it never lies on its own arrow.

`labels.js` has no DOM and no three.js, so Node runs it. The tests that run Node are skipped where there is no `node`.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

LABELS = Path(__file__).resolve().parents[1] / "topopt" / "dashboard" / "js" / "labels.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
W, H = 16, 4    # the size of a one-name label: the dashboard uses 1.6 and 0.4 times the arrow length
L = 50.0        # an arrow length
DOWN, UP, LEVEL = [0, 0, -1], [0, 0, 1], [1, 0, 0]


def call(fn, *args):
    script = (f"import {{ {fn} }} from {json.dumps(LABELS.as_uri())};\n"
              f"console.log(JSON.stringify({fn}(...{json.dumps(list(args))})));\n")
    done = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-500:]
    return json.loads(done.stdout)


def shifts(points, way=-1, w=W):
    return call("labelShifts", points, w, H, [way] * len(points))


def arrow(tail, d, ci):
    return {"tail": tail, "dir": d, "ci": ci}


# --- labelShifts: labels that would lie on top of each other ---------------------------------------------------------------------


@needs_node
def test_labels_that_collide_are_stacked_in_load_case_order():
    assert shifts([[0, 0, 10], [0, 0, 10]]) == [0, -H]                    # the later label goes past the earlier one
    assert shifts([[0, 0, 10], [0, 0, 10], [0, 0, 10]]) == [0, -H, -2 * H]
    assert shifts([[0, 0, 10], [0, 0, 10]], way=1) == [0, H]
    assert shifts([[0, 0, 9], [0, 0, 11]]) == [0, -6]                     # a label just above an earlier one goes below it


@needs_node
def test_labels_apart_do_not_move():
    assert shifts([[0, 0, 0], [W, 0, 0], [0, 0, -H], [-12, -12, 0]]) == [0, 0, 0, 0]
    assert shifts([]) == []


@needs_node
def test_a_moved_label_does_not_cover_a_label_past_it():
    # the third label collides with the first; one label height down it would cover the second, so it goes below both
    assert shifts([[0, 0, 0], [3, 0, -4], [0, 0, 0]]) == [0, 0, -8]


@needs_node
def test_two_labels_collide_when_their_half_widths_overlap():
    """A label with two names is wider. Two labels collide when their centres are closer than half the sum of their widths, so a wide
    label does not push a narrow one that it does not reach."""
    assert shifts([[0, 0, 0], [100, 0, 0]], w=[150, 40]) == [0, 0]        # 95 < 100: apart
    assert shifts([[0, 0, 0], [100, 0, 0]], w=[170, 40]) == [0, -H]       # 105 > 100: they collide


# --- sameArrowGroups: arrows at the same place share one label ---------------------------------------------------------------------


@needs_node
def test_arrows_at_the_same_place_are_one_group():
    """The footbridge: `uniform` and `half_left` have their (first) force on the same region, so `half_left`'s arrow is drawn over the
    arrow of `uniform` and hides it. The two cases share one label; `half_right` is alone."""
    a = {"tail": [70, 0, 140], "dir": DOWN}
    b = {"tail": [210, 0, 140], "dir": DOWN}
    assert call("sameArrowGroups", [a, a, b], 1.0) == [[0, 1], [2]]
    assert call("sameArrowGroups", [a, b, a], 1.0) == [[0, 2], [1]]          # by the first index, each group in case order
    assert call("sameArrowGroups", [a, a, a], 1.0) == [[0, 1, 2]]
    assert call("sameArrowGroups", [], 1.0) == []


@needs_node
def test_arrows_apart_or_in_other_directions_are_not_one_group():
    a = {"tail": [0, 0, 0], "dir": DOWN}
    assert call("sameArrowGroups", [a, {"tail": [0, 0, 0], "dir": UP}], 1.0) == [[0], [1]]          # the opposite force
    assert call("sameArrowGroups", [a, {"tail": [1.0, 0, 0], "dir": DOWN}], 1.0) == [[0], [1]]      # the tails 1 apart: not < 1
    assert call("sameArrowGroups", [a, {"tail": [0.9, 0, 0], "dir": DOWN}], 1.0) == [[0, 1]]
    import math
    tilt = lambda deg: [math.sin(math.radians(deg)), 0, -math.cos(math.radians(deg))]           # noqa: E731
    assert call("sameArrowGroups", [a, {"tail": [0, 0, 0], "dir": tilt(2)}], 1.0) == [[0, 1]]
    assert call("sameArrowGroups", [a, {"tail": [0, 0, 0], "dir": tilt(5)}], 1.0) == [[0], [1]]


# --- labelPlaces: past the tail, never on the arrow ---------------------------------------------------------------------------------


def past_the_tail(place, a):
    """The label centre lies before the tail along the force: its projection on the direction is negative."""
    return sum((p - t) * d for p, t, d in zip(place, a["tail"], a["dir"])) < 0


@needs_node
def test_the_footbridge_labels_stay_above_their_tails_and_do_not_move():
    tz = 89.5 + L
    arrows = [arrow([70, 0, tz], DOWN, 0), arrow([70, 0, tz], DOWN, 1), arrow([210, 0, tz], DOWN, 2)]
    groups = call("sameArrowGroups", arrows, 0.135 * L)
    assert groups == [[0, 1], [2]]
    places = call("labelPlaces", arrows, groups, L, [2.5 * L, 1.6 * L])        # one wide label with two names, one with one name
    assert places == [[70, 0, pytest.approx(tz + 0.32 * L)], [210, 0, pytest.approx(tz + 0.32 * L)]]
    assert all(p[2] > tz for p in places)                                    # above the tail of the downward arrows


@needs_node
def test_a_label_that_would_cover_an_earlier_one_moves_away_from_its_tip():
    """Two arrows close together (not at the same place): the later label moves past the earlier one away from the tip of its arrow.
    Down to the tip it would lie on its own shaft; this was the half_left label of the footbridge clip before."""
    tz = 100.0
    down = [arrow([0, 0, tz], DOWN, 0), arrow([10, 0, tz], DOWN, 1)]
    places = call("labelPlaces", down, [[0], [1]], L, [1.6 * L, 1.6 * L])
    assert places[0][2] == pytest.approx(tz + 0.32 * L)
    assert places[1][2] == pytest.approx(tz + 0.72 * L)                      # one label height above the first one: up
    up = [arrow([0, 0, tz], UP, 0), arrow([10, 0, tz], UP, 1)]
    places = call("labelPlaces", up, [[0], [1]], L, [1.6 * L, 1.6 * L])
    assert places[1][2] < places[0][2] < tz                                  # an upward force: below the tail, and on down


@needs_node
def test_every_label_lies_past_the_tail_of_its_arrow():
    # tails of the same direction 8 to 16 apart (more than the 6.75 of the same place), so the labels collide and move
    arrows = [arrow([0, 0, 0], DOWN, 0), arrow([10, 0, 0], DOWN, 1), arrow([0, 12, 0], DOWN, 2), arrow([0, 0, 0], UP, 3),
              arrow([10, 10, 0], UP, 4), arrow([0, 0, 0], LEVEL, 5), arrow([0, 0, 8], LEVEL, 6), arrow([0, 10, 8], [0.6, 0, -0.8], 7)]
    groups = call("sameArrowGroups", arrows, 0.135 * L)
    assert groups == [[i] for i in range(len(arrows))]
    places = call("labelPlaces", arrows, groups, L, [1.6 * L] * len(groups))
    assert len(places) == len(groups) == len(arrows)
    for g, p in zip(groups, places):
        assert past_the_tail(p, arrows[g[0]]), (g, p)
    assert len({round(p[2], 6) for p in places}) > 2                          # some labels did move


# --- the scene ----------------------------------------------------------------------------------------------------------------


def test_the_scene_gives_one_label_to_each_group_and_places_it_with_labels_js():
    app = (LABELS.parent / "app.js").read_text()
    assert 'import { labelPlaces, sameArrowGroups } from "./labels.js";' in app
    assert "const groups = sameArrowGroups(arrows, len * 0.135);" in app             # 0.135: the radius of the arrow head
    assert "headL = len * 0.32, shaftL = len - headL, r = len * 0.045;" in app and "new THREE.ConeGeometry(r * 3, headL, 16)" in app
    assert "makeLabel(group.map((i) => ({ text: drawn[i].name, color: drawn[i].color })))" in app
    assert "lab.scale.set(len * 0.4 * lab.userData.aspect, len * 0.4, 1);" in app      # one-name label: 256 x 64, so 1.6 len wide
    assert "cv.width = Math.max(256, Math.ceil(textW + 24)); cv.height = 64;" in app
    assert "labelPlaces(arrows, groups, len, labels.map((l) => l.scale.x))" in app
    assert "labelShifts" not in app                                                      # through labelPlaces only
