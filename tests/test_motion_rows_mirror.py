"""A5 rows with the mirror of A4a (CONTRACTS.md sections 26.1 and 31): merge-gate tests.

The part is the cantilever of `test_mirror.write_yaml` (symmetric about y = 0 and z = 0), with probe regions that are
symmetric about both planes (`mid`, `a`, `b`) and rows of both kinds. The chain rule averages every gradient by the transpose
of the forward average (`Model.gradient_wrt_x`), so the rows need no code for the mirror. These tests check that.
"""

import numpy as np
import pytest

import test_motion_rows as base
from test_mirror import box, keep_out_corner, two_planes, write_yaml
from topopt import symmetry
from topopt.config import load_config
from topopt.formulation import Formulation, weighted_compliance
from topopt.mma import MmaState
from topopt.optimize import Model, formulation_from_config
from topopt.problem import build_problem

LIMITS = {
    "displacement": [
        # across the plane y = 0: only the case `mixed` has a load that is not symmetric about it (see the last test)
        {"name": "mid_y", "region": "mid", "limit_mm": 0.01, "direction": [0, 1, 0], "cases": ["mixed"]},
        {"name": "mid_z", "region": "mid", "limit_mm": 0.02, "direction": [0, 0, -1], "cases": ["down", "mixed"]},
        {"name": "tip_n", "region": "tip", "limit_mm": 0.03},
    ],
    "alignment": [
        {"region_a": "a", "region_b": "b", "axis": [1, 0, 0], "shift_limit_mm": 0.003, "tilt_limit_rad": 5e-4,
         "cases": ["down", "mixed"]},
    ],
}


def case_yaml(tmp_path, planes, edit=None, limits=LIMITS, **kw):
    box(tmp_path / "mid.stl", (6, -4, -4), (10, 4, 4))
    box(tmp_path / "a.stl", (3, -4, -4), (5, 4, 4))
    box(tmp_path / "b.stl", (11, -4, -4), (13, 4, 4))

    def add(d):
        d["regions"]["measure"] = {"mid": "mid.stl", "a": "a.stl", "b": "b.stl"}
        d["limits"] = limits
        if edit:
            edit(d)
    return write_yaml(tmp_path, planes, edit=add, **kw)


def model_of(path):
    m = Model(build_problem(load_config(path)))
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13
    return m


@pytest.mark.parametrize("variant", ["one_plane", "two_planes_with_orphans"])
def test_row_gradients_match_finite_differences_at_an_asymmetric_x(tmp_path, variant):
    """The rows of A5 against central differences of x, with the mirror average inside the model."""
    if variant == "one_plane":
        path = case_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}])
    else:   # a keep-out corner: the two maps do not commute, orphans
        path = case_yaml(tmp_path, two_planes(), edit=keep_out_corner(tmp_path))
    m = model_of(path)
    form = formulation_from_config(m.problem.config, m.n_design, m)
    form = Formulation(weighted_compliance(), form.constraints)
    # No row here has a load that is symmetric about a plane across its direction: such a row has the value eps and the
    # gradient 0, and a finite difference of a function that is 0 up to rounding says nothing (see the last test).
    err = {n: e for n, e in base.fd_error(m, form, steps=(1e-3, 1e-4)).items() if n != "volume"}
    assert len(err) == 1 + 2 + 3 + 4 and max(err.values()) < base.FD_DIRECT, err
    m.solver.free()


def test_row_gradients_are_symmetric_about_the_plane(tmp_path):
    m = model_of(case_yaml(tmp_path, [{"normal": "y", "at_mm": 0.0}]))
    form = formulation_from_config(m.problem.config, m.n_design, m)
    x = np.random.default_rng(4).uniform(0.2, 0.8, m.n_design)
    ev = base.evaluate(m, form, x)
    perm, = symmetry.perms_of(m)
    for r, c in enumerate(form.constraints):
        scale = np.abs(ev.dg[r]).max() or 1.0
        assert np.abs(ev.dg[r] - ev.dg[r][perm]).max() <= 1e-12 * scale, c.name
    m.solver.free()


def row_values(m, form, x):
    ev = base.evaluate(m, form, x)
    return {e["name"]: e["value"] for e in ev.constraints}, ev


def test_a_displacement_across_the_plane_is_zero_for_a_symmetric_load(tmp_path):
    """Case `down` has a force along z: it is symmetric about y = 0. The mean y displacement of a symmetric region is 0.

    The smooth norm then has the value eps and the gradient 0 (CONTRACTS.md section 31, combinations). Without the mirror the
    same x (not symmetric) gives a value above eps.
    """
    only_down = lambda d: d.__setitem__("load_cases", [c for c in d["load_cases"] if c["name"] == "down"])   # noqa: E731
    limits = {"displacement": [{"name": "mid_y", "region": "mid", "limit_mm": 0.01, "direction": [0, 1, 0]}]}
    eps = rm_eps(limits["displacement"][0]["limit_mm"])
    x = None
    values = {}
    for key, planes in (("mirror", [{"normal": "y", "at_mm": 0.0}]), ("plain", None)):
        sub = base.sub(tmp_path, key)
        m = model_of(case_yaml(sub, planes, edit=only_down, limits=limits))
        form = formulation_from_config(m.problem.config, m.n_design, m)
        x = np.random.default_rng(2).uniform(0.2, 0.8, m.n_design) if x is None else x
        vals, ev = row_values(m, form, x)
        values[key] = (vals["displacement:mid_y:down"], np.abs(ev.dg[-1]).max())
        m.solver.free()
    assert values["mirror"][0] == pytest.approx(eps, rel=1e-12)    # y = 0 up to rounding
    assert values["mirror"][1] < 1e-9 * values["plain"][1]         # the gradient is 0
    assert values["plain"][0] > eps * (1 + 1e-6) and values["plain"][1] > 0   # no mirror: the part is not symmetric


def rm_eps(limit):
    from topopt.rows_motion import EPS_FRACTION
    return EPS_FRACTION * limit
