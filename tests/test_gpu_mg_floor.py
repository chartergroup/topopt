"""The stop test of gpu-mg at the FP64 floor (G2; CONTRACTS.md section 20).

On a design with a near-void region around stiff strips (a soft start: start density = volume fraction, penal 5 or more), the
FP64 true residual of any solve has a floor above 1e-8: about eps * || |K| |u| || / ||f||. A direct solver with a backward
error of 1e-16 stops there too. The rule under test: a solve is accepted when its relative residual is below `gpu_tol`, or when
its normwise backward error `eta = |r| / (| |K| |u| | + |f|)` is at most `gpu_backward_error` (0 turns the rule off).

Part 1 uses a dense stand-in for the host operator and for the device PCG: no OpenCL context. Part 2 runs the real engine on the
OpenCL device of the other gpu-mg tests (pocl on this project's Linux host).
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("pyopencl")

import test_gpu_mg  # noqa: E402
from test_gpu_mg import SMALL, case_yaml, need_case  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt.config import GPU_BACKWARD_ERROR_DEFAULT, UserError, load_config  # noqa: E402
from topopt.gpu_mg import engine  # noqa: E402
from topopt.gpu_mg.model import GpuModel, GpuParams  # noqa: E402
from topopt.problem import build_problem  # noqa: E402
from topopt.snapshot import RunWriter  # noqa: E402

dev = test_gpu_mg.dev      # the fixture of the other gpu-mg tests (the OpenCL device; a CPU device on the host of the tests)
ROUNDING = 1.1e-16     # the unit roundoff of FP64
opencl = pytest.mark.opencl_context       # see tests/conftest.py: skipped on a Mac without TOPOPT_TEST_REAL_GPU=1


# --- part 1: a dense stand-in ---------------------------------------------------------------------------


class DenseHost:
    """The part of `HostMG` that the residual code uses, on a dense matrix K: `apply_fine`, `apply_fine_abs`, `levels[0].fix`.
    The vectors are (n, 3) like the vectors of the fine level.

    `rounding` > 0 models the rounding error of an FP64 matrix-vector product: each entry of `K x` gets an error of
    `rounding * (|K| |x|)_i` with a fixed random sign (the bound of the error of a product; the unit roundoff is 1.1e-16). The
    residual of any solve then has a floor of about `rounding * || |K||x| || / ||b||`, whatever the library of the dense solve rounds to.
    """

    def __init__(self, K, n_nodes, rounding=0.0):
        self.K, self.absK, self.rounding = K, np.abs(K), rounding
        self.sign = np.random.default_rng(5).choice([-1.0, 1.0], size=K.shape[0])
        self.levels = [SimpleNamespace(fix=np.zeros((n_nodes, 3), np.uint8))]

    def apply_fine(self, x, y):
        v = self.K @ x.reshape(-1)
        if self.rounding:
            v = v + self.rounding * self.sign * (self.absK @ np.abs(x).reshape(-1))
        y[...] = v.reshape(y.shape)

    def apply_fine_abs(self, x, y):
        y[...] = (self.absK @ np.abs(x).reshape(-1)).reshape(y.shape)


# The cutoff of the stand-in, relative to the largest singular value. The null singular values of an exactly singular K come out at the
# rounding level, which depends on the LAPACK build and the CPU: up to about n eps ||K|| (8e-14 for the chain of 30 nodes). The default
# cutoff of np.linalg.pinv (1e-15 ||K||) is inside that band, so on one CI runner it kept a null value, the solve became unbounded and
# the backward error test accepted it. The regular chain of these tests has 8.4e-11 ||K|| as its smallest singular value (support 1e-8).
SINGULAR_RTOL = 1e-12


class DensePcg:
    """The device PCG as a direct solve of the dense matrix (`pinv` for a singular one). `solve_refined` calls `pcg(rhs, rtol, maxit)`."""

    def __init__(self, K):
        self.K, self.calls = K, 0
        U, s, Vt = np.linalg.svd(K)                 # one SVD for the test and the cut: a second SVD can round the null values otherwise
        keep = s > SINGULAR_RTOL * s.max()
        if not keep.all():                          # singular: the minimum-norm solution, with the null space cut at the same level
            self.Kinv = (Vt[keep].T / s[keep]) @ U[:, keep].T
        else:
            self.Kinv = np.linalg.inv(K)

    def pcg(self, rhs, rtol, maxit=300):
        self.calls += 1
        d = np.stack([(self.Kinv @ r.reshape(-1)).reshape(r.shape) for r in rhs])
        return d, {"executed": 1}


def chain(n_nodes, support, rounding=0.0):
    """A stiff chain on a soft support: the Laplacian of a chain (3 dofs per node, free-free) plus a spring `support` at the last
    node. A load at the last node moves the whole chain by about 1 / support, so |K||u| is about 2 / support times the load: the
    soft start in miniature. `support = 0` leaves a free rigid mode."""
    L = np.diag(np.r_[1.0, 2.0 * np.ones(n_nodes - 2), 1.0]) - np.diag(np.ones(n_nodes - 1), 1) - np.diag(np.ones(n_nodes - 1), -1)
    K = np.kron(L, np.eye(3))
    K[-3:, -3:] += support * np.eye(3)
    B = np.zeros((1, n_nodes, 3))
    B[0, -1, 2] = 1.0
    return K, B, DenseHost(K, n_nodes, rounding), DensePcg(K)


def refine(g, host, B, rtol, inner_rtol=1e-2, **kw):
    """`engine.solve_refined` with the stand-ins (duck typing: the loop needs `g.pcg`, `host.apply_fine`, `host.apply_fine_abs`, `host.levels[0].fix`)."""
    return engine.solve_refined(g, host, B, rtol, inner_rtol, **kw)  # type: ignore[arg-type]


def check(host, B, X, rtol, limit):
    """`engine.check_solution` with the stand-in host."""
    return engine.check_solution(host, B, X, rtol, limit)  # type: ignore[arg-type]


def floor_of(host, g, B, rtol=1e-30):
    """The residual that the restart loop reaches when it cannot go lower: run it to the end with an unreachable target."""
    _, info = refine(g, host, B, rtol, maxref=3)
    return min(max(v) for v in info["rel_hist"][1:])


def test_the_floor_of_the_stand_in_is_above_1e_8_and_eta_is_at_machine_precision():
    _, B, host, g = chain(30, 1e-8, ROUNDING)
    _, info = refine(g, host, B, 1e-30, maxref=3, backward_error=1e-14)
    assert info["accepted_by"] == "backward_error"                       # the first step reaches the floor at once
    assert min(max(v) for v in info["rel_hist"][1:]) > 1e-8              # a floor above the old target ...
    assert 1e-18 < info["eta_hist"][-1][0] < 1e-14                      # ... with a backward error of a few eps


def test_a_history_at_the_floor_passes_with_a_small_eta_and_fails_with_the_rule_off():
    _, B, host, g = chain(30, 1e-8, ROUNDING)
    floor = floor_of(host, g, B)
    target = floor / 10                                                  # no FP64 solve reaches it
    X, info = refine(g, host, B, target, maxref=12, backward_error=1e-14)
    assert info["converged"] and info["accepted_by"] == "backward_error" and info["steps"] == 1
    assert max(info["rel_hist"][-1]) > target and info["eta_hist"][-1][0] <= 1e-14
    X0, off = refine(g, host, B, target, maxref=12, backward_error=0.0)      # the old behavior
    assert not off["converged"] and off["accepted_by"] is None and off["steps"] == 12 and not off["stagnated"]
    assert all(e is None for e in off["eta_hist"])                       # the rule is off: eta is not computed
    assert np.array_equal(X0[0], X[0]) or np.allclose(X0, X, rtol=1e-6)  # the same solution; only the stop test differs


def test_a_target_above_the_floor_is_accepted_by_the_residual_and_eta_is_not_computed():
    _, B, host, g = chain(30, 1e-8, ROUNDING)
    floor = floor_of(host, g, B)
    _, info = refine(g, host, B, floor * 10, maxref=12, backward_error=1e-14)
    assert info["accepted_by"] == "tol" and all(e is None for e in info["eta_hist"])
    _, same = refine(g, host, B, floor * 10, maxref=12, backward_error=0.0)
    assert same["accepted_by"] == "tol" and same["rel_hist"] == info["rel_hist"]


def test_the_rule_off_loop_is_the_loop_of_d19c7fb():
    """A copy of the loop of d19c7fb (without the backward error) gives the same history, steps and solution."""
    _, B, host, g = chain(30, 1e-8, ROUNDING)

    def old_loop(g, host, B, rtol, inner_rtol, maxref, maxit):
        nr, u, y = B.shape[0], np.zeros_like(B), np.zeros(B.shape[1:])
        fixed = host.levels[0].fix.reshape(-1).astype(bool)
        bn = np.array([np.linalg.norm(B[c]) for c in range(nr)])
        its_per, rel_hist, converged = [], [], False
        for k in range(maxref + 1):
            r = np.empty_like(B)
            for c in range(nr):
                host.apply_fine(u[c], y)
                r[c] = B[c] - y
                r[c].reshape(-1)[fixed] = 0.0
            rn = np.array([np.linalg.norm(r[c]) for c in range(nr)])
            rel = np.where(bn > 0, rn / bn, 0.0)
            rel_hist.append(rel.tolist())
            if np.all(rel < rtol):
                converged = True
                break
            if k == maxref:
                break
            live = rn > 0
            scale = np.where(live, rn, 1.0)
            d, info = g.pcg(r / scale[:, None, None], inner_rtol, maxit=maxit)
            u += d * np.where(live, scale, 0.0)[:, None, None]
            its_per.append(info["executed"])
        return u, len(its_per), rel_hist, converged

    for rtol in (1e-3, 1e-12, 1e-30):          # converges after 2 steps; reaches the floor; never converges
        ref_u, ref_steps, ref_hist, ref_ok = old_loop(g, host, B, rtol, 1e-2, 12, 300)
        u, info = refine(g, host, B, rtol, maxref=12, maxit=300, backward_error=0.0)
        assert np.array_equal(u, ref_u) and info["steps"] == ref_steps
        assert info["rel_hist"] == ref_hist and info["converged"] == ref_ok and not info["stagnated"]


def test_a_singular_system_with_a_free_rigid_mode_still_fails_and_the_loop_stops_early():
    """The stand-in solves with the minimum-norm solution (`pinv`), so the solution stays bounded and `eta` stays large.

    The limit of the rule (CONTRACTS.md section 20): it assumes a bounded solution. `eta` divides by `|| |K||u| ||`, so a solve with
    a large null-space part in `u` can have a small `eta` (the review of G2 measured eta 2.6e-15 at a relative residual of 0.18 for an
    inverse of K + 1e-14 I). `topopt run` stops a floating piece before the solve (`connectivity.floating_error`), and `E_MIN` keeps K regular.
    """
    _, B, host, g = chain(30, 0.0)               # the whole chain can move: the load has a part along the rigid mode
    _, info = refine(g, host, B, 1e-8, maxref=12, backward_error=1e-14)
    assert not info["converged"] and info["accepted_by"] is None
    assert min(e[0] for e in info["eta_hist"][1:]) > 1e-3          # eta stays large: the residual is not rounding
    assert info["stagnated"] and info["steps"] == 3 and len(info["rel_hist"]) == 4      # the loop stopped: no fall of 10 % in two steps
    _, off = refine(g, host, B, 1e-8, maxref=12, backward_error=0.0)
    assert not off["converged"] and off["steps"] == 12             # the old behavior: all the steps


@pytest.mark.parametrize("size", [1e-15, 3e-15, 1e-14])
def test_the_singular_stand_in_does_not_depend_on_the_rounding_of_the_svd(size):
    """The stand-in of the singular chain sees K with a change at the rounding level of a dense SVD (n eps ||K|| is 8e-14 here), as
    another LAPACK build or CPU does. Its solve stays the minimum-norm one, so the loop still fails and stops early. With the default
    cutoff of np.linalg.pinv the stand-in kept a null value at these sizes and the backward error test accepted an unbounded solve."""
    K, B, host, _ = chain(30, 0.0)
    E = np.random.default_rng(0).standard_normal(K.shape)
    _, info = refine(DensePcg(K + size * (E + E.T) / 2), host, B, 1e-8, maxref=12, backward_error=1e-14)
    assert not info["converged"] and info["accepted_by"] is None
    assert min(e[0] for e in info["eta_hist"][1:]) > 1e-3
    assert info["stagnated"] and info["steps"] == 3 and len(info["rel_hist"]) == 4


def test_the_stagnation_rule_does_not_stop_a_loop_that_still_falls():
    """Each step falls by 100 times: no stop. The stand-in PCG solves with a relative error of 1e-2."""
    _, B, host, g = chain(30, 1e-2)
    exact = g.pcg

    def inexact(rhs, rtol, maxit=300):
        d, info = exact(rhs, rtol, maxit)
        return d * (1 - 1e-2), info                                   # each step removes 99 % of the residual
    g.pcg = inexact
    _, info = refine(g, host, B, 1e-12, maxref=12, backward_error=1e-14)
    assert info["converged"] and not info["stagnated"] and info["accepted_by"] == "tol" and info["steps"] >= 4


def inexact_pcg(g, fall):
    """The stand-in PCG with an inner solve that removes only the part `fall` of the residual in each step (`1 - fall` of it stays)."""
    exact = g.pcg

    def pcg(rhs, rtol, maxit=300):
        d, info = exact(rhs, rtol, maxit)
        return d * fall, info
    return pcg


def test_a_loop_that_passes_through_an_eta_of_5e_15_takes_one_more_step_with_the_default_limit():
    """The shape of the d3b start on the real GPU: the step before the floor has eta 5e-15 (a relative residual of 1e-5 against a floor
    of 8e-8), the next step is at the floor (eta 4e-17). The limit 1e-14 accepted the first (G2 review, M1); the default 1e-15 goes on."""
    _, B, host, g = chain(30, 1e-8, ROUNDING)
    g.pcg = inexact_pcg(g, 1 - 1e-5)                                    # each step removes all but 1e-5 of the residual
    _, loose = refine(g, host, B, 1e-30, maxref=12, backward_error=1e-14)
    assert loose["accepted_by"] == "backward_error" and loose["steps"] == 1
    eta1, rel1 = loose["eta_hist"][1][0], loose["rel_hist"][1][0]
    assert 2e-15 < eta1 < 1e-14 and rel1 > 3e-6                         # accepted 100 times above the rel of the floor ...
    _, info = refine(g, host, B, 1e-30, maxref=12, backward_error=GPU_BACKWARD_ERROR_DEFAULT)
    assert GPU_BACKWARD_ERROR_DEFAULT == 1e-15
    assert info["accepted_by"] == "backward_error" and info["steps"] == 2 and not info["stagnated"]     # ... the default takes one more step
    assert info["eta_hist"][1][0] > GPU_BACKWARD_ERROR_DEFAULT and info["eta_hist"][2][0] < 1e-16 * 1e2
    assert info["rel_hist"][2][0] < rel1 / 50                           # and ends near the floor of the residual


def test_the_stagnation_stop_counts_only_the_columns_that_are_not_accepted_yet():
    """Two load cases in one block (G2 review, m1). Column 0 sits at its floor and stays flat (eta 4e-17: accepted by the backward error).
    Column 1 falls by 25 % per step, above the target. The block must go on until column 1 meets the target: 7 steps."""
    n = 30
    K, _, host, _ = chain(n, 1e-8, ROUNDING)
    Kinv = np.linalg.inv(K)
    B = np.zeros((2, n, 3))
    B[0, -1, 2] = 1.0                                                   # excites the soft mode: |K||u| / |f| is about 2e9
    B[1, 5, 0], B[1, 6, 0] = 1.0, -1.0                                  # a self-equilibrated pair: a small |K||u|, no floor problem

    def pcg(rhs, rtol, maxit=300):
        d = np.stack([(Kinv @ r.reshape(-1)).reshape(r.shape) for r in rhs])
        d[1] *= 0.25                                                    # column 1: each step removes 25 % of its residual
        return d, {"executed": 1}
    g = SimpleNamespace(pcg=pcg)
    u0 = np.stack([(Kinv @ B[c].reshape(-1)).reshape(n, 3) for c in range(2)])
    e = np.random.default_rng(0).standard_normal((n, 3))                # an error along a stiff direction: eta of column 1 stays large
    u0[1] += e * (6e-8 * np.linalg.norm(B[1]) / np.linalg.norm((K @ e.reshape(-1)).reshape(n, 3)))
    _, info = refine(g, host, B, 1e-8, maxref=12, u0=u0, backward_error=GPU_BACKWARD_ERROR_DEFAULT)
    assert info["converged"] and not info["stagnated"] and info["steps"] == 7
    assert info["eta_hist"][0][0] < GPU_BACKWARD_ERROR_DEFAULT          # column 0: accepted by the backward error from the start, flat
    assert max(r[0] for r in info["rel_hist"]) > 5e-8 > 1e-8           # ... above the target all the time
    assert info["rel_hist"][-1][1] < 1e-8                               # column 1 met the target
    _, alone = refine(SimpleNamespace(pcg=lambda rhs, rtol, maxit=300: (pcg(np.concatenate([rhs, rhs]), rtol)[0][1:2], {"executed": 1})),
                      host, B[1:2].copy(), 1e-8, maxref=12, u0=u0[1:2].copy(), backward_error=GPU_BACKWARD_ERROR_DEFAULT)
    assert alone["converged"] and alone["steps"] == 7                   # the same column alone needs the same 7 steps


def test_a_loop_that_falls_by_25_percent_per_step_does_not_stop():
    """The stop is for a loop that has stopped falling: 10 % over two steps. A fall of 25 % per step (44 % over two steps) goes on."""
    _, B, host, g = chain(30, 1e-2)
    g.pcg = inexact_pcg(g, 0.25)                                        # each step removes 25 % of the residual
    _, info = refine(g, host, B, 0.1, maxref=12, backward_error=GPU_BACKWARD_ERROR_DEFAULT)
    assert info["converged"] and not info["stagnated"] and info["accepted_by"] == "tol" and info["steps"] == 9      # 0.75**9 < 0.1
    assert min(e[0] for e in info["eta_hist"][1:-1]) > 1e-6             # eta was far above the limit all the way: only the stop rule could end it
    assert engine.STAGNATION_FALL == 0.10


def test_backward_errors_divides_by_the_norm_of_the_load_too():
    """eta = |r| / (|| |K||u| || + |f|). With K = I and u = b / 2: r = b / 2, |K||u| = |b| / 2, so eta = 1/3 (without the load: 1)."""
    n = 5
    host = DenseHost(np.eye(3 * n), n)
    b = np.arange(1.0, 3 * n + 1).reshape(1, n, 3)
    u = 0.5 * b
    r, rn = engine.true_residual(host, b, u)                           # type: ignore[arg-type]
    bn = np.array([np.linalg.norm(b[0])])
    eta = engine.backward_errors(host, u, rn, bn, np.array([True]))    # type: ignore[arg-type]
    assert eta[0] == pytest.approx(1.0 / 3.0, rel=1e-14)
    # u = 0: |K||u| = 0, so eta = |r| / |f| = 1 (no division by zero); a column that `want` does not select is not computed (nan)
    two = np.concatenate([b, b])
    u2 = np.concatenate([u, np.zeros_like(u)])
    _, rn2 = engine.true_residual(host, two, u2)                       # type: ignore[arg-type]
    eta2 = engine.backward_errors(host, u2, rn2, np.array([bn[0], bn[0]]), np.array([False, True]))     # type: ignore[arg-type]
    assert np.isnan(eta2[0]) and eta2[1] == pytest.approx(1.0, rel=1e-14)


def test_check_solution_is_the_acceptance_of_a_first_pass():
    _, B, host, g = chain(30, 1e-8, ROUNDING)
    X = g.pcg(B, 1e-8)[0]
    floor = floor_of(host, g, B)
    near = check(host, B, X, floor * 10, 1e-14)
    assert near["ok"] and near["accepted_by"] == "tol" and np.isnan(near["eta"]).all()       # eta is not computed when it is not needed
    at = check(host, B, X, floor / 10, 1e-14)
    assert at["ok"] and at["accepted_by"] == "backward_error" and at["eta"][0] <= 1e-14 and at["rel"][0] > floor / 10
    off = check(host, B, X, floor / 10, 0.0)
    assert not off["ok"] and off["accepted_by"] is None
    wrong = X + 0.1 * np.linalg.norm(X) * np.random.default_rng(1).standard_normal(X.shape) / np.sqrt(X.size)   # 10 % of noise in every dof
    bad = check(host, B, wrong, floor / 10, 1e-14)
    assert not bad["ok"] and bad["accepted_by"] is None and bad["eta"][0] > 1e-6
    soft = check(host, B, X * 1.01, floor / 10, 1e-14)       # 1 % wrong along the soft mode: still a tiny backward error
    assert soft["rel"][0] > 1e-3 and soft["eta"][0] < 1e-9                    # (a backward error does not see the error of u)
    assert at["seconds"] >= 0


def test_the_absolute_operator_is_the_sum_of_the_absolute_element_matrices():
    from topopt.fe import edof_table, hex8_ke
    from topopt.gpu_mg.hierarchy import HostMG
    shape = (4, 2, 2)                                        # 16 elements; the nodes of the face i = 0 are fixed
    edof, aidx = edof_table(shape, np.ones(shape, bool))
    ny1, nz1 = shape[1] + 1, shape[2] + 1
    fixed = np.array([3 * ((j * nz1) + k) + a for j in range(ny1) for k in range(nz1) for a in range(3)])
    h = HostMG(shape, edof, aidx, fixed, 1.0, 0.3, min_free=20)
    Ee = np.random.default_rng(2).uniform(0.1, 2.0, edof.shape[0])
    h.setup_host(Ee)
    L = h.levels[0]
    ke = hex8_ke(1.0, 0.3, 1.0)
    ref = np.zeros((3 * L.nn, 3 * L.nn))
    for e in range(L.nel):
        d = (3 * L.enodes[e][:, None] + np.arange(3)[None, :]).reshape(-1)
        ref[np.ix_(d, d)] += Ee[e] * np.abs(ke)
    x = np.random.default_rng(3).standard_normal((L.nn, 3))
    x[L.fix.astype(bool)] = 0.0
    y = np.zeros_like(x)
    h.apply_fine_abs(x, y)
    expect = (ref @ np.abs(x).reshape(-1)).reshape(-1, 3)
    expect[L.fix.astype(bool)] = 0.0
    assert np.allclose(y, expect, rtol=1e-12, atol=1e-12) and (y >= 0).all()
    plain = np.zeros_like(x)
    h.apply_fine(x, plain)
    assert (np.abs(plain) <= y + 1e-12).all()                 # |K x| is at most |K| |x|


# --- part 2: the real engine on the OpenCL device ---------------------------------------------------------------------
# A target of 1e-30 stands for the floor: no FP64 solve reaches it, and the residual of every solve ends at the rounding floor
# with a backward error of a few unit roundoffs (about 1e-16). Part 1 shows with a stand-in that the floor of a soft start is
# above 1e-8; the user test (d3b, 0.75 mm) had 2.0e-8 and 1e-16.

CASES = ["C", "B"]
UNREACHABLE = 1e-30


def lines_of(folder):
    return [json.loads(line) for line in (folder / "gpu_mg.jsonl").read_text().splitlines()]


def settings_case(tmp_path, case, **settings):
    p = case_yaml(tmp_path, case=case, h=2.0, iterations=3)
    d = yaml.safe_load(p.read_text())
    d["settings"].update(settings)
    p.write_text(yaml.safe_dump(d))
    return p


@opencl
@pytest.mark.parametrize("case", CASES)
def test_a_solve_at_the_floor_is_accepted_by_the_backward_error_and_the_run_says_so_once(tmp_path, case, dev, capsys):
    need_case(case, dev)
    p = settings_case(tmp_path, case, gpu_tol=UNREACHABLE)                       # gpu_backward_error: the default (1e-15)
    assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet"]) == 0
    recs = lines_of(tmp_path / "r")
    assert len(recs) == 4 and all(r["accepted_by"] == "backward_error" for r in recs)
    for r in recs:
        assert 0 < r["eta"] <= GPU_BACKWARD_ERROR_DEFAULT and r["true_rel"] > 0
        assert all(it["accepted_by"] == "backward_error" and it["eta"] <= GPU_BACKWARD_ERROR_DEFAULT for it in r["iterations"])
    log = (tmp_path / "r" / "run.log").read_text()
    first = [line for line in log.splitlines() if "backward-error rule accepted a solve" in line]
    assert len(first) == 1                                                       # one line for each run, not for each solve
    assert "(optimizer iteration 0)" in first[0] and "the target is 1e-30 (settings.gpu_tol)" in first[0]
    assert "the backward error is" in first[0] and "(limit 1e-15, settings.gpu_backward_error)" in first[0]
    assert "cannot go lower" not in first[0] and "about 4e-17" in first[0]            # facts only (G2 R1: the old sentence was not tested by the code)
    count = [line for line in log.splitlines() if "solves were accepted by the backward error" in line]
    assert count == ["gpu-mg: 4 of 4 solves were accepted by the backward error (settings.gpu_backward_error 1e-15); "
                     "the others by the relative residual 1e-30 (settings.gpu_tol)"]       # the count comes once, at the end of the run


@opencl
def test_the_same_run_with_the_rule_off_stops_and_leaves_the_record_of_the_failed_solve(tmp_path, dev, capsys):
    """`gpu_backward_error: 0` is the old behavior: the target is the only test. The run ends with exit code 1, and the failed solve
    is in `gpu_mg.jsonl` (before the error): the residual history that shows that the loop sat at the floor. Case C: every solve is a
    restart loop with the true residual. (Case B on a small grid: see the next test.)"""
    p = settings_case(tmp_path, "C", gpu_tol=UNREACHABLE, gpu_backward_error=0)
    assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet"]) == 1
    err = capsys.readouterr().err
    assert "did not reach the relative residual 1e-30" in err and "the rule is off" in err
    assert "rounding floor" not in err                                          # eta was not computed: no sentence about its value (G2 R1, n2)
    recs = lines_of(tmp_path / "r")
    assert len(recs) == 1 and recs[0]["failed"] is True and recs[0]["kind"] == "main" and recs[0]["iteration"] == 0
    it = recs[0]["iterations"][0]
    assert it["failed"] is True and len(it["rel_hist"]) == 13 and it["refine_steps"] == 12     # all steps: no early stop with the rule off
    assert it["inner"] and "accepted_by" not in it and "accepted_by" not in recs[0]
    assert it["eta_hist"] == [None] * 13 and "eta" not in it                                    # eta is not computed with the rule off
    assert min(it["rel_hist"][-6:]) > 1e-20 and max(it["rel_hist"][-6:]) / min(it["rel_hist"][-6:]) < 100   # the last steps are flat: the floor
    status = json.loads((tmp_path / "r" / "status.json").read_text())
    assert status["state"] == "error"


@opencl
def test_with_the_rule_off_case_b_accepts_a_first_pass_on_its_recurrence_residual_as_before(tmp_path, dev):
    """The recurrence residual of the 64-bit PCG of a small grid falls without a floor, so a target of 1e-30 is reached and the pass is
    accepted with no check of the true residual: the behavior of d19c7fb, kept for `gpu_backward_error: 0`."""
    need_case("B", dev)
    p = settings_case(tmp_path, "B", gpu_tol=UNREACHABLE, gpu_backward_error=0)
    assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet"]) == 0
    recs = lines_of(tmp_path / "r")
    assert all(r["accepted_by"] == "tol" and "eta" not in r and "true_rel" not in r for r in recs)


@opencl
@pytest.mark.parametrize("case", CASES)
def test_a_forced_failure_writes_its_record_before_the_error_and_the_text_gives_facts_and_options(tmp_path, case, dev):
    need_case(case, dev)
    prob = build_problem(load_config(case_yaml(tmp_path, case=case, h=2.0, iterations=2)))
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    m = GpuModel(prob, out, GpuParams(min_free=300, max_iter_first=1, max_iter=1, max_refine=3))               # the rule is on (the default 1e-15)
    out.step(0, "solve")
    with pytest.raises(UserError) as e:
        m.solve(m.stiffness(np.full(m.n_design, 0.5)))
    text = str(e.value)
    recs = lines_of(tmp_path / "run")
    assert recs == m.solver.records and len(recs) == 1                                         # the file has the record
    rec = recs[0]
    assert rec["failed"] and rec["kind"] == "main" and rec["iteration"] == 0 and rec["solve"] == 0 and rec["solves"] == 1
    it = rec["iterations"][0]
    assert it["failed"] and len(it["rel_hist"]) >= 2 and all(isinstance(v, float) for v in it["rel_hist"])
    assert it["inner"] and it["eta"] > GPU_BACKWARD_ERROR_DEFAULT and rec["eta"] == it["eta"]    # not converged: eta is far above rounding
    assert it["eta_hist"][-1] == it["eta"] and "accepted_by" not in it
    # facts: the target, what was reached, the backward error and its limit
    assert "did not reach the relative residual 1e-08" in text and "settings.gpu_tol" in text
    assert f"Reached {it['true_rel']:.2e}" in text and f"backward error {it['eta']:.2e} (limit 1e-15)" in text
    assert f"case {case}" in text and "FP64 rounding floor has a backward error of about 4e-17 to 2e-16" in text
    # options: the levers, as options and not as orders
    for lever in ("a larger settings.gpu_tol", "a larger settings.gpu_backward_error", "a lower settings.penal",
                  "a higher settings.volume_fraction", "set settings.solver to 'auto'"):
        assert lever in text
    assert "gpu_case" not in text if case == "B" else "settings.gpu_case 'B'" in text           # no case B in case B; case C is no cure
    assert "'C'" not in text
    # each lever is an option of its own (G2 R1, n3): the case option stands between the semicolons, not inside the stiffer-design option
    assert "settings.volume_fraction); " in text
    assert ("); settings.gpu_case 'B' (this device has 64-bit float); or set settings.solver" in text) if case == "C" else "); or set settings.solver" in text
    assert ("up to 1 PCG iterations in each of 3 refinement steps" in text) if case == "C" else "then 3 restart steps of up to 1 PCG iterations" in text   # n4
    m.solver.free()
    log = (tmp_path / "run" / "run.log").read_text() if (tmp_path / "run" / "run.log").exists() else ""
    assert "accepted by the backward error" not in log


@opencl
def test_the_option_of_case_b_is_for_case_c_on_a_device_with_64_bit_float_only(tmp_path, dev):
    import dataclasses
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=2)))
    m, off = GpuModel(prob, None, SMALL), GpuModel(prob, None, GpuParams(min_free=300, backward_error=0.0))
    try:
        s = m.solver
        assert s.case == "C" and (not s.dev.fp64 or "settings.gpu_case 'B'" in s._failure_text(1e-5, 1e-9, "x", False))
        s.dev = dataclasses.replace(s.dev, fp64=False)
        assert "gpu_case" not in s._failure_text(1e-5, 1e-9, "x", False)
        assert "the loop stopped early" in s._failure_text(1e-5, 1e-9, "x", True)
        assert "the backward error was not computed (the rule is off)" in off.solver._failure_text(1e-5, None, "x", False)
        assert "a settings.gpu_backward_error above 0 (now 0: the rule is off)" in off.solver._failure_text(1e-5, None, "x", False)
    finally:
        m.solver.free()
        off.solver.free()


@opencl
def test_the_loop_takes_the_settings_and_the_one_off_stress_solve_keeps_the_tight_tolerance(tmp_path, dev, monkeypatch):
    from test_gpu_mg import DirectModel
    from topopt.gpu_mg import model as gm
    from topopt.optimize import reduced_system
    prob = build_problem(load_config(settings_case(tmp_path, "C", gpu_tol=0.5, gpu_backward_error=3e-15)))
    m = GpuModel(prob)
    assert (m._params.tol, m._params.backward_error) == (0.5, 3e-15)                         # the loop: the settings
    assert (GpuParams().tol, GpuParams().backward_error) == (1e-8, GPU_BACKWARD_ERROR_DEFAULT) == (1e-8, 1e-15)    # the recorded defaults
    seen = []
    init = gm.GpuSolver.__init__

    def spy(self, *a, **k):
        seen.append(a[7])
        init(self, *a, **k)
    monkeypatch.setattr(gm.GpuSolver, "__init__", spy)
    edof, aidx, free = reduced_system(prob)
    rho = np.full(m.n_design, 0.5)
    U = gm.solve_design(prob, edof, aidx, free, m.stiffness(rho), m.f_red)
    assert (seen[0].tol, seen[0].backward_error) == (1e-8, 3e-15)                            # tol: GpuParams.tol; the rule: the setting
    ref = DirectModel(prob)
    res = ref.stiffness(rho) @ U - ref.f_red
    assert np.linalg.norm(res) / np.linalg.norm(ref.f_red) < 2e-8                              # the stress step is solved tightly
    m.solver.free()


@opencl
def test_the_rule_off_and_the_old_target_leave_the_record_as_it_was(tmp_path, dev):
    """`gpu_tol: 1e-8` and `gpu_backward_error: 0`: no check of the true residual after the first pass, and no eta."""
    for case in CASES:
        need_case(case, dev)
        prob = build_problem(load_config(settings_case(tmp_path / case, case, gpu_tol=1e-8, gpu_backward_error=0)))
        m = GpuModel(prob, None, GpuParams(min_free=300, tol=1e-8, backward_error=0.0))
        m.solve(m.stiffness(np.full(m.n_design, 0.6)))
        its = m.solver.records[-1]["iterations"][0]
        assert its["accepted_by"] == "tol" and {"eta", "check_s", "failed", "pass1_true_rel"}.isdisjoint(its)
        assert case == "C" or "true_rel" not in its            # case B: the first pass is accepted on the recurrence residual, as at d19c7fb
        m.solver.free()


@opencl
def test_a_first_pass_that_stops_on_the_recurrence_gets_a_check_of_the_true_residual(tmp_path, dev, monkeypatch):
    need_case("B", dev)
    prob = build_problem(load_config(case_yaml(tmp_path, case="B", h=2.0, iterations=2)))
    m = GpuModel(prob, None, GpuParams(min_free=300, tol=1e-6))
    try:
        m.solve(m.stiffness(np.full(m.n_design, 0.6)))
        its = m.solver.records[-1]["iterations"][0]
        assert its["accepted_by"] == "tol" and "refine_steps" not in its and "eta" not in its      # eta is not needed above the target
        assert 0 < its["true_rel"] < 1e-6 and its["check_s"] >= 0 and its["executed"] < 100
        # the check says "not accepted": the solve goes on in the restart loop from the result of the pass
        real = engine.check_solution
        calls = []

        def refuse_once(*a, **k):
            out = real(*a, **k)
            if not calls:
                out = dict(out, ok=False, accepted_by=None)
            calls.append(out)
            return out
        monkeypatch.setattr(engine, "check_solution", refuse_once)
        m.solve(m.stiffness(np.full(m.n_design, 0.6)))
        its = m.solver.records[-1]["iterations"][0]
        assert len(calls) == 1 and its["pass1_true_rel"] == pytest.approx(float(calls[0]["rel"].max())) and "refine_steps" in its
        assert its["accepted_by"] == "tol" and its["true_rel"] < 1e-6
        # the restart starts at the result of the pass (G2 review, m7): its true residual is below the target, so the loop needs 0 steps.
        # A restart from zero would need 3 steps or more (the inner PCG removes about 99 % of the residual in each step).
        assert its["refine_steps"] == 0 and its["inner"] == [] and its["true_rel"] == pytest.approx(its["pass1_true_rel"])
    finally:
        m.solver.free()


@opencl
def test_the_resume_says_nothing_new_about_the_first_solve_that_the_rule_accepted(tmp_path, dev):
    """One `run.log` line for each run: a resume that finds a record of the first part with `accepted_by: backward_error` does not repeat it."""
    prob = build_problem(load_config(case_yaml(tmp_path, case="C", h=2.0, iterations=2)))
    out = RunWriter(tmp_path / "run", quiet=True)
    out.clean()
    params = GpuParams(min_free=300, tol=UNREACHABLE)

    def part(iterations, resume_from=None):
        resume = None if resume_from is None else SimpleNamespace(ck=SimpleNamespace(iteration=resume_from, meta={"complete": False}))
        m = GpuModel(prob, out, params, resume=resume)
        for it in iterations:
            out.step(it, "solve")
            m.solve(m.stiffness(np.full(m.n_design, 0.5)))
        m.solver.free()

    part(range(0, 3))
    part(range(2, 5), resume_from=1)                                  # the checkpoint is at iteration 1; iterations 2 to 4 again
    log = (tmp_path / "run" / "run.log").read_text().splitlines()
    assert len([line for line in log if "backward-error rule accepted a solve" in line]) == 1
    counts = [line for line in log if "solves were accepted by the backward error" in line]
    assert [c.split(" solves")[0] for c in counts] == ["gpu-mg: 3 of 3", "gpu-mg: 5 of 5"]       # each end of a process: the whole run so far


@opencl
def test_a_resume_of_a_gpu_mg_checkpoint_from_before_G2_says_what_the_first_part_ran_at(tmp_path, dev):
    """G2 review, m2. `effective_settings` leaves the two numbers out at their defaults, so the input comparison shows nothing when the
    checkpoint is from before them (the first part ran at 1e-8 and without the rule) or from a topopt with other defaults. The
    checkpoint of a gpu-mg run records the numbers (`meta["gpu_mg"]`); the resume writes one `resume:` fact when they differ."""
    from topopt import checkpoint as C
    p = settings_case(tmp_path, "C", gpu_tol=1e-6)                     # the default written out (`case_yaml` gives the parity tests 1e-8)
    run = tmp_path / "run"
    assert main(["run", str(p), "--out", str(run), "--quiet"]) == 0
    cfg = load_config(p)
    problem = build_problem(cfg)

    def facts(solver="gpu-mg"):
        plan = C.plan_resume(run, cfg, problem, solver, allow_changed=solver != "gpu-mg")      # another solver is a change: it needs the override
        return [f for f in plan.facts if "settings.gpu_tol" in f]

    ck = C.read_checkpoint(run)
    assert ck.meta["gpu_mg"] == {"tol": 1e-6, "backward_error": GPU_BACKWARD_ERROR_DEFAULT, "max_iter": 1000}      # recorded at each save (P2b: the first-pass cap too)
    assert facts() == []                                              # the same numbers: nothing to say
    # a checkpoint from before G2: no record
    meta = {k: v for k, v in ck.meta.items() if k != "gpu_mg"}
    C.write_checkpoint(run, ck.arrays, meta)
    (one,) = facts()
    assert one.startswith("resume: the checkpoint was written before settings.gpu_tol and settings.gpu_backward_error existed")
    assert "first part ran at the fixed target 1e-8 and without the backward-error rule" in one
    assert "this part runs at settings.gpu_tol 1e-06 and settings.gpu_backward_error 1e-15" in one and "--allow-changed-inputs" in one
    assert facts("pardiso") == []                                     # the fact is for the solver gpu-mg
    # the same run folder, resumed by the command: the line is in run.log once
    assert main(["run", str(p), "--out", str(run), "--resume", "--quiet"]) == 0
    log = (run / "run.log").read_text().splitlines()
    assert len([line for line in log if "written before settings.gpu_tol" in line]) == 1
    # a record with other numbers (a default of topopt changed between two versions): the YAML is the same, so the fact names both
    ck = C.read_checkpoint(run)
    assert "gpu_mg" not in ck.meta                                    # the resume of a finished run writes no checkpoint
    C.write_checkpoint(run, ck.arrays, dict(ck.meta, gpu_mg={"tol": 1e-6, "backward_error": 1e-14}))
    (two,) = facts()
    assert two == ("resume: the first part ran at settings.gpu_tol 1e-06 and settings.gpu_backward_error 1e-14; this part runs at "
                   "settings.gpu_tol 1e-06 and settings.gpu_backward_error 1e-15 (the YAML is the same: a default of topopt changed)")
    # G2 R2 (N1): the two checks of `_gpu_mg_fact` that name no other case. `plan_resume` only; no new run.
    # (a) a checkpoint of another solver, resumed with gpu-mg: the change of solver is the line, and no fact says how the first part ran
    ck = C.read_checkpoint(run)
    other = dict(ck.meta, inputs=dict(ck.meta["inputs"], solver="pardiso"))
    other.pop("gpu_mg", None)
    C.write_checkpoint(run, ck.arrays, other)
    plan = C.plan_resume(run, cfg, problem, "gpu-mg", allow_changed=True)
    assert plan.changes == ["the solver that runs: pardiso -> gpu-mg"], plan.changes             # the change of solver is in the changes ...
    assert any("the inputs changed" in f and "pardiso -> gpu-mg" in f for f in plan.facts)       # ... and in its own line
    assert not [f for f in plan.facts if "settings.gpu_tol" in f or "written before" in f], plan.facts
    # (b) the YAML changes gpu_tol, the record has other numbers: the change line says it; no line says "the YAML is the same"
    C.write_checkpoint(run, ck.arrays, dict(ck.meta, gpu_mg={"tol": 1e-6, "backward_error": GPU_BACKWARD_ERROR_DEFAULT}))
    d = yaml.safe_load(p.read_text())
    d["settings"]["gpu_tol"] = 1e-8
    p.write_text(yaml.safe_dump(d))
    cfg8 = load_config(p)
    plan = C.plan_resume(run, cfg8, problem, "gpu-mg", allow_changed=True)
    assert [c for c in plan.changes if c.startswith("settings.gpu_tol")], plan.changes
    assert not [f for f in plan.facts if "a default of topopt changed" in f or "the first part ran at settings.gpu_tol" in f], plan.facts
