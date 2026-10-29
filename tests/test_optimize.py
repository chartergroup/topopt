"""Tests for the optimizer: gradients, a small run, and the run folder."""

import json

import numpy as np
import pytest
import trimesh
import yaml

from topopt.config import load_config
from topopt.optimize import Model, beta_at, run
from topopt.problem import build_problem
from topopt.snapshot import RunWriter
from topopt.solver_choice import resolve_solver


def box(path, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def make_case(tmp_path, h=1.0, iterations=30):
    """Cantilever: design x 0..20, fixed plate x -2..0, load box at the tip."""
    box(tmp_path / "design.stl", (0, 0, 0), (20, 10, 10))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(tmp_path / "tip.stl", (18, 4, 0), (20, 6, 3))
    d = {
        "name": "cantilever",
        "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
        "load_cases": [
            {"name": "down", "weight": 3.0, "forces": [{"region": "tip", "force": [0, 0, -2.0]}]},
            {"name": "side", "weight": 1.0, "forces": [{"region": "tip", "force": [1.0, 3.0, 0]}]},
        ],
        "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": iterations,
                     "beta_schedule": [[0, 1], [10, 2], [20, 4]]},
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_beta_schedule():
    s = ((0, 1.0), (30, 2.0), (50, 4.0))
    assert [beta_at(s, i) for i in (0, 29, 30, 49, 50, 80)] == [1.0, 1.0, 2.0, 2.0, 4.0, 4.0]


def test_gradients_match_finite_differences(tmp_path):
    prob = build_problem(load_config(make_case(tmp_path, h=2.0)))
    m = Model(prob)
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 0.7, m.n_design)
    beta = 2.0
    c0, dc, v0, dv = m.evaluate(x, beta)
    eps, tol = 1e-6, 1e-5
    if m.solver_name == "amg":
        # AMG-CG is not exact: a tiny step gives no change in c. Use a bigger step and a looser limit.
        m.solver.rtol = 1e-13
        c0, dc, v0, dv = m.evaluate(x, beta)
        eps, tol = 1e-3, 1e-4
    for e in rng.choice(m.n_design, size=min(12, m.n_design), replace=False):
        xp, xm = x.copy(), x.copy()
        xp[e] += eps
        xm[e] -= eps
        cp, _, vp, _ = m.evaluate(xp, beta)
        cm, _, vm, _ = m.evaluate(xm, beta)
        fd_c = (cp - cm) / (2 * eps)
        fd_v = (vp - vm) / (2 * eps)
        assert abs(fd_c - dc[e]) / max(abs(dc[e]), 1e-12) < tol
        assert abs(fd_v - dv[e]) / max(abs(dv[e]), 1e-12) < tol
    m.solver.free()


def test_gradients_at_the_run_tolerance(tmp_path):
    """The same check with the AMG tolerance of a real run (rtol 1e-5, KB: amg-warm-start-stale-solution).

    The solver is not exact here, so the limit is loose. A stale warm start (0 CG
    iterations after a small change) would give a wrong gradient far above it.
    """
    p = make_case(tmp_path, h=2.0)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "amg"     # AMG runs on every platform
    p.write_text(yaml.safe_dump(d))
    m = Model(build_problem(load_config(p)))
    assert m.solver_name == "amg"
    assert m.solver.rtol == pytest.approx(1e-5)
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 0.7, m.n_design)
    beta, eps = 2.0, 1e-2
    _c, dc, _v, _dv = m.evaluate(x, beta)
    for e in rng.choice(m.n_design, size=min(12, m.n_design), replace=False):
        xp, xm = x.copy(), x.copy()
        xp[e] += eps
        xm[e] -= eps
        cp = m.evaluate(xp, beta)[0]
        cm = m.evaluate(xm, beta)[0]
        assert abs((cp - cm) / (2 * eps) - dc[e]) / max(abs(dc[e]), 1e-12) < 5e-2
    m.solver.free()


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("small")
    cfg = load_config(make_case(tmp, h=1.0, iterations=30))
    prob = build_problem(cfg)
    out_dir = tmp / "run"
    out = RunWriter(out_dir, quiet=True)
    out.clean()
    run(prob, out)
    return prob, out_dir


def test_cantilever_converges(small_run):
    prob, out_dir = small_run
    hist = [json.loads(l) for l in (out_dir / "history.jsonl").read_text().splitlines()]
    assert [h["iteration"] for h in hist] == list(range(31))
    assert hist[-1]["compliance"] < 0.9 * hist[0]["compliance"]
    assert abs(hist[-1]["volume_fraction"] - 0.3) < 0.01 * 0.3 + 0.003  # within 1% (relative)
    assert abs(hist[-1]["volume_fraction"] / 0.3 - 1) < 0.01
    assert 0 <= hist[-1]["grayness"] <= 1
    assert len(hist[0]["compliance_cases"]) == 2


def test_run_folder_matches_contract(small_run):
    prob, out_dir = small_run
    n = int(np.prod(prob.shape))
    assert (out_dir / "labels.bin").stat().st_size == n
    assert (out_dir / "labels.bin").read_bytes() == prob.labels.tobytes()
    snaps = sorted((out_dir / "snapshots").iterdir())
    assert [s.name for s in snaps] == [f"it_{i:04d}.bin" for i in range(31)]
    assert all(s.stat().st_size == n for s in snaps)
    assert not list(out_dir.rglob("*.tmp"))
    last = np.frombuffer(snaps[-1].read_bytes(), np.uint8)
    flat = prob.labels.reshape(-1)
    assert np.all(last[flat == 2] == 255) and np.all(last[flat == 3] == 255)
    assert np.all(last[(flat == 0) | (flat == 4)] == 0)
    run_json = json.loads((out_dir / "run.json").read_text())
    assert run_json["solver"] == resolve_solver("auto")   # the solver that really ran
    for key in ("format", "name", "yaml_path", "started_at", "shape", "origin", "h", "n_active",
                "n_free_dofs", "iterations", "volume_fraction", "load_cases", "regions",
                "solver", "mass_design_space_g"):
        assert key in run_json
    assert run_json["shape"] == list(prob.shape)
    f0 = run_json["load_cases"][0]["forces"][0]
    assert set(f0) == {"region", "force", "center", "kind"} and len(f0["center"]) == 3 and f0["kind"] == "force"
    assert abs(f0["center"][0] - 19.0) < 1e-6   # tip box x 18..20, voxel centres 18.5, 19.5
    assert run_json["n_active"] == int(((prob.labels >= 1) & (prob.labels <= 3)).sum())
    status = json.loads((out_dir / "status.json").read_text())
    assert set(status) == {"state", "iteration", "step", "elapsed_s", "message"}
    assert status["step"] == "snapshot" and status["iteration"] == 30
    hist0 = json.loads((out_dir / "history.jsonl").read_text().splitlines()[0])
    assert set(hist0) == {"iteration", "elapsed_s", "compliance", "compliance_cases",
                          "volume_fraction", "beta", "change", "grayness", "t_iter_s",
                          "objective", "constraints"}   # the last two: additive, A1
    log = (out_dir / "run.log").read_text()
    assert "it  30" in log


def test_step_markers_in_source():
    import re
    from pathlib import Path
    import topopt.optimize as mod
    text = Path(mod.__file__).read_text()
    steps = re.findall(r"# \[step:(\w+)\]", text)
    assert steps == ["setup", "assemble", "solve", "sensitivity", "filter", "update", "snapshot"]
    assert "# [end]" in text
