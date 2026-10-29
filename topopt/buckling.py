"""Linear buckling: the eigen solve that does not depend on the element (CONTRACTS.md section 33).

This module knows only the matrices `Kg` and `K` and a factor of `K` with a `.solve` method. It does not import
the finite element code, so a check with other elements (the tet P2 model of `topopt verify`) can use it.
The hex8 voxel part is in `topopt.buckling_voxel`.

The problem is `K phi = lambda (-Kg) phi`. `Kg` is the geometric stiffness from the stress of one load case,
so `lambda` is the factor on that load case at which the model buckles (linear buckling). The code solves
the inverse form `(-Kg) phi = mu K phi`, `mu = 1 / lambda`, with ARPACK in regular inverse mode:
`eigsh(-Kg, M=K, Minv=<factor of K>.solve, which="LA")`. This is the call of the pipeline
(`pp_core_final.py`, `p2_buckling.py`). It needs no shift and no second factor. A `mu` of zero or below
is no buckling load: its factor is `inf`.

The result is a voxel estimate or a model estimate. It is not a verdict (KB: cheap-check-as-verdict).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

DEFAULT_MODES = 6
TOL = 1e-8                  # ARPACK tolerance on mu (the pipeline default)
MAXITER = 100               # ARPACK restarts. The row on the hook at 2 mm needs 50 to 140 solves for one evaluation with the cold start (median 64).
NCV_MIN = 24                # Lanczos vectors: max(2 k + 1, 24), as in the P2 check of the pipeline
DEFAULT_MAX_SECONDS = 120.0  # time budget of one eigen solve (an option: buckling.max_seconds)
NEAR_REPEATED_REL = 1e-2    # two factors closer than this (relative) are "near-repeated"
RESIDUAL_FACT = 1e-6        # a relative residual above this is a fact (100 x TOL; normal values are 1e-8 to 1e-10)
RERUN = "topopt buckling <run_dir>"   # the command that repeats the check of a run (the voxel check)
EIGEN_SOLVER = "arpack regular inverse mode (scipy eigsh, which=LA, M=K, Minv=K factor)"
EIGEN_SOLVER_JOINTS = ("arpack regular inverse mode (scipy eigsh, which=LA, M=S, Minv=grid block of the bordered solve; "
                       "S = K + Kb - C D^-1 C^T, joints)")
ENERGY_BOX_SHARE = 0.9      # the box of `energy_location` holds this share of the strain energy of the mode

# Status of one load case.
OK, NO_POSITIVE, NOT_CONVERGED, TIME_BUDGET, ERROR = (
    "ok", "no_positive_factor", "not_converged", "time_budget", "error")


class BudgetExceeded(Exception):
    """The eigen solve used its time budget."""


def buckling_modes(Kg, K, fac, k: int = DEFAULT_MODES, tol: float = TOL, ncv: int | None = None,
                   v0=None, maxiter: int = MAXITER, max_seconds: float | None = None) -> dict:
    """The `k` lowest buckling load factors of `K phi = lambda (-Kg) phi`.

    `K` is symmetric positive definite (sparse). `Kg` is symmetric (sparse), of the same size, with the
    sign of the geometric stiffness (compression gives a negative `Kg`). `fac` is an object with
    `fac.solve(b)` that solves `K x = b` for one vector. It must hold the factor of exactly this `K`.
    For a run with joints (CONTRACTS.md section 38) `K` is the condensed stiffness `S` (`joints.CondensedStiffness`, a
    `LinearOperator` with `@` and `toarray()`), and `fac.solve` is the grid block of the bordered solve, which is `S^-1`.
    `max_seconds` is a time budget for the iteration: ARPACK stops when it is over.

    Returns a dict:

    - `mu`, `lam`: arrays, sorted by `lam` (lowest first). `lam` is `inf` where `mu <= 0`.
    - `phi`: (n, m) K-normalised eigenvectors. `residual`: relative residual of each pair.
    - `status`: `ok`, `not_converged` or `time_budget`. `converged`: how many pairs converged.
      With a status other than `ok`, `lam` holds only the pairs that converged (maybe none).
      They are not always the lowest ones.
    - `seconds`, `solves`, `solve_seconds`, `k`, `ncv`, `tol`, `maxiter`, `max_seconds`, `method`.
    """
    n = K.shape[0]
    k = int(max(1, min(int(k), n - 1)))
    ncv = int(max(2 * k + 1, NCV_MIN) if ncv is None else ncv)
    t0 = time.perf_counter()
    info = {"k": k, "ncv": min(ncv, n), "tol": tol, "maxiter": maxiter, "max_seconds": max_seconds}
    if ncv >= n:       # too small for ARPACK: the dense solver gives all pairs
        mu, phi = _dense_pairs(Kg, K)
        info.update(method="dense", status=OK, solves=0, solve_seconds=0.0, ncv=n)
        return _finish(Kg, K, mu, phi, k, t0, info, converged=True)

    state = {"solves": 0, "t_solve": 0.0}

    def matvec(v):
        if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
            raise BudgetExceeded
        s0 = time.perf_counter()
        x = np.asarray(fac.solve(np.asarray(v, dtype=float).ravel()), dtype=float).ravel()
        state["t_solve"] += time.perf_counter() - s0
        state["solves"] += 1
        return x

    Minv = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    if v0 is None:
        v0 = np.random.default_rng(0).standard_normal(n)   # fixed seed: the same result each run
    negKg = (-Kg).tocsc()
    info["method"] = "arpack"
    try:
        mu, phi = spla.eigsh(negKg, k=k, M=K, Minv=Minv, which="LA", tol=tol, maxiter=maxiter,
                             ncv=ncv, v0=v0)
        status, converged = OK, True
    except spla.ArpackNoConvergence as e:
        mu, phi = e.eigenvalues, e.eigenvectors
        status, converged = NOT_CONVERGED, False
    except BudgetExceeded:
        mu, phi = np.zeros(0), np.zeros((n, 0))
        status, converged = TIME_BUDGET, False
    info.update(status=status, solves=state["solves"], solve_seconds=state["t_solve"])
    return _finish(Kg, K, mu, phi, k, t0, info, converged)


def _dense_pairs(Kg, K):
    import scipy.linalg as sla
    mu, phi = sla.eigh(-Kg.toarray(), K.toarray())
    return mu, phi


def _finish(Kg, K, mu, phi, k, t0, info, converged):
    mu = np.asarray(mu, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).reshape(K.shape[0], mu.size)
    order = np.argsort(-mu)[:k]
    mu, phi = mu[order], phi[:, order]
    lam = np.full(mu.shape, np.inf)
    pos = mu > 0.0
    lam[pos] = 1.0 / mu[pos]
    resid = np.zeros(mu.shape)
    for j in range(mu.size):   # |(-Kg) phi - mu K phi| over |Kg phi| + |mu| |K phi|
        kp = K @ phi[:, j]
        gp = Kg @ phi[:, j]
        den = np.linalg.norm(gp) + abs(mu[j]) * np.linalg.norm(kp)
        resid[j] = np.linalg.norm(-gp - mu[j] * kp) / den if den > 0 else 0.0
    # Lowest factor first. A pair with `mu <= 0` (factor inf) goes last.
    order = np.argsort(lam, kind="stable")
    out = dict(info)
    out.update(mu=mu[order], lam=lam[order], phi=phi[:, order], residual=resid[order],
               converged=int(mu.size), seconds=time.perf_counter() - t0)
    return out


def near_repeated(lam, rel: float = NEAR_REPEATED_REL):
    """Look for near-repeated factors in the sorted positive factors `lam`.

    Return `(flag, min_rel_gap, pairs)`. A gap is `(lam[i+1] - lam[i]) / lam[i]`. `flag` is True when a gap
    is below `rel`. `pairs` lists the 1-based mode numbers `[i, i+1]` of those gaps. `min_rel_gap` is None
    with fewer than two factors. A repeated factor has no single mode shape, and its derivative is not defined.
    """
    lam = np.asarray(lam, dtype=float)
    lam = lam[np.isfinite(lam)]
    if lam.size < 2:
        return False, None, []
    gaps = np.diff(lam) / lam[:-1]
    pairs = [[int(i) + 1, int(i) + 2] for i in np.flatnonzero(gaps < rel)]
    return bool(pairs), float(gaps.min()), pairs


@dataclass(frozen=True)
class CaseTexts:
    """The parts of the facts and options of a case record that name the model (voxels or tets).

    Every text can use `{rerun}`, the command that repeats the check. `residual_cause` is the end of the residual fact.
    """
    no_positive_option: str
    not_converged_option: str
    time_budget_option: str
    residual_cause: str
    residual_option: str


VOXEL_TEXTS = CaseTexts(
    no_positive_option="Look at the stress of this case in the refine page (topopt refine).",
    not_converged_option="Ask for fewer modes (`{rerun} --modes N`), or look at the stress of this case (topopt refine).",
    time_budget_option="Run `{rerun} --max-seconds S` with a larger S, or use a larger voxel_size.",
    residual_cause="K can be close to singular: a part that is held only by the stiffness of void voxels (E_min) does this.",
    residual_option="Look for a loose piece or a thin neck in the part (topopt refine, result/final.stl), "
                    "and compare with the other density (`{rerun} --density gray` or `printed`).",
)


def case_summary(res: dict, name: str, modes: int, rerun: str = RERUN, texts: CaseTexts = VOXEL_TEXTS) -> dict:
    """The record of one load case for metrics.json (section 33) from a `buckling_modes` result.

    `load_factors` are the positive factors that converged, lowest first. A case with no positive factor,
    or with an ARPACK that does not converge, is a fact (`status`, `facts`, `options`), not an error.
    `rerun` is the command that the options name for a new run of the check. `texts` holds the option texts that
    name the model: the default is the voxel check, and the P2 tet check of `topopt verify` has its own.
    """
    lam = res["lam"][np.isfinite(res["lam"])]
    flag, gap, pairs = near_repeated(lam)
    status = res["status"]
    facts, options = [], []
    partial = []
    if status == OK and lam.size == 0:
        status = NO_POSITIVE
        facts.append("Every eigenvalue mu is zero or below: this load case has no compression that can "
                     "buckle the model (for example a pure tension state).")
        options.append(texts.no_positive_option.format(rerun=rerun))
    elif status == NOT_CONVERGED:
        facts.append(f"ARPACK did not converge: {res['converged']} of {res['k']} pairs in {res['solves']} solves "
                     f"({res['seconds']:.1f} s, maxiter {res['maxiter']}). The case may have no compression, or "
                     "the lowest modes are not separated.")
        options.append(texts.not_converged_option.format(rerun=rerun))
        partial = [float(v) for v in lam]
        lam = lam[:0]
    elif status == TIME_BUDGET:
        facts.append(f"The eigen solve stopped at the time budget of {res['max_seconds']:g} s after "
                     f"{res['solves']} solves.")
        options.append(texts.time_budget_option.format(rerun=rerun))
    elif status == OK and lam.size < res["k"]:
        facts.append(f"{res['k'] - lam.size} of the {res['k']} pairs have mu <= 0 (no buckling load).")
    if status in (NOT_CONVERGED, TIME_BUDGET):
        flag, gap, pairs = False, None, []
    resid = float(res["residual"].max()) if res["residual"].size else None
    if resid is not None and resid > RESIDUAL_FACT:
        facts.append(f"The eigen solve did not reach its tolerance: the relative residual of the pairs is up to "
                     f"{resid:.1e} (above {RESIDUAL_FACT:.0e}; a normal case has 1e-8 to 1e-10), so the factors "
                     f"of this case are less exact than usual. {texts.residual_cause}")
        options.append(texts.residual_option.format(rerun=rerun))
    out = {
        "name": name, "status": status,
        "load_factors": [float(v) for v in lam],
        "near_repeated": flag, "near_repeated_rel": NEAR_REPEATED_REL, "min_rel_gap": gap,
        "near_repeated_modes": pairs,
        "modes_requested": int(modes), "modes_converged": int(res["converged"]),
        "residual_max": resid,
        "solves": int(res["solves"]), "eigen_seconds": round(float(res["seconds"]), 3),
        "facts": facts, "options": options,
    }
    if partial:
        out["partial_load_factors"] = partial
    return out


def energy_location(e, centers, share: float = ENERGY_BOX_SHARE) -> dict:
    """Where the strain energy of a mode is: the energy-weighted centre, and the box of the elements that hold `share` of it.

    `e` (n,) is the strain energy of the mode in each element and `centers` (n, 3) the element centres in mm. Return
    `lowest_mode_energy_center_mm` and `lowest_mode_energy_box_mm` (`[[x0, y0, z0], [x1, y1, z1]]`, element centres),
    both rounded to 0.001 mm; an empty dict if the total energy is not positive.
    """
    e = np.asarray(e, dtype=float)
    centers = np.asarray(centers, dtype=float)
    total = float(e.sum())
    if not total > 0.0:
        return {}
    order = np.argsort(-e)
    n = int(np.searchsorted(np.cumsum(e[order]) / total, share)) + 1
    box = centers[order[:n]]
    return {"lowest_mode_energy_center_mm": [round(float(v), 3) for v in (e[:, None] * centers).sum(0) / total],
            "lowest_mode_energy_box_mm": [[round(float(v), 3) for v in box.min(0)],
                                          [round(float(v), 3) for v in box.max(0)]]}


def write_metrics_key(run_dir, result: dict) -> Path:
    """Write `result` as the key `buckling` of `result/metrics.json`, the last key. Return the file.

    If `metrics.json` does not exist, write `result/buckling.json` instead. A file is written to
    `<name>.tmp` first, then moved (CONTRACTS.md section 5).
    """
    result_dir = Path(run_dir) / "result"
    result_dir.mkdir(exist_ok=True)
    path = result_dir / "metrics.json"
    if path.is_file():
        metrics = json.loads(path.read_text())
        metrics.pop("buckling", None)      # a new check replaces the old one and goes last
        metrics["buckling"] = result
        data = json.dumps(metrics, indent=1)
    else:
        path = result_dir / "buckling.json"
        data = json.dumps(result, indent=1)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)
    return path


def sparse_symmetry_error(A) -> float:
    """Relative asymmetry `max|A - A^T| / max|A|` of a sparse matrix (0 if A is empty)."""
    top = abs(A).max() if A.nnz else 0.0
    return float(abs(A - A.T).max() / top) if top > 0 else 0.0


__all__ = ["buckling_modes", "case_summary", "near_repeated", "write_metrics_key", "sparse_symmetry_error",
           "energy_location", "CaseTexts", "VOXEL_TEXTS", "ENERGY_BOX_SHARE",
           "BudgetExceeded", "DEFAULT_MODES", "DEFAULT_MAX_SECONDS", "EIGEN_SOLVER", "EIGEN_SOLVER_JOINTS", "NEAR_REPEATED_REL",
           "RESIDUAL_FACT", "RERUN",
           "OK", "NO_POSITIVE", "NOT_CONVERGED", "TIME_BUDGET", "ERROR"]
