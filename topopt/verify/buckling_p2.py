"""The linear buckling check of `topopt verify` on the P2 tet model (B3, CONTRACTS.md sections 16 and 33).

For each load case: the stress of the static solve at the 14 quadrature points of every tet gives the geometric stiffness
`Kg` (`P2Model.geometric_stiffness`), and `topopt.buckling.buckling_modes` gives the lowest load factors with the plain
factor of K that `topopt verify` already holds (no second factor). The record goes to `verify.json` as the key `buckling`,
in the form of section 33. It has the tet value, the voxel value of the run (`result/metrics.json`, key `buckling`, when
the run has it) and their ratio. The two numbers come from two models (P2 tets of the printed STL, hex8 voxels of the
optimizer). The ratio is a fact about the two models, not an error of one of them. There is no pass or fail.

A case without compression, an ARPACK that does not converge, a time budget that is used and a memory estimate above the
limit are facts with options, and `topopt verify` goes on.

Mode facts (K1-C). Each converged mode of a case gets a record in the key `mode_facts`: a local or a global mode, and the region of
the largest motion with its position (`mode_facts`). A part with joints and no fixed node (it floats on its supports) also gets the
rigid-body test (`rigid_body_basis`, `rule_trigger`): when the lowest mode is a motion of the supports, the check solves again without the
six rigid-body motions of the part and writes both results. The ideas are from `mode_diagnostics`, `rigid_body_basis` and the `rigid_basis`
argument of `buckling_modes` in `pipeline/lib/p2_buckling.py` of an earlier pipeline of the same author (no licence header; the task text names
`pipeline/s5_verify/`, the file is in `pipeline/lib/`). The code is new: it uses the K1-A model (`supports_p2`), and the measures of "local"
and "region" are defined here.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import scipy.sparse.linalg as spla

from .. import buckling as bk
from . import memory
from .names import opt, style
from .report import projected_text

KIND = ("linear buckling of the P2 tet model of the printed STL (quadratic tets). It is a model that does not depend on the voxel "
        "model: another mesh and another element. A model estimate, not a verdict.")
MEANING = "load_factors are multipliers on the load case as the YAML gives it (the objective weights are not applied)"
FACTOR = ("the plain factor of the matrix of the static solve (no second factor, and not the refined one). With springs the matrix holds the spring "
          "matrix; with joints it is the condensed stiffness S = K + Kb - C D^-1 C^T")
MODELS = ("tet_over_voxel is the difference of two models: P2 tets of the printed STL against hex8 voxels of the optimizer. "
          "It is not an error or a bias of one of them. The parts of one such difference (the element, the geometry and the "
          "region rules, the load model) were measured on one design (see CONTRACTS.md section 33).")
NOT_USED = ("buckling.density", "buckling.max_gb")
LOOSE_PIECE = {"applicable": False,
               "reason": "A body with no fixed node and no joint or spring facet makes K singular, and verify stops with exit code 1 "
                         "before the solve (CONTRACTS.md section 16). A body that a joint or a spring holds is held, not loose. So the tet "
                         "model has no loose piece that the stiffness of void holds, and the loose-piece fact of the voxel check does not apply."}
THREADS_NOTE = ("The seconds of a solve with pardiso depend on the MKL thread count and on the load of the machine; the load "
                "factors do not.")
THREAD_VARS = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def rerun_text() -> str:
    """The command that repeats the check, in the style of the reader (command line or MCP tool)."""
    return "topopt verify <run_dir>" if style() == "cli" else "verify_run"


def tet_texts() -> bk.CaseTexts:
    """The option texts of a case record for the tet model (the voxel texts name voxels and the refine page)."""
    where = "verify.json (cases, von_mises) and fields.npz"
    return bk.CaseTexts(
        no_positive_option=f"A buckling load needs a compressive stress in the part. Look at the stress of this case in {where}.",
        not_converged_option=f"Ask for fewer modes (buckling.modes in the YAML of the run, then `{{rerun}}` again), or look at "
                             f"the stress of this case in {where}.",
        time_budget_option=f"Set a larger buckling.max_seconds in the YAML of the run and run `{{rerun}}` again, use a faster "
                           f"factor ({opt('solver')} cholesky on macOS, pardiso on Linux x86_64 with the extra 'pardiso', or "
                           f"cholespy with the extra 'verify'), or use a larger {opt('mesh_size')} (fewer dofs, so a faster solve).",
        residual_cause="K can be close to singular, or the mesh has tets of poor quality.",
        residual_option="Look at the quality of the mesh (mesh.quality_min and the tets below 0.05 in verify.json) and at "
                        "linear_solve of this case.",
    )


def voxel_values(info) -> tuple[dict, str | None]:
    """The voxel values of the run: `({case: {...}}, why)`. `why` is None, or the reason why there is no usable value.

    They come from the key `buckling` of `result/metrics.json`. A case entry has `lowest` (the lowest load factor, or None),
    `status`, `density`, `solver` and, when the voxel record has them, the energy centre and box of its lowest mode.
    """
    path = info.run_dir / "result" / "metrics.json"
    try:
        rec = json.loads(path.read_text()).get("buckling")
    except (OSError, ValueError, AttributeError):
        return {}, "result/metrics.json cannot be read"
    if not isinstance(rec, dict):
        return {}, "result/metrics.json has no key buckling (the run did not do the voxel check)"
    if rec.get("status") != "done":
        return {}, f"the voxel check of the run has the status {rec.get('status')!r}" + (
            f" ({rec['reason']})" if rec.get("reason") else "")
    out = {}
    for c in rec.get("cases", []):
        lf = c.get("load_factors") or []
        entry = dict(lowest=float(lf[0]) if lf else None, status=c.get("status"), density=rec.get("density"),
                     solver=rec.get("solver"), source="result/metrics.json, key buckling")
        for key in ("lowest_mode_energy_center_mm", "lowest_mode_energy_box_mm"):
            if key in c:
                entry[key.replace("lowest_mode_", "mode_")] = c[key]
        out[c.get("name")] = entry
    return out, None


def threads_fact() -> dict:
    return {k: os.environ.get(k) for k in THREAD_VARS}


def term_bytes(cfg, model, K, Kff, free) -> int:
    """The extra peak memory of the check for this model, from its real counts (`topopt.verify.memory.buckling_bytes`)."""
    return memory.buckling_bytes(model.N, cfg.buckling.modes, free_dofs=len(free), nnz_k=K.nnz, nnz_kff=Kff.nnz, tets=model.nel,
                                 cases=len(cfg.load_cases))


def _header(cfg, solver: str) -> dict:
    b = cfg.buckling
    return {"kind": KIND, "meaning": MEANING, "solver": solver, "factor": FACTOR, "eigen_solver": bk.EIGEN_SOLVER,
            "modes": int(b.modes), "tol": bk.TOL, "maxiter": bk.MAXITER, "max_seconds": b.max_seconds,
            "stress": "the stress of the static solve at the 14 points of the order-5 rule of every tet (not computed again)",
            "not_used_keys": list(NOT_USED), "loose_piece": LOOSE_PIECE, "models": MODELS}


def skip_record(cfg, solver: str, reason: str, facts: list, options: list, **extra) -> dict:
    """The record of a check that did not start: a fact with its options. `topopt verify` goes on."""
    out = _header(cfg, solver)
    out.update(status="skipped", reason=reason, cases=[], lowest_load_factor=None, facts=facts, options=options, **extra)
    return out


def _error_option(rerun: str) -> str:
    return (f"Read the message, then run `{rerun}` again with more memory ({opt('max_gb')}, {opt('force')}), a larger "
            f"{opt('mesh_size')}, or with the check off (buckling.check: false in the YAML).")


def error_record(cfg, solver: str, message: str) -> dict:
    out = _header(cfg, solver)
    out.update(status="error", reason="exception", message=message, cases=[], lowest_load_factor=None,
               facts=[f"The check stopped: {message}"],
               options=[_error_option(rerun_text())])
    return out


def memory_decision(cfg, solver: str, term: int, after_bytes: int, lim: dict, force: bool, size: float):
    """Is the memory estimate with the check above the limit? Return `(skip record or None, memory facts)`.

    The estimate is the estimate of the run (the factor) plus the term of the check. The limit is the smaller of the
    available RAM and `--max-gb`; `buckling.max_gb` of the YAML is not used. With `--force` the check starts.
    """
    limit = lim["limit_bytes"]
    mem = dict(term_bytes=int(term), estimate_without_check_bytes=int(after_bytes),
               estimate_with_check_bytes=int(after_bytes + term), limit_bytes=limit, limit_source=lim["limit_source"],
               model="memory.buckling_bytes: 12 B x (nnz(K)/3 + 2 nnz(Kff)/3) + 8 B x free dofs x (ncv + 8) + 672 B per tet and load case + 200 MB")
    if limit is None or after_bytes + term <= limit or force:
        return None, mem
    facts = [f"The memory estimate with the check is {memory.gb_text(after_bytes + term)} ({memory.gb_text(after_bytes)} for "
             f"the factor and the stress check, {memory.gb_text(term)} for the check), above the limit {memory.gb_text(limit)} "
             f"({lim['limit_source']}). The check is not started. The stress check of verify goes on."]
    options = [f"A larger {opt('max_gb')}, or {opt('force')} to run anyway (the machine can swap or stop the process).",
               f"A larger {opt('mesh_size')} (fewer dofs, so a smaller term).",
               f"A factor that needs less memory ({opt('solver')} pardiso on Linux x86_64 with the extra 'pardiso', or cholespy "
               "with the extra 'verify') if the factor is SuperLU.",
               "Leave the check off (buckling.check: false in the YAML)."]
    return skip_record(cfg, solver, "memory", facts, options, memory=mem, size_mm=float(size)), mem


# --------------------------------------------------------------------------- #
# Mode facts (K1-C)
# --------------------------------------------------------------------------- #

# The thresholds of the mode facts and of the rigid-body test. The first values are from the author's measurements on two test parts (a global
# column, energy volume share 0.416; a fin on a block, 0.096). They are module constants: no YAML key and no CLI option. Each constant has one
# meaning, and the texts MODE_RULE and RIGID_RULE are built from them (`mode_rule`, `rigid_rule`), so a change of a constant changes the text.
MODE_ENERGY_SHARE = 0.9      # the tets that hold this share of the strain energy of a mode are the tets of its volume share
LOCAL_VOLUME_SHARE = 0.25    # a mode is local when those tets fill at most this share of the part volume
SUPPORT_ENERGY_SHARE = 0.5   # the supports hold at least this share of the energy of a mode (support_energy_share): the rigid-body test runs, and kind is null
DESIGN = "design"            # the name of the part outside every named region


def mode_rule(energy: float = MODE_ENERGY_SHARE, local: float = LOCAL_VOLUME_SHARE, support: float = SUPPORT_ENERGY_SHARE) -> str:
    """The text `mode_rule` of the record, from the three thresholds (the arguments default to the module constants)."""
    return ("A mode is local when the tets that hold {energy:g} % of its strain energy fill at most {local:g} % of the part volume (tets taken by "
            "strain energy density, highest first), else global. Local is relative to the part volume: a slender member that is a small share of "
            "the part gets local for its own mode. When the supports hold {support:g} or more of the energy of the mode (support_energy_share), "
            "kind is null: the tets hold only the rest of the energy, and kind would describe that rest. The region of a mode is the named region "
            "that holds the node of the largest motion (a region of the YAML: fixed, load, joint, spring or measure), else 'design'. The motion is "
            "the displacement of the mode without its rigid-body part when the part floats on its supports. These are facts about the shape of "
            "the mode, not a limit. The three values are first values from the author's measurements on two test parts, and module constants."
            ).format(energy=100 * energy, local=100 * local, support=support)


def rigid_rule(support: float = SUPPORT_ENERGY_SHARE) -> str:
    """The text `rigid_body_rule` of the record, from the threshold of the trigger (the argument defaults to the module constant)."""
    return ("The test runs when the part has joints or springs and no fixed node (it floats on its supports) and the supports hold at least "
            "{support:g} of the strain energy of the lowest mode (support_energy_share >= {support:g}), that is when the lowest mode is mostly a "
            "motion of the part on its supports. Then Kg is projected on the motions of the part without the six rigid-body motions (Kg' = P^T Kg P) "
            "and the eigen problem is solved again. The projector weights all nodes alike, so on a graded mesh the projected value depends a "
            "little on the grading. load_factors stays the result without the projection, and the gate and the ratchet read it; "
            "rigid_body.load_factors is the result with it, a fact next to it. The projection is a test of the part, not a model of the "
            "supports: a part on soft supports can also fail by swinging on them, and the beams have no geometric stiffness. In "
            "rigid_body.mode_facts the rigid_body_fraction can be far from zero: the projected modes are orthogonal to the rigid-body motions with the "
            "stiffness S as weight, and the fraction is measured without weight.").format(support=support)


MODE_RULE = mode_rule()
RIGID_RULE = rigid_rule()


def rigid_body_basis(node_xyz: np.ndarray, scale_mm: float | None = None) -> np.ndarray:
    """The six rigid-body motions of the nodes, `(3 nn, 6)`: three translations and three rotations about the centroid.

    `node_xyz` is `(3, nn)`. A rotation column is divided by `scale_mm` (default: the largest extent of the nodes) so that the six
    columns have a similar size.
    """
    X = np.asarray(node_xyz, dtype=float).T
    if scale_mm is None:
        scale_mm = max(float(np.ptp(X, axis=0).max()), 1e-12)
    r = (X - X.mean(axis=0)) / scale_mm
    nn = X.shape[0]
    R = np.zeros((3 * nn, 6))
    R[0::3, 0] = R[1::3, 1] = R[2::3, 2] = 1.0
    R[0::3, 4], R[0::3, 5] = r[:, 2], -r[:, 1]
    R[1::3, 3], R[1::3, 5] = -r[:, 2], r[:, 0]
    R[2::3, 3], R[2::3, 4] = r[:, 1], -r[:, 0]
    return R


class _Projected(spla.LinearOperator):
    """`sign * P Kg P` for the projector `P = I - R (R^T R)^-1 R^T`, as an operator. `tocsc()` returns the operator itself, so that
    `topopt.buckling.buckling_modes` takes it in place of a sparse matrix."""

    def __init__(self, Kg, R, sign: float = 1.0):
        super().__init__(dtype=np.float64, shape=Kg.shape)
        self.Kg, self.R, self.sign = Kg, R, sign
        self.G = np.linalg.inv(R.T @ R)

    def _p(self, v):
        return v - self.R @ (self.G @ (self.R.T @ v))

    def _matvec(self, v):
        return self.sign * self._p(self.Kg @ self._p(np.asarray(v, dtype=np.float64)))

    def _matmat(self, X):
        return self._matvec(X)

    def _adjoint(self):
        return self

    def __neg__(self):
        return _Projected(self.Kg, self.R, -self.sign)

    def tocsc(self):
        return self

    def toarray(self):
        return self._matvec(np.eye(self.shape[0]))


def _local_volume(e, vol) -> float:
    """The share of the volume that holds `MODE_ENERGY_SHARE` of the energy `e` (per tet), the tets taken by energy density."""
    total = float(e.sum())
    if not total > 0.0:
        return float("nan")
    order = np.argsort(-(e / vol))
    n = int(np.searchsorted(np.cumsum(e[order]) / total, MODE_ENERGY_SHARE)) + 1
    return float(vol[order[:n]].sum() / vol.sum())


def mode_facts(model, phi_full, load_factor: float, E: float, nu: float, region_masks: dict | None = None, basis=None,
               s_norm: float | None = None) -> dict:
    """The facts of one mode: `kind`, the energy volume share, the region of the largest motion and its position.

    `phi_full` is the mode on all `model.N` dofs. `region_masks` is `{name: bool mask over the P2 nodes}`. With `basis` (the rigid-body
    basis, a floating part) the motion is the mode without its rigid part and `rigid_body_fraction` is given. `s_norm` is `phi^T S phi`;
    with it `support_energy_share` (the share of that energy that is not strain energy of the tets) is given.
    """
    e = model.strain_energy(phi_full, E, nu)
    share = _local_volume(e, model.vol)
    ok = bool(np.isfinite(share))
    out = dict(load_factor=float(load_factor), kind=("local" if share <= LOCAL_VOLUME_SHARE else "global") if ok else None,
               energy_volume_share=round(share, 4) if ok else None)
    u = np.asarray(phi_full, dtype=float)
    if basis is not None:
        c, *_ = np.linalg.lstsq(basis, u, rcond=None)
        res = u - basis @ c
        out["rigid_body_fraction"] = round(float(1.0 - (res @ res) / (u @ u)), 4) if u @ u > 0 else None
        u = res
    if s_norm is not None and s_norm > 0:
        out["support_energy_share"] = round(float(max(0.0, 1.0 - 2.0 * float(e.sum()) / s_norm)), 4)
        if out["support_energy_share"] >= SUPPORT_ENERGY_SHARE and out["kind"] is not None:
            out["kind"] = None      # the tets hold the smaller part of the energy: their shape is not the shape of the mode
            out["kind_note"] = (f"kind is null: the supports hold {out['support_energy_share']:.2f} of the energy of this mode "
                                "(support_energy_share), so the tets hold only the rest and kind would describe that rest")
    mag = np.linalg.norm(u.reshape(-1, 3), axis=1)
    peak = int(np.argmax(mag))
    peak_mag = float(mag[peak])
    named = {n: m for n, m in (region_masks or {}).items() if m.any()}
    rms = {n: float(np.sqrt(np.mean(mag[m] ** 2)) / peak_mag) if peak_mag > 0 else 0.0 for n, m in named.items()}
    inside = [n for n, m in named.items() if m[peak]]
    region = max(inside, key=lambda n: rms[n]) if inside else DESIGN
    rms[DESIGN] = float(np.sqrt(np.mean(mag ** 2)) / peak_mag) if peak_mag > 0 else 0.0
    out["largest_motion"] = dict(region=region, position_mm=[round(float(v), 3) for v in model.node_xyz[:, peak]],
                                 region_rms_over_peak={n: round(v, 4) for n, v in rms.items()})
    return out


def rule_trigger(facts: list) -> bool:
    """Does the lowest mode ask for the rigid-body test (`RIGID_RULE`)? `facts` is the list of `mode_facts` of the case, lowest first.

    The trigger decides only whether the projected value is computed as a fact. It does not decide the value that the gate reads.
    """
    if not facts:
        return False
    return (facts[0].get("support_energy_share") or 0.0) >= SUPPORT_ENERGY_SHARE      # the rigid_body_fraction is 0.98 or more on every part that was measured, so it decides nothing


def _add_mode_facts(rec, res, model, Kff, free, E, nu, region_masks, floating, held) -> None:
    """Write `mode_facts` (a list, lowest mode first) into the case record `rec` from the eigenvectors of `res`.

    `held` is True when joints or springs hold the part: `Kff` is then the stiffness with the supports, and `support_energy_share` is given.
    """
    basis = rigid_body_basis(model.node_xyz) if floating else None
    facts = []
    for j, lam in enumerate(rec["load_factors"]):
        phi = np.zeros(model.N)
        phi[free] = res["phi"][:, j]
        s_norm = float(res["phi"][:, j] @ (Kff @ res["phi"][:, j])) if held else None
        facts.append(mode_facts(model, phi, lam, E, nu, region_masks, basis, s_norm))
    rec["mode_facts"] = facts


def _rigid_body_test(rec, model, Kgff, Kff, fac, free, b, name, texts, rerun, E, nu, region_masks) -> None:
    """Solve the case again with the six rigid-body motions of the part removed from `Kg` and write `rigid_body` into `rec`."""
    basis = rigid_body_basis(model.node_xyz)
    res = bk.buckling_modes(_Projected(Kgff, basis[free]), Kff, fac, k=b.modes, max_seconds=b.max_seconds)
    sub = bk.case_summary(res, name, b.modes, rerun=rerun, texts=texts)
    rb = dict(applied=True, rule=RIGID_RULE, status=sub["status"], load_factors=sub["load_factors"], solves=sub["solves"],
              eigen_seconds=sub["eigen_seconds"], residual_max=sub["residual_max"])
    if sub["status"] == bk.OK and sub["load_factors"]:
        _add_mode_facts(sub, res, model, Kff, free, E, nu, region_masks, True, True)
        rb["mode_facts"] = sub["mode_facts"]
        low, low0 = sub["load_factors"][0], rec["load_factors"][0]
        rb["lowest_over_unprojected"] = float(low / low0)
        rec["facts"].append(
            f"The lowest mode of this case is a motion of the part on its supports (rigid_body_fraction "
            f"{rec['mode_facts'][0]['rigid_body_fraction']:.2f}, support_energy_share {rec['mode_facts'][0]['support_energy_share']:.2f}). "
            f"Its load factor is {low0:.6g}. With the six rigid-body motions of the part removed from Kg the lowest load factor is {low:.6g}.")
        rec["options"].append(f"The part itself buckles at {low:.6g}; the lowest mode ({low0:.6g}) is a swing on the supports. The gate and the "
                              "ratchet read the lowest load factor, not the higher one: it would loosen buckling.min_factor, a requirement, "
                              "and only the user can agree to that. Compare the two results (key rigid_body). The beams have no geometric "
                              "stiffness, so neither is a buckling load of the whole assembly.")
    else:
        rec["facts"].append(f"The solve without the rigid-body motions of the part gave the status {sub['status']}.")
    rec["rigid_body"] = rb


def _case(model, cfg, fac, Kff, free, sig, name, texts, rerun, voxel, stiffness_over_optimizer, E, nu, region_masks=None,
          floating=False, held=False) -> dict:
    t0 = time.perf_counter()
    Kg = model.geometric_stiffness(sig)
    Kgff = Kg[free][:, free].tocsr()
    del Kg
    t_kg = time.perf_counter() - t0
    b = cfg.buckling
    res = bk.buckling_modes(Kgff, Kff, fac, k=b.modes, max_seconds=b.max_seconds)
    rec = bk.case_summary(res, name, b.modes, rerun=rerun, texts=texts)
    rec["kg_seconds"] = round(t_kg, 3)
    rec["seconds_per_solve"] = round(float(res["solve_seconds"]) / res["solves"], 4) if res["solves"] else None
    lowest = rec["load_factors"][0] if rec["load_factors"] else None
    if rec["status"] == bk.OK and lowest is not None:
        phi = np.zeros(model.N)
        phi[free] = res["phi"][:, 0]
        rec.update(bk.energy_location(model.strain_energy(phi, E, nu), model.centroids()))
    if rec["status"] == bk.OK and rec["load_factors"]:
        _add_mode_facts(rec, res, model, Kff, free, E, nu, region_masks, floating, held)
        if floating and rule_trigger(rec["mode_facts"]):
            try:
                _rigid_body_test(rec, model, Kgff, Kff, fac, free, b, name, texts, rerun, E, nu, region_masks)
            except Exception as exc:      # the test is an extra: its failure must not take the measured factors of the case away
                msg = " ".join(f"{type(exc).__name__}: {exc}".split())
                rec["rigid_body"] = dict(applied=True, rule=RIGID_RULE, status=bk.ERROR, message=msg, load_factors=[])
                rec["facts"].append(f"The solve without the rigid-body motions of the part stopped: {msg}")
    if stiffness_over_optimizer is not None:
        rec["stiffness_over_optimizer"] = stiffness_over_optimizer
    v = voxel.get(name)
    rec["voxel"] = v
    rec["tet_over_voxel"] = float(lowest / v["lowest"]) if lowest is not None and v and v.get("lowest") else None
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    return rec


def run_check(model, cfg, info, *, fac, Kff, free, sigs: dict, cases: dict, solver: str, E: float, nu: float,
              mem: dict, say=lambda _m: None, supports: dict | None = None, region_masks: dict | None = None,
              held: bool | None = None) -> dict:
    """The check of every load case. `sigs` is `{case name: stress (nel, 14, 6)}`; each array is freed when its `Kg` is built.

    `fac` holds the factor of exactly this `Kff` and nothing else may be factored while this runs.
    With joints (K1-A, `supports_p2`), `Kff` is the condensed stiffness `S = K + Kb - C D^-1 C^T` and `fac.solve` the grid block of the bordered
    solve; `supports` (`SupportSet.buckling_note`) is then written into the record: the geometric stiffness is that of the tets only, the beams have none.
    `held` is True when joints or springs hold the part (default: `supports is not None`, that is joints; a set with springs only has no `supports` note and
    passes `held=True`): the part then floats when it has no fixed node, and the rigid-body test can run. `cases` is the dict of
    the cases of the level (`stiffness_over_optimizer`). Never raises for a failure of one case: that case gets `status: error`.
    """
    t_start = time.perf_counter()
    peak0 = memory.peak_rss_bytes()
    held = supports is not None if held is None else bool(held)
    floating = held and len(free) == model.N and bool(np.array_equal(free, np.arange(model.N)))
    texts, rerun = tet_texts(), rerun_text()
    voxel, why = voxel_values(info)
    out = _header(cfg, solver)
    out.update(status="done", free_dofs=int(len(free)), threads=threads_fact(), threads_note=THREADS_NOTE, memory=mem)
    out["mode_rule"] = MODE_RULE
    if supports is not None:
        out["eigen_solver"] = bk.EIGEN_SOLVER_JOINTS
        out["supports"] = supports
    if held:
        out["rigid_body_rule"] = RIGID_RULE if floating else (
            "The rigid-body test did not run: the part has a fixed node, so it does not float on its supports.")
    recs = []
    for case in cfg.load_cases:
        name = case.name
        try:
            rec = _case(model, cfg, fac, Kff, free, sigs.pop(name), name, texts, rerun, voxel,
                        cases.get(name, {}).get("stiffness_over_optimizer"), E, nu,
                        region_masks, floating, held)
        except Exception as exc:     # a failure of one case is a fact in its record; the other cases go on
            msg = " ".join(f"{type(exc).__name__}: {exc}".split())
            rec = dict(name=name, status=bk.ERROR, load_factors=[], message=msg, facts=[f"The check of this case stopped: {msg}"],
                       options=[_error_option(rerun)],
                       voxel=voxel.get(name), tet_over_voxel=None)
        low = rec["load_factors"][0] if rec["load_factors"] else None
        line = (f"  buckling: case {name}: " + (f"lowest load factor {low:.6g}" if low is not None else rec["status"])
                + projected_text(rec)
                + (f", voxel {rec['voxel']['lowest']:.6g}, tet over voxel {rec['tet_over_voxel']:.3f}"
                   if rec.get("tet_over_voxel") is not None else "")
                + (f", {rec['solves']} solves, {rec['seconds']:.1f} s" if "solves" in rec else ""))
        say(line)
        recs.append(rec)
    out["cases"] = recs
    found = [(r["load_factors"][0], r["name"]) for r in recs if r.get("load_factors")]
    out["lowest_load_factor"] = {"case": min(found)[1], "value": min(found)[0]} if found else None
    facts, options = [], []
    if supports is not None:
        facts.append(supports["fact"])
        options.append(supports["option"])
    if why is not None:
        facts.append(f"There is no voxel value to compare with: {why}.")
        options.append("Run `topopt buckling <run_dir>` for the voxel value of the run, then run verify again.")
    for r in recs:
        facts.extend(f"{r['name']}: {t}" for t in r.get("facts", []))
        options.extend(f"{r['name']}: {t}" for t in r.get("options", []))
    out["facts"], out["options"] = facts, options
    out["seconds"] = round(time.perf_counter() - t_start, 3)
    peak1 = memory.peak_rss_bytes()
    out["peak_rss_before_bytes"], out["peak_rss_bytes"] = int(peak0), int(peak1)
    out["peak_grew_bytes"] = int(max(peak1 - peak0, 0))
    return out
