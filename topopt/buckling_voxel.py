"""Linear buckling check of the final voxel design (CONTRACTS.md section 33).

Opt-in: the YAML block `buckling` (key `check: true`) or `topopt run --buckling`. It runs after the
post-process. It never changes the optimizer, the snapshots or `final.stl`. It adds the key `buckling` to
`result/metrics.json`.

What it does, for each load case of the run:

1. K of the final design, and one factor of K (a direct factor: Apple Cholesky or PARDISO).
2. The static solution `u` of the load case, and the stress at the centre of each hex8 voxel,
   `sigma = D0 B_c u` (full modulus, as `topopt.refine.stress`).
3. The geometric stiffness `Kg = sum_e rho_e^p Kg_e(sigma_e)`. Weight `rho^p` (no E_min): a void voxel adds no
   geometric stiffness (the pipeline rule `E_G = rho^p E0`).
4. The lowest load factors from `topopt.buckling.buckling_modes` (the element-agnostic part).

The load factor is the multiplier on the load case as the YAML gives it. The objective weights do not apply.
It is a voxel estimate, not a verdict: full integration hex8 voxels are too stiff in bending (a column of
6 voxels across reads 1.8 percent high, one of 3 voxels reads 7 percent high).

Density: `printed` (default) is the design at the print threshold 0.5. `gray` is the optimizer density. A gray
member with a density of 0.2 to 0.5 can give a false low mode that the print does not have.
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field

import numpy as np

from . import buckling as bk

THRESHOLD = 0.5        # the print threshold (the post-process makes the mesh at 0.5)
GRAY_LOW, GRAY_HIGH = 0.05, 0.95   # a design voxel between these is counted as gray
DIRECT_SOLVERS = ("cholesky", "pardiso")   # the solvers with a factor of K that has `.solve`
# A share of the lowest mode at or above this level is a fact (CONTRACTS.md section 33). The level sits in the
# gap of the measured cases: real modes have 0.0005 (hook), 0.0026 (a design box that is 98 percent void) and 0.09
# (a gray neck); false modes (a loose piece, an edge-only strut, a thin gray member) have 0.94 to 1.0.
FALSE_MODE_SHARE = 0.5
ENERGY_BOX_SHARE = bk.ENERGY_BOX_SHARE     # the box of `lowest_mode_energy_box_mm` holds this share of the strain energy
NNZ_PER_DOF = 81       # nonzeros of a row of K: 27 nodes x 3 dofs (an inner node; the hook has 75)
KIND = ("linear buckling of the voxel model (hex8 elements). A voxel estimate, not a verdict: "
        "the elements are too stiff in bending, and the result is for this density only.")
MEANING = ("load_factors are multipliers on the load case as the YAML gives it "
           "(the objective weights are not applied)")


# --------------------------------------------------------------------------- Kg of hex8 voxels

@functools.lru_cache(maxsize=1)
def _unit_blocks() -> np.ndarray:
    """The six constant matrices `G_c` (6, 24, 24) of a unit cube: `Kg_e = h * sum_c sigma_c G_c`.

    `Kg_e = int grad(N_a)^T sigma grad(N_b) dV` for a constant stress `sigma` (the stress at the centre),
    placed on the three directions (`kron(., I3)`). The integral of the gradient products is exact with the
    2x2x2 Gauss rule of `fe.hex8_ke`. `c` runs over the Voigt order sx, sy, sz, txy, tyz, txz.
    """
    from .fe import LOCAL_NODES
    sgn = 2.0 * LOCAL_NODES - 1.0
    g = 1.0 / np.sqrt(3.0)
    M = np.zeros((3, 3, 8, 8))
    for gx in (-g, g):
        for gy in (-g, g):
            for gz in (-g, g):
                dn = np.empty((8, 3))
                for a in range(8):
                    sx, sy, sz = sgn[a]
                    dn[a, 0] = 0.125 * sx * (1 + sy * gy) * (1 + sz * gz)
                    dn[a, 1] = 0.125 * sy * (1 + sx * gx) * (1 + sz * gz)
                    dn[a, 2] = 0.125 * sz * (1 + sx * gx) * (1 + sy * gy)
                dn *= 2.0                           # chain rule with h = 1
                M += np.einsum("ai,bj->ijab", dn, dn) * 0.125   # detj = (h / 2)^3
    comps = ((0, 0, False), (1, 1, False), (2, 2, False), (0, 1, True), (1, 2, True), (0, 2, True))
    out = np.empty((6, 24, 24))
    for c, (i, j, shear) in enumerate(comps):
        m8 = M[i, j] + (M[j, i] if shear else 0.0)
        out[c] = np.kron(m8, np.eye(3))
    return out


def hex8_kg_blocks(h: float) -> np.ndarray:
    """The six matrices `G_c` (6, 24, 24) of a cube of edge `h` (see `_unit_blocks`). They scale with `h`."""
    return _unit_blocks() * float(h)


def element_stress(u_all, edof, E: float, nu: float, h: float) -> np.ndarray:
    """Stress rows (n_el, 6) at the voxel centres: `D0 B_c u`, sx, sy, sz, txy, tyz, txz (MPa).

    `u_all` holds the displacement of all grid dofs. The modulus is the full `E` (no `rho^p`).
    The helpers are the ones of the refine step. The follow-up of PR #22 moves them to `fe.py`.
    """
    from .refine.stress import isotropic_D, strain_matrix_centre
    Bc = strain_matrix_centre(h)
    D0 = isotropic_D(E, nu)
    return (np.asarray(u_all)[edof] @ Bc.T) @ D0.T


def hex8_kg(asm, edof, u_all, h: float, E: float, nu: float, weight):
    """Geometric stiffness `Kg` (csc, the free dofs of `asm`) of one displacement field.

    `asm` is the `fe.Assembler` of the model. `Kg` is linear in the six stress components, so six calls of
    `asm.K` with the matrices `G_c` instead of `ke` build it (the pattern of K is reused; no new kernel).
    `weight` (n_el,) is `rho^p`: it scales the stress of each voxel. A voxel with weight 0 adds nothing.
    """
    sig = element_stress(u_all, edof, E, nu, h)
    G = hex8_kg_blocks(h)
    weight = np.asarray(weight, dtype=float)
    ke_saved = asm.ke
    Kg = None
    try:
        for c in range(6):
            asm.ke = np.ascontiguousarray(G[c])
            part = asm.K(sig[:, c] * weight)
            if Kg is None:
                Kg = part
            else:
                Kg.data += part.data     # the six parts share one pattern
    finally:
        asm.ke = ke_saved
    return Kg


# --------------------------------------------------------------------------- the model

@dataclass
class VoxelModel:
    """The hex8 model of one density. `rho` is the density used (solid voxels 1).

    `Ee` is the stiffness of each voxel (with E_min). `weight` is `rho^p` for `Kg` (no E_min).
    `rho_optimizer` is the optimizer density (None when the model is built by hand). `loose` (n_el, bool) marks
    the solid voxels that are not face-connected to a fixed region through solid voxels (None: not known).
    `centers` (n_el, 3) are the voxel centres in mm (None: not known). `density` is the density rule.
    """
    edof: np.ndarray
    free: np.ndarray
    Ee: np.ndarray
    weight: np.ndarray
    rho: np.ndarray
    h: float
    E: float
    nu: float
    n_dof_all: int
    rho_optimizer: np.ndarray | None = None
    loose: np.ndarray | None = None
    centers: np.ndarray | None = None
    density: str = "printed"
    _asm: object = field(default=None, repr=False)

    @property
    def asm(self):
        if self._asm is None:
            from .fe import Assembler, hex8_ke
            self._asm = Assembler(self.edof, hex8_ke(1.0, self.nu, self.h), self.free)
        return self._asm


def loose_mask(solid, anchor) -> np.ndarray:
    """The solid voxels of the grid `solid` (bool) that are not face-connected, through solid voxels, to a voxel of
    `anchor` (bool, the fixed region). A piece that touches the rest on an edge or a corner only is loose too:
    the voxel model joins such voxels at the shared nodes, but no stress goes through them like through a face."""
    from scipy import ndimage
    solid = np.asarray(solid, dtype=bool)
    labels, _n = ndimage.label(solid)          # the default structure of `label` has 6 neighbours (faces)
    keep = np.unique(labels[solid & np.asarray(anchor, dtype=bool)])
    return solid & ~np.isin(labels, keep[keep > 0])


def build_model(problem, density_u8, density: str = "printed") -> VoxelModel:
    """The model of the final design: the last snapshot `density_u8` (0..255), and the density rule.

    `printed`: design voxels at or above 0.5 are solid, the others are void (the printed part).
    `gray`: the optimizer density as it is. Fixed and load voxels are solid in both.
    The stiffness has `E_min` for void, as the optimizer has (`optimize.E_MIN`).
    """
    if density not in ("printed", "gray"):
        raise ValueError("density must be printed or gray")
    from .filter import simp_E
    from .optimize import E_MIN, reduced_system
    from .springs import support_mask
    from .voxelize import DESIGN
    cfg = problem.config
    st, mat = cfg.settings, cfg.material
    snap = np.asarray(density_u8, dtype=np.uint8).reshape(-1)
    if snap.size != int(np.prod(problem.shape)):
        raise ValueError("the snapshot does not match the grid shape")
    edof, active_index, free = reduced_system(problem)
    is_design = problem.labels.reshape(-1)[active_index] == DESIGN
    rho_opt = np.ones(active_index.size)
    rho_opt[is_design] = snap[active_index[is_design]].astype(np.float64) / 255.0
    rho = rho_opt.copy()
    if density == "printed":
        rho[is_design] = (rho_opt[is_design] >= THRESHOLD).astype(np.float64)
    Ee = np.full(active_index.size, mat.E)
    Ee[is_design] = mat.E * simp_E(rho[is_design], st.penal, 1.0, E_MIN)
    # Loose pieces are a property of the printed geometry (threshold 0.5 of the optimizer density), for both densities.
    solid = np.zeros(int(np.prod(problem.shape)), bool)
    solid[active_index] = rho_opt >= THRESHOLD
    solid = solid.reshape(problem.shape)
    loose = loose_mask(solid, support_mask(problem)).reshape(-1)[active_index]    # FIXED, and a spring region (section 35)
    ijk = np.stack(np.unravel_index(active_index, problem.shape), axis=1)
    centers = np.asarray(problem.origin, dtype=float) + float(problem.h) * (ijk + 0.5)
    return VoxelModel(edof=edof, free=free, Ee=Ee, weight=rho ** st.penal, rho=rho, h=float(problem.h),
                      E=float(mat.E), nu=float(mat.nu),
                      n_dof_all=3 * int(np.prod(np.array(problem.shape) + 1)), rho_optimizer=rho_opt,
                      loose=loose, centers=centers, density=density)


def estimate_bytes(grid_voxels: int, free_dofs: int, modes: int = bk.DEFAULT_MODES) -> int:
    """Rough peak memory of the check in bytes: the model of the run (`cli.estimate_memory_bytes`:
    grid arrays and the direct factor), plus K and Kg (12 bytes per nonzero, `NNZ_PER_DOF` of them for each
    dof), plus the Lanczos vectors. The hook (115,836 dofs) measured 1.92 GB against this estimate 2.08 GB."""
    from .cli import estimate_memory_bytes
    n = int(free_dofs)
    mats = 2 * NNZ_PER_DOF * 12 * n
    vecs = 8 * n * (max(2 * int(modes) + 1, bk.NCV_MIN) + 8)
    return int(estimate_memory_bytes(int(grid_voxels), n) + mats + vecs)


def voxel_energy(model: VoxelModel, phi) -> np.ndarray:
    """The strain energy of the mode `phi` (free dofs) in each voxel (n_el)."""
    from .fe import hex8_ke
    ph = np.zeros(model.n_dof_all)
    ph[model.free] = phi
    pe = ph[model.edof]
    ke = hex8_ke(1.0, model.nu, model.h)
    return model.Ee * np.einsum("ij,ij->i", pe @ ke, pe)


def strain_energy_share(model: VoxelModel, phi, mask) -> float:
    """Share of the strain energy of the mode `phi` (free dofs) in the voxels where `mask` is true."""
    e = voxel_energy(model, phi)
    total = float(e.sum())
    return float(e[mask].sum() / total) if total > 0 else 0.0


def mode_location(model: VoxelModel, phi) -> dict:
    """Where the mode `phi` lives, as the fields of a case record (all of them facts, no verdict):

    - `lowest_mode_energy_share_rho_lt_half`: the share of the strain energy in voxels with a density below 0.5
      (void voxels, or gray voxels of the gray density).
    - `lowest_mode_motion_share_loose`: of the squared nodal motion of the solid voxels (density 0.5 or more), the
      share that is in the loose solid voxels (`VoxelModel.loose`). Only when the model knows them.
    - `lowest_mode_energy_center_mm` and `lowest_mode_energy_box_mm`: the energy-weighted centre, and the box
      of the voxel centres that hold `ENERGY_BOX_SHARE` of the energy. Only when the model has the centres.
    """
    e = voxel_energy(model, phi)
    total = float(e.sum())
    if not total > 0.0:
        return {}
    out = {"lowest_mode_energy_share_rho_lt_half": round(float(e[model.rho < THRESHOLD].sum() / total), 6)}
    if model.loose is not None:
        ph = np.zeros(model.n_dof_all)
        ph[model.free] = phi
        move = (ph[model.edof] ** 2).sum(axis=1)
        solid = model.rho >= THRESHOLD       # void voxels share the nodes of the solid ones: leave them out
        out["lowest_mode_motion_share_loose"] = round(float(move[model.loose].sum() / max(move[solid].sum(), 1e-300)), 6)
    if model.centers is not None:
        out.update(bk.energy_location(e, model.centers, ENERGY_BOX_SHARE))
    return out


def false_mode_facts(model: VoxelModel, loc: dict, partial: bool = False) -> tuple[list, list]:
    """Facts and options when the lowest mode lives in voxels that are not a solid member (CONTRACTS.md section 33).

    The level is `FALSE_MODE_SHARE`. The result is a signal for the reader and not a verdict: the factor stays in
    the record. `partial` is true when the mode is the lowest converged one of a case that did not converge.
    """
    share = loc.get("lowest_mode_energy_share_rho_lt_half", 0.0)
    move = loc.get("lowest_mode_motion_share_loose") or 0.0
    if share < FALSE_MODE_SHARE and move < FALSE_MODE_SHARE:
        return [], []
    mode = "lowest converged mode" if partial else "lowest mode"
    facts = []
    if share >= FALSE_MODE_SHARE:
        facts.append(f"{100 * share:.2f} percent of the strain energy of the {mode} is in voxels with a density "
                     "below 0.5 (void or gray voxels). A mode that is held by such voxels can be a loose piece, "
                     "a hinge between voxels that touch only on an edge or a corner, or a thin gray member. It "
                     "may not be a buckling mode of a solid member of the printed part. Look at the mode before "
                     "you use this load factor.")
    n_loose = int(model.loose.sum()) if model.loose is not None else 0
    if move >= FALSE_MODE_SHARE and n_loose:
        n_solid = int((model.rho >= THRESHOLD).sum())
        facts.append(f"{n_loose} solid voxels ({100 * n_loose / max(n_solid, 1):.1f} percent of the solid voxels) "
                     "are not face-connected to a fixed region through solid voxels (a loose piece, or a contact "
                     f"on an edge or a corner only), and {100 * move:.1f} percent of the motion of the {mode} "
                     "is in them.")
    if "lowest_mode_energy_box_mm" in loc:
        lo, hi = loc["lowest_mode_energy_box_mm"]
        facts.append(f"The voxels with {100 * ENERGY_BOX_SHARE:.0f} percent of the strain energy of this mode lie "
                     f"in the box {lo} to {hi} mm (centre {loc['lowest_mode_energy_center_mm']}).")
    other = "gray" if model.density == "printed" else "printed"
    options = ["Look at the part in that box for a loose piece or a thin neck (topopt refine <run_dir>, "
               "result/final.stl).",
               f"Run `{bk.RERUN} --density {other}` and compare the load factors of the two densities."]
    return facts, options


def run_cases(model: VoxelModel, fac, f_red, names, *, modes: int = bk.DEFAULT_MODES, tol: float = bk.TOL,
              maxiter: int = bk.MAXITER, max_seconds: float | None = bk.DEFAULT_MAX_SECONDS,
              log=None, K=None) -> list[dict]:
    """The eigen solve of each load case. `fac` holds the factor of the K of `model` and nothing else
    may be factored while this runs. `f_red` (n_free, n_cases) are the loads on the free dofs.
    `K` is the stiffness matrix of the model if the caller has it (the factor was made from it).
    Return one record for each case (`topopt.buckling.case_summary`, plus `seconds` and where the mode lives).
    """
    asm = model.asm
    if K is None:
        K = asm.K(model.Ee)
    U = fac.solve(np.ascontiguousarray(f_red))
    U = U.reshape(model.free.size, -1)
    out = []
    for i, name in enumerate(names):
        t0 = time.perf_counter()
        u_all = np.zeros(model.n_dof_all)
        u_all[model.free] = U[:, i]
        Kg = hex8_kg(asm, model.edof, u_all, model.h, model.E, model.nu, model.weight)
        t_kg = time.perf_counter() - t0
        res = bk.buckling_modes(Kg, K, fac, k=modes, tol=tol, maxiter=maxiter, max_seconds=max_seconds)
        rec = bk.case_summary(res, name, modes)
        rec["kg_seconds"] = round(t_kg, 3)
        loc = {}
        if res["lam"].size and np.isfinite(res["lam"][0]):
            loc = mode_location(model, res["phi"][:, 0])
            rec.update(loc)
            partial = rec["status"] != bk.OK
            rec["shares_of_mode"] = "lowest converged (not always the lowest)" if partial else "lowest"
            facts, options = false_mode_facts(model, loc, partial)
            rec["facts"] += facts
            rec["options"] += options
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        if log is not None:
            lam = rec["load_factors"]
            line = (f"buckling: case {name}: " + (f"lowest load factor {lam[0]:.6g}, {len(lam)} found, "
                    if lam else f"{rec['status']}, ") + f"{rec['solves']} solves, {rec['seconds']:.1f} s")
            if "lowest_mode_energy_share_rho_lt_half" in loc:
                line += (", strain energy of the lowest mode in voxels below 0.5: "
                         f"{100 * loc['lowest_mode_energy_share_rho_lt_half']:.2f} percent")
            log(line)
        out.append(rec)
    return out


# --------------------------------------------------------------------------- the check of a run

def _mb(nbytes) -> int:
    return int(round(nbytes / 2 ** 20))


def gb(nbytes) -> str:
    """Bytes as GB (1e9 bytes), the unit of `buckling.max_gb`. Every memory text of the check uses it."""
    return f"{nbytes / 1e9:.1f} GB"


def memory_budget(max_gb: float | None):
    """The budget in bytes: `max_gb` (GB, 1e9 bytes) if given, else the RAM of this machine (None if not known)."""
    from .cli import physical_ram_bytes
    return int(max_gb * 1e9) if max_gb is not None else physical_ram_bytes()


def skip_decision(name, n_active, max_active, est, budget, max_gb=None, avail=None):
    """Why the check cannot run, or None. `name` is the resolved solver. Return `(reason, facts, options)`.

    The same rule serves the check of a run (`check_problem`) and `topopt check` (a warning before the run).
    """
    if name not in DIRECT_SOLVERS:
        if avail is None:
            from .cli import backend_availability
            avail = backend_availability()
        options = []
        for d in DIRECT_SOLVERS:
            info = avail.get(d, {})
            if info.get("available"):
                options.append(f"Set settings.solver: {d} in the YAML and run `{bk.RERUN}` (the optimization does not "
                               f"need to run again). A direct factor of this model needs about {gb(est)}.")
            else:
                options.append(f"{d}: {info.get('note', 'not known')}")
        options.append("An iterative path (LOBPCG with an AMG preconditioner) is not in this version.")
        return ("solver_not_direct",
                [f"The check needs a direct factor of K with a solve call. The solver of this run is {name}: "
                 f"it has no such factor (amg is an inexact solve, gpu-mg has no matrix). "
                 f"The memory estimate of a direct factor of this model is {gb(est)}. "
                 "The run goes on without the check."], options)
    if n_active > max_active:
        return ("voxel_guard",
                [f"The model has {n_active} active voxels, above the voxel guard of {max_active} "
                 "(settings.max_active_voxels). The check is not started."],
                ["Use a larger voxel_size, or raise settings.max_active_voxels if this machine has the memory."])
    if budget is not None and est > budget:
        src = f"buckling.max_gb = {max_gb:g}" if max_gb is not None else "the RAM of this machine"
        return ("memory",
                [f"The memory estimate of the check is {gb(est)}, above the budget of {gb(budget)} ({src}; "
                 "GB = 1e9 bytes). The check is not started."],
                ["Use a larger voxel_size (this needs a new run).",
                 f"Run `{bk.RERUN} --max-gb G` with a larger G, or leave max_gb out to use the RAM of this machine."])
    return None


def _skipped(result, reason, facts, options):
    result.update(status="skipped", reason=reason, facts=facts, options=options, cases=[],
                  lowest_load_factor=None)
    return result


def preflight(cfg, solver, grid_voxels: int, n_active: int, free_dofs: int, ram_bytes=None) -> dict:
    """What `topopt check` says about a check that the YAML turns on: will it run, and if not, why.

    `solver` is the resolved solver name, or None when this platform cannot run the solver of the YAML.
    Return a dict with `will_run`, `reason`, `memory_estimate_bytes`, `facts` and `options`.
    """
    b = cfg.buckling
    est = estimate_bytes(grid_voxels, free_dofs, b.modes)
    out = {"check": True, "modes": b.modes, "density": b.density, "memory_estimate_bytes": est,
           "will_run": True, "reason": None, "facts": [], "options": []}
    if solver is None:
        skip = ("solver_unavailable",
                ["The solver of this run cannot start here, so the check cannot run either."],
                [f"Set settings.solver in the YAML to a solver that runs on this machine, then run `{bk.RERUN}`."])
    else:
        budget = int(b.max_gb * 1e9) if b.max_gb is not None else ram_bytes
        skip = skip_decision(solver, n_active, cfg.settings.max_active_voxels, est, budget, b.max_gb)
    if skip:
        out.update(will_run=False, reason=skip[0], facts=skip[1], options=skip[2])
    if cfg.has_joints():      # joints (section 38): the check runs with the condensed stiffness S; the beams have no Kg
        from .joints import BUCKLING_BEAMS_FACT, BUCKLING_BEAMS_OPTION
        out["facts"] = out["facts"] + [BUCKLING_BEAMS_FACT]
        out["options"] = out["options"] + [BUCKLING_BEAMS_OPTION]
    return out


def check_problem(problem, density_u8, *, modes: int = bk.DEFAULT_MODES, density: str = "printed",
                  max_seconds: float | None = bk.DEFAULT_MAX_SECONDS, max_gb: float | None = None,
                  tol: float = bk.TOL, maxiter: int = bk.MAXITER, log=None) -> dict:
    """The linear buckling check of the final design. Return the dict for `metrics.json["buckling"]`.

    The result is always a dict. A solver without a direct factor (amg, gpu-mg), a model above the voxel
    guard or above the memory budget, and a load case without a buckling load are facts (`status`,
    `facts`, `options`), not errors. The memory budget is `max_gb` (GB) if given, else the RAM of this machine.
    """
    from .config import UserError
    from .solver_choice import resolve_solver
    from .verify.memory import peak_rss_bytes
    from .voxelize import DESIGN, LOAD
    t_start = time.perf_counter()
    cfg = problem.config
    st = cfg.settings
    result = {"kind": KIND, "status": "done", "meaning": MEANING, "density": density,
              "threshold": THRESHOLD if density == "printed" else None, "solver": None,
              "eigen_solver": bk.EIGEN_SOLVER, "modes": int(modes), "tol": tol, "maxiter": maxiter,
              "max_seconds": max_seconds, "facts": [], "options": []}
    labels = problem.labels
    n_active = int(((labels >= DESIGN) & (labels <= LOAD)).sum())
    grid_voxels = int(np.prod(problem.shape))
    result["active_voxels"] = n_active
    joints = problem.joints
    if joints is not None:      # joints (section 38): the eigen problem uses S = K + Kb - C D^-1 C^T; the beams have no Kg
        from .joints import BUCKLING_BEAMS_FACT, BUCKLING_BEAMS_OPTION, buckling_note
        result["joints"] = buckling_note(len(joints.joints))
        result["facts"].append(BUCKLING_BEAMS_FACT)
        result["options"].append(BUCKLING_BEAMS_OPTION)
        result["eigen_solver"] = bk.EIGEN_SOLVER_JOINTS

    try:
        name = resolve_solver(st.solver)
    except UserError as e:
        return _skipped(result, "solver_unavailable", [f"The solver of this run cannot start here: {e}"],
                        [f"Set settings.solver in the YAML to a solver that runs on this machine, then run `{bk.RERUN}`."])
    result["solver"] = name

    model = build_model(problem, density_u8, density)
    n_free = int(model.free.size)
    est = estimate_bytes(grid_voxels, n_free, modes)
    result.update(free_dofs=n_free, memory_estimate_bytes=est)
    rho_opt = model.rho_optimizer
    result["solid_voxels"] = int((model.rho >= THRESHOLD).sum())
    result["gray_voxels"] = int(((rho_opt > GRAY_LOW) & (rho_opt < GRAY_HIGH)).sum())
    result["loose_solid_voxels"] = int(model.loose.sum())
    budget = memory_budget(max_gb)
    result["memory_budget_bytes"] = budget
    skip = skip_decision(name, n_active, st.max_active_voxels, est, budget, max_gb)
    if skip:
        return _skipped(result, *skip)

    from .optimize import make_solver
    peak0 = peak_rss_bytes()
    result["peak_rss_before_mb"] = _mb(peak0)
    solver, _name = make_solver(st.solver, model.free, problem.shape, problem.h, joints)
    try:
        t0 = time.perf_counter()
        K = model.asm.K(model.Ee)
        if joints is not None:      # springs and bonds: the same K + Kb as the run (`joints.support_matrix`)
            from .joints import condensed_border, support_matrix
            support_matrix(problem, model.asm, model.free).add_to(K)
        elif problem.springs is not None:     # springs (section 35): the same K + Kb as the run; Kg has no spring term (no prestress)
            problem.springs.matrix(model.asm, model.free).add_to(K)
        solver.factor(K)
        result["factor_seconds"] = round(time.perf_counter() - t0, 3)
        f_red = np.ascontiguousarray(problem.forces[:, model.free].T)
        S = K if joints is None else condensed_border(joints, model.free).condense(K)   # the matrix that `solver` inverts
        cases = run_cases(model, solver, f_red, problem.case_names, modes=modes, tol=tol, maxiter=maxiter,
                          max_seconds=max_seconds, log=log, K=S)
    finally:
        solver.free()
    result["cases"] = cases
    found = [(c["load_factors"][0], c["name"]) for c in cases if c["load_factors"]]
    result["lowest_load_factor"] = ({"case": min(found)[1], "value": min(found)[0]} if found else None)
    result["seconds"] = round(time.perf_counter() - t_start, 3)
    peak1 = peak_rss_bytes()
    result["peak_rss_mb"] = _mb(peak1)
    result["peak_grew"] = bool(peak1 > peak0)
    result["peak_note"] = ("peak_rss_mb is the peak of the whole process. When peak_grew is false, the check "
                           "stayed below an earlier peak (the run itself).")
    for c in cases:
        for key in ("facts", "options"):
            result[key].extend(f"{c['name']}: {t}" for t in c[key])
    return result


def error_result(density: str, message: str, reason: str = "exception") -> dict:
    """The record of a check that stopped with an error or an interrupt (`status: error`). A fact, not a crash."""
    again = f"Run `{bk.RERUN}` again"
    options = {"exception": [f"{again} with a direct solver and enough memory (settings.solver, buckling.max_gb), "
                             "or leave the check off."],
               "interrupted": [f"{again} to repeat the check."]}.get(reason, [again + "."])
    return {"kind": KIND, "status": "error", "reason": reason, "meaning": MEANING, "density": density,
            "message": message, "cases": [], "lowest_load_factor": None,
            "facts": [f"The check stopped: {message}"], "options": options}


def run_check(run_dir, problem=None, *, log=None, yaml_vs_run=None, **overrides) -> dict:
    """Run the check on the run folder `run_dir` and write the key `buckling` of `result/metrics.json`.

    `problem` is the `Problem` of the run. If it is None, it is rebuilt from the YAML named in run.json.
    `yaml_vs_run` (the result of `verify.run.compare_yaml_problem`, given by `topopt buckling`) goes into the record as the key
    `yaml_vs_run`: the differences between the YAML and run.json are a fact next to the load factors.
    The settings come from the YAML block `buckling`. A keyword (`modes`, `density`, `max_seconds`, `max_gb`,
    `tol`, `maxiter`) overrides them. Return the result dict. The check does not stop the caller on a skip
    or on a load case without a buckling load: those are facts in the result. A `KeyboardInterrupt` goes up
    to the caller and writes nothing (the caller decides what the record is).
    """
    import json
    from pathlib import Path

    from .post import _last_snapshot
    run_dir = Path(run_dir)
    if problem is None:
        from .refine.build import problem_from_run
        problem = problem_from_run(run_dir, json.loads((run_dir / "run.json").read_text()))
    b = problem.config.buckling
    opts = dict(modes=b.modes, density=b.density, max_seconds=b.max_seconds, max_gb=b.max_gb)
    opts.update(overrides)
    snap = np.frombuffer(_last_snapshot(run_dir).read_bytes(), np.uint8)
    try:
        result = check_problem(problem, snap, log=log, **opts)
    except Exception as e:   # an error is a fact in metrics.json, not a silent gap
        result = error_result(opts["density"], " ".join(f"{type(e).__name__}: {e}".split()))
    if yaml_vs_run is not None:
        result["yaml_vs_run"] = {k: yaml_vs_run[k] for k in ("matches", "differences", "note")}
    path = bk.write_metrics_key(run_dir, result)
    if log is not None:
        log(summary_line(result) + f" ({path.name})")
        for fact in result["facts"]:
            log("warning: buckling: " + fact)
        for option in result["options"]:
            log("buckling: option: " + option)
    return result


def summary_line(result: dict) -> str:
    """One log line about a result."""
    if result["status"] != "done":
        return f"buckling check {result['status']}: {result.get('message') or result.get('reason', '')}"
    low = result.get("lowest_load_factor")
    head = (f"lowest load factor {low['value']:.6g} ({low['case']})" if low
            else "no positive load factor in any case")
    return (f"buckling check (voxel estimate, {result['density']} design, {result['solver']}): {head}; "
            f"{result['seconds']:.1f} s, peak {result['peak_rss_mb']} MB")
