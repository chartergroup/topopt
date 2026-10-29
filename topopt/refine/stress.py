"""Von Mises stress of the final design. Used by the refine step and by the stress report of the post step.

The optimizer loop does not call `compute_stress`. `topopt.post.finish` calls it for a run with a stress row (`run.json`
has the key `stress`): `finish` calls `stress_report.stress_section`, and that function calls `compute_stress` for the
last design and again for the design cut at rho 0.5. It keeps the first result in the `cache` that it gets, and
`topopt run` passes that result to `build_refine`, so the refine step does not solve a second time. A run with no stress
row, and `topopt refine <run_dir>`, call `compute_stress` from `build_refine`. The helpers `isotropic_D` and
`strain_matrix_centre` are defined in `topopt.fe` and only re-exported here; the voxel buckling check
(`topopt/buckling_voxel.py`) imports them from this module.

Method: rebuild the stiffness of the last snapshot with the same SIMP rule
as the optimizer (`E_min + rho^p (E - E_min)`), make one solve for each load
case, and find the von Mises stress at the centre of each hex8 voxel.

Stress rule (one choice, kept simple): the stress is `D0 * strain` with the
full material modulus `E`, not `E(rho) * strain`. A voxel with rho >= 0.5 is
solid in the printed part (the mesh threshold is 0.5). A scale by `rho^p`
would lower the stress of the boundary voxels, and the surface of the part is
where the boundary voxels are. So the surface colours would show the density
noise and not the stress.
"""
from __future__ import annotations

import numpy as np

from topopt.config import UserError
from topopt.fe import Assembler, hex8_ke, isotropic_D, strain_matrix_centre, von_mises  # noqa: F401  (the three helpers moved to topopt.fe in A2)
from topopt.filter import simp_E
from topopt.joints import support_matrix
from topopt.optimize import E_MIN, make_solver, reduced_system
from topopt.voxelize import DESIGN

RHO_MIN = 0.5  # only voxels with rho >= RHO_MIN are in the result


def compute_stress(problem, density_u8: np.ndarray, log=None, solver: str | None = None) -> dict:
    """Von Mises stress (MPa) of the final design, for each load case.

    `density_u8` is the last snapshot (flat or (nx, ny, nz), 0..255). `log` (a function of one line, or None) is for `settings.solver` gpu-mg
    case C with joints: this solve makes its own extension of the harmonic columns (cold), and `log` gets one line with its iterations and seconds.
    It also gets the line that says a gpu-mg solve of `settings.solver: auto` failed and this solve went on with the CPU solver.
    `solver` is the solver that the run ended on (`run.json` `solver`), for `settings.solver: auto` (`solver_fallback.stress_solver`). Return a
    dict for voxels with rho >= 0.5: `centers` (n, 3) in mm (the frame of
    final.stl), `rho`, `vm_cases` (n_cases, n), `vm_max` (n,), `case_names`,
    and `compliance_cases` (n_cases,) for a check against the optimizer. `design` (n,) says which
    of these voxels are design voxels (the others are forced solid: fixed or load). A2 added it.
    """
    cfg = problem.config
    st, mat = cfg.settings, cfg.material
    h = problem.h
    snap = np.asarray(density_u8, dtype=np.uint8).reshape(-1)
    if snap.size != int(np.prod(problem.shape)):
        raise ValueError("the snapshot does not match the grid shape")

    edof, active_index, free = reduced_system(problem)
    label = problem.labels.reshape(-1)[active_index]
    is_design = label == DESIGN
    rho = np.ones(active_index.size)
    rho[is_design] = snap[active_index[is_design]].astype(np.float64) / 255.0

    # Same SIMP stiffness as `Model.stiffness`: solid voxels have full stiffness.
    Ee = np.full(active_index.size, mat.E)
    Ee[is_design] = mat.E * simp_E(rho[is_design], st.penal, 1.0, E_MIN)
    f_red = np.ascontiguousarray(problem.forces[:, free].T)
    from topopt.solver_fallback import stress_solver
    name, fallback = stress_solver(cfg, solver)
    U = None
    if name == "gpu-mg":    # matrix-free: no global matrix and no factor (CONTRACTS.md section 20)
        from topopt.gpu_mg.device import GpuFailure
        from topopt.gpu_mg.model import solve_design
        try:
            U = solve_design(problem, edof, active_index, free, Ee, f_red, log=log)
        except GpuFailure as failure:
            if fallback is None:        # settings.solver is gpu-mg: the error is the answer
                raise
            from topopt.solver_fallback import cpu_memory_problem, reason_text
            reason = f"{failure.kind}: {reason_text(failure)}"
            memory = cpu_memory_problem(problem, fallback)
            if memory:
                raise UserError(f"solver gpu-mg stopped in the stress solve ({reason}), and the solve cannot go on with {fallback}: {memory}. "
                                "Options: a larger settings.voxel_size, or a machine with more RAM") from failure
            if log is not None:
                log(f"solver: gpu-mg stopped in this solve ({reason}); this solve goes on with {fallback}")
            name = fallback
    if U is None:
        assembler = Assembler(edof, hex8_ke(1.0, mat.nu, h), free)
        K = assembler.K(Ee)
        kb = support_matrix(problem, assembler, free)
        if kb is not None:      # springs and joints (CONTRACTS.md sections 35, 38): the same K + Kb as the optimizer
            kb.add_to(K)
        lin, _name = make_solver(name, free, problem.shape, h, problem.joints)   # the solver of the run (`name` is the resolved name of `settings.solver`)
        try:
            lin.factor(K)
            U = lin.solve(f_red)
        finally:
            lin.free()
    compliance = np.einsum("ij,ij->j", f_red, U)

    keep = rho >= RHO_MIN
    Bc = strain_matrix_centre(h)
    D0 = isotropic_D(mat.E, mat.nu)
    edof_keep = edof[keep]
    u_all = np.zeros(3 * int(np.prod(np.array(problem.shape) + 1)))
    vm_cases = np.empty((U.shape[1], int(keep.sum())))
    for i in range(U.shape[1]):
        u_all[:] = 0.0
        u_all[free] = U[:, i]
        eps = u_all[edof_keep] @ Bc.T      # (n, 6)
        vm_cases[i] = von_mises(eps @ D0.T)

    flat = active_index[keep]
    nx, ny, nz = problem.shape
    ijk = np.stack([flat // (ny * nz), (flat // nz) % ny, flat % nz], axis=1)
    centers = np.asarray(problem.origin, np.float64) + h * (ijk + 0.5)
    return {
        "centers": centers,
        "rho": rho[keep],
        "vm_cases": vm_cases,
        "vm_max": vm_cases.max(axis=0),
        "case_names": list(problem.case_names),
        "compliance_cases": compliance,
        "design": is_design[keep],
    }
