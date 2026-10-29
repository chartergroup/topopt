"""The self-test of `settings.solver: auto` (CONTRACTS.md section 20): one small gpu-mg solve against a direct solve.

`auto` picks gpu-mg only on a GPU that passes this test. The test builds a small cantilever in memory (no STL file, no run
folder), solves it once with gpu-mg on the device, solves it again with SuperLU on the host (scipy, always there), and
compares the displacements. It is the method of `tools/gpu_mg_parity.py`: the gpu-mg tolerance is `device.SELFTEST_TOL`, and the
test passes when the relative difference of the displacements is at most `device.SELFTEST_MAX_DIFF`.

`run(dev, case)` never raises an exception of the device or of the solve: a failure is a result with `passed` False and the
text in `error`. The cache for the process is in `device.self_test`; this module has no cache. It imports pyopencl (through the
engine) and numba, so it loads only when the test runs.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# The cantilever: elements along x, y and z, and the element size in mm. 3,888 free dofs: above the 3,000 of the coarsest
# level, so the hierarchy has more than one level. The numbers are the geometry of a test body, not limits.
SHAPE = (16, 8, 8)
H_MM = 1.0
DESIGN_SEED = 1             # the stiffness of the elements is a fixed random design (SIMP-like range)
DESIGN_RANGE = (0.3, 0.9)


def build(shape=SHAPE, h=H_MM):
    """The in-memory problem: `(problem, edof, active_index, free_dofs, Ee, f_red, K)`.

    The part is a block of design voxels, clamped at x = 0 (all dofs of the nodes of that plane). Two load cases load the nodes of a
    patch at the tip (x = max). `K` is the reduced stiffness matrix of the design, assembled for the direct solve.
    """
    from ..config import Config, Material, Settings
    from ..fe import Assembler, hex8_ke
    from ..filter import simp_E
    from ..optimize import E_MIN, reduced_system
    from ..problem import Problem
    from ..voxelize import DESIGN

    nx, ny, nz = shape
    material = Material()
    settings = Settings(voxel_size=h, volume_fraction=0.3, iterations=1, filter_radius=1.5 * h, solver="gpu-mg")
    config = Config(name="gpu-mg self-test", yaml_path=Path("gpu-mg-self-test"), regions=None, load_cases=(), material=material,
                    settings=settings)
    node = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k          # noqa: E731  (the node numbering of CONTRACTS.md section 3)
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    clamp = [node(0, j, k) for j in range(ny + 1) for k in range(nz + 1)]
    fixed = np.sort(np.array([3 * n + c for n in clamp for c in range(3)], dtype=np.int64))
    tip = [node(nx, j, k) for j in range(ny // 4, 3 * ny // 4 + 1) for k in range(nz // 4, 3 * nz // 4 + 1)]
    forces = np.zeros((2, 3 * n_nodes))
    for n in tip:                      # a down load and a sideways load, as in `tools/gpu_mg_parity.py`
        forces[0, 3 * n + 2] = -2.0 / len(tip)
        forces[1, 3 * n + 0] = 1.0 / len(tip)
        forces[1, 3 * n + 1] = 3.0 / len(tip)
    problem = Problem(config=config, origin=np.zeros(3), h=h, shape=(nx, ny, nz), labels=np.full((nx, ny, nz), DESIGN, dtype=np.uint8),
                      load_region_ids=np.full((nx, ny, nz), -1, dtype=np.int16), fixed_dofs=fixed, forces=forces,
                      case_names=["down", "side"], case_weights=np.array([3.0, 1.0]))
    edof, active_index, free_dofs = reduced_system(problem)
    rho = np.random.default_rng(DESIGN_SEED).uniform(*DESIGN_RANGE, size=edof.shape[0])
    Ee = material.E * simp_E(rho, settings.penal, 1.0, E_MIN)
    f_red = np.ascontiguousarray(forces[:, free_dofs].T)
    K = Assembler(edof, hex8_ke(1.0, material.nu, h), free_dofs).K(Ee)
    return problem, edof, active_index, free_dofs, Ee, f_red, K


def run(dev, case, tol: float, max_diff: float) -> dict:
    """The self-test on `dev` (a `DeviceInfo`) in the precision `case`. Return the result dict; never raise for a device or solve failure.

    Keys: `passed`, `rel_diff` (the largest relative difference of the displacements over the load cases, or None),
    `seconds` (all of the test, with the kernel build), `free_dofs`, `tol`, `max_diff`, `device`, `case`, `error` (None, or the text).
    """
    import scipy.sparse.linalg as spla

    t0 = time.perf_counter()
    result = {"passed": False, "rel_diff": None, "seconds": None, "free_dofs": None, "tol": tol, "max_diff": max_diff,
              "device": dev.name, "case": case, "error": None}
    solver = None
    try:
        from .model import GpuParams, GpuSolver      # loads pyopencl: an import error is a failed test, not an exception
        problem, edof, active_index, free_dofs, Ee, f_red, K = build()
        result["free_dofs"] = int(free_dofs.size)
        ref = spla.splu(K.tocsc()).solve(f_red)
        solver = GpuSolver(free_dofs, edof, active_index, problem, dev, case, f_red.shape[1], GpuParams(tol=tol))
        solver.set_design(Ee)
        U = solver.solve(f_red)
        diffs = [float(np.linalg.norm(U[:, j] - ref[:, j]) / np.linalg.norm(ref[:, j])) for j in range(ref.shape[1])]
        result["rel_diff"] = max(diffs)
        result["passed"] = bool(np.isfinite(result["rel_diff"]) and result["rel_diff"] <= max_diff)
    except Exception as e:      # a device error, a kernel that does not build, a solve that is not accepted: the test is not passed
        result["error"] = " ".join(f"{type(e).__name__}: {e}".split())[:400]
    finally:
        if solver is not None:
            try:
                solver.free()
            except Exception:       # the result is known already; a device that fails to release is not a second fact
                pass
        result["seconds"] = round(time.perf_counter() - t0, 2)
    return result
