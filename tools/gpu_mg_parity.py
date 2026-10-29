"""Compare one gpu-mg solve (cases B and C) with a direct solve, on a small cantilever.

    python tools/gpu_mg_parity.py [--h 1.0] [--device auto] [--cases B,C] [--tol 1e-8]

It builds a cantilever (design box, fixed plate, tip load, two load cases), makes one SIMP design,
and solves it with the direct solver of this platform (Accelerate Cholesky on a Mac, PARDISO on Linux,
when they are available) and with SuperLU, and with gpu-mg on an OpenCL device. It prints the relative
difference of the displacements and of the compliance for each load case.

The exit code is 1 if a relative difference of the displacements is above `--limit` (default 1e-6).
On a machine with a GPU this is the first check to run: it needs no long run.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import trimesh
import yaml

from topopt.config import UserError, load_config
from topopt.gpu_mg import device
from topopt.gpu_mg.model import GpuModel, GpuParams
from topopt.optimize import Model
from topopt.problem import build_problem
from topopt.solver import LinearSolver
from topopt.solver_choice import resolve_solver


def box(path, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    m.export(path)


def cantilever(folder: Path, h: float, solver: str = "auto", gpu_case: str = "auto", gpu_device: str = "auto"):
    box(folder / "design.stl", (0, 0, 0), (20, 10, 10))
    box(folder / "plate.stl", (-2, 0, 0), (0, 10, 10))
    box(folder / "tip.stl", (18, 4, 0), (20, 6, 3))
    d = {"name": "cantilever",
         "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [
             {"name": "down", "weight": 3.0, "forces": [{"region": "tip", "force": [0, 0, -2.0]}]},
             {"name": "side", "weight": 1.0, "forces": [{"region": "tip", "force": [1.0, 3.0, 0]}]}],
         "settings": {"voxel_size": h, "volume_fraction": 0.3, "iterations": 1, "solver": solver,
                      "gpu_case": gpu_case, "gpu_device": gpu_device}}
    path = folder / "c.yaml"
    path.write_text(yaml.safe_dump(d))
    return build_problem(load_config(path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h", type=float, default=1.0, help="voxel size in mm (1.0 gives 7260 free dofs)")
    ap.add_argument("--device", default="auto", help="settings.gpu_device: auto, a number or a part of a name")
    ap.add_argument("--cases", default="B,C", help="precision cases to test")
    ap.add_argument("--tol", type=float, default=1e-8, help="relative residual of the gpu-mg solve. The backward-error rule is on at its default "
                                                                    "(GpuParams.backward_error, as in `topopt run`); a solve at the FP64 floor is accepted by it")
    ap.add_argument("--limit", type=float, default=1e-6, help="largest accepted relative difference")
    args = ap.parse_args(argv)

    try:
        dev, auto_case = device.select(args.device, "auto")
    except UserError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"device: {dev.label()}; case auto = {auto_case}")
    folder = Path(tempfile.mkdtemp(prefix="gpu_mg_parity_"))
    prob = cantilever(folder, args.h)
    ref = Model(dataclasses.replace(prob, config=_with_solver(prob.config, "amg")))   # for its matrix only
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 0.9, ref.n_design)
    _rt, rho = ref.density(x, 2.0)
    K = ref.stiffness(rho)
    print(f"free dofs: {ref.n_free}, elements: {ref.n_el}")

    solutions = {}
    t0 = time.perf_counter()
    solutions["superlu"] = (spla.splu(K.tocsc()).solve(ref.f_red), time.perf_counter() - t0)
    platform_solver = resolve_solver("auto")
    if platform_solver in ("cholesky", "pardiso"):
        s = LinearSolver(platform_solver)
        t0 = time.perf_counter()
        s.factor(K)
        solutions[platform_solver] = (s.solve(ref.f_red), time.perf_counter() - t0)
        s.free()
    base = platform_solver if platform_solver in solutions else "superlu"

    rows, worst, ran, failed = [], 0.0, 0, []
    for case in [c.strip() for c in args.cases.split(",") if c.strip()]:
        try:
            device.choose_case(dev, case)
        except UserError as e:
            print(f"case {case}: skipped. {e}")
            continue
        m = None
        try:
            cfg_prob = cantilever(Path(tempfile.mkdtemp(prefix="gpu_mg_parity_")), args.h,
                                  solver="gpu-mg", gpu_case=case, gpu_device=args.device)
            m = GpuModel(cfg_prob, None, GpuParams(tol=args.tol))
            t0 = time.perf_counter()
            U = m.solve(m.stiffness(rho))
            t_solve = time.perf_counter() - t0
        except UserError as e:       # the build, an OpenCL error or a solve that does not converge: a fact
            print(f"case {case}: FAIL. {e}")
            failed.append(case)
            continue
        finally:
            if m is not None:
                m.solver.free()
        ran += 1
        rec = m.solver.records[-1]
        its = _iterations(rec["iterations"][0])
        for name, (Ur, t_ref) in solutions.items():
            du = max(np.linalg.norm(U[:, j] - Ur[:, j]) / np.linalg.norm(Ur[:, j]) for j in range(U.shape[1]))
            c, cr = np.einsum("ij,ij->j", ref.f_red, U), np.einsum("ij,ij->j", ref.f_red, Ur)
            dc = float(np.max(np.abs(c - cr) / np.abs(cr)))
            rows.append((name, case, du, dc, its, t_solve, t_ref))
            if name == base:
                worst = max(worst, du)

    print("\n| reference | gpu-mg case | max rel diff of u | max rel diff of compliance | iterations | gpu-mg solve s | reference s |")
    print("|---|---|---|---|---|---|---|")
    for name, case, du, dc, its, t_solve, t_ref in rows:
        print(f"| {name} | {case} | {du:.2e} | {dc:.2e} | {its} | {t_solve:.2f} | {t_ref:.2f} |")
    print(f"\nreference for the exit code: {base}. Note: the gpu-mg time includes the first kernel build "
          f"on devices that compile at the first launch; it is not a speed measurement.")
    if failed:
        print(f"FAIL: case {', '.join(failed)} did not run (see the text above)")
        return 1
    if not ran:
        print("FAIL: no case ran, so nothing was compared")
        return 1
    ok = worst < args.limit
    print("PASS" if ok else f"FAIL: a relative difference is above {args.limit:g}")
    return 0 if ok else 1


def _iterations(its: dict) -> str:
    """Case C: refinement steps. Case B: the PCG, and the restart steps when the PCG did not reach the tolerance."""
    if "pcg" in its:
        if "refine_steps" not in its:
            return f"{max(i for i in its['pcg'] if i)} PCG"
        return (f"{its['executed']} PCG (not converged), then {sum(its['inner'])} inner in "
                f"{its['refine_steps']} restart steps")
    return f"{sum(its['inner'])} inner in {its['refine_steps']} refinement steps"


def _with_solver(config, solver):
    return dataclasses.replace(config, settings=dataclasses.replace(config.settings, solver=solver))


if __name__ == "__main__":
    raise SystemExit(main())
