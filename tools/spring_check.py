"""Springs against analytic values: a series spring and a cantilever on a face spring (CONTRACTS.md section 35).

Run: python tools/spring_check.py [--lumped]
It builds the stiffness matrix with `topopt.fe.Assembler`, adds the face matrices of `topopt.springs` to the x = 0
face of a box, solves with scipy and prints the engine value, the analytic value and the relative error.
`--lumped` uses the lumped rule (a diagonal matrix) for the comparison: it over-stiffens the rotation of the support.
"""

import argparse

import numpy as np
import scipy.sparse.linalg as spla

from topopt.fe import Assembler, edof_table, hex8_ke
from topopt.springs import FACE_UNIT, SpringMatrix, face_nodes, face_triplets, sum_duplicates


def solve_box(shape, h, E, nu, k_total, load, lumped=False) -> float:
    """Work of the end load `load` (3 N, uniform traction) on a box with a spring of `k_total` (3 N/mm) on x = 0."""
    n_node = int(np.prod(np.array(shape) + 1))
    n_dof = 3 * n_node
    edof, _ = edof_table(shape, np.ones(shape, bool))
    free = np.arange(n_dof)
    asm = Assembler(edof, hex8_ke(1.0, nu, h), free)
    K = asm.K(np.full(edof.shape[0], E))
    first, last = np.zeros(shape, bool), np.zeros(shape, bool)
    first[0], last[-1] = True, True
    root = face_nodes([(0, -1, first)], shape)
    unit = np.eye(4) / 4.0 if lumped else FACE_UNIT
    rows, cols, vals = sum_duplicates(*face_triplets(root, np.asarray(k_total, float) / len(root), unit), n_dof)
    SpringMatrix(rows, cols, vals, asm._indptr, asm._indices, free).add_to(K)
    end = face_nodes([(0, 1, last)], shape)
    f = np.zeros(n_dof)
    share = np.bincount(end.ravel(), minlength=n_node) / (4.0 * len(end))     # consistent nodal loads of a uniform traction
    for d in range(3):
        f[d::3] = load[d] * share
    return float(f @ spla.spsolve(K.tocsc(), f))


def series(L=10.0, a=1.0, E=3500.0, kx=100.0, F=10.0, lumped=False):
    """Bar with a spring on one end and a force on the other (nu = 0): c = F^2 (1/kx + L/(E A))."""
    got = solve_box((int(L), int(a), int(a)), 1.0, E, 0.0, (kx, kx, kx), (F, 0, 0), lumped)
    return got, F * F * (1.0 / kx + L / (E * a * a))


def cantilever(n, L=40.0, b=8.0, E=3500.0, nu=0.3, ka=2.0, F=1.0, lumped=False):
    """Cantilever on a spring of `ka` N/mm3 on x = 0, transverse force F, n voxels across: the tip deflection.

    Analytic: F L^3 / (3 E I) + F L^2 / (ka I) + F / (ka A) + F L / (kappa G A), kappa = 10 (1 + nu) / (12 + 11 nu).
    """
    h = b / n
    A, I, G = b * b, b ** 4 / 12.0, E / (2.0 * (1.0 + nu))
    kappa = 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)
    got = solve_box((round(L / h), n, n), h, E, nu, (ka * A,) * 3, (0, 0, F), lumped) / F
    return got, F * L ** 3 / (3 * E * I) + F * L ** 2 / (ka * I) + F / (ka * A) + F * L / (kappa * G * A)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lumped", action="store_true", help="use the lumped face rule")
    a = ap.parse_args()
    for name, (got, ref) in [("series spring (compliance, N mm)", series(lumped=a.lumped))] + [
            (f"cantilever tip, {n} voxels across (mm)", cantilever(n, lumped=a.lumped)) for n in (2, 4, 8)]:
        print(f"{name:42s} engine {got:.8g}  analytic {ref:.8g}  error {100 * (got / ref - 1):+.4f} percent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
