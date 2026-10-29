#!/usr/bin/env python3
"""Check robust mode on one problem file, on its own grid (CONTRACTS.md section 22b). No run folder.

    python tools/robust_check.py problem.yaml [--beta 4] [--entry 0]

The script loads the YAML through the normal path (`load_config`, `build_problem`, `Model`) and prints, for an
even design at the volume fraction:

- whether robust mode is on, and eta_n and eta_e;
- the mean density of the nominal design and of the eroded design (the volume row uses the first);
- the compliance of the eroded design (what the optimizer minimises) and of the nominal design (one more solve);
- a central finite difference of the eroded compliance and of the nominal volume for one design variable, with
  the analytic slope (the best of five steps from 1e-2 to 1e-4: the error falls with the step, then the noise of
  the solves raises it again, see pitfalls/fd-check-noise-floor).

Use a small grid (voxel_size 2 mm or more): the check takes one solve for each step of the difference.
Exit code 0 when the best-step errors are below 1e-4, 1 when not (or the file is wrong).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from topopt.config import UserError, load_config
from topopt.optimize import Model
from topopt.problem import build_problem

LIMIT = 1e-4   # a correct gradient: the best step gives 1e-5 or less here, 1e-7 on the small test fixture. A wrong slope: 1e-2 or more


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("yaml")
    ap.add_argument("--beta", type=float, default=4.0, help="Heaviside beta of the check (default %(default)s)")
    ap.add_argument("--entry", type=int, default=0, help="index of the design variable for the difference (default 0)")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.yaml)
        problem = build_problem(cfg)
        m = Model(problem)
    except UserError as e:
        print(f"error: {e}")
        return 1
    st = cfg.settings
    if m.solver_name == "amg":
        m.solver.rtol = 1e-13   # AMG-CG is not exact at its default tolerance
    x = np.full(m.n_design, st.volume_fraction)
    e = min(max(a.entry, 0), m.n_design - 1)
    print(f"{cfg.name}: grid {problem.shape}, {m.n_design} design voxels, {m.n_free} free dofs, solver {m.solver_name}")
    if m.robust is None:
        print("settings.robust is off: the stiffness sees the nominal density (the default path)")
    else:
        print(f"settings.robust is eroded: eta_n {m.robust.eta_n:g}, eta_e {m.robust.eta_e:g} ({m.robust.source})")
    rho_tilde, rho = m.density(x, a.beta)
    rho_e = m.stiffness_density(rho_tilde, rho, a.beta)
    print(f"beta {a.beta:g}: mean density nominal {rho.mean():.6f}, eroded {rho_e.mean():.6f} "
          f"(the volume row uses the nominal one)")
    c, dc, v, dv = m.evaluate(x, a.beta)
    print(f"compliance of the stiffness density (eroded in robust mode): {c:.9g}")
    if m.robust is not None:
        U = m.solve(m.stiffness(rho))
        cn = float(m.weights @ np.einsum("ij,ij->j", m.f_red, U))
        print(f"compliance of the nominal design (one more solve): {cn:.9g}  (eroded / nominal = {c / cn:.4f})")
    best = [np.inf, np.inf]
    for eps in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        xp, xm = x.copy(), x.copy()
        xp[e] += eps
        xm[e] -= eps
        cp, _, vp, _ = m.evaluate(xp, a.beta)
        cm, _, vm, _ = m.evaluate(xm, a.beta)
        fd = [(cp - cm) / (2 * eps), (vp - vm) / (2 * eps)]
        err = [abs(fd[0] - dc[e]) / abs(dc[e]), abs(fd[1] - dv[e]) / abs(dv[e])]
        best = [min(b, r) for b, r in zip(best, err)]
        print(f"  entry {e}, step {eps:g}: dc/dx fd {fd[0]:.8g} analytic {dc[e]:.8g} (rel {err[0]:.1e}); "
              f"dv/dx fd {fd[1]:.8g} analytic {dv[e]:.8g} (rel {err[1]:.1e})")
    ok = max(best) < LIMIT
    print(f"best-step relative error: compliance {best[0]:.1e}, volume {best[1]:.1e}: {'ok' if ok else 'TOO LARGE'}")
    m.solver.free()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
