"""Independent P2 stress check of the final STL of a run (CONTRACTS.md section 16).

    from topopt.verify import verify_run
    result = verify_run("runs/live", estimate_only=True)       # the estimate, no meshing
    result = verify_run("runs/live", allowable_mpa=40.0)        # mesh, solve, write runs/live/verify/

This package needs the `verify` extra (gmsh through the `tet` extra, scikit-fem, cholespy). It does not
import any of them when you import it. The core of topopt never imports this package at module level.
The library parts are `p2` (quadratic tets), `stats` (von Mises and percentiles), `factor` (linear solvers and
refinement), `regions` and `bc` (regions, loads and supports), `memory` (estimates). The command is `topopt verify`.
"""
from __future__ import annotations

__all__ = ["EXIT_COULD_NOT_RUN", "EXIT_LIMIT", "EXIT_OK", "verify_run"]


def __getattr__(name: str):
    if name in __all__:
        from . import run
        return getattr(run, {"verify_run": "verify_run", "EXIT_OK": "EXIT_OK",
                             "EXIT_LIMIT": "EXIT_LIMIT", "EXIT_COULD_NOT_RUN": "EXIT_COULD_NOT_RUN"}[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
