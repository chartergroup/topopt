"""Linear solvers for the P2 system, and the iterative refinement. CONTRACTS.md section 16.

A P2 system is one symmetric positive definite matrix. It is factored once, and every load case is
a right-hand side. A factor object has `solve(b)` (one vector or many columns) and `free()`. These
are the calls of `topopt.solver.LinearSolver`, so the factors here are thin adapters.

Names of the factors (`FACTOR_NAMES`):

- `cholesky`: Apple Accelerate sparse Cholesky, through `topopt.solver.LinearSolver`. macOS on Apple silicon.
- `pardiso`: Intel MKL PARDISO, through `topopt.solver.LinearSolver`. Linux x86_64 with the `pardiso` extra.
- `cholespy`: CHOLMOD through the package cholespy (`verify` extra). All platforms, one thread.
  License: GPL-2.0-or-later in effect: the wheel bundles CHOLMOD Supernodal, SuiteSparse 5.4.0 (the code of cholespy
  is BSD-3-Clause). Same rule as gmsh. It is optional and is imported inside `CholespyFactor` only.
  CHOLMOD stops on a pivot of zero or below, but it is often silent on a singular matrix, because roundoff gives a tiny positive
  pivot (on numpy 2.3.5 for the floating body of the tests; on numpy 2.5.3 it raised for that body but was silent on 17 of 17 matrices
  held at one node or one line, E3b review). So, like SuperLU, the factor makes a test solve (`FACTOR_TEST_RESIDUAL`). The test solve
  finds most singular matrices, but it can miss one with one free rotation (a part held along one line). `verify_run` therefore also
  stops on the relative residual of a real load case (`FACTOR_TEST_RESIDUAL` or more, for every factor).
- `superlu`: scipy SuperLU (`splu`). All platforms, one thread, BSD-3-Clause, always installed (scipy is a main dependency).
  It is the last fallback: on the hook it is faster than cholespy at 1.0 mm (23 s against 45 s) but uses about 2 times the memory,
  and its fill grows fast with the size and with how compact the part is.

`resolve_factor(None)` ("auto") takes the first one that works here, in this order. AMG is never chosen:
it is an iterative method, so its error depends on its tolerance. Every direct factor gives the same refined solution.
AMG is a possible later option for a low-memory run: the rigid-body modes of a P2 system come from the node
coordinates, and `RefinedFactor` around an AMG-CG solve would converge to the direct result (not built, not measured).
`resolve_factor(name)` takes the name as given and raises `ValueError` for an unknown name. A factor that cannot run is
`SolverUnavailable` (a `UserError`, exit code 1), with the note of every factor. Which factor ran is a printed fact
(`topopt.verify.report`), never a silent switch. The check is independent of the voxel model because of the P2 tet
model and its own mesh, not because of the factor: the factor of the optimizer can be the same one.

`RefinedFactor` wraps any of them. It adds refinement steps with a residual in double-double
arithmetic (ported from `pipeline/lib/p2_solver.py`, revision d43bcf3). Because of it, the solution
does not depend on the factor: the factors give the same refined solution to about 1e-9 or better.
"""
from __future__ import annotations

import contextlib
import re
import shlex
import sys
from pathlib import Path

import numba as nb
import numpy as np
import scipy.sparse as sp

from ..uv_command import uv_command
from ..solver_choice import (PYPARDISO_PIN, install_command, is_linux_x86_64, is_macos_arm64, platform_name,
                             pypardiso_problem)
from .errors import SingularSystem, SolverUnavailable, VerifyError, VerifyUnavailable
from .names import opt

FACTOR_NAMES = ("cholesky", "pardiso", "cholespy", "superlu")
AUTO_ORDER = FACTOR_NAMES
CHOLESPY_PIN = "cholespy==2.2.0"
SKFEM_PIN = "scikit-fem==12.0.2"
# A test solve (after the factor of SuperLU or of cholespy) with a larger relative residual: the factor of a singular matrix.
# Measured on the hook P2 matrix at 2.0 mm (SuperLU): a valid matrix gives 4e-13, a floating part gives 3, a part with one fixed
# node gives 6; a valid matrix that is very ill-conditioned (a floating part held by a spring of 1e-12 times the diagonal) gives
# 3e-6. So 1e-3 sits between them. cholespy, floating body of tests/test_verify_bodies.py (1,275 dofs), numpy 2.3.5, where CHOLMOD
# does not stop: 0.8 (and 1e12 in the solution), against 3e-14 for a valid matrix of the same mesh.
# The limits on both sides (E3b review, re-measured for the path spring on both numpy pins): a valid matrix is refused from a spring of
# 3e-15 times the diagonal (path Laplacian of 400 nodes: cholespy 2.9e-3, SuperLU 2.2e-3; with 1e-14 they give 6.5e-4 and 2.8e-4 and are
# accepted) or from a stiffness contrast of 1e10 (a bar with a middle third of E x 1e-10: 6.4e-3; with E x 1e-9: 7.2e-4, accepted).
# A singular matrix with one free rotation can pass: the 6 mm cube held at one edge gives 7.7e-4 (cholespy) and 4.2e-4 (SuperLU), and a
# fixed vector cannot tell it from the valid bar (7.2e-4). The residual of a real load case can (0.2 to 0.34 on that cube, 3e-12 on the
# hook at 1.0 mm), so `verify_run` stops on it with the same constant.
FACTOR_TEST_RESIDUAL = 1e-3


# --------------------------------------------------------------------------- #
# Which factors work here
# --------------------------------------------------------------------------- #

def _shim_problem() -> str | None:
    """Why the Accelerate shim does not load, or None if it loads. Only asked on macOS arm64."""
    from .. import solver
    try:
        solver._accel_lib()
    except Exception as exc:                  # RuntimeError (file missing) or OSError (cannot load)
        return f"{type(exc).__name__}: {exc}"
    return None


def verify_install_command() -> str:
    """The command that installs the whole `verify` extra (gmsh, scikit-fem and cholespy) into the Python that runs topopt now.

    One helper for every text that gives the command: each text says "the 'verify' extra", so the command must install all three
    in each route. A source tree gets `-e '<root>[verify]'` (the forms of `solver_choice.install_command`). An installed package
    (a wheel or a non-editable install: no `pyproject.toml` beside it) gets the three pins of the extra; `gmsh` only where the
    extra has a gmsh wheel (the marker of the extra). `tests/test_verify_run.py` checks the pins against `pyproject.toml`.
    """
    from ..tet.gmsh_probe import GMSH_PIN, has_wheel_platform
    py = shlex.quote(sys.executable)
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        return f"{uv_command()} pip install --python {py} -e {shlex.quote(str(root) + '[verify]')}"
    pins = ([GMSH_PIN] if has_wheel_platform() else []) + [SKFEM_PIN, CHOLESPY_PIN]
    return f"{uv_command()} pip install --python {py} {' '.join(shlex.quote(pin) for pin in pins)}"


def _cholespy_error(exc: BaseException) -> tuple[str, str]:
    """`("missing", text)` if the package is not installed, else `("load_failed", text)` (installed, but the import fails:
    for example a C++ runtime that is too old, or a module under cholespy that is missing). `text` is the exception."""
    if isinstance(exc, ModuleNotFoundError) and exc.name in (None, "cholespy"):
        return "missing", str(exc)
    return "load_failed", f"{type(exc).__name__}: {exc}"


def cholespy_problem() -> tuple[str, str] | None:
    """Why cholespy does not load, or None if it loads (the form of `solver_choice.pypardiso_problem`)."""
    try:
        import cholespy  # noqa: F401
    except Exception as exc:
        return _cholespy_error(exc)
    return None


def cholespy_note(problem: tuple[str, str]) -> str:
    """What to tell the user about a cholespy that does not load: the real error, and the step that fits it.

    Only a missing package gets the hint to install the extra. A package that is installed and fails to load gets the
    error text (the extra is there already), as `solver_choice.resolve_solver` does for pypardiso.
    """
    kind, text = problem
    if kind == "missing":
        return f"cholespy is not installed ({text}). The 'verify' extra holds it; install the extra with: {verify_install_command()}"
    return f"cholespy is installed but does not load: {text}. Reinstall it with: {uv_command()} pip install --reinstall {CHOLESPY_PIN}"


def _superlu_problem() -> str | None:
    try:
        from scipy.sparse.linalg import splu  # noqa: F401
    except Exception as exc:                  # scipy is a main dependency: only a broken install gets here
        return f"{type(exc).__name__}: {exc}"
    return None


def factor_availability() -> dict:
    """Which factors can run here: `{name: {"available": bool, "note": str}}`.

    Imports pypardiso only on Linux x86_64 and the Accelerate shim only on macOS arm64 (the rules of
    `solver_choice`). Imports cholespy and scipy.sparse.linalg.
    """
    out = {}
    if is_macos_arm64():
        problem = _shim_problem()
        out["cholesky"] = {"available": problem is None,
                           "note": "Apple Accelerate sparse Cholesky" if problem is None else
                           f"the shim does not load ({problem}); build it with topopt/native/build.sh"}
    else:
        out["cholesky"] = {"available": False,
                           "note": f"Apple silicon only (this platform is {platform_name()})"}
    if not is_linux_x86_64():
        out["pardiso"] = {"available": False,
                          "note": f"Linux x86_64 only (this platform is {platform_name()})"}
    else:
        problem = pypardiso_problem()
        if problem is None:
            out["pardiso"] = {"available": True, "note": f"Intel MKL PARDISO ({PYPARDISO_PIN})"}
        elif problem[0] == "missing":
            out["pardiso"] = {"available": False,
                              "note": f"pypardiso is not installed. Install it with: {install_command()}"}
        else:
            out["pardiso"] = {"available": False, "note": f"pypardiso is installed but does not load: {problem[1]}"}
    problem = cholespy_problem()
    out["cholespy"] = {"available": problem is None,
                       "note": "CHOLMOD through cholespy, one thread" if problem is None else cholespy_note(problem)}
    problem = _superlu_problem()
    out["superlu"] = {"available": problem is None,
                      "note": "scipy SuperLU, one thread (always installed; the last fallback)" if problem is None else
                      f"scipy SuperLU does not load ({problem}). Check the install of scipy"}
    return out


def check_name(name: str | None) -> None:
    """Raise `ValueError` if `name` is not None, "auto" or one of `FACTOR_NAMES`."""
    if name is not None and name != "auto" and name not in FACTOR_NAMES:
        extra = (" amg is not offered for verify: it is an iterative method, so its error depends on its "
                 "tolerance, and every direct factor gives the same refined solution. It can come later as a low-memory "
                 "option (the refinement of `RefinedFactor` could wrap it)." if name == "amg" else "")
        raise ValueError(f"unknown solver {name!r} for verify. Use one of: auto, {', '.join(FACTOR_NAMES)}.{extra}")


def resolve_factor(name: str | None = None, availability: dict | None = None) -> str:
    """The name of the factor that runs: one of `FACTOR_NAMES`.

    `name` None or "auto": the first factor of `AUTO_ORDER` that works here (see the module text).
    Raise `SolverUnavailable` (a `UserError`; the text lists the facts and the fix) if none works.
    Any other `name` is taken as given. Raise `ValueError` if it is not one of `FACTOR_NAMES`,
    and `SolverUnavailable` if it cannot run here.
    `availability` is the result of `factor_availability()` (read it once, for a fake platform in tests).
    """
    check_name(name)
    avail = factor_availability() if availability is None else availability
    if name in (None, "auto"):
        for cand in AUTO_ORDER:
            if avail.get(cand, {}).get("available"):
                return cand
        lines = "; ".join(f"{k}: {v['note']}" for k, v in avail.items())
        raise SolverUnavailable(f"no direct linear solver can run here ({lines}). Options: make one of these factors work "
                                f"(each note says how), then run again", dict(requested="auto", factors=avail))
    if not avail.get(name, {}).get("available"):
        others = [k for k, v in avail.items() if k != name and v["available"]]
        flag = opt("solver")
        opts = f" Options: {flag} {(', ' + flag + ' ').join(others)}, or auto." if others else ""
        raise SolverUnavailable(f"solver {name!r} cannot run here: {avail.get(name, {}).get('note', 'unknown')}.{opts}",
                                dict(requested=name, factors=avail))
    return name


# --------------------------------------------------------------------------- #
# A singular matrix
# --------------------------------------------------------------------------- #

# What the solvers say when the matrix is singular or not positive definite (found by running each one on a floating body):
#   pardiso   RuntimeError "PARDISO factor failed: ... error code -4 ... (zero pivot ...)" (code -7: diagonal matrix is singular)
#   cholespy  ValueError "Matrix is not positive definite!" (CHOLMOD also prints a warning to stderr). Often not: CHOLMOD is silent
#             on many singular matrices (a tiny positive pivot) on both numpy pins; the floating body of the tests is silent on numpy
#             2.3.5. `CholespyFactor` makes a `LinAlgError` "Singular matrix" from the residual of a test solve (FACTOR_TEST_RESIDUAL),
#             as `SuperluFactor` does. That test can miss a matrix with one free rotation: `verify_run` also checks the load cases.
#   cholesky  RuntimeError "Accelerate SparseFactor failed, status -1 or -2" (factorization failed, matrix is singular)
#   numpy/scipy dense or sparse paths: LinAlgError "Singular matrix", RuntimeError "Factor is exactly singular"
#   superlu   no error: with no pivoting it gives a near-zero pivot and a wrong solution. `SuperluFactor` makes a `LinAlgError`
#             "Singular matrix" from the residual of a test solve (FACTOR_TEST_RESIDUAL).
_SINGULAR_TEXT = re.compile(r"zero pivot|not positive definite|singular|error code -[47]\b|SparseFactor failed, status -[12]\b|"
                            r"non-positive diagonal", re.I)


def singular_reason(exc: BaseException) -> str | None:
    """The text of `exc` if it says that the matrix is singular or not positive definite, else None."""
    if isinstance(exc, np.linalg.LinAlgError) or _SINGULAR_TEXT.search(str(exc)):
        return f"{type(exc).__name__}: {exc}"
    return None


def singular_message(solver: str, detail: str, dofs: int | None = None, context: str = "", supports: bool = False) -> str:
    """The user text for a singular matrix: what is singular, the likely causes, the options.

    With `supports` (the YAML has a joint or a spring, K1-A) the text names them as a cause and as an option: a spring that has stiffness
    on one axis holds that axis only, so the part moves along the other two.
    """
    n = f" ({dofs:,} free dofs)" if dofs else ""
    held = (", a spring or a joint that holds fewer than the six rigid-body motions (a spring with stiffness on one axis leaves the other two "
            "axes free), or a joint or spring region with few facets") if supports else ""
    held_opt = (f"; a stiffness on all three axes of a spring, a joint, a second spring on another face or a fixed region, so that the "
                f"supports hold all six motions; {opt('region_band')} larger also takes more facets into a joint or spring region") if supports else ""
    return (f"the stiffness matrix of the tet model{n} is singular: the {solver} solver stopped on a zero pivot, a matrix "
            f"that is not positive definite, or a solve with a large residual ({detail}). A singular matrix means that a part of the mesh can move with no "
            f"strain. Likely causes: a floating body (a piece of the part with no support), a load or a support that has no "
            f"contact with the part, a fixed region with too few nodes (one line of nodes lets the part turn){held}, or "
            f"degenerate tets. {context}Options: look at the result and the regions (topopt render <run>); "
            f"{opt('region_band')} larger to take more nodes into a fixed region{held_opt}; {opt('mesh_size')} for another mesh; "
            f"or change the optimizer problem so that the part connects the load to the supports.")


@contextlib.contextmanager
def singular_guard(solver: str, dofs: int | None = None):
    """Turn a singular-matrix error of a solver into `SingularSystem` (exit code 1). Any other error goes through."""
    try:
        yield
    except SingularSystem:
        raise
    except Exception as exc:
        reason = singular_reason(exc)
        if reason is None:
            raise
        raise SingularSystem(singular_message(solver, reason, dofs),
                             dict(reason="singular_matrix", solver=solver, detail=reason, dofs=dofs)) from exc


# --------------------------------------------------------------------------- #
# Factors
# --------------------------------------------------------------------------- #

def _spd_csr(A: sp.spmatrix) -> sp.csr_matrix:
    A = A.tocsr()
    if np.any(A.diagonal() <= 0):
        raise ValueError("non-positive diagonal in the stiffness matrix")
    return A


def _test_solve_residual(solve, apply, n: int) -> float:
    """The relative residual `|A x - b| / |b|` of a test solve after a factor: `b` is a fixed random vector (the same every time),
    `solve(b)` gives `x`, and `apply(x)` gives `A x`. `A` is the (scaled) matrix that the factor was made for."""
    b = np.random.default_rng(0).normal(size=n)
    return float(np.linalg.norm(apply(solve(b)) - b) / np.linalg.norm(b))


def _test_solve_error(label: str, residual: float) -> np.linalg.LinAlgError:
    """The error for a test solve with a relative residual of `residual` that is not below `FACTOR_TEST_RESIDUAL`.
    `singular_guard` turns it into `SingularSystem`."""
    return np.linalg.LinAlgError(
        f"Singular matrix: {label} gave a test solve with a relative residual of {residual:.1e} (threshold "
        f"{FACTOR_TEST_RESIDUAL:.0e}; a very ill-conditioned valid matrix can also come near it, and another factor, "
        f"named with {opt('solver')}, may factor that)")


class LinearSolverFactor:
    """The `topopt.solver.LinearSolver` (cholesky or pardiso) behind the factor interface."""

    def __init__(self, A: sp.spmatrix, method: str):
        from ..solver import LinearSolver
        self.n = A.shape[0]
        self.name = method
        with singular_guard(method, self.n):
            A = _spd_csr(A)
            self._ls = LinearSolver(method)
            self._ls.factor(A.tocsc())

    def solve(self, b: np.ndarray) -> np.ndarray:
        with singular_guard(self.name, self.n):
            return self._ls.solve(np.asarray(b, float))

    def free(self):
        if self._ls is not None:
            self._ls.free()
        self._ls = None


class CholespyFactor:
    """CHOLMOD through cholespy. The matrix is scaled by its diagonal (as in `LinearSolver`).

    CHOLMOD raises "Matrix is not positive definite!" for a pivot of zero or below, but a singular matrix whose pivot comes out
    as a tiny positive number (roundoff) gives no error and a wrong solution; this is common, on both numpy pins. So the factor makes
    the same test solve as `SuperluFactor`: a relative residual of `FACTOR_TEST_RESIDUAL` or more is a `LinAlgError`, which
    `singular_guard` turns into `SingularSystem`. The residual is that of the matrix that CHOLMOD factored: the lower triangle, made
    symmetric. Like the factor itself, the test does not see an upper-only matrix (the matrix must be the full symmetric matrix, or its
    lower triangle).

    The test solve finds most singular matrices, not all: with one fixed vector it can miss a matrix with one free rotation (a part
    held along one line; see the comment of `FACTOR_TEST_RESIDUAL` for the numbers). The residual of a real load case does not miss it,
    so `verify_run` also stops on that residual, with the same constant.
    """

    name = "cholespy"

    def __init__(self, A: sp.spmatrix):
        try:
            from cholespy import CholeskySolverD, MatrixType
        except ImportError as exc:
            raise SolverUnavailable(
                "solver 'cholespy' needs the package cholespy (GPL-2.0-or-later in effect: the wheel bundles CHOLMOD "
                f"Supernodal, SuiteSparse 5.4.0). {cholespy_note(_cholespy_error(exc))}", dict(requested="cholespy")) from exc
        self.n = A.shape[0]
        with singular_guard(self.name, self.n):
            A = _spd_csr(A)
            self.s = 1.0 / np.sqrt(A.diagonal())
            As = sp.diags(self.s) @ A @ sp.diags(self.s)
            C = sp.tril(As, format="coo")        # cholespy reads the lower triangle only (tested: same solution)
            del As
            self._f = CholeskySolverD(self.n, C.row.astype(np.int32), C.col.astype(np.int32),
                                      C.data.astype(np.float64), MatrixType.COO)
            d = C.diagonal()
            r = _test_solve_residual(lambda b: self._solve_scaled(np.ascontiguousarray(b[:, None]))[:, 0],
                                     lambda x: C @ x + C.T @ x - d * x, self.n)      # lower triangle made symmetric
            if not r < FACTOR_TEST_RESIDUAL:                     # also true for NaN
                self._f = None
                raise _test_solve_error("cholespy", r)

    def _solve_scaled(self, B: np.ndarray) -> np.ndarray:
        """Solve the scaled system for the columns of `B` (a C-contiguous float64 array of shape (n, k))."""
        X = np.zeros_like(B)
        self._f.solve(B, X)
        return X

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, float)
        one = b.ndim == 1
        B = np.ascontiguousarray(b.reshape(self.n, -1) * self.s[:, None])
        with singular_guard(self.name, self.n):
            X = self._solve_scaled(B)
        X = X * self.s[:, None]
        return X[:, 0] if one else X

    def free(self):
        self._f = None


class SuperluFactor:
    """scipy SuperLU (`splu`): a direct LU factor that is always installed. The matrix is scaled by its diagonal.

    The matrix is symmetric positive definite, so SuperLU runs with the order `MMD_AT_PLUS_A`, symmetric mode and no
    pivoting (`diag_pivot_thresh=0`). That keeps the fill small: on the hook matrix (115,836 dofs) it needed 274 s
    and 11.8 GB against 396 s and 22.6 GB with the default order. One thread. The fill grows fast with the size of the
    system and with how compact the part is (a cube needs several times the memory of a hook of the same dofs).

    SuperLU does not stop on a singular matrix when it does not pivot: it gives a near-zero pivot and a solution that is
    wrong. A test solve (a fixed vector) finds most of these matrices: a relative residual of `FACTOR_TEST_RESIDUAL` or more is a
    `LinAlgError`, which `singular_guard` turns into `SingularSystem`. `CholespyFactor` makes the same test. It can miss a matrix
    with one free rotation, which the residual of a real load case in `verify_run` stops.

    The matrix must be the full symmetric matrix (both triangles), as for the other factors except `cholespy`, which reads
    the lower triangle. SuperLU reads every entry, so an upper-only or lower-only matrix gives a wrong solution and no error
    (the test solve is made with the matrix that was given, so it does not see it).
    """

    name = "superlu"

    def __init__(self, A: sp.spmatrix):
        from scipy.sparse.linalg import splu
        self.n = A.shape[0]
        with singular_guard(self.name, self.n):
            A = _spd_csr(A).astype(np.float64, copy=False)       # splu keeps a float32 matrix in single precision
            self.s = 1.0 / np.sqrt(A.diagonal())
            As = (sp.diags(self.s) @ A @ sp.diags(self.s)).tocsc()
            del A
            try:
                self._lu = splu(As, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
            except MemoryError as exc:
                raise VerifyError(
                    f"SuperLU ran out of memory in the factor ({self.n:,} free dofs, {As.nnz:,} matrix entries). Its fill "
                    f"grows fast with the size of the system and with how compact the part is. scipy SuperLU has 32-bit "
                    f"indices, so near 2^31 factor entries (about 17 GB) more memory may not help (not measured here). "
                    f"Options: a larger {opt('mesh_size')}; a factor with less memory: pypardiso (Linux x86_64, extra "
                    f"'pardiso') or cholespy (extra 'verify'), then {opt('solver')} NAME; or a machine with more memory.") from exc
            r = _test_solve_residual(self._lu.solve, lambda x: As @ x, self.n)
            if not r < FACTOR_TEST_RESIDUAL:                     # also true for NaN
                self._lu = None
                raise _test_solve_error("SuperLU", r)

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, float)
        one = b.ndim == 1
        B = np.ascontiguousarray(b.reshape(self.n, -1) * self.s[:, None])
        with singular_guard(self.name, self.n):
            X = self._lu.solve(B)
        X = X * self.s[:, None]
        return X[:, 0] if one else X

    def free(self):
        self._lu = None


def make_factor(A: sp.spmatrix, name: str):
    """Factor the matrix `A` with the factor `name` (one of `FACTOR_NAMES`; use `resolve_factor` first)."""
    if name in ("cholesky", "pardiso"):
        return LinearSolverFactor(A, name)
    if name == "cholespy":
        return CholespyFactor(A)
    if name == "superlu":
        return SuperluFactor(A)
    raise ValueError(f"unknown solver {name!r}")


# --------------------------------------------------------------------------- #
# Compensated residual and refinement (ported from pipeline/lib/p2_solver.py)
# --------------------------------------------------------------------------- #

@nb.njit(cache=True)
def _two_sum(a, b):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


@nb.njit(cache=True)
def _two_prod(a, b):
    p = a * b
    c = 134217729.0 * a
    ah = c - (c - a)
    al = a - ah
    c = 134217729.0 * b
    bh = c - (c - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


@nb.njit(cache=True, parallel=True)
def _residual_dd(indptr, indices, data, xhi, xlo, b, out):
    """out = b - A (xhi + xlo), accurate as if computed in twice the working precision."""
    n = indptr.shape[0] - 1
    for i in nb.prange(n):
        s = 0.0
        c = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            a = data[k]
            p, e = _two_prod(a, xhi[j])
            s, t = _two_sum(s, p)
            c += t + e + a * xlo[j]
        t, e2 = _two_sum(b[i], -s)
        out[i] = t + (e2 - c)


@nb.njit(cache=True)
def _max_abs_row_sum(indptr, data):
    """max over rows of sum |a_ij| (no temporary copy of the matrix)."""
    best = 0.0
    for i in range(indptr.shape[0] - 1):
        t = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            t += abs(data[k])
        if t > best:
            best = t
    return best


class RefinedFactor:
    """Solve with a factor, then refine with accurate residuals. One or many right-hand sides.

    `A` is the matrix, `fac` any object with `solve(b)` and `free()` (a `LinearSolver` works too),
    `steps` the number of refinement steps (the verifier uses 3). It does not test for convergence.
    """

    def __init__(self, A: sp.spmatrix, fac, steps: int = 2, accurate: bool = True):
        self.A = A.tocsr()
        self.fac = fac
        self.steps = int(steps)
        self.accurate = accurate
        self.n = A.shape[0]
        self._ip = self.A.indptr.astype(np.int64, copy=False)
        self.norm_inf = float(_max_abs_row_sum(self._ip, self.A.data))

    def _residual(self, xhi, xlo, b):
        if self.accurate:
            out = np.empty(self.n)
            _residual_dd(self._ip, self.A.indices, self.A.data, xhi, xlo, b, out)
            return out
        return b - self.A @ (xhi + xlo)

    def _solve1(self, b):
        xhi = np.ascontiguousarray(self.fac.solve(b))
        xlo = np.zeros(self.n)
        for _ in range(self.steps):
            r = self._residual(xhi, xlo, b)
            dx = np.ascontiguousarray(self.fac.solve(r))
            # double-double accumulation: (xhi, xlo) += dx
            s = xhi + dx
            bb = s - xhi
            e = (xhi - (s - bb)) + (dx - bb)
            xlo = xlo + e
            xhi = s
        return xhi + xlo

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, float)
        if b.ndim == 1:
            return self._solve1(np.ascontiguousarray(b))
        return np.column_stack([self._solve1(np.ascontiguousarray(b[:, j])) for j in range(b.shape[1])])

    def backward_error(self, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Normwise backward error |b - A x|_inf / (|A|_inf |x|_inf + |b|_inf) per column (residual in double-double)."""
        b = np.asarray(b, float).reshape(self.n, -1)
        x = np.asarray(x, float).reshape(self.n, -1)
        out = np.empty(b.shape[1])
        z = np.zeros(self.n)
        for j in range(b.shape[1]):
            r = np.empty(self.n)
            _residual_dd(self._ip, self.A.indices, self.A.data, np.ascontiguousarray(x[:, j]), z,
                         np.ascontiguousarray(b[:, j]), r)
            den = self.norm_inf * np.abs(x[:, j]).max() + np.abs(b[:, j]).max()
            out[j] = np.abs(r).max() / den if den > 0 else np.nan             # b = 0 and x = 0: not defined
        return out

    def residual_norms(self, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """|b - A x|_2 / |b|_2 per column, residual in double-double. NaN for a column with b = 0 (not defined)."""
        b = np.asarray(b, float).reshape(self.n, -1)
        x = np.asarray(x, float).reshape(self.n, -1)
        out = np.empty(b.shape[1])
        z = np.zeros(self.n)
        for j in range(b.shape[1]):
            r = np.empty(self.n)
            _residual_dd(self._ip, self.A.indices, self.A.data, np.ascontiguousarray(x[:, j]), z,
                         np.ascontiguousarray(b[:, j]), r)
            nb = np.linalg.norm(b[:, j])
            out[j] = np.linalg.norm(r) / nb if nb > 0 else np.nan             # b = 0: not defined
        return out

    def free(self):
        self.fac.free()
