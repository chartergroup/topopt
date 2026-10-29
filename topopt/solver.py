"""Linear solvers for the reduced stiffness matrix.

"cholesky": Apple Accelerate sparse Cholesky (C shim in topopt/native). macOS only.
"pardiso": Intel MKL PARDISO through pypardiso. Linux x86_64 only, optional extra.
"amg": PyAMG smoothed aggregation as CG preconditioner, with warm start.
"auto": the platform rule in solver_choice.py. On macOS with the shim, callers that give a
        nullspace also get the size rule found by benchmark runs.
"""
from __future__ import annotations

import ctypes
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from .solver_choice import accelerate_library_path, import_pypardiso, resolve_solver

_HERE = Path(__file__).resolve().parent
_LIB = None

# Auto rule: Cholesky up to this many free dofs, else AMG. Set from benchmark runs.
AUTO_MAX_DOFS_CHOLESKY = 300_000
ORDERING = "metis"  # default | amd | metis | colamd
_ORDER_CODE = {"default": 0, "amd": 2, "metis": 3, "colamd": 4}


# Common PARDISO error codes (MKL documentation), for the message of a failed factor.
_PARDISO_CODES = {
    -1: "input data is not consistent",
    -2: "not enough memory",
    -3: "reordering problem",
    -4: "zero pivot: the matrix is not positive definite (a singular or unsupported system?)",
    -5: "unclassified internal error",
    -6: "reordering failed",
    -7: "diagonal matrix is singular",
    -9: "not enough memory for the out-of-core mode",
}


def _pardiso_reason(exc: Exception) -> str:
    """The text of a PARDISO exception, with the meaning of the error code if we know it."""
    text = str(exc)
    m = re.search(r"error code\s*(-?\d+)", text)
    code = _PARDISO_CODES.get(int(m.group(1))) if m else None
    return f"{text} ({code})" if code else text


def _accel_lib():
    global _LIB
    if _LIB is None:
        path = accelerate_library_path()
        if not path.exists():
            raise RuntimeError(f"{path} not found. Run topopt/native/build.sh")
        lib = ctypes.CDLL(str(path))
        lib.af_factor.restype = ctypes.c_void_p
        lib.af_factor.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.af_solve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.af_free.argtypes = [ctypes.c_void_p]
        lib.af_factor_ex.restype = ctypes.c_void_p
        lib.af_factor_ex.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int)]
        lib.af_refactor.restype = ctypes.c_int
        lib.af_refactor.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _LIB = lib
    return _LIB


class LinearSolver:
    """factor(K) once, then solve(B) for many right-hand sides.

    nullspace: optional (n, 6) rigid-body modes for AMG (fe.rigid_body_modes).
    rtol: relative residual target for AMG-CG.
    reuse_amg: rebuild the AMG hierarchy only every N calls of factor().
    """

    def __init__(self, method: str = "auto", nullspace=None, rtol: float = 1e-5,
                 maxiter: int = 200, reuse_amg: int = 1, ordering: str = ORDERING,
                 block3: bool = False, theta: float = 0.05,
                 chol_block: bool = False):
        if method not in ("auto", "cholesky", "pardiso", "amg"):
            raise ValueError("method must be auto, cholesky, pardiso or amg")
        self.requested = method
        self.method = method
        self.nullspace = nullspace
        self.rtol = rtol
        self.maxiter = maxiter
        self.reuse_amg = max(1, int(reuse_amg))
        self.ordering = ordering
        self.block3 = block3  # dofs come in x,y,z triplets per node: use 3x3 blocks in AMG
        self.theta = theta
        self.chol_block = chol_block  # 3x3 blocks in Cholesky. Same speed as scalar in tests.
        self._h_key = None
        self.n = None
        self._h = None
        self._lower_key = None
        self._pat_ref = None
        self._x_prev = None
        self._n_factor = 0
        self._ml = None
        self.last_iters = []
        # PARDISO backend (Linux x86_64). The Cholesky knobs ordering and chol_block do not apply.
        self._ps = None            # pypardiso.PyPardisoSolver, made at the first factor()
        self._pd_ref = None        # (indptr, indices, nnz) of the cached pattern
        self._pd_pat = None        # upper triangle pattern in CSR, see _pardiso_pattern
        self._pd_ready = False     # PARDISO holds an analysis (and a factor)
        self._pd_A = None          # scaled upper triangle of the last factor

    # ---------------------------------------------------------------- factor
    def factor(self, K: sp.csc_matrix, nullspace=None):
        K = K.tocsc() if not sp.isspmatrix_csc(K) else K
        n = K.shape[0]
        if nullspace is not None:
            self.nullspace = nullspace
        method = self.requested
        if method == "auto":
            method = resolve_solver("auto")
            if method == "cholesky":   # macOS: the size rule for callers that give a nullspace
                method = "cholesky" if (n <= AUTO_MAX_DOFS_CHOLESKY or self.nullspace is None) else "amg"
        elif method in ("cholesky", "pardiso"):
            resolve_solver(method)     # UserError where this platform cannot run the solver
        if method == "amg" and self.nullspace is None:
            raise ValueError("AMG needs the rigid-body nullspace")
        if self.n != n:
            self._x_prev = None
            self._ml = None
        self.n = n
        self.method = method
        d = K.diagonal()
        if np.any(d <= 0):
            raise ValueError("non-positive diagonal in the stiffness matrix")
        s_new = 1.0 / np.sqrt(d)
        self.s = s_new
        if method == "cholesky":
            self._factor_chol(K)
        elif method == "pardiso":
            self._factor_pardiso(K)
        else:
            self._factor_amg(K)
        self._n_factor += 1

    def _lower_pattern(self, K):
        """Cache the lower-triangle pattern. The Assembler shares its index arrays."""
        ref = self._pat_ref
        same = ref is not None and ref[2] == K.nnz and (
            (ref[0] is K.indptr and ref[1] is K.indices)
            or (np.array_equal(ref[0], K.indptr) and np.array_equal(ref[1], K.indices)))
        key = self._lower_key
        if not same:
            self._pat_ref = (K.indptr, K.indices, K.nnz)  # keep a reference: no address reuse
            key = self._lower_key = object()
            self._h_key = None
        if not same:
            if self.chol_block and self.n % 3 == 0:
                self._lower = self._block_pattern(K)
            else:
                col = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(K.indptr))
                mask = K.indices >= col
                ip = np.zeros(self.n + 1, np.int64)
                np.cumsum(np.bincount(col[mask], minlength=self.n), out=ip[1:])
                self._lower = ("scalar", mask, ip, np.ascontiguousarray(K.indices[mask], dtype=np.int32),
                               col[mask])
        return self._lower

    def _block_pattern(self, K):
        """3x3 block lower triangle. perm maps block entries to K.data (-1 = zero)."""
        tag = sp.csc_matrix((np.arange(1, K.nnz + 1, dtype=np.float64), K.indices, K.indptr),
                            shape=K.shape)
        # K is symmetric: BSR rows of K are block columns of the CSC. Row-major
        # blocks of B_ij equal column-major blocks of B_ji.
        tb = tag.tocsr().tobsr(blocksize=(3, 3))
        nb = self.n // 3
        rows = np.repeat(np.arange(nb, dtype=np.int32), np.diff(tb.indptr))
        keep = tb.indices >= rows  # block column >= block row: lower triangle of the CSC
        ip = np.zeros(nb + 1, np.int64)
        np.cumsum(np.bincount(rows[keep], minlength=nb), out=ip[1:])
        ri = np.ascontiguousarray(tb.indices[keep], dtype=np.int32)
        perm = tb.data[keep].reshape(-1).astype(np.int64) - 1  # -1 where zero
        # scaling factors per entry: s[row] * s[col] with scalar dofs
        blk_row = ri.astype(np.int64)  # block row of the CSC entry
        blk_col = rows[keep].astype(np.int64)
        return ("block", perm, ip, ri, blk_row, blk_col)

    def _factor_chol(self, K):
        lo = self._lower_pattern(K)
        lib = _accel_lib()
        if lo[0] == "scalar":
            _, mask, ip, ri, colm = lo
            va = np.ascontiguousarray(K.data[mask] * self.s[ri] * self.s[colm])
            bs, nrow = 1, self.n
        else:
            _, perm, ip, ri, br, bc = lo
            d = np.where(perm >= 0, K.data[np.maximum(perm, 0)], 0.0).reshape(-1, 3, 3)
            sm = self.s.reshape(-1, 3)
            # d[i0, i1] = K[3*bc+i0, 3*br+i1] (symmetric K)
            va = np.ascontiguousarray(d * sm[bc][:, :, None] * sm[br][:, None, :])
            bs, nrow = 3, self.n // 3
        if self._h and self._h_key is self._lower_key:
            # same pattern: reuse the symbolic factor
            if lib.af_refactor(self._h, va.ctypes.data) != 0:
                raise RuntimeError("Accelerate refactor failed")
            return
        self._free_handle()
        st = ctypes.c_int(0)
        self._h = lib.af_factor_ex(nrow, ip.ctypes.data, ri.ctypes.data, va.ctypes.data,
                                   _ORDER_CODE[self.ordering], None, bs, ctypes.byref(st))
        if not self._h or st.value != 0:
            self._h = None
            raise RuntimeError(f"Accelerate SparseFactor failed, status {st.value}")
        self._h_key = self._lower_key

    def _pardiso_pattern(self, K):
        """Upper triangle of K as a CSR pattern, for PARDISO matrix type 2.

        Return `(perm, rows, cols, indptr, changed)`. `perm` maps the entries to
        `K.data`, `rows` and `cols` are the row and column of each entry.
        The pattern is cached: the Assembler shares its index arrays.
        """
        ref = self._pd_ref
        same = ref is not None and ref[2] == K.nnz and (
            (ref[0] is K.indptr and ref[1] is K.indices)
            or (np.array_equal(ref[0], K.indptr) and np.array_equal(ref[1], K.indices)))
        if not same:
            self._pd_ref = (K.indptr, K.indices, K.nnz)  # keep a reference: no address reuse
            # Tag each entry with its place in K.data (+1), then let scipy cut the upper triangle.
            tag = sp.csc_matrix((np.arange(1, K.nnz + 1, dtype=np.float64), K.indices, K.indptr),
                                shape=K.shape)
            up = sp.triu(tag.tocsr(), format="csr")
            up.sort_indices()  # PARDISO needs sorted column indices in each row
            perm = up.data.astype(np.int64) - 1
            rows = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(up.indptr))
            self._pd_pat = (perm, rows, up.indices, up.indptr)
        return (*self._pd_pat, not same)

    def _factor_pardiso(self, K):
        perm, rows, cols, indptr, changed = self._pardiso_pattern(K)
        # Same scaling as the Cholesky path: S K S has a unit diagonal.
        va = K.data[perm] * self.s[rows] * self.s[cols]
        A = sp.csr_matrix((va, cols, indptr), shape=K.shape, copy=False)
        if self._ps is None:
            self._ps = import_pypardiso().PyPardisoSolver(mtype=2)  # real, symmetric positive definite
        ps = self._ps
        if self._pd_ready and changed:
            ps.free_memory(everything=True)    # new pattern: drop the old analysis
            self._pd_ready = False
        # Phase 12: analysis and factor. Phase 22: factor only, same pattern as before.
        # pypardiso 0.4.7 (pinned) has no public call for phase 22, and factorize()
        # would copy the matrix and analyse it again at each iteration.
        ps.set_phase(22 if self._pd_ready else 12)
        self._pd_A = None          # no valid factor until the call below works
        try:
            ps._call_pardiso(A, np.zeros((self.n, 1)))
        except Exception as e:
            # The factor is not valid now. Release it, so that solve() cannot use a stale one.
            self._free_pardiso(force=True)
            raise RuntimeError(f"PARDISO factor failed: {_pardiso_reason(e)}") from e
        self._pd_ready = True
        self._pd_A = A

    def _factor_amg(self, K):
        import pyamg
        s = self.s
        S = sp.diags(s)
        self._A = (S @ K @ S).tocsr()
        self._K = K
        if self._ml is None or self._n_factor % self.reuse_amg == 0:
            Bs = self.nullspace / s[:, None]
            A = self._A.tobsr(blocksize=(3, 3)) if (self.block3 and self.n % 3 == 0) else self._A
            self._ml = pyamg.smoothed_aggregation_solver(
                A, B=Bs, symmetry="symmetric", max_coarse=100,
                strength=("symmetric", {"theta": self.theta}))
            self._M = self._ml.aspreconditioner()

    # ----------------------------------------------------------------- solve
    def solve(self, B: np.ndarray, warm: bool = True) -> np.ndarray:
        """Solve K X = B with the stored factor.

        `warm=False` is for extra solves (adjoint loads) between two main solves.
        AMG then starts from zero and keeps its warm start for the main solves.
        The direct solvers ignore `warm`.
        """
        B = np.asarray(B, dtype=np.float64)
        one = B.ndim == 1
        B = B.reshape(self.n, -1)
        if self.method == "cholesky":
            X = self._chol_apply(B)
        elif self.method == "pardiso":
            X = self._pardiso_apply(B)
        else:
            m = B.shape[1]
            if not warm:
                x_prev = np.zeros((self.n, m))
            else:
                if self._x_prev is None or self._x_prev.shape[1] != m:
                    self._x_prev = np.zeros((self.n, m))
                x_prev = self._x_prev
            X = np.empty((self.n, m))
            iters = []
            for c in range(m):
                it = [0]

                def cb(_xk, it=it):
                    it[0] += 1
                y, info = cg(self._A, self.s * B[:, c], x0=x_prev[:, c] / self.s,
                             M=self._M, rtol=self.rtol, maxiter=self.maxiter, callback=cb)
                if info != 0:
                    raise RuntimeError(f"AMG-CG did not converge, info={info}")
                X[:, c] = y * self.s
                iters.append(it[0])
            if warm:     # an extra solve keeps the warm start and last_iters of the main solve
                self._x_prev = X.copy()
                self.last_iters = iters
        return X[:, 0] if one else X

    def _chol_apply(self, B):
        X = np.asfortranarray(B * self.s[:, None])
        _accel_lib().af_solve(self._h, self.n, X.shape[1], X.ctypes.data)
        return X * self.s[:, None]

    def _pardiso_apply(self, B):
        if self._ps is None or not self._pd_ready or self._pd_A is None:
            raise RuntimeError("PARDISO has no factor: call factor() first "
                               "(the last factor failed, or free() was called)")
        self._ps.set_phase(33)    # solve with the stored factor
        X = self._ps._call_pardiso(self._pd_A, np.asfortranarray(B * self.s[:, None]))
        return X * self.s[:, None]

    # ------------------------------------------------------------ checkpoint
    def checkpoint_state(self) -> dict:
        """State that a resume needs (CONTRACTS.md section 24.2). AMG starts each CG from the last solution (`warm`).

        The direct solvers keep no state between passes: an empty dict.
        """
        return {} if self._x_prev is None else {"x_prev": self._x_prev}

    def restore_checkpoint_state(self, state: dict) -> list[str]:
        """Put the saved state back. Return facts about what the checkpoint lacks."""
        if "x_prev" in state:
            self._x_prev = np.array(state["x_prev"], dtype=np.float64)
            self.n = self._x_prev.shape[0]   # `factor` resets the warm start when the size differs from `n`
            return []
        if self.requested == "amg":
            return ["the checkpoint has no AMG warm start: the first solve starts from zero"]
        return []

    # --------------------------------------------------------------- cleanup
    def _free_pardiso(self, force=False):
        """Release the PARDISO memory. `force`: also if no factor is marked as ready."""
        self._pd_A = None
        held = self._pd_ready or force
        self._pd_ready = False
        if self._ps is not None and held:
            try:
                self._ps.free_memory(everything=True)
            except Exception:   # after an error PARDISO may have nothing to free
                pass

    def _free_handle(self):
        if self._h:
            _accel_lib().af_free(self._h)
            self._h = None

    def free(self):
        self._free_handle()
        self._free_pardiso()

    def __del__(self):
        try:
            self._free_handle()
            self._free_pardiso()
        except Exception:
            pass
