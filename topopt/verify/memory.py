"""Memory and size estimates of a P2 check, and the measured peak. CONTRACTS.md section 16, "Memory model".

The model is `7.5 GB x dofs / 1e6 x 1.25`. The 7.5 GB per million dofs is the constant of the
pipeline verifier (one Apple run). The factor 1.25 is a margin: the G1 Linux study measured 4.05 to
4.47 GB at 431k dofs, where the constant gives 3.2 GB, and the hook runs measured 11 to 14 % above it.
Both are facts, not promises: the estimate is on the safe side for the hook.

SuperLU (`superlu`, the last fallback factor) has its own model, because its fill depends on the dofs and on how compact
the part is (see `superlu_fill_gb`; the model uses the square of the dofs, the hook measured about N^1.65). The other
factors keep the rule above. The SuperLU model gives a central value and a range (`memory_range_bytes`), not a margin: the
guard compares the central value with the limit.

The linear buckling check (B3) adds a term on top of the peak of the factor: `buckling_bytes` (see below). It is not part of the rule
above. The guard of the run does not count it: a check that does not fit is skipped with a record and the facts, and the stress check goes on.

Before meshing the number of dofs is not known. `estimate_dofs` gives a number from the volume `V` and the
surface area `A` of the STL, and the mesh size `h`:

    dofs ~ 6.35 V / h^3 + 17.0 A / h^2

The two constants are a least-squares fit (relative error) to 27 meshes of 5 shapes (the hook at 7 sizes, a box, a
plate, a sphere and a cylinder at 4 to 5 sizes each, `mesh_stl` with 1 thread). The estimate was between 0.86 and
1.15 times the real number of dofs, 5 % on average. The fit holds for a size above the edge length of the STL. A size
below it does not refine the surface, and the real number of dofs is then lower than the estimate.
"""
from __future__ import annotations

import sys

from .names import opt

GB = 1e9
GB_PER_MDOF = 7.5              # the pipeline constant: GB per million dofs
SAFETY = 1.25                  # margin on top of it
DOFS_PER_V_H3, DOFS_PER_A_H2 = 6.35, 17.0

# SuperLU (scipy `splu`, order MMD_AT_PLUS_A, as `SuperluFactor` runs it). Peak memory of the process, central value:
#   rule = 0.50 GB + 1.63 GB x (million dofs) + fill,  fill = 65,000 GB x C^2 x (million dofs)^2,  C = V / A^1.5
# C is dimensionless: 0.016 for the hook (thin and rough), 0.027 for a plate 60 x 40 x 5, 0.040 for a bar 80 x 10 x 10,
# 0.068 for a cube, 0.094 for a sphere. The base (0.50 + 1.63 x) is the verify process on the hook before the factor.
# The fill is fitted on 17 meshes: the hook at 7 sizes (45k to 662k dofs) and 4 other shapes (24k to 230k dofs); `mesh_stl`,
# 1 thread, one Linux x86_64 host. The model over the measured peak (no margin) was 0.94 to 1.3 on the hook, 1.0 to 1.45 on
# the cube, the sphere and the bar, and 0.60 to 0.87 on the plate: 0.60 to 1.45 on all 17 meshes. So the peak was 0.69 to 1.67
# times the central value (`SUPERLU_MODEL_OVER_PEAK`). The square of the dofs is the model; the fill of the hook grew about as
# N^1.65. Factor time is about 20 s per GB of fill: modelled over measured 0.4 to 1.9 for the 4 other shapes. These are facts
# about those shapes, not promises: the real fill depends on the mesh.
SUPERLU_BASE_GB, SUPERLU_BASE_GB_PER_MDOF = 0.50, 1.63
SUPERLU_FILL_GB = 65000.0
SUPERLU_S_PER_GB = 20.0
SUPERLU_MODEL_OVER_PEAK = (0.60, 1.45)   # lowest and highest value of model / measured peak on the 17 meshes
SUPERLU_INT32_FILL_GB = 15.0             # scipy SuperLU has 32-bit indices: near 2^31 entries (about 17 GB) of fill
HOOK_COMPACTNESS = 0.016       # C of the hook: used when the shape is not known


def compactness(volume_mm3: float, area_mm2: float) -> float:
    """C = V / A^1.5 of a closed surface (dimensionless): small for a thin or rough part, 0.094 for a sphere."""
    return float(volume_mm3) / float(area_mm2) ** 1.5


def superlu_fill_gb(dofs: float, c: float | None = None) -> float:
    """Modelled memory (GB) of the SuperLU fill for a P2 system of `dofs` dofs and compactness `c` (None: the hook)."""
    return SUPERLU_FILL_GB * (HOOK_COMPACTNESS if c is None else c) ** 2 * (dofs / 1e6) ** 2


def estimate_dofs(volume_mm3: float, area_mm2: float, size_mm: float) -> int:
    """Rough number of P2 dofs of a mesh of size `size_mm` (see the module text; about +-15 %)."""
    return int(round(DOFS_PER_V_H3 * volume_mm3 / size_mm ** 3 + DOFS_PER_A_H2 * area_mm2 / size_mm ** 2))


# Time model (seconds). Measured on one Linux x86_64 host (22 threads), the hook at 1.5 to 0.8 mm, 2 load cases,
# 8 runs. A fit of the stages, not a promise: the number is a fact about that host.
#   mesh, P2 model, regions, stiffness:  3.6 + 2.3 x (dofs / 1e5)
#   solve and stress, for each load case:  0.5 x (dofs / 1e5)
#   factor:  pardiso 0.52 x (dofs / 1e5)^1.07, cholespy 1.9 x (dofs / 1e5)^2.30
# There is no measurement of the Apple Accelerate factor on a P2 system, so `cholesky` has no estimate.
TIME_BASE_S, TIME_PER_1E5_S, TIME_CASE_PER_1E5_S = 1.2, 3.0, 0.85
#   superlu factor:  20 s x (the modelled fill in GB), see `superlu_fill_gb` (the solves cost more with a larger fill: not modelled)
FACTOR_TIME = {"pardiso": (0.59, 1.07), "cholespy": (1.88, 2.41)}
TIME_BACKENDS = ("pardiso", "cholespy", "superlu")
TIME_HOST = "one Linux x86_64 host with 22 threads"


def factor_seconds(dofs: float, backend: str, c: float | None = None):
    """Estimated time (s) of one factor of a P2 system of `dofs` dofs, or None if there is no measurement for `backend` (the factor part of `estimate_seconds`)."""
    if backend not in TIME_BACKENDS:
        return None
    if backend == "superlu":
        return SUPERLU_S_PER_GB * superlu_fill_gb(dofs, c)
    a, b = FACTOR_TIME[backend]
    return a * (dofs / 1e5) ** b


def estimate_seconds(dofs: float, backend: str, n_cases: int, sensitivity: bool = True, c: float | None = None):
    """Estimated run time (s) of one mesh level, or None if there is no measurement for `backend`.

    `sensitivity` adds the two extra load bands (a solve and a stress evaluation for each case). The model is the
    fit that the comment above gives. The estimate is for the host of that fit (a busy, shared host) and is
    within a factor of 1.5 of the 12 measured runs there (mean error about 18 %). A quiet host can be faster.
    `c` is the compactness of the part (see `compactness`): it matters for `superlu` only.
    """
    factor = factor_seconds(dofs, backend, c)
    if factor is None:
        return None
    x = dofs / 1e5
    solves = n_cases * (3 if sensitivity else 1)
    return float(TIME_BASE_S + TIME_PER_1E5_S * x + factor + TIME_CASE_PER_1E5_S * x * solves)


# The linear buckling check (B3) on top of the peak of the factor. Measured on the hook (Linux x86_64, 2 load cases, 6 modes):
# the peak grew by 72 to 234 MB at 139k dofs, 199 to 389 MB at 359k dofs and 300 MB at 662k dofs, and the term was 0.32, 0.53 and
# 0.81 GB: 1.35 to 4.4 times the growth at the research sizes, and 2.6 to 2.7 times at 139k dofs in the runs of the B3 report (one
# shape, one host; other shapes and more load cases are not measured). So the term is above the measured growth. Pieces, in bytes:
#   12 x (nnz(K)/3 + 2 nnz(Kff)/3)   Kg of all dofs (a scalar block matrix x I3, so nnz(K)/3 entries), its free rows and columns,
#                                    and the copy of -Kg that ARPACK needs; 12 bytes per entry (8 value + 4 index)
#   8 x free x (ncv + 8)             the Lanczos vectors of ARPACK and a few work vectors
#   672 x tets x cases              the stress at 14 points of every tet (14 x 6 x 8 bytes) of every load case: verify keeps the
#                                    stress of all cases until the check starts, and frees a case when its eigen solve ends
#   200 MB                           the temporaries of the assembly of Kg in chunks
# Before meshing only the dofs are known. Then nnz(K) = BUCKLING_NNZ_PER_DOF x dofs, tets = BUCKLING_TETS_PER_DOF x dofs, and every dof
# is counted as free (the hook has 56 % free dofs). These ratios are one-host facts of P2 meshes of the hook at 3 sizes, not promises.
BUCKLING_NNZ_PER_DOF = 77.0      # nonzeros of K per dof: 72.4, 75.0 and 76.6 at 139k, 359k and 662k dofs
BUCKLING_TETS_PER_DOF = 0.21     # tets per dof: 0.184, 0.197 and 0.204
BUCKLING_STRESS_BYTES_PER_TET = 672
BUCKLING_TEMP_BYTES = 200e6
BUCKLING_BYTES_PER_ENTRY = 12


def buckling_bytes(dofs: float, modes: int = 6, *, free_dofs: float | None = None, nnz_k: float | None = None,
                   nnz_kff: float | None = None, tets: float | None = None, cases: int = 1) -> int:
    """The extra peak memory (bytes) of the linear buckling check (B3), on top of the peak of the factor.

    With `dofs` only (before meshing), `nnz_k`, `tets` come from the ratios above and all dofs are free. With the real
    `free_dofs`, `nnz_k`, `nnz_kff` and `tets` (after the matrices are built) the term is for this model. `cases` is the
    number of load cases: the stress of every case is kept until the check starts.
    """
    from ..buckling import NCV_MIN
    ncv = max(2 * int(modes) + 1, NCV_MIN)
    nnz_k = BUCKLING_NNZ_PER_DOF * dofs if nnz_k is None else nnz_k
    free = dofs if free_dofs is None else free_dofs
    nnz_kff = nnz_k * (free / dofs if dofs else 1.0) if nnz_kff is None else nnz_kff
    tets = BUCKLING_TETS_PER_DOF * dofs if tets is None else tets
    return int(BUCKLING_BYTES_PER_ENTRY * (nnz_k / 3.0 + 2.0 * nnz_kff / 3.0) + 8.0 * free * (ncv + 8)
               + BUCKLING_STRESS_BYTES_PER_TET * tets * max(int(cases), 1) + BUCKLING_TEMP_BYTES)


def rule_bytes(dofs: float, backend: str | None = None, c: float | None = None) -> float:
    """The rule of the pipeline: 7.5 GB per million dofs, in bytes. For `superlu`: its model (see the module text)."""
    if backend == "superlu":
        return GB * (SUPERLU_BASE_GB + SUPERLU_BASE_GB_PER_MDOF * dofs / 1e6 + superlu_fill_gb(dofs, c))
    return GB_PER_MDOF * GB * dofs / 1e6


def estimate_bytes(dofs: float, backend: str | None = None, c: float | None = None) -> int:
    """Peak memory estimate in bytes, the number that the guard compares with the limit.

    `7.5 GB x dofs / 1e6 x 1.25` for the factors with the pipeline rule. For `superlu`: the central value of its model,
    with no margin (the range of `memory_range_bytes` says how far the peak went on the measured shapes).
    """
    if backend == "superlu":
        return int(rule_bytes(dofs, backend, c))
    return int(rule_bytes(dofs, backend, c) * SAFETY)


def memory_range_bytes(dofs: float, backend: str | None, c: float | None = None):
    """`(low, central, high)` peak in bytes from the measured spread of the model, or None if `backend` has no model of its own.

    low = central / 1.45 and high = central / 0.60: the lowest and highest value of model / peak on 17 meshes of 5 shapes.
    """
    if backend != "superlu":
        return None
    mid = rule_bytes(dofs, backend, c)
    lo, hi = SUPERLU_MODEL_OVER_PEAK
    return (mid / hi, mid, mid / lo)


def int32_fact(dofs: float, c: float | None = None) -> str:
    """A fact when the modelled SuperLU fill is near the limit of 32-bit indices, else an empty text."""
    fill = superlu_fill_gb(dofs, c)
    if fill <= SUPERLU_INT32_FILL_GB:
        return ""
    return (f"The modelled SuperLU fill is {fill:.0f} GB. scipy SuperLU has 32-bit indices, so near 2^31 factor entries "
            f"(about 17 GB) more RAM may not help (not measured here).")


def range_note(dofs: float, backend: str | None, c: float | None, limit_bytes) -> str:
    """The fact for a run that passes the guard (central value below the limit) while the upper end of the range is above it."""
    r = memory_range_bytes(dofs, backend, c)
    if r is None or limit_bytes is None or not (r[1] <= limit_bytes < r[2]):
        return ""
    return (f"the upper end of the SuperLU memory range, {gb_text(r[2])}, is above the limit {gb_text(limit_bytes)}; the central "
            f"value, {gb_text(r[1])}, is below it, so the run starts. The peak was 0.69 to 1.67 times the central value on the "
            f"shapes that were measured; the model is uncertain for a shape that was not measured. If the run stops for memory (exit code 2), the options are a larger {opt('mesh_size')}, "
            f"pypardiso (Linux x86_64, extra 'pardiso') or cholespy (extra 'verify') with {opt('solver')} NAME")


def notes(dofs: float, backend: str | None, c: float | None, limit_bytes) -> list[str]:
    """The facts about an estimate that the user should see next to the number (empty for a factor with no model of its own)."""
    if backend != "superlu":
        return []
    out = [range_note(dofs, backend, c, limit_bytes)]
    fact = int32_fact(dofs, c)
    if fact:
        out.append(f"{fact} The options are a larger {opt('mesh_size')}, or pypardiso (Linux x86_64, extra 'pardiso') or "
                   f"cholespy (extra 'verify') with {opt('solver')} NAME.")
    return [t for t in out if t]


def model_text(backend: str | None = None, c: float | None = None) -> str:
    """The rule behind a memory estimate, in words (for the messages)."""
    if backend == "superlu":
        return (f"SuperLU model: 0.50 GB + 1.63 GB per million dofs + 65,000 GB x C^2 x (million dofs)^2 of fill, "
                f"C = {(HOOK_COMPACTNESS if c is None else c):.3f}")
    return "7.5 GB per million dofs"


def peak_rss_bytes() -> int:
    """Peak resident memory of this process in bytes (macOS: ru_maxrss is bytes, Linux: kilobytes)."""
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(v) if sys.platform == "darwin" else int(v) * 1024


def children_peak_rss_bytes() -> int:
    """Peak resident memory of the largest finished child process (the gmsh child), in bytes."""
    import resource
    v = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(v) if sys.platform == "darwin" else int(v) * 1024


def available_ram_bytes():
    """RAM that this process can use now, in bytes, or None if the system does not say.

    On Linux it is MemAvailable, or the room left in the cgroup limit if that is lower. On macOS it is
    free + inactive + speculative pages (`vm_stat`).
    """
    from ..environment import _ram
    return _ram()["available_bytes"]


def gb_text(nbytes) -> str:
    return f"{nbytes / GB:.2f} GB"


def limits(max_gb: float | None) -> dict:
    """The limits that apply: `{"available_bytes", "max_gb_bytes", "limit_bytes", "limit_source"}`.

    The limit is the smaller of the available RAM and `max_gb` (if given). `limit_bytes` is None when neither is known.
    """
    avail = available_ram_bytes()
    cap = None if max_gb is None else int(max_gb * GB)
    cands = [(v, s) for v, s in ((avail, "available RAM"), (cap, opt("max_gb"))) if v is not None]
    limit, source = min(cands) if cands else (None, None)
    return {"available_bytes": avail, "max_gb_bytes": cap, "limit_bytes": limit, "limit_source": source}
