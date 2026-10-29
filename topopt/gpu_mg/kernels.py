"""OpenCL C source of the gpu-mg solver: matrix-free hex8 multigrid with a PCG on the outside.

Ported from the G1b research code. The source is made by `source(case, nr)`. It is plain
OpenCL C 1.2 (no work-group functions of OpenCL 2.0), so it builds on Apple OpenCL.

Precision cases. RV is the type of the V-cycle, RO the type of the outer PCG:

  "C": RV = RO = float. The program source has no 64-bit float type and no extension pragma for it.
  "B": RV = float, RO = the 64-bit type. The source holds the pragma that the device needs.

`nr` is the number of right-hand sides (columns) that one kernel launch handles. The columns
are stored one after the other with the stride n3 = 3 * nn of the fine level.

Algorithm of the V-cycle: Chebyshev smoother of degree 2 on 3x3 block Jacobi, interval
[lmax/4, lmax], V(1,1), rediscretized coarse levels, trilinear transfer, and a dense inverse on
the coarsest level (one work-group for each row). Fused kernels: one Chebyshev step with the
residual, the restriction with the first Chebyshev step of the next level, the p.Ap dot product
in the matrix-vector product, the r.r dot product in the update of x and r, and device-side
scalars (alpha, beta).

Every work-group reduction uses `wg_sum_*` (a tree in local memory). The local memory is a kernel
argument, so the host can pick the work-group size at run time. The work-group size must be a
power of two.

Springs and joints (P2b). `source(case, nr, border=...)` with a border makes a program with the macro `BORDER`. The four kernels that apply
the operator of a level (`resid_v`, `cheb`, `cheb_dot`, `spmv_dot_o`) then take one more argument, the vector `wb`, and add `wb` to the
result of the element gather at the nodes that carry bit 8 of the fix mask: `wb = (Kb - C D^-1 C^T) x` of the level, made by `bord_w_*`
(and `bord_t_*`) just before the launch of the kernel that reads it. Without a border the program has no `BORDER` and the kernels are the
kernels of the solver before P2b. `NK` is the number of border dofs of the joints (6 for each joint, 0 for springs only).

Deflation of the rigid modes (P2d). `source(..., defl_dk=DK)` appends three kernels (`defl_dot`, `defl_coef`, `defl_apply`) to the program of case C
with joints, and nothing else changes: the text before them is the text without `defl_dk`. `DK` is the number of columns of `Z` (`deflation.py`).
Case B and a program without joints never get them.
"""
from __future__ import annotations

import numpy as np

from ..fe import LOCAL_NODES

CASES = ("B", "C")
TYPES = {"B": ("float", "double"), "C": ("float", "float")}   # (RV, RO)

# The names of all kernels of the program. The host queries the work-group limit of each.
KERNEL_NAMES = ("resid_v", "spmv_dot_o", "cheb", "cheb_dot", "restrict_k", "restrict_c0", "prolong_add",
                "gemv", "dotrr", "scal_rr", "p_update", "upd_xr", "init_k")

# The kernels of the support terms (P2b), made only for a program with a border. `_v`: type RV (the V-cycle), `_o`: type RO (the outer
# operator of case B; case C uses the `_v` kernels for it).
BORDER_KERNEL_NAMES = ("bord_w_v", "bord_w_o", "bord_t_v", "bord_t_o")

# The kernels of the deflation of the rigid modes (P2d), made only for case C with joints and `defl_dk` > 0.
DEFL_KERNEL_NAMES = ("defl_dot", "defl_coef", "defl_apply")

HEADER = r"""
@PRAGMA@
typedef @RV@ RV;
typedef @RO@ RO;
#define NR @NR@

// sum of v over the work-group; the result is in every work-item. W (local size) is a power of two.
RV wg_sum_v(RV v, __local RV* scr)
{
  const int lid = get_local_id(0);
  scr[lid] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = (int)get_local_size(0) >> 1; s > 0; s >>= 1) {
    if (lid < s) scr[lid] += scr[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const RV t = scr[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

RO wg_sum_o(RO v, __local RO* scr)
{
  const int lid = get_local_id(0);
  scr[lid] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = (int)get_local_size(0) >> 1; s > 0; s >>= 1) {
    if (lid < s) scr[lid] += scr[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const RO t = scr[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}
"""


def _pairs():
    ln = LOCAL_NODES.astype(int)
    pairs = {}
    for a in range(8):
        for b in range(8):
            o = ln[b] - ln[a]
            q = int((o[0] + 1) * 9 + (o[1] + 1) * 3 + (o[2] + 1))
            pairs.setdefault(q, []).append((a, b))
    return pairs, ln


def gen_gather_stencil(T, S, nr):
    """Stencil form (used for the 32-bit type): loop over the 27 neighbour nodes. The x of each
    neighbour is loaded once and used for all (element, local node) pairs that it belongs to."""
    pairs, _ = _pairs()
    o = [f"static void gather_{S}(const int n, __global const {T}* x, const int n3, const int nn, "
         f"__global const int* nb27, __global const int* nelem, __global const int* enodes, "
         f"__global const {T}* Ee, __constant {T}* kel, __private {T}* s)\n{{\n"
         f"  {T} t[8][{3 * nr}];\n  #pragma unroll\n  for (int a = 0; a < 8; a++) {{\n"
         f"    #pragma unroll\n    for (int i = 0; i < {3 * nr}; i++) t[a][i] = 0;\n  }}\n"]
    for q in range(27):
        o.append(f"  {{ const int m = nb27[{q} * nn + n]; if (m >= 0) {{\n")
        for c in range(nr):
            o.append(f"    const {T} u{c}0 = x[{c} * n3 + 3 * m], u{c}1 = x[{c} * n3 + 3 * m + 1], "
                     f"u{c}2 = x[{c} * n3 + 3 * m + 2];\n")
        for (a, b) in pairs[q]:
            base = (a * 8 + b) * 9
            for c in range(nr):
                for r in range(3):
                    o.append(f"    t[{a}][{3 * c + r}] += kel[{base + 3 * r}] * u{c}0 + "
                             f"kel[{base + 3 * r + 1}] * u{c}1 + kel[{base + 3 * r + 2}] * u{c}2;\n")
        o.append("  } }\n")
    o.append(f"  #pragma unroll\n  for (int i = 0; i < {3 * nr}; i++) s[i] = 0;\n")
    for a in range(8):
        o.append(f"  {{ const int e = nelem[n * 8 + {a}]; if (e >= 0) {{ const {T} E = Ee[e];\n    #pragma unroll\n"
                 f"    for (int i = 0; i < {3 * nr}; i++) s[i] += E * t[{a}][i]; }} }}\n")
    o.append("}\n")
    return "".join(o)


GATHER_ELEM = r"""
static void gather_@S@(const int n, __global const @T@* x, const int n3, const int nn, __global const int* nb27,
                       __global const int* nelem, __global const int* enodes, __global const @T@* Ee, __constant @T@* kel, __private @T@* s)
{
  #pragma unroll
  for (int i = 0; i < 3 * NR; i++) s[i] = 0;
  for (int a = 0; a < 8; a++) {
    const int e = nelem[n * 8 + a];
    if (e < 0) continue;
    @T@ t[3 * NR];
    #pragma unroll
    for (int i = 0; i < 3 * NR; i++) t[i] = 0;
    #pragma unroll
    for (int b = 0; b < 8; b++) {
      const int m = enodes[e * 8 + b];
      __constant @T@* k = kel + (a * 8 + b) * 9;
      const @T@ k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4], k5 = k[5], k6 = k[6], k7 = k[7], k8 = k[8];
      #pragma unroll
      for (int c = 0; c < NR; c++) {
        const @T@ u0 = x[c * n3 + 3 * m], u1 = x[c * n3 + 3 * m + 1], u2 = x[c * n3 + 3 * m + 2];
        t[3 * c]     += k0 * u0 + k1 * u1 + k2 * u2;
        t[3 * c + 1] += k3 * u0 + k4 * u1 + k5 * u2;
        t[3 * c + 2] += k6 * u0 + k7 * u1 + k8 * u2;
      }
    }
    const @T@ E = Ee[e];
    #pragma unroll
    for (int i = 0; i < 3 * NR; i++) s[i] += E * t[i];
  }
}
"""

# r = b - A x (Dirichlet dofs zero), V-cycle type
RESID = r"""
__kernel void resid_v(__global const RV* b, __global const RV* x, __global RV* r, __global const int* nb27, __global const int* nelem,
                      __global const int* enodes, __global const RV* Ee, __constant RV* kel,
                      __global const uchar* fix, const int nn, const int n3
#ifdef BORDER
                      , __global const RV* wb
#endif
                      )
{
  const int n = get_global_id(0);
  if (n >= nn) return;
  RV s[3 * NR];
  gather_v(n, x, n3, nn, nb27, nelem, enodes, Ee, kel, s);
  const uchar fm = fix[n];
#ifdef BORDER
  if (fm & 8) {
    #pragma unroll
    for (int c = 0; c < NR; c++) { s[3 * c] += wb[c * n3 + 3 * n]; s[3 * c + 1] += wb[c * n3 + 3 * n + 1]; s[3 * c + 2] += wb[c * n3 + 3 * n + 2]; }
  }
#endif
  #pragma unroll
  for (int c = 0; c < NR; c++) {
    r[c * n3 + 3 * n]     = (fm & 1) ? 0 : b[c * n3 + 3 * n] - s[3 * c];
    r[c * n3 + 3 * n + 1] = (fm & 2) ? 0 : b[c * n3 + 3 * n + 1] - s[3 * c + 1];
    r[c * n3 + 3 * n + 2] = (fm & 4) ? 0 : b[c * n3 + 3 * n + 2] - s[3 * c + 2];
  }
}
"""

# y = A x and per work-group partial sums of x.y (one for each column): pa[c * ngroups + group]. Outer type.
SPMV_DOT = r"""
__kernel void spmv_dot_o(__global const RO* x, __global RO* y, __global const int* nb27, __global const int* nelem, __global const int* enodes,
                         __global const RO* Ee, __constant RO* kel, __global const uchar* fix, const int nn, const int n3,
                         __global RO* pa, __local RO* scr
#ifdef BORDER
                         , __global const RO* wb
#endif
                         )
{
  const int n = get_global_id(0);
  RO dd[NR];
  #pragma unroll
  for (int c = 0; c < NR; c++) dd[c] = 0;
  if (n < nn) {
    RO s[3 * NR];
    gather_o(n, x, n3, nn, nb27, nelem, enodes, Ee, kel, s);
    const uchar fm = fix[n];
#ifdef BORDER
    if (fm & 8) {
      #pragma unroll
      for (int c = 0; c < NR; c++) { s[3 * c] += wb[c * n3 + 3 * n]; s[3 * c + 1] += wb[c * n3 + 3 * n + 1]; s[3 * c + 2] += wb[c * n3 + 3 * n + 2]; }
    }
#endif
    #pragma unroll
    for (int c = 0; c < NR; c++) {
      const RO y0 = (fm & 1) ? 0 : s[3 * c], y1 = (fm & 2) ? 0 : s[3 * c + 1], y2 = (fm & 4) ? 0 : s[3 * c + 2];
      y[c * n3 + 3 * n] = y0; y[c * n3 + 3 * n + 1] = y1; y[c * n3 + 3 * n + 2] = y2;
      dd[c] = x[c * n3 + 3 * n] * y0 + x[c * n3 + 3 * n + 1] * y1 + x[c * n3 + 3 * n + 2] * y2;
    }
  }
  #pragma unroll
  for (int c = 0; c < NR; c++) {
    const RO tot = wg_sum_o(dd[c], scr);
    if (get_local_id(0) == 0) pa[c * get_num_groups(0) + get_group_id(0)] = tot;
  }
}
"""

# V-cycle kernels (type RV)
VCYCLE = r"""
// one Chebyshev step: r = b - A x ; d = c1 d + c2 Dinv r ; xo = x + d
__kernel void cheb(__global const RV* b, __global const RV* x, __global RV* xo, __global RV* d, __global const RV* dinv,
                   __global const int* nb27, __global const int* nelem, __global const int* enodes, __global const RV* Ee, __constant RV* kel,
                   __global const uchar* fix, const int nn, const int n3, const RV c1, const RV c2
#ifdef BORDER
                   , __global const RV* wb
#endif
                   )
{
  const int n = get_global_id(0);
  if (n >= nn) return;
  RV s[3 * NR];
  gather_v(n, x, n3, nn, nb27, nelem, enodes, Ee, kel, s);
  const uchar fm = fix[n];
#ifdef BORDER
  if (fm & 8) {
    #pragma unroll
    for (int c = 0; c < NR; c++) { s[3 * c] += wb[c * n3 + 3 * n]; s[3 * c + 1] += wb[c * n3 + 3 * n + 1]; s[3 * c + 2] += wb[c * n3 + 3 * n + 2]; }
  }
#endif
  __global const RV* dv = dinv + 6 * n;
  const RV a0 = dv[0], a1 = dv[1], a2 = dv[2], a3 = dv[3], a4 = dv[4], a5 = dv[5];
  #pragma unroll
  for (int c = 0; c < NR; c++) {
    const int o = c * n3 + 3 * n;
    const RV r0 = (fm & 1) ? 0 : b[o] - s[3 * c];
    const RV r1 = (fm & 2) ? 0 : b[o + 1] - s[3 * c + 1];
    const RV r2 = (fm & 4) ? 0 : b[o + 2] - s[3 * c + 2];
    const RV z0 = a0 * r0 + a1 * r1 + a2 * r2;
    const RV z1 = a1 * r0 + a3 * r1 + a4 * r2;
    const RV z2 = a2 * r0 + a4 * r1 + a5 * r2;
    const RV e0 = (c1 != 0 ? c1 * d[o] : 0) + c2 * z0;
    const RV e1 = (c1 != 0 ? c1 * d[o + 1] : 0) + c2 * z1;
    const RV e2 = (c1 != 0 ? c1 * d[o + 2] : 0) + c2 * z2;
    d[o] = e0; d[o + 1] = e1; d[o + 2] = e2;
    xo[o] = x[o] + e0; xo[o + 1] = x[o + 1] + e1; xo[o + 2] = x[o + 2] + e2;
  }
}

// the same, and per work-group partial sums of r_o.z and r_o.zold for the flexible PCG (z = the new xo): pa[(k * NR + c) * ngroups + group]
__kernel void cheb_dot(__global const RV* b, __global const RV* x, __global RV* xo, __global RV* d, __global const RV* dinv,
                       __global const int* nb27, __global const int* nelem, __global const int* enodes, __global const RV* Ee, __constant RV* kel,
                       __global const uchar* fix, const int nn, const int n3, const RV c1, const RV c2,
                       __global const RO* ro, __global const RV* zo, __global RO* pa, __local RO* scr
#ifdef BORDER
                       , __global const RV* wb
#endif
                       )
{
  const int n = get_global_id(0);
  RO s0[NR], s1[NR];
  #pragma unroll
  for (int c = 0; c < NR; c++) { s0[c] = 0; s1[c] = 0; }
  if (n < nn) {
    RV s[3 * NR];
    gather_v(n, x, n3, nn, nb27, nelem, enodes, Ee, kel, s);
    const uchar fm = fix[n];
#ifdef BORDER
    if (fm & 8) {
      #pragma unroll
      for (int c = 0; c < NR; c++) { s[3 * c] += wb[c * n3 + 3 * n]; s[3 * c + 1] += wb[c * n3 + 3 * n + 1]; s[3 * c + 2] += wb[c * n3 + 3 * n + 2]; }
    }
#endif
    __global const RV* dv = dinv + 6 * n;
    const RV a0 = dv[0], a1 = dv[1], a2 = dv[2], a3 = dv[3], a4 = dv[4], a5 = dv[5];
    #pragma unroll
    for (int c = 0; c < NR; c++) {
      const int o = c * n3 + 3 * n;
      const RV r0 = (fm & 1) ? 0 : b[o] - s[3 * c];
      const RV r1 = (fm & 2) ? 0 : b[o + 1] - s[3 * c + 1];
      const RV r2 = (fm & 4) ? 0 : b[o + 2] - s[3 * c + 2];
      const RV z0 = a0 * r0 + a1 * r1 + a2 * r2;
      const RV z1 = a1 * r0 + a3 * r1 + a4 * r2;
      const RV z2 = a2 * r0 + a4 * r1 + a5 * r2;
      const RV e0 = (c1 != 0 ? c1 * d[o] : 0) + c2 * z0;
      const RV e1 = (c1 != 0 ? c1 * d[o + 1] : 0) + c2 * z1;
      const RV e2 = (c1 != 0 ? c1 * d[o + 2] : 0) + c2 * z2;
      d[o] = e0; d[o + 1] = e1; d[o + 2] = e2;
      const RV x0 = x[o] + e0, x1 = x[o + 1] + e1, x2 = x[o + 2] + e2;
      xo[o] = x0; xo[o + 1] = x1; xo[o + 2] = x2;
      const RO q0 = ro[o], q1 = ro[o + 1], q2 = ro[o + 2];
      s0[c] = q0 * (RO)x0 + q1 * (RO)x1 + q2 * (RO)x2;
      s1[c] = q0 * (RO)zo[o] + q1 * (RO)zo[o + 1] + q2 * (RO)zo[o + 2];
    }
  }
  const int ng = get_num_groups(0);
  #pragma unroll
  for (int c = 0; c < NR; c++) {
    const RO t0 = wg_sum_o(s0[c], scr);
    const RO t1 = wg_sum_o(s1[c], scr);
    if (get_local_id(0) == 0) { pa[(0 * NR + c) * ng + get_group_id(0)] = t0; pa[(1 * NR + c) * ng + get_group_id(0)] = t1; }
  }
}

__constant RV WR[27] = {@WR@};

// rc = P^T rf (gather, table rtab[27][nc])
__kernel void restrict_k(__global const RV* rf, __global RV* rc, __global const int* rtab, __global const uchar* fixc,
                         const int nc, const int n3c, const int n3f)
{
  const int n = get_global_id(0);
  if (n >= nc) return;
  RV s[3 * NR];
  #pragma unroll
  for (int i = 0; i < 3 * NR; i++) s[i] = 0;
  for (int q = 0; q < 27; q++) {
    const int f = rtab[q * nc + n];
    if (f < 0) continue;
    const RV w = WR[q];
    #pragma unroll
    for (int c = 0; c < NR; c++) {
      s[3 * c]     += w * rf[c * n3f + 3 * f];
      s[3 * c + 1] += w * rf[c * n3f + 3 * f + 1];
      s[3 * c + 2] += w * rf[c * n3f + 3 * f + 2];
    }
  }
  const uchar fm = fixc[n];
  #pragma unroll
  for (int c = 0; c < NR; c++) {
    rc[c * n3c + 3 * n]     = (fm & 1) ? 0 : s[3 * c];
    rc[c * n3c + 3 * n + 1] = (fm & 2) ? 0 : s[3 * c + 1];
    rc[c * n3c + 3 * n + 2] = (fm & 4) ? 0 : s[3 * c + 2];
  }
}

// restriction + the zero-guess Chebyshev step of the coarse level
__kernel void restrict_c0(__global const RV* rf, __global RV* rc, __global RV* xc, __global RV* dc, __global const RV* dinvc,
                          __global const int* rtab, __global const uchar* fixc, const int nc, const int n3c, const int n3f, const RV c2)
{
  const int n = get_global_id(0);
  if (n >= nc) return;
  RV s[3 * NR];
  #pragma unroll
  for (int i = 0; i < 3 * NR; i++) s[i] = 0;
  for (int q = 0; q < 27; q++) {
    const int f = rtab[q * nc + n];
    if (f < 0) continue;
    const RV w = WR[q];
    #pragma unroll
    for (int c = 0; c < NR; c++) {
      s[3 * c]     += w * rf[c * n3f + 3 * f];
      s[3 * c + 1] += w * rf[c * n3f + 3 * f + 1];
      s[3 * c + 2] += w * rf[c * n3f + 3 * f + 2];
    }
  }
  const uchar fm = fixc[n];
  __global const RV* dv = dinvc + 6 * n;
  const RV a0 = dv[0], a1 = dv[1], a2 = dv[2], a3 = dv[3], a4 = dv[4], a5 = dv[5];
  #pragma unroll
  for (int c = 0; c < NR; c++) {
    const int o = c * n3c + 3 * n;
    const RV r0 = (fm & 1) ? 0 : s[3 * c], r1 = (fm & 2) ? 0 : s[3 * c + 1], r2 = (fm & 4) ? 0 : s[3 * c + 2];
    rc[o] = r0; rc[o + 1] = r1; rc[o + 2] = r2;
    const RV e0 = c2 * (a0 * r0 + a1 * r1 + a2 * r2);
    const RV e1 = c2 * (a1 * r0 + a3 * r1 + a4 * r2);
    const RV e2 = c2 * (a2 * r0 + a4 * r1 + a5 * r2);
    dc[o] = e0; dc[o + 1] = e1; dc[o + 2] = e2; xc[o] = e0; xc[o + 1] = e1; xc[o + 2] = e2;
  }
}

// xf += P xc (table ptab[8][nf], weight pw[nf]); Dirichlet dofs of xf zeroed
__kernel void prolong_add(__global const RV* xc, __global RV* xf, __global const int* ptab, __global const RV* pw,
                          __global const uchar* fixf, const int nf, const int n3f, const int n3c)
{
  const int n = get_global_id(0);
  if (n >= nf) return;
  RV s[3 * NR];
  #pragma unroll
  for (int i = 0; i < 3 * NR; i++) s[i] = 0;
  for (int q = 0; q < 8; q++) {
    const int cn = ptab[q * nf + n];
    if (cn < 0) break;
    #pragma unroll
    for (int c = 0; c < NR; c++) {
      s[3 * c]     += xc[c * n3c + 3 * cn];
      s[3 * c + 1] += xc[c * n3c + 3 * cn + 1];
      s[3 * c + 2] += xc[c * n3c + 3 * cn + 2];
    }
  }
  const RV w = pw[n];
  const uchar fm = fixf[n];
  #pragma unroll
  for (int c = 0; c < NR; c++) {
    const int o = c * n3f + 3 * n;
    xf[o]     = (fm & 1) ? 0 : xf[o] + w * s[3 * c];
    xf[o + 1] = (fm & 2) ? 0 : xf[o + 1] + w * s[3 * c + 1];
    xf[o + 2] = (fm & 4) ? 0 : xf[o + 2] + w * s[3 * c + 2];
  }
}

// coarsest level: x = Ainv b, Ainv symmetric (zero rows / columns on Dirichlet dofs). One work-group for each row.
__kernel void gemv(__global const RV* Ainv, __global const RV* b, __global RV* x, const int nd, __local RV* scr)
{
  const int i = get_group_id(0), lid = get_local_id(0), W = get_local_size(0);
  RV s[NR];
  #pragma unroll
  for (int c = 0; c < NR; c++) s[c] = 0;
  for (int j = lid; j < nd; j += W) {
    const RV a = Ainv[(long)i * nd + j];
    #pragma unroll
    for (int c = 0; c < NR; c++) s[c] += a * b[c * nd + j];
  }
  #pragma unroll
  for (int c = 0; c < NR; c++) {
    const RV t = wg_sum_v(s[c], scr);
    if (lid == 0) x[c * nd + i] = t;
  }
}
"""

# outer PCG kernels (type RO for the PCG vectors, RV for the V-cycle vectors z, zold, rv)
OUTER = r"""
__kernel void dotrr(__global const RO* r, __global RO* pa, const int n3, const int G, __local RO* scr)
{
  const int gid = get_group_id(0), c = gid / G, g = gid - c * G;
  const int lid = get_local_id(0), W = get_local_size(0);
  RO s0 = 0;
  for (int i = g * W + lid; i < n3; i += G * W) { const RO v = r[c * n3 + i]; s0 += v * v; }
  s0 = wg_sum_o(s0, scr);
  if (lid == 0) pa[c * G + g] = s0;
}

// sc slots (each NR long): 0 rzold, 1 beta, 2 alpha, 3 rr
__kernel void scal_rr(__global const RO* pa, __global RO* sc, const int G, __local RO* scr)
{
  const int c = get_group_id(0), lid = get_local_id(0), W = get_local_size(0);
  RO s0 = 0;
  for (int g = lid; g < G; g += W) s0 += pa[c * G + g];
  s0 = wg_sum_o(s0, scr);
  if (lid == 0) sc[3 * NR + c] = s0;
}

// beta from the partials of cheb_dot (NG1 groups); p = z + beta p ; zold = z.  grid: NR * G groups, grid-stride over the column
__kernel void p_update(__global const RV* z, __global RO* p, __global RV* zo, __global RO* sc, __global const RO* pa,
                       const int n3, const int G, const int NG1, const int flex, __local RO* scr)
{
  const int gid = get_group_id(0), c = gid / G, g = gid - c * G;
  const int lid = get_local_id(0), W = get_local_size(0);
  RO s0 = 0, s1 = 0;
  for (int k = lid; k < NG1; k += W) { s0 += pa[(0 * NR + c) * NG1 + k]; s1 += pa[(1 * NR + c) * NG1 + k]; }
  s0 = wg_sum_o(s0, scr);
  s1 = wg_sum_o(s1, scr);
  const RO rzold = sc[0 * NR + c];
  const RO beta = flex ? (s0 - s1) / rzold : s0 / rzold;
  for (int i = g * W + lid; i < n3; i += G * W) {
    const RV zz = z[c * n3 + i];
    p[c * n3 + i] = (RO)zz + beta * p[c * n3 + i];
    zo[c * n3 + i] = zz;
  }
}

// alpha = rz / pap (partials NG1 of cheb_dot, NG2 of spmv_dot); x += alpha p ; r -= alpha Ap ; rv = (RV) r ; partial r.r ;
// first (zero-guess) Chebyshev step of level 0 on rv:  xv = dv = c2 Dinv rv.  grid: NR * G groups, grid-stride over the nodes
__kernel void upd_xr(__global RO* x, __global RO* r, __global const RO* p, __global const RO* Ap, __global RV* rv,
                     __global RV* xv, __global RV* dv, __global const RV* dinv, __global RO* sc, __global const RO* P1, __global const RO* P2,
                     __global RO* P3, const int nn, const int n3, const int G, const int NG1, const int NG2, const RV c2, __local RO* scr)
{
  const int gid = get_group_id(0), c = gid / G, g = gid - c * G;
  const int lid = get_local_id(0), W = get_local_size(0);
  RO s0 = 0, s1 = 0;
  for (int k = lid; k < NG1; k += W) s0 += P1[(0 * NR + c) * NG1 + k];
  for (int k = lid; k < NG2; k += W) s1 += P2[c * NG2 + k];
  s0 = wg_sum_o(s0, scr);
  s1 = wg_sum_o(s1, scr);
  const RO alpha = s0 / s1;
  if (g == 0 && lid == 0) { sc[0 * NR + c] = s0; sc[2 * NR + c] = alpha; }
  RO rr = 0;
  for (int n = g * W + lid; n < nn; n += G * W) {
    const int o = c * n3 + 3 * n;
    const RO r0 = r[o] - alpha * Ap[o], r1 = r[o + 1] - alpha * Ap[o + 1], r2 = r[o + 2] - alpha * Ap[o + 2];
    x[o] += alpha * p[o]; x[o + 1] += alpha * p[o + 1]; x[o + 2] += alpha * p[o + 2];
    r[o] = r0; r[o + 1] = r1; r[o + 2] = r2;
    rr += r0 * r0 + r1 * r1 + r2 * r2;
    const RV v0 = (RV)r0, v1 = (RV)r1, v2 = (RV)r2;
#ifndef SAME_RV
    rv[o] = v0; rv[o + 1] = v1; rv[o + 2] = v2;
#endif
    __global const RV* dd = dinv + 6 * n;
    const RV e0 = c2 * (dd[0] * v0 + dd[1] * v1 + dd[2] * v2);
    const RV e1 = c2 * (dd[1] * v0 + dd[3] * v1 + dd[4] * v2);
    const RV e2 = c2 * (dd[2] * v0 + dd[4] * v1 + dd[5] * v2);
    xv[o] = e0; xv[o + 1] = e1; xv[o + 2] = e2; dv[o] = e0; dv[o + 1] = e1; dv[o + 2] = e2;
  }
  rr = wg_sum_o(rr, scr);
  if (lid == 0) P3[c * G + g] = rr;
}

// start: r = b ; rv = (RV) r ; x = p = zold = 0 ; first Chebyshev step of level 0.  grid: nn nodes
__kernel void init_k(__global const RO* b, __global RO* r, __global RV* rv, __global RO* x, __global RO* p, __global RV* zo,
                     __global RV* xv, __global RV* dv, __global const RV* dinv, const int nn, const int n3, const RV c2)
{
  const int n = get_global_id(0);
  if (n >= nn) return;
  __global const RV* dd = dinv + 6 * n;
  #pragma unroll
  for (int c = 0; c < NR; c++) {
    const int o = c * n3 + 3 * n;
    const RO r0 = b[o], r1 = b[o + 1], r2 = b[o + 2];
    r[o] = r0; r[o + 1] = r1; r[o + 2] = r2;
    const RV v0 = (RV)r0, v1 = (RV)r1, v2 = (RV)r2;
#ifndef SAME_RV
    rv[o] = v0; rv[o + 1] = v1; rv[o + 2] = v2;
#endif
    x[o] = 0; x[o + 1] = 0; x[o + 2] = 0; p[o] = 0; p[o + 1] = 0; p[o + 2] = 0; zo[o] = 0; zo[o + 1] = 0; zo[o + 2] = 0;
    const RV e0 = c2 * (dd[0] * v0 + dd[1] * v1 + dd[2] * v2);
    const RV e1 = c2 * (dd[1] * v0 + dd[3] * v1 + dd[4] * v2);
    const RV e2 = c2 * (dd[2] * v0 + dd[4] * v1 + dd[5] * v2);
    xv[o] = e0; xv[o + 1] = e1; xv[o + 2] = e2; dv[o] = e0; dv[o + 1] = e1; dv[o + 2] = e2;
  }
}
"""


# The support terms (P2b): `w = (Kb - C D^-1 C^T) x` at the bonded nodes of a level, written to `w` (the vector layout of the level, the other
# nodes are not touched). `bn` (nb): the bonded nodes, `kptr` (nb + 1), `kcol` (the node of each block), `kval` (nnz, 9): the 3 x 3 blocks of Kb, row
# major. With joints (NK > 0): `cb` (nb, 3, NK) is C, `di` (NK, NK) is D^-1, and `t` (NR, NK) is C^T x from `bord_t`; each work-group first makes
# `q = D^-1 t` in local memory (`qs`, NR * NK numbers) and every node then takes `- C q`.
BORDER_KERNEL = r"""
#if NK > 0
// t[c * NK + j] = sum over the bonded nodes of (C^T x)_j. One work-group for each border dof j; a power-of-two local size.
__kernel void bord_t_@S@(__global const @T@* x, __global const int* bn, __global const @T@* cb, __global @T@* t, const int nb, const int n3,
                         __local @T@* scr)
{
  const int j = get_group_id(0), lid = get_local_id(0), W = get_local_size(0);
  @T@ s[NR];
  #pragma unroll
  for (int c = 0; c < NR; c++) s[c] = 0;
  for (int i = lid; i < nb; i += W) {
    const int n = bn[i];
    const @T@ c0 = cb[(3 * i) * NK + j], c1 = cb[(3 * i + 1) * NK + j], c2 = cb[(3 * i + 2) * NK + j];
    #pragma unroll
    for (int c = 0; c < NR; c++) s[c] += c0 * x[c * n3 + 3 * n] + c1 * x[c * n3 + 3 * n + 1] + c2 * x[c * n3 + 3 * n + 2];
  }
  #pragma unroll
  for (int c = 0; c < NR; c++) {
    const @T@ tot = wg_sum_@S@(s[c], scr);
    if (lid == 0) t[c * NK + j] = tot;
  }
}
#endif

__kernel void bord_w_@S@(__global const @T@* x, __global @T@* w, __global const int* bn, __global const int* kptr, __global const int* kcol,
                         __global const @T@* kval, __global const @T@* cb, __global const @T@* t, __global const @T@* di, const int nb, const int n3,
                         __local @T@* qs)
{
  const int i = get_global_id(0);
#if NK > 0
  for (int a = get_local_id(0); a < NR * NK; a += get_local_size(0)) {
    const int c = a / NK, r = a - c * NK;
    @T@ q = 0;
    for (int b = 0; b < NK; b++) q += di[r * NK + b] * t[c * NK + b];
    qs[a] = q;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
  if (i >= nb) return;
  @T@ s[3 * NR];
  #pragma unroll
  for (int k = 0; k < 3 * NR; k++) s[k] = 0;
  for (int p = kptr[i]; p < kptr[i + 1]; p++) {
    const int m = kcol[p];
    __global const @T@* k = kval + 9 * p;
    #pragma unroll
    for (int c = 0; c < NR; c++) {
      const @T@ u0 = x[c * n3 + 3 * m], u1 = x[c * n3 + 3 * m + 1], u2 = x[c * n3 + 3 * m + 2];
      s[3 * c]     += k[0] * u0 + k[1] * u1 + k[2] * u2;
      s[3 * c + 1] += k[3] * u0 + k[4] * u1 + k[5] * u2;
      s[3 * c + 2] += k[6] * u0 + k[7] * u1 + k[8] * u2;
    }
  }
#if NK > 0
  for (int r = 0; r < 3; r++) {
    #pragma unroll
    for (int c = 0; c < NR; c++) {
      @T@ acc = 0;
      for (int j = 0; j < NK; j++) acc += cb[(3 * i + r) * NK + j] * qs[c * NK + j];
      s[3 * c + r] -= acc;
    }
  }
#endif
  const int n = bn[i];
  #pragma unroll
  for (int c = 0; c < NR; c++) { w[c * n3 + 3 * n] = s[3 * c]; w[c * n3 + 3 * n + 1] = s[3 * c + 1]; w[c * n3 + 3 * n + 2] = s[3 * c + 2]; }
}
"""

# Deflation of the rigid modes in the inner PCG (P2d, "D3"). `Z` and `AZ = A Z` are DK vectors of the n3 dofs of level 0 (`zm[j * n3 + i]`,
# `az[j * n3 + i]`), `einv` is `E^-1` (DK x DK, row major), all in the outer type. The output `z` of the V-cycle is projected in place so that it is
# A-orthogonal to `Z`:  c = (AZ)^T z,  g = E^-1 c,  z <- z - Z g.  Three launches, between the last Chebyshev launch of the V-cycle and `p_update`:
#   defl_dot    partial sums of c_j for each work-group. Grid: NR * G groups (G = the groups of `dotrr`); pd[(c * DK + j) * G + g]
#   defl_coef   the sums over the groups and the product with E^-1. Grid: NR groups, a power-of-two local size; gc[c * DK + i]
#   defl_apply  z <- z - Z gc, and the two partial dots r.z and r.zold that `cheb_dot` made from the unprojected z, made again for the projected z
#               (same layout `pa[(k * NR + c) * ngroups + group]`, same groups as `cheb_dot`), so `p_update` and `upd_xr` stay as they are.
# Only plain float arithmetic and the reduction of the other kernels (`wg_sum_o`): no 64-bit type and no feature the other kernels do not use.
DEFL_KERNEL = r"""
__kernel void defl_dot(__global const RV* z, __global const RO* az, __global RO* pd, const int n3, const int G, __local RO* scr)
{
  const int gid = get_group_id(0), c = gid / G, g = gid - c * G;
  const int lid = get_local_id(0), W = get_local_size(0);
  RO s[DK];
  for (int j = 0; j < DK; j++) s[j] = 0;
  for (int i = g * W + lid; i < n3; i += G * W) {
    const RO zz = (RO)z[c * n3 + i];
    for (int j = 0; j < DK; j++) s[j] += az[j * n3 + i] * zz;
  }
  for (int j = 0; j < DK; j++) {
    const RO t = wg_sum_o(s[j], scr);
    if (lid == 0) pd[(c * DK + j) * G + g] = t;
  }
}

__kernel void defl_coef(__global const RO* pd, __global const RO* einv, __global RO* gc, const int G, __local RO* scr)
{
  const int c = get_group_id(0), lid = get_local_id(0), W = get_local_size(0);
  RO s[DK];
  for (int j = 0; j < DK; j++) {
    RO t = 0;
    for (int g = lid; g < G; g += W) t += pd[(c * DK + j) * G + g];
    s[j] = wg_sum_o(t, scr);
  }
  for (int i = lid; i < DK; i += W) {
    RO a = 0;
    for (int j = 0; j < DK; j++) a += einv[i * DK + j] * s[j];
    gc[c * DK + i] = a;
  }
}

__kernel void defl_apply(__global RV* z, __global const RO* zm, __global const RO* gc, __global const RO* ro, __global const RV* zo,
                         __global RO* pa, const int nn, const int n3, __local RO* scr)
{
  const int n = get_global_id(0);
  RO s0[NR], s1[NR];
  #pragma unroll
  for (int c = 0; c < NR; c++) { s0[c] = 0; s1[c] = 0; }
  if (n < nn) {
    for (int c = 0; c < NR; c++) {
      RO g[DK];
      for (int j = 0; j < DK; j++) g[j] = gc[c * DK + j];
      for (int a = 0; a < 3; a++) {
        const int o = c * n3 + 3 * n + a;
        RO v = (RO)z[o];
        for (int j = 0; j < DK; j++) v -= g[j] * zm[j * n3 + 3 * n + a];
        z[o] = (RV)v;
        s0[c] += ro[o] * (RO)((RV)v);
        s1[c] += ro[o] * (RO)zo[o];
      }
    }
  }
  const int ng = get_num_groups(0);
  for (int c = 0; c < NR; c++) {
    const RO t0 = wg_sum_o(s0[c], scr);
    const RO t1 = wg_sum_o(s1[c], scr);
    if (get_local_id(0) == 0) { pa[(0 * NR + c) * ng + get_group_id(0)] = t0; pa[(1 * NR + c) * ng + get_group_id(0)] = t1; }
  }
}
"""


def build_options(case: str) -> str:
    """Compiler options of the program. OpenCL C 1.2 is the level that Apple supports."""
    opts = "-cl-std=CL1.2 -cl-mad-enable"
    if case == "C":
        opts += " -DSAME_RV"      # RV and RO are the same type: the V-cycle reads the residual of the PCG
    return opts


def source(case: str, nr: int, border: bool = False, nk: int = 0, defl_dk: int = 0) -> str:
    """The OpenCL C source of the program for precision case `case` ("B" or "C") and `nr` columns.

    `border` (P2b): the program of a run with springs or joints. It defines `BORDER` (the operator kernels add the support terms) and holds the
    kernels `bord_w_*` that make them. Case B has them for both types (`_v`, `_o`), case C for the one type (`_v`). `nk`: the number of border
    dofs of the joints (6 for each joint); with `nk` > 0 the kernels `bord_t_*` (`C^T x`) are in the program too.

    `defl_dk` (P2d) > 0: the program of case C with joints (`nk` > 0) gets the three deflation kernels for `defl_dk` columns of `Z`, appended after all
    other kernels (the text before them is the text with `defl_dk` 0). Any other program with `defl_dk` > 0 is an error.
    """
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}, got {case!r}")
    if defl_dk and not (case == "C" and border and nk > 0 and defl_dk > 0):
        raise ValueError(f"the deflation kernels are for case C with joints (case {case!r}, border {border}, nk {nk}, defl_dk {defl_dk})")
    rv, ro = TYPES[case]
    pragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" if case == "B" else ""
    head = (HEADER.replace("@PRAGMA@", pragma).replace("@RV@", rv).replace("@RO@", ro)
            .replace("@NR@", str(nr)))
    if border:
        head += f"\n#define BORDER\n#define NK {int(nk)}\n"
    wr = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                nz = (di != 0) + (dj != 0) + (dk != 0)
                wr.append(repr(0.5 ** nz) + ("f" if rv == "float" else ""))
    parts = [head]
    # The V-cycle type is always the 32-bit type: stencil form. The outer type is in stencil form
    # for case C and in element-loop form for case B.
    parts.append(gen_gather_stencil("RV", "v", nr))
    if ro == "float":
        parts.append(gen_gather_stencil("RO", "o", nr))
    else:
        parts.append(GATHER_ELEM.replace("@T@", "RO").replace("@S@", "o"))
    parts += [RESID, SPMV_DOT, VCYCLE.replace("@WR@", ", ".join(wr)), OUTER]
    if border:
        parts.append(BORDER_KERNEL.replace("@T@", "RV").replace("@S@", "v"))
        if ro != rv:
            parts.append(BORDER_KERNEL.replace("@T@", "RO").replace("@S@", "o"))
    if defl_dk:
        parts.append(f"\n#define DK {int(defl_dk)}\n" + DEFL_KERNEL)
    return "".join(parts)


def kernel_dtype(case: str):
    """numpy types (V-cycle, outer) of a case."""
    return (np.float32, np.float32) if case == "C" else (np.float32, np.float64)
