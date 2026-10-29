"""The OpenCL engine of gpu-mg: device buffers, the V-cycle, the PCG and the FP64 refinement.

Ported from the G1b research code (`GpuMG`). This module imports pyopencl at the top, so import it
only when the solver `gpu-mg` runs. `topopt.gpu_mg.device` is the light module for facts and errors.

Cases (see kernels.py):
  "C"  all 32-bit on the device. The solve is an FP64 residual refinement on the CPU around an
       inner PCG on the device (`solve_refined`). Needs no 64-bit float on the device.
  "B"  32-bit V-cycle, 64-bit outer PCG on the device. Needs a device with 64-bit floats.

Deflation (P2d). In case C with joints the engine can project the output of the V-cycle A-orthogonal to the columns of `Z` (the rigid modes of the bodies
of the part and, since D3h, the harmonic columns of the bonded patches; `deflation.Deflation`, kernels `defl_dot`, `defl_coef`, `defl_apply`) and `pcg`
makes the coarse correction of these columns in FP64 around the device loop. The harmonic columns depend on the design: the model makes them with
`precondition` (the V-cycle alone) in an FP64 PCG on the host, and `upload_deflation` copies `Z`, `AZ` and `E^-1` to the device for each design. A
requested deflation that the device cannot run is not used: the engine then is the engine of before (the same launches), and `defl_off` holds the reason
as a text.

Recorded defaults (not user settings): the work-group size comes from the device and from the
kernel query (`WORK_GROUP_SIZE` of each kernel); `WG_CAP` and `WGV_CAP` are only upper bounds for
the search. `SYNC_EVERY = 0` means no `finish` inside a solve and a speculative check of the
residual one iteration ahead (the mode of the G1b timings).
"""
from __future__ import annotations

import time

import numpy as np
import pyopencl as cl

from . import kernels
from .hierarchy import HostMG
from .transfer import build_transfer_tables      # the tables are host code (P2b: the springs and joints build the same transfer as a matrix)

WG_CAP = 128      # upper bound for the work-group size of the node kernels (stencil, register heavy)
WGV_CAP = 256     # upper bound for the work-group size of the vector and reduction kernels
SYNC_EVERY = 0    # 0: no finish inside a solve. N > 0: finish after every N kernels (slower, for a diagnosis)
NR_BLOCK = 2      # columns (load cases) that one kernel launch handles
CHEB_ALPHA = 4.0  # Chebyshev interval is [lmax / CHEB_ALPHA, lmax]
BORDER_BIT = 8    # bit of the fix mask of a node that carries a support term (springs, joints) on its level (P2b)


def roundup(n: int, w: int) -> int:
    return ((n + w - 1) // w) * w


def pow2_floor(n: int) -> int:
    """The largest power of two that is at most `n` (at least 1)."""
    return 1 << (max(1, int(n)).bit_length() - 1)


class Op:
    """One kernel launch: the kernel with its arguments, the global and the local size."""
    __slots__ = ("name", "k", "gs", "ls")

    def __init__(self, name, k, gs, ls):
        self.name, self.k, self.gs, self.ls = name, k, (int(gs),), (int(ls),)


def build_nb27(L):
    """(27, nn) table: compact id of the neighbour node at offset (di, dj, dk) in {-1, 0, 1}^3 that
    shares an active element with the node (-1: none)."""
    pairs, ln = kernels._pairs()
    nb = -np.ones((27, L.nn), np.int32)
    for a in range(8):
        e = L.nelem[:, a]
        ok = e >= 0
        for b in range(8):
            o = ln[b] - ln[a]
            q = int((o[0] + 1) * 9 + (o[1] + 1) * 3 + (o[2] + 1))
            nb[q, ok] = L.enodes[e[ok], b]
    return nb


def fixmask(fix, bonded=None):
    """The uchar mask of each node: bits 0 to 2 are the Dirichlet axes, `BORDER_BIT` marks a node of `bonded` (P2b)."""
    m = (fix[:, 0] | (fix[:, 1] << 1) | (fix[:, 2] << 2)).astype(np.uint8)
    if bonded is not None and len(bonded):
        m[bonded] |= BORDER_BIT
    return m


class GpuMG:
    """Multigrid-PCG on an OpenCL device for one hierarchy topology.

    `host` is the `HostMG`. `device` is a pyopencl device. The numbers of a design (`Ee`, `dinv`,
    `lmax`, the coarse inverse) are uploaded with `upload_numerics` before each `pcg`.

    `deflation` (P2d): a `deflation.Deflation` of the hierarchy, for case C with joints only. The engine uses it when the device can run its kernels and
    holds its buffers; else `defl` is None and `defl_off` says why. With `deflation` None the engine is the engine of before P2d.
    """

    def __init__(self, host: HostMG, device, case="C", nr=NR_BLOCK, sync_every=SYNC_EVERY,
                 wg_cap=WG_CAP, wgv_cap=WGV_CAP, deflation=None):
        if case not in kernels.CASES:
            raise ValueError(f"case must be one of {kernels.CASES}, got {case!r}")
        self.case = case
        self.nr = nr
        self.sync_every = sync_every
        self.host = host
        self.border = getattr(host, "border", None)       # the support terms on every level (springs, joints), or None (P2b)
        self.levels = host.levels
        self.nlev = len(self.levels)
        self.RV, self.RO = kernels.kernel_dtype(case)
        self.same = self.RV is self.RO
        self.dev = device
        self.ctx = cl.Context([device])
        self.q = cl.CommandQueue(self.ctx)
        t0 = time.perf_counter()
        self.nk = 0 if self.border is None else int(self.border.k)           # border dofs of the joints (6 for each joint), 0 for springs only
        self.defl = None                # the deflation in use (P2d), or None
        self.defl_off = None            # why a requested deflation is not in use (a text), or None
        self.defl_requested = deflation
        self.dk = 0                     # the columns of Z in the program (0: the program has no deflation kernels)
        self.wd = 0                     # the work-group size of defl_dot and defl_coef
        self.z_uploaded = False         # Z depends on the topology only: it is copied to the device once
        self.Zd = self.AZd = self.Einv_d = self.PD = self.GC = None
        if deflation is not None:
            if case != "C" or not self.nk:
                raise ValueError("the deflation is for case C with joints")
            if deflation.dk > 0:
                self.dk = int(deflation.dk)
            else:
                self.defl_off = "no rigid mode is free: every body has all its dofs fixed (and no bonded patch can move)"
        self._build_program(nr)
        self.t_build = time.perf_counter() - t0
        self._choose_work_groups(wg_cap, wgv_cap)
        self.nbytes = 0
        self.nbytes_parts = {}
        self._alloc_topology()
        self._alloc_vectors()
        self._alloc_border()
        self._alloc_deflation()
        self.coef_targets = []
        self._lmax = None
        asm, pcg_ops = self._assemble(), self._pcg_ops()
        self._asm_ops = asm                                   # the V-cycle alone (`precondition`)
        self._ops_plain = asm + pcg_ops                       # the launches of one PCG iteration without deflation (the engine of before P2d)
        self._ops_defl = (asm + self._defl_ops() + pcg_ops) if self.dk and not self.defl_off else None
        self.iter_ops = self._ops_plain if self._ops_defl is None else self._ops_defl
        self.defl = None if self._ops_defl is None else deflation
        self.defl_active = self.defl is not None      # the model switches it for a design whose host numbers failed (`use_deflation`)
        self.t_coarse = 0.0             # host seconds of the coarse correction in `pcg`, summed over the calls

    def _build_program(self, nr):
        """Build the program. A program with the deflation kernels that does not build is built again without them (`defl_off` says why)."""
        border = self.border is not None
        opts = kernels.build_options(self.case)
        try:
            self.source = kernels.source(self.case, nr, border=border, nk=self.nk, defl_dk=self.dk)
            self.prg = cl.Program(self.ctx, self.source).build(options=opts)
        except cl.Error as e:
            if not self.dk:
                raise
            self.defl_off = (f"the OpenCL compiler could not build the deflation kernels for {self.dk} columns: "
                             + " ".join(str(e).split())[:300])
            self.dk = 0
            self.source = kernels.source(self.case, nr, border=border, nk=self.nk)
            self.prg = cl.Program(self.ctx, self.source).build(options=opts)

    # ---------------------------------------------------------------- work-group sizes
    def kernel(self, name):
        return cl.Kernel(self.prg, name)

    def kernel_limit(self, name) -> int:
        """The work-group size limit of a kernel on this device (a query of the built kernel)."""
        return int(self.kernel(name).get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, self.dev))

    def _choose_work_groups(self, wg_cap, wgv_cap):
        node = ("resid_v", "spmv_dot_o", "cheb", "cheb_dot", "restrict_k", "restrict_c0", "prolong_add", "init_k")
        if self.border is not None:
            node += ("bord_w_v",) + (() if self.same else ("bord_w_o",))
        vect = ("gemv", "dotrr", "scal_rr", "p_update", "upd_xr")
        if self.nk:
            vect += ("bord_t_v",) + (() if self.same else ("bord_t_o",))
        dev_max = int(self.dev.max_work_group_size)
        self.limits = {n: self.kernel_limit(n) for n in node + vect}
        self.wg = pow2_floor(min(wg_cap, dev_max, *(self.limits[n] for n in node)))
        self.wgv = pow2_floor(min(wgv_cap, dev_max, *(self.limits[n] for n in vect)))
        if self.dk and not self.defl_off:
            # P2d: the limits of the deflation kernels never lower `wg` or `wgv` (the engine without deflation has the sizes of before). `defl_apply` runs
            # with `wg` (its partial sums have the layout of `cheb_dot`); `defl_dot` and `defl_coef` run with `wd`, a power of two.
            self.limits.update({n: self.kernel_limit(n) for n in kernels.DEFL_KERNEL_NAMES})
            if self.limits["defl_apply"] < self.wg:
                self.defl_off = (f"the device allows work-group size {self.limits['defl_apply']} for the deflation kernel defl_apply, and the node kernels "
                                 f"use {self.wg}")
            else:
                self.wd = pow2_floor(min(self.wgv, dev_max, self.limits["defl_dot"], self.limits["defl_coef"]))

    def _local(self, itemsize, w):
        return cl.LocalMemory(int(itemsize) * int(w))

    # ---------------------------------------------------------------- buffers
    def buf(self, host=None, nbytes=None, flags=None, part="other"):
        mf = cl.mem_flags
        if host is not None:
            host = np.ascontiguousarray(host)
            b = cl.Buffer(self.ctx, (flags or mf.READ_WRITE) | mf.COPY_HOST_PTR, hostbuf=host)
            n = host.nbytes
        else:
            b = cl.Buffer(self.ctx, flags or mf.READ_WRITE, int(nbytes))
            n = int(nbytes)
        self.nbytes += n
        self.nbytes_parts[part] = self.nbytes_parts.get(part, 0) + n
        return b

    def _alloc_topology(self):
        L = self.levels
        outer_stencil = self.RO is np.float32
        self.nb27, self.nelem, self.enodes, self.fix = [], [], [], []
        self.ptab, self.pw, self.rtab = [None] * self.nlev, [None] * self.nlev, [None] * self.nlev
        for l, Lv in enumerate(L):
            need_en = (l == 0 and not outer_stencil)
            self.nelem.append(self.buf(Lv.nelem.astype(np.int32), part="topology"))
            nb = self.buf(build_nb27(Lv), part="topology")
            en = self.buf(Lv.enodes.astype(np.int32), part="topology") if need_en else nb  # dummy pointer if unused
            self.nb27.append(nb)
            self.enodes.append(en)
            self.fix.append(self.buf(fixmask(Lv.fix, None if self.border is None else self.border.levels[l].nodes), part="topology"))
        for l in range(self.nlev - 1):
            pt, pw, rt = build_transfer_tables(L[l], L[l + 1])
            self.ptab[l] = self.buf(pt, part="transfer")
            self.pw[l] = self.buf(pw.astype(self.RV), part="transfer")
            self.rtab[l + 1] = self.buf(rt, part="transfer")
        isv = np.dtype(self.RV).itemsize
        self.Ee, self.dinv = [], []
        for Lv in L:
            self.Ee.append(self.buf(nbytes=Lv.nel * isv, part="numerics"))
            self.dinv.append(self.buf(nbytes=Lv.nn * 6 * isv, part="numerics"))
        ke1 = L[0].kel
        mf = cl.mem_flags
        self.kel_v = self.buf(np.ascontiguousarray(ke1.reshape(-1), self.RV), flags=mf.READ_ONLY, part="numerics")
        if self.same:
            self.kel_o, self.Ee_o = self.kel_v, self.Ee[0]
        else:
            self.kel_o = self.buf(np.ascontiguousarray(ke1.reshape(-1), self.RO), flags=mf.READ_ONLY, part="numerics")
            self.Ee_o = self.buf(nbytes=L[0].nel * np.dtype(self.RO).itemsize, part="numerics")
        self.nd = L[-1].nn * 3
        self.Ainv = self.buf(nbytes=self.nd * self.nd * isv, part="numerics")

    def _alloc_vectors(self):
        nr = self.nr
        isv = np.dtype(self.RV).itemsize
        iso = np.dtype(self.RO).itemsize
        self.b, self.x, self.y, self.d, self.r = [], [], [], [], []
        for l, Lv in enumerate(self.levels):
            sz = nr * Lv.nn * 3 * isv
            self.b.append(self.buf(nbytes=sz, part="vectors") if (l > 0 or not self.same) else None)
            self.x.append(self.buf(nbytes=sz, part="vectors"))
            self.y.append(self.buf(nbytes=sz, part="vectors"))
            self.d.append(self.buf(nbytes=sz, part="vectors"))
            self.r.append(self.buf(nbytes=sz, part="vectors"))
        nn0 = self.levels[0].nn
        n3 = nn0 * 3
        self.n3 = n3
        szo = nr * n3 * iso
        self.b_o = self.buf(nbytes=szo, part="vectors")
        self.x_o = self.buf(nbytes=szo, part="vectors")
        self.r_o = self.buf(nbytes=szo, part="vectors")
        self.p_o = self.buf(nbytes=szo, part="vectors")
        self.Ap_o = self.buf(nbytes=szo, part="vectors")
        if self.same:
            self.b[0] = self.r_o
        self.zo = self.buf(nbytes=nr * n3 * isv, part="vectors")
        self.G = int(min(256, max(1, -(-nn0 // (self.wgv * 4)))))
        self.NG1 = roundup(nn0, self.wg) // self.wg      # groups of cheb_dot and of spmv_dot
        self.NG2 = self.NG1
        self.P1 = self.buf(nbytes=2 * nr * self.NG1 * iso, part="vectors")
        self.P2 = self.buf(nbytes=nr * self.NG2 * iso, part="vectors")
        self.P3 = self.buf(nbytes=nr * self.G * iso, part="vectors")
        self.sc = self.buf(nbytes=8 * nr * iso, part="vectors")
        self.hostsc = np.zeros(8 * nr, self.RO)

    def _alloc_border(self):
        """The support terms (P2b): for each level the table of the bonded nodes, the 3 x 3 blocks of `Kb`, the blocks of `C` (joints) and the vector
        `wb` that the operator kernels read. `D^-1` is one small matrix for all levels. They do not depend on the design, so they are uploaded
        once. A level with no bonded node has no table and gets the dummy buffer (no node of it carries `BORDER_BIT`, so the kernels do not read it)."""
        nl, nr, nk = self.nlev, self.nr, self.nk
        self.bnb = [0] * nl
        self.bn, self.kptr, self.kcol, self.kval, self.cb, self.wb = [None] * nl, [None] * nl, [None] * nl, [None] * nl, [None] * nl, [None] * nl
        self.kval_o, self.cb_o, self.wb_o, self.Di, self.Di_o, self.tb, self.tb_o = None, None, None, None, None, None, None
        if self.border is None:
            return
        isv, iso = np.dtype(self.RV).itemsize, np.dtype(self.RO).itemsize
        dummy = self.buf(nbytes=16, part="border")
        self.dummy = dummy
        if nk:
            Dinv = self.border.Dinv
            self.Di = self.buf(np.ascontiguousarray(Dinv, self.RV), part="border")
            self.tb = self.buf(nbytes=nr * nk * isv, part="border")
            if not self.same:
                self.Di_o = self.buf(np.ascontiguousarray(Dinv, self.RO), part="border")
                self.tb_o = self.buf(nbytes=nr * nk * iso, part="border")
        for l, Lv in enumerate(self.levels):
            nodes, kptr, kcol, kval, cb = self.border.device_tables(l)
            self.bnb[l] = int(nodes.size)
            if not nodes.size:
                self.wb[l] = dummy
                continue
            self.bn[l], self.kptr[l], self.kcol[l] = self.buf(nodes, part="border"), self.buf(kptr, part="border"), self.buf(kcol, part="border")
            self.kval[l] = self.buf(np.ascontiguousarray(kval.reshape(-1), self.RV), part="border")
            self.cb[l] = self.buf(np.ascontiguousarray(cb.reshape(-1), self.RV), part="border") if nk else dummy
            self.wb[l] = self.buf(nbytes=nr * Lv.nn * 3 * isv, part="border")
            if l == 0 and not self.same:       # case B: the outer operator of level 0 is in the 64-bit type
                self.kval_o = self.buf(np.ascontiguousarray(kval.reshape(-1), self.RO), part="border")
                self.cb_o = self.buf(np.ascontiguousarray(cb.reshape(-1), self.RO), part="border") if nk else dummy
                self.wb_o = self.buf(nbytes=nr * Lv.nn * 3 * iso, part="border")
        if self.same:
            self.kval_o, self.cb_o, self.wb_o, self.Di_o, self.tb_o = self.kval[0], self.cb[0], self.wb[0], self.Di, self.tb
        elif self.wb_o is None:
            self.wb_o = self.cb_o = dummy

    def max_alloc(self) -> int:
        """The size of the largest buffer that the device allows in bytes (0: the device gave no size)."""
        return int(getattr(self.dev, "max_mem_alloc_size", 0) or 0)

    def _alloc_deflation(self):
        """The buffers of the deflation (P2d), all in the 32-bit type: `Z` and `AZ` (`dk` vectors of the dofs of level 0, so 2 * dk * n3 * 4 bytes), `E^-1`
        (dk x dk), the partial sums (nr * dk * G) and the coefficients (nr * dk). A buffer that the device cannot make turns the deflation off (`defl_off`)."""
        d = self.defl_requested
        if d is None or not self.dk or self.defl_off:
            return
        iso = np.dtype(self.RO).itemsize
        vec = self.dk * self.n3 * iso
        cap = self.max_alloc()
        if cap and vec > cap:
            self.defl_off = (f"one deflation buffer ({self.dk} columns of {self.n3} dofs, {vec / 2**20:.0f} MB) is larger than the largest buffer of the "
                             f"device ({cap / 2**20:.0f} MB)")
            return
        n0, p0 = self.nbytes, self.nbytes_parts.get("defl", 0)
        made = []
        try:
            for name, size in (("Zd", vec), ("AZd", vec), ("Einv_d", self.dk * self.dk * iso), ("PD", self.nr * self.dk * self.G * iso),
                               ("GC", self.nr * self.dk * iso)):
                b = self.buf(nbytes=size, part="defl")
                made.append(b)
                setattr(self, name, b)
        except cl.Error as e:
            for b in made:
                b.release()
            self.nbytes, self.nbytes_parts["defl"] = n0, p0
            self.Zd = self.AZd = self.Einv_d = self.PD = self.GC = None
            self.defl_off = f"the device could not make the deflation buffers ({2 * vec / 2**20:.0f} MB for Z and AZ): " + " ".join(str(e).split())[:200]

    def _defl_ops(self):
        """The three launches of the deflation (P2d), between the last Chebyshev launch of the V-cycle and `p_update`."""
        nr, G, wg, wd, dk = self.nr, self.G, self.wg, self.wd, self.dk
        i32 = np.int32
        n3, nn0 = self.n3, self.levels[0].nn
        iso = np.dtype(self.RO).itemsize
        k1 = self.kernel("defl_dot")
        k1.set_args(self.vout, self.AZd, self.PD, i32(n3), i32(G), self._local(iso, wd))
        k2 = self.kernel("defl_coef")
        k2.set_args(self.PD, self.Einv_d, self.GC, i32(G), self._local(iso, wd))
        k3 = self.kernel("defl_apply")
        k3.set_args(self.vout, self.Zd, self.GC, self.r_o, self.zo, self.P1, i32(nn0), i32(n3), self._local(iso, wg))
        return [Op("defl_dot", k1, nr * G * wd, wd), Op("defl_coef", k2, nr * wd, wd), Op("defl_apply", k3, roundup(nn0, wg), wg)]

    def use_deflation(self, on):
        """Switch the deflation on or off for the next solves (the model does it for a design whose host numbers failed). Off: the launches and the PCG
        of before P2d. On needs a deflation that the engine holds (`defl` is not None)."""
        if on and self.defl is None:
            raise ValueError("this engine has no deflation: " + str(self.defl_off))
        self.defl_active = bool(on)
        self.iter_ops = self._ops_defl if on else self._ops_plain

    def _bord_ops(self, l, xbuf, outer=False):
        """The launches that make `wb = (Kb - C D^-1 C^T) x` of level `l` for the vector `xbuf` (none without a border or without a bonded node):
        `bord_t` (`C^T x`, one work-group for each border dof; joints only) and `bord_w` (the product with `Kb`, the term `- C D^-1 (C^T x)`).
        `outer`: the 64-bit operator of level 0 in case B (case C has one type and the same buffers)."""
        nb = self.bnb[l]
        if self.border is None or not nb:
            return []
        outer = outer and not self.same
        sfx = "o" if outer else "v"
        kval, cb, w, di, tb = ((self.kval_o, self.cb_o, self.wb_o, self.Di_o, self.tb_o) if outer
                               else (self.kval[l], self.cb[l], self.wb[l], self.Di, self.tb))
        isz = np.dtype(self.RO if outer else self.RV).itemsize
        n3 = np.int32(3 * self.levels[l].nn)
        ops = []
        if self.nk:
            kt = self.kernel(f"bord_t_{sfx}")
            kt.set_args(xbuf, self.bn[l], cb, tb, np.int32(nb), n3, self._local(isz, self.wgv))
            ops.append(Op("bord_t", kt, self.nk * self.wgv, self.wgv))
        k = self.kernel(f"bord_w_{sfx}")
        k.set_args(xbuf, w, self.bn[l], self.kptr[l], self.kcol[l], kval, cb, tb if self.nk else self.dummy, di if self.nk else self.dummy,
                   np.int32(nb), n3, self._local(isz, max(1, self.nr * self.nk)))
        ops.append(Op("bord_w", k, roundup(nb, self.wg), self.wg))
        return ops

    # ---------------------------------------------------------------- kernels and op lists
    def _topo(self, l):
        return (self.nb27[l], self.nelem[l], self.enodes[l])

    def _cheb_kernel(self, l, b, x, y, d, kind, dot=False):
        Lv = self.levels[l]
        k = self.kernel("cheb_dot" if dot else "cheb")
        args = [b, x, y, d, self.dinv[l], *self._topo(l), self.Ee[l], self.kel_v, self.fix[l],
                np.int32(Lv.nn), np.int32(3 * Lv.nn), self.RV(0), self.RV(0)]
        if dot:
            args += [self.r_o, self.zo, self.P1, self._local(np.dtype(self.RO).itemsize, self.wg)]
        if self.border is not None:
            args.append(self.wb[l])
        k.set_args(*args)
        self.coef_targets.append((l, kind, k, 13, 14))
        return k

    def _assemble(self):
        """The fused V-cycle as a list of launches. The first Chebyshev step of level 0 is in
        `init_k` and `upd_xr`, and the last launch also makes the partial dot products of the PCG."""
        L, nl, wg = self.levels, self.nlev, self.wg
        i32 = np.int32
        n3l = [3 * Lv.nn for Lv in L]
        ops = []

        def node(nm, k, nn):
            return Op(nm, k, roundup(nn, wg), wg)

        for l in range(nl - 1):
            Lv = L[l]
            b, x, y, d, r = self.b[l], self.x[l], self.y[l], self.d[l], self.r[l]
            k = self._cheb_kernel(l, b, x, y, d, "step1")
            ops += self._bord_ops(l, x)
            ops.append(node("cheb_pre1", k, Lv.nn))
            k = self.kernel("resid_v")
            args = [b, y, r, *self._topo(l), self.Ee[l], self.kel_v, self.fix[l], i32(Lv.nn), i32(n3l[l])]
            k.set_args(*(args + [self.wb[l]] if self.border is not None else args))
            ops += self._bord_ops(l, y)
            ops.append(node("resid", k, Lv.nn))
            N = L[l + 1]
            if l + 1 < nl - 1:
                k = self.kernel("restrict_c0")
                k.set_args(r, self.b[l + 1], self.x[l + 1], self.d[l + 1], self.dinv[l + 1], self.rtab[l + 1],
                           self.fix[l + 1], i32(N.nn), i32(n3l[l + 1]), i32(n3l[l]), self.RV(0))
                self.coef_targets.append((l + 1, "c0", k, 10, None))
                ops.append(node("restrict_c0", k, N.nn))
            else:
                k = self.kernel("restrict_k")
                k.set_args(r, self.b[l + 1], self.rtab[l + 1], self.fix[l + 1], i32(N.nn), i32(n3l[l + 1]), i32(n3l[l]))
                ops.append(node("restrict", k, N.nn))
                gk = self.kernel("gemv")
                gk.set_args(self.Ainv, self.b[l + 1], self.x[l + 1], i32(n3l[l + 1]),
                            self._local(np.dtype(self.RV).itemsize, self.wgv))
                ops.append(Op("gemv", gk, n3l[l + 1] * self.wgv, self.wgv))
        out = [None] * nl
        out[nl - 1] = self.x[nl - 1]
        for l in range(nl - 2, -1, -1):
            Lv = L[l]
            b, x, y, d = self.b[l], self.x[l], self.y[l], self.d[l]
            k = self.kernel("prolong_add")
            k.set_args(out[l + 1], y, self.ptab[l], self.pw[l], self.fix[l], i32(Lv.nn), i32(n3l[l]), i32(n3l[l + 1]))
            ops.append(node("prolong", k, Lv.nn))
            k = self._cheb_kernel(l, b, y, x, d, "post0")
            ops += self._bord_ops(l, y)
            ops.append(node("cheb_post0", k, Lv.nn))
            if l == 0:
                k = self._cheb_kernel(l, b, x, y, d, "step1", dot=True)
                ops += self._bord_ops(l, x)
                ops.append(node("cheb_post1_dot", k, Lv.nn))
            else:
                k = self._cheb_kernel(l, b, x, y, d, "step1")
                ops += self._bord_ops(l, x)
                ops.append(node("cheb_post1", k, Lv.nn))
            out[l] = y
        self.vout = out[0]
        return ops

    def _pcg_ops(self):
        nr, G, wgv, wg = self.nr, self.G, self.wgv, self.wg
        i32 = np.int32
        n3, nn0 = self.n3, self.levels[0].nn
        iso = np.dtype(self.RO).itemsize
        ops = []
        k = self.kernel("p_update")
        k.set_args(self.vout, self.p_o, self.zo, self.sc, self.P1, i32(n3), i32(G), i32(self.NG1), i32(1),
                   self._local(iso, wgv))
        ops.append(Op("p_update", k, nr * G * wgv, wgv))
        k = self.kernel("spmv_dot_o")
        args = [self.p_o, self.Ap_o, *self._topo(0), self.Ee_o, self.kel_o, self.fix[0], i32(nn0), i32(n3), self.P2, self._local(iso, wg)]
        k.set_args(*(args + [self.wb_o] if self.border is not None else args))
        ops += self._bord_ops(0, self.p_o, outer=True)         # P2b: Kb p (and the joint term) before the product
        ops.append(Op("spmv_dot", k, roundup(nn0, wg), wg))
        self.k_upd = self.kernel("upd_xr")
        self.k_upd.set_args(self.x_o, self.r_o, self.p_o, self.Ap_o, self.b[0], self.x[0], self.d[0], self.dinv[0],
                            self.sc, self.P1, self.P2, self.P3, i32(nn0), i32(n3), i32(G), i32(self.NG1),
                            i32(self.NG2), self.RV(0), self._local(iso, wgv))
        ops.append(Op("upd_xr", self.k_upd, nr * G * wgv, wgv))
        k = self.kernel("scal_rr")
        k.set_args(self.P3, self.sc, i32(G), self._local(iso, wgv))
        ops.append(Op("scal_rr", k, nr * wgv, wgv))
        self.k_init = self.kernel("init_k")
        self.k_init.set_args(self.b_o, self.r_o, self.b[0], self.x_o, self.p_o, self.zo, self.x[0], self.d[0],
                             self.dinv[0], i32(nn0), i32(n3), self.RV(0))
        self.c0_level0_kernels = [(self.k_upd, 17), (self.k_init, 11)]
        self.dotrr_k = self.kernel("dotrr")
        self.dotrr_k.set_args(self.r_o, self.P3, i32(n3), i32(G), self._local(iso, wgv))
        self.scal_rr0 = self.kernel("scal_rr")
        self.scal_rr0.set_args(self.P3, self.sc, i32(G), self._local(iso, wgv))
        self.init_ops = [Op("init", self.k_init, roundup(nn0, wg), wg)]
        self.dotrr_ops = [Op("dotrr", self.dotrr_k, nr * G * wgv, wgv), Op("scal_rr", self.scal_rr0, nr * wgv, wgv)]
        cl.enqueue_fill_buffer(self.q, self.P1, np.zeros(1, self.RO), 0, 2 * nr * self.NG1 * iso)
        return ops

    def set_chebyshev(self, lmax):
        """Set the Chebyshev coefficients (degree 2) from the lmax of each level except the coarsest."""
        c = {}
        for l in range(self.nlev - 1):
            lm = lmax[l]
            ln = lm / CHEB_ALPHA
            theta = 0.5 * (lm + ln)
            delta = 0.5 * (lm - ln)
            sigma = theta / delta
            rho0 = 1.0 / sigma
            rho1 = 1.0 / (2.0 * sigma - rho0)
            c[l] = {"c2_0": 1.0 / theta, "c1_1": rho1 * rho0, "c2_1": 2.0 * rho1 / delta}
        RV = self.RV
        for l, kind, k, i1, i2 in self.coef_targets:
            if kind == "c0":
                k.set_arg(i1, RV(c[l]["c2_0"]))
            elif kind == "post0":
                k.set_arg(i1, RV(0.0))
                k.set_arg(i2, RV(c[l]["c2_0"]))
            else:
                k.set_arg(i1, RV(c[l]["c1_1"]))
                k.set_arg(i2, RV(c[l]["c2_1"]))
        for k, idx in self.c0_level0_kernels:
            k.set_arg(idx, RV(c[0]["c2_0"]))

    # ---------------------------------------------------------------- running
    def run(self, ops):
        """Enqueue the launches in order. With `sync_every` > 0, finish after every N launches."""
        q, se, cnt = self.q, self.sync_every, 0
        for o in ops:
            cl.enqueue_nd_range_kernel(q, o.k, o.gs, o.ls)
            cnt += 1
            if se and cnt % se == 0:
                q.finish()
        if se:
            q.finish()

    # ---------------------------------------------------------------- numerics upload
    def upload_numerics(self, host: HostMG):
        """Copy the numbers of the last `host.setup_host` to the device.

        The stiffness of level l is multiplied by 2^l: the device keeps the matrix of level 0
        and the element size doubles on each coarser level.
        """
        q = self.q
        for l, Lv in enumerate(self.levels):
            cl.enqueue_copy(q, self.Ee[l], np.ascontiguousarray(Lv.Ee0 * (2.0 ** l), self.RV))
            cl.enqueue_copy(q, self.dinv[l], np.ascontiguousarray(Lv.dinv, self.RV))
        if not self.same:
            cl.enqueue_copy(q, self.Ee_o, np.ascontiguousarray(self.levels[0].Ee0, self.RO))
        cl.enqueue_copy(q, self.Ainv, np.ascontiguousarray(host.Ainv_full, self.RV))
        self._lmax = [getattr(Lv, "lmax", 1.0) for Lv in self.levels]
        self.set_chebyshev(self._lmax)
        q.finish()

    def upload_deflation(self):
        """Copy `Z` (when it changed), `AZ` and `E^-1` of the design to the device (P2d). Call it after `upload_numerics` and after `Deflation.set_design`. `Z`
        has harmonic columns that depend on the design, so it is copied for each design; with rigid columns only it depends on the topology and is copied once."""
        if not self.defl_active:
            return
        d, q = self.defl, self.q
        if not d.ready:
            raise RuntimeError("gpu-mg: the deflation has no numbers for this design: call Deflation.set_design after setup_host")
        if d.z_changes or not self.z_uploaded:
            cl.enqueue_copy(q, self.Zd, d.Zt32)
            self.z_uploaded = True
        cl.enqueue_copy(q, self.AZd, d.AZt32)
        cl.enqueue_copy(q, self.Einv_d, d.Einv32)
        q.finish()

    def precondition(self, R):
        """The V-cycle of the device alone, `z = M r` for the `nr` residuals `R` (nr, n3) (P2d, "D3h"). Return `z` (nr, n3) float64. It needs the numbers of
        the design (`upload_numerics`). It uses the buffers of the PCG as work space, so call it between solves and not inside one."""
        q, nr, n3 = self.q, self.nr, self.n3
        R = np.asarray(R)
        if R.shape != (nr, n3):
            raise ValueError(f"precondition: the residuals have shape {R.shape}, the engine takes {(nr, n3)}")
        cl.enqueue_copy(q, self.b_o, np.ascontiguousarray(R, self.RO))
        self.run(self.init_ops)
        self.run(self._asm_ops)
        z = np.empty((nr, n3), self.RV)
        cl.enqueue_copy(q, z, self.vout)
        q.finish()
        return z.astype(np.float64)

    # ---------------------------------------------------------------- PCG
    def pcg(self, B, rtol, maxit=300):
        """Solve for `nr` right-hand sides. `B`: (nr, nn0, 3) float64 with zeros on the Dirichlet dofs.

        Return `(X, info)`. `X` is (nr, nn0, 3) float64. The loop stops when the recurrence
        residual |r| / |b| of all columns is below `rtol`. `info`: `iters` (the first iteration
        with a residual below `rtol`, per column; None if not reached), `executed` (iterations
        run), `converged`, `relres` (history), `bnorm`, `t_loop`.

        With the deflation (P2d) the columns of `Z` (the rigid modes, and the harmonic columns of the bonded patches) are solved exactly on the host in FP64
        around the device loop: `a = E^-1 Z^T b`, the device solves `b2 = b - A Z a` with the projection in each iteration, and the result is `x + Z a`. `b2` has
        `Z^T b2 = 0`, and the solution of `A y = b2` is A-orthogonal to `Z`. `rtol` and `relres` then refer to the 2-norm of `b2`. `I - A Z E^-1 Z^T` is an oblique
        projector: `b2` is not larger than `b` in the `A^-1` norm only. In the 2-norm it can be larger (measured on a small dumbbell, one host, pocl: 1.3 times for a random
        `b` with 18 columns and 23 times in the worst case; 1.01 and 5.3 times with the 6 rigid columns). So `rtol` is not a bound on `|r| / |b|`. The acceptance of a solve is
        the FP64 check of the host, which measures the true residual `b - A x` and not this one. `info["deflated"]` is True.
        """
        d = self.defl if self.defl_active else None
        if d is None:
            return self._pcg_device(B, rtol, maxit)
        t0 = time.perf_counter()
        a, B2 = d.coarse(np.asarray(B, dtype=np.float64).reshape(B.shape[0], -1))
        t_pre = time.perf_counter() - t0
        X, info = self._pcg_device(B2.reshape(B.shape), rtol, maxit)
        t0 = time.perf_counter()
        X = X + d.correction(a).reshape(X.shape)
        self.t_coarse += t_pre + time.perf_counter() - t0
        info["deflated"] = True
        return X, info

    def _pcg_device(self, B, rtol, maxit):
        """The PCG loop on the device (the plain engine, or the engine with the projection launches in `iter_ops`)."""
        q = self.q
        nr, n3 = self.nr, self.n3
        isz = np.dtype(self.RO).itemsize
        cl.enqueue_copy(q, self.b_o, np.ascontiguousarray(B.reshape(nr, n3), self.RO))
        self.hostsc[:] = 0
        self.hostsc[0:nr] = np.inf
        cl.enqueue_copy(q, self.sc, self.hostsc)
        self.run(self.init_ops)
        self.run(self.dotrr_ops)
        out = np.empty(8 * nr, self.RO)
        cl.enqueue_copy(q, out, self.sc)
        bb = out[3 * nr:4 * nr].astype(np.float64).copy()
        thr = (rtol ** 2) * bb
        hist = np.zeros((maxit + 1, nr), self.RO)
        t0 = time.perf_counter()
        nit = 0
        if self.sync_every == 0:
            # Speculative check: read the residual of iteration k without a wait, and test
            # the residual of iteration k - 1. One iteration more runs than the test needs.
            pending = None
            for it in range(maxit):
                self.run(self.iter_ops)
                ev = cl.enqueue_copy(q, hist[it], self.sc, src_offset=3 * nr * isz, is_blocking=False)
                nit = it + 1
                if pending is not None:
                    pending.wait()
                    if np.all(hist[it - 1] < thr):
                        break
                pending = ev
        else:
            sc_host = np.empty(8 * nr, self.RO)
            for it in range(maxit):
                self.run(self.iter_ops)
                cl.enqueue_copy(q, sc_host, self.sc, is_blocking=True)
                hist[it] = sc_host[3 * nr:4 * nr]
                nit = it + 1
                if np.all(hist[it] < thr):
                    break
        q.finish()
        t_loop = time.perf_counter() - t0
        X = np.empty(nr * n3, self.RO)
        cl.enqueue_copy(q, X, self.x_o)
        its = []
        for c in range(nr):
            idx = np.flatnonzero(hist[:nit, c].astype(np.float64) < thr[c])
            its.append(int(idx[0]) + 1 if idx.size else None)
        with np.errstate(divide="ignore", invalid="ignore"):
            relres = np.sqrt(hist[:nit].astype(np.float64) / bb[None, :])
        info = {"iters": its, "executed": nit, "converged": all(i is not None for i in its),
                "relres": relres, "bnorm": np.sqrt(bb), "t_loop": t_loop}
        return X.astype(np.float64).reshape(nr, -1, 3), info

    def release(self):
        """Release the device buffers. The object cannot solve after this."""
        try:
            self.q.finish()
        except cl.Error:
            pass
        seen = set()
        for name, v in list(vars(self).items()):
            for it in (v if isinstance(v, list) else [v]):
                if isinstance(it, cl.Buffer) and id(it) not in seen:
                    seen.add(id(it))
                    try:
                        it.release()
                    except cl.Error:
                        pass
        self.iter_ops = self.init_ops = self.dotrr_ops = self.coef_targets = None
        self._ops_plain = self._ops_defl = None
        self.c0_level0_kernels = None


# A restart loop is stagnant when the largest residual of the columns that are not accepted yet fell by less than this fraction
# over two steps (internal stop condition of the loop, G2: it applies only when the backward-error rule is on and eta is above the limit).
STAGNATION_FALL = 0.10

# The last restart step (P2b). A step with the fixed inner tolerance 1e-2 divides the true residual by about 100. When the residual is
# below 50 times the target, that overshoots: the step ends 9 to 32 times below the target in the measured r3_run, and the extra iterations
# are lost. A step whose start residual is `rel` needs only the factor `LAST_STEP_MARGIN * rtol / rel` to land at `LAST_STEP_MARGIN * rtol`
# (half the target), so `solve_refined(..., last_step_margin=LAST_STEP_MARGIN)` loosens the inner tolerance of that step to this number. It never
# makes a step stricter than `inner_rtol`. The margin is a safety factor between the recurrence residual of the inner PCG and the true residual.
LAST_STEP_MARGIN = 0.5


def true_residual(host: HostMG, B, u, y=None, fixed=None):
    """The FP64 residual `B - K u` of each column on the host. Return `(r, rn)`: `r` like `B` (zero on the Dirichlet dofs), `rn` the norms."""
    y = np.zeros(B.shape[1:]) if y is None else y
    fixed = host.levels[0].fix.reshape(-1).astype(bool) if fixed is None else fixed
    r = np.empty_like(B)
    for c in range(B.shape[0]):
        host.apply_fine(u[c], y)
        r[c] = B[c] - y
        r[c].reshape(-1)[fixed] = 0.0
    return r, np.array([np.linalg.norm(r[c]) for c in range(B.shape[0])])


def backward_errors(host: HostMG, u, rn, bn, want, y=None):
    """The normwise backward error `eta = |r| / (| |K| |u| | + |b|)` of the columns that `want` selects (Rigal and Gaches).

    `rn` and `bn` are the norms of the residual and of the right-hand side. `eta` is the size of the smallest relative change of
    K that makes `u` the exact solution, so FP64 rounding in a solve of any method keeps it near 1e-16 (the unit roundoff), however
    large the residual is. A column that `want` does not select gets `nan` (not computed). `|K| |u|` is `host.apply_fine_abs`.
    """
    y = np.zeros(u.shape[1:]) if y is None else y
    eta = np.full(len(rn), np.nan)
    for c in np.flatnonzero(want):
        host.apply_fine_abs(u[c], y)
        den = np.linalg.norm(y) + bn[c]
        eta[c] = rn[c] / den if den > 0 else 0.0
    return eta


def check_solution(host: HostMG, B, X, rtol, backward_error):
    """The true residual of the result `X` of a first pass, and the acceptance (G2, item 4).

    `backward_error` > 0 turns the rule on: a column is accepted when its relative residual is below `rtol`, or when its backward
    error is at most `backward_error`. Return a dict: `rel` and `eta` (per column; `eta` is `nan` where it was not computed),
    `ok` (all columns accepted), `accepted_by` ("tol", "backward_error", or None when `ok` is False) and `seconds`.
    """
    t0 = time.perf_counter()
    bn = np.array([np.linalg.norm(B[c]) for c in range(B.shape[0])])
    _, rn = true_residual(host, B, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(bn > 0, rn / bn, 0.0)
    eta = backward_errors(host, X, rn, bn, rel >= rtol) if backward_error > 0 else np.full(len(rn), np.nan)
    ok = bool(np.all((rel < rtol) | (eta <= backward_error)))
    by = None if not ok else ("tol" if np.all(rel < rtol) else "backward_error")
    return {"rel": rel, "eta": eta, "ok": ok, "accepted_by": by, "seconds": time.perf_counter() - t0}


def solve_refined(g: GpuMG, host: HostMG, B, rtol, inner_rtol, maxref=12, maxit=300, u0=None, backward_error=0.0, last_step_margin=None):
    """FP64 residual refinement (a restart loop) on the CPU around the inner PCG on the device.

    Case C uses it for every solve (32-bit inner PCG). Case B uses it as a restart when its plain
    64-bit PCG did not reach `rtol` (the engine of case B works in the same loop).
    `B`: (nr, nn0, 3) float64. `u0` (optional, same shape) is the start vector: the loop starts
    from the true residual of `u0`. The residual is computed with the FP64 matrix-vector product of
    the host (numba). A column with a zero residual is not solved again. Return `(X, info)`.
    `info`: `steps`, `inner_iters` (executed iterations of each step), `rel_hist`, `converged`,
    `t_cpu_residual`, `t_gpu_inner`, and (G2) `accepted_by`, `eta_hist`, `stagnated`, and (P2b) `nonfinite` (the residual was not finite: the loop
    stopped at once).

    A column is accepted when its true relative residual is below `rtol`. With `backward_error` > 0 (G2) it is also accepted when
    its normwise backward error is at most that number: the FP64 residual of a solve has a floor of about eps * |K||u|, and on a
    design with a near-void region around stiff strips the floor is above 1e-8 for any solver. The backward error is computed
    only in a step where some column is above `rtol`. With `backward_error` > 0 the loop also stops when the largest residual of the
    columns that are not accepted yet fell by less than `STAGNATION_FALL` over two steps (`stagnated`; the loop cannot go lower and
    eta is above the limit; a column that a test accepted does not count). With `backward_error` = 0 the loop is the loop of d19c7fb.
    `eta_hist` has one entry for each residual check: a list with the backward error of each column (None where not computed), or None.

    `last_step_margin` (P2b, `LAST_STEP_MARGIN` in the model) is None for the loop of d19c7fb: every step solves to `inner_rtol`. A number `m`
    makes a step solve to `max(inner_rtol, m * rtol / rel)`, with `rel` the largest relative residual of the columns that no test accepts yet.
    The step then ends near `m * rtol` instead of 9 to 32 times below `rtol`. A step that starts at `rel` of `m * rtol / inner_rtol` or more
    (50 times `rtol` for `m` 0.5 and `inner_rtol` 1e-2) is unchanged.
    `info["inner_tols"]` has the inner tolerance of each step.
    """
    nr = B.shape[0]
    u = np.zeros_like(B) if u0 is None else np.array(u0, dtype=np.float64)
    y = np.zeros(B.shape[1:])
    fixed = host.levels[0].fix.reshape(-1).astype(bool)
    bn = np.array([np.linalg.norm(B[c]) for c in range(nr)])
    its_per, rel_hist, eta_hist, tols = [], [], [], []
    t_cpu = t_gpu = 0.0
    converged = stagnated = nonfinite = False
    accepted_by = None
    for k in range(maxref + 1):
        t0 = time.perf_counter()
        r, rn = true_residual(host, B, u, y, fixed)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(bn > 0, rn / bn, 0.0)
        eta = None
        if backward_error > 0 and np.any(rel >= rtol) and (k > 0 or u0 is not None):      # u = 0 has nothing to measure
            eta = backward_errors(host, u, rn, bn, rel >= rtol, y)
        t_cpu += time.perf_counter() - t0
        rel_hist.append(rel.tolist())
        eta_hist.append(None if eta is None else [None if np.isnan(v) else float(v) for v in eta])
        if not np.all(np.isfinite(rel)):         # P2b: the inner PCG broke down (a 32-bit operator that is not positive definite): nothing to refine
            nonfinite = True
            break
        if np.all(rel < rtol):
            converged, accepted_by = True, "tol"
            break
        pending = rel >= rtol          # the columns that no test accepts yet
        if eta is not None:
            accepted = (rel < rtol) | (eta <= backward_error)
            if np.all(accepted):
                converged, accepted_by = True, "backward_error"
                break
            pending = ~accepted        # the stagnation test looks at the columns that no test accepts yet (G2 R1: a column at its floor stays flat)
            if k >= 2 and np.max(rel[pending]) > (1.0 - STAGNATION_FALL) * np.max(np.asarray(rel_hist[k - 2])[pending]):
                stagnated = True
                break
        if k == maxref:
            break
        t0 = time.perf_counter()
        live = rn > 0
        scale = np.where(live, rn, 1.0)
        rhs = r / scale[:, None, None]
        if not live.all():              # a zero column would break the PCG: give it a copy of a live one
            rhs[~live] = rhs[np.flatnonzero(live)[0]]
        step_tol = inner_rtol if last_step_margin is None else max(inner_rtol, last_step_margin * rtol / float(np.max(rel[pending])))
        tols.append(step_tol)
        d, info = g.pcg(rhs, step_tol, maxit=maxit)
        u += d * np.where(live, scale, 0.0)[:, None, None]
        t_gpu += time.perf_counter() - t0
        its_per.append(info["executed"])
    return u, {"steps": len(its_per), "inner_iters": its_per, "inner_iters_total": int(sum(its_per)),
               "rel_hist": rel_hist, "converged": converged, "t_cpu_residual": t_cpu, "t_gpu_inner": t_gpu,
               "accepted_by": accepted_by, "eta_hist": eta_hist, "stagnated": stagnated, "inner_tols": tols, "nonfinite": nonfinite}
