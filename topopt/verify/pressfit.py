"""The press-fit check at the bores of a part (`topopt verify`, key `pressfit`). CONTRACTS.md section 40.

The cup of a bore is a linear two-sided Winkler spring on the wall facets of the bore (`bore.py`),
loaded by the interference. The spring acts along the radial unit vector from the axis at each facet quadrature point:

    p = k (d_r - u_r)       the contact pressure (MPa); `k` is the spring stiffness (MPa/mm, a spring for each unit of area)
    d_r = diametral_interference_mm / 2     the radial interference (mm); `u_r` is the radial displacement of the wall

The model is linear. One solve for each cup variant gives the press-fit stress at the bore: `(K + Ks) u = f` on the free dofs, with
`Ks = sum over the bores of int k (u . n)(v . n) dA` and `f = sum over the bores of int k d_r (v . n) dA`. `p` can be below 0 (the wall lifts off
the cup): there is no contact iteration. The supports are those of the static solve. The load cases are solved with no cup spring, so the Goodman
`sa` (the case) and `sm` (the press fit, `fatigue.py`) come from two solves.

The cup is given as `k`, or as `e_mpa`, `nu` and `wall_mm`, from which the Lame stiffness is computed for each bore (`cup_k`). The statistics at a bore
are those of `bore.SampleSet` (p99.9 and max of the von Mises and of the hoop stress). The contact pressure has two forms. The area mean
(`int p dA / A`, on the facet basis of the assembly) is the radial spring force over the area; it is reliable for every `k`. The vertex values
(mean, max, min) are the numbers of the pipeline; they are not a stress measure for a stiff spring.

One factor is alive at a time (CONTRACTS section 16, "Memory model"): `run_level` frees the factor of the static solve before the stage, and the
stage frees the factor of each variant before the next. A stage that does not fit in memory is skipped with a record (facts and options);
`topopt verify` goes on.

The scikit-fem import is inside the functions that need it. This module has no part name, and no material, cup or interference value.
"""
from __future__ import annotations

import time

import numpy as np

from . import memory
from .names import opt, style
from .p2 import STRESS_ORDER, quadrature

KIND = "pressfit"
FACET_QUAD_ORDER = 4       # the facet rule of the assembly, the one of the pipeline (pressfit.py:63; plan 6.2, section 13); the radial vector n(x) is not a polynomial, so it is not exact
STATISTICS = ("p99_9", "max")      # the statistics that feed the Goodman check (the same as `fatigue.STATISTICS`)

RULE = ("p = k (d_r - u_r) per unit of area on the wall facets of the bore: a two-sided linear spring along the radial unit vector from the axis "
        "at each facet quadrature point (it can pull as well as push; there is no contact iteration). d_r = diametral_interference_mm / 2. "
        "One linear solve for each cup variant, with the supports of the static solve. The load cases are solved with no cup spring. "
        "hoop = e_t . sigma . e_t at the sample points of the bore; the statistics are p99.9 and max.")
FACT_VERTEX = ("The vertex values of the contact pressure (mean, max, min) of a stiff spring are differences of near-equal numbers and are not a "
               "stress measure. On thick steel rings with k = 1e6 MPa/mm the vertex mean was 4.5 % to 20 % above the closed form and the vertex max "
               "was 45 % above it. On one measured part the vertex mean was 2.9 times the area mean. The area mean (the radial spring force over the "
               "area) was within 0.4 % of the closed form on every ring mesh for every k. The hoop p99.9, the statistic that feeds sm, was +0.5 % to +2.7 % "
               "above the closed form on eight ring meshes (+1.2 % to +1.5 % on the two meshes of the tests) and +5 % to +7 % on a ninth, where the top "
               "points were at the bore mouths: it is a point statistic of the P2 stress at the surface and it moves with the mesh.")
FACT_FORCE = ("net_spring_force_N is the force of the spring on the part, the integral of p n over the wall facets; the sum over the bores is minus the "
              "reaction of the clamps (solve.clamp_reaction_N). load_vector_net_N is the net of the load vector k d_r n: it is not a force on the "
              "part for a large k.")
FACT_SERVICE = ("The load cases of the Goodman check are solved with no cup spring. The Goodman sa (the case) and sm (the press fit) come from two "
                "solves.")
FACT_INTERFERENCE = ("The interference of a bore is the diametral value of the YAML (with its source text in the block bore_checks of this report): "
                     "the radial interference is half of it.")
FACT_SUPPORTS = "The supports are those of the static solve (the clamped fixed regions). The cup spring acts on the wall facets of the bores only."
FACT_COMBINED = ("combined is the stress of a load case plus the stress of the press fit at the sample points of the bore (a sum of two linear solutions, as "
                 "the pipeline check does). The cup spring does not act in the solve of the load case, so the sum is not a solve of the assembled problem.")


def rerun_text() -> str:
    return "topopt verify <run_dir>" if style() == "cli" else "verify_run"


# --------------------------------------------------------------------------- #
# The cup
# --------------------------------------------------------------------------- #

def cup_k(R: float, wall: float, E: float, nu: float) -> float:
    """The stiffness of a cup (MPa/mm): the inverse of the radial compliance at the outer radius `R` of a thick cylinder with a free inner surface
    (radius `R - wall`) under an outer pressure, in plane stress. It is the expression of `pressfit.py:55-58` of the pipeline, term for term
    (the same bits). `wall = R` is a solid disc."""
    ri = R - wall
    c = R / E * ((R * R + ri * ri) / (R * R - ri * ri) - nu)
    return 1.0 / c


def stiffness(cup, spec) -> float:
    """The spring stiffness (MPa/mm) of `cup` (a `config.Cup`) at the bore `spec` (a `config.BoreSpec`): its `k`, or the Lame stiffness at the bore radius."""
    if cup.k_mpa_per_mm is not None:
        return float(cup.k_mpa_per_mm)
    return cup_k(float(spec.radius_mm), float(cup.wall_mm), float(cup.e_mpa), float(cup.nu))


# --------------------------------------------------------------------------- #
# The spring of one bore on one mesh
# --------------------------------------------------------------------------- #

class Spring:
    """The facet basis and the radial directions of one `bore.Bore` on one mesh. Build it once; `matrices` and `evaluate` take `k` and `d_r` of a variant.

    `n` is the radial unit vector from the axis at each facet quadrature point (not the normal of the facet), as in `pressfit.py:64` of the pipeline.
    """

    def __init__(self, model, bore):
        from skfem import ElementTetP2, ElementVector, FacetBasis
        spec = bore.spec
        self.bore = bore
        self.name = spec.name
        a = np.asarray(spec.axis, float)
        c = np.asarray(spec.point, float)
        self.facets = np.asarray(bore.ids)
        self.basis = FacetBasis(model.mesh, ElementVector(ElementTetP2()), facets=self.facets, intorder=FACET_QUAD_ORDER)
        X = np.asarray(self.basis.global_coordinates().value)                 # (3, nf, nq)
        d = X - c[:, None, None]
        rad = d - a[:, None, None] * np.einsum("i,ijk->jk", a, d)[None]
        self.n = rad / np.linalg.norm(rad, axis=0)
        self.dx = np.asarray(self.basis.dx)                                    # (nf, nq): the quadrature weight times the facet Jacobian
        self.area = float(self.dx.sum())
        self.nodes = np.unique(model.mesh.facets[:, self.facets])             # the vertex nodes of the facets (the vertices come first in the P2 numbering)
        xn = model.p[:, self.nodes].T - c
        rn = xn - (xn @ a)[:, None] * a
        self.er = rn / np.linalg.norm(rn, axis=1)[:, None]

    def nbytes(self) -> int:
        """The bytes of the arrays that this spring holds (the radial directions, the weights, the node data)."""
        return int(self.n.nbytes + self.dx.nbytes + self.nodes.nbytes + self.er.nbytes + self.facets.nbytes)

    def matrices(self, k: float, dr: float):
        """`(Ks, f)` of this bore: the spring matrix (N, N, CSR) and the load vector (N,) for the stiffness `k` and the radial interference `dr`."""
        from skfem import BilinearForm, LinearForm, asm
        from skfem.helpers import dot

        @BilinearForm
        def form(u, v, w):
            return w.k * dot(u, w.n) * dot(v, w.n)

        @LinearForm
        def load(v, w):
            return w.k * w.dr * dot(v, w.n)

        kk = np.full(self.n.shape[1:], float(k))
        dd = np.full(self.n.shape[1:], float(dr))
        return asm(form, self.basis, k=kk, n=self.n).tocsr(), asm(load, self.basis, k=kk, dr=dd, n=self.n)

    def evaluate(self, u: np.ndarray, k: float, dr: float) -> dict:
        """The contact pressure and the net force of this bore for the displacement `u`.

        The area mean and the net force are on the facet basis of the assembly. The vertex values (`mean`, `max`, `min`; `nodes`, `negative_nodes`) are
        `p = k (d_r - u_r)` at the vertex nodes of the facets, as the pipeline does.
        """
        uq = np.asarray(self.basis.interpolate(u).value)                       # (3, nf, nq)
        p = k * (dr - np.einsum("ijk,ijk->jk", uq, self.n))
        net = [float((p * self.n[c] * self.dx).sum()) for c in range(3)]
        ur = np.einsum("ij,ij->i", u.reshape(-1, 3)[self.nodes], self.er)
        pv = k * (dr - ur)
        return dict(contact_pressure_MPa=dict(area_mean=float((p * self.dx).sum() / self.area), mean=float(pv.mean()), max=float(pv.max()),
                                              min=float(pv.min()), nodes=int(len(self.nodes)), negative_nodes=int((pv < 0).sum()),
                                              mean_radial_displacement_mm=float(ur.mean())),
                    net_spring_force_N=net)


def assemble(springs, ks: dict, drs: dict):
    """`(Ks, f, load_vector_net)` of all bores for one variant. `ks` and `drs` are `{bore name: value}`; `load_vector_net` is `{bore name: [Fx, Fy, Fz]}`."""
    Ks, f, net = None, None, {}
    for sp in springs:
        Km, fv = sp.matrices(ks[sp.name], drs[sp.name])
        Ks = Km if Ks is None else Ks + Km
        f = fv.copy() if f is None else f + fv
        net[sp.name] = [float(x) for x in fv.reshape(-1, 3).sum(axis=0)]
    return Ks.tocsr(), f, net


def csr_bytes(A) -> int:
    return int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)


def stress_array_bytes(model) -> int:
    """The bytes of the array `sig` (nel, nq, 6) of `model.stress_qp`, the stress at the quadrature points of every tet. The stage holds it for one variant, while it
    reads the sample points (6 is the number of Voigt components)."""
    return int(model.nel * len(quadrature(STRESS_ORDER)[1]) * 6 * np.dtype(float).itemsize)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

def _header(bc) -> dict:
    return dict(kind=KIND, rule=RULE, statistics=list(STATISTICS), cups=[c.name for c in bc.cups])


def skip_record(bc, reason: str, facts: list, options: list, **extra) -> dict:
    """The record of a stage that did not start or has nothing to report: facts with options. `topopt verify` goes on."""
    return dict(_header(bc), status="skipped", reason=reason, variants=[], facts=facts, options=options, **extra)


def error_record(bc, message: str) -> dict:
    return dict(_header(bc), status="error", reason="exception", message=message, variants=[],
                facts=[f"The check stopped: {message}"],
                options=[f"Read the message, then run `{rerun_text()}` again with more memory ({opt('max_gb')}, {opt('force')}), a larger "
                         f"{opt('mesh_size')}, or with the check off (remove the `press_fit` block from the YAML, or use `--no-bore-checks`)."])


def memory_decision(bc, term: int, after_bytes: int, lim: dict, force: bool, size: float):
    """Is the memory estimate with the stage above the limit? Return `(skip record or None, memory facts)`.

    The estimate is the estimate of the run (the factor; one factor at a time) plus `term`, the bytes that the stage holds on top of the factor: the
    spring matrix of a variant, its load vector and the arrays of the springs. The limit is the smaller of the available RAM and `--max-gb`.
    With `--force` the stage starts.
    """
    limit = lim["limit_bytes"]
    mem = dict(extra_bytes=int(term), estimate_without_stage_bytes=int(after_bytes), estimate_with_stage_bytes=int(after_bytes + term),
               limit_bytes=limit, limit_source=lim["limit_source"],
               model="the estimate of the factor (one factor at a time) + the nbytes of the spring matrix and the load vector of the largest variant "
                     "+ the arrays of the springs (the radial directions and weights of the facet quadrature points) + the stress of one variant at the quadrature "
                     "points of every tet + the stress of each load case at the sample points (the key `combined`)")
    if limit is None or after_bytes + term <= limit or force:
        return None, mem
    facts = [f"The memory estimate with the press-fit stage is {memory.gb_text(after_bytes + term)} ({memory.gb_text(after_bytes)} for the factor, "
             f"{memory.gb_text(term)} for the spring matrix and the vectors), above the limit {memory.gb_text(limit)} ({lim['limit_source']}). "
             "The stage is not started. The stress check of verify goes on, and the Goodman check has no row for a cup variant."]
    options = [f"A larger {opt('max_gb')}, or {opt('force')} to run anyway (the machine can swap or stop the process).",
               f"A larger {opt('mesh_size')} (fewer dofs, so a smaller factor).",
               f"A factor that needs less memory ({opt('solver')} pardiso on Linux x86_64 with the extra 'pardiso', or cholespy with the extra "
               "'verify') if the factor is SuperLU.",
               "Leave the check off (remove the `press_fit` block from the YAML, or use `--no-bore-checks`)."]
    return skip_record(bc, "memory", facts, options, memory=mem, size_mm=float(size)), mem


def hoop_values(record: dict | None) -> dict | None:
    """`{variant: {bore: {statistic: hoop stress}}}` for `fatigue.build_record` (`sm`), or None when the record has no result."""
    if not record or record.get("status") != "done":
        return None
    return {v["name"]: {b["name"]: {s: b[f"hoop_{s}_MPa"] for s in STATISTICS} for b in v["bores"]} for v in record["variants"]}


def estimate_before_meshing(bc, estimate_bytes: int, dofs: float, solver: str, c: float | None) -> dict:
    """The key `pressfit` of an entry of `estimates_before_meshing` (plan 6.6). The factor is the one of the entry (one factor at a time).

    The time is `variants x (factor time + one solve and stress)`, from the fit of `memory.py`; it is None where the fit has no value (`cholesky`).
    """
    n = len(bc.cups)
    fs = memory.factor_seconds(dofs, solver, c)
    secs = None if fs is None else float(n * (fs + memory.TIME_CASE_PER_1E5_S * dofs / 1e5))
    return dict(variants=n, factorizations=n, estimate_bytes=int(estimate_bytes), estimated_seconds=secs,
                note="one factor at a time: the memory estimate is the one of the entry; the time is the variants times (one factor, one solve and "
                     "the stress at the sample points)")


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #

def _solve_variant(model, K, springs, cup, bc, built, *, free, fixed, solver: str, E: float, nu: float, refine_steps: int):
    """One cup variant: factor `(K + Ks)_ff`, solve, stress at the sample sets, contact pressure, reaction of the clamps. The factor is freed."""
    from .factor import RefinedFactor, make_factor
    specs = {s.name: s for s in bc.bores}
    ks = {sp.name: stiffness(cup, specs[sp.name]) for sp in springs}
    drs = {sp.name: 0.5 * float(specs[sp.name].diametral_interference_mm) for sp in springs}
    t0 = time.perf_counter()
    Ks, f, load_net = built if built is not None else assemble(springs, ks, drs)
    A = (K + Ks)[free][:, free].tocsr()
    t1 = time.perf_counter()
    fac = make_factor(A, solver)
    try:
        t2 = time.perf_counter()
        rf = RefinedFactor(A, fac, steps=refine_steps)
        rhs = f[free][:, None]
        x = rf.solve(rhs)
        if not np.all(np.isfinite(x)):
            raise np.linalg.LinAlgError("the solution has NaN or infinite values")
        resid = float(rf.residual_norms(rhs, x)[0])
        berr = float(rf.backward_error(rhs, x)[0])
        t3 = time.perf_counter()
    finally:
        fac.free()
    u = np.zeros(model.N)
    u[free] = x[:, 0]
    del A, rf, fac
    sig, _lam, _w = model.stress_qp(u, E, nu)
    bores = []
    points = {}
    for sp in springs:
        s = specs[sp.name]
        smp = sp.bore.sample
        st = smp.stats(sig)
        points[sp.name] = smp.points(sig)
        pr = sp.evaluate(u, ks[sp.name], drs[sp.name])
        bores.append(dict(name=sp.name, k_mpa_per_mm=ks[sp.name], diametral_interference_mm=float(s.diametral_interference_mm),
                          radial_interference_mm=drs[sp.name], facets=int(len(sp.facets)), area_mm2=sp.area,
                          net_spring_force_N=pr["net_spring_force_N"], load_vector_net_N=load_net[sp.name],
                          contact_pressure_MPa=pr["contact_pressure_MPa"], **{k: v for k, v in st.items() if k.startswith(("hoop_", "vm_"))},
                          n_qp=st["n_qp"]))
    del sig
    r = np.zeros(model.N)
    r[fixed] = K[fixed] @ u + Ks[fixed] @ u - f[fixed]                        # the force that the clamps put on the part, dof by dof
    reaction = r.reshape(-1, 3).sum(axis=0)
    var = dict(name=cup.name, k_form=cup.form, sources=dict(cup.sources), bores=bores,
               solve=dict(relative_residual=resid, backward_error=berr, refinement_steps=int(refine_steps), assemble_seconds=round(t1 - t0, 3),
                          factor_seconds=round(t2 - t1, 3), solve_seconds=round(t3 - t2, 3), seconds=round(time.perf_counter() - t0, 3),
                          max_displacement_mm=float(np.linalg.norm(u.reshape(-1, 3), axis=1).max()),
                          clamp_reaction_N=[float(v) for v in reaction]))
    if cup.form == "lame":
        var.update(e_mpa=float(cup.e_mpa), nu=float(cup.nu), wall_mm=float(cup.wall_mm))
    return var, points


def combined_stress(springs, service_points: dict, variant_points: dict) -> dict:
    """`{bore: {case: {vm_p99_9_MPa, vm_max_MPa, hoop_p99_9_MPa, hoop_max_MPa}}}` of one cup variant: the stress of each load case plus the stress of the press fit
    at the sample points (linear superposition, plan 6.8)."""
    out = {}
    for sp in springs:
        smp = sp.bore.sample
        out[sp.name] = {}
        for case, by_bore in service_points.items():
            st = smp.stats_at(by_bore[sp.name] + variant_points[sp.name])
            out[sp.name][case] = {f"{m}_{s}_MPa": st[f"{m}_{s}_MPa"] for m in ("vm", "hoop") for s in STATISTICS}
    return out


def _facts(bc, variants: list, dropped: list, reasons: dict | None = None) -> tuple[list, list]:
    facts = [f"Rule: {RULE}", FACT_INTERFERENCE, FACT_SUPPORTS, FACT_SERVICE, FACT_FORCE, FACT_VERTEX]
    neg = [(v["name"], b["name"], b["contact_pressure_MPa"]["negative_nodes"], b["contact_pressure_MPa"]["nodes"])
           for v in variants for b in v["bores"] if b["contact_pressure_MPa"]["negative_nodes"]]
    if neg:
        facts.append("The wall lifts off the cup (p below 0) at some vertex nodes: " + "; ".join(f"{v}/{b}: {n} of {m} nodes" for v, b, n, m in neg[:6])
                     + (" ..." if len(neg) > 6 else "") + ". The spring is linear and pulls there: there is no contact iteration.")
    if dropped:
        why = reasons or {}
        facts.append("No press-fit result for the dropped bore(s) " + ", ".join(f"{n} ({why[n]})" if n in why else n for n in dropped)
                     + ". The reason is `no_facets` (no wall facet), `no_sample_set` (no sample point) or `error` (an exception of the bore check); bore_facts has the facts and the options.")
    options = ["A smaller mesh size, and `--levels` to see how the hoop stress at the bore changes with the mesh (the bore stress is not smoothed).",
               "Remove the `press_fit` block from the YAML (or use `--no-bore-checks`) to leave the check off."]
    return facts, options


def run_stage(bc, bores, model, K, *, free, fixed, solver: str, E: float, nu: float, refine_steps: int, lim: dict, force: bool,
              after_bytes: int, size: float, service_points: dict | None = None, say=lambda _m: None) -> dict:
    """The press-fit check of one level: the record `pressfit`. Call it after the factor of the static solve is freed.

    `bc` is `Config.bore_checks`, `bores` the `bore.Bore` list of the mesh, `K` the stiffness matrix and `free`, `fixed` the dofs of the static solve.
    `after_bytes` is the memory estimate of the run with the real dofs. The stage does the memory decision (a skip record) and then one solve for
    each cup variant. An exception goes to the caller (`run_level` turns it into `error_record`).
    `service_points` is `{case: {bore: stress (n_qp, 6)}}` of the load cases at the sample points (`bore.SampleSet.points`): with it the record has the key
    `combined` (case plus press fit, plan 6.8).
    """
    t_all = time.perf_counter()
    ok = [b for b in bores if b.ok]
    dropped = [b.spec.name for b in bores if not b.ok]
    if not ok:
        return skip_record(bc, "no_bore", ["No bore has a sample set, so the cup spring has no wall facet to act on. The facts of `bore_facts` say why."],
                           ["Check the geometry of the bores against the part, or the knobs of `bore_checks` (see `bore_facts`).",
                            "Remove the `press_fit` block from the YAML (or use `--no-bore-checks`) to leave the check off."])
    rss_before = memory.peak_rss_bytes()
    springs = [Spring(model, b) for b in ok]
    specs = {s.name: s for s in bc.bores}

    def terms(cup):
        return ({sp.name: stiffness(cup, specs[sp.name]) for sp in springs},
                {sp.name: 0.5 * float(specs[sp.name].diametral_interference_mm) for sp in springs})

    first = assemble(springs, *terms(bc.cups[0]))
    term = csr_bytes(first[0]) + int(first[1].nbytes) + sum(sp.nbytes() for sp in springs)
    term += stress_array_bytes(model)                                                                 # the stress of one variant at every tet (review n-2, D3)
    term += sum(int(a.nbytes) for case in (service_points or {}).values() for a in case.values())     # the stress of each case at the sample points (combined)
    skip, mem = memory_decision(bc, term, after_bytes, lim, force, size)
    say(f"  press fit: memory estimate with the stage {memory.gb_text(after_bytes + term)} (+{term / 1e6:.2f} MB for the spring matrix, the stress arrays and the vectors)"
        + ("" if skip is None else "; above the limit, the stage is skipped and the stress check goes on"))
    if skip is not None:
        return skip
    variants, combined = [], {}
    for i, cup in enumerate(bc.cups):
        var, points = _solve_variant(model, K, springs, cup, bc, first if i == 0 else None, free=free, fixed=fixed, solver=solver, E=E, nu=nu,
                                      refine_steps=refine_steps)
        variants.append(var)
        if service_points:
            combined[cup.name] = combined_stress(springs, service_points, points)
        del points
        say(f"  press fit {cup.name}: " + ", ".join(
            f"{b['name']}: hoop p99.9 {b['hoop_p99_9_MPa']:.4g} MPa, contact pressure mean {b['contact_pressure_MPa']['area_mean']:.4g} MPa"
            for b in var["bores"]) + f" ({var['solve']['seconds']:.1f} s)")
        if i == 0:
            first = None
    facts, options = _facts(bc, variants, dropped, {b.spec.name: b.drop_reason for b in bores if not b.ok})
    if combined:
        facts.append(FACT_COMBINED)
    mem.update(peak_rss_before_stage_bytes=int(rss_before), peak_rss_after_stage_bytes=int(memory.peak_rss_bytes()))
    return dict(_header(bc), status="done", reason=None, model=dict(text=RULE, facet_quadrature_order=FACET_QUAD_ORDER, solver=solver,
                                                                     refinement_steps=int(refine_steps)),
                memory=mem, variants=variants, dropped_bores=dropped, facts=facts, options=options, seconds=round(time.perf_counter() - t_all, 3),
                **({"combined": combined} if combined else {}))

