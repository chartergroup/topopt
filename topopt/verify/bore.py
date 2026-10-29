"""The bores of a part on the P2 tet model: the wall facets, the sample set and its statistics (`topopt verify`, B4).

One module serves the Goodman check (`fatigue.py`) and the press-fit check (B4b):
both use the same bore facets and the same sample set, so `sa` and `sm` are statistics over the same points.

A bore is a cylinder: `point`, unit `axis`, `radius_mm`, `length_mm` (`topopt.config.BoreSpec`). The contact range goes from `point`
along the axis for `length_mm`. A boundary facet of the tet mesh is a wall facet of the bore when all four tests hold. The numbers are the
four knobs of the YAML block (`BoreKnobs`; the defaults are the values of the pipeline check, `pressfit.py:45`, `:50`, `:127`):

1. all three vertices have `|r - R| <= radius_tolerance_mm`;
2. the centroid has `end_margin_mm <= s <= length - end_margin_mm`;
3. the unit facet normal has `|n . axis| < normal_limit` (and the facet has an area above 0);
4. the outward normal of the facet points toward the axis (the outward direction is away from the fourth vertex of the tet that owns the
   facet). The pipeline has no such test; the C2 code has one (`interfaces.py`). It keeps a boss or a shaft of the same radius out of the set.

The sample set is the set of quadrature points (order `STRESS_ORDER`, 14 for each tet) of the tet that owns each wall facet, with
`|r - R| < sample_band_mm` (a point exactly on the axis is left out: it has no radial direction; it can be in the band only when R < sample_band_mm). The weight of a point is its quadrature weight times the tet volume. The statistics are p99.9 by volume
(`BORE_PERCENTILE`, step rule, no interpolation), max, median and min of the von Mises stress and of the hoop stress.

This module has no import of an extra at module level and no part name.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .names import opt
from .p2 import STRESS_ORDER, quadrature
from .stats import von_mises, weighted_percentile

BORE_PERCENTILE = 99.9      # the percentile of the bore statistics (the pipeline: goodman.py:12, pressfit.py:131; plan 4.3)


def _perp_basis(axis: np.ndarray):
    """Two unit vectors perpendicular to `axis` and to each other. The helper vector is the coordinate axis with the smallest component."""
    e = np.zeros(3)
    e[int(np.argmin(np.abs(axis)))] = 1.0
    e1 = np.cross(axis, e)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def find_facets(mesh, spec, knobs) -> dict:
    """The wall facets of one bore and the numbers that describe them (plan 4.2, 4.4). Return a dict with `ids` and the facts.

    `ids` are the scikit-fem facet ids (`mesh.facets` columns). The facts are plain numbers and lists: the `funnel`, `facets`, `area_mm2`,
    the analytic areas, `vertex_radius_mm`, `largest_vertex_deviation_mm`, `vertex_s_range_mm`, `mean_facet_edge_mm`, the angular gap and
    width, and `near_miss`.
    """
    a = np.asarray(spec.axis, float)
    c = np.asarray(spec.point, float)
    R, ell = float(spec.radius_mm), float(spec.length_mm)
    bd = mesh.boundary_facets()
    fv = mesh.facets[:, bd]                                  # (3, nf)
    xyz = mesh.p[:, fv]                                      # (coordinate, vertex, facet)
    d = xyz - c[:, None, None]
    s = np.einsum("i,ijk->jk", a, d)                         # (3, nf) axial coordinate of each vertex
    rv = d - a[:, None, None] * s[None]                      # radial vector of each vertex
    r = np.linalg.norm(rv, axis=0)                           # (3, nf)
    dev = np.abs(r - R).max(axis=0)                          # the largest radial deviation of the facet vertices
    sc = s.mean(axis=0)
    cross = np.cross((xyz[:, 1] - xyz[:, 0]).T, (xyz[:, 2] - xyz[:, 0]).T)
    twice = np.linalg.norm(cross, axis=1)
    area = 0.5 * twice
    pos = area > 0
    nrm = np.zeros_like(cross)
    nrm[pos] = cross[pos] / twice[pos, None]
    # the outward normal: away from the fourth vertex of the tet that owns the facet (the vertex ids of a tet sum to the facet ids plus the fourth)
    own = mesh.f2t[0, bd]
    opp = mesh.t[:, own].sum(axis=0) - fv.sum(axis=0)
    cen = xyz.mean(axis=1).T
    flip = np.einsum("ij,ij->i", nrm, mesh.p[:, opp].T - cen) > 0
    nout = np.where(flip[:, None], -nrm, nrm)
    rc = rv.mean(axis=1).T                                   # radial vector of the centroid
    t1 = dev <= knobs.radius_tolerance_mm
    t2 = (sc >= knobs.end_margin_mm) & (sc <= ell - knobs.end_margin_mm)
    t3 = pos & (np.abs(nrm @ a) < knobs.normal_limit)
    t4 = np.einsum("ij,ij->i", nout, rc) < 0                 # toward the axis (the sign of a product: no constant)
    ch = t1 & t2 & t3 & t4
    funnel = dict(boundary=int(len(bd)), radius=int(t1.sum()), axial=int((t1 & t2).sum()), normal=int((t1 & t2 & t3).sum()), side=int(ch.sum()))
    analytic = 2.0 * math.pi * R * (ell - 2.0 * knobs.end_margin_mm)
    full = 2.0 * math.pi * R * ell
    out = dict(ids=bd[ch], funnel=funnel, facets=int(ch.sum()), area_mm2=float(area[ch].sum()), analytic_area_mm2=analytic,
               analytic_area_full_mm2=full)
    out["area_over_analytic"] = out["area_mm2"] / analytic
    nm = t2 & t3 & t4 & ~t1 & (dev < knobs.sample_band_mm)   # a near miss: it fails the radius test only, and lies within the sample band
    out["near_miss"] = dict(facets=int(nm.sum()), area_mm2=float(area[nm].sum()),
                            deviation_mm=[float(dev[nm].min()), float(dev[nm].max())] if nm.any() else None)
    if ch.any():
        edges = np.stack([np.linalg.norm(xyz[:, 1] - xyz[:, 0], axis=0), np.linalg.norm(xyz[:, 2] - xyz[:, 1], axis=0),
                          np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=0)])
        out.update(vertex_radius_mm=[float(r[:, ch].min()), float(r[:, ch].max())], largest_vertex_deviation_mm=float(dev[ch].max()),
                   vertex_s_range_mm=[float(s[:, ch].min()), float(s[:, ch].max())], mean_facet_edge_mm=float(edges[:, ch].mean()))
        e1, e2 = _perp_basis(a)
        th = np.arctan2(np.einsum("i,ijk->jk", e2, rv[:, :, ch]), np.einsum("i,ijk->jk", e1, rv[:, :, ch]))          # (3, n)
        thc = np.arctan2(np.einsum("i,ik->k", e2, rc[ch].T), np.einsum("i,ik->k", e1, rc[ch].T))
        width = 2.0 * np.abs(np.angle(np.exp(1j * (th - thc[None])))).max(axis=0)
        srt = np.sort(thc)
        gap = np.diff(np.r_[srt, srt[0] + 2.0 * math.pi])
        out.update(largest_angular_gap_deg=float(np.degrees(gap.max())), largest_facet_width_deg=float(np.degrees(width.max())),
                   closes_cylinder=bool(gap.max() <= width.max()))
    else:
        out.update(vertex_radius_mm=None, largest_vertex_deviation_mm=None, vertex_s_range_mm=None, mean_facet_edge_mm=None,
                   largest_angular_gap_deg=None, largest_facet_width_deg=None, closes_cylinder=None)
    return out


class SampleSet:
    """The sample set of one bore (plan 4.3) and its statistics. Build it once for a mesh; `stats(sig)` for each stress field."""

    def __init__(self, model, ids: np.ndarray, spec, band_mm: float):
        a = np.asarray(spec.axis, float)
        c = np.asarray(spec.point, float)
        lam, w = quadrature(STRESS_ORDER)
        el = np.unique(model.mesh.f2t[0, ids])                 # the tet that owns each wall facet (one tet layer)
        el = el[el >= 0]
        X = model.qp_points(lam, el)                           # (m, nq, 3)
        d = X - c
        s = d @ a
        rv = d - s[..., None] * a
        r = np.linalg.norm(rv, axis=-1)
        near = (np.abs(r - float(spec.radius_mm)) < band_mm) & (r > 0)      # a point on the axis has no radial direction (no hoop): it is not in the set
        ei, qi = np.nonzero(near)
        self.tets = int(len(el))
        self.elem = el[ei]                                     # tet id of each sample point
        self.qp = qi                                           # quadrature point index of each sample point
        self.weight = (w[qi] * model.vol[self.elem]).astype(float)
        self.n_qp = int(len(ei))
        self.volume_mm3 = float(self.weight.sum()) if self.n_qp else 0.0
        self.band_mm = float(band_mm)
        if self.n_qp:
            er = rv[ei, qi]
            er = er / np.linalg.norm(er, axis=1)[:, None]
            et = np.cross(a, er)
            # hoop = et . sigma . et for the Voigt order xx, yy, zz, xy, yz, xz
            self._hoop = np.stack([et[:, 0] ** 2, et[:, 1] ** 2, et[:, 2] ** 2, 2 * et[:, 0] * et[:, 1], 2 * et[:, 1] * et[:, 2],
                                   2 * et[:, 0] * et[:, 2]], axis=1)

    def info(self) -> dict:
        return dict(tets=self.tets, points=self.n_qp, volume_mm3=self.volume_mm3, band_mm=self.band_mm)

    def points(self, sig: np.ndarray) -> np.ndarray:
        """The stress (n_qp, 6) at the sample points, out of the stress `sig` (nel, nq, 6) of all tets."""
        return sig[self.elem, self.qp]

    def stats(self, sig: np.ndarray) -> dict:
        """The statistics of the von Mises stress (`vm_*`) and the hoop stress (`hoop_*`) at the sample points of the stress `sig` (nel, nq, 6).

        The names are those of `pressfit.json` of the pipeline: `<vm|hoop>_<p99_9|max|median|min>_MPa` and `n_qp`.
        """
        return self.stats_at(self.points(sig))

    def stats_at(self, sg: np.ndarray) -> dict:
        """The same statistics for the stress `sg` (n_qp, 6) at the sample points (`points`; the sum of two such arrays is the stress of a sum of two fields)."""
        out = dict(n_qp=self.n_qp)
        for name, arr in (("vm", von_mises(sg)), ("hoop", (sg * self._hoop).sum(axis=1))):
            p = weighted_percentile(arr, self.weight, [BORE_PERCENTILE, 50.0])
            out[f"{name}_p99_9_MPa"] = float(p[0])
            out[f"{name}_max_MPa"] = float(arr.max())
            out[f"{name}_median_MPa"] = float(p[1])
            out[f"{name}_min_MPa"] = float(arr.min())
        return out


@dataclass
class Bore:
    """One bore on one mesh: its spec, the wall facet ids, the sample set (None when the bore is dropped) and the `bore_facts` entry."""
    spec: object
    ids: np.ndarray
    sample: SampleSet | None
    entry: dict

    @property
    def ok(self) -> bool:
        return self.sample is not None

    @property
    def drop_reason(self) -> str:
        """Why the bore has no sample set: `error` (an exception), `no_facets` or `no_sample_set`."""
        if self.entry["status"] == "error":
            return "error"
        return "no_facets" if self.entry["facets"] == 0 else "no_sample_set"

    def fail(self, exc: Exception, stage: str) -> None:
        """Drop the bore after an exception of `stage` ("build" or "stats"). The other bores and `topopt verify` go on (CONTRACTS.md section 40.4)."""
        self.sample = None
        self.entry = _error_entry(self.spec, exc, stage)


def _funnel_text(f: dict) -> str:
    return (f"{f['boundary']} boundary facets, {f['radius']} within the radius tolerance, {f['axial']} also in the axial range, "
            f"{f['normal']} also with a wall normal, {f['side']} also with the outward normal toward the axis")


def _facts_and_options(spec, knobs, g: dict, sample: SampleSet | None) -> tuple[list, list]:
    facts, options = [], []
    n = g["facets"]
    if n == 0:
        facts.append(f"No boundary facet of the mesh passes the four tests of the bore '{spec.name}' ({_funnel_text(g['funnel'])}). "
                     "The bore is not checked; the other bores go on.")
        options += ["Check point, axis, radius_mm and length_mm against the verified part (the file is in run.verified_file of the report).",
                    f"A larger radius_tolerance_mm (now {knobs.radius_tolerance_mm} mm): the facets of a coarse or remeshed surface can lie off the radius.",
                    f"A smaller end_margin_mm (now {knobs.end_margin_mm} mm), or a smaller mesh size ({opt('mesh_size')}).",
                    "Remove the `bore_checks` block from the YAML (or use `--no-bore-checks`) to leave the check off."]
    else:
        facts.append(f"{n} wall facets, area {g['area_mm2']:.1f} mm2: {g['area_over_analytic']:.3f} of the analytic area "
                     f"2 pi R (length - 2 end margin) = {g['analytic_area_mm2']:.1f} mm2 (the full 2 pi R length is {g['analytic_area_full_mm2']:.1f} mm2). "
                     f"Funnel: {_funnel_text(g['funnel'])}.")
        facts.append(f"The largest vertex deviation from the radius is {g['largest_vertex_deviation_mm']:.3f} mm (radius_tolerance_mm {knobs.radius_tolerance_mm}). "
                     f"The vertices of the facets lie at s = {g['vertex_s_range_mm'][0]:.2f} to {g['vertex_s_range_mm'][1]:.2f} mm "
                     f"(contact range 0 to {spec.length_mm:g} mm); the mean facet edge is {g['mean_facet_edge_mm']:.2f} mm.")
        if g["closes_cylinder"] is False:
            facts.append(f"The facets do not close a cylinder: the largest gap between neighbouring facet centroids is {g['largest_angular_gap_deg']:.1f} degrees "
                         f"and the largest facet is {g['largest_facet_width_deg']:.1f} degrees wide. The stress at the bore is for the part of the wall "
                         "that the facets cover.")
            options += ["Check that the bore of the part is a full cylinder in the verified mesh, and the axis and point of the YAML."]
        if sample is None or sample.n_qp == 0:
            facts.append(f"No quadrature point of the {n} owning tets lies within sample_band_mm ({knobs.sample_band_mm} mm) of the "
                         f"radius. The bore is not checked; the other bores go on.")
            options += [f"A larger sample_band_mm (now {knobs.sample_band_mm} mm), or a smaller mesh size ({opt('mesh_size')}).",
                        "Remove the `bore_checks` block from the YAML (or use `--no-bore-checks`) to leave the check off."]
    nm = g["near_miss"]
    if nm["facets"]:
        facts.append(f"{nm['facets']} facets (area {nm['area_mm2']:.1f} mm2) pass the axial, normal and side tests and lie within sample_band_mm of the radius, "
                     f"but fail the radius tolerance (deviation {nm['deviation_mm'][0]:.3f} to {nm['deviation_mm'][1]:.3f} mm). They carry no spring and are "
                     "not in the wall area of the report.")
        options += [f"A larger radius_tolerance_mm (now {knobs.radius_tolerance_mm} mm; these facets then join the set), or check radius_mm.",
                    f"A smaller mesh size ({opt('mesh_size')}) so that a facet row is narrower than the feature at the bore mouth."]
    return facts, options


STAGES = {"build": "build of the wall facets and the sample set", "stats": "statistics of the stress at the sample set"}


def _error_entry(spec, exc: Exception, stage: str) -> dict:
    """The `bore_facts` entry of a bore that stopped with an exception: the name, the error type and its text, and an option. It has none of the funnel keys."""
    message = " ".join(f"{type(exc).__name__}: {exc}".split())
    facts = [f"The bore '{spec.name}' stopped at the {STAGES[stage]}: {message}. The bore is not checked; the other bores and the rest of the report go on."]
    options = ["Run the check again. If the message is the same, check point, axis, radius_mm and length_mm of the bore against the verified part "
               "(the file is in run.verified_file of the report), and tell the maintainers of topopt about the message.",
               "Remove this bore from the `bore_checks` block of the YAML, or remove the block (or use `--no-bore-checks`) to leave the check off."]
    return dict(name=spec.name, status="error", geometry_from=spec.geometry_from, stage=stage, message=message, facts=facts, options=options)


def build_bore(model, spec, knobs) -> Bore:
    g = find_facets(model.mesh, spec, knobs)
    ids = g.pop("ids")
    sample = SampleSet(model, ids, spec, knobs.sample_band_mm) if len(ids) else None
    if sample is not None and sample.n_qp == 0:
        sample = None
    g["ids"] = ids
    facts, options = _facts_and_options(spec, knobs, g, sample)
    del g["ids"]
    entry = dict(name=spec.name, status="ok" if sample is not None else "empty", geometry_from=spec.geometry_from, **g,
                 sample_set=None if sample is None else sample.info(), facts=facts, options=options)
    return Bore(spec, ids, sample, entry)


def build(model, bc) -> list:
    """The `Bore` of every bore of `Config.bore_checks` on this mesh. A bore whose build raises is dropped with a fact (`Bore.fail`); the others go on."""
    out = []
    for spec in bc.bores:
        try:
            out.append(build_bore(model, spec, bc.knobs))
        except Exception as exc:     # a failure of one bore is a fact in its entry, not a stop of verify
            b = Bore(spec, np.zeros(0, dtype=np.int64), None, {})
            b.fail(exc, "build")
            out.append(b)
    return out
