"""Exact interfaces and machining stock (CONTRACTS.md section 23).

An interface is an exact shape that the post step unites with the final mesh or cuts out of it, after the smoothing:
a cylindrical bore (cut), a boss (union), a tube (a ring union, then a cut of its hole), a plane face (cut), or an STL solid
(union or cut). The shapes are built with manifold3d, so a bore has its radius and a face has its position, not the voxel
staircase. A keep-out region wins over a union shape (the label rule): the step cuts the region STL files again after the unions.
A face cuts the whole half-space on the side of its normal inside the grid box. With `radius_mm` it cuts only the disc (a cylinder
about the line through `point` along the normal, from the plane to the grid box). `voxelize.taken_facts` counts what each cut takes
from the frozen regions before the run. After the run, `integrity` measures what stays of each frozen region in the finished mesh,
`measure` gives the complete-surface share of an interface with `full_surface`, and `stock_effect_warnings` finds a stock that has
no effect.

Machining stock: with `stock_mm` above 0 the step makes two meshes from the same smooth mesh. The as-printed mesh
(`result/final.stl`) leaves the stock on each machined surface: a bore is smaller, a boss or tube is larger, a face moves
into the cut side. The nominal mesh (`result/nominal.stl`) is the finished shape. The optimizer sees the part of each
shape that is the same in both meshes (voxel labels): bores and faces as in the as-printed mesh, bosses and tubes as nominal.

The method is ported from the s9 print pass of the pipeline (`build_c3.build`: union the exact solids, subtract one union of
cutters, nominal and as-printed variants by one stock value). The shapes, the YAML and the labels are new and general. The
parts that are not general (keys, steerer profile, material table, fillet fill) are not ported. The press fit and the Goodman check are not in this module: they are
checks of `topopt verify` (CONTRACTS.md section 40).
"""
from __future__ import annotations

import io
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Interface
from .resource_errors import reraise_if_resource

log = logging.getLogger("topopt.interfaces")

NOMINAL, AS_PRINTED, LABEL = "nominal", "as_printed", "label"
SAG_MM = 0.002          # largest chord sag of a cylinder facet (the wall of a facet lies up to this far inside the radius)
MIN_SEGMENTS = 16
BOX_MARGIN_MM = 2.0     # a face cutter reaches this far past the grid box (plus the stock)
FEATURE_TOL_MM = 1e-4   # added to the sag: float32 resolution of the STL and rounding of the shapes
PLANE_WINDOW_MM = 0.05  # a face is searched within this distance of its expected position
PLANE_FLAT_MM = 5e-5    # a triangle of a cut plane has its three vertices this close to one plane (float32 noise is under 2e-5)
CLEAN_TOL_MM = 1e-5     # manifold3d simplify tolerance, as in smooth_sdf (see pitfall float32-stl-reload-pinch)
SHELL_MM = 0.0005       # thickness of the shell next to a wall that gives the expected wall area (volume of the shell / thickness)
COVERAGE_MIN = 0.8      # a measured surface must cover this share of its expected area to be "within tolerance"
AREA_MIN_MM2 = 1e-2     # an expected area below this is no surface to find (a shape in empty space): no coverage
KEPT_MIN = 0.97         # a frozen region keeps at least this share of its volume (check C; under review)
FULL_COVERAGE_MIN = 0.98   # a surface with `full_surface` shows at least this share of its full area (check D)
STOCK_EFFECT_MIN = 0.05    # the as-printed surface has at least this share of the finished area, else the stock has no effect (check E)


def segments(radius: float, sag: float = SAG_MM) -> int:
    """Number of segments of a circle so that the chord sag `r (1 - cos(pi/n))` is at most `sag` (even, at least 16)."""
    if radius <= sag:
        return MIN_SEGMENTS
    n = math.ceil(math.pi / math.acos(1.0 - sag / radius))
    return max(MIN_SEGMENTS, n + (n % 2))


def facet_sag(radius: float) -> float:
    """The chord sag of the facets that `segments` gives for this radius (mm)."""
    return radius * (1.0 - math.cos(math.pi / segments(radius)))


def _frame(axis) -> np.ndarray:
    """3 x 3 matrix with the columns u, v, a (a right-handed frame; `a` is the unit axis)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    e = np.zeros(3)
    e[int(np.argmin(np.abs(a)))] = 1.0
    u = np.cross(a, e)
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(a, u), a])


def _arrays(man):
    """float32-valued (verts float64, faces int64) of a manifold. Return None for an empty manifold."""
    if man.is_empty():
        return None
    me = man.to_mesh()
    return np.asarray(me.vert_properties, np.float64)[:, :3], np.asarray(me.tri_verts, np.int64)


# -- the shapes -------------------------------------------------------------------------------------
def _cylinder(point, axis, r_out: float, r_in, length: float):
    import manifold3d as m3d

    n = segments(r_out)
    if r_in is None:
        man = m3d.Manifold.cylinder(length, r_out, r_out, n)
    else:
        cs = m3d.CrossSection.circle(r_out, n) - m3d.CrossSection.circle(r_in, segments(r_in))
        man = m3d.Manifold.extrude(cs, length)
    return man.transform(np.column_stack([_frame(axis), np.asarray(point, float)]))


def _ring(point, axis, r_in: float, r_out: float, length: float):
    """A thin tube with the same number of segments on both walls (a shell of uniform thickness next to a wall)."""
    import manifold3d as m3d

    n = segments(r_in)
    cs = m3d.CrossSection.circle(r_out, n) - m3d.CrossSection.circle(r_in, n)
    return m3d.Manifold.extrude(cs, length).transform(np.column_stack([_frame(axis), np.asarray(point, float)]))


def _cutter_of_face(point, normal, offset: float, box, radius=None):
    """The part of the grid box (and a margin) that lies beyond the plane `point + offset * normal` (the side of `normal`).

    With `radius` (the `radius_mm` of a face) the cutter is only the cylinder of that radius about the line through `point`
    along `normal`: from the plane to the margin past the far corner of the grid box. Without it, the cutter is the whole half-space.
    """
    import manifold3d as m3d

    n = np.asarray(normal, float)
    lo, hi = (np.asarray(b, float) for b in box)
    if radius is not None:
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        length = float(((corners - np.asarray(point, float)) @ n).max()) + BOX_MARGIN_MM - offset
        if length <= 0.0:   # the whole box lies before the plane: the disc cuts nothing
            return m3d.Manifold()
        man = m3d.Manifold.cylinder(length, radius, radius, segments(radius))
        return man.transform(np.column_stack([_frame(n), np.asarray(point, float) + offset * n]))
    m = BOX_MARGIN_MM + offset
    cube = m3d.Manifold.cube(tuple(hi - lo + 2 * m)).translate(tuple(lo - m))
    return cube.trim_by_plane(tuple(n), float(np.dot(point, n) + offset))


def pieces(spec: Interface) -> list[tuple[str, str]]:
    """The steps of one interface as `(op, part)`. A tube is a ring (union) and its hole (cut); every other kind is one step."""
    if spec.kind == "tube":
        return [("union", "ring"), ("cut", "hole")]
    return [(spec.op, "main")]


def piece_key(spec: Interface, part: str) -> str:
    """The name of a step in the dicts of the labels and the region checks (the hole of a tube is `<name> (hole)`)."""
    return spec.name if part in ("main", "ring") else f"{spec.name} ({part})"


def build_shape(spec: Interface, stock: float, box=None, part: str = "main"):
    """The manifold (float64) of an interface (or of one part of it) with `stock` mm of stock. `box` is the grid box; a face needs it."""
    if spec.kind == "bore":
        return _cylinder(spec.point, spec.axis, spec.radius_mm - stock, None, spec.length_mm)
    if spec.kind == "boss":
        return _cylinder(spec.point, spec.axis, spec.radius_mm + stock, None, spec.length_mm)
    if spec.kind == "tube":
        if part == "hole":   # the hole: a full cylinder of the as-printed inner radius, cut after the unions
            return _cylinder(spec.point, spec.axis, spec.inner_radius_mm - stock, None, spec.length_mm)
        return _cylinder(spec.point, spec.axis, spec.radius_mm + stock, spec.inner_radius_mm - stock, spec.length_mm)
    if spec.kind == "face":
        return _cutter_of_face(spec.point, spec.axis, stock, box, spec.radius_mm)
    raise ValueError(f"no built-in shape for kind {spec.kind!r}")


def _stock_of(spec: Interface, which: str, op: str | None = None) -> float:
    """Stock used for a shape: as-printed has it, nominal has none, the labels use the part that both meshes share.

    `op` is the operation of the step (a tube has a union step and a cut step); it defaults to the one of the interface.
    """
    if spec.kind == "stl" or which == NOMINAL:
        return 0.0
    if which == AS_PRINTED or (which == LABEL and (op or spec.op) == "cut"):
        return spec.stock_mm
    return 0.0


class Shapes:
    """The exact shapes of a list of interfaces on one grid box. Each shape is built once, as a float32 mesh."""

    def __init__(self, specs, box):
        self.specs = list(specs)
        self.box = (np.asarray(box[0], float), np.asarray(box[1], float))
        self._mesh: dict = {}
        self._man: dict = {}

    def arrays(self, spec: Interface, which: str, part: str = "main"):
        """(verts, faces) of the shape, float32-valued, or None if the shape is empty (a face beyond the part)."""
        s = _stock_of(spec, which, dict((p, o) for o, p in pieces(spec))[part])
        key = (spec.name, part, s)
        if key not in self._mesh:
            if spec.kind == "stl":
                import trimesh

                t = trimesh.load(spec.path, file_type="stl", force="mesh")
                t.merge_vertices()
                self._mesh[key] = (np.asarray(t.vertices, np.float32).astype(np.float64), np.asarray(t.faces, np.int64))
            else:
                self._mesh[key] = _arrays(build_shape(spec, s, self.box, part))
        return self._mesh[key]

    def manifold(self, spec: Interface, which: str, part: str = "main"):
        from .post import _to_manifold

        s = _stock_of(spec, which, dict((p, o) for o, p in pieces(spec))[part])
        key = (spec.name, part, s)
        if key not in self._man:
            arr = self.arrays(spec, which, part)
            self._man[key] = None if arr is None else _to_manifold(*arr)
        return self._man[key]

    def trimesh(self, spec: Interface, which: str, part: str = "main"):
        import trimesh

        arr = self.arrays(spec, which, part)
        return None if arr is None else trimesh.Trimesh(arr[0], arr[1], process=False)


# -- labels (called by voxelize) --------------------------------------------------------------------
def _shapes_for(config, box) -> Shapes:
    return Shapes(config.interfaces, box)


def add_meshes(config) -> dict:
    """Name -> trimesh of each union step, as the optimizer sees it (nominal size). They widen the grid like a region."""
    out = {}
    if not config.interfaces:
        return out
    from .voxelize import load_mesh

    sh = _shapes_for(config, ((0, 0, 0), (1, 1, 1)))   # a union shape does not depend on the grid box
    for spec in config.interfaces:
        for op, part in pieces(spec):
            if op == "union":
                out[piece_key(spec, part)] = load_mesh(spec.path) if spec.kind == "stl" else sh.trimesh(spec, LABEL, part)
    return out


def cut_meshes(config, box) -> dict:
    """Name -> trimesh of each cut step, as the optimizer sees it (the as-printed size). They do not widen the grid.

    The hole of a tube is a cut step with the key `<name> (hole)`.
    """
    out = {}
    if not config.interfaces:
        return out
    from .voxelize import load_mesh

    sh = _shapes_for(config, box)
    for spec in config.interfaces:
        for op, part in pieces(spec):
            if op != "cut":
                continue
            m = load_mesh(spec.path) if spec.kind == "stl" else sh.trimesh(spec, LABEL, part)
            if m is not None:
                out[piece_key(spec, part)] = m
    return out


def render_shapes(config, region_bounds) -> tuple[dict, dict, list[str]]:
    """The meshes of the interface shapes for `topopt render` of a YAML: `(unions, cuts, notes)`.

    `unions` and `cuts` are dicts name -> `(verts, faces)` (a tube is a union ring and a cut hole `<name> (hole)`). They are the
    shapes of the labels (`add_meshes`, `cut_meshes`). `region_bounds` is `(lo, hi)`, the bounds of all region STLs; with the union
    shapes it gives the grid box. A cut is clipped to the grid box, so a face shows its cut zone in the box and not the margin that
    the exact step uses. A cut that cannot be clipped is drawn as it is, and `notes` says so.
    """
    unions = {k: (np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)) for k, m in add_meshes(config).items()}
    lo, hi = (np.asarray(b, float) for b in region_bounds)
    for v, _f in unions.values():
        lo, hi = np.minimum(lo, v.min(axis=0)), np.maximum(hi, v.max(axis=0))
    cuts, notes = {}, []
    for k, m in cut_meshes(config, (lo, hi)).items():
        v, f = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
        try:
            import manifold3d as m3d
            from .post import _to_manifold

            box = m3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
            arr = _arrays(_to_manifold(v, f) ^ box)
            if arr is not None:
                v, f = arr
        except Exception as exc:   # draw the shape whole
            notes.append(f"interface '{k}': the cut shape could not be clipped to the grid box ({type(exc).__name__}: {exc}); it is drawn whole")
        cuts[k] = (v, f)
    return unions, cuts, notes


def voxel_facts(config, labels, origin, h: float, shape, support=None) -> list[dict]:
    """Facts about the voxels of each interface step, for `topopt check` and `topopt run`.

    One dict for each step: `name` (the hole of a tube is `<name> (hole)`), `kind`, `op`, `voxels` (inside the shape),
    `held` (of them, the voxels that carry the label of the shape: LOAD for a union, KEEP_OUT for a cut; a keep-out region
    or a fixed or load region can win a voxel). A union step also has `piece_voxels` and `has_fixed`: the number of active
    voxels in the 6-connected piece that it is in, and whether that piece has a fixed voxel. A piece without a fixed
    voxel makes the stiffness matrix singular. `support` (bool, grid shape) names the voxels that hold the part; the
    default is the FIXED voxels. With springs it is `springs.support_mask(problem)`: a spring region holds a piece too.
    """
    if not config.interfaces:
        return []
    from scipy import ndimage

    from .voxelize import DESIGN, FIXED, KEEP_OUT, LOAD, interface_add_meshes, interface_masks

    union, cut = interface_masks(config, interface_add_meshes(config), origin, h, shape, warn=False)
    labels = np.asarray(labels)
    active = (labels >= DESIGN) & (labels <= LOAD)
    comp, n = ndimage.label(active)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    fixed_comp = set(np.unique(comp[(labels == FIXED) if support is None else support]).tolist())
    out = []
    for spec in config.interfaces:
        for op, part in pieces(spec):
            key = piece_key(spec, part)
            mk = (union if op == "union" else cut)[key]
            f = {"name": key, "kind": spec.kind, "op": op, "voxels": int(mk.sum())}
            if op == "union":
                own = mk & (labels == LOAD)
                f["held"] = int(own.sum())
                ids = set(np.unique(comp[own]).tolist()) - {0}
                f["piece_voxels"] = int(sum(sizes[i] for i in ids))
                f["has_fixed"] = bool(ids & fixed_comp) if ids else True
            else:
                f["held"] = int((mk & (labels == KEEP_OUT)).sum())
            out.append(f)
    return out


def floating_message(facts) -> str | None:
    """The error text for a union step whose piece of active voxels has no fixed voxel, or None."""
    bad = [f for f in facts if f["op"] == "union" and f["held"] and not f["has_fixed"]]
    if not bad:
        return None
    lines = [f"interface '{f['name']}' ({f['kind']}) holds {f['held']} voxels in a piece of {f['piece_voxels']} active voxels "
             "that has no fixed voxel" for f in bad]
    return ("; ".join(lines) + ": the stiffness matrix is singular and the solver cannot run. "
            "Options: move the shape so that it touches the part (a design, fixed or load voxel), or remove it from `interfaces`")


# -- the run ------------------------------------------------------------------------------------------
def specs_from_run(run: dict, yaml_dir=None) -> list[Interface]:
    """The `interfaces` key of run.json as a list of `Interface`. An STL is searched at the stored path, then in the YAML folder.

    The YAML folder is that of `run["yaml_path"]` as it is: give `runinfo.resolved_run(run, run_dir)` for a relative path."""
    if yaml_dir is None:
        yaml_dir = Path(run.get("yaml_path", "")).parent
    out = []
    for i, d in enumerate(run.get("interfaces") or []):
        spec = Interface.from_dict(d, yaml_dir, f"interfaces[{i}]")
        if spec.kind == "stl" and not spec.path.is_file() and (Path(yaml_dir) / spec.path.name).is_file():
            spec = Interface(spec.name, spec.kind, spec.op, 0.0, path=Path(yaml_dir) / spec.path.name)
        out.append(spec)
    return out


@dataclass
class Context:
    """The interfaces of one run: the specs, the grid box and the shapes.

    `frozen_paths` are the STL files of the fixed and load regions, if all of them exist (else None). The exact step
    uses them to find bodies that no frozen shape touches. `keep_out_paths` are the keep-out region STL files that exist,
    `keep_out_missing` the names of those that do not. A union step is cut by the keep-out regions (the label rule).
    """
    specs: list
    box: tuple
    shapes: Shapes
    has_stock: bool
    frozen_paths: list | None = None
    keep_out_paths: list = field(default_factory=list)
    keep_out_missing: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    near_mm: float = 0.0      # the exact step also judges a body of the method mesh that lies this close to a union shape
    regions: list = field(default_factory=list)   # (role, name, path) of the fixed, load and solid region STL files that exist (check C)
    taken: list = field(default_factory=list)     # the `interface_taken` facts of run.json (check A): they name the cuts that overlap a region
    _keep: object = None

    def steps(self, op: str):
        """`(spec, part)` of every step with this operation, in the order of the list."""
        return [(s, part) for s in self.specs for o, part in pieces(s) if o == op]

    def pin_meshes(self):
        """(frozen, cut): trimeshes for the SDF step. Frozen: unions as nominal. Cut: cutters as the labels have them."""
        fz = [m for s, part in self.steps("union") if (m := self.shapes.trimesh(s, LABEL, part)) is not None]
        ct = [m for s, part in self.steps("cut") if (m := self.shapes.trimesh(s, LABEL, part)) is not None]
        return fz, ct

    def union_meshes(self, which: str = AS_PRINTED) -> list:
        """The trimeshes of the union steps (a boss, the wall of a tube, an STL union) at the size of `which`. The envelope of the SDF step has them."""
        return [m for s, part in self.steps("union") if (m := self.shapes.trimesh(s, which, part)) is not None]

    def cutters(self, which: str = AS_PRINTED) -> dict:
        """Name -> trimesh of each cutter, for the region checks (`interface:<name>`)."""
        out = {}
        for s, part in self.steps("cut"):
            if (m := self.shapes.trimesh(s, which, part)) is not None:
                out[f"interface:{piece_key(s, part)}"] = m
        return out

    def removed(self, which: str):
        """Manifold of what the cuts and the keep-out regions take away in this variant, or None if there is none.

        The expected wall area of a boss (`expected_area`) leaves this out: a cut that removes a part of a boss wall does not
        make the wall missing.
        """
        mans = [m for s, part in self.steps("cut") if (m := self.shapes.manifold(s, which, part)) is not None]
        keep = self.keep_manifold()
        if keep is not None:
            mans.append(keep)
        if not mans:
            return None
        out = mans[0]
        for m in mans[1:]:
            out = out + m
        return out

    def keep_manifold(self):
        """The union of the keep-out region STL files that exist, or None."""
        if self._keep is None and self.keep_out_paths:
            from .smooth_sdf import _load_manifold

            mans = [_load_manifold(q) for q in self.keep_out_paths]
            self._keep = mans[0]
            for m in mans[1:]:
                self._keep = self._keep + m
        return self._keep


def from_run(run: dict, origin, h: float, shape) -> Context | None:
    """The `Context` of a run, or None if the run has no interfaces."""
    specs = specs_from_run(run)
    if not specs:
        return None
    from .post import UPSAMPLE, check_regions, region_paths

    lo = np.asarray(origin, float)
    box = (lo, lo + float(h) * np.asarray(shape, float))
    fixed, load, keep = region_paths(run)
    files = list(check_regions(fixed, load).values())
    frozen = files if files and all(Path(q).is_file() for q in files) else None
    ctx = Context(specs, box, Shapes(specs, box), any(s.kind != "stl" and s.stock_mm > 0 for s in specs), frozen,
                  [Path(q) for q in keep.values() if Path(q).is_file()],
                  [Path(q).name for q in keep.values() if not Path(q).is_file()], near_mm=float(h) / UPSAMPLE)
    for k, q in fixed.items():
        if Path(q).is_file():
            ctx.regions.append(("fixed", Path(k).stem, Path(q)))
    for k, q in load.items():
        if Path(q).is_file():
            ctx.regions.append(("solid", Path(k[6:]).stem, Path(q)) if k.startswith("solid:") else ("load", k, Path(q)))
    ctx.taken = list(run.get("interface_taken") or [])
    for s, part in ctx.steps("union"):   # a union shape must lie in the grid box (the loader makes the grid hold it)
        arr = ctx.shapes.arrays(s, LABEL, part)
        outside = arr is not None and bool((arr[0].min(axis=0) < box[0] - 1e-6).any() or (arr[0].max(axis=0) > box[1] + 1e-6).any())
        if outside:
            note = (f"interface '{s.name}' lies partly outside the grid box of the run: the SDF step and the labels do not hold "
                    "the part outside; the exact step still unites it. Run the YAML again so that the grid holds it")
            ctx.notes.append(note)
            log.warning(note)
    return ctx


# -- the exact step --------------------------------------------------------------------------------------
@dataclass
class Result:
    """What the exact step made. `variants` has `as_printed` (the final mesh) and, with stock, `nominal`."""
    context: Context
    applied: bool = False
    error: str | None = None
    variants: dict = field(default_factory=dict)    # name -> (verts, faces)
    deltas: dict = field(default_factory=dict)      # variant -> {interface name: volume change, mm3}
    clean: dict = field(default_factory=dict)       # variant -> facts of the float32 clean-up
    dropped: dict = field(default_factory=dict)     # variant -> {"bodies", "volume_mm3"} removed (no frozen shape touches them)
    keep_out: dict = field(default_factory=dict)    # variant -> {"files", "removed_mm3"}: the keep-out regions cut from the unions
    t_s: dict = field(default_factory=dict)


def _cut_keep_out(ctx: Context, which: str, body, vol: float):
    """Cut the keep-out regions out of the body after the unions (a keep-out region wins over a union shape).

    Return `(body, volume, info)`. `info` is None if no union shape touches a keep-out region (nothing to cut): the method
    cut the regions already. The test is cheap (the union shapes against the regions, not the whole body).
    """
    unions = [m for s, part in ctx.steps("union") if (m := ctx.shapes.manifold(s, which, part)) is not None]
    keep = ctx.keep_manifold()
    if not unions or keep is None:
        return body, vol, None
    if sum((u ^ keep).volume() for u in unions) <= OVERLAP_MM3:
        return body, vol, None
    body = body - keep
    v = body.volume()
    return body, v, {"files": len(ctx.keep_out_paths), "removed_mm3": round(vol - v, 4)}


def _variant(base, ctx: Context, which: str):
    """(manifold, deltas, keep_out): the unions in the listed order, the keep-out regions, then the cuts.

    `deltas` is the volume change of each interface (a tube: ring and hole together).
    """
    body, vol, deltas = base, base.volume(), {}

    def run(body, vol, op):
        for spec, part in ctx.steps(op):
            man = ctx.shapes.manifold(spec, which, part)
            if man is None:
                deltas.setdefault(spec.name, 0.0)
                continue
            body = body + man if op == "union" else body - man
            v = body.volume()
            deltas[spec.name] = deltas.get(spec.name, 0.0) + (v - vol)
            vol = v
        return body, vol

    body, vol = run(body, vol, "union")
    body, vol, keep = _cut_keep_out(ctx, which, body, vol)
    body, vol = run(body, vol, "cut")
    return body, deltas, keep


OVERLAP_MM3 = 1e-6   # a body shares volume with a union shape if the common volume is above this
TOUCH_MM = 1e-3   # a body touches a frozen shape if the gap between them is under this (a float32 file has 1e-5 or better)


def _vertex_keys(verts32) -> np.ndarray:
    """One uint64 key for each float32 vertex (a hash of the bits). Equal vertices give equal keys."""
    a = np.ascontiguousarray(verts32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (a[:, 0] * np.uint64(0x9E3779B97F4A7C15)) ^ (a[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)) ^ (a[:, 2] * np.uint64(0x165667B19E3779F9))


def _split_off(parts, base) -> set:
    """Indices of the bodies in `parts` that the booleans made: they are the only ones the floating-body rule may remove.

    The method mesh `base` can hold several bodies (islands that the `sdf` step keeps within 0.35 fine voxel of a frozen region).
    A body of the result that shares a vertex with a base body, and is the only descendant of every base body it comes from,
    is that base body, not changed in topology: it stays, as it does in a run without interfaces. Booleans keep the
    vertices that they do not change, so the shared vertices tell which base body a piece comes from. A piece with no
    base body (a new body of a union shape) and a piece of a base body that is now in two or more pieces are judged.
    If the base mesh is one body, all pieces are judged.
    """
    base_parts = base.decompose()
    if len(base_parts) <= 1:
        return set(range(len(parts)))
    keys_b = [np.unique(_vertex_keys(np.asarray(b.to_mesh().vert_properties, np.float32)[:, :3])) for b in base_parts]
    anc = []
    for p in parts:
        kp = _vertex_keys(np.asarray(p.to_mesh().vert_properties, np.float32)[:, :3])
        anc.append([j for j, kb in enumerate(keys_b) if np.isin(kp, kb).any()])
    count = [sum(j in a for a in anc) for j in range(len(base_parts))]
    return {i for i, a in enumerate(anc) if not a or any(count[j] >= 2 for j in a)}


def _drop_floating(body, ctx: Context, which: str, base):
    """Remove the bodies that the booleans split off and that no frozen shape touches. Return (manifold, {"bodies": n, "volume_mm3": v}).

    Frozen shapes: the fixed and load STL files (gap rule), and the union interfaces (shared volume). It is the rule of the
    `sdf` step (a body that touches no frozen region is removed), applied only to the pieces that the exact step makes
    (`_split_off`) and to the bodies of the method mesh that lie within `Context.near_mm` (one fine voxel) of a union shape
    (the `sdf` step pins the union shapes, so it keeps an island next to one; the exact union can leave it floating). Any other
    body of the method mesh stays as the method left it, so the same run with and without an interface gives the same bodies
    away from the interfaces. If no judged body touches, the largest body stays.
    Without all fixed and load files (`Context.frozen_paths` is None) nothing is removed: the facts show the bodies.
    """
    import manifold3d as m3d

    from .smooth_sdf import _load_manifold

    out = {"bodies": 0, "volume_mm3": 0.0}
    if ctx.frozen_paths is None:
        return body, out
    parts = body.decompose()
    if len(parts) <= 1:
        return body, out
    judged = _split_off(parts, base)
    unions = [m for s, part in ctx.steps("union") if (m := ctx.shapes.manifold(s, which, part)) is not None]
    if ctx.near_mm > 0 and unions:   # the sdf step keeps an island next to a union shape (it pins the shape); the exact union can leave it floating
        judged |= {i for i, p in enumerate(parts) if i not in judged and any(p.min_gap(u, ctx.near_mm) < ctx.near_mm for u in unions)}
    if not judged:
        return body, out
    frozen = [_load_manifold(q) for q in ctx.frozen_paths]

    def touches(p) -> bool:
        # a region file: the gap rule of the sdf step. A union shape: the body must share volume with it (the union
        # already merged every body that did; a body that only kisses the shape at an edge is still a separate body).
        return (any(p.min_gap(f, TOUCH_MM) < TOUCH_MM for f in frozen)
                or any((p ^ u).volume() > OVERLAP_MM3 for u in unions))

    keep = [i for i in range(len(parts)) if i not in judged or touches(parts[i])]
    if not keep:
        keep = [max(range(len(parts)), key=lambda i: parts[i].volume())]
    gone = [parts[i] for i in range(len(parts)) if i not in keep]
    if not gone:
        return body, out
    out = {"bodies": len(gone), "volume_mm3": round(float(sum(p.volume() for p in gone)), 4)}
    log.warning("interface step removed %d body(ies) of %.3f mm3 in the %s mesh that no frozen shape touches "
                "(pieces that the booleans split off, and bodies next to a union shape)",
                out["bodies"], out["volume_mm3"], which)
    return m3d.Manifold.compose([parts[i] for i in keep]), out


def apply_step(ctx: Context, verts, faces):
    """Unite and cut the exact shapes. Return `(verts, faces, Result)`; the mesh is the as-printed mesh.

    On any error the mesh comes back unchanged, one warning is logged, and `Result.applied` is false.
    """
    from . import smooth_sdf

    res = Result(ctx)
    t0 = time.monotonic()
    try:
        base = smooth_sdf._manifold_of(np.asarray(verts, np.float64), np.asarray(faces, np.int64))
        res.t_s["base"] = round(time.monotonic() - t0, 3)
        which = [AS_PRINTED, NOMINAL] if ctx.has_stock else [AS_PRINTED]   # without stock the two are the same shape
        if ctx.keep_out_missing and ctx.steps("union"):
            log.warning("keep-out region file(s) %s not found: a union interface that overlaps such a region is not cut back "
                        "from it in the exact step (the labels hold the region as KEEP_OUT)", ", ".join(ctx.keep_out_missing))
        for w in which:
            t1 = time.monotonic()
            body, res.deltas[w], keep = _variant(base, ctx, w)
            if keep is not None:
                res.keep_out[w] = keep
            body, res.dropped[w] = _drop_floating(body, ctx, w, base)
            v, f = smooth_sdf._from_manifold(body)
            v, f = smooth_sdf._orient(v, f)
            v = v.astype(np.float32).astype(np.float64)
            v, f, res.clean[w] = smooth_sdf._cleanup_float32(v, f, CLEAN_TOL_MM)
            res.variants[w] = (v, f)
            res.t_s[w] = round(time.monotonic() - t1, 3)
    except Exception as exc:   # keep the mesh of the method; the record says what happened
        reraise_if_resource(exc, "the exact interface shapes (union and cut)")
        res.error = f"{type(exc).__name__}: {exc}"
        res.variants, res.deltas, res.clean, res.dropped, res.keep_out = {}, {}, {}, {}, {}
        log.warning("interface step failed, the mesh has no exact interfaces: %s", res.error)
        res.t_s["total"] = round(time.monotonic() - t0, 3)
        return verts, faces, res
    res.applied = True
    res.t_s["total"] = round(time.monotonic() - t0, 3)
    v, f = res.variants[AS_PRINTED]
    return v, f, res


# -- measurement of a mesh against the spec -----------------------------------------------------------
def _face_data(V, F):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(n, axis=1)
    ok = a2 > 1e-12
    nrm = np.zeros_like(n)
    nrm[ok] = n[ok] / a2[ok, None]
    return nrm, a2 / 2.0


def _measure_radius(V, F, point, axis, radius: float, length: float, outward: bool) -> dict:
    """Find the cylinder wall of radius `radius` and report its mean radius and the area found.

    The facets lie up to the sag inside the radius. A face counts if its normal is across the axis and on the right side:
    away from the axis for the outer wall of a boss or tube (`outward`), toward the axis for a hole (the material is outside).
    """
    a = np.asarray(axis, float)
    d = V - np.asarray(point, float)
    t = d @ a
    rho = np.linalg.norm(d - t[:, None] * a, axis=1)
    sag = facet_sag(radius)
    band = 2 * FEATURE_TOL_MM
    on = (rho >= radius - sag - band) & (rho <= radius + band) & (t >= -band) & (t <= length + band)
    nrm, area = _face_data(V, F)
    sel = on[F].all(axis=1) & (np.abs(nrm @ a) < 0.05) & (area > 0)
    if sel.any():
        cen = V[F[sel]].mean(axis=1) - np.asarray(point, float)
        rad = cen - (cen @ a)[:, None] * a
        side = np.einsum("ij,ij->i", nrm[sel], rad) / np.maximum(np.linalg.norm(rad, axis=1), 1e-12)
        keep = side > 0.5 if outward else side < -0.5
        idx = np.flatnonzero(sel)
        sel = np.zeros_like(sel)
        sel[idx[keep]] = True
    tol = sag + FEATURE_TOL_MM
    out = {"expected_mm": round(radius, 6), "tolerance_mm": round(tol, 6), "found": bool(sel.any())}
    if sel.any():
        ids = np.unique(F[sel])
        out.update({"measured_mm": round(float(rho[ids].mean()), 6), "max_mm": round(float(rho[ids].max()), 6),
                    "min_mm": round(float(rho[ids].min()), 6), "area_mm2": round(float(area[sel].sum()), 3),
                    "faces": int(sel.sum())})
        out["deviation_mm"] = round(out["measured_mm"] - radius, 6)
        out["within_tolerance"] = bool(abs(out["deviation_mm"]) <= tol)
    else:
        out["within_tolerance"] = False
    return out


def _measure_plane(V, F, point, normal, offset: float, radius=None) -> dict:
    """Find the planar faces at `offset` mm from the plane through `point` (along `normal`) and report their mean position.

    A triangle counts if its normal is `normal` (a face cut leaves the outward normal on the side of the cutter), its three
    vertices lie within `PLANE_FLAT_MM` of one plane, and that plane is within `PLANE_WINDOW_MM` of `offset`. The flat test
    keeps out the faces of the smooth surface that are nearly parallel. The normal test keeps out a wall that faces the
    other way (the wall of a thin bore on the same plane). With `radius` (a bounded face) only the triangles inside the disc count.
    """
    n = np.asarray(normal, float)
    d = (V - np.asarray(point, float)) @ n
    nrm, area = _face_data(V, F)
    sel = ((np.abs(d[F] - offset) <= PLANE_WINDOW_MM).all(axis=1) & (nrm @ n > 0.9999) & (area > 0)
           & (np.ptp(d[F], axis=1) <= PLANE_FLAT_MM))
    if radius is not None and sel.any():
        off = (V - np.asarray(point, float)) - d[:, None] * n
        sel &= (np.linalg.norm(off, axis=1)[F] <= radius + FEATURE_TOL_MM).all(axis=1)
    out = {"expected_mm": round(offset, 6), "tolerance_mm": FEATURE_TOL_MM, "found": bool(sel.any())}
    if sel.any():
        pos = float(np.average(d[F][sel].mean(axis=1), weights=area[sel]))
        out.update({"measured_mm": round(pos, 6), "area_mm2": round(float(area[sel].sum()), 3), "faces": int(sel.sum())})
        out["deviation_mm"] = round(pos - offset, 6)
        out["within_tolerance"] = bool(abs(out["deviation_mm"]) <= FEATURE_TOL_MM)
    else:
        out["within_tolerance"] = False
    return out


def mesh_body(V, F):
    """The manifold of a mesh (float32 input, as a slicer reads the file), or None if the mesh is not a closed manifold."""
    from .post import _to_manifold

    try:
        return _to_manifold(np.asarray(V, np.float64), np.asarray(F, np.int64))
    except Exception:
        return None


def _features(spec: Interface, which: str) -> list[tuple[str, str, float]]:
    """`(key, role, size)` of each surface that `measure` looks for. Role: `hole` (the wall faces the axis, material outside),
    `outer` (the wall faces away from the axis) or `plane` (a face cut). Size: the radius, or the offset of the plane."""
    s = _stock_of(spec, which)
    if spec.kind == "bore":
        return [("radius_mm", "hole", spec.radius_mm - s)]
    if spec.kind == "boss":
        return [("radius_mm", "outer", spec.radius_mm + s)]
    if spec.kind == "tube":
        return [("radius_mm", "outer", spec.radius_mm + s), ("inner_radius_mm", "hole", spec.inner_radius_mm - s)]
    if spec.kind == "face":
        return [("position_mm", "plane", s)]
    return []


def expected_area(spec: Interface, role: str, size: float, body, box, removed=None) -> float | None:
    """Area (mm2) of the exact surface that the mesh `body` (a manifold) should show, from the mesh itself.

    A thin shell of `SHELL_MM` next to the wall has the volume area x thickness. For a hole wall and a face cut the shell
    lies in the material (outside the hole, under the plane): the area is the volume of the body in the shell, so the area
    is where the surface has material behind it. For the outer wall of a boss or tube the shell lies in the free space:
    the area is the volume of the shell that the body does not fill, so a missing boss gives its whole wall. The area does
    not depend on how much of the surface the mesh really shows. `removed` (a manifold: the cuts and the keep-out regions of
    the variant) is left out of the free space of an outer wall: a cut that takes a part of the wall away does not make the
    wall missing. Return None if `body` is None.
    """
    if body is None:
        return None
    t = SHELL_MM
    if role == "plane":
        slab = (_cutter_of_face(spec.point, spec.axis, size - t, box, spec.radius_mm)
                - _cutter_of_face(spec.point, spec.axis, size, box, spec.radius_mm))
        vol = (slab ^ body).volume()
    else:
        ring = _ring(spec.point, spec.axis, size, size + t, spec.length_mm)
        if role == "hole":
            vol = (ring ^ body).volume()
        else:
            free = ring - body
            vol = (free if removed is None else free - removed).volume()
    return float(vol) / t


def measure(spec: Interface, which: str, V, F, body=None, box=None, removed=None) -> dict:
    """Measure the finished surface of one built-in interface in a mesh. `which` is `as_printed` or `nominal`.

    Keys: `radius_mm` (bore, boss, tube outer wall), `inner_radius_mm` (tube), `position_mm` (face: the signed distance of the
    plane from the YAML point along the normal; it is the stock for the as-printed mesh). An STL solid has nothing to measure.
    With `body` (the manifold of the mesh, `mesh_body`) and `box` (the grid box; `removed` is `Context.removed(which)`) each
    feature also gets `expected_area_mm2` (see `expected_area`) and `coverage` (the area found divided by it; None if the surface has no expected area, for
    example a bore in empty space). A feature is `within_tolerance` only if its coverage is at least `COVERAGE_MIN`.
    `coverage` does not count a wall that is not there (the expected area is where the mesh has material). For an interface with
    `full_surface`, each feature also gets `full_area_mm2` (2 pi r L for a cylinder wall, pi r^2 for a bounded face) and
    `full_coverage` (`area_mm2` divided by it). An interface without `full_surface` has neither key.
    """
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    out = {}
    for key, role, size in _features(spec, which):
        if role == "plane":
            feat = _measure_plane(V, F, spec.point, spec.axis, size, spec.radius_mm)
        else:
            feat = _measure_radius(V, F, spec.point, spec.axis, size, spec.length_mm, outward=(role == "outer"))
        if body is not None and box is not None:
            area = expected_area(spec, role, size, body, box, removed)
            feat["expected_area_mm2"] = round(area, 3)
            if area >= AREA_MIN_MM2:
                feat["coverage"] = round(feat.get("area_mm2", 0.0) / area, 3)
                feat["within_tolerance"] = bool(feat["within_tolerance"] and feat["coverage"] >= COVERAGE_MIN)
            else:
                feat["coverage"] = None
        if spec.full_surface:   # check D: the area of the complete surface, and the share of it that the mesh shows
            full = math.pi * spec.radius_mm ** 2 if role == "plane" else 2.0 * math.pi * size * spec.length_mm
            feat["full_area_mm2"] = round(full, 3)
            feat["full_coverage"] = round(feat.get("area_mm2", 0.0) / full, 3)
        out[key] = feat
    return out


def _stock_between(spec: Interface, ap: dict, nom: dict) -> dict:
    """The stock that the two meshes show: nominal size minus as-printed size for a bore or an inner wall; as-printed minus nominal else."""
    out = {}
    for k in ap:
        if not (ap[k].get("found") and nom.get(k, {}).get("found")):
            out[k] = None
            continue
        a, n = ap[k]["measured_mm"], nom[k]["measured_mm"]
        out[k] = round(n - a if (spec.kind == "bore" or k == "inner_radius_mm") else a - n, 6)
    return out


def _load_stl_bytes(data: bytes):
    import trimesh

    t = trimesh.load(io.BytesIO(data), file_type="stl")
    return np.asarray(t.vertices, np.float64), np.asarray(t.faces, np.int64)


def mesh_facts(data: bytes, dens: float) -> dict:
    """Volume, mass and the reload check (`smooth_sdf.stl_facts`) of an STL file as written."""
    from . import smooth_sdf

    facts = smooth_sdf.stl_facts(data)
    return {"volume_mm3": facts["volume_mm3"], "mass_g": round(facts["volume_mm3"] / 1000.0 * dens, 2),
            "triangles": facts["triangles"], "stl_roundtrip": facts}


def _union_all(mans):
    out = None
    for m in mans:
        out = m if out is None else out + m
    return out


def _region_word(role: str) -> str:
    return "interface shape" if role == "interface" else f"{role} region"


def integrity(ctx: Context, V, F) -> list[dict]:
    """What stays of each frozen region in the finished mesh (check C; CONTRACTS.md section 23, "What stays of a frozen region").

    The frozen regions are the fixed, load and solid region STL files and the union interfaces (the nominal shapes of a boss, a tube
    ring and an STL union). For each one: `kept = volume(mesh and (region minus holes)) / volume(region minus holes)`. The holes are the
    nominal shapes of the `bore` interfaces and the tube holes: a hole that the YAML asks for is not a loss. The nominal shape of each face, STL cut or
    keep-out region that check A judged as a trim for this region (`warn` false) is taken out too. One boolean for each region.
    `V`, `F` are the finished mesh (`nominal.stl`, or `final.stl` when the run has no stock).

    One dict for each region: `region`, `role` (fixed, load, solid, interface), `volume_mm3` (region minus holes), `kept`,
    `lost_mm3`, `cuts` (the cuts of `interface_taken` that overlap it: `name`, `kind`, `role` = `hole`, `trim` or `loss`), `warn` (`kept < KEPT_MIN`). A region whose
    file cannot be read gives `kept: None` and a `note`; it is no warning. A mesh that is not a closed manifold gives one entry with
    `region: None` and a `note`.
    """
    from .smooth_sdf import _load_manifold

    body = mesh_body(V, F)
    if body is None:
        return [{"region": None, "role": None, "kept": None, "warn": False, "note": "the finished mesh is not a closed manifold"}]
    holes = _union_all(m for s, part in ctx.steps("cut") if (s.kind == "bore" or part == "hole")
                       and (m := ctx.shapes.manifold(s, NOMINAL, part)) is not None)
    cutters = {}   # name -> nominal shape of a cut that the YAML asks for: a face, an STL cut, a keep-out region
    for s_, part in ctx.steps("cut"):
        if s_.kind in ("face", "stl") and (m := ctx.shapes.manifold(s_, NOMINAL, part)) is not None:
            cutters[s_.name] = m
    for q in ctx.keep_out_paths:
        from .smooth_sdf import _load_manifold as _lm
        cutters.setdefault(Path(q).stem, _lm(q))
    todo = [(role, name, path) for role, name, path in ctx.regions]
    todo += [("interface", s.name, None) for s, part in ctx.steps("union")]
    out = []
    for role, name, path in todo:
        rec = {"region": name, "role": role, "volume_mm3": None, "kept": None, "lost_mm3": None, "cuts": [], "warn": False}
        try:
            if path is not None:
                man = _load_manifold(path)
            else:
                spec, part = next((s, p) for s, p in ctx.steps("union") if s.name == name)
                man = ctx.shapes.manifold(spec, NOMINAL, part)
            mine = [t for t in ctx.taken if t["region"] == name and t["region_role"] == role]
            trims = [t["cut"] for t in mine if t["cut_kind"] in ("face", "stl", "keep-out region") and not t.get("warn")
                     and t["cut"] in cutters]
            eff = None if man is None else (man if holes is None else man - holes)
            if eff is not None and trims:   # a cut that check A judged as a trim is asked for: it is not a loss
                eff = eff - _union_all(cutters[c] for c in trims)
            vol = 0.0 if eff is None else float(eff.volume())
            if vol > 1e-9:
                kept_vol = min(vol, float((eff ^ body).volume()))
                rec.update(volume_mm3=round(vol, 3), kept=round(kept_vol / vol, 4), lost_mm3=round(vol - kept_vol, 3),
                           warn=bool(kept_vol / vol < KEPT_MIN))
            else:
                continue
        except Exception as exc:   # a region file that is no closed manifold: a fact, not a stop
            rec["note"] = f"not measured: {type(exc).__name__}: {exc}"
        rec["cuts"] = [{"name": t["cut"], "kind": t["cut_kind"],
                        "role": "hole" if t["cut_kind"] in ("bore", "tube hole") else ("loss" if t.get("warn") or t["cut"] not in cutters else "trim")}
                       for t in ctx.taken if t["region"] == name and t["region_role"] == role]
        out.append(rec)
    return out


def integrity_warnings(records: list | None, mesh_file: str = "nominal.stl") -> list[str]:
    """One warning for each region of `integrity` with `warn`: the region keeps less than `KEPT_MIN` of its volume. Facts, then options."""
    out = []
    for r in records or []:
        if not r.get("warn"):
            continue
        names = [f"{c['name']} ({c['kind']}, {c['role']})" for c in r["cuts"]]
        cuts = ("The cuts of the check before the run that overlap it: " + ", ".join(names) + ". A face without 'radius_mm' cuts "
                "everything on the side of its normal." if r["cuts"] else
                "No cut of the check before the run overlaps it; look at the keep-out regions and at the render of the part.")
        out.append(f"{_region_word(r['role'])} '{r['region']}' keeps {100 * r['kept']:.1f} % of its volume in {mesh_file} "
                   f"(the limit is {100 * KEPT_MIN:g} %): {r['lost_mm3']:g} of {r['volume_mm3']:g} mm3 are missing. "
                   "The holes of the bore and tube interfaces are not counted. " + cuts +
                   " Options: bound the cut (a face takes 'radius_mm'), move the cut or the region, or accept the loss")
    return out


def stock_effect_warnings(items: list[dict]) -> list[str]:
    """Check E: a warning for each surface with `stock_mm` above 0 whose as-printed area is under `STOCK_EFFECT_MIN` of the finished area.

    The areas are the `area_mm2` of `measured` in the as-printed and the nominal mesh. The stock then has no effect: the region next
    to the surface starts at the finished size, so the as-printed surface has no material behind it.
    """
    out = []
    for it in items:
        if it.get("kind") == "stl" or not it.get("stock_mm", 0) > 0 or NOMINAL not in it:
            continue
        for key, ap in it[AS_PRINTED]["measured"].items():
            full = (it[NOMINAL]["measured"].get(key) or {}).get("area_mm2")
            if not full or full < AREA_MIN_MM2:
                continue
            got = ap.get("area_mm2", 0.0)
            if got < STOCK_EFFECT_MIN * full:
                what = f"interface '{it['name']}'" + (f" ({key})" if len(it[AS_PRINTED]["measured"]) > 1 else "")
                tail = ("The region next to it starts at the finished size. Make the region reach the as-printed size, or use a "
                        "'tube' interface, which adds the stock" if it["kind"] in ("bore", "face") else
                        "Check that the position of the interface is the position of the surface")
                out.append(f"{what}: stock_mm {it['stock_mm']:g} has no effect: the as-printed part has {got:.1f} mm2 of this surface "
                           f"against {full:.0f} mm2 finished. {tail}")
    return out


def full_surface_tests(items: list[dict]) -> list[dict]:
    """The test of each interface with `full_surface` (check D): the finished mesh shows at least `FULL_COVERAGE_MIN` of its full area.

    One dict for each such interface: `interface`, `variant` (`nominal`, else `as_printed`), `full_coverage` (the lowest of its
    surfaces: a tube has two), `threshold`, `met`, and `surfaces` (`key`, `area_mm2`, `full_area_mm2`, `full_coverage`).
    """
    out = []
    for it in items:
        if not it.get("full_surface"):
            continue
        variant = NOMINAL if NOMINAL in it else AS_PRINTED
        surf = [{"key": k, "area_mm2": f.get("area_mm2", 0.0), "full_area_mm2": f["full_area_mm2"], "full_coverage": f["full_coverage"]}
                for k, f in it[variant]["measured"].items() if "full_coverage" in f]
        if not surf:
            continue
        low = min(surf, key=lambda x: x["full_coverage"])
        out.append({"interface": it["name"], "variant": variant, "full_coverage": low["full_coverage"], "threshold": FULL_COVERAGE_MIN,
                    "met": bool(low["full_coverage"] >= FULL_COVERAGE_MIN), "surfaces": surf})
    return out


def full_surface_warnings(tests: list[dict]) -> list[str]:
    """One warning for each `full_surface` test that is not met: the facts (area found, full area) and the options."""
    out = []
    for t in tests:
        if not t["met"]:
            low = min(t["surfaces"], key=lambda x: x["full_coverage"])
            out.append(f"interface '{t['interface']}' has full_surface, and the {t['variant']} mesh shows {t['full_coverage']:.3g} of its "
                       f"full area (the limit is {t['threshold']:g}): {low['area_mm2']:g} of {low['full_area_mm2']:g} mm2. "
                       "Options: change the region or the interface so that the surface has material all round, or accept the gap")
    return out


def verdict_extra(rec: dict | None) -> tuple[dict, str]:
    """`(keys, sentence)` for `metrics.verdict` from the `interfaces` record: the tests of `full_surface` and the regions that keep too little.

    Both are empty when there is nothing to report, so the verdict of a run without them keeps its bytes. They are facts of the part,
    not tests of the convergence: `state`, `converged` and `feasible` do not read them.
    """
    if not rec:
        return {}, ""
    keys, parts = {}, []
    tests = rec.get("full_surface_tests") or []
    if tests:
        keys["interface_tests"] = tests
        parts.append("Interface tests (full surface): " + "; ".join(
            f"{t['interface']} {t['full_coverage']:.3g} of its full area (threshold {t['threshold']:g}: {'met' if t['met'] else 'not met'})"
            for t in tests) + ".")
    low = [r for r in rec.get("integrity") or [] if r.get("warn")]
    if low:
        keys["region_integrity"] = [{"region": r["region"], "role": r["role"], "kept": r["kept"], "threshold": KEPT_MIN} for r in low]
        parts.append("Frozen regions that keep less than the limit: " + "; ".join(
            f"{r['region']} {r['kept']:.3g} of its volume (threshold {KEPT_MIN:g}: not met)" for r in low) + ".")
    return keys, (" " + " ".join(parts)) if parts else ""


def items_record(ctx: Context, deltas: dict, final: tuple, nominal: tuple | None) -> list[dict]:
    """The per-interface records: spec size, measured size per mesh, volume change, stock. `final` and `nominal` are (verts, faces)."""
    items = []
    body_f = mesh_body(*final)
    body_n = mesh_body(*nominal) if nominal is not None else None
    for spec in ctx.specs:
        it = {"name": spec.name, "kind": spec.kind, "op": spec.op, "stock_mm": spec.stock_mm}
        if spec.kind == "stl":
            it["file"] = spec.path.name
        if spec.full_surface:   # the key is there only when the YAML asks for it (check D)
            it["full_surface"] = True
        it[AS_PRINTED] = {"volume_change_mm3": round(deltas[AS_PRINTED].get(spec.name, 0.0), 4),
                          "measured": measure(spec, AS_PRINTED, *final, body=body_f, box=ctx.box, removed=ctx.removed(AS_PRINTED))}
        if nominal is not None:
            it[NOMINAL] = {"volume_change_mm3": round(deltas[NOMINAL].get(spec.name, 0.0), 4),
                           "measured": measure(spec, NOMINAL, *nominal, body=body_n, box=ctx.box, removed=ctx.removed(NOMINAL))}
            if spec.kind != "stl":
                it["stock_measured_mm"] = _stock_between(spec, it[AS_PRINTED]["measured"], it[NOMINAL]["measured"])
        items.append(it)
    return items


def write_outputs(res: Result, result_dir: Path, final_bytes: bytes, dens: float, calibration=None):
    """Write `nominal.stl` (only with stock), reload both files, and return the `interfaces` record of metrics.json.

    `final_bytes` are the bytes of `final.stl` as written. `calibration` is the list of rounds of the `sdf` step (the
    `mesh.calibration` record), or None. Returns None if the run has no interfaces (`res` is None).
    """
    if res is None:
        return None
    from . import smooth_sdf
    from .post import _write_atomic

    ctx = res.context
    nominal_path = Path(result_dir) / "nominal.stl"
    rec = {"applied": bool(res.applied), "error": res.error, "count": len(ctx.specs), "stock": bool(ctx.has_stock),
           "sag_mm": SAG_MM, "t_s": res.t_s}
    if ctx.notes:
        rec["notes"] = list(ctx.notes)
    if not res.applied:
        if nominal_path.exists():
            nominal_path.unlink()
        return rec
    nom_arr = res.variants.get(NOMINAL)
    nom_bytes = None
    if nom_arr is not None:
        import trimesh

        nom_bytes = trimesh.Trimesh(nom_arr[0], nom_arr[1], process=False).export(file_type="stl")
        _write_atomic(nominal_path, nom_bytes)
    elif nominal_path.exists():   # no stock in this run: an old nominal.stl would be wrong
        nominal_path.unlink()
    t0 = time.monotonic()
    final = _load_stl_bytes(final_bytes)
    nominal = _load_stl_bytes(nom_bytes) if nom_bytes is not None else None
    rec["files"] = {"final.stl": "as_printed", "nominal.stl": "nominal" if nom_bytes is not None else None}
    rec[AS_PRINTED] = mesh_facts(final_bytes, dens)
    rec[NOMINAL] = mesh_facts(nom_bytes, dens) if nom_bytes is not None else None
    if nom_bytes is not None:
        rec["stock_volume_mm3"] = round(rec[AS_PRINTED]["volume_mm3"] - rec[NOMINAL]["volume_mm3"], 3)
    rec["clean"] = res.clean
    rec["bodies_removed"] = res.dropped
    if res.keep_out:
        rec["keep_out_cut"] = res.keep_out   # a union shape overlapped a keep-out region: the region was cut again
    if calibration:   # `mesh.volume_error_pct` is the as-printed mesh against the label volume, so it holds the stock;
        rec["label_volume_error_pct"] = round(100.0 * float(calibration[-1]["error"]), 3)   # this is the error of the label shapes
    rec["items"] = items_record(ctx, res.deltas, final, nominal)
    t1 = time.monotonic()
    if ctx.regions or ctx.steps("union"):   # check C: what stays of each frozen region (one boolean for each region)
        rec["integrity"] = integrity(ctx, *(nominal if nominal is not None else final))
        res.t_s["integrity"] = round(time.monotonic() - t1, 3)
    tests = full_surface_tests(rec["items"])   # check D
    if tests:
        rec["full_surface_tests"] = tests
    warns = (integrity_warnings(rec.get("integrity"), "nominal.stl" if nominal is not None else "final.stl")
             + stock_effect_warnings(rec["items"]) + full_surface_warnings(tests))
    if warns:
        rec["warnings"] = warns   # `topopt run` writes them to run.log (cli.finish); no second log line here
    for name, part in ((AS_PRINTED, rec[AS_PRINTED]), (NOMINAL, rec[NOMINAL])):
        facts = None if part is None else part["stl_roundtrip"]
        if facts is not None and not (facts["ok"] and facts["bodies"] == 1 and facts["zero_area_faces"] == 0
                                       and facts["open_edges"] == 0 and facts["nonmanifold_edges"] == 0):
            log.warning("the %s STL file is not one closed manifold after the float32 reload: %d bodies, %d open edges, "
                        "%d non-manifold edges, %d zero-area faces", name, facts["bodies"], facts["open_edges"],
                        facts["nonmanifold_edges"], facts["zero_area_faces"])
    bad = [(it["name"], v) for it in rec["items"] for v in (AS_PRINTED, NOMINAL) if v in it
           for feat in it[v]["measured"].values()
           if not feat["within_tolerance"] and (feat.get("found") or (feat.get("expected_area_mm2") or 0.0) >= AREA_MIN_MM2
                                                or "expected_area_mm2" not in feat)]
    if bad:
        log.warning("interface surface not within tolerance, not covered, or not found: %s",
                    ", ".join(sorted({f"{n} ({v})" for n, v in bad})))
    rec["t_s"] = {**res.t_s, "measure": round(time.monotonic() - t0, 3)}
    return rec
