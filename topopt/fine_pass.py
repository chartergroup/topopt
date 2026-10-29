"""Fine pass: a new run on a finer voxel grid that starts from the design of a finished run (#318).

A fine pass reads a finished parent run, maps its design onto the design voxels of a finer grid of the same problem,
and gives that design to `optimize.run` as the start design (`run(..., x0=FineStart)`). The optimizer state starts new.

This module has four parts.

1. The voxel of the part. `part_voxel` gives the largest voxel that keeps enough voxels across the thinnest load, fixed and solid
   region, the smallest keep-out region, and `settings.min_member_mm`. It reuses the rules that exist: `region_rules` (the check of
   load and solid regions) and `members.MEMBER_ACROSS` (the 3-voxel line of `min_member_mm`). It adds no threshold.
   `part_voxel_block` and `fact_text` make the facts that `check`, the run log, the part card and the MCP tools show. `fine_grid_guards` tests the
   planned fine grid against the guards of the run (`max_active_voxels`, `max_grid_voxels`, the memory estimate) before the first run.
2. The settings of a fine pass. `with_fine_settings` gives the config with the fine voxel, the maximum pass count, the beta schedule that goes
   on from the parent (`shift_schedule`), and the stop rule `settled` (the loss has settled: `verdict.stop_decision`). `resolve_fine` reads
   the parent and checks the voxel.
3. The parent run. `read_parent` reads the run folder.
4. The start design. `FineStart` maps the design of the parent onto the design voxels of a model (`start_design`) and fits the mean density
   to the volume target.

The mapping reads the position of each fine voxel centre in millimetres. It does not read voxel indices, so two grids with other origins
and other voxel sizes are mapped by place (the origin is the minimum corner of all regions, and a mirror plane can move it by up to one voxel).
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import FINE_OBJECTIVE_CHANGE, UserError   # the default tolerance of `settled`: one constant, next to the loader of `settings.fine_pass`
from .members import MEMBER_ACROSS
from .region_rules import MIN_REGION_THICK, REGION_ACROSS
from .verdict import SETTLED as SETTLED_RULE   # the name of the stop rule of a fine pass: one place, `verdict.stop_decision`
from .voxelize import DESIGN, FIXED, LOAD

STEP_MM = 0.05            # the step of the voxel picker (`regions_builder.auto_voxel` rounds up to it); the derived voxel rounds down to it
# The measure of the thickness of a region (`inscribed_radius`) has these precision constants (they are not limits for the user). Measured
# 2026-10-26 on a shared Linux host (22 cores, trimesh 5.1 with rtree, no embree), one thread, CPU time; the diameter of the largest sphere
# against the exact value (a linear program for a convex body). Accuracy: slabs (tilted too) and the 200 x 200 x 2 mm plate are exact (0.00 %);
# cuboids are 0.016 % low at worst (2 x 2 x 3 mm: 1.9998); a rod of 64 sections 0.01 %; the wall of a tube of 96 sections 0.03 %; a sphere of
# 5120 faces 0.12 %, of 1280 faces 0.34 %; a cone 0.55 %; a plate and a ball in one mesh 0.34 % (as the ball alone); an 800 face sphere up to 3 % in the worst of
# six orientations. Never above the exact value: every sphere that it finds lies inside the region. Cost: 0.03 s for a box or the plate, 0.1 s
# for a rod, 0.1 to 0.15 s for the regions of the hook (380 faces or fewer: the screw holes), the arm and the ladle bowl (3360 faces),
# 0.7 s for the 800 face gravy space of the ladle, 1.2 s for a sphere of 1280 faces, 5.3 s for one of 5120 faces. The cost depends on the number of
# faces, not on the size of the region or on a grid. A real region that is convex was compared too: the motor shaft of the arm, 6.9956 against 6.9962.
THICK_SAMPLES = 512       # points of the first search: surface points, chosen by area, each gives the largest sphere that touches the surface there
THICK_TRIES = 32          # new points around the best one in each round of the local search
THICK_TOL = 1e-4          # relative precision of a radius: a search stops when a sphere is empty to this, or can not beat the best by this
THICK_SLACK = 5e-4        # the measure of a box is below its exact thickness by up to 0.016 % (measured): a voxel within 0.05 % above a step counts as on it
THICK_ROUNDS = 20         # rounds of the local search: the distance of the new points halves each round (from the spacing of the first points down to 1e-6 of it)
CHECK_STEPS = 8           # the most steps of 0.05 mm that the check of the voxel on the grid takes (`check_voxel`): a few steps, not a search
SOURCES = ("x", "rho")    # what the start design is mapped from: the design variables of the checkpoint, or the density of the last snapshot
MIN_WEIGHT = 1e-9         # a fine voxel with less source weight than this has no source voxel near it: it takes the nearest source voxel
CHUNK = 500_000           # fine voxels in one call of the interpolation (a bound on memory)
SAME_VOXEL = 1e-9         # two voxel sizes in mm that differ by less than this are the same voxel


# --- the voxel of the part ----------------------------------------------------------------------------

def region_across_needed() -> float:
    """Voxels across that a load or solid region needs: the rule `thick >= MIN_REGION_THICK` of the region check (`region_rules`).

    The check measures `thick`, the largest distance in voxels to the outside. It shows `2 * thick - 1` as the voxels across.
    """
    return REGION_ACROSS


def _outward(mesh):
    """`(mesh, normals)`: the face normals of a closed mesh, all pointing out of the solid. A mesh with mixed winding is repaired on a copy."""
    import trimesh

    if not mesh.is_winding_consistent:
        mesh = mesh.copy()
        trimesh.repair.fix_normals(mesh)
    normals = np.asarray(mesh.face_normals, float)
    return mesh, (-normals if mesh.volume < 0 else normals)


def _half_chord(mesh, p: np.ndarray, n: np.ndarray, diagonal: float) -> np.ndarray:
    """Half the length of the chord that starts at each surface point `p` and goes inward along the normal `n` (out of the solid).

    A sphere that touches the surface at `p` has its centre on that line, and it can not hold the other end of the chord, so its radius is at
    most half the chord: a start value for `_shrink` that is never too small.
    """
    half = np.full(len(p), diagonal / 2.0)
    loc, ray, _tri = mesh.ray.intersects_location(p - n * (1e-6 * diagonal), -n, multiple_hits=False)
    if len(ray):
        half[ray] = 0.5 * np.linalg.norm(loc - p[ray], axis=1)
    return half


def _shrink(mesh, p: np.ndarray, n: np.ndarray, start: np.ndarray, floor: float = 0.0) -> tuple[np.ndarray, float]:
    """The radius of the largest sphere inside the solid that touches the surface at each point `p` (outward normal `n`). Return `(radius, floor)`.

    The search starts with the radius `start` and shrinks it: the nearest surface point `q` to the centre that is inside the sphere gives the
    radius of the sphere that touches both `p` and `q` (`|p - q|^2 / (2 n.(p - q))`, the centre stays on the normal). It stops when the sphere is
    empty, to a relative `THICK_TOL`. The radius only falls, so a point whose radius has fallen to `floor` (the best radius so far, plus
    `THICK_TOL`) can not win: it stops and its radius is 0. Every radius that is not 0 is the radius of an empty sphere, so it lies inside the solid.
    `floor` is returned, raised to the best radius found.
    """
    import trimesh

    r = np.asarray(start, float).copy()
    live = r > floor * (1.0 + THICK_TOL)
    done = np.zeros(len(p), bool)
    for _ in range(40):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        centre = p[idx] - n[idx] * r[idx, None]
        q, d, _tri = trimesh.proximity.closest_point(mesh, centre)
        empty = d >= r[idx] * (1.0 - THICK_TOL)
        v = p[idx] - q
        denom = 2.0 * np.einsum("ij,ij->i", n[idx], v)
        ok = (~empty) & (denom > 1e-12)
        new = np.where(ok, np.einsum("ij,ij->i", v, v) / np.where(ok, denom, 1.0), r[idx])
        r[idx[empty]] = np.minimum(r[idx], d)[empty]     # the largest empty sphere at this centre
        r[idx[ok]] = np.minimum(new, r[idx])[ok]
        done[idx[empty]] = True
        live[idx[~ok]] = False
        if empty.any():
            floor = max(floor, float(r[idx[empty]].max()))
        live &= ~done
        live &= r > floor * (1.0 + THICK_TOL)
    idx = np.flatnonzero(live)
    if idx.size:   # the search did not end in 40 steps: the largest empty sphere at the centre is a smaller sphere that fits
        centre = p[idx] - n[idx] * r[idx, None]
        _q, d, _tri = trimesh.proximity.closest_point(mesh, centre)
        r[idx] = np.minimum(r[idx], d)
        done[idx] = True
        floor = max(floor, float(r[idx].max()))
    return np.where(done, r, 0.0), floor


def _body_radius(mesh, floor: float = 0.0) -> tuple[float, np.ndarray | None]:
    """`(radius, centre)` of the largest sphere inside one closed body. A body that can not beat `floor` gives radius 0.

    First search: `THICK_SAMPLES` surface points, chosen by area with a fixed seed. Each gives the largest sphere that touches the surface
    there (`_shrink`). Local search: in `THICK_ROUNDS` rounds, `THICK_TRIES` new surface points near the best one; the distance halves each round;
    a better sphere replaces the best. The last sphere is tested with the signed distance of its centre.
    """
    import trimesh

    mesh, normals = _outward(mesh)
    rng = np.random.default_rng(0)
    lo, hi = mesh.bounds
    diagonal = float(np.linalg.norm(hi - lo))
    if not mesh.area > 0.0:
        raise UserError("a region has no surface area")
    p, face = trimesh.sample.sample_surface(mesh, THICK_SAMPLES, seed=0)
    p, n = np.asarray(p), normals[face]
    r, floor = _shrink(mesh, p, n, _half_chord(mesh, p, n, diagonal), floor)
    best = int(np.argmax(r))
    if r[best] <= 0.0:
        return 0.0, None
    bp, bn, br = p[best], n[best], float(r[best])
    step = float(np.sqrt(mesh.area / THICK_SAMPLES))
    for _ in range(THICK_ROUNDS):
        q, _d, tri = trimesh.proximity.closest_point(mesh, bp + rng.normal(size=(THICK_TRIES, 3)) * step)
        q, qn = np.asarray(q), normals[tri]
        rr, _floor = _shrink(mesh, q, qn, np.minimum(1.25 * br, _half_chord(mesh, q, qn, diagonal)), br)
        i = int(np.argmax(rr))
        if rr[i] > br:
            bp, bn, br = q[i], qn[i], float(rr[i])
        step *= 0.5
    centre = bp - bn * br
    inside = float(trimesh.proximity.signed_distance(mesh, centre[None])[0])
    return (min(br, inside), centre) if inside > 0.0 else (0.0, None)


def inscribed_radius(mesh) -> tuple[float, np.ndarray | None]:
    """`(radius, centre)` of the largest sphere inside a closed mesh (the mesh of a region), in the unit of the mesh.

    This is the quantity that the region check measures: the largest distance from a point of the region to its outside. A region of several bodies
    gives the largest sphere of any body. The bodies go by volume, largest first; a body whose volume is below that of a sphere with the best radius
    so far is not searched (it can not hold a larger sphere). The radius is never above the true value: the sphere that is returned lies inside.
    A body keeps the face order of the mesh. `mesh.split` groups the faces with an unstable sort, whose order differs between x86_64 and arm64;
    the search of `_body_radius` samples by face, so with `split` a plate and a ball in one mesh measured 0.29 % low on Linux x86_64 and 2.2 % low
    on macOS arm64 (the ball alone: 0.34 % on both).
    """
    import trimesh

    if mesh.body_count > 1:
        faces = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))
        bodies = mesh.submesh([np.sort(f) for f in faces], only_watertight=False, repair=True)   # the arguments of `split(only_watertight=False)`
    else:
        bodies = [mesh]
    bodies = sorted(bodies, key=lambda b: -abs(float(b.volume)))
    best, centre = 0.0, None
    for body in bodies:
        if (3.0 * abs(float(body.volume)) / (4.0 * math.pi)) ** (1.0 / 3.0) <= best:
            break
        r, c = _body_radius(body, best)
        if r > best:
            best, centre = r, c
    return best, centre


def thickness_mm(mesh) -> float:
    """The thickness of a region in mm: twice the radius of the largest sphere inside its mesh (`inscribed_radius`)."""
    return 2.0 * inscribed_radius(mesh)[0]


def round_down_to_step(h: float) -> float:
    """`h` rounded down to `STEP_MM`, and not below one step."""
    return max(STEP_MM, math.floor(h / STEP_MM + 1e-9) * STEP_MM)


def grid_origin(config, extra_meshes, h: float) -> np.ndarray:
    """The origin of the grid of a run at voxel `h`: the lowest corner of all regions (the design too, and the shapes of the union interfaces).

    This is what `voxelize.voxelize` does, with the mirror planes: each plane becomes a plane of the grid. `extra_meshes` are the meshes that the
    caller has already (the regions it measured); the design and the interface shapes are read here.
    """
    from . import symmetry
    from .voxelize import interface_add_meshes, load_mesh

    reg = config.regions
    meshes = [load_mesh(reg.design), *extra_meshes, *interface_add_meshes(config).values()]
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    return symmetry.shift_lo(config.mirror, lo, h)


def region_thick(mesh, origin: np.ndarray, h: float, max_voxels: float) -> float | None:
    """`thick` of the region check for a region on the grid with this `origin` and voxel `h`: the largest distance, in voxels, to the outside.

    The region is voxelized on the voxel centres of that grid (the box of the region only, so a small region costs little), then the same
    measure as `cli.check_data` is applied (`scipy.ndimage.distance_transform_edt` of the padded mask). Return None when the box has more than
    `max_voxels` voxels: the fine run refuses such a grid (`settings.max_grid_voxels`), so there is nothing to test.
    """
    from scipy import ndimage

    from .voxelize import grid_from_bounds, inside_mask

    lo, hi = mesh.bounds
    first = np.floor((lo - origin) / h + 1e-9)
    corner = origin + first * h
    nf = np.maximum(np.ceil((hi - corner) / h - 1e-6), 1.0)
    if float(np.prod(nf)) > max_voxels:
        return None
    corner, shape = grid_from_bounds(corner, hi, h)
    mask = inside_mask(mesh, corner, h, shape)
    return float(ndimage.distance_transform_edt(np.pad(mask, 1)).max()) if mask.any() else 0.0


def check_voxel(config, regions, voxel: float, limiting: tuple | None = None) -> dict:
    """Test the voxel on the grid of the fine pass, and go down in steps of `STEP_MM` until the regions pass the region check.

    `regions` is a list of `(role, name, mesh)`. The regions that are tested are the load and solid regions (the ones that `topopt check` tests
    with `MIN_REGION_THICK`) and the region `limiting` = `(role, name)` that sets the voxel. For each the check of `cli.check_data` is made on the
    grid of the fine pass at `voxel` (`region_thick`): it passes when `thick` is at least `MIN_REGION_THICK`. The thickness gives the voxel as a
    number; the grid decides how many voxel centres a region holds, so a region can have fewer voxels across than the number says (a slab of 2.8
    voxels holds 2 or 3 rows). At most `CHECK_STEPS` steps are taken. Return `{"voxel_mm", "start_mm", "steps", "region", "ok", "skipped"}`: `region`
    is the region that failed last (None when the start voxel passed), `ok` is false when the steps ran out, `skipped` is true when the grid of a
    region is larger than `settings.max_grid_voxels` (the guards of the fine run refuse it; the start voxel is returned).
    """
    limit = float(config.settings.max_grid_voxels)
    tested = [(role, name, mesh) for role, name, mesh in regions if role in ("load", "solid") or (role, name) == limiting]
    start = h = round_down_to_step(voxel)
    failed = None
    for steps in range(CHECK_STEPS + 1):
        origin = grid_origin(config, [m for _role, _name, m in regions], h)
        bad = None
        for role, name, mesh in tested:
            thick = region_thick(mesh, origin, h, limit)
            if thick is None:
                return {"voxel_mm": round(start, 4), "start_mm": round(start, 4), "steps": 0, "region": None, "ok": True, "skipped": True}
            if thick < MIN_REGION_THICK:
                bad = name
                break
        if bad is None:
            return {"voxel_mm": round(h, 4), "start_mm": round(start, 4), "steps": steps, "region": failed, "ok": True, "skipped": False}
        failed = bad
        if steps < CHECK_STEPS and h - STEP_MM >= STEP_MM - SAME_VOXEL:
            h = (round(h / STEP_MM) - 1) * STEP_MM
        else:
            break
    return {"voxel_mm": round(h, 4), "start_mm": round(start, 4), "steps": steps, "region": failed, "ok": False, "skipped": False}


def part_voxel(config, verify: bool = False) -> dict:
    """The voxel that the part needs. Return a dict with the features, the one that sets the voxel, and the voxel in mm.

    The features are the fixed, load and solid regions, the keep-out regions, and `settings.min_member_mm`. Each gives a voxel. A region gives
    `thickness / REGION_ACROSS` (2.8 voxels across, the rule of the region check), with the thickness of `thickness_mm`; `min_member_mm` gives
    `min_member_mm / MEMBER_ACROSS`. The result `voxel_mm` is the smallest voxel of the features, rounded DOWN to `STEP_MM`. There is one value: no
    band and no levels. `limiting` is the feature that sets it, with its thickness in mm and the voxels across now (at the voxel of the run) and needed.
    `verify` true (a fine pass that is planned or starts) tests the voxel on the grid of the fine pass, and goes down in steps of `STEP_MM` if a region
    has too few voxel centres across (`check_voxel`); the result `checked` has the record, else `checked` is None. A plain check and the end of a run
    use the measured value: they do not voxelize.
    `recommend` is true when the voxel is below the voxel of the run by `STEP_MM` or more: a voxel that is equal to the voxel of the run after
    the rounding gives no recommendation.
    """
    from .voxelize import load_mesh

    st, reg = config.settings, config.regions
    items = ([("fixed", p.stem, p) for p in reg.fixed] + [("load", lr.name, lr.path) for lr in reg.load]
             + [("solid", p.stem, p) for p in reg.solid] + [("keep_out", p.stem, p) for p in reg.keep_out])
    need = region_across_needed()
    meshes = [(role, name, load_mesh(path)) for role, name, path in items]
    features = []
    for role, name, mesh in meshes:
        t = thickness_mm(mesh)
        features.append({"role": role, "name": name, "thickness_mm": round(t, 3), "across_at_run": round(t / st.voxel_size, 2),
                         "across_needed": round(need, 2), "voxel_mm": t / need, "rule": "region_rules.MIN_REGION_THICK"})
    if st.min_member_mm is not None:
        t = float(st.min_member_mm)
        features.append({"role": "min_member_mm", "name": "settings.min_member_mm", "thickness_mm": t, "across_at_run": round(t / st.voxel_size, 2),
                         "across_needed": MEMBER_ACROSS, "voxel_mm": t / MEMBER_ACROSS, "rule": "members.MEMBER_ACROSS"})
    if not features:
        raise UserError("the problem has no region to measure")
    limiting = min(features, key=lambda f: f["voxel_mm"])
    slack = 1.0 if limiting["role"] == "min_member_mm" else 1.0 + THICK_SLACK     # a measured thickness is within THICK_SLACK of the exact one; min_member_mm is exact
    voxel = round_down_to_step(limiting["voxel_mm"] * slack)
    checked = check_voxel(config, meshes, voxel, (limiting['role'], limiting['name'])) if verify else None
    if checked is not None:
        voxel = checked["voxel_mm"]
    for f in features:
        f["voxel_mm"] = round(f["voxel_mm"], 4)
    run_voxel = float(st.voxel_size)
    return {"voxel_mm": round(voxel, 4), "limiting": limiting, "features": features, "run_voxel_mm": run_voxel, "checked": checked,
            "recommend": bool(run_voxel - voxel >= STEP_MM - SAME_VOXEL)}


def fine_estimates(check: dict, config, voxel_mm: float, max_passes: int) -> dict:
    """Counts and estimates for a fine pass at `voxel_mm`, from the dict of `cli.check_data` (the run at its own voxel).

    The counts scale with the cube of the voxel ratio (measured 2026-10-26 on a 5-inch drone arm: 20,496 active voxels at 1.0 mm gave 59,758 at 0.7 mm
    against 59,571 built, and 73,188 free dofs gave 213,000 against 199,041). The time is the time model of `topopt check`
    (`cli.estimate_seconds`) for `max_passes` passes: an upper bound, because a fine pass ends when its loss has settled. The same model gives
    the time of a run from zero at the fine voxel with `settings.iterations` passes (`time_s_from_zero`). The memory is
    `cli.estimate_memory_bytes` with the filter of the fine grid (its shape: the grid of the run scaled by the voxel ratio). A filter radius that follows the voxel (the default, 1.5 voxels; the wall rule, max(w / L, 1.5 voxels))
    follows it here too. A bordered copy of K (joints) is not counted. The numbers are estimates.
    """
    from .cli import estimate_memory_bytes, estimate_seconds, physical_ram_bytes
    from .members import filter_size, radius_for_grid

    st = config.settings
    h = float(check["grid"]["voxel_mm"])
    ratio = (h / float(voxel_mm)) ** 3
    solver = check["solver"]["resolved"] or "cholesky"
    n_stress = sum(1 for c in config.load_cases if c.stress_limit_mpa is not None)
    active, free_vox, free_dofs = (int(round(check[k] * ratio)) for k in ("active_voxels", "free_design_voxels", "free_dofs"))
    grid = int(round(check["grid"]["voxels"] * ratio))
    radius = radius_for_grid(st, float(voxel_mm))
    shape = [int(math.ceil(n * h / float(voxel_mm))) for n in check["grid"]["shape"]]
    fbytes = filter_size(radius, float(voxel_mm), free_vox, shape)["held_bytes_estimate"]
    kw = dict(stress_rows=n_stress, n_design=free_vox, step_acceptance=st.stress_step_acceptance, robust_mode=st.robust != "off")
    _, t_max = estimate_seconds(free_dofs, int(max_passes), solver, **kw)
    _, t_zero = estimate_seconds(free_dofs, int(st.iterations), solver, **kw)
    _, t_run = estimate_seconds(int(check["free_dofs"]), int(st.iterations), solver,
                                **{**kw, "n_design": int(check["free_design_voxels"])})
    return {"voxel_mm": float(voxel_mm), "run_voxel_mm": h, "voxel_ratio": round(ratio, 2), "active_voxels": active,
            "free_design_voxels": free_vox, "free_dofs": free_dofs, "grid_voxels": grid, "max_passes": int(max_passes), "solver": solver,
            "time_s_max": round(t_max), "time_s_from_zero": round(t_zero), "time_s_run": round(t_run),
            "memory_bytes": int(estimate_memory_bytes(grid, free_dofs, fbytes)), "ram_bytes": physical_ram_bytes()}


GUARD_SETTINGS = {"max_active_voxels": "active_voxels", "max_grid_voxels": "grid_voxels"}   # the two voxel guards: setting -> key of `fine_estimates`


def fine_grid_guards(config, check: dict, voxel_mm: float, max_passes: int | None = None) -> dict:
    """Test the planned fine grid against the guards of the run: `settings.max_active_voxels`, `settings.max_grid_voxels` and the memory block.

    `check` is the dict of `cli.check_data` (the run at its own voxel); the counts of the fine grid scale with the cube of the voxel ratio
    (`fine_estimates`). The memory block is the memory estimate above the RAM of this machine: the same estimate that `topopt check` warns
    with ("a run may swap or crash"); in a fine pass it is a failure, because the fine grid is the largest of the two runs.
    Return `{"voxel_mm", "ok", "failed", "voxel_pass_mm", "text"}`. `failed` lists `{"guard", "value", "limit", "text"}` for each guard that
    the grid fails. `voxel_pass_mm` is the finest voxel (a multiple of `STEP_MM`, with the margin of 3 % in the voxel that `voxelize.suggest_voxel_size` has,
    because the counts are estimates) that passes all three, or None when no voxel finer than the run voxel does. `text` is the fact with the options (empty when `ok`): `--fine-voxel H` (MCP `fine_voxel`), or a higher setting.
    """
    from .cli import gib
    from .voxelize import suggest_voxel_size

    st = config.settings
    run_h = float(check["grid"]["voxel_mm"])
    passes = max(1, int(st.iterations if max_passes is None else max_passes))

    def failures(v: float) -> list[dict]:
        est = fine_estimates(check, config, v, passes)
        out = []
        for key, name in GUARD_SETTINGS.items():
            value, limit = est[name], getattr(st, key)
            if value > limit:
                out.append({"guard": f"settings.{key}", "value": int(value), "limit": int(limit),
                            "text": f"settings.{key}: about {int(value)} {'active' if name == 'active_voxels' else 'grid'} voxels, the limit is {int(limit)}"})
        if est["ram_bytes"] and est["memory_bytes"] > est["ram_bytes"]:
            out.append({"guard": "memory", "value": int(est["memory_bytes"]), "limit": int(est["ram_bytes"]),
                        "text": f"the memory estimate is {gib(est['memory_bytes'])} and this machine has {gib(est['ram_bytes'])} RAM"})
        return out

    failed = failures(float(voxel_mm))
    res = {"voxel_mm": float(voxel_mm), "ok": not failed, "failed": failed, "voxel_pass_mm": None, "text": ""}
    if not failed:
        return res
    # The finest voxel that passes: the counts need at least the voxel that `suggest_voxel_size` gives (it has a margin of 3 %), the memory is
    # tested on the same steps. The search goes up in steps of `STEP_MM` and stops below the run voxel: a voxel that is not finer is no fine pass.
    counts = {"max_active_voxels": check["active_voxels"], "max_grid_voxels": check["grid"]["voxels"]}
    low = max(float(voxel_mm), *(suggest_voxel_size(run_h, counts[g], getattr(st, g)) for g in counts))
    v = max(STEP_MM, math.ceil(low / STEP_MM - 1e-9) * STEP_MM)
    while v < run_h - SAME_VOXEL:
        if not failures(v):
            res["voxel_pass_mm"] = round(v, 4)
            break
        v += STEP_MM
    options = []
    if res["voxel_pass_mm"] is not None:
        options.append(f"--fine-voxel {res['voxel_pass_mm']:g} (MCP fine_voxel) is the finest voxel that passes")
    kinds = {f["guard"] for f in failed}
    raise_text = ", ".join(sorted(k for k in kinds if k.startswith("settings.")))
    if raise_text:
        options.append(f"raise {raise_text} in the YAML if this machine has the memory")
    if "memory" in kinds:
        options.append("use a machine with more memory")
    res["text"] = (f"fine pass: the fine grid at {voxel_mm:g} mm fails " + "; ".join(f["text"] for f in failed) + ". A fine pass at this voxel stops "
                   "when it builds the grid. Options: " + "; or ".join(options) + ".")
    if res["voxel_pass_mm"] is None:
        res["text"] += " No voxel finer than the voxel of this run passes at these settings."
    return res


def name_fine_option(exc: UserError) -> UserError:
    """The error of a fine pass with the voxel named as `--fine-voxel` and not as `settings.voxel_size`.

    The guards of the build (`voxelize`) say "Try voxel_size X or more" and "voxel_size H": for a fine pass the voxel comes from `--fine-voxel`
    (MCP `fine_voxel`), or from the voxel that the part needs, and `settings.voxel_size` is the voxel of the first run. A message that does not
    name `voxel_size` is returned as it is.
    """
    import re

    text = str(exc)
    if "voxel_size" not in text:
        return exc
    text = re.sub(r"Try voxel_size ([0-9.]+) or more", r"Try --fine-voxel \1 or more (MCP fine_voxel)", text)
    text = text.replace("Use a larger voxel_size", "Use a larger --fine-voxel (MCP fine_voxel)")
    text = text.replace("Use a smaller voxel_size", "Use a smaller --fine-voxel (MCP fine_voxel)")
    text = re.sub(r"at voxel_size ([0-9.]+)", r"at the fine voxel \1 mm", text)
    if "voxel_size" in text:
        text += " This is a fine pass: its voxel is --fine-voxel (MCP fine_voxel) or the voxel that the part needs, not settings.voxel_size."
    return UserError(text)


def part_voxel_block(config, check: dict | None = None, max_passes: int | None = None) -> dict:
    """The facts of the voxel of the part, as one dict: the key `fine_pass` of `check_data`, `check_problem`, and `part_voxel` of run.json.

    Keys: `voxel_mm` (the voxel of the part: one value, rounded down to `STEP_MM`), `run_voxel_mm`, `recommend` (the voxel is below the voxel of
    the run by `STEP_MM` or more), `limiting` (the feature that sets the voxel: `role`, `name`, `thickness_mm`, `across_at_run`, `across_needed`),
    `max_passes`, `tolerance` (the objective change at which the fine pass ends). This is the measured value (`part_voxel` without `verify`): a
    fine pass that is planned or starts tests it on the grid (`check_voxel`) and can go down by a few steps. With
    `recommend` and the dict `check` of `cli.check_data`, also `estimate` (counts, time and memory of a fine pass: `fine_estimates`) and `guards`.
    With `recommend`, `fact` is the text line. A measure that fails gives `recommend` false and `error` with the reason: the fact never stops a
    check or a run. `max_passes` is `settings.iterations` of the YAML when not given: the first run, which is the parent, has that many passes.
    """
    max_passes = max(1, int(config.settings.iterations if max_passes is None else max_passes))
    try:
        res = part_voxel(config)
    except Exception as exc:   # a fact for the user must not stop a check or a run; the reason is in the block
        return {"recommend": False, "voxel_mm": None, "run_voxel_mm": float(config.settings.voxel_size), "max_passes": max_passes,
                "error": f"{type(exc).__name__}: {exc}"}
    lim = res["limiting"]
    block = {"voxel_mm": res["voxel_mm"], "run_voxel_mm": res["run_voxel_mm"], "recommend": res["recommend"], "max_passes": max_passes,
             "tolerance": tolerance_of(config)[0], "limiting": {k: lim[k] for k in ("role", "name", "thickness_mm", "across_at_run", "across_needed")}}
    if res["recommend"]:
        if check is not None:
            try:
                block["estimate"] = fine_estimates(check, config, res["voxel_mm"], max_passes)
                block["guards"] = {k: v for k, v in fine_grid_guards(config, check, res["voxel_mm"], max_passes).items() if k != "voxel_mm"}
            except Exception as exc:
                block["estimate_error"] = f"{type(exc).__name__}: {exc}"
        block["fact"] = fact_text(block)
    return block


def _duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def fact_text(block: dict) -> str:
    """The one fact line of a part that needs a finer voxel. The same words go to `topopt check`, the run log, the part card and the MCP tools.

    It says what sets the voxel, what the run uses, the option, when the fine pass ends, and what is known about the cost. The numbers of
    time and memory are estimates. It does not say that a fine pass is faster: the first run is extra time.
    """
    lim = block["limiting"]
    what = ("the setting min_member_mm" if lim["role"] == "min_member_mm"
            else f"the {lim['role'].replace('_', '-')} region '{lim['name']}'")
    text = (f"fine pass: the part needs a voxel of {float(block['voxel_mm']):g} mm or less ({what}: {lim['thickness_mm']:g} mm thick, "
            f"{lim['across_at_run']:g} voxels across at the voxel of this run, {lim['across_needed']:g} needed). This run uses "
            f"{block['run_voxel_mm']:g} mm. A fine pass tests the voxel on its grid and goes down in steps of {STEP_MM:g} mm if a region has too few "
            f"voxels across there. Option: a fine pass runs the same problem on the finer voxel and starts from the design of a "
            f"finished run (`topopt run <yaml> --out <new folder> --fine-from <finished run folder>`, MCP run_topopt with fine_from; "
            f"`--fine-pass` makes the first run and the fine pass in one command). The fine pass ends when the loss has settled (the change "
            f"of the objective between passes is at most {block.get('tolerance', FINE_OBJECTIVE_CHANGE):g}), or at the maximum of passes. "
            f"In two measured parts the result came within 0.5 % and 2.1 % of the compliance of a run from zero at the fine voxel, with fewer "
            f"passes at the fine voxel. The first run is extra time.")
    est = block.get("estimate")
    if est:
        text += (f" Estimate at {est['voxel_mm']:g} mm: about {est['active_voxels']} active voxels and {est['free_dofs']} free dofs; "
                 f"at most {est['max_passes']} passes take about {_duration(est['time_s_max'])} (less if the loss settles first); "
                 f"about {est['memory_bytes'] / 2**30:.1f} GiB of memory (the time and the memory are estimates).")
    guards = block.get("guards")
    if guards and not guards.get("ok", True):
        text += " " + guards["text"]
    return text


def check_fact(part: dict) -> str | None:
    """The fact about the test of the voxel on the grid of the fine pass (`check_voxel`), or None when the test passed at the first voxel.

    `part` is the dict of `part_voxel` with `verify`. The text says which region had too few voxels across, how many steps of `STEP_MM` the voxel
    went down, and the voxel that is used.
    """
    chk = part.get("checked")
    if not chk:
        return None
    lim = part["limiting"]
    if chk["skipped"]:
        return ("fine pass: the test of the voxel on the grid of the fine pass was skipped: the grid of a region has more voxels than "
                "settings.max_grid_voxels, and the run refuses such a grid. The voxel is the measured one.")
    if not chk["steps"]:
        return None
    base = (f"fine pass: the part needs a voxel of {chk['start_mm']:g} mm (the {lim['role'].replace('_', '-')} region '{lim['name']}': "
            f"{lim['thickness_mm']:g} mm thick). On the grid of the fine pass at that voxel the region '{chk['region']}' has too few voxels across "
            f"(the largest distance to the outside is below {MIN_REGION_THICK:g} voxels: the check of regions), ")
    if chk["ok"]:
        return base + (f"so the voxel goes down {chk['steps']} step{'s' if chk['steps'] > 1 else ''} of {STEP_MM:g} mm to {chk['voxel_mm']:g} mm, "
                       "where every tested region passes.")
    return base + (f"and {chk['steps']} steps of {STEP_MM:g} mm down to {chk['voxel_mm']:g} mm did not help. The voxel {chk['voxel_mm']:g} mm is used; "
                   "the region may stay thin there. Option: --fine-voxel H (MCP fine_voxel) forces a voxel.")


# --- the settings of a fine pass ----------------------------------------------------------------------

def shift_schedule(schedule, offset: int) -> tuple:
    """The beta schedule of a fine pass: the stages of `schedule` seen from pass `offset` of the parent run.

    A fine pass goes on with the schedule of the problem, as `--continue-from` does: pass `j` of the fine pass has the beta of pass
    `offset + j` of the parent. `offset` is the number of passes that the parent chain ran (the pass of the parent checkpoint, plus the
    offset of the parent when the parent is a fine pass itself). The stage that is active at `offset` starts at pass 0; a stage that starts
    later keeps its place, moved by `offset`. So the beta of pass 0 is the beta that the parent ended with, when the YAML of the parent is used.
    """
    from .optimize import beta_at

    offset = int(offset)
    if offset < 0:
        raise ValueError("the offset of the schedule must be 0 or more")
    first = (0, float(beta_at(schedule, offset)))
    later = [(int(it) - offset, float(b)) for it, b in schedule if int(it) > offset]
    return (first, *later)


def with_fine_settings(config, voxel_mm: float, max_passes: int, offset: int, tolerance: float = FINE_OBJECTIVE_CHANGE, early_stop: bool = True):
    """The config of a fine pass: the fine voxel, the maximum pass count, the beta schedule seen from pass `offset` of the parent, and the stop rule.

    A filter radius that follows the voxel (`filter_radius_source` is `default`: 1.5 voxels; `wall`: max(w / L, 1.5 voxels) of the wall rule,
    `members.radius_for_grid`) follows it. A radius that the YAML gave (`filter_radius`, or `min_member_mm`) stays the same in mm. `settings.iterations` of the YAML is not used: it is the pass count of the first
    run. The fine run has `max_passes` as `settings.iterations`: the maximum, because the fine pass ends earlier when its loss has settled.
    The stop rules are on, with `objective_change` = `tolerance` (the rule `settled` of `verdict.stop_decision`); the window, the grayness limit
    and the time budget stay as the YAML has them. `early_stop` false (`--no-stop-rules`, MCP `stop_rules=false`) is as in a plain run: all stop
    rules are off (`settled`, `converged` and the time budget), and the fine pass runs to its maximum.
    """
    st = config.settings
    from .members import radius_for_grid
    radius = radius_for_grid(st, float(voxel_mm))
    rules = dataclasses.replace(st.stop_rules, enabled=bool(early_stop), objective_change=float(tolerance))
    return dataclasses.replace(config, settings=dataclasses.replace(
        st, voxel_size=float(voxel_mm), iterations=int(max_passes), filter_radius=radius,
        beta_schedule=shift_schedule(st.beta_schedule, offset), stop_rules=rules))


def check_fine_passes(value) -> int:
    """`--fine-passes` and `run_topopt(fine_passes=...)`: the maximum pass count of the fine run, a whole number of 1 or more.

    Raise `UserError` with options.
    """
    n = None
    if isinstance(value, int) and not isinstance(value, bool):
        n = value
    elif isinstance(value, str) and value.strip().isdecimal():
        n = int(value.strip())
    if n is None or n < 1:
        raise UserError(f"--fine-passes must be a whole number of 1 or more (the maximum pass count of the fine run), got {value!r}. "
                        "Options: give a number such as 30; without it the maximum is the pass count of the parent run.")
    return n


def check_fine_tol(value) -> float:
    """`--fine-tol` and `run_topopt(fine_tol=...)`: the tolerance of the stop rule `settled`, a number above 0 (a relative change of the objective).

    Raise `UserError` with options.
    """
    try:
        if isinstance(value, bool):
            raise ValueError
        t = float(value)
    except (TypeError, ValueError):
        t = float("nan")
    if not (t > 0.0 and math.isfinite(t)):
        raise UserError(f"--fine-tol must be a number above 0 (the relative change of the objective between two passes at which the fine pass "
                        f"ends), got {value!r}. Options: give a number such as 5e-4 (a smaller number runs more passes); without it the "
                        f"tolerance is settings.fine_pass.objective_change of the YAML, or {FINE_OBJECTIVE_CHANGE:g}.")
    return t


def tolerance_of(config, fine_tol=None) -> tuple[float, str]:
    """`(tolerance, source)` of the stop rule `settled` for a fine pass made from `config`.

    The source is `option` (`--fine-tol` or MCP `fine_tol`, which wins), `yaml` (`settings.fine_pass.objective_change`) or `default`
    (`FINE_OBJECTIVE_CHANGE`). A bad `fine_tol` raises `UserError` with options.
    """
    if fine_tol is not None:
        return check_fine_tol(fine_tol), "option"
    given = config.settings.fine_pass.objective_change
    return (float(given), "yaml") if given is not None else (FINE_OBJECTIVE_CHANGE, "default")


def check_fine_voxel(value) -> float:
    """`--fine-voxel` and `run_topopt(fine_voxel=...)`: a number of mm above 0. Raise `UserError` with options."""
    try:
        if isinstance(value, bool):
            raise ValueError
        h = float(value)
    except (TypeError, ValueError):
        h = float("nan")
    if not (h > 0.0 and math.isfinite(h)):
        raise UserError(f"--fine-voxel must be a number of mm above 0, got {value!r}. Options: give a voxel such as 0.65; without it the "
                        "fine pass takes the voxel that the part needs (`topopt check` prints it).")
    return h


@dataclass
class FineSetup:
    """What `resolve_fine` found: the config of the fine run and the facts that come before the build of the problem."""
    parent_dir: Path
    parent: "Parent"
    config: object          # the config of the fine run (voxel, maximum pass count, schedule, stop rule)
    voxel_mm: float
    voxel_from: str         # "fine_voxel" (the user gave it) or "part_voxel" (the voxel that the part needs)
    max_passes: int
    max_from: str           # "fine_passes" (the user gave it) or "parent" (the pass count of the parent run)
    tolerance: float
    tolerance_from: str     # "option" (`--fine-tol`, MCP `fine_tol`), "yaml" (`settings.fine_pass.objective_change`) or "default" (`FINE_OBJECTIVE_CHANGE`)
    offset: int
    part: dict | None       # `part_voxel(config, verify=True)` of the config as given (`voxel_mm`, `limiting`, `checked`), or None when the user gave the voxel
    facts: list
    early_stop: bool = True  # false: `--no-stop-rules`, all stop rules are off and the fine pass runs to its maximum


def resolve_fine(config, parent_dir, new_dir=None, fine_voxel=None, fine_passes=None, fine_tol=None, parent=None, early_stop: bool = True) -> FineSetup:
    """Read the parent, choose the fine voxel, the maximum and the tolerance, and make the config of the fine run. The parent is only read.

    `fine_voxel` (mm) is the voxel of the user; without it the voxel is the one that the part needs (`part_voxel` with `verify`: the measured voxel,
    tested on the grid of the fine pass, and lower by a few steps of `STEP_MM` if a region has too few voxels across there; a fact says it). A voxel equal to the one of the
    parent is refused (`--continue-from` adds passes on the same grid). A voxel coarser than the parent is allowed, with a fact.
    `fine_passes` is the maximum pass count; without it the maximum is the pass count of the parent run (the pass of its last checkpoint).
    `fine_tol` is the tolerance of the stop rule `settled`; without it `settings.fine_pass.objective_change` of the YAML, else `FINE_OBJECTIVE_CHANGE`
    (`tolerance_of`).
    `parent` is a `Parent` that the caller read already (the one read goes to every step). `early_stop` false is `--no-stop-rules`: no stop rule at
    all. Raise `UserError` with options.
    """
    from . import checkpoint

    parent_dir = Path(parent_dir)
    if not parent_dir.is_dir():
        raise UserError(f"--fine-from: {parent_dir} is not a folder. Give the run folder of a finished run (the --out of the earlier `topopt run`).")
    if new_dir is not None:
        checkpoint.check_new_folder(parent_dir, new_dir)
    max_passes = None if fine_passes is None else check_fine_passes(fine_passes)
    tol, tol_from = tolerance_of(config, fine_tol)
    if parent is None:
        parent = read_parent(parent_dir)
    facts = list(parent.facts)
    max_from = "parent" if max_passes is None else "fine_passes"
    if max_passes is None:
        max_passes = max(1, int(parent.iteration))
    part = None
    if fine_voxel is None:
        part = part_voxel(config, verify=True)
        voxel, how = float(part["voxel_mm"]), "part_voxel"
        note = check_fact(part)
        if note:
            facts.append(note)
    else:
        voxel, how = check_fine_voxel(fine_voxel), "fine_voxel"
    if abs(voxel - parent.h) < SAME_VOXEL:
        raise UserError(
            f"--fine-from: the fine voxel is {voxel:g} mm, the same as the voxel of the parent run {parent_dir}. A fine pass changes the grid. "
            "Options: --continue-from PARENT with --more-iterations N adds passes on the same grid; --fine-voxel H gives another voxel"
            + ("" if how == "fine_voxel" else f" (the voxel that the part needs is {voxel:g} mm: the parent run already has it)") + ".")
    if how == "part_voxel" and voxel > parent.h:
        raise UserError(
            f"--fine-from: the voxel that the part needs, {voxel:g} mm, is not finer than the voxel of the parent run {parent_dir}, {parent.h:g} mm. "
            "A fine pass with it would make the grid coarser. Options: --fine-voxel H forces a voxel (a coarser voxel than the parent is allowed, with a fact); "
            "--continue-from PARENT with --more-iterations N adds passes on the same grid.")
    if voxel > parent.h:
        facts.append(f"fine pass: the fine voxel {voxel:g} mm is coarser than the voxel of the parent run, {parent.h:g} mm. The run goes on; "
                     "a coarse pass keeps less detail than the parent has.")
    offset = parent.iteration + parent.offset
    return FineSetup(parent_dir, parent, with_fine_settings(config, voxel, max_passes, offset, tol, early_stop), voxel, how, max_passes, max_from, tol,
                     tol_from, offset, part, facts, bool(early_stop))


def fine_record(setup: FineSetup, source: str) -> dict:
    """The `fine_from` record of run.json and of each checkpoint of the fine run (JSON data). `FineStart` adds `map`, `volume` and `origin_shift_mm`.

    `max_passes`, `tolerance`, `voxel_mm` and `schedule_offset` are what `checkpoint.config_of_fine` needs to build the config of a resume again.
    """
    p = setup.parent
    meta = p.ck.meta if p.ck is not None else {}
    rec = {"run_dir": str(setup.parent_dir.resolve()), "iteration": int(p.iteration), "parent_iterations": meta.get("n_iterations"),
           "parent_voxel_mm": float(p.h), "parent_shape": [int(v) for v in p.shape], "schedule_offset": int(setup.offset),
           "voxel_mm": float(setup.voxel_mm), "voxel_from": setup.voxel_from,
           "max_passes": int(setup.max_passes), "max_from": setup.max_from,
           "stop_rule": SETTLED_RULE if setup.early_stop else None, "tolerance": float(setup.tolerance), "tolerance_from": setup.tolerance_from,
           "window": int(setup.config.settings.stop_rules.window), "source": source,
           "checkpoint": p.ck.file if p.ck is not None else None,
           "checkpoint_sha256": p.ck.sha256 if p.ck is not None else None, "topopt_version": meta.get("topopt_version")}
    if setup.part is not None:
        lim = setup.part["limiting"]
        rec["part_voxel_mm"] = setup.part["voxel_mm"]
        rec["limiting"] = lim["name"]
        rec["limiting_role"] = lim["role"]
        rec["limiting_thickness_mm"] = lim["thickness_mm"]
        rec["grid_check"] = setup.part["checked"]       # the test of the voxel on the grid of the fine pass: start_mm, steps, region, ok, skipped
    return rec


# --- the parent run ------------------------------------------------------------------------------------

@dataclass
class Parent:
    """The data of a finished parent run that the mapping reads.

    `labels` is the voxel class of each parent voxel (`voxelize.EMPTY`, `DESIGN`, ...). `rho` is the physical density of the last
    snapshot (0 to 1; the fixed and load voxels are 1), or None when the folder has no snapshot. `x` is the design variable of each
    parent voxel from the checkpoint (NaN outside the design voxels), or None when the folder has no valid checkpoint. `ck` is the
    checkpoint (None without one). `offset` is the offset of the beta schedule that the parent itself carries (a parent that is a fine pass).
    """
    run_dir: Path
    shape: tuple
    origin: np.ndarray
    h: float
    labels: np.ndarray
    rho: np.ndarray | None
    x: np.ndarray | None
    iteration: int
    volume_fraction: float | None
    facts: list
    ck: object = None
    offset: int = 0

    def release(self) -> None:
        """Let go of the arrays (the voxel classes, the density, the design variable and the checkpoint). The numbers stay.

        The mapping reads the arrays once. A caller holds the `Parent` (the plan of the run) for as long as the run lasts, so without this the
        three grids of the parent and its checkpoint stay in memory through all passes of the fine pass.
        """
        self.labels = self.rho = self.x = self.ck = None


def last_snapshot(run_dir: Path) -> tuple[int, Path]:
    """`(pass, path)` of the snapshot with the highest pass number. Raise `UserError` when there is none."""
    best = None
    folder = Path(run_dir) / "snapshots"
    if folder.is_dir():
        for p in folder.iterdir():
            name = p.name
            if name.startswith("it_") and name.endswith(".bin") and name[3:-4].isdigit():
                n = int(name[3:-4])
                if best is None or n > best[0]:
                    best = (n, p)
    if best is None:
        raise UserError(f"{run_dir}: there is no snapshot in snapshots/, so there is no density to map")
    return best


def read_parent(run_dir) -> Parent:
    """Read a finished run folder. Raise `UserError` with options when it is not a finished run or has no design to map.

    The design variable of the checkpoint is the source of the start design. Without a valid checkpoint the density of the last
    snapshot is the source (a fact says so). A checkpoint that is not complete is refused: the run did not finish.
    """
    from .checkpoint import design_ids, read_checkpoint

    d = Path(run_dir)
    try:
        run = json.loads((d / "run.json").read_text())
    except (OSError, ValueError):
        raise UserError(f"--fine-from: {d} has no readable run.json. Give the run folder of a finished run (the --out of `topopt run`).") from None
    shape = tuple(int(v) for v in run["shape"])
    n_vox = int(np.prod(shape))
    raw = (d / "labels.bin").read_bytes() if (d / "labels.bin").is_file() else b""
    if len(raw) != n_vox:
        raise UserError(f"--fine-from: {d}/labels.bin has {len(raw)} bytes and the grid {list(shape)} needs {n_vox}. The run folder is not complete.")
    labels = np.frombuffer(raw, np.uint8).reshape(shape)
    facts, x_full, ck, it = [], None, None, None
    try:
        ck = read_checkpoint(d)
    except UserError as exc:
        facts.append(f"fine pass: the parent has no usable checkpoint ({str(exc).splitlines()[0]}); the density of its last snapshot is the "
                     "source of the start design")
    if ck is not None:
        if not ck.meta.get("complete"):
            raise UserError(f"--fine-from: the checkpoint of {d} is at pass {ck.iteration} of {ck.meta.get('n_iterations')} and the run did not finish. "
                            "Options: finish it with `topopt run ... --resume`; or give the folder of a finished run.")
        ids = design_ids(ck.arrays["design_mask"], n_vox)
        if ids.size != int((labels == DESIGN).sum()):
            raise UserError(f"--fine-from: the design voxels of the checkpoint of {d} ({ids.size}) are not those of labels.bin "
                            f"({int((labels == DESIGN).sum())}). The run folder is damaged. Option: give another finished run.")
        x_full = np.full(n_vox, np.nan)
        x_full[ids] = ck.arrays["x"]
        x_full = x_full.reshape(shape)
        it = ck.iteration
    rho, snap_it = None, None
    try:
        snap_it, snap = last_snapshot(d)
        sraw = snap.read_bytes()
        if len(sraw) == n_vox:
            rho = np.frombuffer(sraw, np.uint8).reshape(shape).astype(np.float64) / 255.0
    except UserError:
        pass
    if ck is None:
        if rho is None:
            raise UserError(f"--fine-from: {d} has no usable checkpoint and no snapshot, so there is no design to map. Give another finished run.")
        if _state_of(d) != "done":
            raise UserError(f"--fine-from: {d} has no checkpoint, and its status.json does not say that the run is done. "
                            "Options: give the folder of a finished run.")
        it = snap_it
    elif snap_it is not None and snap_it != it:
        facts.append(f"fine pass: the checkpoint is at pass {it} and the last snapshot is pass {snap_it}")
    fine = (ck.meta.get("fine_from") if ck is not None else None) or run.get("fine_from") or {}
    return Parent(d, shape, np.asarray(run["origin"], float), float(run["h"]), labels, rho, x_full, int(it),
                  run.get("volume_fraction"), facts, ck, int(fine.get("schedule_offset", 0)))


def _state_of(d: Path) -> str | None:
    try:
        return json.loads((d / "status.json").read_text()).get("state")
    except (OSError, ValueError):
        return None


# --- the mapping ---------------------------------------------------------------------------------------

def _interp(field: np.ndarray, weight: np.ndarray, centres: np.ndarray, origin: np.ndarray, h: float):
    """Trilinear interpolation of `field` and `weight` at the points `centres` (mm). Return `(num, den)`.

    The grid is cell-centred: voxel `i` has its centre at `origin + h * (i + 0.5)`. Both arrays are zero outside the grid,
    so a point near the border takes less weight. `num / den` is the weighted mean of the source voxels around the point.
    """
    from scipy.ndimage import map_coordinates

    f = np.pad(field * weight, 1)
    w = np.pad(weight, 1)
    num = np.empty(len(centres))
    den = np.empty(len(centres))
    for a in range(0, len(centres), CHUNK):
        u = ((centres[a:a + CHUNK] - origin) / h - 0.5 + 1.0).T      # +1: the pad
        num[a:a + CHUNK] = map_coordinates(f, u, order=1, mode="constant", cval=0.0)
        den[a:a + CHUNK] = map_coordinates(w, u, order=1, mode="constant", cval=0.0)
    return num, den


def map_field(parent: Parent, centres: np.ndarray, source: str = "x") -> tuple[np.ndarray, dict]:
    """The parent design at the points `centres` (mm), as a mean of the source voxels around each point.

    `source` "x": the design variables of the checkpoint. Only the design voxels of the parent are sources: the filter of the new run
    counts the fixed and load voxels as density 1 by itself, so they must not enter the variable twice.
    `source` "rho": the physical density of the last snapshot. The fixed and load voxels are sources with density 1; the empty and
    keep-out voxels are no sources (they are no part of the design).
    A point with no source voxel around it takes the value of the nearest source voxel. Return `(values, info)`.
    """
    if source not in SOURCES:
        raise UserError(f"the source of the start design must be one of {', '.join(SOURCES)}, got {source!r}")
    lab = parent.labels
    if source == "x":
        if parent.x is None:
            raise UserError("the parent has no checkpoint, so the design variables are not known. Options: source 'rho' (the snapshot density)")
        valid = lab == DESIGN
        field = np.where(valid, parent.x, 0.0)
    else:
        if parent.rho is None:
            raise UserError("the parent has no snapshot, so the density is not known. Options: source 'x' (the design variables of the checkpoint)")
        valid = (lab == DESIGN) | (lab == FIXED) | (lab == LOAD)
        field = np.where(valid, parent.rho, 0.0)
    weight = valid.astype(np.float64)
    num, den = _interp(field, weight, centres, parent.origin, parent.h)
    ok = den > MIN_WEIGHT
    out = np.empty(len(centres))
    out[ok] = num[ok] / den[ok]
    far = int((~ok).sum())
    if far:
        from scipy import ndimage

        _d, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
        cell = np.clip(np.floor((centres[~ok] - parent.origin) / parent.h).astype(int), 0, np.array(parent.shape) - 1)
        out[~ok] = field[idx[0][tuple(cell.T)], idx[1][tuple(cell.T)], idx[2][tuple(cell.T)]]
    info = {"points": len(centres), "no_source": far, "weak_source": int((ok & (den < 0.5)).sum()),
            "source": source, "parent_iteration": parent.iteration}
    return out, info


def fit_volume(model, x: np.ndarray, beta: float, target: float) -> tuple[np.ndarray, float, float, float]:
    """Shift `x` by one number `s` (then clip to 0..1) so that the mean physical density of the design voxels is `target`.

    The density is the one of the first pass: the filter and the projection of `beta`. The mean grows with `s`, so a bisection
    finds `s`. Return `(x, s, mean, mean_before)`: `mean_before` is the mean density at `s = 0`. If `target` cannot be reached, `s` is
    the end of the range (-1 or 1) and `mean` shows the miss.
    """
    from . import symmetry
    from .filter import filtered_density, heaviside_project
    from .optimize import ETA

    def mean_rho(s: float) -> float:
        xs = np.clip(x + s, 0.0, 1.0)
        rho_tilde = filtered_density(model.F, model.b, symmetry.average(model, xs))
        return float(heaviside_project(rho_tilde, beta, ETA).mean())

    before = mean_rho(0.0)
    lo, hi = -1.0, 1.0
    m_lo, m_hi = mean_rho(lo), mean_rho(hi)
    if target <= m_lo:
        return np.clip(x + lo, 0.0, 1.0), lo, m_lo, before
    if target >= m_hi:
        return np.clip(x + hi, 0.0, 1.0), hi, m_hi, before
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean_rho(mid) < target:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return np.clip(x + s, 0.0, 1.0), s, mean_rho(s), before


def parent_mean_density(parent: Parent) -> float | None:
    """The mean physical density of the design voxels of the parent at its last pass, or None when the folder does not say.

    It is the `volume_fraction` of the history record of the pass of the checkpoint (the last record when that pass has none), and else the mean of
    the density of the last snapshot over the design voxels (8 bit, so it is exact to 0.002).
    """
    best = None
    try:
        for line in (Path(parent.run_dir) / "history.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if rec.get("volume_fraction") is not None and (best is None or int(rec.get("iteration", -1)) <= parent.iteration):
                best = float(rec["volume_fraction"])
    except (OSError, ValueError, TypeError):
        best = None
    if best is None and parent.rho is not None:
        best = float(parent.rho[parent.labels == DESIGN].mean())
    return best


def volume_target(parent: Parent, config) -> tuple[float, str]:
    """`(target, how)`: the mean density that the start design is fitted to.

    A run with the objective `compliance` has the volume row, so the target is `settings.volume_fraction` (`how`: `volume_fraction`). A run with
    the objective `mass` has no volume row: `settings.volume_fraction` is only its start density. The target is then the final mean density of
    the parent (`how`: `parent`), so that the fit removes the drift of the mapping and does not push the design to a number that the run never
    had. A parent that does not say its mean density gives the fraction of the YAML (`how`: `volume_fraction`).
    """
    fraction = float(config.settings.volume_fraction)
    if config.settings.objective != "mass":
        return fraction, "volume_fraction"
    mean = parent_mean_density(parent)
    return (fraction, "volume_fraction") if mean is None else (mean, "parent")


def start_design(parent: Parent, model, beta: float, source: str = "x", log=None, record: dict | None = None) -> np.ndarray:
    """The start design of the fine run: the parent design on the design voxels of `model`, with the mean density fitted to the volume target.

    `beta` is the Heaviside beta of pass 0 of the fine run. `log` is called with each fact line. `record` (a dict) gets the keys `map`
    and `volume` (the numbers of the facts), for run.json.
    """
    say = log or (lambda line: None)
    problem = model.problem
    centres = model.design_centers()
    values, info = map_field(parent, centres, source)
    target, target_from = volume_target(parent, problem.config)
    x0 = np.clip(values, 0.0, 1.0)
    x0, shift, mean, before = fit_volume(model, x0, beta, target)
    shift_mm = np.abs(np.asarray(problem.origin) - parent.origin).max()
    say(f"fine pass: start design from pass {parent.iteration} of {parent.run_dir} (grid {list(parent.shape)}, voxel {parent.h:g} mm) "
        f"onto {model.n_design} design voxels (voxel {problem.h:g} mm), source {source}; the origins differ by {shift_mm:.3f} mm "
        f"({shift_mm / parent.h:.3f} parent voxels)")
    if target_from == "parent":
        goal = (f"it was fitted to the final mean density of the parent, {target:.5f}, with a shift of {shift:+.4f} of the design variable (mean density "
                f"{mean:.5f}). settings.objective is mass: the run has no volume row, and settings.volume_fraction is only its start density, so it is not "
                "the target here")
    else:
        goal = f"it was fitted to the volume target {target:.5f} with a shift of {shift:+.4f} of the design variable (mean density {mean:.5f})"
    say(f"fine pass: {info['no_source']} of {info['points']} design voxels had no parent voxel around them and took the nearest one; "
        f"{info['weak_source']} had less than half of the weight. The mean density of the mapped design is {before:.5f}; {goal}")
    if abs(mean - target) > 1e-4:
        say(f"fine pass: the fit did not reach the {'mean density of the parent' if target_from == 'parent' else 'volume target'} "
            f"(mean density {mean:.5f}, target {target:.5f}): the shift is at the end of its range")
    if record is not None:
        record["map"] = {"points": info["points"], "no_source": info["no_source"], "weak_source": info["weak_source"]}
        record["volume"] = {"target": target, "target_from": target_from, "mean_before": before, "shift": shift, "mean_after": mean}
        record["origin_shift_mm"] = float(shift_mm)
    return x0


class FineStart:
    """The start design of a fine run, for `optimize.run(..., x0=FineStart)`.

    Calling it with the `Model` maps the design of the parent and fits the volume. Then `lines` holds the fact lines for the run log, and
    `record` (the `fine_from` record) has the numbers of the mapping. `optimize.run` calls it before it writes run.json.
    """

    def __init__(self, parent: Parent, record: dict, beta: float, source: str, extra_lines=()):
        self.parent, self.record, self.beta, self.source = parent, record, float(beta), source
        self.lines: list[str] = list(extra_lines)

    def __call__(self, model) -> np.ndarray:
        x = start_design(self.parent, model, self.beta, self.source, log=self.lines.append, record=self.record)
        self.parent.release()    # the arrays of the parent (three grids and the checkpoint) are not needed after the mapping, also not in the plan
        self.parent = None
        return x


def plan_line(setup: FineSetup) -> str:
    """The first log line of a fine pass: where it starts from, the voxel, the maximum, and when it ends."""
    p, st = setup.parent, setup.config.settings
    how = "the voxel of --fine-voxel" if setup.voxel_from != "part_voxel" else "the voxel that the part needs"
    cap = ("the pass count of the parent run" if setup.max_from == "parent" else "--fine-passes")
    tol = {"default": "the default", "yaml": "settings.fine_pass.objective_change"}.get(setup.tolerance_from, "--fine-tol")
    head = f"fine pass: new run from the design of {setup.parent_dir} (pass {p.iteration}, voxel {p.h:g} mm) at voxel {setup.voxel_mm:g} mm ({how}). "
    ends = (f"It ends when the loss has settled (objective change at or below {setup.tolerance:g}, {tol}, over {st.stop_rules.window} passes at one beta), "
            f"or after {setup.max_passes} passes at most ({cap}). " if setup.early_stop else
            f"All stop rules are off (--no-stop-rules): it runs {setup.max_passes} passes ({cap}). ")
    return head + ends + f"The beta schedule goes on from pass {setup.offset} of the earlier runs."


def fine_start(setup: FineSetup) -> FineStart:
    """The start design of the fine run of `setup`, for `optimize.run(problem, out, x0=...)`. It holds the `fine_from` record and the log lines."""
    from .optimize import beta_at

    if setup.parent.labels is None:     # a start before this one let go of the arrays (`Parent.release`): a restart at pass 0 after a failure of gpu-mg reads the parent again
        setup = dataclasses.replace(setup, parent=read_parent(setup.parent.run_dir))
    source, facts = choose_source(setup.parent)
    return FineStart(setup.parent, fine_record(setup, source), beta_at(setup.config.settings.beta_schedule, 0), source,
                     extra_lines=[plan_line(setup), *setup.facts, *facts])


def choose_source(parent: Parent) -> tuple[str, list]:
    """`(source, facts)`: `x` when the parent has a checkpoint, else `rho` with a fact."""
    if parent.x is not None:
        return "x", []
    return "rho", ["fine pass: the source of the start design is the density of the last snapshot of the parent (no checkpoint). That density has the "
                   "filter and the projection in it already, so the fine pass filters it twice: the start is worse than with the design variable "
                   "(measured 2026-10-26 on a 5-inch drone arm at 0.7 mm, AMG solver: weighted compliance 854 against 726 at pass 0)."]


def end_line(verdict: dict, rec: dict, n_passes: int | None = None) -> str | None:
    """The log line that says how a fine pass ended: the loss settled, or the maximum of passes ended it first. None for any other end.

    `verdict` is the verdict of the run (`verdict.build_verdict`), `rec` the `fine_from` record. `n_passes` is the pass count of this run when it is
    not the maximum of the record (a continuation of a fine pass has K + N). A run that a request or a time budget ended has no line here: its
    verdict text says it.
    """
    stop, state = verdict.get("stop"), verdict.get("state")
    n, tol = int(rec["max_passes"] if n_passes is None else n_passes), float(rec["tolerance"])
    if rec.get("stop_rule") != SETTLED_RULE:
        return f"fine pass: all stop rules were off (--no-stop-rules), so the fine pass ran its {n} passes." if state == "iteration_limit" else None
    if state == "stopped_by_rule" and stop and stop.get("rule") == SETTLED_RULE:
        return (f"fine pass: the loss settled at pass {stop['iteration']} (the maximum is {n} passes): the objective change is at or below "
                f"{tol:g} over the last {rec['window']} passes.")
    if state == "iteration_limit":
        oc = ((verdict.get("measures") or {}).get("objective_change") or {})
        value = oc.get("value")
        if value is None:
            why = (f"the objective change was not judged: it needs {oc.get('passes_needed')} passes in a row at one beta, "
                   f"the fine pass has {oc.get('passes_at_beta')} at beta {oc.get('beta', 0):g}")
        else:
            why = f"the objective change is {value:.3g} over the last {rec['window']} passes, above the tolerance {tol:g}"
        return (f"fine pass: the maximum of {n} passes ended the fine pass before the loss settled ({why}). "
                "Options: run the fine pass again from the same parent with a higher --fine-passes (MCP fine_passes), or with a larger "
                "--fine-tol (MCP fine_tol).")
    return None
