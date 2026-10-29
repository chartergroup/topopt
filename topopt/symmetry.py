"""Mirror symmetry of the design by an exact voxel index map (CONTRACTS.md section 26.1).

The YAML key `mirror` lists up to three axis-aligned planes with distinct axes. Each plane is a grid
plane. Only the grid origin on the plane axis moves, so that the plane lies on a voxel face (parity
"face") or through the centres of a voxel layer (parity "centre"). With `k` the position of the plane in
half voxels above the grid origin, the voxel `i` along the axis has the partner `k - 1 - i`, and the node
`i` has the partner `k - i`. `k` is even for the face parity and odd for the centre parity.

The map on the design voxels is exact and it is an involution `P`. A design voxel whose image is not a
design voxel (an orphan) maps to itself, so it stays a free variable. The model applies the map in two
places (`Model.density` and `Model.gradient_wrt_x`):

    forward:   x -> 0.5 (x + x[perm]), once for each plane in turn, then the filter, then the projection
    reverse:   the gradient after the transposed chain -> 0.5 (g + g[perm]), the planes in reverse order

The reverse order makes the second step the exact transpose of the first, also with orphans (two maps
commute only without orphans). The method is the one of the pipeline (`pp_core_final.py` `sym` and
`symT`). It does not need the filter to commute with the map. The map is exact here, because the grid
is a grid: the pipeline has a mesh and uses the nearest centroid.

This module has no import from the rest of the package at load time, so `config.py` can use it. It loads
numpy only inside the functions that need it: `topopt.cli` must not load numpy at import. Every function
that reports returns facts (numbers and text). None of them blocks a run.
"""

from __future__ import annotations

import math

AXES = ("x", "y", "z")
ALIGNS = ("auto", "face", "centre")
MAX_PLANES = 3
SYMMETRIC_TOL = 1e-9   # a load asymmetry at or below this value is reported as "symmetric"
GRID_TOL = 1e-4        # of a voxel: a region face this close to the grid plane counts as on it (float32 STL vertices)
ORPHAN_MOST = 0.5      # a share of orphans at or above this value gets the words "little effect" and the options

# The one sentence that the refusals of the YAML loader share: what is supported, and the option.
MIRROR_HELP = (
    "Supported: a list of up to 3 planes with distinct axes, each {normal: x|y|z, at_mm: <mm>, "
    "align: auto|face|centre (optional)}. To mirror about another plane, re-orient the part (the STL files "
    "and the loads) so that the plane is axis-aligned")


# --- the YAML values ---------------------------------------------------------


def axis_of_normal(value) -> str:
    """The axis name ("x", "y" or "z") of a plane normal. Raise `ValueError` with a one-line fact.

    `value` is an axis name (a sign is allowed and has no meaning: a plane has no direction) or a list of 3
    numbers with one non-zero component. A vector that is not along an axis is refused: the plane is not
    axis-aligned.
    """
    if isinstance(value, str):
        name = value.strip().lower().lstrip("+-")
        if name in AXES:
            return name
        raise ValueError(f"normal {value!r} is not x, y or z")
    ok = (isinstance(value, (list, tuple)) and len(value) == 3
          and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value))
    if not ok:
        raise ValueError(f"normal must be x, y, z or a vector of 3 numbers, got {value!r}")
    size = max(abs(v) for v in value)
    if size == 0:
        raise ValueError("normal is the zero vector")
    along = [i for i, v in enumerate(value) if abs(v) > 1e-9 * size]
    if len(along) != 1:
        raise ValueError(f"normal {list(value)} is not axis-aligned")
    return AXES[along[0]]


# --- the grid origin and the position of the plane ----------------------------


def _half_steps(plane, lo_axis: float, h: float) -> int:
    """The smallest `k` with the plane at or above `lo_axis` after the shift, for the parity of `plane.align`.

    `k` counts half voxels from the new origin to the plane. The new origin is `at_mm - k h / 2`. It is
    above `lo_axis` by at most `GRID_TOL` h (a rounding of the STL vertices, which cuts no voxel centre) and below it by
    less than one voxel.
    """
    k = math.ceil((plane.at_mm - lo_axis) / (h / 2.0) - 2.0 * GRID_TOL)   # GRID_TOL h in mm: float32 STL rounding
    if plane.align == "face" and k % 2 != 0:
        k += 1
    elif plane.align == "centre" and k % 2 == 0:
        k += 1
    return k


def shift_lo(planes, lo, h: float):
    """The lower corner of the grid with each plane on a grid plane. `voxelize` calls this once.

    Without planes it returns `lo` itself. With `align` "auto" the shift is the smallest one (below half a
    voxel). With "face" or "centre" it is the smallest shift that gives that parity (below one voxel).
    """
    if not planes:
        return lo
    lo = lo.copy()
    for p in planes:
        a = AXES.index(p.axis)
        lo[a] = p.at_mm - _half_steps(p, float(lo[a]), h) * h / 2.0
    return lo


def half_steps_of(origin, axis: str, at_mm: float, h: float) -> int:
    """`k` of a plane on a grid with this origin: the plane is `k` half voxels above the origin."""
    return int(round((at_mm - float(origin[AXES.index(axis)])) / (h / 2.0)))


def parity_of(k: int) -> str:
    return "face" if k % 2 == 0 else "centre"


def label_mismatches(labels, axis: int, k: int) -> int:
    """Voxels whose label differs from the label of their mirror image.

    A voxel whose image is outside the grid is compared with EMPTY. Each pair of voxels counts twice.
    """
    import numpy as np
    n = labels.shape[axis]
    image = k - 1 - np.arange(n)
    inside = (image >= 0) & (image < n)
    shape = [1, 1, 1]
    shape[axis] = n
    mirrored = np.where(inside.reshape(shape), np.take(labels, np.clip(image, 0, n - 1), axis=axis), 0)
    return int((labels != mirrored).sum())


# --- the parity choice ---------------------------------------------------------


def resolve(config):
    """Decide the parity of each plane with `align` "auto". Return `(config, planes)`.

    `config` has every `align` set to "face" or "centre" (its `voxelize` gives the final grid).
    `planes` is a list with one dict for each plane (`None` without a mirror key): axis, at_mm, align (as
    given), parity (the one that is used) and `label_mismatches` ({"face": n, "centre": n}, `None` for a
    parity that was not tried). The other parity costs one more voxelization of the regions for each plane.
    The count of the plane is the one of the grid at that step: the planes are decided in the list order,
    and each later plane sees the parity that an earlier plane got.

    With `align` "auto", the parity with fewer mismatches is used. With `align` "face" or "centre", the given
    parity is used and the other parity is a fact only: the dict of the plane gets `other_parity`
    ({"parity", "label_mismatches", "orphans"} of the other grid), and `label_mismatches` of the other parity
    stays `None`. The key is missing for "auto" and when the other grid cannot be built.
    """
    if not config.mirror:
        return config, None
    import dataclasses
    from .config import UserError
    from .voxelize import voxelize

    def measure(planes, orphans_of=None):   # the (k, mismatches) of each plane on the grid that `planes` gives
        origin, h, shape, labels, _ids = voxelize(dataclasses.replace(config, mirror=tuple(planes)))
        out = []
        for p in planes:
            k = half_steps_of(origin, p.axis, p.at_mm, h)
            out.append((k, label_mismatches(labels, AXES.index(p.axis), k)))
        if orphans_of is None:
            return out
        k = out[orphans_of][0]   # the orphans of one plane (the grid is at hand: no new voxelization)
        return out, sum(design_map(labels, AXES.index(planes[orphans_of].axis), k)[1].values())

    state = list(config.mirror)
    info = [{"axis": p.axis, "at_mm": p.at_mm, "align": p.align, "parity": None,
             "label_mismatches": {"face": None, "centre": None}} for p in state]
    current = measure(state)
    for i, p in enumerate(state):
        k, count = current[i]
        near = parity_of(k)   # the parity of the smallest shift (a given `align` has its own parity here)
        info[i]["label_mismatches"][near] = count
        info[i]["parity"] = near
        other = "centre" if near == "face" else "face"
        trial = list(state)
        trial[i] = dataclasses.replace(p, align=other)
        if p.align != "auto":   # the parity is given: it stays, and the other parity is a fact
            try:
                tried, orphans = measure(trial, i)
            except UserError:   # the other grid cannot be built (for example the grid guard): no fact
                continue
            info[i]["other_parity"] = {"parity": other, "label_mismatches": tried[i][1], "orphans": orphans}
            continue
        try:
            tried = measure(trial)
        except UserError:   # the other grid cannot be built (for example the grid guard): the smallest shift stays
            tried = None
        if tried is not None:
            info[i]["label_mismatches"][other] = tried[i][1]
        if tried is not None and tried[i][1] < count:   # fewer mismatches: use the other parity
            info[i]["parity"], current = other, tried
        state[i] = dataclasses.replace(p, align=info[i]["parity"])
    return dataclasses.replace(config, mirror=tuple(state)), info


# --- the index map ---------------------------------------------------------------


def design_map(labels, axis: int, k: int):
    """The partner map of the design voxels for one plane. Return `(perm, causes)`.

    `perm[p]` is the index (in the order of the design voxels, which is the order of `x`) of the mirror
    image of design voxel `p`. An orphan has `perm[p] = p`. `causes` counts the orphans by what their image is:
    outside_grid, empty, keep_out, fixed, load (the label LOAD: a load region or a solid region), and `other`
    (a label code that has no name here: 0 today). The causes add up to the number of orphans.
    """
    import numpy as np
    from .voxelize import DESIGN, EMPTY, FIXED, KEEP_OUT, LOAD
    shape = labels.shape
    flat = labels.reshape(-1)
    design = np.flatnonzero(flat == DESIGN)
    ijk = list(np.unravel_index(design, shape))
    image = k - 1 - ijk[axis]
    inside = (image >= 0) & (image < shape[axis])
    ijk[axis] = np.clip(image, 0, shape[axis] - 1)
    image_flat = np.ravel_multi_index(tuple(ijk), shape)
    design_index = np.full(flat.size, -1, dtype=np.intp)
    design_index[design] = np.arange(design.size)
    partner = np.where(inside, design_index[image_flat], -1)
    perm = np.where(partner >= 0, partner, np.arange(design.size))
    orphan = partner < 0
    image_label = np.where(inside, flat[image_flat], 255)
    causes = {"outside_grid": int((orphan & ~inside).sum())}
    for name, code in (("empty", EMPTY), ("keep_out", KEEP_OUT), ("fixed", FIXED), ("load", LOAD)):
        causes[name] = int((orphan & (image_label == code)).sum())
    causes["other"] = int(orphan.sum()) - sum(causes.values())
    return perm, causes


def perms_of(model) -> tuple:
    """The index maps of a model, one for each plane. An empty tuple without a mirror key.

    The maps are built from `model.problem` at the first call and kept on the model.
    """
    perms = model.__dict__.get("_mirror_perms")
    if perms is None:
        info = getattr(model.problem, "mirror", None)
        perms = () if not info else tuple(
            design_map(model.problem.labels, AXES.index(p["axis"]), p["k"])[0] for p in info["planes"])
        model._mirror_perms = perms
    return perms


def average(model, v, transpose: bool = False):
    """The mirror average of `v` (the last axis is the design voxels): `0.5 (v + v[perm])`, plane by plane.

    `transpose=False` is the forward step: planes in list order. `transpose=True` is the step for the
    gradient: planes in reverse order, so that it is the exact transpose of the forward step. Without a
    mirror key, `v` itself comes back (the same object).
    """
    perms = perms_of(model)
    if not perms:
        return v
    for perm in (perms[::-1] if transpose else perms):
        v = 0.5 * (v + v[..., perm])
    return v


def record(model, x, rho) -> None:
    """Keep the mirror deviation of the last call of `Model.density` in `problem.mirror["end"]`.

    `x` is the averaged design (the input of the filter) and `rho` is the physical density. The values are
    `max|v - v[perm]|` for each plane. After the last pass of a run they are the end values.
    The optimizer variable itself is not the design: with two planes and orphans it is symmetric about the first plane only.
    """
    perms = perms_of(model)
    if perms:
        import numpy as np
        model.problem.mirror["end"] = {
            "x_deviation": [float(np.abs(x - x[p]).max()) for p in perms],
            "rho_deviation": [float(np.abs(rho - rho[p]).max()) for p in perms]}


# --- the loads ---------------------------------------------------------------------


def load_asymmetry(forces, shape, axis: int, k: int) -> list:
    """`|f - R f| / |f|` for each load case about one plane.

    `forces` is the `(n_cases, 3 n_nodes)` array of the problem. `R` moves each nodal force to the mirror
    node (the node `i` along the axis goes to `k - i`) and flips the force component along the axis. A node
    with no image in the grid has no partner. The ratio is 0 for a symmetric load and 2 for an antisymmetric
    load. It is 0 for a load case with no force.
    """
    import numpy as np
    node_shape = tuple(s + 1 for s in shape)
    out = []
    for f in forces:
        fv = f.reshape(-1, 3)
        nodes = np.flatnonzero(fv.any(axis=1))
        vec = fv[nodes]
        ijk = np.stack(np.unravel_index(nodes, node_shape), axis=1)
        ijk[:, axis] = k - ijk[:, axis]
        ok = (ijk[:, axis] >= 0) & (ijk[:, axis] <= shape[axis])
        image = vec[ok].copy()
        image[:, axis] *= -1.0
        ids = np.concatenate([nodes, np.ravel_multi_index(tuple(ijk[ok].T), node_shape)])
        uniq, inv = np.unique(ids, return_inverse=True)
        diff = np.zeros((uniq.size, 3))
        np.add.at(diff, inv, np.concatenate([vec, -image]))
        size = float(np.sqrt((vec ** 2).sum()))
        out.append(float(np.sqrt((diff ** 2).sum())) / size if size > 0 else 0.0)
    return out


# --- the facts of a problem ----------------------------------------------------------


def problem_facts(info, origin, h: float, labels, forces, case_names) -> dict:
    """The mirror facts of a built problem (`Problem.mirror`, and the key `mirror` of run.json).

    `info` is the list that `resolve` returns. The result adds, for each plane: `k`, `plane_index` (the node
    index `k / 2` for the face parity, the voxel index `(k - 1) / 2` for the centre parity), the label
    mismatches of the grid that is used, the orphans (`count`, `share` of the design voxels, and
    the count for each cause), and `grid_range_mm` (the extent of the grid along the plane axis).
    A plane with a given `align` (face or centre) also has `other_parity` when `resolve` measured it: the
    other parity, its label mismatches and its orphans. `load_asymmetry` has one entry for each load case.

    Raise `RuntimeError` (an internal error, exit code 2) if a plane is not a grid plane of this origin: `2 (at_mm - origin) / h` must be a whole
    number `k` within `2 GRID_TOL`. `voxelize` makes it so. The check stops a map about another plane (for example
    after a change of the origin in `voxelize`) with the numbers, instead of a silent wrong map.
    """
    from .voxelize import DESIGN
    n_design = int((labels == DESIGN).sum())
    planes = []
    for p in info:
        axis = AXES.index(p["axis"])
        k = half_steps_of(origin, p["axis"], p["at_mm"], h)
        half_steps = (p["at_mm"] - float(origin[axis])) / (h / 2.0)
        if abs(half_steps - k) > 2.0 * GRID_TOL:
            raise RuntimeError(
                f"the mirror plane {p['axis']} = {p['at_mm']:g} mm is not a grid plane: origin "
                f"{float(origin[axis]):.6g} mm, voxel {h:g} mm, the plane is {half_steps:.6f} half voxels above the origin "
                f"(a whole number is needed, nearest {k}). The origin must come from voxelize with the mirror key")
        parity = parity_of(k)
        mismatches = dict(p["label_mismatches"])
        mismatches[parity] = label_mismatches(labels, axis, k)
        _perm, causes = design_map(labels, axis, k)
        count = sum(causes.values())
        planes.append({
            "axis": p["axis"], "at_mm": p["at_mm"], "align": p["align"], "parity": parity, "k": k,
            "plane_index": k // 2 if parity == "face" else (k - 1) // 2,
            "label_mismatches": mismatches,
            "orphans": {"count": count, "share": count / n_design if n_design else 0.0, **causes},
            "grid_range_mm": [float(origin[axis]), float(origin[axis]) + labels.shape[axis] * h]})
        if p.get("other_parity"):   # a given `align`: the counts of the other parity (a fact; `resolve`)
            planes[-1]["other_parity"] = dict(p["other_parity"])
    ratios = [load_asymmetry(forces, labels.shape, AXES.index(p["axis"]), p["k"]) for p in planes]
    return {"planes": planes, "design_voxels": n_design,
            "load_asymmetry": [{"case": name, "planes": [r[i] for r in ratios]}
                               for i, name in enumerate(case_names)]}



# --- text for the user ------------------------------------------------------------------


def plane_text(p) -> str:
    """`y = 0 mm` for a plane (a dict of facts)."""
    return f"{p['axis']} = {p['at_mm']:g} mm"


def _orphan_text(mirror, p) -> str:
    """The orphan line of one plane: the count, the causes, and (for most or all) the effect and the options."""
    o = p["orphans"]
    n = mirror["design_voxels"]
    text = (f"plane {plane_text(p)}: {o['count']} of {n} design voxels have no design "
            f"partner ({100 * o['share']:.2f} percent). They stay free variables")
    if o["count"]:
        other = f", other {o['other']}" if o.get("other") else ""
        text += (f" (image outside the grid {o['outside_grid']}, empty {o['empty']}, keep_out "
                 f"{o['keep_out']}, fixed {o['fixed']}, load {o['load']}{other})")
    options = ("Options: check at_mm, move the plane, make the regions symmetric about it, or remove the "
               "mirror key")
    lo, hi = p["grid_range_mm"]
    if n and o["count"] == n:
        text += (f". The mirror has no effect on this plane. The grid spans {p['axis']} = {lo:.6g} to {hi:.6g} mm. "
                 f"{options}")
    elif o["share"] >= ORPHAN_MOST:
        text += (f". Most design voxels have no partner, so the mirror has little effect on this plane. The grid "
                 f"spans {p['axis']} = {lo:.6g} to {hi:.6g} mm. {options}")
    return text


def _other_parity_text(p) -> str:
    """The line about a given `align` whose other parity has fewer label mismatches. Empty if it has not."""
    other = p.get("other_parity")
    mm = p["label_mismatches"].get(p["parity"])
    if not other or mm is None or other["label_mismatches"] >= mm:
        return ""

    def counts(label_mismatches, orphans):
        return f"{label_mismatches} label mismatches" + (f" and {orphans} orphans" if orphans is not None else "")

    used, fewer = p["parity"], other["parity"]
    return (f"plane {plane_text(p)}: parity {fewer} has fewer label mismatches than parity {used}, which align {p['align']} "
            f"sets ({fewer}: {counts(other['label_mismatches'], other.get('orphans'))}; "
            f"{used}: {counts(mm, p['orphans']['count'])}). The run keeps {used}. "
            f"Options: set align: auto, or set align: {fewer}")


def _load_verdict(mirror, p, ratio) -> str:
    """The words after the asymmetry value of one load case about one plane."""
    if ratio <= SYMMETRIC_TOL:
        return "symmetric"
    o = p["orphans"]
    n = mirror["design_voxels"]
    if n and o["count"] == n:
        text = ("not symmetric: the mirror has no effect on this plane (no design voxel has a partner), so the "
                "load is not mirrored and the design is not symmetric about it")
    elif o["count"]:
        text = (f"not symmetric: the design is symmetric on the paired voxels ({100 * (1 - o['share']):.1f} percent "
                f"of the design voxels), so the result is the best such design for this load")
    else:
        text = "not symmetric: the design is symmetric, so the result is the best symmetric design for this load"
    if o["load"]:
        text += (f". {o['load']} design voxels have a load voxel as their image (cause load): the load region is "
                 f"not symmetric about this plane, which can make a symmetric force read as not symmetric")
    return text


def summary_lines(mirror) -> list:
    """Text lines (no prefix) about the planes, the orphans and the load asymmetry. Empty without a mirror key."""
    if not mirror:
        return []
    lines = []
    for p in mirror["planes"]:
        where = ("on a voxel face" if p["parity"] == "face" else "through the centres of a voxel layer")
        mm = p["label_mismatches"]
        counts = ", ".join(f"{name} {mm[name]}" + (" (used)" if name == p["parity"] else "")
                           for name in ("face", "centre") if mm[name] is not None)
        lines.append(f"plane {plane_text(p)}: parity {p['parity']}, the plane goes {where} "
                     f"(align {p['align']}). Label mismatches under the mirror: {counts}")
        lines.append(_orphan_text(mirror, p))
        fewer = _other_parity_text(p) if p["align"] != "auto" else ""   # a given align whose other parity fits better
        if fewer:
            lines.append(fewer)
    for entry in mirror["load_asymmetry"]:
        for p, ratio in zip(mirror["planes"], entry["planes"]):
            lines.append(f"load case '{entry['case']}', plane {plane_text(p)}: |f - R f| / |f| = {ratio:.3g} "
                         f"(0 is symmetric, 2 is antisymmetric): {_load_verdict(mirror, p, ratio)}")
    return lines


def end_lines(mirror) -> list:
    """Text lines about the mirror deviation at the end of a run (empty if none is recorded)."""
    end = (mirror or {}).get("end")
    if not end:
        return []
    lines = []
    for p, dx, dr in zip(mirror["planes"], end["x_deviation"], end["rho_deviation"]):
        note = "x is the averaged design that the filter reads"
        if mirror["design_voxels"] and p["orphans"]["count"] == mirror["design_voxels"]:
            note += "; no design voxel has a partner, so these values say nothing about symmetry"
        lines.append(f"plane {plane_text(p)}: end of run max|x - x[perm]| = {dx:.3g}, max|rho - rho[perm]| = {dr:.3g} "
                     f"({note})")
    return lines
