"""Pieces of the active voxels that no support holds (CONTRACTS.md section 3, "Pieces without a support").

A piece is a set of active voxels that share faces (6 neighbours). A voxel that touches another one only at an edge
or at a corner is in another piece: the voxel model joins the two at the shared nodes only, and the joint is a hinge.
A piece holds a support when it has a FIXED voxel, or, with springs, a voxel of a spring region, or, with joints
(from A7b), a voxel of a joint region (`springs.support_mask` is the one rule for this). A piece with no support can
move as a rigid body, so the stiffness matrix is singular and a direct solver can stop on a zero pivot with a message
that does not name the voxels, and a solver that goes on (it perturbs the pivot) gives the piece a displacement that
is not defined.

`unsupported_pieces` gives the facts, `message` gives the text of the `UserError` that `topopt run` raises before the
first solve, and `topopt check` prints the same text as a warning. The mirror key does not change the result: the
planes tie the design variables of partner voxels, and they add no support to a piece.

`support_touch` and `touch_message` are the one rule for "a support does not touch the design space" (a FIXED voxel, or
a voxel of a spring or joint region, that shares a face with a DESIGN voxel). `topopt check` and the part builder
(`regions_builder.check_yaml`) both call them and both give a warning. It is a fact and never an error: a support that
reaches the design only through load or solid voxels holds the part, and the run works. A run cannot work only when a
piece has no support, and `unsupported_pieces` covers that case. The two tools use it the same way: `message` (with
`RUN_STOPS`) is an ERROR in the builder and a warning in `topopt check` when a piece has no support. When every piece holds a
support, `held_pieces_message` is a warning in both tools, and the run works.

The module loads numpy and scipy only inside the functions: `topopt.cli` must not load them at import.
"""

from __future__ import annotations

SHOW_PIECES = 5    # pieces that the facts describe in detail, and that the message names (the counts cover all)
SHOW_VOXELS = 3    # voxels named for each piece: the first ones in the order (i, j, k)


def unsupported_pieces(problem) -> dict:
    """Facts about the pieces of active voxels of `problem` and the pieces that hold no support.

    Keys: `pieces` (all pieces), `unsupported_pieces`, `unsupported_voxels` (in all of them), `springs` (the problem
    has spring regions: they count as supports), `joints` (True only when the problem has joints, from A7b: joint regions
    count as supports too; the key is absent otherwise), and `listed`: one dict for each unsupported piece, the largest first,
    at most `SHOW_PIECES`. A listed piece has `voxels`, `labels` (counts of `design`, `fixed`, `load`, `solid`;
    `solid` is a forced-solid voxel with no load: a solid region or an interface shape), `box_mm` (`[lo, hi]`, the
    box of the piece) and `first` (at most `SHOW_VOXELS` voxels: `ijk`, `mm` (the centre) and `label`).
    """
    import numpy as np
    from scipy import ndimage

    from .springs import support_mask
    from .voxelize import DESIGN, FIXED, LOAD

    labels = np.asarray(problem.labels)
    active = (labels >= DESIGN) & (labels <= LOAD)
    comp, n = ndimage.label(active)            # the default structure has 6 neighbours (faces)
    facts = {"pieces": int(n), "unsupported_pieces": 0, "unsupported_voxels": 0,
             "springs": problem.springs is not None, "listed": []}
    if getattr(problem, "joints", None) is not None:        # `Problem.joints` comes with A7b: main has no such attribute
        facts["joints"] = True
    if n == 0:
        return facts
    held = np.zeros(n + 1, dtype=bool)
    held[np.unique(comp[support_mask(problem)])] = True
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    loose = np.nonzero(~held[1:])[0] + 1                     # the ids of the pieces with no support
    if not loose.size:
        return facts
    loose = loose[np.lexsort((loose, -sizes[loose]))]        # the largest first, then by id
    facts["unsupported_pieces"] = int(loose.size)
    facts["unsupported_voxels"] = int(sizes[loose].sum())
    h = float(problem.h)
    origin = np.asarray(problem.origin, dtype=np.float64)
    slices = ndimage.find_objects(comp)
    names = {int(DESIGN): "design", int(FIXED): "fixed"}
    for i in (int(v) for v in loose[:SHOW_PIECES]):
        sl = slices[i - 1]
        ijk = np.argwhere(comp[sl] == i) + np.array([s.start for s in sl])   # C order: (i, j, k) rising
        lab = labels[tuple(ijk.T)]
        ids = np.asarray(problem.load_region_ids)[tuple(ijk.T)]
        counts = {"design": int((lab == DESIGN).sum()), "fixed": int((lab == FIXED).sum()),
                  "load": int(((lab == LOAD) & (ids >= 0)).sum()), "solid": int(((lab == LOAD) & (ids < 0)).sum())}
        first = []
        for p, lb, rid in zip(ijk[:SHOW_VOXELS], lab[:SHOW_VOXELS], ids[:SHOW_VOXELS]):
            name = names.get(int(lb)) or ("load" if rid >= 0 else "solid")
            first.append({"ijk": [int(v) for v in p], "mm": [round(float(v), 3) for v in origin + h * (p + 0.5)],
                          "label": name})
        facts["listed"].append({
            "voxels": int(sizes[i]), "labels": counts,
            "box_mm": [[round(float(v), 3) for v in origin + h * ijk.min(axis=0)],
                       [round(float(v), 3) for v in origin + h * (ijk.max(axis=0) + 1)]],
            "first": first})
    return facts


def _list(items: list, last: str) -> str:
    """`a`, `a and b` (`last` "and"), `a, b and c`: the words of one clause."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {last} {items[-1]}"


def message(facts: dict) -> str | None:
    """The `UserError` text for `unsupported_pieces` facts: the count, a few voxel positions and the options. None if every piece holds a support."""
    k = facts["unsupported_pieces"]
    if not k:
        return None
    holders = [name for name, key in (("spring", "springs"), ("joint", "joints")) if facts.get(key)]   # what else holds a piece
    support = _list(["no fixed voxel"] + [f"no {name} region" for name in holders], "and")
    parts = []
    for n, piece in enumerate(facts["listed"], 1):
        lb = ", ".join(f"{c} {name}" for name, c in piece["labels"].items() if c)
        lo, hi = piece["box_mm"]
        box = ", ".join(f"{a} {lo[d]:g} to {hi[d]:g}" for d, a in enumerate("xyz"))
        first = "; ".join("({}, {}, {}) at ({:g}, {:g}, {:g}) mm {}".format(*v["ijk"], *v["mm"], v["label"])
                          for v in piece["first"])
        shown = "first voxels" if piece["voxels"] > len(piece["first"]) else "voxels"
        parts.append(f"piece {n}: {piece['voxels']} voxels ({lb}), box {box} mm, {shown} (i, j, k): {first}")
    more = k - len(facts["listed"])
    return (f"the active voxels make {facts['pieces']} separate pieces (voxels that share a face are one piece). {k} of them "
            f"{'has' if k == 1 else 'have'} {support}, {facts['unsupported_voxels']} voxels in all. "
            + " | ".join(parts) + (f" | and {more} more" if more > 0 else "")
            + ". A piece with no support can move as a rigid body, so the stiffness matrix is singular: a direct solver can "
            "stop on a zero pivot, and a solver that goes on gives the piece a displacement that is not defined. "
            "A voxel that touches the rest only at an edge or a corner is a separate piece. "
            "Options: move or enlarge a region so that the piece shares a face with a part that is held, "
            "add " + _list(["a fixed region"] + [f"a {name}" for name in holders], "or") + " on the piece, "
            "take the piece out of the regions, or change voxel_size (the voxel centres decide which voxels a region gets)")


def support_touch(problem) -> dict:
    """Facts: does a support voxel share a face (6 neighbours) with a DESIGN voxel?

    Keys: `touches` (bool) and `held_by` (the supports other than FIXED that the problem has: "spring", "joint").
    When no support touches the design, also: `support_voxels`, `design_voxels`, `nearest_voxels` (the distance from a
    design voxel to the nearest support voxel, centre to centre, in voxels: 1 is a shared face), `nearest_mm` and
    `design_held` (True when every design voxel is in a piece that holds a support: the support reaches the design through
    load or solid voxels). The distance and the pieces are computed only in this case.
    """
    import numpy as np
    from scipy import ndimage

    from .springs import support_mask
    from .voxelize import DESIGN, LOAD

    labels = np.asarray(problem.labels)
    design = labels == DESIGN
    support = support_mask(problem)
    held_by = [w for w, p in (("spring", problem.springs), ("joint", getattr(problem, "joints", None))) if p is not None]
    facts = {"touches": bool((ndimage.binary_dilation(support) & design).any()), "held_by": held_by}
    if facts["touches"] or not design.any():   # (`voxelize` refuses a problem with no design voxel: no distance to give)
        return facts
    dist = ndimage.distance_transform_edt(~support)         # in voxels, to the nearest support voxel
    nearest = float(dist[design].min())
    comp, _n = ndimage.label((labels >= DESIGN) & (labels <= LOAD))
    in_support = np.isin(comp[design], np.unique(comp[support]))
    facts.update({"support_voxels": int(support.sum()), "design_voxels": int(design.sum()),
                  "nearest_voxels": round(nearest, 2), "nearest_mm": round(nearest * float(problem.h), 3),
                  "design_held": bool(in_support.all())})
    return facts


def touch_message(facts: dict) -> str | None:
    """The warning text for `support_touch` facts: the fact, what it means for the run, and the options. None when a support touches the design."""
    if facts["touches"] or "nearest_voxels" not in facts:
        return None
    what = "no fixed voxel" + "".join(f" and no {w} region" for w in facts["held_by"])
    holder = " or ".join(["fixed"] + [f"{w} region" for w in facts["held_by"]])
    head = (f"{what} touches a design voxel: the nearest design voxel is {facts['nearest_voxels']:g} voxels "
            f"({facts['nearest_mm']:g} mm) from a {holder} voxel, centre to centre (1 voxel is a shared face). ")
    if facts["design_held"]:
        meaning = ("Every design voxel is in a piece that holds a support, so the design is held through load or solid voxels. "
                   "This does not stop `topopt run`; the load path goes through those voxels, which stay solid. ")
    else:
        meaning = ("Some design voxels are in a piece with no support (see the message about the pieces), so the design is "
                   "not held there. ")
    return (head + meaning + "Options: overlap the design space with the support region by one voxel or more, move one of "
            "them so that they share a face, or keep it if the path through the solid voxels is what you want")


RUN_STOPS = " (`topopt run` stops with this error)"    # the end of the `topopt check` warning of this fact


def held_pieces_message(sizes: list, held_by: list) -> str:
    """The warning text for a problem with more than one piece where every piece holds a support (the run works).

    `sizes` are the voxel counts of the pieces, the largest first; `held_by` are the supports other than FIXED ("spring", "joint").
    `topopt check` and the part builder print this same text.
    """
    return (f"the active voxels make {len(sizes)} separate pieces (largest sizes {sizes[:5]}); "
            "each holds a fixed voxel" + "".join(f" or a {w} region" for w in held_by)
            + ", so a piece cannot move as a rigid body, and a load on one piece does not reach the others")


def floating_error(problem) -> str | None:
    """The `UserError` text of `topopt run` for `problem`, or None when every piece holds a support."""
    return message(unsupported_pieces(problem))
