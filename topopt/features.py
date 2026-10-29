"""Holes, bosses and flat faces of an STL (CONTRACTS.md section 45). Read only: nothing is written.

`find_features(stl)` reads a triangle mesh and lists what a person sees in the part: the cylindrical holes (axis, radius, length, through or
blind), the cylindrical bosses and shafts, and the flat faces (outward normal, area). The reply is the input of a later step, for example
the `bore` and `tube` interfaces of section 23: the keys of a hole (`point`, `axis`, `radius_mm`, `length_mm`) are the keys of a `bore`.

Method (every number is a named constant below, and a keyword argument of `find_features`):

1. Read the STL as the file holds it (float32 vertices; equal vertices are one vertex). No repair. Find the neighbour faces (two faces
   that share an edge), and the angle between their normals.
2. A smooth region is a set of faces that neighbour angles of at most `smooth_angle_deg` join. A flat region is the same with
   `flat_angle_deg`, and each face of it lies within `flat_angle_deg` of the region normal.
3. A smooth region is a cylinder when (a) the normals of its faces are all within `axis_normal_deg` of the plane across one axis (the axis
   is the direction with the smallest normal scatter, area weighted), and (b) the vertices projected on the plane across the axis lie on
   one circle: the RMS distance to the circle fit is at most `fit_residual_frac` of the radius. The radius is the radius of the wall
   vertices. A tessellated wall lies up to `chord_sag_mm` inside it (the entry gives it).
4. A cylinder is listed when its faces cover at least `arc_min_deg` of the circle (the union of the angular spans of the faces). Less is an
   arc (a fillet, the end of a slot): it is counted in the facts, not listed. The entry says if the ring is closed.
5. Hole or boss: the sum of area times (face normal . direction away from the axis). Negative: the normals point to the axis, the
   material is outside the cylinder, so it is a hole. Positive: a boss or a shaft. The sign of the normals is made outward by the sign of the
   volume of the mesh.
6. Through or blind: at each end of the wall, the faces across the rim edges vote. Their normal along the direction that leaves the hole
   decides: positive, the surface leads away from the hole (the end is `open`); negative, the surface closes the end (`closed`, a bottom, a
   cone, a step). Both ends open: `through`. One closed end: `blind`. Both closed: `closed` (a cavity). No vote: `unknown`. This looks at
   the rim only. A stepped hole (a counterbore) is two cylinders on one axis: the big one reports `blind` (its step end is closed), the small
   one reports `through`, and both name each other in `coaxial_with`.
7. A flat face is a flat region that no cylinder owns, whose area is at least `flat_min_area_mm2`, and that is not a facet of a curved
   surface: at most `flat_soft_share_max` of its boundary length joins a neighbour by a soft edge (an angle above `flat_angle_deg` and at
   most `smooth_angle_deg`). A facet of a sphere is soft on all sides; a face of a box is hard on all sides.

The faces of a hole wall that join a fillet or a shallow chamfer (all neighbour angles within `smooth_angle_deg`) make one smooth region
that is not a cylinder: such a hole is not found. The facts count the smooth regions that are neither cylinder nor flat.

Order: holes by radius, then by the centre (x, y, z); bosses the same; flat faces by area (the largest first), then by the centroid. Ids
(`H1`, `B1`, `F1`) follow the order. Lengths are mm and the vectors are unit vectors (rounded to 6 digits). Time is about 5 s for a million
faces; there is no limit on the faces. Standard library, numpy, scipy and trimesh only.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .config import UserError

# --- the numbers of the method (CONTRACTS.md section 45). The reply `method.numbers` gives the value that was used. ----------------------------
SMOOTH_ANGLE_DEG = 30.0       # two neighbour faces are one smooth region when their normals differ by at most this. A cylinder of n segments has the angle
                              # 360/n between wall facets: 30 deg finds n >= 12. Edges of 90 deg (wall to plate) always split. A fillet or a chamfer
                              # under this angle joins the wall to the plate and the hole is not found.
FLAT_ANGLE_DEG = 1.0          # a flat region: neighbour faces within this angle, and every face within this angle of the region normal. File rounding
                              # (float32) gives 1e-3 deg or less on a face of normal size; a draft angle of 1 deg or more is not flat.
FIT_RESIDUAL_FRAC = 0.01      # a cylinder: the RMS distance of the wall vertices to the circle, as a share of the radius. A tessellated wall has its
                              # vertices on the circle (distance ~0); a cone, a sphere patch or free form gives more.
AXIS_NORMAL_DEG = 2.0         # a cylinder: every face normal within this angle of the plane across the axis (a cone or a draft wall of more is not a cylinder).
ARC_MIN_DEG = 180.0           # a cylinder is listed when its faces cover at least this angle of the circle. A closed ring covers 360 deg.
FLAT_MIN_AREA_MM2 = 1.0       # a flat region under this area is counted in the facts, not listed (a mesh of free form has many small flat pieces).
FLAT_SOFT_SHARE_MAX = 0.5     # a flat region is a face of the part when at most this share of its boundary length is soft (a facet of a curved surface has all soft).
NEAR_LIMIT_SHARE = 0.5        # a fact only: a value above this share of its limit is "near the limit". It does not change what is found.
ANGLE_EPS_RAD = 1e-6          # float rounding of an angle: a gap of the angular cover under this is no gap, and an angle that is this much above a limit still passes
                              # (a ring of 12 segments has exactly 30 deg between facets). Not a user number.
ZERO_AREA_MM2 = 1e-12         # a face with a smaller area has no normal (the same number as the print check). It is left out.

NUMBERS = {
    "smooth_angle_deg": SMOOTH_ANGLE_DEG, "flat_angle_deg": FLAT_ANGLE_DEG, "fit_residual_frac": FIT_RESIDUAL_FRAC,
    "axis_normal_deg": AXIS_NORMAL_DEG, "arc_min_deg": ARC_MIN_DEG, "flat_min_area_mm2": FLAT_MIN_AREA_MM2,
    "flat_soft_share_max": FLAT_SOFT_SHARE_MAX, "near_limit_share": NEAR_LIMIT_SHARE, "angle_eps_rad": ANGLE_EPS_RAD, "zero_area_mm2": ZERO_AREA_MM2,
}
OPTION_NAMES = ("smooth_angle_deg", "flat_angle_deg", "fit_residual_frac", "axis_normal_deg", "arc_min_deg", "flat_min_area_mm2", "flat_soft_share_max")   # the keyword arguments

MM_DIGITS = 4        # lengths: 0.1 micrometre (the float32 file resolution is 0.01 micrometre for a part of 1 m: finer digits would be noise)
UNIT_DIGITS = 6      # unit vectors
AREA_DIGITS = 3
ANGLE_DIGITS = 3


def _fixed(value, digits: int) -> float:
    return float(np.round(value, digits)) + 0.0          # + 0.0: -0.0 becomes 0.0


def _vec(v, digits: int) -> list[float]:
    return [_fixed(x, digits) for x in np.asarray(v, float)]


def _check_option(name: str, value, low: float, high: float | None = None, low_open: bool = True) -> float:
    ok = (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
          and (value > low if low_open else value >= low) and (high is None or value <= high))
    if not ok:
        bound = ("above " if low_open else "at least ") + f"{low:g}" + (f" and at most {high:g}" if high is not None else "")
        raise UserError(f"find_features: {name} takes a number {bound}, got {value!r}")
    return float(value)


# --- reading the mesh ------------------------------------------------------------------------------------------------------------------


def _read(stl):
    """`(vertices, faces, name)` of a path or a trimesh mesh. A path that is not there, or not an STL, raises `UserError`."""
    if hasattr(stl, "vertices") and hasattr(stl, "faces"):
        joined = stl.copy()
        joined.merge_vertices()                                  # a mesh from a caller may hold the same vertex twice: neighbour faces need one vertex
        return np.asarray(joined.vertices, float), np.asarray(joined.faces, np.int64), "<mesh>"
    path = Path(stl).expanduser()
    if path.is_dir():
        raise UserError(f"{path} is a folder. Give an STL file")
    if not path.is_file():
        raise UserError(f"file not found: {path}")
    from .printcheck import load_stl_float32
    V, F = load_stl_float32(path)      # raises UserError for a file that is not an STL (it checks the bytes first)
    return V, F, str(path)


def _perp_basis(axis: np.ndarray):
    """Two unit vectors across `axis` and across each other."""
    e = np.zeros(3)
    e[int(np.argmin(np.abs(axis)))] = 1.0
    e1 = np.cross(axis, e)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def _components(m: int, a: np.ndarray, b: np.ndarray, keep: np.ndarray):
    """Connected faces. `a`, `b`: the face pairs that share an edge; `keep`: which pairs join. Returns `(count, label of each face)`."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(int(keep.sum()), np.int8), (a[keep], b[keep])), shape=(m, m))
    return connected_components(graph, directed=False)


def _angular_cover(theta_c: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Total angle in rad (0 to 2 pi) that the intervals `[theta_c + lo, theta_c + hi]` cover on the circle."""
    start = np.mod(theta_c + lo, 2.0 * math.pi)
    end = start + (hi - lo)
    order = np.argsort(start)
    start, end = start[order], end[order]
    wrap = end.max() - 2.0 * math.pi                            # the part of an interval that passes 2 pi covers the start of the circle
    reach = np.maximum(np.maximum.accumulate(end), wrap)
    nxt = np.r_[start[1:], start[0] + 2.0 * math.pi]
    gap = nxt - reach
    return float(2.0 * math.pi - gap[gap > ANGLE_EPS_RAD].sum())


# --- one cylinder candidate -----------------------------------------------------------------------------------------------------------


def _fit_cylinder(Vc, F, normal, area, centroid, faces, axis, opts):
    """Test the smooth region `faces` as a cylinder about `axis`. Returns `(fit, reason)`: a dict of the facts, or None and the reason."""
    n = normal[faces]
    sin_lim = math.sin(math.radians(opts["axis_normal_deg"]))
    dev = np.abs(n @ axis)
    if dev.max() > sin_lim:
        return None, "normal"
    vids = np.unique(F[faces])
    P = Vc[vids]
    e1, e2 = _perp_basis(axis)
    x, y, s = P @ e1, P @ e2, P @ axis
    # Kasa circle fit: x^2 + y^2 = 2 cx x + 2 cy y + (R^2 - cx^2 - cy^2). The points are centred first.
    x0, y0 = x.mean(), y.mean()
    xs, ys = x - x0, y - y0
    A = np.stack([2.0 * xs, 2.0 * ys, np.ones_like(xs)], axis=1)
    sol = np.linalg.lstsq(A, xs ** 2 + ys ** 2, rcond=None)[0]
    cx, cy = sol[0], sol[1]
    radius2 = sol[2] + cx ** 2 + cy ** 2
    if not (math.isfinite(radius2) and radius2 > 0.0):
        return None, "residual"
    R = math.sqrt(radius2)
    dist = np.hypot(xs - cx, ys - cy)
    res = dist - R
    rms = float(np.sqrt(np.mean(res ** 2)))
    if rms > opts["fit_residual_frac"] * R:
        return None, "residual"
    cx, cy = cx + x0, cy + y0                                   # the centre in the frame of e1, e2
    # the angular cover of the faces about the centre
    tri = Vc[F[faces]]                                          # (k, 3, 3)
    tx, ty = tri @ e1 - cx, tri @ e2 - cy
    theta = np.arctan2(ty, tx)                                  # (k, 3)
    cen = centroid[faces]
    theta_c = np.arctan2(cen @ e2 - cy, cen @ e1 - cx)
    d = np.angle(np.exp(1j * (theta - theta_c[:, None])))       # each vertex angle relative to the face centroid, in (-pi, pi]
    lo, hi = d.min(axis=1), d.max(axis=1)
    cover = _angular_cover(theta_c, lo, hi)
    width = hi - lo
    n_hat = np.stack([n @ e1, n @ e2], axis=1)
    rad = np.stack([np.cos(theta_c), np.sin(theta_c)], axis=1)
    side = float((area[faces] * np.einsum("ij,ij->i", n_hat, rad)).sum())     # negative: the normals point to the axis (a hole)
    return dict(
        axis=axis, e1=e1, e2=e2, cx=cx, cy=cy, radius=R, rms=rms, max_dev=float(np.abs(res).max()), normal_dev=math.degrees(math.asin(min(1.0, float(dev.max())))),
        s_lo=float(s.min()), s_hi=float(s.max()), cover=cover, closed_ring=bool(2.0 * math.pi - cover <= ANGLE_EPS_RAD),
        segments=int(round(2.0 * math.pi / float(np.median(width)))) if np.median(width) > 0 else 0,
        hole=side < 0.0, area=float(area[faces].sum()), faces=int(len(faces)),
    ), None


def _end_states(fit, inside, across, edge_a, edge_b, Vc, normal):
    """The state of the two ends of a wall: `open`, `closed` or `unknown`, for the end at `s_lo` and the end at `s_hi` (module docstring, step 6)."""
    if len(inside) == 0:
        return ["unknown", "unknown"]
    axis = fit["axis"]
    mid = 0.5 * (Vc[edge_a] + Vc[edge_b])
    length = np.linalg.norm(Vc[edge_b] - Vc[edge_a], axis=1)
    s = mid @ axis
    high = s > 0.5 * (fit["s_lo"] + fit["s_hi"])
    lead = normal[across] @ axis                                # the normal of the face across the rim along the axis
    states = []
    for end_high in (False, True):
        pick = high == end_high
        vote = float((length[pick] * (lead[pick] if end_high else -lead[pick])).sum()) if pick.any() else 0.0
        states.append("open" if vote > 0.0 else "closed" if vote < 0.0 else "unknown")
    return states


# --- the function ---------------------------------------------------------------------------------------------------------------------


def find_features(stl, *, smooth_angle_deg: float = SMOOTH_ANGLE_DEG, flat_angle_deg: float = FLAT_ANGLE_DEG,
                  fit_residual_frac: float = FIT_RESIDUAL_FRAC, axis_normal_deg: float = AXIS_NORMAL_DEG,
                  arc_min_deg: float = ARC_MIN_DEG, flat_min_area_mm2: float = FLAT_MIN_AREA_MM2,
                  flat_soft_share_max: float = FLAT_SOFT_SHARE_MAX) -> dict:
    """The holes, bosses and flat faces of an STL (a path or a trimesh mesh), in mm. Read only. See the module docstring for the method.

    Returns a JSON-safe dict with `mesh`, `holes`, `bosses`, `flat_faces`, `method` and `facts` (CONTRACTS.md section 45).
    Raises `UserError` for a file that is not there or not an STL, and for a number that is not valid.
    """
    t0 = time.perf_counter()
    opts = dict(
        smooth_angle_deg=_check_option("smooth_angle_deg", smooth_angle_deg, 0.0, 89.0),
        flat_angle_deg=_check_option("flat_angle_deg", flat_angle_deg, 0.0, 45.0),
        fit_residual_frac=_check_option("fit_residual_frac", fit_residual_frac, 0.0, 1.0),
        axis_normal_deg=_check_option("axis_normal_deg", axis_normal_deg, 0.0, 45.0),
        arc_min_deg=_check_option("arc_min_deg", arc_min_deg, 0.0, 360.0),
        flat_min_area_mm2=_check_option("flat_min_area_mm2", flat_min_area_mm2, 0.0, low_open=False),
        flat_soft_share_max=_check_option("flat_soft_share_max", flat_soft_share_max, 0.0, 1.0, low_open=False),
    )
    V, F, name = _read(stl)
    m = int(len(F))
    if m == 0 or len(V) == 0:
        raise UserError(f"the mesh {name} has no faces")
    if not np.isfinite(V).all():
        raise UserError(f"the mesh {name} has non-finite coordinates (NaN or infinity): the file is damaged")
    import trimesh
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    origin = 0.5 * (V.min(axis=0) + V.max(axis=0))
    Vc = V - origin                                              # work near the origin: no rounding loss for a part far from it
    tri = Vc[F]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice = np.linalg.norm(cross, axis=1)
    area = 0.5 * twice
    live = area > ZERO_AREA_MM2
    normal = np.zeros_like(cross)
    normal[live] = cross[live] / twice[live, None]
    centroid = tri.mean(axis=1)
    volume = float(np.einsum("ij,ij->", tri[:, 0], cross) / 6.0)
    flipped = volume < 0.0
    if flipped:
        normal = -normal                                         # the file has inward normals: make them outward (the volume has the sign)
    del tri, cross, twice

    watertight = bool(mesh.is_watertight)
    winding = bool(mesh.is_winding_consistent)
    adj = np.asarray(mesh.face_adjacency, np.int64).reshape(-1, 2)
    adj_edges = np.asarray(mesh.face_adjacency_edges, np.int64).reshape(-1, 2)
    both = live[adj[:, 0]] & live[adj[:, 1]]
    dot = np.einsum("ij,ij->i", normal[adj[:, 0]], normal[adj[:, 1]])
    smooth_keep = both & (dot >= math.cos(math.radians(opts["smooth_angle_deg"]) + ANGLE_EPS_RAD))
    flat_keep = both & (dot >= math.cos(math.radians(opts["flat_angle_deg"]) + ANGLE_EPS_RAD))

    # --- smooth regions and the cylinder candidates
    n_smooth, smooth = _components(m, adj[:, 0], adj[:, 1], smooth_keep)
    region_area = np.bincount(smooth, weights=area, minlength=n_smooth)
    scatter = np.empty((n_smooth, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            scatter[:, i, j] = scatter[:, j, i] = np.bincount(smooth, weights=area * normal[:, i] * normal[:, j], minlength=n_smooth)
    lam, vec = np.linalg.eigh(scatter)
    sin2 = math.sin(math.radians(opts["axis_normal_deg"])) ** 2
    c = math.radians(opts["arc_min_deg"])
    spread = 0.25 * (0.5 - math.sin(c) / (2.0 * c))              # a speed filter only: the middle eigenvalue share of normals that cover `arc_min_deg`
    cand = np.nonzero((region_area > 0.0) & (lam[:, 0] <= sin2 * region_area) & (lam[:, 1] >= spread * region_area))[0]
    order = np.argsort(smooth, kind="stable")
    first = np.searchsorted(smooth[order], np.arange(n_smooth + 1))

    rejected = {"normal": 0, "residual": 0}
    arcs = 0
    found: list[tuple[int, dict]] = []
    for cid in cand:
        faces = order[first[cid]:first[cid + 1]]
        faces = faces[live[faces]]
        fit, why = _fit_cylinder(Vc, F, normal, area, centroid, faces, vec[cid, :, 0], opts)
        if fit is None:
            rejected[why] += 1
        elif math.degrees(fit["cover"]) < opts["arc_min_deg"] - 1e-9:
            arcs += 1
            found.append((int(cid), dict(fit, listed=False)))
        else:
            found.append((int(cid), dict(fit, listed=True)))
    cylinder_region = np.zeros(n_smooth, bool)
    for cid, fit in found:
        cylinder_region[cid] = True

    # --- through or blind: the faces across the rim of each listed wall
    holes_raw, bosses_raw = [], []
    pair_in = pair_out = pair_ea = pair_eb = np.zeros(0, np.int64)
    if any(fit["listed"] and fit["hole"] for _, fit in found):
        cross_pair = smooth[adj[:, 0]] != smooth[adj[:, 1]]
        ia = np.r_[adj[cross_pair, 0], adj[cross_pair, 1]]
        ib = np.r_[adj[cross_pair, 1], adj[cross_pair, 0]]
        ea = np.r_[adj_edges[cross_pair, 0], adj_edges[cross_pair, 0]]
        eb = np.r_[adj_edges[cross_pair, 1], adj_edges[cross_pair, 1]]
        srt = np.argsort(smooth[ia], kind="stable")
        pair_in, pair_out, pair_ea, pair_eb = ia[srt], ib[srt], ea[srt], eb[srt]
        pair_first = np.searchsorted(smooth[pair_in], np.arange(n_smooth + 1))
    for cid, fit in found:
        if not fit["listed"]:
            continue
        if fit["hole"]:
            lo, hi = pair_first[cid], pair_first[cid + 1]
            fit["ends"] = _end_states(fit, pair_in[lo:hi], pair_out[lo:hi], pair_ea[lo:hi], pair_eb[lo:hi], Vc, normal)
            holes_raw.append(fit)
        else:
            bosses_raw.append(fit)

    def entry(fit, kind):
        """The entry of a hole or a boss. The axis is oriented here (the sign of the largest component positive; a blind hole points from its
        open end to its closed end) and `point` is the centre of the first end face."""
        axis, s_lo, s_hi = fit["axis"], fit["s_lo"], fit["s_hi"]
        ends = list(fit.get("ends", []))                         # [state at s_lo, state at s_hi]
        flip = bool(axis[int(np.argmax(np.abs(axis)))] < 0.0)
        if kind == "hole" and sorted(ends) == ["closed", "open"]:
            flip = ends[0] == "closed"                            # the axis runs from s_lo to s_hi: flip when the open end is s_hi
        a = -axis if flip else axis
        lo, hi = (-s_hi, -s_lo) if flip else (s_lo, s_hi)
        if flip:
            ends = ends[::-1]
        foot = fit["cx"] * fit["e1"] + fit["cy"] * fit["e2"]       # the point of the axis in the plane s = 0 (the frame of the work near the origin)
        p0 = foot + a * lo + origin
        p1 = foot + a * hi + origin
        out = {"axis": _vec(a, UNIT_DIGITS), "point": _vec(p0, MM_DIGITS), "end_point": _vec(p1, MM_DIGITS),
               "radius_mm": _fixed(fit["radius"], MM_DIGITS), "diameter_mm": _fixed(2.0 * fit["radius"], MM_DIGITS),
               "length_mm": _fixed(hi - lo, MM_DIGITS)}
        if kind == "hole":
            out["depth"] = ("through" if ends == ["open", "open"] else "blind" if sorted(ends) == ["closed", "open"]
                            else "closed" if ends == ["closed", "closed"] else "unknown")
            out["ends"] = ends
        out.update(arc_deg=_fixed(math.degrees(fit["cover"]), ANGLE_DIGITS), closed_ring=fit["closed_ring"], segments=fit["segments"],
                   chord_sag_mm=_fixed(fit["radius"] * (1.0 - math.cos(math.pi / fit["segments"])), 6) if fit["segments"] >= 3 else None,
                   area_mm2=_fixed(fit["area"], AREA_DIGITS), faces=fit["faces"],
                   fit={"rms_mm": _fixed(fit["rms"], 6), "max_mm": _fixed(fit["max_dev"], 6), "normal_deviation_deg": _fixed(fit["normal_dev"], ANGLE_DIGITS)})
        out["_centre"] = 0.5 * (p0 + p1)
        return out

    def ordered(entries, prefix):
        entries.sort(key=lambda e: (round(e["radius_mm"], 3), *(round(float(v), 3) for v in e["_centre"])))
        for i, e in enumerate(entries, 1):
            e["id"] = f"{prefix}{i}"
            e.pop("_centre")
        return [{"id": e.pop("id"), **e} for e in entries]

    holes = ordered([entry(f, "hole") for f in holes_raw], "H")
    bosses = ordered([entry(f, "boss") for f in bosses_raw], "B")
    _name_coaxial(holes, opts)

    # --- flat regions
    n_flat, flat = _components(m, adj[:, 0], adj[:, 1], flat_keep)
    fl_area = np.bincount(flat, weights=area, minlength=n_flat)
    mean_n = np.stack([np.bincount(flat, weights=area * normal[:, k], minlength=n_flat) for k in range(3)], axis=1)
    norm = np.linalg.norm(mean_n, axis=1)
    mean_n[norm > 0] /= norm[norm > 0, None]
    cosdev = np.einsum("ij,ij->i", normal, mean_n[flat])
    worst = np.full(n_flat, 1.0)
    np.minimum.at(worst, flat[live], cosdev[live])
    owned = cylinder_region[smooth]                              # the faces of a cylinder, listed or not (a fillet, the end of a slot)
    owner = np.zeros(n_flat, bool)
    owner[flat[owned]] = True
    flat_ok = (fl_area > 0.0) & ~owner & (worst >= math.cos(math.radians(opts["flat_angle_deg"]) + ANGLE_EPS_RAD))
    drift = int(((fl_area > 0.0) & ~owner & ~flat_ok).sum())
    edge_len = np.linalg.norm(Vc[adj_edges[:, 1]] - Vc[adj_edges[:, 0]], axis=1)
    between = both & (flat[adj[:, 0]] != flat[adj[:, 1]])                  # an edge between two flat regions (or a flat region and a curved face)
    soft_len = np.zeros(n_flat)
    hard_len = np.zeros(n_flat)
    for side in (0, 1):
        soft_len += np.bincount(flat[adj[between & smooth_keep, side]], weights=edge_len[between & smooth_keep], minlength=n_flat)
        hard_len += np.bincount(flat[adj[between & ~smooth_keep, side]], weights=edge_len[between & ~smooth_keep], minlength=n_flat)
    soft_share = soft_len / np.maximum(soft_len + hard_len, 1e-300)
    facet = flat_ok & (soft_share > opts["flat_soft_share_max"])            # a facet of a curved surface
    flat_ok &= ~facet
    big = flat_ok & (fl_area >= opts["flat_min_area_mm2"])
    small = flat_ok & ~big
    ids = np.nonzero(big)[0]
    cen_sum = np.stack([np.bincount(flat, weights=area * centroid[:, k], minlength=n_flat) for k in range(3)], axis=1)
    cen = cen_sum / np.maximum(fl_area, 1e-300)[:, None] + origin
    count = np.bincount(flat, minlength=n_flat)
    rows = []
    for k in ids:
        rows.append((float(fl_area[k]), cen[k], mean_n[k], int(count[k]), math.degrees(math.acos(min(1.0, float(worst[k]))))))
    rows.sort(key=lambda r: (-round(r[0], 3), *(round(float(v), 3) for v in r[1])))
    flat_faces = [{"id": f"F{i}", "normal": _vec(r[2], UNIT_DIGITS), "area_mm2": _fixed(r[0], AREA_DIGITS), "centroid": _vec(r[1], MM_DIGITS),
                   "offset_mm": _fixed(float(r[2] @ r[1]), MM_DIGITS), "faces": r[3], "max_normal_deviation_deg": _fixed(r[4], ANGLE_DIGITS)}
                  for i, r in enumerate(rows, 1)]

    # --- the facts
    # a smooth region that is neither a cylinder nor one flat region (free form, a cone, a hole wall that joins a fillet)
    low = np.full(n_smooth, n_flat, np.int64)
    high = np.full(n_smooth, -1, np.int64)
    np.minimum.at(low, smooth, flat)
    np.maximum.at(high, smooth, flat)
    curved = (region_area > 0.0) & ~cylinder_region & (low != high)
    curved_faces = int(curved[smooth].sum())
    facts = _facts(opts, holes, bosses, watertight=watertight, winding=winding, flipped=flipped, mesh=mesh, live=live, arcs=arcs,
                   small=small, fl_area=fl_area, drift=drift, facets=int(facet.sum()), facet_area=float(fl_area[facet].sum()), curved=int(curved.sum()), curved_faces=curved_faces)
    bounds = np.stack([V.min(axis=0), V.max(axis=0)])
    reply = {
        "ok": True,
        "mesh": {
            "source": name, "faces": m, "vertices": int(len(V)), "unit": "mm (an STL has no unit; the values are mm if the file is in mm)",
            "watertight": watertight, "winding_consistent": winding, "normals_flipped": flipped,
            "zero_area_faces": int((~live).sum()), "bounds_mm": [_vec(bounds[0], MM_DIGITS), _vec(bounds[1], MM_DIGITS)],
            "size_mm": _vec(bounds[1] - bounds[0], MM_DIGITS),
            "volume_mm3": _fixed(abs(volume), AREA_DIGITS) if watertight else None,
        },
        "holes": holes,
        "bosses": bosses,
        "flat_faces": flat_faces,
        "method": {
            "numbers": {k: (opts[k] if k in opts else v) for k, v in NUMBERS.items()},
            "counts": {"smooth_regions": int(n_smooth), "cylinder_candidates": int(len(cand)), "cylinders": len(found), "rejected_normal": rejected["normal"],
                       "rejected_fit": rejected["residual"], "arcs_not_listed": arcs, "flat_regions_not_listed": int(small.sum()), "flat_regions_curved_facets": int(facet.sum())},
            "text": ("Smooth regions (neighbour normals within smooth_angle_deg) are tested as cylinders: normals across one axis (axis_normal_deg), "
                     "vertices on one circle (fit_residual_frac of the radius), at least arc_min_deg of the circle. Normals to the axis: a hole; "
                     "away from it: a boss. Through or blind from the faces across the rim of the wall. Flat faces: regions within flat_angle_deg, "
                     "no cylinder, at least flat_min_area_mm2. Radii are the radii of the wall vertices; a tessellated wall lies chord_sag_mm inside."),
        },
        "facts": facts,
        "timing_s": round(time.perf_counter() - t0, 3),
    }
    return reply


def _name_coaxial(holes: list[dict], opts: dict) -> None:
    """Add `coaxial_with` (the ids of the other holes on one axis) to each hole. Parallel within `axis_normal_deg`; the lines closer than
    `fit_residual_frac` of the smaller radius."""
    k = len(holes)
    for h in holes:
        h["coaxial_with"] = []
    if k < 2:
        return
    ax = np.array([h["axis"] for h in holes], float)
    pt = np.array([h["point"] for h in holes], float)
    rad = np.array([h["radius_mm"] for h in holes], float)
    cos_lim = math.cos(math.radians(opts["axis_normal_deg"]))
    step = 256
    for start in range(0, k, step):
        rows = slice(start, min(k, start + step))
        par = np.abs(ax[rows] @ ax.T) >= cos_lim                              # (r, k)
        d = pt[None, :, :] - pt[rows][:, None, :]                               # from the point of row i to the point of hole j
        along = np.einsum("ijk,ik->ij", d, ax[rows])
        off = np.linalg.norm(d - along[:, :, None] * ax[rows][:, None, :], axis=2)
        near = off <= opts["fit_residual_frac"] * np.minimum(rad[rows][:, None], rad[None, :])
        hit = par & near
        for r, i in enumerate(range(start, min(k, start + step))):
            hit[r, i] = False
            holes[i]["coaxial_with"] = [holes[j]["id"] for j in np.nonzero(hit[r])[0]]


def _facts(opts, holes, bosses, *, watertight, winding, flipped, mesh, live, arcs, small, fl_area, drift, facets, facet_area, curved, curved_faces) -> list[str]:
    """Plain sentences for what a reader should know."""
    out = []
    if not watertight:
        edges = mesh.edges_sorted
        from trimesh import grouping
        count = len(grouping.group_rows(edges, require_count=1))
        out.append(f"The mesh is not watertight ({count:,} edges belong to one face only). What can be found is listed; the repair of a mesh is not part of this "
                   "tool. A wall with an open edge can be missed, and an end that the mesh does not close can read `unknown`.")
    if not winding:
        out.append("The face winding is not consistent (neighbour faces run the same way round their shared edge). The sign of a normal can be wrong "
                   "on some faces, and a hole can read as a boss or the other way.")
    if flipped:
        out.append("The volume of the mesh is negative: the file has inward normals. The analysis turned them outward; the file is unchanged.")
    zero = int((~live).sum())
    if zero:
        out.append(f"{zero:,} faces have no area (under {ZERO_AREA_MM2:g} mm2) and are left out.")
    if holes or bosses:
        out.append("Radius: the radius of the wall vertices (a circle fit). A tessellated wall lies up to chord_sag_mm inside it. Each entry has `segments` and `chord_sag_mm`.")
    share = NEAR_LIMIT_SHARE
    for e in holes + bosses:
        near = []
        if e["fit"]["rms_mm"] > share * opts["fit_residual_frac"] * e["radius_mm"]:
            near.append(f"fit RMS {e['fit']['rms_mm']:g} mm is more than {share:g} of its limit {opts['fit_residual_frac'] * e['radius_mm']:g} mm")
        if e["fit"]["normal_deviation_deg"] > share * opts["axis_normal_deg"]:
            near.append(f"a face normal is {e['fit']['normal_deviation_deg']:g} deg off the plane across the axis (limit {opts['axis_normal_deg']:g})")
        if e["arc_deg"] < opts["arc_min_deg"] + (1.0 - share) * (360.0 - opts["arc_min_deg"]) and not e["closed_ring"]:
            near.append(f"it covers {e['arc_deg']:g} deg of the circle, not a closed ring (arc_min_deg is {opts['arc_min_deg']:g})")
        if near:
            out.append(f"{e['id']} is near a limit of the method: " + "; ".join(near) + ".")
    for e in holes:
        if e["depth"] == "unknown":
            out.append(f"{e['id']}: the faces across the rim do not tell if the ends are open or closed (the mesh is open there, or the rim has no faces).")
        if e["depth"] == "closed":
            out.append(f"{e['id']}: both ends are closed: a cavity inside the part, not a hole to the outside.")
        if e["coaxial_with"]:
            out.append(f"{e['id']} shares its axis with {', '.join(e['coaxial_with'])} (a stepped hole, or two holes in line): `depth` is for this cylinder alone.")
    if arcs:
        out.append(f"{arcs} cylindrical regions cover under arc_min_deg ({opts['arc_min_deg']:g} deg) of a circle (fillets, the ends of slots): not listed.")
    if small.sum():
        out.append(f"{int(small.sum())} flat regions with an area under flat_min_area_mm2 ({opts['flat_min_area_mm2']:g} mm2, {float(fl_area[small].sum()):.1f} mm2 in all) are not listed.")
    if facets:
        out.append(f"{facets:,} flat regions ({facet_area:.1f} mm2 in all) are facets of curved surfaces (more than flat_soft_share_max = "
                   f"{opts['flat_soft_share_max']:g} of their boundary joins the next face at an angle of at most smooth_angle_deg). Not listed.")
    if drift:
        out.append(f"{drift} flat-looking regions are not flat: their face normals turn by more than flat_angle_deg ({opts['flat_angle_deg']:g} deg) from the region normal "
                   "(a curve of small facets). Not listed.")
    if curved:
        out.append(f"{curved:,} smooth regions ({curved_faces:,} faces) are neither a cylinder nor flat (free form, cones, or a hole wall that joins a fillet "
                   "or a chamfer under smooth_angle_deg). Their flat parts and holes are not listed.")
    return out
