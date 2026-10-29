"""Software render of STL parts for a vision model (CONTRACTS.md section 15).

No display, no OpenGL, no VTK. An orthographic z-buffer rasterizer (numba) with flat
shading, crease and silhouette lines. Back faces are magenta. Needs numpy, numba, trimesh,
manifold3d (only for `section`) and Pillow.

    from topopt.render import render
    facts = render("runs/live")                                   # a run folder
    facts = render([("part.stl", "part"), ("plate.stl", "fixed")], section="y=0")

The result is a dict of facts: PNG paths with sizes, per mesh facts, back face share,
missing files, the font and the time. The same inputs give the same PNG bytes.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont

from .config import UserError
from .render_input import (DEFAULT_SIZE, DEFAULT_VIEWS, NAMED_VIEWS, ROLES, YAML_ROLES, check_options, normalize_sources,  # noqa: F401
                           parse_section, section_views,
                           parse_size, parse_views)

# --- documented choices --------------------------------------------------------------------


# One color per role (RGB).
ROLE_COLORS = {
    "part": (154, 167, 180),      # grey-blue
    "design": (120, 165, 235),    # light blue
    "fixed": (40, 90, 200),       # dark blue
    "load": (214, 39, 40),        # red
    "keep-out": (150, 100, 40),   # brown
    "interface": (50, 160, 90),   # green: a union interface shape (a problem YAML only)
    "cut": (235, 70, 110),        # red: a cut interface shape, drawn as a ghost (a problem YAML only)
}
ALL_ROLES = ROLES + YAML_ROLES
LEGEND_LABEL = {"interface": "union shape", "cut": "cut shape"}
BACK_COLOR = (217, 26, 191)       # magenta: back faces (flipped or open surface)
CUT_COLOR = (232, 140, 60)        # orange: cut faces of a section, drawn flat
# Ghost styles: (tint alpha, line alpha, line color = role color times this factor).
GHOST_LIGHT = (0.14, 0.6, 0.55)   # design space, keep-out: a faint tint with thin lines
GHOST_STRONG = (0.35, 0.9, 0.8)   # fixed and load regions when a part is drawn
EDGE_LEVEL = 0.12                 # grey level of the lines of solid meshes
# Force arrows of a problem YAML (CONTRACTS.md section 15). They are drawing constants, not limits. The shape is the one of
# `addLoadArrows` in the dashboard (topopt/dashboard/js/app.js), so the picture before a run and the picture during a run agree.
ARROW_LENGTH_SHARE = 0.18         # arrow length / diagonal of the drawing: the dashboard rule. The same length for every arrow
ARROW_SHAFT_RADIUS = 0.045        # shaft radius / arrow length (dashboard)
ARROW_HEAD_LENGTH = 0.32          # head length / arrow length (dashboard)
ARROW_HEAD_RADIUS = 3.0           # head base radius / shaft radius (dashboard)
ARROW_SEGMENTS = 16               # sides of the shaft and of the head: round enough at tile size, few triangles
ARROW_SHADE_MIN = 0.75            # brightest shading is 1.0: a narrow range keeps the case color easy to match
# One color per load case (RGB), in the order of the load cases; the colors repeat after the last one. None is a color of the
# role colors above, of the back face or of the cut face (CIEDE2000 distance 19 or more to each; checked 2026-10-23).
CASE_COLORS = (
    (120, 0, 200),     # violet
    (176, 150, 0),     # gold
    (0, 140, 150),     # teal
    (120, 0, 80),      # plum
    (0, 100, 40),      # dark green
)
MEM_BASE_MB, MEM_MB_PER_MTRI = 450, 450   # rough memory model: 477 MB for the hook, 1.0 GB at 1.4 M triangles

MAX_LONG_EDGE = 1568              # the montage; larger images are scaled down by vision models
SS = 2                            # supersampling per axis
LABEL_PX = 18                     # tile labels and header text; 16 px is the smallest to read

# Font chain: the first file that opens wins. The last step is the font that Pillow bundles.
FONT_CHAIN = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
)
PILLOW_DEFAULT = "pillow-default"
_font_used: dict = {}


def load_font(size: int):
    """Return a font of about `size` pixels. Never raises.

    Order: the file in env var TOPOPT_FONT, then FONT_CHAIN, then `ImageFont.load_default(size=N)`
    (Pillow 10.1 and newer bundle a font, so this works with no system fonts). The name that
    won is in `font_name()`.
    """
    env = os.environ.get("TOPOPT_FONT")
    for path in ([env] if env else []) + list(FONT_CHAIN):
        try:
            font = ImageFont.truetype(path, size)
        except (OSError, ValueError):
            continue
        _font_used["name"] = path
        return font
    _font_used["name"] = PILLOW_DEFAULT
    return ImageFont.load_default(size=size)


def font_name() -> str:
    """The font of the last `load_font` call: a file path or `pillow-default`."""
    return _font_used.get("name", PILLOW_DEFAULT)


# --- rasterizer ----------------------------------------------------------------------------

@njit(cache=True)
def raster(P, F, W, H, zbuf, idbuf):
    """Fill zbuf and idbuf. P: (n,3) screen x, screen y (pixels), depth (smaller is closer)."""
    for t in range(F.shape[0]):
        a, b, c = F[t, 0], F[t, 1], F[t, 2]
        x0, y0, z0 = P[a, 0], P[a, 1], P[a, 2]
        x1, y1, z1 = P[b, 0], P[b, 1], P[b, 2]
        x2, y2, z2 = P[c, 0], P[c, 1], P[c, 2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        xmin = max(int(np.floor(min(x0, min(x1, x2)))), 0)
        xmax = min(int(np.ceil(max(x0, max(x1, x2)))), W - 1)
        ymin = max(int(np.floor(min(y0, min(y1, y2)))), 0)
        ymax = min(int(np.ceil(max(y0, max(y1, y2)))), H - 1)
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                px = x + 0.5
                py = y + 0.5
                l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                l2 = 1.0 - l0 - l1
                if l0 >= -1e-7 and l1 >= -1e-7 and l2 >= -1e-7:
                    z = l0 * z0 + l1 * z1 + l2 * z2
                    if z < zbuf[y, x]:
                        zbuf[y, x] = z
                        idbuf[y, x] = t


def _basis(d):
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)                      # the camera sits at +d and looks along -d
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.99 else np.array([0, 1.0, 0])
    right = np.cross(up, d)
    right /= np.linalg.norm(right)
    return right, np.cross(d, right), d


def _face_normals(V, F):
    """Unit normals from the winding (the file normals are not used). Zero for a flat triangle."""
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(n, axis=1)
    return n / np.where(ln > 0, ln, 1.0)[:, None]


class _Layer:
    """Triangles to draw: vertices, faces, face normals, face colors, per-vertex depth bias, cut-face flags."""

    def __init__(self, V, F, fn, fcol, vbias, fcut=None):
        self.V, self.F, self.fn, self.fcol, self.vbias = V, F, fn, fcol, vbias
        self.fcut = np.zeros(len(F), bool) if fcut is None else fcut


def arrow_mesh(tail, tip):
    """One closed arrow (a shaft and a cone head) from `tail` to `tip`, as `(V, F)`. The winding points outward.

    The proportions are `ARROW_*`: they are shares of the arrow length, so the arrow looks the same at every size.
    """
    tail, tip = np.asarray(tail, float), np.asarray(tip, float)
    length = float(np.linalg.norm(tip - tail))
    d = (tip - tail) / length
    u = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)                                               # u, v, d are a right-handed set
    n = ARROW_SEGMENTS
    ang = 2 * np.pi * np.arange(n) / n
    radial = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v     # (n, 3) unit vectors around the axis
    r = ARROW_SHAFT_RADIUS * length
    base = tip - d * ARROW_HEAD_LENGTH * length                      # where the head starts
    V = np.vstack([tail[None], tail + r * radial, base + r * radial, base + ARROW_HEAD_RADIUS * r * radial, tip[None]])
    centre, a0, a1, b, apex = 0, 1, 1 + n, 1 + 2 * n, 1 + 3 * n
    F = []
    for k in range(n):
        k1 = (k + 1) % n
        F += [(centre, a0 + k1, a0 + k),                             # the end of the shaft
              (a0 + k, a0 + k1, a1 + k), (a0 + k1, a1 + k1, a1 + k),  # the shaft
              (a1 + k, b + k1, b + k), (a1 + k, a1 + k1, b + k1),    # the back of the head (a ring)
              (b + k, b + k1, apex)]                                 # the cone
    return V, np.array(F, np.int64)


def arrow_layer(arrows):
    """One `_Layer` for all arrows. `arrows`: a list of `(tail, tip, rgb)`."""
    Vs, Fs, cols, off = [], [], [], 0
    for tail, tip, rgb in arrows:
        V, F = arrow_mesh(tail, tip)
        Vs.append(V)
        Fs.append(F + off)
        cols.append(np.tile(np.array(rgb, float) / 255, (len(F), 1)))
        off += len(V)
    V, F = np.vstack(Vs), np.vstack(Fs)
    return _Layer(V, F, _face_normals(V, F), np.vstack(cols), np.zeros(len(V)))


def _pass(layer, right, upv, dirn, c, scale, ox, oy, W, H, span):
    """Rasterize one layer. Returns zbuf, idbuf (size H x W in supersampled pixels)."""
    Vc = layer.V - c
    X, Y = Vc @ right, Vc @ upv
    Z = -(Vc @ dirn) + layer.vbias * span          # smaller is closer to the camera
    P = np.stack([X * scale + ox, -Y * scale + oy, Z], 1)
    zb = np.full((H, W), np.inf, np.float32)
    ib = np.full((H, W), -1, np.int32)
    raster(np.ascontiguousarray(P, np.float64), np.ascontiguousarray(layer.F, np.int64), W, H, zb, ib)
    return zb, ib


def _edges(ib, zb, fn, span):
    """Mask of crease (35 degrees), silhouette and depth-jump pixels."""
    hit = ib >= 0
    n = fn[np.where(hit, ib, 0)]
    ibp, zbp = np.pad(ib, 1, mode="edge"), np.pad(zb, 1, mode="edge")
    edge = np.zeros_like(hit)
    h, w = hit.shape
    for dy, dx in ((0, 1), (1, 0)):
        nb_id = ibp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        nb_z = zbp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        nb_n = fn[np.where(nb_id >= 0, nb_id, 0)]
        crease = hit & (nb_id >= 0) & ((n * nb_n).sum(-1) < np.cos(np.radians(35)))
        sil = hit != (nb_id >= 0)
        with np.errstate(invalid="ignore"):
            depth = hit & (nb_id >= 0) & (np.abs(zb - nb_z) > 0.02 * span)
        edge |= crease | sil | depth
    return edge


def fit_scale(allv, d, W, H):
    """Pixels per mm (supersampled) that fit the projection of `allv` into a W x H tile."""
    right, upv, _ = _basis(d)
    c = (allv.min(0) + allv.max(0)) / 2
    A, B = (allv - c) @ right, (allv - c) @ upv
    return 0.85 * min(W * SS / (np.ptp(A) + 1e-9), H * SS / (np.ptp(B) + 1e-9))


def render_view(solid, ghosts, d, W, H, scale, arrows=None):
    """Render one view into an (H, W, 3) uint8 array. Returns `(image, stats)`.

    `scale`: supersampled pixels per mm (the same for all tiles of a montage).
    `arrows`: the `arrow_layer` of the force arrows, or None. They are drawn last, over everything (as in the dashboard).
    `stats`: `visible_px` (pixels of solid meshes, supersampled), `back_px` (of them back faces) and
    `cut_px` (visible cut-face pixels).
    """
    right, upv, dirn = _basis(d)
    allv = np.vstack([solid.V] + [g.V for g, *_ in ghosts] + ([arrows.V] if arrows is not None else []))
    c = (allv.min(0) + allv.max(0)) / 2
    A = (allv - c) @ right
    B = (allv - c) @ upv
    span = float(np.ptp((allv - c) @ dirn)) + 1e-9
    Ws, Hs = W * SS, H * SS
    ox = Ws / 2 - (A.max() + A.min()) / 2 * scale
    oy = Hs / 2 + 12 * SS + (B.max() + B.min()) / 2 * scale     # a little lower: room for the label
    zb, ib = _pass(solid, right, upv, dirn, c, scale, ox, oy, Ws, Hs, span)
    hit = ib >= 0
    idx = np.where(hit, ib, 0)
    n = solid.fn[idx]
    facing = n @ dirn                                           # below 0: a back face toward the camera
    light = -0.35 * right + 0.45 * upv + 0.82 * dirn            # key light: camera, up and left
    lam = np.clip(n @ light / np.linalg.norm(light), 0, 1)
    shade = (0.35 + 0.65 * lam)[..., None]
    img = np.where(hit[..., None], solid.fcol[idx] * shade, 1.0)
    back = hit & (facing < 0)
    img[back] = np.array(BACK_COLOR) / 255 * (0.5 + 0.5 * np.clip(-facing[back], 0, 1))[:, None]
    edge = _edges(ib, zb, solid.fn, span)
    img[edge] = EDGE_LEVEL
    for g, rgb, (alpha, ealpha, emul) in ghosts:
        gz, gi = _pass(g, right, upv, dirn, c, scale, ox, oy, Ws, Hs, span)
        gh = gi >= 0
        tint = np.array(rgb) / 255
        img = np.where(gh[..., None], img * (1 - alpha) + tint * alpha, img)
        with np.errstate(invalid="ignore"):
            seen = gz <= zb + 0.002 * span                      # lines that no solid mesh hides
        ge = _edges(gi, gz, g.fn, span) & seen
        img = np.where(ge[..., None], img * (1 - ealpha) + tint * emul * ealpha, img)
    cut = hit & solid.fcut[idx] & (facing >= 0)                 # cut faces: flat orange, over the ghosts
    img = np.where((cut & ~edge)[..., None], np.array(CUT_COLOR) / 255, img)
    if arrows is not None:
        az, ai = _pass(arrows, right, upv, dirn, c, scale, ox, oy, Ws, Hs, span)
        ah = ai >= 0
        an = arrows.fn[np.where(ah, ai, 0)]
        alam = np.clip(an @ light / np.linalg.norm(light), 0, 1)
        acol = arrows.fcol[np.where(ah, ai, 0)] * (ARROW_SHADE_MIN + (1 - ARROW_SHADE_MIN) * alam)[..., None]
        img = np.where(ah[..., None], acol, img)
        img[_edges(ai, az, arrows.fn, span)] = EDGE_LEVEL        # the same line grey as the solid meshes
    out = (img.reshape(H, SS, W, SS, 3).mean((1, 3)) * 255 + 0.5).astype(np.uint8)
    return out, {"visible_px": int(hit.sum()), "back_px": int(back.sum()), "cut_px": int(cut.sum())}


def nice_length(target):
    """The largest of 1, 2, 5 times a power of 10 that is at most `target`."""
    if target <= 0:
        return 1.0
    p = 10.0 ** np.floor(np.log10(target))
    for m in (5, 2, 1):
        if m * p <= target:
            return float(m * p)
    return float(p)


def _load_mesh(path: Path):
    import trimesh
    try:
        m = trimesh.load(str(path), force="mesh")
        V = np.asarray(m.vertices, float)
        F = np.asarray(m.faces, np.int64)
    except Exception as exc:
        raise UserError(f"cannot read the mesh {path}: {type(exc).__name__}: {exc}")
    if len(F) == 0:
        raise UserError(f"the mesh {path} has no triangles")
    return m, V, F


# --- section -------------------------------------------------------------------------------

def _trim(V, F, axis, value, side):
    """Keep the side `side` (+1: AXIS >= VALUE, -1: AXIS <= VALUE) with manifold3d.

    Returns `(status, V, F, note)`. `status` is `ok`, `empty` (no material left) or `failed`.
    An inside-out mesh (negative volume) is flipped first, and `note` says so.
    """
    try:
        import manifold3d as M
    except ImportError as exc:
        return "failed", None, None, f"manifold3d is not installed: {exc}"
    try:
        note = None

        def build(faces):
            return M.Manifold(M.Mesh64(vert_properties=np.ascontiguousarray(V, np.float64),
                                       tri_verts=np.ascontiguousarray(faces, np.uint64)))
        mf = build(F)
        if mf.status() == M.Error.NoError and not mf.is_empty() and mf.volume() < 0:
            mf = build(F[:, ::-1])
            note = "the mesh is inside out (negative volume): it was flipped for the section"
        if mf.status() != M.Error.NoError or mf.is_empty():
            return "failed", None, None, f"not a valid closed solid ({mf.status().name}): the section was not made"
        normal = [0.0, 0.0, 0.0]
        normal[axis] = float(side)
        me = mf.trim_by_plane(tuple(normal), float(side) * float(value)).to_mesh64()
    except Exception as exc:
        return "failed", None, None, f"the section failed: {type(exc).__name__}: {exc}"
    Vn = np.asarray(me.vert_properties, float)[:, :3]
    Fn = np.asarray(me.tri_verts, np.int64)
    if len(Fn) == 0:
        return "empty", None, None, "no material on the kept side: the mesh is not drawn"
    return "ok", Vn, Fn, note


def _clip_by_center(V, F, axis, value, side):
    """Fallback for a mesh that cannot be cut: drop the triangles whose center is on the removed side."""
    keep = side * (V[F][:, :, axis].mean(1) - value) >= 0
    return F[keep]


# --- main ----------------------------------------------------------------------------------

def _hex(c):
    return "#%02x%02x%02x" % tuple(c)


def _yaml_inputs(yaml_path: Path):
    """`(items, extras, notes, cfg)` of a problem YAML: the region STLs as `[(path, role)]`, the interface shapes as `[(name, verts, faces, role)]`, the config.

    The roles are those of the regions (a solid region is drawn like a load region, as for a run folder). A union interface shape has
    the role `interface`, a cut shape the role `cut`. The cuts are clipped to the grid box (`interfaces.render_shapes`).
    """
    import trimesh
    from .config import load_config

    cfg = load_config(yaml_path)
    reg = cfg.regions
    items = [(reg.design, "design")]
    items += [(p, "fixed") for p in reg.fixed] + [(p, "keep-out") for p in reg.keep_out]
    items += [(lr.path, "load") for lr in reg.load] + [(p, "load") for p in reg.solid]
    extras, notes = [], []
    if cfg.interfaces:
        from . import interfaces
        bounds = []
        for p, _role in items:
            try:
                bounds.append(trimesh.load(str(p), force="mesh").bounds)
            except Exception as exc:
                raise UserError(f"cannot read the mesh {p}: {type(exc).__name__}: {exc}")
        b = np.array(bounds)
        unions, cuts, notes = interfaces.render_shapes(cfg, (b[:, 0].min(axis=0), b[:, 1].max(axis=0)))
        extras += [(f"interface {k}", v, f, "interface") for k, (v, f) in unions.items()]
        extras += [(f"interface {k}", v, f, "cut") for k, (v, f) in cuts.items()]
    return items, extras, notes, cfg


def _region_center(V, F, closed):
    """`(point, how)`: the volume centroid of a closed mesh (watertight, consistent winding), else the center of its bounding box.

    `how` is `region centroid` or `region bbox center`."""
    if closed:
        t = V[F]
        det = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))
        vol = det.sum() / 6.0
        if vol != 0.0:
            return (det[:, None] * (t[:, 0] + t[:, 1] + t[:, 2]) / 4.0).sum(0) / (6.0 * vol), "region centroid"
    return (V.min(0) + V.max(0)) / 2, "region bbox center"


def _r3(v):
    return [round(float(x), 3) for x in v]


def _g6(x):
    return float(f"{float(x):.6g}")


def load_arrows(cfg, geometry, diagonal):
    """The force arrows of a problem YAML: `(arrows, facts, notes)`.

    One arrow for each force entry with a force, in each load case. It ends (the tip) at the point where the force acts: `at` of
    a remote force, else the volume centroid of the load region STL (the run takes the mean of the load voxels: about a voxel
    away, more if the region sticks out of the grid). It points along the force and has the length `ARROW_LENGTH_SHARE` times
    `diagonal` (mm), whatever the size of the force. The colors follow the load cases (`CASE_COLORS`). A moment and a pressure
    get no arrow. `geometry`: `{str(region path): (V, F, closed)}`. `arrows`: `[(tail, tip, rgb)]`.
    """
    from .loads import entry_kind
    length = ARROW_LENGTH_SHARE * diagonal
    centers = {lr.name: _region_center(*geometry[str(lr.path)]) for lr in cfg.regions.load}
    arrows, cases, notes, sizes = [], [], [], []
    for ci, case in enumerate(cfg.load_cases):
        rgb = CASE_COLORS[ci % len(CASE_COLORS)]
        drawn, skipped = [], []
        for f in case.forces:
            force = np.array(f.force, float)
            size = float(np.linalg.norm(force))
            if size == 0.0:       # `entry_kind`: a moment alone has a zero force
                skipped.append({"region": f.region, "kind": "moment", "moment_nmm": _r3(f.moment or (0, 0, 0)),
                                "reason": "a moment has no arrow; only a force has one"})
                continue
            point, how = (np.array(f.at, float), "at") if f.at is not None else centers[f.region]
            tip = point
            tail = tip - force / size * length
            entry = {"region": f.region, "kind": entry_kind(f),
                     "force_n": [_g6(x) for x in force], "size_n": _g6(size),
                     "point_mm": _r3(point), "point_source": how,
                     "arrow_tail_mm": _r3(tail), "arrow_tip_mm": _r3(tip), "color": _hex(rgb)}
            if f.moment is not None:
                entry["moment_nmm"] = _r3(f.moment)
                entry["moment_drawn"] = False
            if how == "region bbox center":
                notes.append(f"load case {case.name!r}, region {f.region!r}: the mesh is not a closed solid, so the arrow ends at the center of its bounding box")
            arrows.append((tail, tip, rgb))
            sizes.append(size)
            drawn.append(entry)
        for p in case.pressures:
            skipped.append({"region": p.region, "kind": "pressure", "value_mpa": p.value_mpa,
                            "reason": "a pressure acts along the face normals of its region; it has no arrow"})
        if not drawn:
            notes.append(f"load case {case.name!r} has no force arrow: " + (
                "it has no force entry with a force (a moment or a pressure gets no arrow)" if skipped else "it has no force entry"))
        cases.append({"case": case.name, "color": _hex(rgb) if drawn else None, "arrows": drawn, "not_drawn": skipped})
    if len(cfg.load_cases) > len(CASE_COLORS):
        notes.append(f"the YAML has {len(cfg.load_cases)} load cases and there are {len(CASE_COLORS)} arrow colors: the colors repeat")
    if sizes and max(sizes) > min(sizes):
        notes.append(f"the forces differ in size ({min(sizes):.6g} N to {max(sizes):.6g} N), and every arrow has the same length: "
                     "the length does not show the size of the force")
    facts = {"drawn": True, "arrow_count": len(arrows), "length_mm": _g6(length),
             "length_rule": f"{ARROW_LENGTH_SHARE} of the diagonal of the drawing, the same for every arrow: the length does not show the size of the force",
             "tip_rule": "the arrow tip is at point_mm, the point where the force acts; the arrow points along force_n",
             "cases": cases}
    return arrows, facts, notes


def render(sources, *, roles=None, views=None, section=None, size=None, out_dir=None,
           regions=None, max_triangles=None, loads=None) -> dict:
    """Render STL meshes into one PNG per view and one labeled montage. Returns facts (a dict).

    sources: a run folder (renders `result/final.stl` and the region STLs of `run.json`), or a problem YAML (the region STLs
        and the interface shapes, before a run: fixed, load and union shapes solid; design space, keep-out regions and cut
        shapes as ghosts), or a path, or a list of paths. In a list an item can be `(path, role)`.
    roles: optional list of roles for the list: one role for all, or one per file. Roles:
        part (default), design, fixed, load, keep-out.
    views: names from NAMED_VIEWS or `NAME:dx,dy,dz` (letters, digits, `_`, `-` in the name). Default: iso,
        front, right, top; with a section: cut, iso-cut and two side views of the kept part.
    section: `"x=30"` or `"x>=30"` keeps x >= 30; `"x<=30"` keeps x <= 30. Axis x, y or z, VALUE in mm.
    size: one tile, `"WxH"` or `(w, h)`. Default 640x480. The montage long edge is at most 1568 px.
    out_dir: default `<run>/renders` or `renders` beside the first STL.
    regions: how the non-part meshes are drawn. `"ghost"` (default when a part is drawn): design and
        keep-out as faint ghosts, fixed and load as strong ghosts. `"solid"` (default without a part):
        fixed, load and keep-out solid; the design space stays a ghost when other meshes are drawn.
        `"off"`: only the part.
    max_triangles: stop with a UserError when the meshes have more triangles than this (default: no limit).
    loads: for a problem YAML, `False` draws no force arrows (default: one arrow for each force entry of each load case, along the
        force, ending at the point where it acts; a color for each load case; `load_arrows` in the facts). With another input
        the option has no effect (a note says so).

    Raises `UserError` for wrong input.
    """
    t0 = time.perf_counter()
    sec = parse_section(section) if section else None
    view_list = parse_views(views, section)
    tw, th = parse_size(size)
    check_options(regions, max_triangles, loads)
    kind, run_dir, items, missing, notes = normalize_sources(sources, roles)
    extras, yaml_path, cfg = [], None, None
    if kind == "yaml":   # a problem YAML: the regions and the interface shapes, before a run
        yaml_path = items[0][0]
        items, extras, notes, cfg = _yaml_inputs(yaml_path)
        if regions is not None:
            notes.append("regions has no effect for a problem YAML: fixed, load and union shapes are solid; design, keep-out and cut shapes are ghosts")
            regions = None
    elif loads is not None:
        notes.append("loads has no effect: force arrows are drawn for a problem YAML only")
    has_part = any(r == "part" for _, r in items)
    if regions == "off" and not has_part:
        raise UserError("regions=off needs a mesh with the role part")
    regions = regions or ("ghost" if has_part else "solid")

    if out_dir is not None:
        out = Path(out_dir).expanduser().resolve()
    else:
        out = (run_dir if run_dir else yaml_path.parent if yaml_path else items[0][0].parent) / "renders"
    prefix = yaml_path.stem if yaml_path else items[0][0].stem
    tag = ""
    if sec is not None:
        tag = f"_section-{'xyz'[sec[0]]}{sec[1]:g}" + ("-below" if sec[2] < 0 else "")
    inputs = {p.resolve() for p, _ in items} | ({yaml_path.resolve()} if yaml_path else set())

    def mode_of(role):
        if role == "part":
            return "solid"
        if kind == "yaml":
            return "solid" if role in ("fixed", "load", "interface") else "ghost"
        if regions == "off":
            return "off"
        if role == "design":
            return "ghost" if len(items) > 1 else "solid"
        if role == "keep-out":
            return "ghost" if regions == "ghost" else "solid"
        return "ghost" if regions == "ghost" else "solid"           # fixed, load

    # load, check, cut
    import trimesh

    def loaded():   # (file name, path or None, role, mesh, V, F): the files first, then the interface shapes of a YAML
        for path, role in items:
            yield (path.name, str(path), role, *_load_mesh(path))
        for name, V, F, role in extras:
            yield name, None, role, trimesh.Trimesh(V, F, process=False), V, F

    meshes, facts_meshes, tri_total, geometry = [], [], 0, {}
    for fname, fpath, role, m, V, F in loaded():
        tri_total += len(F)
        if max_triangles is not None and tri_total > max_triangles:
            raise UserError(f"the meshes have more than {max_triangles} triangles ({tri_total} so far; "
                            f"memory is about {MEM_MB_PER_MTRI} MB per million triangles). "
                            "Raise max_triangles, or render fewer or simpler meshes")
        wt = bool(m.is_watertight)
        wc = bool(m.is_winding_consistent)
        t = V[F]
        vol = float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
        known = wt and wc
        if role == "load" and fpath:
            geometry[fpath] = (V, F, known)       # before a section: the arrow starts at the region as the YAML gives it
        if wt and not wc:
            notes.append(f"{fname}: the winding is not consistent, so volume_mm3 and inside_out are not given")
        mode = mode_of(role)
        f = {"file": fname, "path": fpath, "role": role, "drawn": mode,
             "bbox_mm": {"min": [round(float(x), 3) for x in V.min(0)],
                         "max": [round(float(x), 3) for x in V.max(0)],
                         "size": [round(float(x), 3) for x in np.ptp(V, axis=0)]},
             "triangles": int(len(F)), "watertight": wt, "winding_consistent": wc,
             "volume_mm3": round(vol, 1) if known else None,
             "inside_out": bool(vol < 0) if known else None}
        cut = None
        if sec is not None and mode != "off":
            axis, val, side = sec
            status, Vn, Fn, note = _trim(V, F, axis, val, side)
            if status == "ok":
                tol = 1e-6 * max(1.0, abs(val), float(np.ptp(V, axis=0).max()))
                cut = np.all(np.abs(Vn[Fn][:, :, axis] - val) < tol, axis=1)
                V, F = Vn, Fn
                f["section"] = {"applied": True, "kept_triangles": int(len(F)), "cut_faces": int(cut.sum())}
            elif status == "empty":
                f["section"] = {"applied": True, "kept_triangles": 0, "note": note}
                f["drawn"] = mode = "removed"
            else:
                f["section"] = {"applied": False, "note": note}
                if role != "part":                     # the part is drawn whole; a region loses its removed side
                    F = _clip_by_center(V, F, axis, val, side)
                    f["section"]["note"] += "; the triangles with their center on the removed side were dropped"
                    if len(F) == 0:
                        f["drawn"] = mode = "removed"
            if status == "ok" and note:
                f["section"]["note"] = note
        meshes.append((V, F, role, mode, cut if role in ("part", "design") and mode == "solid" else None))
        facts_meshes.append(f)
    drawn_tris = sum(len(F) for _, F, _, mode, _ in meshes if mode not in ("off", "removed"))
    if sec is not None and not any(f.get("section", {}).get("applied") for f in facts_meshes):
        notes.append("the section could not run on any mesh; the render shows the meshes whole")

    def layer(sel):
        Vs, Fs, cols, biases, cuts, off = [], [], [], [], [], 0
        for V, F, role, mode, cut in sel:
            col = np.tile(np.array(ROLE_COLORS[role], float) / 255, (len(F), 1))
            Vs.append(V)
            Fs.append(F + off)
            cols.append(col)
            cuts.append(np.zeros(len(F), bool) if cut is None else cut)
            biases.append(np.full(len(V), 0.0 if role == "part" else 0.002))
            off += len(V)
        V, F = np.vstack(Vs), np.vstack(Fs)
        return _Layer(V, F, _face_normals(V, F), np.vstack(cols), np.concatenate(biases), np.concatenate(cuts))

    solid_sel = [m for m in meshes if m[3] == "solid" and len(m[1])]
    if not solid_sel:
        raise UserError("nothing to draw: the section leaves no material of the solid meshes")
    solid = layer(solid_sel)
    ghosts = []
    for r in ALL_ROLES:                                  # light ghosts first, then the strong ones
        sel = [m for m in meshes if m[3] == "ghost" and m[2] == r and len(m[1])]
        if sel:
            ghosts.append((layer(sel), ROLE_COLORS[r], GHOST_LIGHT if r in ("design", "keep-out") else GHOST_STRONG))
    ghosts.sort(key=lambda g: g[2][0])

    # force arrows (a problem YAML)
    arrows, arrow_facts = None, None
    if cfg is not None:
        if loads is False:
            arrow_facts = {"drawn": False, "note": "loads is false: no force arrow is drawn"}
        else:
            lo = np.min([f["bbox_mm"]["min"] for f in facts_meshes], axis=0)
            hi = np.max([f["bbox_mm"]["max"] for f in facts_meshes], axis=0)
            arrow_list, arrow_facts, arrow_notes = load_arrows(cfg, geometry, float(np.linalg.norm(hi - lo)))
            notes += arrow_notes
            arrows = arrow_layer(arrow_list) if arrow_list else None

    # tile size: the montage must fit MAX_LONG_EDGE
    cols_n = 2 if len(view_list) > 1 else 1
    rows_n = (len(view_list) + cols_n - 1) // cols_n
    head_h = 2 * (LABEL_PX + 8) + 4
    while max(cols_n * tw, head_h + rows_n * th + 2 * (LABEL_PX + 12)) > MAX_LONG_EDGE:
        tw, th = int(tw * 0.95), int(th * 0.95)

    font = load_font(LABEL_PX)
    allv = np.vstack([solid.V] + [g.V for g, *_ in ghosts] + ([arrows.V] if arrows is not None else []))
    scale = min(fit_scale(allv, d, tw, th) for _, d in view_list)      # one scale for all tiles
    px_per_mm = scale / SS
    bar_mm = nice_length(tw / 5 / px_per_mm)
    bar_px = bar_mm * px_per_mm
    bar_text = f"{bar_mm:g} mm"

    tiles, view_facts, vis, back, cut_total = [], [], 0, 0, 0
    for name, d in view_list:
        arr, st = render_view(solid, ghosts, d, tw, th, scale, arrows)
        tile = Image.fromarray(arr)
        dt = ImageDraw.Draw(tile)
        dt.text((8, 6), name, fill=(0, 0, 0), font=font)
        by = th - 14
        dt.line([(10, by), (10 + bar_px, by)], fill=(0, 0, 0), width=2)
        dt.line([(10, by - 5), (10, by + 5)], fill=(0, 0, 0), width=2)
        dt.line([(10 + bar_px, by - 5), (10 + bar_px, by + 5)], fill=(0, 0, 0), width=2)
        dt.text((10, by - 8 - LABEL_PX), bar_text, fill=(0, 0, 0), font=font)
        tiles.append(tile)
        vis += st["visible_px"]
        back += st["back_px"]
        cpx = int(round(st["cut_px"] / SS ** 2))
        cut_total += cpx
        view_facts.append({"name": name, "direction": [float(x) for x in d],
                           "back_face_share": round(st["back_px"] / st["visible_px"], 4) if st["visible_px"] else 0.0,
                           "cut_face_visible_px": cpx})

    cut_views = sum(1 for v in view_facts if v["cut_face_visible_px"] > 0)
    section_ok = sec is not None and any(f.get("section", {}).get("applied") for f in facts_meshes)
    if section_ok and cut_total == 0:
        cd = section_views(section)[0][1]
        notes.append("the cut faces are not visible in these views; add the view "
                     f"cut:{cd[0]:g},{cd[1]:g},{cd[2]:g} (a direction from the part to the camera)")

    main = facts_meshes[0]
    bb = main["bbox_mm"]["size"]
    head1 = (f"{main['file']}   bbox {bb[0]:.1f} x {bb[1]:.1f} x {bb[2]:.1f} mm   "
             f"tris {main['triangles']}   watertight {'yes' if main['watertight'] else 'no'}")
    if yaml_path:   # the first mesh is the design space: name the YAML, and the size of the whole drawing
        lo = np.min([f["bbox_mm"]["min"] for f in facts_meshes], axis=0)
        hi = np.max([f["bbox_mm"]["max"] for f in facts_meshes], axis=0)
        head1 = f"{yaml_path.name}   problem YAML   bbox {hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} x {hi[2] - lo[2]:.1f} mm   tris {tri_total}"
    parts = [f"{LEGEND_LABEL.get(r, r)} {sum(1 for f in facts_meshes if f['role'] == r)}" for r in ALL_ROLES
             if r != "part" and any(f["role"] == r for f in facts_meshes)]
    if missing:
        parts.append(f"missing files {len(missing)}")
    if arrows is not None:
        parts.append(f"force arrows {arrow_facts['arrow_count']}")
    if sec is not None:
        op = ">=" if sec[2] > 0 else "<="
        parts.append(f"section {'xyz'[sec[0]]} {op} {sec[1]:g} "
                     + (f"ok, cut visible in {cut_views} of {len(view_facts)} views" if section_ok else "NOT MADE"))
    head2 = "   ".join(parts) if parts else "one mesh"

    # legend: each swatch looks like the way the role is drawn
    legend = []
    for r in ALL_ROLES:
        modes = {f["drawn"] for f in facts_meshes if f["role"] == r} - {"off", "removed"}
        if modes:
            legend.append((LEGEND_LABEL.get(r, r), ROLE_COLORS[r], "ghost" if "ghost" in modes and "solid" not in modes else "solid", r))
    if arrows is not None:       # a swatch for each load case that has an arrow
        legend += [(f"force: {c['case']}", CASE_COLORS[i % len(CASE_COLORS)], "arrow", None)
                   for i, c in enumerate(arrow_facts["cases"]) if c["arrows"]]
    legend.append(("back face", BACK_COLOR, "solid", None))
    if section_ok:
        legend.append(("cut face", CUT_COLOR, "solid", None))
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    items_w = []
    for label, colr, style, role in legend:
        text = label + (" (ghost)" if style == "ghost" else "")
        items_w.append((text, colr, style, role, LABEL_PX + 12 + int(probe.textlength(text, font=font)) + 14))
    rows, cur, x = [], [], 8
    for it in items_w:
        if cur and x + it[4] > cols_n * tw:
            rows.append(cur)
            cur, x = [], 8
        cur.append(it)
        x += it[4]
    rows.append(cur)
    legend_h = len(rows) * (LABEL_PX + 8) + 8

    sheet = Image.new("RGB", (cols_n * tw, head_h + rows_n * th + legend_h), "white")
    dr = ImageDraw.Draw(sheet)
    dr.text((8, 4), head1, fill=(0, 0, 0), font=font)
    dr.text((8, 4 + LABEL_PX + 8), head2, fill=(60, 60, 60), font=font)
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols_n) * tw, head_h + (i // cols_n) * th))
    y = head_h + rows_n * th + 6
    for row in rows:
        x = 8
        for text, colr, style, role, w in row:
            if style == "arrow":      # a small arrow that points right, in the color of the load case
                x0, y0, x1, y1 = x, y + 2, x + LABEL_PX - 2, y + LABEL_PX
                mid, neck = (y0 + y1) / 2, x0 + (x1 - x0) * 0.55
                dr.polygon([(x0, mid - 2), (neck, mid - 2), (neck, y0), (x1, mid), (neck, y1), (neck, mid + 2), (x0, mid + 2)],
                           fill=colr, outline=(0, 0, 0))
            else:
                if style == "ghost":
                    alpha, _, emul = GHOST_LIGHT if role in ("design", "keep-out") else GHOST_STRONG
                    fill = tuple(int(255 * (1 - alpha) + c * alpha) for c in colr)
                    outline = tuple(int(c * emul) for c in colr)
                else:
                    fill, outline = colr, (0, 0, 0)
                dr.rectangle([x, y + 2, x + LABEL_PX - 2, y + LABEL_PX], fill=fill, outline=outline)
            dr.text((x + LABEL_PX + 4, y), text, fill=(0, 0, 0), font=font)
            x += w
        y += LABEL_PX + 8

    # write; never overwrite an input
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UserError(f"cannot make the output folder {out}: {exc.strerror or exc}")
    targets = [(f"{prefix}{tag}_{name}", tile) for (name, _), tile in zip(view_list, tiles)]
    targets.append((f"{prefix}{tag}_montage", sheet))
    paths = []
    for stem, img in targets:
        p = out / f"{stem}.png"
        if p.resolve() in inputs:
            raise UserError(f"the output file {p} is an input file")
        try:
            img.save(p, format="PNG", compress_level=6)
        except OSError as exc:
            raise UserError(f"cannot write {p}: {exc.strerror or exc}")
        paths.append((p, img))
    png = [{"path": str(p), "width": img.width, "height": img.height, "bytes": p.stat().st_size}
           for p, img in paths]
    for vf, e in zip(view_facts, png[:-1]):
        vf.update(e)

    sec_fact = None
    if sec is not None:
        op = ">=" if sec[2] > 0 else "<="
        sec_fact = {"axis": "xyz"[sec[0]], "value": sec[1], "keeps": f"{'xyz'[sec[0]]} {op} {sec[1]:g}",
                    "applied": section_ok, "cut_face_visible_px": cut_total, "views_with_cut": cut_views}
    return {
        "ok": True,
        "input": {"kind": kind, "run_dir": str(run_dir) if run_dir else None, **({"yaml": str(yaml_path)} if yaml_path else {})},
        "out_dir": str(out),
        "montage": png[-1],
        "views": view_facts,
        "section": sec_fact,
        "meshes": facts_meshes,
        "back_face_share": round(back / vis, 4) if vis else 0.0,
        "missing_files": missing,
        "notes": notes,
        "regions": regions,
        "load_arrows": arrow_facts,
        "colors": {**{r: _hex(c) for r, c in ROLE_COLORS.items()},
                   "back_face": _hex(BACK_COLOR), "cut_face": _hex(CUT_COLOR)},
        "ghost_style": {"design, keep-out": f"tint alpha {GHOST_LIGHT[0]}, thin lines",
                        "fixed, load (with a part)": f"tint alpha {GHOST_STRONG[0]}, lines in the role color"},
        "scale": {"px_per_mm": round(px_per_mm, 3), "bar_mm": bar_mm},
        "tile_size": [tw, th],
        "triangles_drawn": int(drawn_tris),
        "memory_estimate_mb": int(MEM_BASE_MB + MEM_MB_PER_MTRI * drawn_tris / 1e6),
        "legend": [{"label": label, "style": style} for label, _, style, _ in legend],
        "font": font_name(),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
