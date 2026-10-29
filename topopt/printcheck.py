"""Print check (CONTRACTS.md section 32).

A check of a finished part for printing. It reads an STL file, or the `result/final.stl` of a run folder, and
gives for each check the value, the limit with its source, a state (`pass`, `fail` or `unknown`) and a reason.

It is geometry only: no solve, no optional package (numpy, scipy and trimesh are core). By default it only
reports. The strict gate is opt-in: `gate.state` is `fail` only when a check that has a limit fails. An `unknown`
never sets it. No run stops on the gate.

The checks (the code of V2, `topopt/print_report.py`, makes the numbers of the last five; it is not written again):

- `mesh_closed`, `bodies`, `closed_voids`: exact counts on the STL as it reads back from the file, with the
  vertices rounded to float32 (the file format).
- `bed_fit`, `overhang`, `support_volume`, `plate_contact`: `print_report.measure_direction`, for the configured
  build direction.
- `wall_thickness`: `print_report.thickness_estimate` (chords along the inward face normal) against `min_wall_mm`.

The limits come from the `print` block of the YAML (`config.PRINT_LIMIT_KEYS`). A limit without a value gives the
state `unknown`: the value is a fact, and the report says which key makes it a limit.

The report has `partial` and `coverage` (TC-2, CONTRACTS.md section 44): which checks ran, failed or were not run. A check that stops
with an exception is a record with the state `unknown` and the key `error` when `check(contain_errors=True)` (the CLI and the MCP tool);
the other checks go on. The default keeps the exception, because the ratchet reads it as a failure of the tool.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from . import check_coverage
from .config import PRINT_DERIVABLE_KEYS, PRINT_LIMIT_KEYS, PrintSettings, UserError, parse_print_block
from .print_report import DEFINITIONS as _V2_DEFINITIONS, HEIGHT_FIT_DEFINITION as _V2_HEIGHT_FIT

GATE_EXIT_CODE = 4          # `--strict` and a failed check. 0 to 3 are taken (`topopt run` and `verify`).
MIN_ELEMENTS_ACROSS = 3     # a member needs at least this many voxels (or mesh edges) across: h <= d_min / 3
VOID_MIN_MM3 = 1e-6         # a closed shell with a smaller volume (negative) is rounding noise, not a void
ZERO_AREA_MM2 = 1e-12       # a face with a smaller area is a zero-area face (the same number as C1, `stl_facts`)
CHORD_TOL_MM = 1e-4         # a chord counts as thinner than the limit only if it is under `limit - CHORD_TOL_MM`
TOP_SHELLS = 10             # the report lists at most this many bodies and voids (largest first)
STATES = ("pass", "fail", "unknown")

# What a closed void means for each process. These are facts for the reason text. They are not limits.
VOID_NOTES = {
    "fdm": "FDM prints a closed void as a hollow; nothing is trapped",
    "sla": "SLA leaves uncured resin in a closed void",
    "lpbf": "LPBF leaves powder in a closed void",
    "sls": "SLS leaves powder in a closed void",
}

DEFINITIONS = {
    "mesh_closed": "The STL is read as float32 and the vertices that are equal are joined. A face with two equal "
                   "corners is left out (it has no area). Closed: every edge joins exactly 2 faces, the winding "
                   "agrees across each edge, the volume is positive, and no closed shell is inside out (a closed "
                   "shell with a negative volume that lies in no other shell).",
    "bodies": "A body is a shell of faces joined along edges that have exactly 2 faces, with an area above "
              f"{ZERO_AREA_MM2:g} mm2. A closed shell that has a negative volume and lies inside a closed shell with "
              "a positive volume is a void (the wall of a cavity), not a body. A closed shell with a negative "
              "volume that lies in no other shell is a body that is written inside out: it counts as a body and "
              "mesh_closed fails. A body inside a void counts as a body.",
    "closed_voids": "A closed shell with a negative volume that lies inside a closed shell with a positive volume: a "
                    "cavity inside the part with no opening. The volume is the volume of the cavity. A cavity that "
                    "only touches the outside along an edge is not closed (the mesh is not manifold there). A "
                    "cavity that two bodies enclose together is not found.",
    "bed_fit": "V2 bed fit in the build frame, with a turn about the build axis (CONTRACTS.md section 18).",
    "overhang": "V2: area of faces with n.b < -cos(angle), plate contact excluded (CONTRACTS.md section 18).",
    "support_volume": _V2_DEFINITIONS["support_volume"],
    "plate_contact": "V2: area of faces that rest on the plate (CONTRACTS.md section 18).",
    "wall_thickness": "V2 chord estimate: 60,000 surface points, a chord along the inward face normal to the first "
                      "wall. The share of the surface area with a chord under min_wall_mm - "
                      f"{CHORD_TOL_MM:g} mm is compared with min_wall_area_pct. The rule of this check: a pass "
                      "needs min_wall_mm to be at least 3 x the resolution (the voxel size of the run, or the "
                      "median mesh edge, or the option); see pitfalls/mesh-coarser-than-members. The share has a "
                      "sampling error of about sqrt(f (1 - f) / samples), given in the value.",
}


# ---------------------------------------------------------------------------
# input and settings (no numpy work here: a wrong input stops first)
# ---------------------------------------------------------------------------


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def resolve_input(path) -> dict:
    """The STL to check. `path` is an STL file or a run folder (it has `result/final.stl`). Raises `UserError`.

    Returns `kind` (`stl` or `run`), `stl` (Path), `run_dir` (Path or None) and `run` (the run.json dict or None).
    """
    if path is None or str(path).strip() == "":
        raise UserError("no input: give an STL file or a run folder")
    p = Path(path).expanduser()
    if not p.exists():
        raise UserError(f"file not found: {p.resolve()}")
    p = p.resolve()
    if p.is_dir():
        final = p / "result" / "final.stl"
        if not final.is_file():
            raise UserError(f"no result/final.stl in {p}: the run is not done, or this is not a run folder. "
                            "Give a run folder or an STL file")
        run = _read_json(p / "run.json")
        return {"kind": "run", "stl": final, "run_dir": p, "run": run if isinstance(run, dict) else None}
    if p.suffix.lower() != ".stl":
        raise UserError(f"{p.name} is not an STL file. Give an STL file or a run folder")
    return {"kind": "stl", "stl": p, "run_dir": None, "run": None}


def _merge_settings(a: PrintSettings, b: PrintSettings) -> PrintSettings:
    """`a` with every key that `b` gave (or that the printer profile of `b` filled) replaced by the value of `b`.

    The printer profile is the one of the source that named a `printer_model` last. A derived key keeps the label of the printer that
    gave its value: the bed of the printer in run.json stays `profile:<that id>` when a later YAML names another printer (MIN-5).
    """
    def pick(key):
        return b if key in b.given or key in b.derived else a

    limits = dict(a.limits)
    limits.update(dict(b.limits))
    profile = b.profile if "printer_model" in b.given else a.profile
    derived, derived_from = [], []
    for key in PRINT_DERIVABLE_KEYS:
        source = pick(key)
        if profile is not None and key in source.derived:
            derived.append(key)
            label = source.key_source(key) or f"profile:{profile.id}"       # profile:<id> of the printer that filled this value
            if label != f"profile:{profile.id}":
                derived_from.append((key, label.removeprefix("profile:")))
    return PrintSettings(
        build_direction=pick("build_direction").build_direction,
        bed_mm=pick("bed_mm").bed_mm,
        printer_type=pick("printer_type").printer_type,
        self_supporting_angle_deg=pick("self_supporting_angle_deg").self_supporting_angle_deg,
        given=tuple(dict.fromkeys(a.given + b.given)),
        limits=tuple((k, limits[k]) for k in PRINT_LIMIT_KEYS if k in limits),
        profile=profile, derived=tuple(derived), derived_from=tuple(derived_from))


def read_yaml_print(yaml_path) -> PrintSettings:
    """The `print` block of a problem YAML (a missing block gives the defaults). Raises `UserError`."""
    import yaml

    p = Path(yaml_path).expanduser()
    if not p.is_file():
        raise UserError(f"YAML file not found: {p.resolve()}")
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise UserError(f"cannot read the YAML {p}: {str(exc).splitlines()[0]}")
    if data is not None and not isinstance(data, dict):
        raise UserError(f"the YAML {p} must be a mapping")
    return parse_print_block((data or {}).get("print"), str(p))


def assemble_settings(inp: dict, yaml_path=None, overrides=None):
    """The print settings and where they come from. Later sources win: defaults, run.json, YAML, options.

    `overrides` is a dict with the keys of the YAML `print` block (the CLI `--set KEY=VALUE`, the MCP `print_settings`).
    Returns `(PrintSettings, sources, origin)`. `sources` is a list of short texts (for `settings.from`). `origin`
    maps each key that a source gave to `run.json`, `yaml` or `option` (the last source that gave it).
    """
    settings, sources, origin = PrintSettings(), ["defaults"], {}

    def take(new: PrintSettings, label: str) -> None:
        nonlocal settings
        for key in list(new.given) + [k for k, _ in new.limits]:
            origin[key] = label
        settings = _merge_settings(settings, new)

    if inp.get("run") and isinstance(inp["run"].get("print"), dict):
        take(PrintSettings.from_dict(inp["run"]["print"]), "run.json")
        sources.append("run.json")
    if yaml_path:
        take(read_yaml_print(yaml_path), "yaml")
        sources.append(f"yaml {Path(yaml_path).expanduser().name}")
    if overrides:
        if not isinstance(overrides, dict):
            raise UserError("print settings must be a mapping of print-block keys, for example {min_wall_mm: 1.2}")
        take(parse_print_block(overrides, "<options>"), "option")
        sources.append("options")
    return settings, sources, origin


def key_source(settings: PrintSettings, origin: dict, key: str) -> str:
    """Where the value of a print key comes from: `run.json`, `yaml`, `option`, `profile:<id>` (the printer profile filled it),
    `assumed` (a default) or `none`."""
    profile_source = settings.key_source(key)
    if profile_source is not None:
        return profile_source
    if key in PRINT_LIMIT_KEYS:
        base = settings.limit_source(key)
    else:
        base = "given" if key in settings.given else "assumed"
    return origin.get(key, "yaml") if base in ("yaml", "given") else base


def _check_resolution(resolution_mm) -> None:
    if resolution_mm is not None and (isinstance(resolution_mm, bool) or not isinstance(resolution_mm, (int, float))
                                      or not np.isfinite(resolution_mm) or resolution_mm < 0):
        raise UserError(f"resolution_mm must be a number of 0 or more, got {resolution_mm!r}")


# ---------------------------------------------------------------------------
# the mesh as the STL file holds it
# ---------------------------------------------------------------------------


def _check_stl_file(path: Path) -> None:
    """Stop with a clear message if the file is not an STL: a binary STL has 84 + 50 n bytes, an ASCII STL starts with
    `solid` and has a `facet`. Raises `UserError`. (trimesh gives other messages for a random file.)"""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError as exc:
        raise UserError(f"cannot read {path}: {exc.strerror or exc}")
    if size == 0:
        raise UserError(f"{path.name} is empty: it is not an STL file")
    if size >= 84 and size == 84 + 50 * int.from_bytes(head[80:84], "little"):
        return                                                    # a binary STL (n may be 0: no faces, said later)
    text = head.lstrip().lower()
    if text.startswith(b"solid") and b"facet" in text:
        return                                                    # an ASCII STL
    if size >= 84:
        n = int.from_bytes(head[80:84], "little")
        raise UserError(f"{path.name} is not an STL file: it has {size:,} bytes, but a binary STL with the "
                        f"{n:,} triangles in its header has {84 + 50 * n:,} bytes (a truncated or broken file), and "
                        "it is not an ASCII STL (no 'solid' and 'facet')")
    raise UserError(f"{path.name} is not an STL file: it has {size} bytes (a binary STL has at least 84) and it "
                    "is not an ASCII STL (no 'solid' and 'facet')")


def load_stl_float32(path: Path):
    """Read an STL and return `(vertices, faces)`: float32 vertices as float64, joined where they are equal.

    The STL format stores float32. Vertices that are equal as float32 are one vertex, so a pinch that the float32
    rounding makes (two faces that touch) shows as an edge with more than 2 faces. Raises `UserError`.
    """
    import trimesh

    _check_stl_file(path)
    try:
        raw = trimesh.load(str(path), file_type="stl", process=False, force="mesh")
        tri = np.asarray(raw.vertices, dtype=np.float32)[np.asarray(raw.faces)]
    except Exception as exc:  # noqa: BLE001 - a broken file is a user error with the reason
        raise UserError(f"cannot read the STL {path}: {type(exc).__name__}: {exc}")
    if len(tri) == 0:
        raise UserError(f"the STL {path} has no faces")
    if not np.isfinite(tri).all():
        raise UserError(f"the STL {path} has non-finite coordinates (NaN or infinity): the file is damaged")
    flat = np.ascontiguousarray(tri.reshape(-1, 3) + np.float32(0.0))   # + 0: -0.0 becomes +0.0
    key = flat.view(np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])).ravel()
    uniq, inv = np.unique(key, return_inverse=True)
    vertices = np.stack([uniq["x"], uniq["y"], uniq["z"]], axis=1).astype(np.float64)
    return vertices, inv.reshape(-1, 3).astype(np.int64)


def _winding_number(point: np.ndarray, tri: np.ndarray) -> float:
    """The winding number of closed triangle surfaces `tri` (k, 3, 3) around `point`: 1 inside a shell with outward
    normals, 0 outside (the sum of the solid angles, Van Oosterom and Strakhov)."""
    a, b, c = tri[:, 0] - point, tri[:, 1] - point, tri[:, 2] - point
    la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", a, c) * lb
           + np.einsum("ij,ij->i", b, c) * la)
    return float(np.arctan2(num, den).sum() / (2.0 * np.pi))


def mesh_health(V: np.ndarray, F: np.ndarray) -> dict:
    """Exact counts on a triangle mesh: closed state, winding, zero-area faces, shells, bodies and voids.

    Edges are the sorted vertex pairs. `open_edges` have 1 face, `nonmanifold_edges` more than 2. Two faces are
    in one shell if they share an edge that has exactly 2 faces (the rule of `trimesh.split`). A face with a
    repeated vertex has no area: it is left out of the edges and the shells (`mesh.stl_roundtrip` of C1 keeps it,
    so a file with such a face can give another count there). A shell is closed if each of its edges has 2 faces.

    The kind of a shell (after the whole mesh is turned right side out if its total volume is negative):
    - a shell with an area of at most `ZERO_AREA_MM2` is degenerate: it is not a body (`degenerate_shells`);
    - a closed shell with a negative volume that lies inside a closed shell with a positive volume is a void;
    - a closed shell with a negative volume that lies in no such shell is a body that is written inside out
      (`inverted_shells`): it counts as a body;
    - every other shell is a body (an open shell is a body).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n, total_faces = len(V), len(F)
    c = V.mean(axis=0)
    area_all = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    zero_area = int((area_all < ZERO_AREA_MM2).sum())
    # A face with a repeated vertex (the float32 rounding made two of its corners one point) has no edge, no area
    # and no volume. It stays in the zero-area count and is left out of the edges and the shells.
    keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F, area = F[keep], area_all[keep]
    m = len(F)
    if m == 0:
        raise UserError("the STL has no face with three different corners")
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    a, b, d = V[i0] - c, V[i1] - c, V[i2] - c
    det = np.einsum("ij,ij->i", a, np.cross(b, d)) / 6.0               # signed volume of each tetrahedron

    ea = np.concatenate([i0, i1, i2])
    eb = np.concatenate([i1, i2, i0])
    face = np.tile(np.arange(m), 3)
    key = np.minimum(ea, eb) * n + np.maximum(ea, eb)
    order = np.argsort(key, kind="stable")
    ks, fs, fwd = key[order], face[order], (ea < eb)[order]
    first = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]) if len(ks) else np.zeros(0, dtype=np.int64)
    count = np.diff(np.r_[first, len(ks)])
    open_edges, nonmanifold = int((count == 1).sum()), int((count > 2).sum())
    two = first[count == 2]                          # the first entry of each edge that has exactly 2 faces
    conflicts = int((fwd[two] == fwd[two + 1]).sum())
    f1, f2 = fs[two], fs[two + 1]
    graph = coo_matrix((np.ones(len(f1)), (f1, f2)), shape=(m, m))
    nshell, label = connected_components(graph, directed=False)
    bad_group = np.repeat(count != 2, count)         # entries that belong to an edge with a count other than 2
    shell_open = np.zeros(nshell, dtype=bool)
    shell_open[label[fs[bad_group]]] = True
    volume = np.bincount(label, weights=det, minlength=nshell)
    shell_area = np.bincount(label, weights=area, minlength=nshell)
    total = float(volume.sum())
    inside_out = total < 0
    if inside_out:
        volume = -volume
        total = -total
    tris = np.bincount(label, minlength=nshell)
    degenerate = shell_area <= ZERO_AREA_MM2
    closed = ~shell_open & ~degenerate
    negative = closed & (volume < -VOID_MIN_MM3)
    is_void = np.zeros(nshell, dtype=bool)
    if negative.any():
        # Is a negative closed shell inside a closed shell with a positive volume? Test the centre of its largest
        # face (a point on the shell, not on the other shell) with the winding number of the positive shells.
        positive = closed & (volume > VOID_MIN_MM3)
        P = V[F] - c                                     # (m, 3, 3): the corners of each face
        if inside_out:
            P = P[:, ::-1]                               # the same turn as the volumes: the normals point out
        lo = np.full((nshell, 3), np.inf)
        hi = np.full((nshell, 3), -np.inf)
        np.minimum.at(lo, label, P.min(axis=1))
        np.maximum.at(hi, label, P.max(axis=1))
        by_shell = np.argsort(label, kind="stable")
        starts = np.r_[0, np.cumsum(tris)]
        for s in np.flatnonzero(negative):
            fs_s = by_shell[starts[s]:starts[s + 1]]
            centre = P[fs_s[np.argmax(area[fs_s])]].mean(axis=0)
            near = positive & np.all(lo <= centre, axis=1) & np.all(centre <= hi, axis=1)   # boxes that hold it
            sel = near[label]
            if sel.any() and _winding_number(centre, P[sel]) > 0.5:
                is_void[s] = True
    inverted = negative & ~is_void
    shells = [{"triangles": int(tris[s]), "volume_mm3": round(abs(float(volume[s])), 3), "closed": bool(not shell_open[s]),
               "inside_out": bool(inverted[s])} for s in range(nshell)]
    bodies = sorted((sh for sh, v, g in zip(shells, is_void, degenerate) if not v and not g),
                    key=lambda sh: -sh["volume_mm3"])
    voids = sorted(({"triangles": sh["triangles"], "volume_mm3": sh["volume_mm3"]}
                    for sh, v in zip(shells, is_void) if v), key=lambda sh: -sh["volume_mm3"])
    return {
        "triangles": int(total_faces), "vertices": int(n),
        "watertight": bool(open_edges == 0 and nonmanifold == 0),
        "winding_consistent": bool(conflicts == 0),
        "open_edges": open_edges, "nonmanifold_edges": nonmanifold, "winding_conflicts": conflicts,
        "zero_area_faces": zero_area, "degenerate_shells": int(degenerate.sum()),
        "inverted_shells": int(inverted.sum()),
        "volume_mm3": round(total, 3), "inside_out": bool(inside_out),
        "shells": int(nshell), "bodies": bodies, "voids": voids,
    }


def mesh_edge_median(V: np.ndarray, F: np.ndarray) -> float:
    """The median length of the triangle edges, in mm (0 for a mesh with no edge)."""
    P = V[F]
    length = np.linalg.norm(np.stack([P[:, 0] - P[:, 1], P[:, 1] - P[:, 2], P[:, 2] - P[:, 0]]), axis=2).ravel()
    length = length[length > 0]
    return float(np.median(length)) if len(length) else 0.0


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------


def _limit(value, key, source, **extra) -> dict:
    return {"value": value, "key": key, "source": source, **extra}


def _check(cid, value, limit, state, reason, method, estimate=False) -> dict:
    assert state in STATES
    return {"id": cid, "state": state, "reason": reason, "value": value, "limit": limit,
            "method": method, "estimate": bool(estimate)}


def _error_text(exc: BaseException, stage: str | None = None) -> str:
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return f"{stage}: {text}" if stage else text


def _error_check(cid: str, error: str) -> dict:
    """The record of a check that stopped with an exception (TC-2): state `unknown`, no value, no limit, and the key `error`.

    `unknown` is the state of a check that gave no verdict, so `summary` counts it and `gate` ignores it (an `unknown` never sets the gate).
    """
    rec = _check(cid, {}, _limit(None, None, "none"), "unknown", f"the check stopped and gave no result: {error}",
                 "none: the check stopped before it gave a result")
    rec["error"] = error
    return rec


def _num(x) -> str:
    """4 significant digits, and whole numbers with a thousands separator from 1,000 (no exponent)."""
    x = float(x)
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def check_mesh_closed(h: dict) -> dict:
    value = {k: h[k] for k in ("triangles", "vertices", "watertight", "winding_consistent", "open_edges",
                               "nonmanifold_edges", "winding_conflicts", "zero_area_faces", "degenerate_shells",
                               "volume_mm3", "inside_out", "inverted_shells")}
    problems = []
    if h["open_edges"]:
        problems.append(f"{h['open_edges']} open edge(s)")
    if h["nonmanifold_edges"]:
        problems.append(f"{h['nonmanifold_edges']} edge(s) with more than 2 faces")
    if h["winding_conflicts"]:
        problems.append(f"{h['winding_conflicts']} edge(s) where the winding disagrees")
    if h["volume_mm3"] <= 0:
        problems.append(f"the volume is {_num(h['volume_mm3'])} mm3")
    elif h["inside_out"]:
        problems.append("the normals point inward (the volume is negative as written)")
    if h["inverted_shells"]:
        problems.append(f"{h['inverted_shells']} shell(s) inside out (a closed shell with inward normals that lies in "
                        "no other shell)")
    if problems:
        state, reason = "fail", "after the float32 reload: " + "; ".join(problems)
    else:
        state = "pass"
        reason = "after the float32 reload every edge joins 2 faces, the winding agrees and the volume is positive"
    if h["zero_area_faces"]:
        reason += f"; {h['zero_area_faces']} zero-area face(s) (a fact, not a limit)"
    if h["degenerate_shells"]:
        reason += f"; {h['degenerate_shells']} shell(s) with no area are not counted as bodies (a fact)"
    return _check("mesh_closed", value, _limit("closed manifold, positive volume", None, "definition"), state, reason,
                  "edge counts on the STL as read (float32)")


def check_bodies(h: dict, settings: PrintSettings, origin: dict) -> dict:
    count = len(h["bodies"])
    limit_value = settings.limit("max_bodies")
    source = key_source(settings, origin, "max_bodies")
    value = {"count": count, "shells": h["shells"], "largest": h["bodies"][:TOP_SHELLS]}
    lim = _limit(limit_value, "max_bodies", source)
    if count <= limit_value:
        state, reason = "pass", f"{count} {'body' if count == 1 else 'bodies'}; the limit is {limit_value} ({source})"
    else:
        sizes = ", ".join(_num(b["volume_mm3"]) for b in h["bodies"][:5])
        state, reason = "fail", (f"{count} bodies; the limit is {limit_value} ({source}). Volumes in mm3, largest "
                                 f"first: {sizes}{', ...' if count > 5 else ''}")
        if h["inverted_shells"]:
            reason += f". {h['inverted_shells']} of them inside out (see mesh_closed)"
    return _check("bodies", value, lim, state, reason, "shells of faces joined along edges")


def check_closed_voids(h: dict, settings: PrintSettings, origin: dict) -> dict:
    voids = h["voids"]
    total = round(sum(v["volume_mm3"] for v in voids), 3)
    largest = voids[0]["volume_mm3"] if voids else 0.0
    value = {"count": len(voids), "total_mm3": total, "largest_mm3": largest, "voids": voids[:TOP_SHELLS]}
    limit_value = settings.limit("max_closed_void_mm3")
    source = key_source(settings, origin, "max_closed_void_mm3")
    lim = _limit(limit_value, "max_closed_void_mm3", source, compared="the largest void")
    note = VOID_NOTES[settings.printer_type]
    if not voids:
        state, reason = "pass", "no closed void"
    elif limit_value is None:
        state, reason = "unknown", (f"{len(voids)} closed void(s), {_num(total)} mm3 in all, the largest "
                                    f"{_num(largest)} mm3. No max_closed_void_mm3 in the print block, so there is "
                                    f"no limit. {note}. A limit of 0 makes any closed void a fail")
    elif largest > limit_value:
        state, reason = "fail", (f"the largest closed void is {_num(largest)} mm3; the limit is "
                                 f"{_num(limit_value)} mm3 ({source}). {note}")
    else:
        state, reason = "pass", (f"{len(voids)} closed void(s), the largest {_num(largest)} mm3; the limit is "
                                 f"{_num(limit_value)} mm3 ({source})")
    return _check("closed_voids", value, lim, state, reason,
                  "closed shells with a negative volume that lie inside a closed shell with a positive volume")


def check_bed_fit(entry: dict, settings: PrintSettings, origin: dict) -> dict:
    from .print_report import fit_phrase

    keys = ("extents_mm", "height_mm", "bed_fit", "bed_turn_deg", "bed_footprint_mm", "bed_margin_mm")
    value = {k: entry[k] for k in keys}
    value["build_direction_label"] = entry["label"]
    source = key_source(settings, origin, "bed_mm")
    extra = {}
    if "height_fit" in entry:      # the printer profile states a build height (CONTRACTS.md section 18.7)
        value["height_fit"], value["height_margin_mm"] = entry["height_fit"], entry["height_margin_mm"]
        extra = {"build_height_mm": settings.build_height_mm, "build_height_source": f"profile:{settings.printer_model}"}
    lim = _limit(list(settings.bed_mm), "bed_mm", source, **extra)
    build = f"build direction {entry['label']} ({key_source(settings, origin, 'build_direction')})"
    state = "pass" if entry["bed_fit"] else "fail"
    reason = f"with {build} the part {fit_phrase(entry)} on a {_num(settings.bed_mm[0])} x " \
             f"{_num(settings.bed_mm[1])} mm bed ({source})"
    if extra:
        reason += (f"; the part is {_num(entry['height_mm'])} mm high and the build height is {_num(settings.build_height_mm)} mm "
                   f"({extra['build_height_source']})")
    return _check("bed_fit", value, lim, state, reason, "V2 bed fit: extents in the build frame, turn about the build axis")


def check_overhang(entry: dict, settings: PrintSettings, origin: dict) -> dict:
    area = entry["overhang_area_mm2"]
    limit_value = settings.limit("max_overhang_area_mm2")
    source = key_source(settings, origin, "max_overhang_area_mm2")
    angle_source = key_source(settings, origin, "self_supporting_angle_deg") \
        if settings.self_supporting_angle_deg is not None else f"default of printer_type {settings.printer_type}"
    value = {"area_mm2": area, "fraction": entry["overhang_fraction"], "angle_deg": settings.angle_deg,
             "angle_source": angle_source, "build_direction_label": entry["label"]}
    lim = _limit(limit_value, "max_overhang_area_mm2", source)
    fact = f"{_num(area)} mm2 of overhang ({_num(100 * entry['overhang_fraction'])} % of the area) below " \
           f"{_num(settings.angle_deg)} degrees from the plate ({angle_source})"
    if limit_value is not None:
        state = "pass" if area <= limit_value else "fail"
        reason = f"{fact}; the limit is {_num(limit_value)} mm2 ({source})"
    elif area == 0:
        state, reason = "pass", "no overhang face"
    else:
        state, reason = "unknown", f"{fact}. No max_overhang_area_mm2 in the print block, so there is no limit"
    return _check("overhang", value, lim, state, reason, "V2 overhang rule n.b < -cos(angle)")


def check_support_volume(entry: dict, settings: PrintSettings, origin: dict) -> dict:
    vol = entry["support_volume_mm3"]
    limit_value = settings.limit("max_support_volume_mm3")
    source = key_source(settings, origin, "max_support_volume_mm3")
    value = {"volume_mm3": vol, "rays": entry["support_rays"], "support_needed": settings.support_needed}
    lim = _limit(limit_value, "max_support_volume_mm3", source)
    if vol is None:
        state, reason = "pass", (f"printer_type {settings.printer_type} needs no support: the powder bed holds the part")
    elif limit_value is not None:
        fact = f"the support estimate is {_num(vol)} mm3 (a solid-column upper bound)"
        state = "pass" if vol <= limit_value else "fail"
        reason = f"{fact}; the limit is {_num(limit_value)} mm3 ({source})"
    elif vol == 0:
        state, reason = "pass", "no overhang face needs support"
    else:
        state, reason = "unknown", (f"the support estimate is {_num(vol)} mm3 (a solid-column upper bound; real "
                                    "supports use less). No max_support_volume_mm3 in the print block, so there "
                                    "is no limit")
    return _check("support_volume", value, lim, state, reason, "V2 support estimate, gap definition", estimate=True)


def check_plate_contact(entry: dict, settings: PrintSettings, origin: dict) -> dict:
    area = entry["plate_contact_area_mm2"]
    limit_value = settings.limit("min_plate_contact_mm2")
    source = key_source(settings, origin, "min_plate_contact_mm2")
    value = {"area_mm2": area, "build_direction_label": entry["label"]}
    lim = _limit(limit_value, "min_plate_contact_mm2", source)
    if limit_value is not None:
        state = "pass" if area >= limit_value else "fail"
        reason = f"{_num(area)} mm2 rests on the plate; the limit is {_num(limit_value)} mm2 ({source})"
    else:
        state, reason = "unknown", (f"{_num(area)} mm2 rests on the plate. No min_plate_contact_mm2 in the print "
                                    "block, so there is no limit")
    return _check("plate_contact", value, lim, state, reason, "V2 plate contact")


def resolution_of(inp: dict, resolution_mm, edge_mm: float) -> dict:
    """The size of the smallest member that the mesh can hold, and where the number comes from.

    An option wins. A run folder gives its voxel size `h` (run.json), but not less than the median mesh edge.
    A bare STL gives the median mesh edge. The mesh does not say if it came from a voxel grid, so a bare STL
    of exact CAD (big flat triangles) reads as coarse: say `resolution_mm=0` for it.
    """
    h = None
    if inp.get("run") and isinstance(inp["run"].get("h"), (int, float)) and inp["run"]["h"] > 0:
        h = float(inp["run"]["h"])
    if resolution_mm is not None:
        r, source = float(resolution_mm), "option"
    elif h is not None:
        r, source = max(h, edge_mm), "voxel size of the run (run.json), or the median mesh edge if that is larger"
    else:
        r, source = edge_mm, "median mesh edge (a bare STL has no voxel size)"
    return {"resolution_mm": round(r, 4), "source": source, "voxel_size_mm": h,
            "mesh_edge_median_mm": round(edge_mm, 4), "elements_across": MIN_ELEMENTS_ACROSS,
            "resolves_from_mm": round(MIN_ELEMENTS_ACROSS * r, 4)}


def _resolution_rule(res: dict) -> str:
    """The rule of this check for a pass, as a fact. (It is a rule of this check, not a fact of the mesh.)"""
    if res["source"] == "option" and res["resolution_mm"] == 0:
        return ("The option says the mesh is exact (resolution 0), so the rule of this check (a pass needs a limit of "
                f"at least {MIN_ELEMENTS_ACROSS} x the resolution) gives no floor")
    return (f"The rule of this check: a pass needs a limit of at least {MIN_ELEMENTS_ACROSS} x the resolution "
            f"({_num(res['resolution_mm'])} mm, {res['source']}), here {_num(res['resolves_from_mm'])} mm "
            "(see pitfalls/mesh-coarser-than-members)")


def _analysis_note(res: dict) -> str:
    """For a run folder: the voxel size is the resolution of the stress analysis, so thin members were under-resolved there."""
    h = res.get("voxel_size_mm")
    if h is None:
        return ""
    return (f". A member under {_num(MIN_ELEMENTS_ACROSS * h)} mm ({MIN_ELEMENTS_ACROSS} x the voxel size {_num(h)} mm "
            "of this run) was under-resolved in the analysis of the run")


def _round_down(x: float, places: int = 4) -> float:
    return math.floor(x * 10 ** places) / 10 ** places


def _exact(x: float) -> str:
    """The number as it is stored, with at most 4 decimals and no trailing zeros.

    `_num` rounds to 4 significant digits, and that can round up. A text must not name a value that is larger than the value in the result.
    """
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _pass_needs(limit_value: float, res: dict) -> tuple[dict, str]:
    """What a `pass` of the thickness check would need when the limit is under the floor of the rule (P4, F4-3).

    The way is a resolution of at most `limit / 3`, rounded DOWN to 4 decimals.
    The resolution of the result has 4 decimals, and a value rounded up gives `unknown` again (limit 2.0 mm gave 0.6667).
    The key and the text have the same value. The limit is the wall that the user needs, so it is not a knob.
    For a run folder the resolution is a voxel size, and the number of voxels grows with its third power.
    So the factor is an estimate that says whether that way is open. It is a fact about the grid, not a limit of the tool.
    The caller has found the measured share within the tolerance (a share over it is a `fail` at any resolution).
    So the rule alone holds the state back for this STL. A run at a finer voxel size makes a new shape.
    """
    target = _round_down(limit_value / MIN_ELEMENTS_ACROSS)
    while MIN_ELEMENTS_ACROSS * target > limit_value and target > 0:      # a float edge: stay on the side of the pass
        target = round(target - 1e-4, 4)
    shown = _exact(target)
    needs = {"resolution_mm_at_most": target}
    h = res.get("voxel_size_mm")
    edge = _num(res["mesh_edge_median_mm"])
    sentences = [f"A pass needs a resolution of at most {shown} mm (the limit divided by {MIN_ELEMENTS_ACROSS})."]
    if res["source"] == "option":
        sentences += [f"Give resolution_mm of {shown} or less (now {_num(res['resolution_mm'])}).",
                      "The measured share of this STL is within the tolerance, so the state is then pass."]
    elif h is not None and h >= res["mesh_edge_median_mm"]:                # the voxel size sets the resolution (`resolution_of`)
        needs["voxel_count_factor"] = round((h / target) ** 3, 1) if target > 0 else None
        sentences += [f"For this run folder that is a voxel size of {shown} mm or less.",
                      f"That is about {_num(needs['voxel_count_factor'])} times the voxels of the {_num(h)} mm grid "
                      f"(an estimate: ({_num(h)} / {shown})^3).",
                      "A run at that voxel size makes a new shape. The state is pass there if its measured share stays within "
                      "the tolerance.",
                      f"For this STL you can give resolution_mm of {shown} or less as an option. Give it only if the mesh "
                      f"holds members that thin (its median edge is {edge} mm).",
                      "The measured share of this STL is within the tolerance, so the state is then pass."]
    else:                                                                   # the median mesh edge sets it
        sentences += [f"That is a mesh with a median edge of {shown} mm or less (now {edge} mm).",
                      f"Or give resolution_mm of {shown} or less as an option. Give it only if the mesh holds members that "
                      "thin (for example exact CAD with big flat faces).",
                      "The measured share of this STL is within the tolerance, so the state is then pass."]
    return needs, " ".join(sentences).rstrip(".")      # the caller adds the next sentence with its own separator


def check_wall_thickness(thin: dict, chords: np.ndarray, total_area: float, settings: PrintSettings, res: dict,
                         closed: bool, origin: dict) -> dict:
    """The thickness check. `thin` is the V2 `thinnest_member` dict and `chords` its chord lengths (mm).

    A chord is thinner than the limit if it is under `limit - CHORD_TOL_MM`: a wall that is exactly at the limit
    reads a few 1e-6 mm short on some positions and turns (float32), and it is not thinner than the limit.
    """
    from .print_report import THICKNESS_MAX_MM

    limit_value = settings.limit("min_wall_mm")
    tol = settings.limit("min_wall_area_pct")
    source = key_source(settings, origin, "min_wall_mm")
    lim = _limit(limit_value, "min_wall_mm", source, tolerance_pct=tol, tolerance_key="min_wall_area_pct",
                 tolerance_source=key_source(settings, origin, "min_wall_area_pct"), chord_tolerance_mm=CHORD_TOL_MM)
    keys = ("thinnest_member_mm", "min_mm", "samples", "percentiles_mm", "area_below_mm2", "area_below_pct",
            "percentile", "max_chord_mm")
    value = {k: thin[k] for k in keys}
    value["resolution"] = res
    method = "V2 chord estimate along the inward face normal on the STL mesh"
    p1 = thin["thinnest_member_mm"]
    facts = f"p{thin['percentile']:g} of the chords is {_num(p1)} mm" if p1 is not None else "no chord"
    tail = f"{_analysis_note(res)}. This is the printed shape, not the stress"
    if not closed:
        return _check("wall_thickness", value, lim, "unknown",
                      "the mesh is not closed, so the inward direction is not defined; the chords are not a "
                      f"measure. {facts}", method, True)
    if limit_value is None:
        below = thin["area_below_mm2"]
        area_facts = (f"; the area with a chord under 0.8 / 1.2 / 2.0 mm is {below['0.8']} / {below['1.2']} / "
                      f"{below['2.0']} mm2") if below else ""
        return _check("wall_thickness", value, lim, "unknown",
                      f"no min_wall_mm in the print block, so there is no limit. {facts}{area_facts}. "
                      f"{_resolution_rule(res)}{tail}", method, True)
    if not thin["samples"]:
        return _check("wall_thickness", value, lim, "unknown", "no chord could be measured on this mesh", method, True)
    if limit_value > THICKNESS_MAX_MM:
        return _check("wall_thickness", value, lim, "unknown",
                      f"the limit {_num(limit_value)} mm is above the chord cap of {THICKNESS_MAX_MM:g} mm, so the "
                      f"chords cannot show a wall that thin. {facts}", method, True)
    n = len(chords)
    share = float((chords < limit_value - CHORD_TOL_MM).mean())
    se_pct = 100.0 * float(np.sqrt(share * (1.0 - share) / n)) if n else 0.0
    area = round(share * total_area, 2)
    pct = round(100.0 * share, 3)
    value.update({"area_below_limit_mm2": area, "area_below_limit_pct": pct,
                  "area_below_limit_sampling_error_pct": round(se_pct, 3)})
    head = (f"{_num(area)} mm2 ({_num(pct)} % of the area, sampling error about {se_pct:.2g} points) has a chord "
            f"under {_num(limit_value)} mm; the tolerance is {_num(tol)} % ({lim['tolerance_source']}); {facts}")
    if abs(pct - tol) < 2.0 * se_pct:
        head += ("; the share is within 2 sampling errors of the tolerance, so another sample can give another "
                 "state")
    if pct > tol:
        return _check("wall_thickness", value, lim, "fail", f"{head}{tail}", method, True)
    if limit_value < MIN_ELEMENTS_ACROSS * res["resolution_mm"]:
        needs, needs_text = _pass_needs(limit_value, res)
        value["pass_needs"] = needs
        return _check("wall_thickness", value, lim, "unknown",
                      f"{head}. The limit {_num(limit_value)} mm is under {MIN_ELEMENTS_ACROSS} x the resolution "
                      f"{_num(res['resolution_mm'])} mm ({res['source']}): by the rule of this check (see "
                      f"pitfalls/mesh-coarser-than-members) this is no pass. {needs_text}{tail}", method, True)
    return _check("wall_thickness", value, lim, "pass", f"{head}. {_resolution_rule(res)}{tail}", method, True)


# ---------------------------------------------------------------------------
# the gate and the report
# ---------------------------------------------------------------------------


def make_gate(checks: list, strict: bool) -> dict:
    """The strict gate. It fails only when a check that has a limit fails; `unknown` never sets it."""
    failed = [c["id"] for c in checks if c["state"] == "fail"]
    keys = [c["limit"]["key"] for c in checks if c["state"] == "fail" and c["limit"].get("key")]
    gate = {"strict": bool(strict), "state": "off", "exit_code": 0, "failed": failed, "override": None}
    if not strict:
        gate["note"] = (f"Report only. With --strict (or strict=true) a failed check sets gate.state to fail and "
                        f"the exit code {GATE_EXIT_CODE}")
        return gate
    gate["state"] = "fail" if failed else "pass"
    if failed:
        gate["exit_code"] = GATE_EXIT_CODE
        change = f" or change the limit in the print block ({', '.join(keys)})" if keys else ""
        fix = (" mesh_closed has no print key; to go on, run without --strict, or repair the mesh."
               if "mesh_closed" in failed else "")
        gate["override"] = ("No run stops on this gate. To go on, run without --strict (strict=false in the "
                            "MCP tool)" + change + "." + fix)
    return gate


def _settings_section(settings: PrintSettings, sources: list, origin: dict, entry: dict) -> dict:
    from .print_report import ASSUMABLE_KEYS

    angle_source = key_source(settings, origin, "self_supporting_angle_deg") \
        if settings.self_supporting_angle_deg is not None else f"default of printer_type {settings.printer_type}"
    out = {
        "build_direction": entry["direction"], "build_direction_label": entry["label"],
        "bed_mm": list(settings.bed_mm), "printer_type": settings.printer_type,
        "self_supporting_angle_deg": settings.angle_deg, "angle_source": angle_source,
        "limits": {k: {"value": settings.limit(k), "source": key_source(settings, origin, k)}
                   for k in PRINT_LIMIT_KEYS},
        "assumed": [k for k in ASSUMABLE_KEYS if k not in settings.given and k not in settings.derived],
        "from": sources,
    }
    if settings.profile is not None:   # a check with no printer profile keeps its settings as they were
        from .print_report import profile_settings
        out.update(profile_settings(settings))
    return out


def check(path, *, yaml_path=None, print_settings=None, resolution_mm=None, strict: bool = False,
          contain_errors: bool = False) -> dict:
    """The print check of an STL file or a run folder. Returns the report (CONTRACTS.md section 32).

    `yaml_path`: a problem YAML whose `print` block gives the settings and limits. `print_settings`: a dict with
    keys of the `print` block (it wins over the YAML). `resolution_mm`: the size of the smallest member that the
    mesh holds (the voxel size; 0 for exact CAD). `strict`: opt in to the gate. Raises `UserError` for an input
    that cannot be read; a failed check is data, not an error.

    `contain_errors`: a stage or a check that stops with an exception gives a record with the state `unknown` and the key `error`, and
    the other checks go on; `coverage.failed` names it. Without it (the default) the exception passes to the caller.
    """
    t_all = time.perf_counter()
    inp = resolve_input(path)
    _check_resolution(resolution_mm)
    settings, sources, origin = assemble_settings(inp, yaml_path, print_settings)
    stl = inp["stl"]
    data = stl.read_bytes()

    def stage(fn, name):
        """`(result, None)`. With `contain_errors` an exception gives `(None, error text)`; without it the exception passes."""
        if not contain_errors:
            return fn(), None
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - recorded as `error` in the check records, see `contain_errors`
            return None, _error_text(exc, name)

    t0 = time.perf_counter()
    V, F = load_stl_float32(stl)
    health_edge, health_err = stage(lambda: (mesh_health(V, F), mesh_edge_median(V, F)), "mesh health")
    health, edge = health_edge or (None, None)
    t_health = time.perf_counter() - t0

    import trimesh

    from . import print_report as pr

    t0 = time.perf_counter()
    b0 = pr._unit(settings.build_direction)

    def measure():
        geometry = pr._Geometry(trimesh.Trimesh(V, F, process=False))
        return geometry, pr.measure_direction(geometry, b0, settings.angle_deg, settings.bed_mm, settings.support_needed,
                                              build_height_mm=settings.build_height_mm)
    geo, geo_err = stage(measure, "build-direction numbers")
    g, entry = geo or (None, None)
    t_v2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    if geo_err:
        thin_chords, thin_err = None, geo_err
    else:
        thin_chords, thin_err = stage(lambda: pr.thickness_estimate(g, return_chords=True), "thickness estimate")
    thin, chords = thin_chords or (None, None)
    t_thin = time.perf_counter() - t0

    def one(cid, stage_error, fn):
        """The record of one check. A stage that it needs stopped, or the check itself raises (`contain_errors`): an error record."""
        if stage_error:
            return _error_check(cid, stage_error)
        record, err = stage(fn, None)
        return record if err is None else _error_check(cid, err)

    wall_error = health_err or thin_err
    checks = [
        one("mesh_closed", health_err, lambda: check_mesh_closed(health)),
        one("bodies", health_err, lambda: check_bodies(health, settings, origin)),
        one("closed_voids", health_err, lambda: check_closed_voids(health, settings, origin)),
        one("bed_fit", geo_err, lambda: check_bed_fit(entry, settings, origin)),
        one("overhang", geo_err, lambda: check_overhang(entry, settings, origin)),
        one("support_volume", geo_err, lambda: check_support_volume(entry, settings, origin)),
        one("plate_contact", geo_err, lambda: check_plate_contact(entry, settings, origin)),
        one("wall_thickness", wall_error, lambda: check_wall_thickness(
            thin, chords, g.total_area, settings, resolution_of(inp, resolution_mm, edge), health["watertight"], origin)),
    ]
    counts = {s: sum(1 for c in checks if c["state"] == s) for s in STATES}
    # a build-direction stage that stopped: the settings section still names the direction (it needs no number of that stage)
    frame = entry or {"direction": [float(v) + 0.0 for v in b0], "label": pr.direction_label(b0)}
    report = {
        "ok": True,
        "input": {"kind": inp["kind"], "stl": str(stl), "run_dir": str(inp["run_dir"]) if inp["run_dir"] else None,
                  "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                  "triangles": health["triangles"] if health else int(len(F)),
                  "total_area_mm2": round(g.total_area, 3) if g is not None else None},
        "settings": _settings_section(settings, sources, origin, frame),
        "checks": checks,
        "summary": {**counts, "failed": [c["id"] for c in checks if c["state"] == "fail"],
                    "unknown_ids": [c["id"] for c in checks if c["state"] == "unknown"]},
        "gate": make_gate(checks, strict),
        "definitions": DEFINITIONS if settings.profile is None else {**DEFINITIONS, "height_fit": _V2_HEIGHT_FIT},
        "note": ("Geometry only: no solve. A print pass is not a structural pass: the stress and the safety factor "
                 "are separate checks. The thickness and the support volume are estimates. Nothing here stops a run."),
    }
    if g is not None and g.inside_out:
        report["warning"] = "the mesh has negative volume: the normals were flipped for the V2 numbers"
    report["timing_s"] = {"mesh_health": round(t_health, 2), "v2_numbers": round(t_v2, 2),
                          "thickness": round(t_thin, 2)}
    report["elapsed_s"] = round(time.perf_counter() - t_all, 2)
    return check_coverage.attach(report, check_coverage.from_printcheck(checks), after="ok")


def _derived_text(s: dict) -> str:
    """The keys that a profile filled. Each key is named with its printer when it is not the printer of the report."""
    own = f"profile:{s['printer_model']}"
    return ", ".join(k if label == own else f"{k} ({label})" for k, label in s["derived"].items())


def format_text(report: dict) -> str:
    """The report as text lines for the terminal. Line 2 is the `partial` line (TC-2) when the report has the keys."""
    i, s = report["input"], report["settings"]
    area = f"area {_num(i['total_area_mm2'])} mm2" if i["total_area_mm2"] is not None else "area not measured"
    lines = [f"print check: {i['stl']}",
             f"  {i['kind']}, {i['triangles']:,} triangles, {area}, "
             f"sha256 {i['sha256'][:12]}",
             f"  settings from: {', '.join(s['from'])}; build direction {s['build_direction_label']}, bed "
             f"{_num(s['bed_mm'][0])} x {_num(s['bed_mm'][1])} mm, printer {s['printer_type']}, angle "
             f"{_num(s['self_supporting_angle_deg'])} deg ({s['angle_source']})"
             + (f"; assumed: {', '.join(s['assumed'])}" if s["assumed"] else "")]
    if s.get("printer_model"):
        lines.append(f"  printer: {s['printer_name']} (profile {s['printer_model']}, {s['profile_from']})"
                     + (f"; build height {_num(s['build_height_mm'])} mm" if s.get("build_height_mm") is not None
                        else "; the profile does not state a build height, so the height is not limited")
                     + (f"; from the profile: {_derived_text(s)}" if s.get("derived") else ""))
    for c in report["checks"]:
        lines.append(f"  {c['state']:<8} {c['id']:<15} {c['reason']}")
    sm = report["summary"]
    lines.append(f"summary: {sm['pass']} pass, {sm['fail']} fail, {sm['unknown']} unknown"
                 + (f"; failed: {', '.join(sm['failed'])}" if sm["failed"] else ""))
    g = report["gate"]
    if g["state"] == "off":
        lines.append(f"gate: off. {g['note']}")
    elif g["state"] == "pass":
        lines.append("gate: pass (strict)")
    else:
        lines.append(f"gate: FAIL (strict), exit code {g['exit_code']}. Failed: {', '.join(g['failed'])}. {g['override']}")
    lines.append(f"time {report['elapsed_s']} s (mesh health {report['timing_s']['mesh_health']} s, V2 numbers "
                 f"{report['timing_s']['v2_numbers']} s, thickness {report['timing_s']['thickness']} s)")
    cov = check_coverage.line_of(report)
    if cov:      # TC-2: line 2, as in the text of `topopt verify`
        lines.insert(1, cov)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the automatic check at the end of a run (CONTRACTS.md section 32, "After every run")
# ---------------------------------------------------------------------------
REPORT_FILE = "print_check.json"   # result/print_check.json: the report of `check` for the run folder, made by the post step


def summary_of(report: dict | None) -> dict | None:
    """The short form of a report for `run_result` and the run summary: `ok`, the counts, the names of the failed checks and of the
    `unknown` checks that have a limit (`unknown_with_limit`: the limit is set and the tool cannot say). `None` for no report.
    A report that failed has `ok: false` and `error`."""
    if not isinstance(report, dict):
        return None
    if not report.get("ok", False):
        return {"ok": False, "error": report.get("error")}
    s = report.get("summary") or {}
    checks = report.get("checks") or []
    return {"ok": True, "pass": s.get("pass"), "fail": s.get("fail"), "unknown": s.get("unknown"),
            "failed": list(s.get("failed") or []),
            "unknown_with_limit": [c["id"] for c in checks
                                   if c.get("state") == "unknown" and (c.get("limit") or {}).get("value") is not None],
            "partial": report.get("partial")}


def summary_line(summary: dict | None) -> str | None:
    """One line of text for the run log and the CLI summary, from `summary_of`."""
    if summary is None:
        return None
    if not summary["ok"]:
        return f"print check: not made ({summary['error']}); `topopt printcheck <run folder>` repeats it"
    text = f"print check: {summary['pass']} pass, {summary['fail']} fail, {summary['unknown']} unknown"
    if summary["failed"]:
        text += f"; failed: {', '.join(summary['failed'])}"
    if summary["unknown_with_limit"]:
        text += f"; unknown with a limit: {', '.join(summary['unknown_with_limit'])}"
    return text + f" ({REPORT_FILE})"


def write_for_run(run_dir) -> dict:
    """The post step calls this once, last: it runs `check` on the run folder (its `run.json` `print` block, no YAML) and writes
    `result/print_check.json`. It reports and gates nothing. A fault gives `{"ok": false, "error": ...}` in the file, as the V2
    section does (`print_report.section_for_run`); this function never raises. Returns the report (or the fault record).

    The old file is removed first, so a report that cannot be written never leaves the states of an earlier post step as current.
    The text is made with the call of `topopt printcheck --json` (`json.dumps(report, indent=1)`), so the file and the direct output are equal.
    """
    run_dir = Path(run_dir)
    out = run_dir / "result" / REPORT_FILE
    tmp = out.with_name(out.name + ".tmp")
    try:
        out.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        report = check(run_dir, contain_errors=True)
        text = json.dumps(report, indent=1)
    except Exception as exc:  # noqa: BLE001 - the run is finished; a fault is a fact in the file
        report = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        text = json.dumps(report, indent=1)
    try:
        tmp.write_text(text)
        tmp.replace(out)
    except Exception as exc:  # noqa: BLE001 - a file that cannot be written is not an error of the run
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return report
