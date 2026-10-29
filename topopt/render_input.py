"""Input checks for `topopt render` (CONTRACTS.md section 15). Standard library only.

The CLI and the MCP tool call `check_inputs` before they import `topopt.render`, so a wrong
input gives a one-line error at once, without loading numba, Pillow or manifold3d.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from .config import UserError

ROLES = ("part", "design", "fixed", "load", "keep-out")
YAML_ROLES = ("interface", "cut")   # drawn only for a problem YAML: a union interface shape, and a cut interface shape
YAML_SUFFIXES = (".yaml", ".yml")
NAMED_VIEWS = {"iso": (1, -1, 0.8), "front": (0, -1, 0), "back": (0, 1, 0), "right": (1, 0, 0),
               "left": (-1, 0, 0), "top": (0, 0, 1), "bottom": (0, 0, -1)}
DEFAULT_VIEWS = ("iso", "front", "right", "top")
DEFAULT_SIZE = (640, 480)         # one tile, pixels

REGION_MODES = ("ghost", "solid", "off")
NAME_OK = re.compile(r"[A-Za-z0-9_-]+")


def parse_section(section):
    """Parse the section text. Returns `(axis index, value, side)`. Raises UserError.

    `x=30` and `x>=30` keep the side with x >= 30 (`side` +1). `x<=30` and `x<30` keep x <= 30 (`side` -1).
    The value is a finite number in mm.
    """
    text = str(section).replace(" ", "").lower()
    m = re.fullmatch(r"([xyz])(>=|<=|=|>|<)(.+)", text)
    try:
        if not m:
            raise ValueError
        value = float(m.group(3))
        if not math.isfinite(value):
            raise ValueError
    except ValueError:
        raise UserError("section must be AXIS=VALUE (keeps AXIS >= VALUE) or AXIS<=VALUE (keeps AXIS <= VALUE), "
                        f"with AXIS x, y or z and a finite VALUE in mm, for example y=0 or x<=30. Got: {section!r}")
    return "xyz".index(m.group(1)), value, (-1 if m.group(2) in ("<=", "<") else 1)


def section_views(section):
    """Default views for a section: the cut face from the removed side, an iso view from that side,
    and two side views of the kept part. Returns a list of `(name, direction)`."""
    axis, _, side = parse_section(section)
    cut = [0.0, 0.0, 0.0]
    cut[axis] = -float(side)                       # the cut face points to the removed side
    iso = list(NAMED_VIEWS["iso"])
    iso[axis] = cut[axis] * abs(iso[axis])
    sides = {0: ("front", "top"), 1: ("right", "top"), 2: ("front", "right")}[axis]
    return [("cut", tuple(cut)), ("iso-cut", tuple(iso))] + [(n, NAMED_VIEWS[n]) for n in sides]


def parse_size(size):
    """`"640x480"` or `(640, 480)` -> `(w, h)`. Raises UserError."""
    if size is None:
        return DEFAULT_SIZE
    try:
        w, h = (int(x) for x in (size.lower().split("x") if isinstance(size, str) else size))
    except (ValueError, TypeError):
        raise UserError(f"size must be WxH in pixels, for example 640x480. Got: {size!r}")
    if not (64 <= w <= 4000 and 64 <= h <= 4000):
        raise UserError(f"size must be between 64 and 4000 pixels per side. Got: {w}x{h}")
    return w, h


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def split_views(text: str) -> list[str]:
    """Split a text of views into one text for each view.

    With a `;` in the text, the views are separated by `;` (the first rule of this tool: it works for every list).
    Without one, the separator is `,`, and a `NAME:dx,dy,dz` view keeps its two next numbers: the commas inside it
    are not separators. So `fl:-0.5,-1,0.25` is one view, and `iso,fl:-0.5,-1,0.25,top` is three. A custom view with
    fewer than three numbers stays as it is. The check of the view then says what is wrong with it.
    """
    if ";" in text:
        return text.split(";")
    parts, out, i = text.split(","), [], 0
    while i < len(parts):
        piece = parts[i]
        i += 1
        if ":" in piece:
            for _ in range(2):                       # the second and the third number of the direction
                if i < len(parts) and _is_number(parts[i].strip()):
                    piece += "," + parts[i]
                    i += 1
        out.append(piece)
    return out


def parse_views(views, section=None):
    """A list of names, or `NAME:dx,dy,dz` (direction from the part to the camera, z up).

    Names: letters, digits, `_` and `-` only (they become file names). With a `section`, the names `cut` and
    `iso-cut` mean the views of `section_views`, and the default views are those of `section_views`.
    `views` is a text (`split_views`) or a list of such texts (each item may also hold a comma list).
    Returns a list of `(name, direction)`. Raises UserError.
    """
    sec = section_views(section) if section else []
    if views is None:
        return sec if sec else [(n, NAMED_VIEWS[n]) for n in DEFAULT_VIEWS]
    views = split_views(views) if isinstance(views, str) else [w for v in views for w in split_views(str(v))]
    named = dict(NAMED_VIEWS)
    named.update(dict(sec))
    out = []
    for v in views:
        v = str(v).strip()
        if not v:
            continue
        if ":" in v:
            name, vec = v.split(":", 1)
            name = name.strip()
            try:
                d = tuple(float(x) for x in vec.split(","))
                assert len(d) == 3 and any(d) and all(math.isfinite(x) for x in d)
            except (ValueError, AssertionError):
                raise UserError(f"view {v!r}: use NAME:dx,dy,dz (three finite numbers, not all zero)")
        elif v in named:
            name, d = v, named[v]
        elif v in ("cut", "iso-cut"):
            raise UserError(f"the view {v!r} needs a section (section=AXIS=VALUE)")
        else:
            raise UserError(f"unknown view {v!r}. Named views: {', '.join(NAMED_VIEWS)}"
                            f"{', cut, iso-cut (with a section)' if not sec else ', cut, iso-cut'}; or NAME:dx,dy,dz")
        if not NAME_OK.fullmatch(name):
            raise UserError(f"view name {name!r}: use letters, digits, _ and - only")
        out.append((name, d))
    if not out:
        raise UserError("no view given")
    return out


def _run_inputs(run_dir: Path):
    """Return `(items, missing, notes)` for a run folder. items: list of (path, role)."""
    final = run_dir / "result" / "final.stl"
    if not final.is_file():
        raise UserError(f"no result/final.stl in {run_dir}: the run is not done, or this is not a run folder")
    items = [(final, "part")]
    missing, notes = [], []
    try:
        run = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        run = {}
    from .runinfo import run_yaml_path
    yaml_path, regions = run_yaml_path(run, run_dir), run.get("regions")   # a relative yaml_path starts at the run folder
    if not isinstance(regions, dict) or yaml_path is None:
        notes.append("run.json has no yaml_path or regions: only final.stl is drawn")
        return items, missing, notes
    base = yaml_path.parent

    def add(role, name):
        p = base / name
        if p.is_file():
            items.append((p, role))
        else:
            missing.append({"role": role, "file": str(name), "path": os.path.normpath(p)})

    if regions.get("design"):
        add("design", regions["design"])
    for name in regions.get("fixed") or []:
        add("fixed", name)
    for name in regions.get("keep_out") or []:
        add("keep-out", name)
    for name in (regions.get("load") or {}).values():
        add("load", name)
    for name in regions.get("solid") or []:   # a solid region is drawn like a load region (forced solid)
        add("load", name)
    return items, missing, notes


def role_list(text: str) -> list[str]:
    """The roles of one `--role` value: a role, or a comma list of roles (`design,fixed`). Raises UserError for an unknown role."""
    out = [r.strip() for r in str(text).split(",") if r.strip()]
    bad = [r for r in out if r not in ROLES]
    if bad or not out:
        raise UserError(f"unknown role {', '.join(repr(r) for r in bad) if bad else repr(text)}. Roles: {', '.join(ROLES)}")
    return out


def normalize_sources(sources, roles):
    """Return `(kind, run_dir or None, items, missing, notes)`. `kind` is `run`, `stl` or `yaml` (one problem YAML: `items` is `[(path, "yaml")]`)."""
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    sources = list(sources or [])
    roles = [r.strip() for item in (roles or []) for r in str(item).split(",") if r.strip()] or None   # a role item may be a comma list
    if not sources:
        raise UserError("no input: give STL files, a run folder or a problem YAML")
    if len(sources) == 1 and not isinstance(sources[0], (tuple, list)) and Path(sources[0]).suffix.lower() in YAML_SUFFIXES:
        p = Path(sources[0]).expanduser().resolve()
        if not p.is_file():
            raise UserError(f"file not found: {p}")
        if roles:
            raise UserError("roles do not apply to a problem YAML: the regions of the YAML give them")
        return "yaml", None, [(p, "yaml")], [], []
    if len(sources) == 1 and not isinstance(sources[0], (tuple, list)) and Path(sources[0]).expanduser().is_dir():
        run_dir = Path(sources[0]).expanduser().resolve()
        items, missing, notes = _run_inputs(run_dir)
        return "run", run_dir, items, missing, notes
    items = []
    for i, s in enumerate(sources):
        role = None
        if isinstance(s, (tuple, list)):
            s, role = s
        if role is None and roles:
            role = roles[0] if len(roles) == 1 else (roles[i] if i < len(roles) else None)
        role = role or "part"
        if role not in ROLES:
            raise UserError(f"unknown role {role!r}. Roles: {', '.join(ROLES)}")
        p = Path(s).expanduser().resolve()
        if p.is_dir():
            raise UserError(f"{p} is a folder. Give one run folder alone, or STL files")
        if not p.is_file():
            raise UserError(f"file not found: {p}")
        items.append((p, role))
    if roles and len(roles) not in (1, len(items)) and not any(isinstance(s, (tuple, list)) for s in sources):
        raise UserError(f"{len(roles)} roles for {len(items)} files: give one role, or one per file")
    return "stl", None, items, [], []


def check_inputs(sources, *, roles=None, views=None, section=None, size=None, regions=None,
                 max_triangles=None, loads=None) -> None:
    """Raise `UserError` for a wrong input. Reads no mesh."""
    parse_views(views, section)
    parse_size(size)
    check_options(regions, max_triangles, loads)
    normalize_sources(sources, roles)


def check_options(regions, max_triangles, loads=None) -> None:
    if loads is not None and not isinstance(loads, bool):
        raise UserError(f"loads must be true or false (false: no force arrows in a render of a problem YAML). Got: {loads!r}")
    if regions is not None and regions not in REGION_MODES:
        raise UserError(f"regions must be one of {', '.join(REGION_MODES)}. Got: {regions!r}")
    if max_triangles is not None and (isinstance(max_triangles, bool) or not isinstance(max_triangles, int)
                                      or max_triangles < 1):
        raise UserError(f"max_triangles must be a whole number of 1 or more. Got: {max_triangles!r}")
