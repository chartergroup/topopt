"""The problem view: a 3D view of a problem inside the chat of a client that shows MCP Apps (tool show_problem).

MCP Apps is the UI extension of MCP (spec 2026-01-26, extension id `io.modelcontextprotocol/ui`). A tool names a `ui://` resource in
`_meta.ui.resourceUri`. The client reads that resource (an HTML page, MIME type `text/html;profile=mcp-app`), shows it in a sandboxed
frame in the chat, and sends the tool result to the page. A client without MCP Apps shows only the text of the result.

The page is one HTML file with no network access: the default sandbox policy of the spec blocks every fetch, and allows inline scripts.
So `page()` puts three.js (the copy of the dashboard, `topopt/dashboard/vendor`) and the view script into the page itself.

What goes where in the result (the spec, and the ChatGPT Apps SDK, which follows it):
- `content`: a short text for the model and for a client without MCP Apps.
- `structuredContent`: the small facts of the view (names, counts, the estimate). The model may read them.
- `_meta[GEOMETRY_KEY]`: the meshes. Only the page reads `_meta`; no client puts it in the model context.
"""
from __future__ import annotations

import base64
import re
from functools import lru_cache
from pathlib import Path

URI = "ui://topopt/problem-view.html"
MIME_TYPE = "text/html;profile=mcp-app"      # the MIME type of an MCP Apps page (spec 2026-01-26)
NAME = "problem-view"
TITLE = "Problem view"
DESCRIPTION = "The 3D view of a problem (tool show_problem): the design space, the fixed, keep-out, solid and load regions, the load arrows and the estimate."
GEOMETRY_KEY = "topopt/problem-view"             # the key in the `_meta` of the result of show_problem
FORMAT = "topopt.problem_view/1"                 # the version of the data that the page reads

# The tool links the page in both forms of the spec: the nested key, and the flat key of the first clients (deprecated in the spec).
TOOL_META = {"ui": {"resourceUri": URI}, "ui/resourceUri": URI}
# No `csp`: the page needs no network (the default policy of the spec). `prefersBorder`: the page draws its own frame.
RESOURCE_META = {"ui": {"prefersBorder": False}}

_HERE = Path(__file__).resolve().parent
_THREE = _HERE.parent / "topopt" / "dashboard" / "vendor" / "three.module.min.js"
_TEMPLATE = _HERE / "problem_view.html"
_SLOT = "/*__THREE__*/"


def _three_as_object(module_text: str) -> str:
    """three.module.min.js as a plain script: the last statement `export{a as B,...};` becomes `const THREE={B:a,...};`.

    An inline module script cannot import another inline script, and the sandbox blocks `blob:` and `data:` scripts. So the module
    text goes into the same script as the view code, and its export list becomes one object.
    """
    text = module_text.rstrip()
    match = re.search(r"export\s*\{([^{}]*)\}\s*;?\s*$", text)
    if match is None:
        raise ValueError(f"{_THREE.name}: no export list at the end of the file")
    pairs = []
    for item in match.group(1).split(","):
        item = item.strip()
        if not item:
            continue
        local, _, public = item.partition(" as ")
        pairs.append(f"{(public or local).strip()}:{local.strip()}")
    return text[: match.start()] + "const THREE={" + ",".join(pairs) + "};"


@lru_cache(maxsize=1)
def page() -> str:
    """The HTML page of the card, with three.js inside it."""
    template = _TEMPLATE.read_text(encoding="utf-8")
    if _SLOT not in template:
        raise ValueError(f"{_TEMPLATE.name}: no {_SLOT} slot")
    three = _three_as_object(_THREE.read_text(encoding="utf-8"))
    return template.replace(_SLOT, three.replace("</script", "<\\/script"))


def _b64(array, dtype) -> str:
    import numpy as np   # here, not at import: the server imports this module (tests/test_render.py)
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def _mesh(path: Path, kind: str, name: str) -> dict:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    return {
        "kind": kind,
        "name": name,
        "file": path.name,
        "triangles": int(len(mesh.faces)),
        "positions": _b64(mesh.vertices, "<f4"),   # x, y, z of each vertex, mm, float32 little endian
        "indices": _b64(mesh.faces, "<u4"),        # three vertex indices for each triangle, uint32 little endian
        "center": [float(v) for v in mesh.bounds.mean(axis=0)],
    }


def regions(cfg) -> list[dict]:
    """The meshes of the problem, in drawing order: the design space first."""
    r = cfg.regions
    out = [_mesh(r.design, "design", r.design.name)]
    out += [_mesh(p, "fixed", p.name) for p in r.fixed]
    out += [_mesh(p, "solid", p.name) for p in r.solid]
    out += [_mesh(p, "keep_out", p.name) for p in r.keep_out]
    out += [_mesh(lr.path, "load", lr.name) for lr in r.load]
    return out


def load_cases(cfg, meshes: list[dict]) -> list[dict]:
    """Each load case with its forces. `center` is the centre of the bounds of the load region (the arrow tip, if the force has no `at`)."""
    centers = {m["name"]: m["center"] for m in meshes if m["kind"] == "load"}
    cases = []
    for lc in cfg.load_cases:
        forces = [{"region": f.region, "force": [float(v) for v in f.force],
                   "at": None if f.at is None else [float(v) for v in f.at], "center": centers.get(f.region)} for f in lc.forces]
        pressures = [{"region": p.region, "value_mpa": float(p.value_mpa), "center": centers.get(p.region)} for p in lc.pressures]
        cases.append({"name": lc.name, "weight": float(lc.weight), "forces": forces, "pressures": pressures})
    return cases


def estimate(check: dict) -> dict:
    """The numbers of the view from a reply of check_problem (`topopt.cli.check_data`)."""
    t = check.get("time_estimate") or {}
    m = check.get("memory_estimate") or {}
    g = check.get("grid") or {}
    solver = check.get("solver") or {}
    return {
        "time_s": t.get("total_s"),
        "solver": t.get("solver") or solver.get("resolved"),
        "solver_line": (solver.get("choice") or {}).get("line"),
        "memory_bytes": m.get("bytes"),
        "ram_bytes": m.get("ram_bytes"),
        "voxel_mm": g.get("voxel_mm"),
        "voxels": g.get("voxels"),
        "free_dofs": check.get("free_dofs"),
        "iterations": check.get("iterations"),
        "volume_fraction": check.get("volume_fraction"),
        "warnings": len(check.get("warnings") or []),
    }


def _round_time(seconds) -> str:
    if seconds is None:
        return "no time estimate"
    if seconds < 90:
        return f"about {seconds:.0f} s"
    return f"about {seconds / 60:.0f} min"


def summary(name: str, meshes: list[dict], cases: list[dict], est: dict, yaml_rel: str) -> str:
    """The text of the result: what the user sees in the view, and the numbers, for the model and for a client without MCP Apps."""
    counts: dict[str, int] = {}
    for m in meshes:
        counts[m["kind"]] = counts.get(m["kind"], 0) + 1
    parts = ", ".join(f"{n} {k.replace('_', '-')}" for k, n in counts.items() if k != "design")
    mem = est.get("memory_bytes")
    mem_text = f", memory about {mem / 1e9:.1f} GB" if mem else ""
    lines = [
        f"3D view of {name} ({yaml_rel}): the design space, {parts} region(s), and {len(cases)} load case(s) "
        f"({', '.join(c['name'] for c in cases)}).",
        f"Estimate: {_round_time(est.get('time_s'))} with {est.get('solver')}{mem_text}; "
        f"voxel {est.get('voxel_mm')} mm, {est.get('voxels')} voxels, {est.get('iterations')} passes.",
        "A client that shows MCP Apps shows the view in the chat. The view's button 'Start the run' starts the run itself if the host lets the view call tools (then call run_status to see it), else it sends a chat message that asks you to start it.",
        f"The full facts and the {est.get('warnings')} warning(s) are in check_problem.",
    ]
    return "\n".join(lines)
