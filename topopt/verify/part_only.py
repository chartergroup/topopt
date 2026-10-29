"""`topopt verify --part-only`: the check of `result/part_only.stl`. CONTRACTS.md sections 9.1 and 16 (added with P4b-3).

A run with `regions.not_printed` has `result/part_only.stl`: the part without the bodies that are in the problem but not in the
print (a plate that the part is bolted to, a load block). The default of `topopt verify` does not change (`nominal.stl` when the run
has one, else `final.stl`). With `--part-only` (MCP `verify_run(part_only=true)`) it meshes `part_only.stl` instead.

Boundary conditions. The region rule is the one of `final.stl`: `regions.select_region`, the same band. A removed body is not in the
mesh, so the rule finds the part next to the surface of the body: the contact face, and what the band takes next to it. There is no
new tolerance and no new number.

- A removed `fixed` region: the P2 nodes that the rule selects are clamped, as for a fixed region that is in the mesh.
- A removed `load` region: the force is a traction on the boundary facets that the rule selects. The check of the resultant stays.
- A removed `solid` or `keep_out` region: no boundary condition. The report says so. When nothing else holds a piece of the part, the
  stiffness matrix is singular, and the error of the solve gives the facts and the options.
- A region that is not removed keeps the rule as it is.

The facts of each removed body are in the block `part_only` of `verify.json` (`removed`): the role, the file, the boundary condition,
the selected nodes and facets, the area of the selected facets, the counts at the three bands, and the contact area of the cut. The
contact area is measured on the mesh, with no tolerance: for two solids that touch and do not overlap, the surface of the union is the
sum of the two surfaces minus twice the contact area (`contact_areas`).
"""
from __future__ import annotations

from pathlib import Path

from .names import opt

FILE = "part_only.stl"                       # result/part_only.stl (the same name as `topopt.post.PART_ONLY_FILE`)
NO_CONDITION_ROLES = ("solid", "keep_out")   # a removed body of these kinds gives no boundary condition

# The keys of the block `part_only` of run.json that verify copies into its own block (a record of the cut, not a result of verify).
RUN_RECORD_KEYS = ("applied", "removed", "volume_before_mm3", "volume_after_mm3", "removed_mm3", "leftovers_dropped", "bodies",
                   "watertight", "shortest_edge_mm", "mesh_method", "notes")

STOCK_NOTE = ("the run has machining stock (result/nominal.stl exists): part_only.stl is cut from the as-printed mesh final.stl, "
              "so it keeps the stock. The finished part without the bodies is not made, and verify makes no second cut. "
              "Without the option, verify meshes nominal.stl.")


def record(run: dict) -> dict:
    """The block `part_only` of run.json, or `{}` for a run without the key."""
    block = run.get("part_only") if isinstance(run, dict) else None
    return dict(block) if isinstance(block, dict) else {}


def missing_text(run_dir, run: dict) -> str:
    """The text of the `UserError` for a run that has no `result/part_only.stl`: the fact, then the options."""
    rec = record(run)
    if not rec:
        fact = "run.json has no part_only block, so the run was made without regions.not_printed"
    elif rec.get("applied") is False:
        fact = f"the post step did not make it ({rec.get('error') or 'no reason recorded'})"
    elif rec.get("applied"):
        fact = "run.json says that the post step made it, but the file is not in the folder (deleted or moved?)"
    else:
        fact = "run.json lists the bodies of regions.not_printed but has no record of the cut, so the post step did not finish"
    return (f"no result/{FILE} in {run_dir}: {fact}. Options: set regions.not_printed in the YAML and run again "
            f"(CONTRACTS.md section 9.1), or verify final.stl (without {opt('part_only')})")


def available_note(mesh_file: str, estimate_only: bool = False) -> str:
    """The fact line for a run that has `part_only.stl` when the option is not set. `mesh_file` is the file that verify meshes.

    With `--estimate-only` nothing is meshed, so the line says "would mesh"."""
    verb = "would mesh" if estimate_only else "meshes"
    return (f"this run has result/{FILE}, the part without the bodies of regions.not_printed. Verify {verb} {mesh_file}. "
            f"Set {opt('part_only')} to mesh {FILE}.")


def yaml_vs_run(cfg, rec: dict) -> dict:
    """Compare `regions.not_printed` of the YAML with the list that the run cut out. A difference is a fact, not an error."""
    run_labels = [str(x) for x in rec.get("removed") or []]
    yaml_labels = [n.label for n in cfg.not_printed]
    diffs = []
    if sorted(run_labels) != sorted(yaml_labels):
        diffs.append(f"regions.not_printed: run.json has {run_labels}, the YAML has {yaml_labels}")
    return dict(matches=not diffs, differences=diffs,
                note="the bodies that verify treats as removed are the ones of the YAML. If they differ from the list that the run "
                     "cut out, part_only.stl does not match the boundary conditions.")


def plan_block(cfg, run: dict, sha256: str, has_nominal: bool) -> dict:
    """The part of the block `part_only` that is known before meshing (it is in the estimate-only result too)."""
    rec = record(run)
    block = dict(on=True, file=FILE, sha256=sha256,
                 run_record={k: rec[k] for k in RUN_RECORD_KEYS if k in rec},
                 yaml_vs_run=yaml_vs_run(cfg, rec),
                 removed=[dict(label=n.label, role=n.role, name=n.name, file=str(n.path)) for n in cfg.not_printed])
    if has_nominal:
        block["notes"] = [STOCK_NOTE]
    return block


def error_context(block: dict) -> str:
    """A sentence for an error of verify (a body with no support, a load that no facet takes, a singular matrix): what the file is."""
    labels = ", ".join(titles(block["removed"])) or "none"
    return (f"The mesh is result/{FILE}. The removed bodies ({labels}) are not in it: a removed fixed body is clamped at its contact "
            f"face, a removed load body takes its force at its contact face, and a removed solid or keep-out body gives no boundary "
            f"condition. To check the part with the bodies, verify final.stl (without {opt('part_only')}), or keep the body in the "
            f"print (take it out of regions.not_printed and run again). ")


def titles(removed: list[dict]) -> list[str]:
    """The name of each removed body in a text: its label, and its file too when another body has the same label.

    Two files of one name in two folders (same role, same file name) are two bodies with one label (`config.py`, `_not_printed`).
    """
    count: dict = {}
    for r in removed:
        count[r["label"]] = count.get(r["label"], 0) + 1
    return [r["label"] if count[r["label"]] == 1 else f"{r['label']} ({r['file']})" for r in removed]


def selection_text(items: list[tuple[str, object, dict]]) -> tuple[str, list[dict]]:
    """`(text, [{label, file, nodes}, ...])` for the removed fixed bodies: how many nodes each one clamps (0: it clamps nothing).

    `items` is `(label, file, facts of the selection)` for each removed fixed body, in the order of the YAML. The list is by body, not by
    label: two bodies can have one label. The text is for the error of a body with no support.
    """
    if not items:
        return "", []
    names = titles([dict(label=label, file=str(file)) for label, file, _ in items])
    parts = [f"{name} selects {f['nodes']:,} nodes within the band of {f['band_mm']:g} mm" if f["nodes"]
             else f"{name} selects no node within the band of {f['band_mm']:g} mm, so it clamps nothing" for name, (_, _, f) in zip(names, items)]
    return ("The removed fixed bodies: " + "; ".join(parts) + ". ",
            [dict(label=label, file=str(file), nodes=int(f["nodes"])) for label, file, f in items])


# --------------------------------------------------------------------------- #
# Contact area
# --------------------------------------------------------------------------- #

def contact_areas(part_path, body_paths: list) -> list[dict]:
    """`[{"area_mm2": ...}, ...]`: the contact area (mm2) of `part_only.stl` with each removed body, in the order of `body_paths`.

    The list is by body, not by label: two bodies can have one label.
    The part and a removed body touch and do not overlap (the file is the mesh minus the body). The surface of their union is the sum
    of the two surfaces minus twice the contact area, so the contact area is half of (area of the part + area of the body - area of the
    union). The union is exact (manifold3d), so there is no tolerance. A body that cannot be read gives `area_mm2` None and the
    error: the contact area is a fact, never a reason to stop.
    """
    out: list[dict] = []
    try:
        from .. import smooth_sdf
        part = smooth_sdf._load_manifold(part_path)
        part_area = float(part.surface_area())
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        return [dict(area_mm2=None, error=f"part_only.stl could not be read as a closed solid: {err}") for _ in body_paths]
    for path in body_paths:
        try:
            body = smooth_sdf._load_manifold(path)
            both = float((part + body).surface_area())
            out.append(dict(area_mm2=max(0.0, 0.5 * (part_area + float(body.surface_area()) - both))))
        except Exception as exc:  # noqa: BLE001
            out.append(dict(area_mm2=None, error=f"{type(exc).__name__}: {exc}"))
    return out


# --------------------------------------------------------------------------- #
# Facts of the removed bodies, from the selections of one level
# --------------------------------------------------------------------------- #

def condition_of(n, sel, used: bool) -> tuple[str, str]:
    """`(boundary condition, note)` of one removed body. `sel` is its `Selection`; `used` is True for a load region that a case uses."""
    if n.role == "fixed":
        if sel.empty:
            return "none", ("no node of the part is within the band of this body, so it clamps nothing. The part is held by the "
                            "other fixed regions, if there are any")
        return "clamp", "the P2 nodes within the band of the body are clamped (all 3 dofs)"
    if n.role == "load":
        if not used:
            return "none", "no load case uses this region, so it carries no force"
        return "traction", ("the force is a uniform traction on the boundary facets whose 3 vertices are within the band of the "
                            "body (the contact face, and what the band takes next to it)")
    return "none", ("the body is not in the mesh and gives no support, so there is no boundary condition here. "
                    "The part has a free surface where the body was")


def body_facts(n, sel, used: bool, contact: dict | None) -> dict:
    """The facts of one removed body: role, file, boundary condition, selection counts and area, contact area."""
    cond, note = condition_of(n, sel, used)
    f = {k: v for k, v in sel.facts.items() if k not in ("region", "role", "file")}
    out = dict(label=n.label, role=n.role, name=n.name, file=str(n.path), boundary_condition=cond, **f, note=note)
    if contact is not None:
        c = dict(contact)
        area = c.get("area_mm2")
        sel_area = float(sel.facts.get("area_mm2", 0.0) or 0.0)
        c["selected_area_over_contact"] = float(sel_area / area) if area and area > 0 else None
        out["contact"] = c
    return out


def body_line(r: dict, title: str | None = None) -> str:
    """One line of text for a removed body: the name (`title`, default the label), what it gives, the selected nodes and facets, the area, the contact area."""
    c = r.get("counts_by_band") or {}
    bands = " / ".join(str(c[k]["nodes"]) for k in ("0", "0.5x", "1x") if k in c)
    what = {"clamp": "clamp", "traction": "traction", "none": "no boundary condition"}[r["boundary_condition"]]
    if r["boundary_condition"] == "none" and r.get("role") == "load":                 # a load region that no load case uses
        what = "no traction (no load case uses this region)"
    text = (f"{title or r['label']}: {what}; {r['nodes']:,} nodes selected" + (f" ({bands} at the bands 0, 0.5x and 1x)" if bands else "")
            + f", {r['facets']:,} facets, {r['area_mm2']:.1f} mm2")
    ct = r.get("contact")
    if ct is not None:
        if ct.get("area_mm2") is None:
            text += "; contact area not measured"
        else:
            text += f"; contact area of the cut {ct['area_mm2']:.1f} mm2"
            if ct.get("selected_area_over_contact") is not None:
                text += f" (selected over contact {ct['selected_area_over_contact']:.2f})"
    return text


def say_line(r: dict, title: str | None = None) -> str:
    """The line of a removed body for the log and the summary (`body_line`, with `removed body` in front)."""
    return "removed body " + body_line(r, title)


def singular_note(removed: list[dict]) -> str | None:
    """The fact that the solve can be singular, when a removed body gives no boundary condition. None when every body has one.

    A load region that no case uses is not in the list: it never held the part."""
    free = [t for r, t in zip(removed, titles(removed)) if r["boundary_condition"] == "none" and r.get("role") != "load"]
    if not free:
        return None
    return (f"{', '.join(free)}: no boundary condition. A removed solid or keep-out body gives no support, and a removed fixed body "
            f"that clamps no node gives none. When nothing else holds a piece of the part, the stiffness matrix is singular. "
            f"Then the solve stops with exit code 1, and the error has the facts and the options. This solve did not stop.")


def level_block(removed: list[dict]) -> dict:
    """The part of the block `part_only` that one level finds: the facts of the removed bodies and the singular note."""
    out: dict = dict(removed=removed)
    note = singular_note(removed)
    if note:
        out["singular_note"] = note
    return out


def merge_block(plan: dict, level: dict) -> dict:
    """The block `part_only` of the report: the plan (known before meshing) with the facts of the reference level."""
    out = dict(plan)
    out["removed"] = level["removed"]
    if "singular_note" in level:
        out["singular_note"] = level["singular_note"]
    return out
