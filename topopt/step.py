"""Faceted STEP file from a finished run (CONTRACTS.md section 42).

`export_step(run_dir)` reads `result/final.stl` (or `result/part_only.stl`), optionally simplifies it with manifold3d, wraps the
triangles as planar faces of a closed solid with OpenCASCADE (OCP), merges coplanar faces into one planar face, writes a STEP file,
reads the file back, and reports what the read-back shows. The faces are facets of the mesh, not editable surfaces: a CAD program can
cut holes in the solid and join it to other parts.

OpenCASCADE is the optional extra `step` (the wheel `cadquery-ocp-novtk`). The core never imports it: `import topopt` and
`import topopt.step` do not load it. `_import_ocp()` is the only place that imports it, inside a function.
A missing extra is a `StepUnavailable` (a `UserError`) with the install command.

Every check is a fact in the report. A failed check does not stop the export: the file is written, and the report says which check
failed (`checks_failed`). The checks have no threshold: `before_write` looks at the model in memory (the analyzer says valid, and the
volume of each solid is positive); the others look at the file as OpenCASCADE reads it (the counts of solids and shells equal the
counts of the build, the model is valid and closed, the length unit is mm, the schema is the one asked for).
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import time
from array import array
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .config import UserError
from .uv_command import uv_command

FORMAT = 1
OCP_DISTRIBUTION = "cadquery-ocp-novtk"
OCP_PIN = "7.9.3.1.1"
EXTRA = "step"
SCHEMAS = {"AP214": "AP214IS", "AP242": "AP242DIS"}     # the CLI name -> the value of the OCCT parameter write.step.schema
DEFAULT_SCHEMA = "AP214"
REPORT_SUFFIX = ".json"       # the report is next to the STEP file and has its name plus this: result/final.step.json
STEP_SUFFIXES = (".step", ".stp")
EXIT_CHECK_FAILED = 3     # the file is written and a check of the read-back failed (a fact; 0 = all checks passed)

DEFAULT_TOLERANCE_VOXEL_FRACTION = 1.0 / 20.0     # settings.tolerance_mm default: the voxel size of the run times this
# The cost model of `--estimate-only` (CONTRACTS.md section 42.5): a fit, made before the check `before_write` existed, to the measured runs of section 42.7 (Linux x86_64, a host with
# a load average of 10 to 18 on 22 threads, 2 threads for the libraries), with weights 1/measured so that a small file counts as much as
# a large one. It is an estimate, not a limit: nothing refuses.
ESTIMATE_STEP_BYTES_PER_FACE = 1100.0
ESTIMATE_SECONDS = (1.3, 1.08e-3, 17.7e-6)        # fixed, for each face, for each triangle of the STL
ESTIMATE_PEAK_BYTES = (526e6, 15.5e3, 0.263e3)    # fixed, for each face, for each triangle of the STL
ESTIMATE_MAX_ERROR_PCT = 32                       # the largest error of the fit over the measured runs
MEASURED_RUNS = 21                                # the number of runs in that fit
DEVIATION_SURFACE_SAMPLES = 20000     # random points on each surface for the deviation, added to the vertices
DEVIATION_VERTEX_SAMPLES = 20000      # at most this many vertices of each mesh for the deviation (an even pick)
DEVIATION_FACE_BATCH = 20000          # faces that are not triangles are meshed in batches of this size: OpenCASCADE holds about 16 KB for each face it meshes
DEVIATION_CHUNK = 2000                # points for each call of the nearest-point search (it holds the candidates of a whole call in memory)

INSTALL_TEXT = (
    "OpenCASCADE (the optional extra `step`, the package {dist}) is {state} in this Python, so topopt cannot write a STEP file.{why}\n"
    "Install it (macOS with Apple silicon, and Linux x86_64; about 250 MB on disk). In the topopt folder:\n"
    "    {uv} pip install -e \".[{extra}]\"\n"
    "With no checkout of topopt:\n"
    "    {uv} pip install \"{dist}=={pin}\"\n"
    "Keep the quotes: zsh reads [...] as a pattern. Then run this command again.")


class StepUnavailable(UserError):
    """The extra `step` (OpenCASCADE) is not installed, or it does not load. The text has the install command."""


def install_text(error: str | None = None) -> str:
    """The message of a missing extra. `error` (the text of the ImportError) says that the package is there and does not load."""
    return INSTALL_TEXT.format(dist=OCP_DISTRIBUTION, pin=OCP_PIN, extra=EXTRA, uv=uv_command(),
                               state="installed but does not load" if error else "not installed",
                               why=f" The import fails with: {error}." if error else "")


def _import_ocp() -> SimpleNamespace:
    """Import OpenCASCADE and return the classes that this module uses. Raises `StepUnavailable`. The one import of OCP.

    It also removes the default printer of OCCT (`Message_PrinterOStream`): OCCT prints transfer statistics to the C stdout, which
    is the protocol channel of the MCP server and noise in a terminal. The report carries the facts instead.
    """
    try:
        from OCP.BRep import BRep_Builder, BRep_Tool
        from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex)
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.BRepTools import BRepTools
        from OCP.GProp import GProp_GProps
        from OCP.Interface import Interface_Static
        from OCP.Message import Message, Message_PrinterOStream
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
        from OCP.StlAPI import StlAPI_Writer
        from OCP.TColStd import TColStd_SequenceOfAsciiString
        from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
        from OCP.TopExp import TopExp, TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shell, TopoDS_Solid, TopoDS_Wire
        from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape
        from OCP.TopLoc import TopLoc_Location
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Plane
        from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    except ImportError as exc:
        raise StepUnavailable(install_text(None if _is_missing(exc) else f"{type(exc).__name__}: {exc}")) from None
    Message.DefaultMessenger_s().RemovePrinters(Message_PrinterOStream.get_type_descriptor_s())
    return SimpleNamespace(**{k: v for k, v in locals().items() if k not in ("exc",)})


def _is_missing(exc: ImportError) -> bool:
    """True when the ImportError says that the package OCP is not there (not that one of its libraries failed to load)."""
    return isinstance(exc, ModuleNotFoundError) or "None in sys.modules" in str(exc)


def ocp_version() -> str | None:
    """The version of the installed OCP wheel, else None. Reads the package metadata only (no import of OCP)."""
    from importlib import metadata
    for dist in (OCP_DISTRIBUTION, "cadquery-ocp"):
        try:
            return f"{dist} {metadata.version(dist)}"
        except metadata.PackageNotFoundError:
            continue
    return None


# --- the mesh -------------------------------------------------------------------------------------------------------------------


def _point_in_shell(V: np.ndarray, sub: np.ndarray, point: np.ndarray) -> bool:
    """True when `point` lies inside the closed triangle surface `sub` (rows of vertex numbers of `V`). The orientation does not matter."""
    from .post import points_inside
    used = np.unique(sub)
    return bool(points_inside(V[used], np.searchsorted(used, sub), point[None, :])[0])


def mesh_shells(V: np.ndarray, F: np.ndarray) -> dict:
    """The bodies and voids of a triangle mesh, by edges (a shell = faces joined along an edge of exactly 2 faces).

    A shell is not manifold when one of its edges has 1 face or more than 2 faces; it is written as it is (a body, not turned).
    A closed shell is a void when it lies inside an odd number of other closed shells, and a body otherwise, whatever the winding of
    the shells is. A body is written with its faces outward and a void with its faces inward: a shell with the other winding is turned
    (`flip`). The volume of the mesh is the volume of the bodies minus the volume of the voids.

    Returns `label` (the shell of each face), `volume` (signed, per shell, as the faces are wound), `kind` (`body` or `void` per
    shell), `flip`, `not_manifold`, `parent` (the closed shell with the smallest volume that holds the shell, else -1),
    `volume_mm3` (the volume of the solid the mesh describes), and `count`.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    m, n = len(F), len(V)
    ea, eb = np.concatenate([F[:, 0], F[:, 1], F[:, 2]]), np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    face = np.tile(np.arange(m), 3)
    key = np.minimum(ea, eb).astype(np.int64) * n + np.maximum(ea, eb)
    order = np.argsort(key, kind="stable")
    ks, fs = key[order], face[order]
    first = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]) if len(ks) else np.zeros(0, np.int64)
    count = np.diff(np.r_[first, len(ks)])
    two = first[count == 2]
    graph = coo_matrix((np.ones(len(two)), (fs[two], fs[two + 1])), shape=(m, m))
    nshell, label = connected_components(graph, directed=False)
    c = V.mean(axis=0)
    a, b, d = V[F[:, 0]] - c, V[F[:, 1]] - c, V[F[:, 2]] - c
    volume = np.bincount(label, weights=np.einsum("ij,ij->i", a, np.cross(b, d)) / 6.0, minlength=nshell)
    not_manifold = np.zeros(nshell, bool)
    not_manifold[label[fs[np.repeat(count != 2, count)]]] = True
    kind = np.array(["body"] * nshell, dtype=object)
    flip = np.zeros(nshell, bool)
    parent = np.full(nshell, -1, np.int64)
    closed = np.flatnonzero(~not_manifold)
    if len(closed) > 1:
        by_shell = np.argsort(label, kind="stable")                  # the faces of shell 0, then of shell 1, ...
        sizes = np.bincount(label, minlength=nshell)
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        corners = V[F[by_shell]]                                       # (faces, 3 corners, 3 coordinates)
        low = np.minimum.reduceat(corners.min(axis=1), starts, axis=0)       # the box of each shell
        high = np.maximum.reduceat(corners.max(axis=1), starts, axis=0)
        del corners
        for s in closed.tolist():
            point = V[F[by_shell[starts[s]]]].mean(axis=0)            # a point of the wall of the shell
            near = np.flatnonzero((low <= point).all(axis=1) & (high >= point).all(axis=1) & ~not_manifold)    # only a closed shell whose box holds the point
            holders = [int(q) for q in near if q != s
                       and _point_in_shell(V, F[by_shell[starts[q]:starts[q] + sizes[q]]], point)]
            if len(holders) % 2:
                kind[s] = "void"
            if holders:
                parent[s] = min(holders, key=lambda q: abs(volume[q]))
    for s in closed.tolist():
        flip[s] = bool(volume[s] > 0) if kind[s] == "void" else bool(volume[s] < 0)
    oriented = np.where(kind == "void", -1.0, 1.0) * np.abs(volume)      # a shell that is not manifold counts as a body, whatever its winding
    return {"label": label, "volume": volume, "kind": kind, "flip": flip, "not_manifold": not_manifold, "parent": parent,
            "volume_mm3": float(oriented.sum()), "count": int(nshell)}


def simplify_mesh(V: np.ndarray, F: np.ndarray, tolerance_mm: float):
    """`(V, F, fact)`: the mesh after manifold3d `simplify(tolerance_mm)`. A mesh that is not a manifold stays as it is (a fact)."""
    from .post import _to_manifold
    before = int(len(F))
    try:
        man = _to_manifold(V, F)
    except ValueError as exc:
        return V, F, {"applied": False, "tolerance_mm": tolerance_mm, "triangles_before": before, "triangles_after": before,
                      "note": f"not simplified: {exc}"}
    out = man.simplify(tolerance_mm).to_mesh()
    V2 = np.asarray(out.vert_properties, np.float64)[:, :3]
    F2 = np.asarray(out.tri_verts, np.int64)
    return V2, F2, {"applied": True, "tolerance_mm": tolerance_mm, "triangles_before": before, "triangles_after": int(len(F2))}


# --- the B-rep ------------------------------------------------------------------------------------------------------------------


def build_brep(ocp: SimpleNamespace, V: np.ndarray, F: np.ndarray, shells: dict | None = None) -> tuple[object, dict]:
    """Wrap a closed triangle mesh as OpenCASCADE solids: `(shape, facts)`.

    The vertices and the edges are shared between the faces (one `TopoDS_Vertex` for each mesh vertex, one `TopoDS_Edge` for each mesh
    edge), so the shell is closed by its topology and no sewing is needed. Each triangle is a planar face. Each body is one solid; a void
    is an inner shell of the solid that holds it. One solid is the shape; several are a compound. `shells` is `mesh_shells(V, F)` when
    the caller has it already.
    Raises `UserError` for an edge shorter than the OCCT point tolerance (1e-7 mm): the simplification joins such vertices.
    """
    o = ocp
    sh = shells if shells is not None else mesh_shells(V, F)
    label, flip = sh["label"], sh["flip"]
    m, n = len(F), len(V)
    ea, eb = np.concatenate([F[:, 0], F[:, 1], F[:, 2]]), np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    key = np.minimum(ea, eb).astype(np.int64) * n + np.maximum(ea, eb)
    uniq, eid = np.unique(key, return_inverse=True)
    ue_a, ue_b = uniq // n, uniq % n
    forward = (ea < eb).reshape(3, m)
    eid = eid.reshape(3, m)
    pts = V.tolist()
    verts = [o.BRepBuilderAPI_MakeVertex(o.gp_Pnt(*p)).Vertex() for p in pts]
    edges = []
    for a, b in zip(ue_a.tolist(), ue_b.tolist()):
        mk = o.BRepBuilderAPI_MakeEdge(verts[a], verts[b])
        if not mk.IsDone():
            raise UserError("the mesh has an edge shorter than 1e-7 mm, which OpenCASCADE cannot make. The simplification joins such "
                            "vertices: give --tolerance-mm 0.00001 or more")
        edges.append(mk.Edge())
    # the normal of each triangle, from the winding (a flipped shell turns it)
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    normal = np.cross(p1 - p0, p2 - p0)
    length = np.linalg.norm(normal, axis=1)
    degenerate = length < 1e-300
    normal = normal / np.where(degenerate, 1.0, length)[:, None]
    builder = o.BRep_Builder()
    shells_out = []
    for s in range(sh["count"]):
        shell = o.TopoDS_Shell()
        builder.MakeShell(shell)
        shells_out.append(shell)
    skipped = 0
    fwd_l, eid_l, nrm, org, lab = forward.T.tolist(), eid.T.tolist(), normal.tolist(), p0.tolist(), label.tolist()
    for i in range(m):
        if degenerate[i]:
            skipped += 1
            continue
        wire = o.TopoDS_Wire()
        builder.MakeWire(wire)
        for k in range(3):
            e = edges[eid_l[i][k]]
            builder.Add(wire, e if fwd_l[i][k] else e.Reversed())
        plane = o.gp_Pln(o.gp_Pnt(*org[i]), o.gp_Dir(*nrm[i]))
        face = o.BRepBuilderAPI_MakeFace(plane, wire, True).Face()
        if flip[lab[i]]:
            face = face.Reversed()
        builder.Add(shells_out[lab[i]], face)
    solids, voids = {}, []
    for s in range(sh["count"]):
        if sh["kind"][s] == "void":
            voids.append(s)
            continue
        solid = o.TopoDS_Solid()
        builder.MakeSolid(solid)
        builder.Add(solid, shells_out[s])
        solids[s] = solid
    for s in voids:      # a void goes to the closed shell that holds it with the smallest volume (a body: a void has an odd depth)
        holder = int(sh["parent"][s])
        builder.Add(solids[holder] if holder in solids else next(iter(solids.values())), shells_out[s])
    if len(solids) == 1:
        shape = next(iter(solids.values()))
    else:
        shape = o.TopoDS_Compound()
        builder.MakeCompound(shape)
        for solid in solids.values():
            builder.Add(shape, solid)
    kind = sh["kind"]
    facts = {"bodies": len(solids), "voids": len(voids), "shells_not_manifold": int(sh["not_manifold"].sum()),
             "inside_out_bodies": int(sum(1 for s in range(sh["count"]) if flip[s] and kind[s] == "body")),
             "inside_out_voids": int(sum(1 for s in range(sh["count"]) if flip[s] and kind[s] == "void")),
             "degenerate_triangles_left_out": skipped, "edges": len(edges)}
    return shape, facts


def merge_planes(ocp: SimpleNamespace, shape):
    """Merge faces that lie in one plane into one planar face (`ShapeUpgrade_UnifySameDomain`, its default tolerances).

    The frozen plates of a part (the faces of the fixed and load regions) become true planes. Collinear edges join too.
    Tolerances above the default merged faces of the curved part into an invalid solid in the measurements, so the default is used:
    a plane merges when its triangles are coplanar to the precision of OpenCASCADE (1e-7 mm). Returns the new shape.
    """
    u = ocp.ShapeUpgrade_UnifySameDomain(shape, True, True, False)
    u.Build()
    return u.Shape()


# --- the STEP file --------------------------------------------------------------------------------------------------------------


def write_step(ocp: SimpleNamespace, shape, path: Path, schema: str, product_name: str) -> dict:
    """Write `shape` as a STEP file at `path` (through a temporary file in the same folder). Returns the facts. Raises `RuntimeError`.

    `rejected` in the facts lists the settings that OpenCASCADE did not take (the unit and the product name): the old value then stays,
    because the parameters belong to the process.
    """
    o = ocp
    writer = o.STEPControl_Writer()      # the writer first: it sets the default values of the parameters
    facts = {"schema_parameter": SCHEMAS[schema], "rejected": []}
    if not o.Interface_Static.SetCVal_s("write.step.schema", SCHEMAS[schema]):
        raise RuntimeError(f"OpenCASCADE does not know the STEP schema {SCHEMAS[schema]}")
    for name, value in (("write.step.unit", "MM"), ("write.step.product.name", product_name)):
        if not o.Interface_Static.SetCVal_s(name, value):
            facts["rejected"].append(f"OpenCASCADE did not take {name} = {value!r}")
    o.Interface_Static.SetIVal_s("write.surfacecurve.mode", 0)       # no pcurves: they only repeat the planes (half the file size)
    status = writer.Transfer(shape, o.STEPControl_AsIs)
    if status != o.IFSelect_RetDone:
        raise RuntimeError(f"OpenCASCADE could not transfer the shape to STEP ({status})")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        if writer.Write(str(tmp)) != o.IFSelect_RetDone:
            raise RuntimeError(f"OpenCASCADE could not write {path}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    facts["bytes"] = path.stat().st_size
    return facts


def read_header_facts(path: Path) -> dict:
    """The schema and the product name of a STEP file, from its text: the head holds FILE_SCHEMA and the PRODUCT entity.

    `product_name` is the name that the file holds (OpenCASCADE adds a number to the name that was set).
    """
    with open(path, "rb") as fh:
        head = fh.read(16384).decode("utf-8", errors="replace")
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", head)
    schema_text = m.group(1) if m else None
    schema = None
    if schema_text:
        if " 214 " in schema_text or "214" in schema_text.split():
            schema = "AP214"
        elif "442" in schema_text or "242" in schema_text.upper() or "AP242" in schema_text.upper():
            schema = "AP242"
        elif " 203 " in schema_text or "CONFIG_CONTROL" in schema_text.upper():
            schema = "AP203"
    q = re.search(r"=\s*PRODUCT\s*\(\s*'((?:[^']|'')*)'", head)
    product = q.group(1).replace("''", "'").replace("\\\\", "\\") if q else None
    return {"schema_text": schema_text, "schema": schema, "product_name": product}


UNIT_NAMES = {"millimetre": "mm", "metre": "m", "centimetre": "cm", "kilometre": "km", "micrometre": "um"}


def read_back(ocp: SimpleNamespace, path: Path):
    """Read the STEP file with OpenCASCADE. Returns `(shape, fact)`: the shape is None when the read fails.

    The fact has `unit`: the length unit that the file declares, from `STEPControl_Reader.FileUnits` (`mm`, or the name that
    OpenCASCADE gives, for example `INCH`; None when the file names none). The reader converts the geometry to mm when it reads, so
    the model of the read-back has the same size in any unit, and only this fact shows the unit of the file.
    """
    o = ocp
    reader = o.STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != o.IFSelect_RetDone:
        return None, {"read": False, "status": str(status), "unit": None}
    lengths = o.TColStd_SequenceOfAsciiString()
    reader.FileUnits(lengths, o.TColStd_SequenceOfAsciiString(), o.TColStd_SequenceOfAsciiString())
    names = [lengths.Value(i).ToCString() for i in range(1, lengths.Length() + 1)]
    unit = "+".join(UNIT_NAMES.get(x, x) for x in names) if names else None
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        return None, {"read": False, "status": "the file holds no shape", "unit": unit}
    return shape, {"read": True, "unit": unit}


def count_shapes(ocp: SimpleNamespace, shape, kind) -> int:
    m = ocp.TopTools_IndexedMapOfShape()
    ocp.TopExp.MapShapes_s(shape, kind, m)
    return int(m.Extent())


def shape_volume(ocp: SimpleNamespace, shape) -> float:
    props = ocp.GProp_GProps()
    ocp.BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def model_facts(ocp: SimpleNamespace, shape) -> dict:
    """The counts of the read-back model: solids, shells, faces (and how many are planar), edges, vertices, and the edges that do
    not join exactly 2 faces (`free_edges` has 1 face, `overshared_edges` more than 2). A closed shell has neither."""
    o = ocp
    faces = o.TopTools_IndexedMapOfShape()
    o.TopExp.MapShapes_s(shape, o.TopAbs_FACE, faces)
    planar = 0
    for i in range(1, faces.Extent() + 1):
        if o.BRepAdaptor_Surface(o.TopoDS.Face_s(faces.FindKey(i))).GetType() == o.GeomAbs_Plane:
            planar += 1
    anc = o.TopTools_IndexedDataMapOfShapeListOfShape()
    o.TopExp.MapShapesAndAncestors_s(shape, o.TopAbs_EDGE, o.TopAbs_FACE, anc)
    free = over = 0
    for i in range(1, anc.Extent() + 1):
        k = anc.FindFromIndex(i).Size()
        free += k == 1
        over += k > 2
    return {"solids": count_shapes(o, shape, o.TopAbs_SOLID), "shells": count_shapes(o, shape, o.TopAbs_SHELL),
            "faces": int(faces.Extent()), "planar_faces": planar, "edges": int(anc.Extent()),
            "vertices": count_shapes(o, shape, o.TopAbs_VERTEX), "free_edges": int(free), "overshared_edges": int(over)}


def shape_triangles(ocp: SimpleNamespace, shape, batch: int = DEVIATION_FACE_BATCH) -> tuple[np.ndarray, np.ndarray]:
    """The triangles of the faces of a shape, as `(V, F)` (the surface of the model, for the distance to the STL).

    A face with 3 vertices and 3 edges is a triangle: its three points are read directly. This is the case for nearly all faces of a
    wrapped mesh, and it costs no memory (meshing every face of a model of 565,000 faces took 10 GB). The other faces (the merged
    planes) are meshed by OpenCASCADE in batches of `batch` faces, and the mesh of a batch is cleaned away after it is read.
    The edges of the model are straight lines, so a flat polygon is its own mesh.
    """
    o = ocp
    faces = o.TopTools_IndexedMapOfShape()
    o.TopExp.MapShapes_s(shape, o.TopAbs_FACE, faces)
    direct = array("d")
    pending, vert_parts, tri_parts, base = [], [], [], 0
    vmap, emap = o.TopTools_IndexedMapOfShape(), o.TopTools_IndexedMapOfShape()

    def flush():
        nonlocal base
        if not pending:
            return
        comp, builder = o.TopoDS_Compound(), o.BRep_Builder()
        builder.MakeCompound(comp)
        for f in pending:
            builder.Add(comp, f)
        o.BRepMesh_IncrementalMesh(comp, 1.0, False, 0.5, True)
        loc = o.TopLoc_Location()
        pts, tris = [], []
        for f in pending:
            tri = o.BRep_Tool.Triangulation_s(f, loc)
            if tri is None:
                continue
            trsf = None if loc.IsIdentity() else loc.Transformation()
            n = tri.NbNodes()
            for i in range(1, n + 1):
                p = tri.Node(i)
                if trsf is not None:
                    p = p.Transformed(trsf)
                pts.append((p.X(), p.Y(), p.Z()))
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                tris.append((base + a - 1, base + b - 1, base + c - 1))
            base += n
        o.BRepTools.Clean_s(comp)
        vert_parts.append(np.asarray(pts, np.float64).reshape(-1, 3))
        tri_parts.append(np.asarray(tris, np.int64).reshape(-1, 3))
        pending.clear()

    for i in range(1, faces.Extent() + 1):
        face = o.TopoDS.Face_s(faces.FindKey(i))
        vmap.Clear()
        emap.Clear()
        o.TopExp.MapShapes_s(face, o.TopAbs_VERTEX, vmap)
        o.TopExp.MapShapes_s(face, o.TopAbs_EDGE, emap)
        if vmap.Extent() == 3 and emap.Extent() == 3:
            for k in range(1, 4):
                p = o.BRep_Tool.Pnt_s(o.TopoDS.Vertex_s(vmap.FindKey(k)))
                direct.extend((p.X(), p.Y(), p.Z()))
        else:
            pending.append(face)
            if len(pending) >= batch:
                flush()
    flush()
    n_points = len(direct) // 3          # three points for each direct triangle
    verts = [np.array(direct, np.float64).reshape(-1, 3)] + vert_parts
    # the points of the direct triangles come first, in threes; the points of the batches follow, so their indices shift by `n_points`
    tris = [np.arange(n_points, dtype=np.int64).reshape(-1, 3)] + [t + n_points for t in tri_parts]
    return np.concatenate(verts), np.concatenate(tris)


def _distances(mesh, points, chunk: int = DEVIATION_CHUNK) -> np.ndarray:
    """Distance of each point to the surface of `mesh`, in chunks: trimesh holds the candidate triangles of all the points of one call."""
    from trimesh.proximity import closest_point
    out = [closest_point(mesh, points[i:i + chunk])[1] for i in range(0, len(points), chunk)]
    return np.concatenate(out) if out else np.zeros(0)


def deviation(V_step, F_step, V_stl, F_stl, seed: int = 1) -> dict:
    """The distance between the STEP surface and the STL surface, sampled, in both directions.

    For each direction: the vertices of the source mesh (an even pick of at most `DEVIATION_VERTEX_SAMPLES`) and
    `DEVIATION_SURFACE_SAMPLES` random points on its surface (area weighted) go to the nearest point on the other surface.
    `max_mm` is the largest distance of all points. `mean_mm` is the mean over the surface points only (they are area weighted, so the
    mean is the mean over the surface). A point set is a sample: the true largest distance can be a little larger.
    """
    import trimesh
    a = trimesh.Trimesh(V_step, F_step, process=False)
    b = trimesh.Trimesh(V_stl, F_stl, process=False)
    rng = np.random.default_rng(seed)

    def one(src, dst):
        verts = np.asarray(src.vertices)
        if len(verts) > DEVIATION_VERTEX_SAMPLES:
            verts = verts[np.linspace(0, len(verts) - 1, DEVIATION_VERTEX_SAMPLES).astype(np.int64)]
        pts, _ = trimesh.sample.sample_surface(src, DEVIATION_SURFACE_SAMPLES, seed=int(rng.integers(1 << 30)))
        d_surf = _distances(dst, np.asarray(pts))
        d_vert = _distances(dst, verts)
        return {"max_mm": float(max(d_surf.max(), d_vert.max())), "mean_mm": float(d_surf.mean()),
                "points": int(len(d_surf) + len(d_vert))}

    return {"method": "sampled", "step_to_stl": one(a, b), "stl_to_step": one(b, a)}


# --- the run folder ------------------------------------------------------------------------------------------------------------


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def resolve_input(run_dir, part_only: bool = False) -> dict:
    """The mesh to wrap: `result/final.stl` of the run folder, or `result/part_only.stl`. Raises `UserError`.

    Returns `run_dir`, `stl` (Path), `mesh` (the file name), `run` (the run.json dict or `{}`) and `name` (the run name).
    """
    if run_dir is None or str(run_dir).strip() == "":
        raise UserError("no run folder: give the folder of a finished run (the --out of `topopt run`)")
    d = Path(run_dir).expanduser()
    if not d.is_dir():
        raise UserError(f"{d} is not a folder. Give the run folder (the --out of `topopt run`)")
    d = d.resolve()
    mesh = "part_only.stl" if part_only else "final.stl"
    stl = d / "result" / mesh
    status = _read_json(d / "status.json")
    state = status.get("state") if isinstance(status, dict) else None
    if state in ("starting", "running", "postprocess"):
        raise UserError(f"the run in {d} is not finished (state {state!r}). Wait until it is done, then run this again")
    if not stl.is_file():
        if part_only:
            raise UserError(f"no result/part_only.stl in {d}: the post step writes it only for a problem with regions.not_printed "
                            "(CONTRACTS.md section 9.1). Leave out --part-only to wrap result/final.stl")
        raise UserError(f"no result/final.stl in {d}: the run is not finished, or this is not a run folder")
    run = _read_json(d / "run.json")
    run = run if isinstance(run, dict) else {}
    return {"run_dir": d, "stl": stl, "mesh": mesh, "run": run, "name": str(run.get("name") or d.name)}


def voxel_size(info: dict) -> tuple[float | None, str | None]:
    """`(voxel size in mm, the file and key that hold it)` of a run folder, or `(None, None)`. `run.json` `h` first; a folder that
    `topopt verify` checked also holds it in `verify/verify.json` (`run.voxel_size_mm`)."""
    def good(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0
    if good(info["run"].get("h")):
        return float(info["run"]["h"]), "run.json h"
    verify = _read_json(info["run_dir"] / "verify" / "verify.json")
    run = verify.get("run") if isinstance(verify, dict) else None
    if isinstance(run, dict) and good(run.get("voxel_size_mm")):
        return float(run["voxel_size_mm"]), "verify/verify.json run.voxel_size_mm"
    return None, None


def resolve_tolerance(tolerance_mm, info: dict) -> tuple[float, str]:
    """`(tolerance in mm, where it comes from)`. An explicit value (0 = no simplification) wins. The default is the voxel size of the
    run times 1/20 (`voxel_size`). A run folder that holds no voxel size gets 0 (no simplification): there is no number to start from.
    Raises `UserError` for a value that is not a number of 0 or more."""
    if tolerance_mm is not None:
        try:
            tol = float(tolerance_mm)
        except (TypeError, ValueError):
            raise UserError(f"tolerance_mm is a number of 0 or more, got {tolerance_mm!r}") from None
        if not math.isfinite(tol) or tol < 0:
            raise UserError(f"tolerance_mm is a number of 0 or more, got {tolerance_mm!r}. 0 means no simplification")
        return tol, "option"
    h, where = voxel_size(info)
    if h is not None:
        return h * DEFAULT_TOLERANCE_VOXEL_FRACTION, f"default: the voxel size of the run ({h:g} mm, {where}) / 20"
    return 0.0, "default: the run folder holds no voxel size (run.json h, verify/verify.json run.voxel_size_mm), so no simplification"


def resolve_schema(schema) -> str:
    """`AP214` or `AP242`; the case and a hyphen do not matter (`ap-242`). Raises `UserError`."""
    key = str(schema or DEFAULT_SCHEMA).upper().replace("-", "")
    if key not in SCHEMAS:
        raise UserError(f"schema is AP214 or AP242, got {schema!r}")
    return key


def report_path_of(target: Path) -> Path:
    """The report goes next to the STEP file and has its name plus `.json`: `result/final.step.json`."""
    return target.with_name(target.name + REPORT_SUFFIX)


def resolve_output(info: dict, out) -> Path:
    """Where the STEP file goes. Default `result/final.step` (`result/part_only.step`). `out` is a file path, or an existing folder.

    An existing file at that path is replaced only when it is a `.step` or `.stp` file. The input STL and any other existing file
    are not: that is a `UserError` with the options. Raises `UserError`."""
    stem = Path(info["mesh"]).stem
    if out is None:
        return info["run_dir"] / "result" / f"{stem}.step"
    p = Path(out).expanduser()
    if p.is_dir():
        return (p / f"{stem}.step").resolve()
    if not p.parent.is_dir():
        raise UserError(f"the folder of --out does not exist: {p.parent.resolve()}")
    p = p.resolve()
    if p == info["stl"]:
        raise UserError(f"--out {p} is the STL that this command reads, and the STEP file would replace it. Give a new file name that "
                        f"ends in .step, or a folder (the file is then <folder>/{stem}.step)")
    if p.exists() and p.suffix.lower() not in STEP_SUFFIXES:
        raise UserError(f"--out {p} is an existing file that is not a .step or .stp file, and the STEP file would replace it. Give a "
                        f"new file name that ends in .step, or a folder (the file is then <folder>/{stem}.step)")
    return p


def check_writable(target: Path) -> None:
    """Raises `UserError` when the folder of the STEP file (and of its report) cannot take a new file. Called before the work."""
    folder = target.parent
    if not folder.is_dir() or not os.access(folder, os.W_OK | os.X_OK):
        raise UserError(f"no write access to the folder {folder}: the STEP file {target.name} and its report "
                        f"{report_path_of(target).name} go there. Give --out with a folder or file that you can write to")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _Phases:
    """The wall time of each phase and the peak memory of the process after it (the peak is the highest so far)."""

    def __init__(self, log):
        from .runinfo import peak_rss_info
        self._peak = peak_rss_info
        self.log = log or (lambda msg: None)
        self.t0 = self.t = time.perf_counter()
        self.seconds: dict = {}
        self.peak_mb: dict = {}

    def done(self, name: str) -> None:
        now = time.perf_counter()
        self.seconds[name] = round(now - self.t, 3)
        self.peak_mb[name] = round(self._peak()[0] / 1e6, 1)
        self.t = now

    def total(self) -> float:
        return round(time.perf_counter() - self.t0, 3)

    def cpu(self) -> float:
        """The CPU time of the process (user + system), in s. On a loaded host it is the steadier number."""
        return round(time.process_time(), 3)


# --- the export ------------------------------------------------------------------------------------------------------------------


def export_step(run_dir, *, part_only: bool = False, out=None, tolerance_mm=None, merge_planes: bool = True,
                schema: str = DEFAULT_SCHEMA, estimate_only: bool = False, log=None) -> dict:
    """Write a faceted STEP file of a finished run and return the report (CONTRACTS.md section 42).

    The run folder is only read. The new files are the STEP file (default `result/final.step`) and its report next to it
    (`result/final.step.json`); `estimate_only` writes none. `tolerance_mm`: the manifold3d simplification tolerance in mm (0 = none;
    default from the voxel size, 0 when the run folder holds none). `merge_planes`: join coplanar faces into planar faces. `schema`:
    `AP214` or `AP242`. `log(text)` gets one line for each step. Raises `UserError` (bad input, a target that would replace another
    file, no write access, a tolerance that removes every triangle, `StepUnavailable` without the extra). A failed check is not an
    error: it is in `checks_failed`, and the file is written.
    """
    from . import printcheck
    from .runinfo import peak_rss_info, write_atomic
    schema = resolve_schema(schema)
    info = resolve_input(run_dir, part_only)
    tol, tol_source = resolve_tolerance(tolerance_mm, info)
    target = resolve_output(info, out)
    ocp = None
    if not estimate_only:
        check_writable(target)
        ocp = _import_ocp()      # the estimate needs no OpenCASCADE: it works before the install
    ph = _Phases(log)
    say = ph.log
    say(f"topopt step: reading {info['stl']}")
    V0, F0 = printcheck.load_stl_float32(info["stl"])
    health = printcheck.mesh_health(V0, F0)
    shells0 = mesh_shells(V0, F0)
    stl_volume = shells0["volume_mm3"]
    ph.done("read_stl")
    input_facts = {"path": str(info["stl"]), "bytes": info["stl"].stat().st_size, "sha256": _sha256(info["stl"]),
                   "triangles": int(len(F0)), "vertices": int(len(V0)), "volume_mm3": stl_volume,
                   "closed": bool(health["watertight"] and health["winding_consistent"]), "bodies": len(health["bodies"]),
                   "voids": len(health["voids"]), "open_edges": health["open_edges"],
                   "nonmanifold_edges": health["nonmanifold_edges"], "zero_area_faces": health["zero_area_faces"]}
    if tol > 0:
        say(f"topopt step: simplify, tolerance {tol:g} mm ({tol_source})")
        V, F, simp = simplify_mesh(V0, F0, tol)
    else:
        V, F = V0, F0
        simp = {"applied": False, "tolerance_mm": 0.0, "triangles_before": int(len(F0)), "triangles_after": int(len(F0)),
                "note": "tolerance 0: the triangles of the STL are the faces"
                        + ("" if tol_source == "option" else ". Option: --tolerance-mm X joins short edges and removes slivers "
                           "(a run folder with a voxel size gets the voxel size / 20)")}
    ph.done("simplify")
    if len(F) == 0:
        raise UserError(f"the tolerance {tol:g} mm removed every triangle of the mesh ({len(F0):,} -> 0): the part is smaller than the "
                        "tolerance, or the STL is not in mm. No file is written. Give a smaller --tolerance-mm, or 0 for none")
    report = {"format": FORMAT, "state": "estimate" if estimate_only else "written", "run_dir": str(info["run_dir"]),
              "mesh": info["mesh"], "input": input_facts,
              "settings": {"tolerance_mm": tol, "tolerance_source": tol_source, "merge_planes": bool(merge_planes), "schema": schema,
                           "part_only": bool(part_only)},
              "simplify": simp, "ocp": None if estimate_only else ocp_version()}
    if estimate_only:
        report["estimate"] = estimate_costs(len(F), len(F0))
        report["timing_s"] = {**ph.seconds, "total": ph.total()}
        report["cpu_s"] = ph.cpu()
        report["peak_rss_bytes"] = peak_rss_info()[0]
        return report
    say(f"topopt step: building the solid from {len(F):,} triangles")
    shells = shells0 if V is V0 else mesh_shells(V, F)
    wrapped_volume = shells["volume_mm3"]
    shape, build = build_brep(ocp, V, F, shells)
    ph.done("build")
    merge = {"applied": False}
    if merge_planes:
        say("topopt step: merging coplanar faces")
        faces_before = count_shapes(ocp, shape, ocp.TopAbs_FACE)
        shape = _merged(ocp, shape)
        merge = {"applied": True, "faces_before": faces_before, "faces_after": count_shapes(ocp, shape, ocp.TopAbs_FACE)}
        ph.done("merge")
    say("topopt step: checking the model in memory")
    build["in_memory"] = check_in_memory(ocp, shape)
    ph.done("check_before_write")
    say(f"topopt step: writing {target}")
    product = info["name"] + (" (part only)" if part_only else "")
    try:
        wrote = write_step(ocp, shape, target, schema, product)
    except (OSError, RuntimeError) as exc:
        raise UserError(f"the STEP file could not be written to {target}: {exc}. Check the path and the free space, or give another "
                        "--out") from None
    del shape
    gc.collect()
    ph.done("write")
    say("topopt step: reading the file back and checking it")
    back, read_fact = read_back(ocp, target)
    header = read_header_facts(target)
    ph.done("read_back")
    model, volume_back, valid, dev, invalid = {}, None, None, None, None
    if back is not None:
        analyzer = ocp.BRepCheck_Analyzer(back)
        valid = bool(analyzer.IsValid())
        if not valid:
            invalid = invalid_parts(ocp, analyzer, back)
        del analyzer
        ph.done("check")
        model = model_facts(ocp, back)
        volume_back = shape_volume(ocp, back)
        say("topopt step: measuring the distance to the STL")
        Vs, Fs = shape_triangles(ocp, back)
        dev = deviation(Vs, Fs, V0, F0)
        ph.done("deviation")
    checks = _checks(read_fact, valid, model, header, schema, build, invalid)
    failed = [k for k, c in checks.items() if c["state"] == "fail"]
    notes = list(wrote["rejected"])
    if header["product_name"] is not None and not header["product_name"].isascii():
        notes.append("the product name has characters outside ASCII. OpenCASCADE writes them as UTF-8 bytes, and ISO 10303-21 "
                     "writes them as \\X2\\ codes, so another reader can show the name wrong")
    report_path = report_path_of(target)
    report.update({
        "output": {"path": str(target), "bytes": wrote["bytes"], "schema": header["schema"], "schema_text": header["schema_text"],
                   "unit": read_fact["unit"], "product_name": header["product_name"], "notes": notes},
        "merge_planes": merge, "build": build, "model": model, "invalid_parts": invalid,
        "volume": _volume_facts(stl_volume, wrapped_volume, volume_back),
        "deviation": dev, "checks": checks, "checks_failed": failed,
        "options": _options(failed, checks, merge_planes, tol), "timing_s": {**ph.seconds, "total": ph.total()},
        "cpu_s": ph.cpu(), "peak_rss_bytes": peak_rss_info()[0], "peak_rss_mb_after_phase": ph.peak_mb})
    try:
        write_atomic(report_path, (json.dumps(report, indent=1) + "\n").encode())
    except OSError as exc:
        raise UserError(f"the STEP file is written ({target}), but the report could not be written to {report_path}: {exc}") from None
    report["report_path"] = str(report_path)
    return report


def _merged(ocp, shape):
    return merge_planes(ocp, shape)


def check_in_memory(ocp: SimpleNamespace, shape) -> dict:
    """The model before it is written: `BRepCheck_Analyzer` on the shape in memory, and the volume of each solid. The analyzer does
    not see a solid with its faces turned inward (it says valid), but the volume of such a solid is negative. Returns `valid`,
    `solids`, `smallest_solid_mm3` and `invalid_parts` (None when valid)."""
    o = ocp
    analyzer = o.BRepCheck_Analyzer(shape)
    valid = bool(analyzer.IsValid())
    invalid = None if valid else invalid_parts(o, analyzer, shape)
    del analyzer
    solids = o.TopTools_IndexedMapOfShape()
    o.TopExp.MapShapes_s(shape, o.TopAbs_SOLID, solids)
    volumes = [shape_volume(o, solids.FindKey(i)) for i in range(1, solids.Extent() + 1)]
    return {"valid": valid, "solids": len(volumes), "smallest_solid_mm3": min(volumes) if volumes else None, "invalid_parts": invalid}


def _volume_facts(stl, wrapped, back) -> dict:
    """The volumes. `roundtrip_error` is a number and has no state: the read-back volume against the volume of the wrapped mesh,
    relative (the measured range is in CONTRACTS.md section 42.4)."""
    out = {"stl_mm3": stl, "wrapped_mm3": wrapped, "step_mm3": back}
    if back is not None and stl > 0:
        out["step_vs_stl_mm3"] = back - stl
        out["step_vs_stl_pct"] = 100.0 * (back - stl) / stl
        out["roundtrip_error"] = abs(back - wrapped) / max(abs(wrapped), 1e-300)
    return out


def _checks(read_fact, valid, model, header, schema, build, invalid=None) -> dict:
    """The checks, each `{"state": "pass" | "fail" | "unknown", "reason": text}`. `before_write` is about the model in memory; the others
    are about the file read back. None of them has a threshold: a count is equal or not, a sign is positive or not, a text is the
    one that was asked for or not."""
    def c(ok, reason):
        return {"state": "unknown" if ok is None else ("pass" if ok else "fail"), "reason": reason}
    mem = build["in_memory"]
    small = mem["smallest_solid_mm3"]
    positive = small is not None and small > 0
    before = c(mem["valid"] and mem["solids"] >= 1 and positive,
               f"BRepCheck_Analyzer says the model in memory is {'valid' if mem['valid'] else 'not valid'}; it has {mem['solids']} solid(s)"
               + ("" if small is None else f", the smallest volume of a solid is {small:,.3f} mm3 "
                  + ("(positive)" if positive else "(not positive: a solid with its faces turned inward has a negative volume)"))
               + _invalid_text(mem["invalid_parts"]))
    if not read_fact.get("read"):
        return {"read_back": c(False, f"OpenCASCADE could not read the file back ({read_fact.get('status')})"), "before_write": before}
    free, over = model["free_edges"], model["overshared_edges"]
    expected = build["bodies"] + build["voids"]
    left_out = build["degenerate_triangles_left_out"]
    return {
        "read_back": c(True, "OpenCASCADE reads the file"),
        "topology": c(model["solids"] == build["bodies"] and model["shells"] == expected,
                      f"the file holds {model['solids']} solid(s) and {model['shells']} shell(s); the build made {build['bodies']} "
                      f"solid(s) and {expected} shell(s) ({build['bodies']} bod{'y' if build['bodies'] == 1 else 'ies'} and "
                      f"{build['voids']} void(s))"),
        "before_write": before,
        "valid_solid": c(valid and model["solids"] >= 1,
                         f"BRepCheck_Analyzer says {'valid' if valid else 'not valid'} on the file as OpenCASCADE reads it (the "
                         f"reader turns faces that point the wrong way); the file holds {model['solids']} solid(s) and "
                         f"{model['shells']} shell(s)" + ("" if model["solids"] else " (a shell that is not closed is not a solid)")
                         + _invalid_text(invalid)),
        "closed": c(free == 0 and over == 0,
                    f"{free} edge(s) with 1 face, {over} edge(s) with more than 2 faces"
                    + (f". {left_out} triangle(s) with no area were left out of the model, and the free edges can come from them"
                       if left_out and free else "")),
        "units": c(read_fact["unit"] == "mm", f"the length unit of the file is {read_fact['unit'] or 'not found'}"),
        "schema": c(header["schema"] == schema, f"the schema of the file is {header['schema'] or 'not found'} ({header['schema_text']}); asked {schema}"),
    }


def invalid_parts(ocp: SimpleNamespace, analyzer, shape, keep: int = 5) -> dict:
    """Which parts of a model `BRepCheck_Analyzer` flags: the number of faces, edges and vertices, the statuses that OpenCASCADE gives,
    and the area (mm2) and status of the first `keep` flagged faces. Called only when the model is not valid; the analyzer has the
    results already, so this is a lookup."""
    o = ocp
    out = {"vertices": 0, "edges": 0, "faces": 0, "statuses": [], "smallest_face_area_mm2": None, "first_faces": []}
    for kind, key in ((o.TopAbs_VERTEX, "vertices"), (o.TopAbs_EDGE, "edges"), (o.TopAbs_FACE, "faces")):
        parts = o.TopTools_IndexedMapOfShape()
        o.TopExp.MapShapes_s(shape, kind, parts)
        for i in range(1, parts.Extent() + 1):
            part = parts.FindKey(i)
            if analyzer.IsValid(part):
                continue
            out[key] += 1
            result = analyzer.Result(part)
            names = sorted({str(x).rsplit(".", 1)[-1].replace("BRepCheck_", "") for x in result.Status()}) if result is not None else []
            names = [n for n in names if n != "NoError"]
            for n in names:
                if n not in out["statuses"]:
                    out["statuses"].append(n)
            if key == "faces":
                props = o.GProp_GProps()
                o.BRepGProp.SurfaceProperties_s(o.TopoDS.Face_s(part), props)
                area = float(props.Mass())
                if out["smallest_face_area_mm2"] is None or area < out["smallest_face_area_mm2"]:
                    out["smallest_face_area_mm2"] = area
                if len(out["first_faces"]) < keep:
                    out["first_faces"].append({"area_mm2": area, "statuses": names})
    return out


def _invalid_text(invalid) -> str:
    if not invalid:
        return ""
    names = {"faces": "face(s)", "edges": "edge(s)", "vertices": "vertex(es)"}
    bits = [f"{invalid[k]:,} {names[k]}" for k in ("faces", "edges", "vertices") if invalid[k]]
    text = f"; flagged: {', '.join(bits) if bits else 'no face, edge or vertex (the solid or its shell)'}"
    if invalid["statuses"]:
        text += f" ({', '.join(invalid['statuses'])})"
    if invalid["smallest_face_area_mm2"] is not None:
        text += f"; the smallest flagged face has {invalid['smallest_face_area_mm2']:.3g} mm2"
    return text


def _options(failed, checks, merged, tolerance_mm=0.0) -> list:
    """What the user can do about a failed check: facts and options, no refusal."""
    out = [f"check {k} failed: {checks[k]['reason']}" for k in failed]
    if failed:
        out.append("the file is written as it is")
        if tolerance_mm <= 0:
            out.append("option: a tolerance above 0 (the default is the voxel size / 20) joins short edges and removes slivers before the faces are made")
        else:
            out.append("option: a different --tolerance-mm changes which short edges and slivers stay")
        if merged:
            out.append("option: --no-merge-planes writes one face for each triangle")
        else:
            out.append("option: leave out --no-merge-planes: coplanar triangles then join into one face")
    return out


def estimate_costs(faces: int, triangles_in: int) -> dict:
    """What the command is expected to cost for a mesh of `triangles_in` triangles that has `faces` faces after the simplification.

    `faces` is an upper bound for the model: the merge of coplanar faces removes 0.7 to 4 percent more on the public parts. The
    numbers are the fit named at `ESTIMATE_SECONDS`; the time depends most on the machine and its load.
    """
    n, t = float(faces), float(triangles_in)
    s0, s1, s2 = ESTIMATE_SECONDS
    m0, m1, m2 = ESTIMATE_PEAK_BYTES
    return {"faces": int(faces), "step_bytes": int(ESTIMATE_STEP_BYTES_PER_FACE * n), "seconds": s0 + s1 * n + s2 * t,
            "peak_rss_bytes": int(m0 + m1 * n + m2 * t),
            "basis": (f"a fit to {MEASURED_RUNS} measured runs on Linux x86_64 (CONTRACTS.md section 42.7): a file is about "
                      f"{ESTIMATE_STEP_BYTES_PER_FACE / 1000:.1f} KB for each face, the time {s1 * 1000:.2f} ms for each face and "
                      f"{s2 * 1e6:.1f} us for each STL triangle plus {s0:.1f} s, the peak memory {m1 / 1000:.0f} KB for each face and "
                      f"{m2 / 1000:.2f} KB for each STL triangle plus {m0 / 1e6:.0f} MB; the largest error of the fit on "
                      f"those runs was {ESTIMATE_MAX_ERROR_PCT} percent (the runs are the data of the fit), the worst errors are estimates "
                      "that are too low, and the time on another machine can differ more")}


# --- the text ------------------------------------------------------------------------------------------------------------------


def exit_code(report: dict) -> int:
    """0: written and every check passed (or an estimate). `EXIT_CHECK_FAILED`: written and a check of the read-back failed."""
    return EXIT_CHECK_FAILED if report.get("checks_failed") else 0


def _mb(n) -> str:
    return f"{n / 1e6:,.1f} MB" if n < 1e9 else f"{n / 1e9:,.2f} GB"


def _rel(path, base) -> str:
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)


def format_text(report: dict) -> str:
    """The report as text for the terminal: facts first, then the failed checks and the options."""
    inp, st, sp = report["input"], report["settings"], report["simplify"]
    base = report["run_dir"]
    lines = []
    est = report["state"] == "estimate"
    lines.append("topopt step: estimate only, nothing is written" if est else
                 f"topopt step: {_rel(report['output']['path'], base)} ({_mb(report['output']['bytes'])}, "
                 f"{report['output']['schema'] or '?'}, {report['output']['unit'] or '?'})")
    lines.append(f"  mesh: result/{report['mesh']}, {inp['triangles']:,} triangles, {inp['volume_mm3']:,.1f} mm3, "
                 f"{'closed' if inp['closed'] else 'NOT closed (' + str(inp['open_edges']) + ' open and ' + str(inp['nonmanifold_edges']) + ' non-manifold edges)'}")
    if sp["applied"]:
        lines.append(f"  simplify: tolerance {st['tolerance_mm']:g} mm ({st['tolerance_source']}): {sp['triangles_before']:,} -> {sp['triangles_after']:,} triangles")
    else:
        lines.append(f"  simplify: none ({sp.get('note') or 'tolerance 0'})")
    if est:
        e = report["estimate"]
        lines.append(f"  faces: about {e['faces']:,}" + ("" if "step_bytes" not in e else
                     f"; STEP file about {_mb(e['step_bytes'])}, time about {e['seconds']:,.0f} s, peak memory about {_mb(e['peak_rss_bytes'])}"))
        if e.get("basis"):
            lines.append(f"  basis: {e['basis']}")
        return "\n".join(lines)
    mp = report["merge_planes"]
    lines.append(f"  merge planes: {mp['faces_before']:,} -> {mp['faces_after']:,} faces" if mp["applied"] else "  merge planes: off")
    md = report["model"]
    if md:
        lines.append(f"  read back: {md['solids']} solid(s), {md['faces']:,} faces ({md['planar_faces']:,} planar), {md['edges']:,} edges, "
                     f"{md['vertices']:,} vertices")
    v = report["volume"]
    if v.get("step_mm3") is not None:
        lines.append(f"  volume: {v['step_mm3']:,.1f} mm3 against {v['stl_mm3']:,.1f} mm3 of the STL ({v['step_vs_stl_pct']:+.3f} %)")
    d = report["deviation"]
    if d:
        a, b = d["step_to_stl"], d["stl_to_step"]
        lines.append(f"  distance to the STL (sampled): STEP to STL max {a['max_mm']:.4f} mm, mean {a['mean_mm']:.4f} mm; "
                     f"STL to STEP max {b['max_mm']:.4f} mm, mean {b['mean_mm']:.4f} mm")
    lines.append("  checks: " + ", ".join(f"{k.replace('_', ' ')} {c['state']}" for k, c in report["checks"].items()))
    t = report["timing_s"]
    lines.append(f"  time {t['total']:.1f} s, CPU {report['cpu_s']:.1f} s ({', '.join(f'{k} {x:.1f}' for k, x in t.items() if k != 'total')}); "
                 f"peak memory {_mb(report['peak_rss_bytes'])}")
    for k in report["checks_failed"]:
        lines.append(f"  FAILED {k}: {report['checks'][k]['reason']}")
    for line in report["options"]:
        if not line.startswith("check "):
            lines.append(f"  {line}")
    for note in report["output"].get("notes") or []:
        lines.append(f"  note: {note}")
    lines.append(f"  report: {_rel(report['report_path'], base)}")
    return "\n".join(lines)
