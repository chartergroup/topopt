"""STL -> tetrahedra with gmsh (CONTRACTS.md section 14).

Ported from the files `pipeline/lib/p2_mesh.py` and `build_mesh` in
`pipeline/s5_verify/p2_verify.py` (commit d43bcf3), without the code of that project.

The steps of `mesh_stl`:

1. Check the probe. If gmsh is not `ok`, raise `TetUnavailable` (one clear error).
2. `prepare_surface`: coarsen the STL to an edge length of `size`, and keep it a closed manifold.
3. gmsh meshes the prepared surface as a fixed boundary (HXT first, Delaunay as the fallback).
4. If gmsh reports a self-intersection, prepare the surface again with a small edge length
   around the reported points, and try again (up to `max_attempts` times).

This module imports gmsh only inside functions, and only after the probe says `ok`.

One mesh at a time per process: gmsh keeps one global state, and `capture_fds` redirects the
file descriptors of the whole process. A module lock (`GMSH_LOCK`) serializes the calls, so
two threads that call `mesh_stl` run one after the other. `mesh_stl` also works off the main thread
(gmsh is started with `interruptible=False`, because the interrupt handler needs the main thread).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import UserError
from . import gmsh_probe as _probe
from .gmsh_probe import GMSH_LOCK, INIT_ARGS

EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

# The quality cut that the P2 report uses (tets below it are "slivers").
SLIVER_QUALITY = 0.05
# Quality below this is called "low" in the facts. It is the mean-ratio quality (1 = regular tet).
LOW_QUALITY = 0.1
# A mesh that fills the surface has the same volume. The pipeline accepts 1e-3.
VOLUME_TOLERANCE = 1e-3


class MeshingFailed(UserError):
    """gmsh could not mesh the surface. `points` are the places where gmsh reported a segment-facet intersection.

    `crossings` (only when `mesh_stl` looked for them, P4): `prepared` and `input`.
    Each is a dict with `count` (the pairs of faces that cross) and `points` (all the places, as lists of 3 mm values).
    `prepared` also has `attempts` (the count of each attempt).
    `stopped` says why the retries ended (`input_crosses_itself`, `no_progress`), or it is absent.
    `same_surface_as_attempt` (with `stopped` `no_progress`, only when it applies) is the attempt whose prepared surface is equal to the
    one of the failed attempt: the refinement did not change the surface, so the next attempt would repeat this one.
    `search_failed` is a list with one entry (`attempt`, `surface`, `error`) for each search that could not run (for example out of memory).
    `surface_retry` (F1-5) is present when the prepared surface of the failed attempt came from a retry of the surface preparation: the
    step of `PreparedSurface.steps` (`weld_short_edges` or `clean_short_edges`, JSON-safe). The text of the error names the retry too.
    """

    def __init__(self, message: str, points=None, crossings: dict | None = None):
        super().__init__(message)
        self.points = list(points or [])
        self.crossings = crossings


class _FlatTets(MeshingFailed):
    """gmsh meshed the prepared surface, and some tets have a determinant of 0 (F4-27). The attempt fails like one with a crossing, but
    the places are the centres of the tets, not crossings of the surface (review MN4)."""


class MeshingCrashed(MeshingFailed):
    """The gmsh child process died on a signal (for example SIGSEGV). `crash` has the facts.

    A crash cannot be caught in the process that runs gmsh, so `mesh_stl` runs gmsh in a child process
    by default. `crashes` lists every crash of one `mesh_stl` call (signal, size, optimize, attempt).
    """

    def __init__(self, message: str, crash: dict | None = None, crashes: list | None = None):
        super().__init__(message, [])
        self.crash = dict(crash or {})
        self.crashes = list(crashes if crashes is not None else ([self.crash] if crash else []))


@dataclass
class TetMesh:
    """A tetrahedral mesh and the facts about it.

    `nodes` (n, 3) float64, in mm, in the frame of the STL. `tets` (m, 4) int64: the node
    indices of each tet, with positive signed volume. `facts` is a dict (see `mesh_stl`).
    """
    nodes: np.ndarray
    tets: np.ndarray
    facts: dict = field(default_factory=dict)

    def volumes(self) -> np.ndarray:
        """Volume of each tet in mm^3 (all positive)."""
        return signed_volumes(self.nodes, self.tets)

    def quality(self) -> np.ndarray:
        """Mean-ratio quality of each tet: 1 for a regular tet, 0 for a flat one."""
        return tet_quality(self.nodes.T, self.tets.T)

    def sha256(self) -> str:
        """Hash of the nodes and the tets. Two equal meshes have the same hash."""
        return mesh_hash(self.nodes, self.tets)

    def save(self, path) -> Path:
        """Write `nodes` (float64), `tets` (int64) and `facts` (as JSON text) to an .npz file.

        The suffix `.npz` is added if the path has none. Return the path of the file.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        np.savez_compressed(path, nodes=np.asarray(self.nodes, np.float64),
                            tets=np.asarray(self.tets, np.int64), facts=np.array(json.dumps(self.facts)))
        return path

    @classmethod
    def load(cls, path) -> "TetMesh":
        """Read a file that `save` wrote. The hash of the loaded mesh equals `facts["sha256"]`."""
        with np.load(Path(path)) as z:
            return cls(np.asarray(z["nodes"], np.float64), np.asarray(z["tets"], np.int64),
                       json.loads(str(z["facts"])))


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def signed_volumes(nodes: np.ndarray, tets: np.ndarray, chunk: int = 400_000) -> np.ndarray:
    """Signed volume of each tet: positive if the node order (0,1,2,3) is right-handed."""
    out = np.empty(len(tets))
    for s in range(0, len(tets), chunk):
        X = nodes[tets[s:s + chunk]]                       # (k, 4, 3)
        out[s:s + chunk] = np.linalg.det(X[:, 1:, :] - X[:, :1, :]) / 6.0
    return out


def zero_volume_centres(p: np.ndarray, t: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """The centres (k, 3) of the tets that have no volume or that the stress check of `topopt verify` cannot invert. `p` is (3, nv)
    and `t` is (4, nel), the layout of the pipeline. A tet is listed when one of two determinants is exactly 0 (no limit):
    the 3 x 3 matrix of the edge vectors (`signed_volumes`: the tet has no volume) or the 4 x 4 matrix `[1 x y z]` that
    `verify.p2.tet_geometry` inverts (`numpy.linalg.inv` raises "Singular matrix" for the same tets, F4-27: f6a at 0.973 mm).

    The two tests are not the same in floating point (review MN1). Of 20,000 random tets in a tilted plane, `inv` refuses 5,230. The
    3 x 3 determinant is exactly 0 for 74 of them (it misses 5,156), and it is exactly 0 for 37 more tets that `inv` inverts (their
    gradients are rounding noise). The first kind stopped `verify` with an internal error. The second kind is a tet of no volume: it is listed too. With only the 4 x 4
    test, f6a at 0.973 mm (`Gmsh+Netgen`, as `verify` uses) lost one of its three listed tets, the refinement of the next attempts
    changed, and the part was refused at attempt 4."""
    flat = []
    for s in range(0, t.shape[1], chunk):
        X = p[:, t[:, s:s + chunk]].transpose(2, 1, 0)                   # (m, 4, 3)
        A = np.ones((X.shape[0], 4, 4))
        A[:, :, 1:] = X
        none = (np.linalg.det(A) == 0.0) | (np.linalg.det(X[:, 1:, :] - X[:, :1, :]) == 0.0)
        flat.append(s + np.flatnonzero(none))
    flat = np.concatenate(flat) if flat else np.empty(0, np.int64)
    return p[:, t[:, flat]].mean(axis=1).T if len(flat) else np.empty((0, 3))


def tet_quality(p: np.ndarray, t: np.ndarray, chunk: int = 400_000) -> np.ndarray:
    """Mean-ratio quality 12 (3V)^(2/3) / sum(l^2): 1 for a regular tet, 0 for a flat one.

    `p` is (3, nv) and `t` is (4, nel), the layout of the pipeline.
    """
    n = t.shape[1]
    q = np.empty(n)
    for s in range(0, n, chunk):
        X = p[:, t[:, s:s + chunk]].transpose(2, 1, 0)
        vol = np.abs(np.linalg.det(X[:, 1:, :] - X[:, :1, :])) / 6.0
        l2 = sum(((X[:, a] - X[:, b]) ** 2).sum(axis=1) for a, b in EDGE_PAIRS)
        q[s:s + chunk] = 12.0 * (3.0 * vol) ** (2.0 / 3.0) / np.maximum(l2, 1e-300)
    return q


def mesh_hash(nodes: np.ndarray, tets: np.ndarray) -> str:
    """sha256 of the node coordinates (float64) and the tet indices (int64), as hex."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(nodes, dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(tets, dtype="<i8").tobytes())
    return h.hexdigest()


def surface_volume(V: np.ndarray, F: np.ndarray) -> float:
    """Volume inside a closed triangle surface (positive for outward normals)."""
    return float(np.einsum("ij,ij->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])) / 6.0)


# --------------------------------------------------------------------------- #
# gmsh calls
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def capture_fds():
    """Capture what C code writes to stdout and stderr (gmsh prints PLC errors with printf)."""
    sys.stdout.flush()
    sys.stderr.flush()
    tmp = tempfile.TemporaryFile(mode="w+b")
    saved = (os.dup(1), os.dup(2))
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    box = {"text": ""}
    try:
        yield box
    finally:
        try:
            import ctypes
            ctypes.CDLL(None).fflush(None)             # C stdio buffers the printf output of gmsh
        except Exception:
            pass
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        tmp.seek(0)
        box["text"] = tmp.read().decode(errors="ignore")
        tmp.close()


_POINT = re.compile(r"intersect at point \(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\)")


def parse_intersection_points(text: str):
    return [tuple(float(g) for g in m.groups()) for m in _POINT.finditer(text)]


def write_ascii_stl(path, V: np.ndarray, F: np.ndarray):
    """ASCII STL with 17 significant digits (a binary STL would round the vertices to float32)."""
    P = V[F]                                                    # (nf,3,3)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    with open(path, "w") as f:
        f.write("solid part\n")
        for i in range(len(F)):
            f.write("facet normal %.9g %.9g %.9g\n outer loop\n" % tuple(n[i]))
            for k in range(3):
                f.write("  vertex %.17g %.17g %.17g\n" % tuple(P[i, k]))
            f.write(" endloop\nendfacet\n")
        f.write("endsolid part\n")


def parse_optimize(optimize: str | None) -> list[str]:
    """The optimizer steps of an `optimize` setting: None or "" is none, "A+B" is two steps in this order.

    "Gmsh" means the default optimizer of gmsh during meshing (`Mesh.Optimize = 1`). Any other name goes to
    `gmsh.model.mesh.optimize` after meshing (for example "Netgen" or "Relocate3D").
    """
    return [x.strip() for x in (optimize or "").split("+") if x.strip()]


def _mesh_once(V, F, algorithm: int, size_max: float, threads: int, optimize: str | None, log: bool, workdir: str):
    import gmsh
    stl = os.path.join(workdir, "surface_for_gmsh.stl")
    write_ascii_stl(stl, V, F)
    try:
        gmsh.initialize(**INIT_ARGS)
        gmsh.option.setNumber("General.Terminal", 1 if log else 0)
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
        gmsh.model.add("part")
        gmsh.merge(stl)
        surfs = gmsh.model.getEntities(2)
        loop = gmsh.model.geo.addSurfaceLoop([e[1] for e in surfs])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        steps = parse_optimize(optimize)
        gmsh.option.setNumber("Mesh.Optimize", 1 if "Gmsh" in steps else 0)
        gmsh.model.mesh.generate(3)
        for name in steps:
            if name != "Gmsh":
                gmsh.model.mesh.optimize(name)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        etypes, etags, enodes = gmsh.model.mesh.getElements(3)
        if len(etypes) != 1 or etypes[0] != 4:
            raise MeshingFailed(f"unexpected element types {list(etypes)}")
        tets = np.asarray(enodes[0], np.int64).reshape(-1, 4)
        ntags = np.asarray(ntags, np.int64)
        P = np.asarray(coords, float).reshape(-1, 3)
        index = np.full(int(ntags.max()) + 1, -1, np.int64)
        index[ntags] = np.arange(len(ntags))
        T = index[tets]
        used = np.unique(T)
        remap = np.full(len(P), -1, np.int64)
        remap[used] = np.arange(len(used))
        return P[used].T.copy(), remap[T].T.copy()
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
        try:
            os.remove(stl)
        except OSError:
            pass


def _try_algorithms(V, F, size, threads, max_size_factor, optimize, algorithms, log, workdir, errors, points):
    """Mesh the closed surface (V, F) with the first of `algorithms` that gives a mesh that fills the surface. Return `(p, t, info)`,
    or None when no algorithm does: the failures are then added to `errors` (text) and `points` (places that gmsh named)."""
    for algo in algorithms:
        t0 = time.time()
        with GMSH_LOCK, capture_fds() as cap:
            try:
                p, t = _mesh_once(V, F, algo, max_size_factor * size, threads, optimize, log, workdir)
                exc = None
            except Exception as e:                            # gmsh raises plain Exception
                exc = e
        if exc is not None:
            errors.append(f"algorithm {algo}: {exc}")
            points.extend(parse_intersection_points(cap["text"]))
            continue
        info = dict(algorithm=int(algo), seconds=time.time() - t0, optimize=optimize,
                    size_max_mm=max_size_factor * size, failed_algorithms=errors)
        # a valid mesh fills the surface volume
        vol_tets = float(np.abs(np.linalg.det((p[:, t[1:]] - p[:, t[:1]]).transpose(2, 1, 0))).sum() / 6.0)
        vol_surf = surface_volume(V, F)
        info["volume_tets_mm3"] = vol_tets
        info["volume_surface_mm3"] = vol_surf
        info["volume_ratio"] = vol_tets / vol_surf
        if abs(info["volume_ratio"] - 1.0) > VOLUME_TOLERANCE:
            errors.append(f"algorithm {algo}: tet volume differs from the surface volume by "
                          f"{info['volume_ratio'] - 1:+.2e}")
            continue
        return p, t, info
    return None


def mesh_surface(V: np.ndarray, F: np.ndarray, size: float, threads: int = 1,
                 max_size_factor: float = 1.5, optimize: str | None = None,
                 algorithms=(10, 1), log: bool = False, workdir: str | None = None, shell_retry: bool = True):
    """Tet mesh of the closed surface (V,F). Returns (p (3,nv), t (4,nel), info).

    Low-level function of the pipeline layout. Most callers use `mesh_stl`. It does not check the probe.
    Raise `MeshingFailed` if no algorithm gives a mesh that fills the surface.

    gmsh gets the whole surface as one volume. If every algorithm fails, gmsh named no crossing, and the surface has more than one
    closed shell, `shell_relations` looks at where the shells lie (F4-45). A shell that lies in the material of another shell adds
    no boundary to the solid, but the volume of the surface counts that material twice, and HXT does not take a second shell. With
    `shell_retry` these shells are left out and the meshing is tried again; `info["shell_retry"]` has the facts. The error of a
    surface that still fails names the shells. A surface that one try meshes is meshed as before. The facts are the same.
    """
    own_dir = None
    if workdir is None:
        own_dir = tempfile.TemporaryDirectory(prefix="topopt_tet_")
        workdir = own_dir.name
    errors = []
    points = []
    try:
        done = _try_algorithms(V, F, size, threads, max_size_factor, optimize, algorithms, log, workdir, errors, points)
        if done is not None:
            return done
        shells = None                                                    # gmsh named a crossing: the shells are not the first cause
        if not points:
            try:
                shells = shell_relations(V, F)
            except Exception:  # noqa: BLE001  (a help to the error text: it must not replace the error of gmsh)
                shells = None
        if shells:
            nested = [k for k, s in enumerate(shells) if s["kind"] == "nested"]
            if nested and shell_retry:
                keep = np.ones(len(F), bool)
                for k in nested:
                    keep[shells[k]["faces"]] = False
                again: list = []
                done = _try_algorithms(V, F[keep], size, threads, max_size_factor, optimize, algorithms, log, workdir, again, points)
                if done is not None:
                    p, t, info = done
                    info["shell_retry"] = dict(
                        step="drop_nested_shells", shells_in_surface=len(shells), volume_before_mm3=float(surface_volume(V, F)),
                        volume_after_mm3=float(info["volume_surface_mm3"]),
                        left_out=[{k2: s[k2] for k2 in ("shell", "volume_mm3", "faces_count", "bbox_min_mm", "bbox_max_mm")}
                                  for s in (shells[k] for k in nested)],
                        whole_surface_errors=list(errors))
                    return p, t, info
                errors.append("after the nested shell(s) were left out: " + "; ".join(again))
            if any(x["kind"] != "outer" for x in shells):          # plain bodies that are apart need no sentence
                errors.append(shells_sentence(shells, left_out=bool(nested and shell_retry)))
        raise MeshingFailed("; ".join(errors), points)
    finally:
        if own_dir is not None:
            own_dir.cleanup()


# --------------------------------------------------------------------------- #
# A surface of several closed shells (F4-45)
# --------------------------------------------------------------------------- #

def winding_number(V: np.ndarray, F: np.ndarray, q, chunk: int = 200_000) -> float:
    """The winding number of the closed triangle surface (V, F) at the point `q`: the sum of the signed solid angles of the faces divided
    by 4 pi. It is 1 inside a body with outward normals, -1 inside a body with inward normals and 0 outside; it is an integer at a
    point that is not on the surface (the point of a vertex of the surface itself gives a half turn: do not ask for that)."""
    q = np.asarray(q, float)
    total = 0.0
    for s in range(0, len(F), chunk):
        P = V[F[s:s + chunk]] - q
        a, b, c = P[:, 0], P[:, 1], P[:, 2]
        la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
        top = np.einsum("ij,ij->i", a, np.cross(b, c))
        bottom = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", a, c) * lb
                  + np.einsum("ij,ij->i", b, c) * la)
        total += 2.0 * np.arctan2(top, bottom).sum()
    return float(total / (4.0 * np.pi))


def shell_relations(V: np.ndarray, F: np.ndarray) -> list:
    """Where the closed shells of a surface (V, F) lie. An empty list when the surface has one shell only.

    A shell is a set of faces that are linked through shared vertices. Each shell is a dict: `shell` (1 to n, the order of the lowest
    vertex), `faces` (the face indices), `faces_count`, `volume_mm3` (signed: negative for a shell with inward normals), `bbox_min_mm`,
    `bbox_max_mm`, `others` (the winding number of all the other shells at one vertex of this shell, rounded to the nearest whole
    number: no limit is used) and `kind`:

    - `outer`: a body with outward normals that is outside every other shell, or inside a cavity of one (`others` is 0);
    - `nested`: a body with outward normals in the material of another body (`others` is 1 or more). The solid is the same with and
      without it; the volume of the surface (the sum of the signed volumes) counts that material twice;
    - `cavity`: a shell with inward normals in the material of one body (`others` is 1);
    - `other`: any other place. This happens when shells cross each other; gmsh names such crossings, and `mesh_surface` then does
      not look at the shells.
    """
    from .surface import _body_euler
    body, _chi = _body_euler(F, len(V))
    count = int(body.max()) + 1 if len(F) else 0
    if count < 2:
        return []
    of_face = body[F[:, 0]]
    shells = []
    for b in range(count):
        faces = np.flatnonzero(of_face == b)
        used = V[np.unique(F[faces])]
        shells.append(dict(shell=b + 1, faces=faces, faces_count=int(len(faces)), volume_mm3=float(surface_volume(V, F[faces])),
                           bbox_min_mm=[float(x) for x in used.min(axis=0)], bbox_max_mm=[float(x) for x in used.max(axis=0)],
                           q=V[F[faces[0], 0]]))
    lo = np.array([s["bbox_min_mm"] for s in shells])
    hi = np.array([s["bbox_max_mm"] for s in shells])
    for i, s in enumerate(shells):
        # a point outside the box of a shell has the winding number 0 for that shell: the sum is left out (no change of the result)
        near = [j for j in range(count) if j != i and np.all(s["q"] >= lo[j]) and np.all(s["q"] <= hi[j])]
        s["others"] = int(np.rint(sum(winding_number(V, F[shells[j]["faces"]], s["q"]) for j in near)))
        if s["volume_mm3"] > 0:
            s["kind"] = "outer" if s["others"] == 0 else "nested" if s["others"] > 0 else "other"
        else:
            s["kind"] = "cavity" if s["others"] == 1 else "other"
        del s["q"]
    return shells


def shells_sentence(shells: list, left_out: bool = False) -> str:
    """The sentence for a `MeshingFailed` about the shells of the surface: how many, their volumes, and what each shell that is not a
    plain body does to the mesh. It ends with the options. There is no full stop at the end."""
    n = len(shells)
    names = ", ".join(f"{s['volume_mm3']:.4g}" for s in shells[:6]) + (", ..." if n > 6 else "")
    parts = []
    for s in shells:
        lo, hi = s["bbox_min_mm"], s["bbox_max_mm"]
        box = ", ".join(f"{a} {lo[i]:.4g} to {hi[i]:.4g}" for i, a in enumerate("xyz"))
        if s["kind"] == "nested":
            parts.append(f"shell {s['shell']} ({s['volume_mm3']:.4g} mm3, {box} mm) lies in the material of another shell, so the volume "
                         "of the surface counts that material twice and HXT does not take a second shell")
        elif s["kind"] == "cavity":
            parts.append(f"shell {s['shell']} ({s['volume_mm3']:.4g} mm3, {box} mm) is a cavity (an enclosed void), and gmsh fills a "
                         "cavity of the one volume it gets")
        elif s["kind"] == "other":
            parts.append(f"shell {s['shell']} ({s['volume_mm3']:.4g} mm3, {box} mm) lies in a place that is not a body, a nested body "
                         f"or a cavity (the other shells give the winding number {s['others']} at its vertex)")
    flagged = "; ".join(parts[:3]) + ("; ..." if len(parts) > 3 else "")
    kinds = {x["kind"] for x in shells}
    options = []
    if kinds & {"nested", "other"}:
        options.append("repair the STL (remove the inner shell, or make one body with a boolean union of the bodies that overlap)")
    if "cavity" in kinds:
        options.append("fill the void in the STL if the part may be solid there (the mesh of this mesher has no void)")
    return (f"the surface has {n} closed shells (volumes {names} mm3): {flagged}"
            + ("; the nested shell(s) were left out and the surface was meshed again, and that failed too" if left_out else "")
            + f". Options: {', or '.join(options)}")


# --------------------------------------------------------------------------- #
# gmsh in a child process
# --------------------------------------------------------------------------- #

def _worker_command(in_path: str, out_path: str) -> list[str]:
    return [sys.executable, "-m", "topopt.tet.worker", in_path, out_path]


def _signal_name(returncode: int) -> str:
    import signal
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal {-returncode}"


def mesh_surface_isolated(V: np.ndarray, F: np.ndarray, size: float, threads: int = 1,
                          max_size_factor: float = 1.5, optimize: str | None = None,
                          algorithms=(10, 1), workdir: str | None = None):
    """Like `mesh_surface`, but gmsh runs in a child Python (`topopt.tet.worker`).

    This process never loads libgmsh. The surface goes in and the nodes and tets come back as .npz files in
    `workdir` (a temp folder by default). Return `(p, t, info)`; `info["seconds_isolation"]` is the time
    of the child call without the gmsh time (start of Python, import of gmsh, file transfer).
    Raise `MeshingFailed` if gmsh reports a failure, and `MeshingCrashed` if the child dies on a signal.
    """
    import subprocess
    own_dir = None
    if workdir is None:
        own_dir = tempfile.TemporaryDirectory(prefix="topopt_tet_")
        workdir = own_dir.name
    try:
        in_path, out_path = os.path.join(workdir, "child_in.npz"), os.path.join(workdir, "child_out.npz")
        params = dict(size=float(size), threads=int(threads), max_size_factor=float(max_size_factor),
                      optimize=optimize, algorithms=list(algorithms))
        np.savez(in_path, V=V, F=F, params=np.array(json.dumps(params)))
        env = dict(os.environ)
        root = str(Path(__file__).resolve().parent.parent.parent)
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        t0 = time.perf_counter()
        done = subprocess.run(_worker_command(in_path, out_path), capture_output=True, text=True, env=env)
        wall = time.perf_counter() - t0
        if done.returncode < 0:
            raise MeshingCrashed(
                f"the gmsh child process died on {_signal_name(done.returncode)} (size {size:g} mm, "
                f"optimize {optimize!r}, algorithms {tuple(algorithms)})",
                dict(signal=_signal_name(done.returncode), returncode=int(done.returncode), size=float(size),
                     optimize=optimize, algorithms=list(algorithms), threads=int(threads)))
        if not os.path.isfile(out_path):
            tail = (done.stderr.strip().splitlines() or ["no output"])[-1][:300]
            raise MeshingFailed(f"the gmsh child process failed (exit code {done.returncode}): {tail}")
        with np.load(out_path) as z:
            if "error" in z.files:
                err = json.loads(str(z["error"]))
                raise MeshingFailed(err["message"], [tuple(x) for x in err["points"]])
            p, t, info = np.asarray(z["p"]), np.asarray(z["t"]), json.loads(str(z["info"]))
        info["seconds_isolation"] = max(wall - info["seconds"], 0.0)
        return p, t, info
    finally:
        if own_dir is not None:
            own_dir.cleanup()


# --------------------------------------------------------------------------- #
# The documented API
# --------------------------------------------------------------------------- #

def _load_surface(path):
    """Read an STL as (V, F). Raise `UserError` if the file is missing or is not an STL."""
    from .surface import load_stl_surface
    path = Path(path)
    if not path.is_file():
        raise UserError(f"STL file not found: {path}")
    try:
        return load_stl_surface(str(path))
    except UserError:
        raise
    except Exception as e:                       # trimesh raises many kinds of errors
        raise UserError(f"cannot read {path} as an STL: {type(e).__name__}: {e}") from e


def _mean_edge(V: np.ndarray, F: np.ndarray) -> float:
    e = np.concatenate([np.linalg.norm(V[F[:, a]] - V[F[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))])
    return float(e.mean())


def _default_size(V: np.ndarray, F: np.ndarray) -> float:
    return 2.5 * _mean_edge(V, F)


def default_size(path) -> float:
    """The edge length `mesh_stl` uses if you give no `size`: 2.5 times the mean edge of the STL.

    The marching-cubes STL of a run (`final.stl`) has edges of about half a voxel, so the default is
    about 1.15 to 1.25 voxels for the `blur` method (measured: hook 1.496 mm with 1.2 mm voxels, ladle
    3.744 mm with 3.25 mm voxels). The `sdf` method has fewer short edges at the faces of the regions,
    so its mean edge is larger. Measured at c080f62 on one host (the default size before F4-27b, at
    de085e1, in brackets): hook 1.94 mm (1.75 mm), 1.6 voxels of 1.2 mm; ladle 4.28 mm (3.36 mm), 1.3
    voxels of 3.25 mm; bracket of `examples/interfaces` 2.16 mm (2.16 mm), 2.2 voxels of 1.0 mm. The
    number follows the part: `topopt verify` prints it as `default mesh size`. That is fine for a
    stress check. For an STL from a CAD tool, the mean edge says little about the thin members: pass
    `size`. The pipeline notes give the rule h <= d_min / 3 (d_min is the
    thinnest member) and used 2.5 mm for a larger part (not shipped).
    """
    return _default_size(*_load_surface(path))


def _near(a: np.ndarray, b: np.ndarray, distance: float, every: bool = False) -> bool:
    """Is any point of `a` (n, 3) within `distance` of a point of `b` (m, 3)? With `every`: is each point of `a`?"""
    if not len(a) or not len(b):
        return False
    from scipy.spatial import cKDTree
    close = np.isfinite(cKDTree(b).query(a, distance_upper_bound=distance)[0])
    return bool(close.all() if every else close.any())


def _crossing_facts(counts: list, prepared, inp, stopped: str | None, failed=(), retry: dict | None = None,
                    same_as: int | None = None) -> dict | None:
    """The `crossings` fact of a `MeshingFailed`.

    It has the prepared surface (the last attempt and the count of each attempt), the STL, why the retries stopped, the attempt whose
    surface was the same (`same_surface_as_attempt`), the retry of the surface preparation that made the surface (`surface_retry`, a step
    of `PreparedSurface.steps`), and the searches that failed. It is `None` if there is nothing to say. A count of `None` is an attempt
    whose search failed.
    """
    facts: dict = {}
    if prepared is not None:
        facts["prepared"] = {"count": int(prepared.total), "points": prepared.points.round(4).tolist(), "attempts": list(counts)}
    if inp is not None:
        facts["input"] = {"count": int(inp.total), "points": inp.points.round(4).tolist()}
    if stopped:
        facts["stopped"] = stopped
    if same_as is not None:
        facts["same_surface_as_attempt"] = int(same_as)
    if retry is not None:
        facts["surface_retry"] = dict(retry)
    if failed:
        facts["search_failed"] = [dict(f) for f in failed]
    return facts or None


def _surface_retry(steps) -> dict | None:
    """The step of the retry that gave the prepared surface (`weld_short_edges` or `clean_short_edges`), as a copy; `None` when the
    first try gave it."""
    return next((dict(s) for s in steps if s.get("step") in ("weld_short_edges", "clean_short_edges")), None)


def _retry_sentence(step: dict, attempt: int) -> str:
    """One sentence for a `MeshingFailed` when the surface that gmsh got came from a retry of the surface preparation (F1-5 review, MJ1).
    It names the retry and the first refusal; a reader of the error alone then knows that the surface is not the one of the first try.
    There is no full stop at the end."""
    where = "" if attempt == 1 else f" in attempt {attempt}"
    first = step.get("refused_first", "reason not recorded")
    if step["step"] == "clean_short_edges":
        n = step["short_edges_before"]
        return (f"The surface that gmsh got{where} is the prepared surface that was refused ({first}), with its {n:,} "
                f"{'edge' if n == 1 else 'edges'} under {step['limit_mm']:g} mm cleaned (clean retry)")
    n = step["short_edges_in_input"]
    return (f"The surface that gmsh got{where} is the input surface with its {n:,} {'edge' if n == 1 else 'edges'} under "
            f"{step['limit_mm']:g} mm welded and prepared again (weld retry; the first preparation was refused: {first})")


def _surface_key(V: np.ndarray, F: np.ndarray) -> str:
    """A hash of the prepared surface (the bytes of `V` and `F`), to see that two attempts got the same surface."""
    h = hashlib.sha256()
    for a, dtype in ((V, np.float64), (F, np.int64)):
        a = np.ascontiguousarray(a, dtype=dtype)
        h.update(repr(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _counts_text(counts: list) -> str:
    """The crossings of the prepared surface in each attempt, in order, for a message (`None`: the search could not run)."""
    return ", ".join("not searched" if c is None else str(c) for c in counts)


def _search_note(failed) -> str:
    """The sentence about searches for crossings that could not run."""
    what = "; ".join(f"attempt {f['attempt']}, {f['surface']} surface: {f['error']}" for f in failed)
    return ("The search for the places where the surface crosses itself could not run (" + what + "), so the refinement used the "
            "place that gmsh named")


def _search(V, F, surface: str, attempt: int, failed: list, say):
    """`find_crossings(V, F)`, or `None` if it cannot run (out of memory, an error of numba).

    The search is a help to the retry. A failure must not replace the error of gmsh.
    It is a note, and the retry refines at the places that gmsh named (P4 review, L3).
    """
    from .intersect import find_crossings
    try:
        return find_crossings(V, F)
    except Exception as err:  # noqa: BLE001  (MemoryError and every error of the compiled code)
        failed.append(dict(attempt=attempt, surface=surface, error=f"{type(err).__name__}: {str(err)[:150]}"))
        say(f"the search for crossings of the {surface} surface failed: {failed[-1]['error']}")
        return None


def _stopped(path, size, why: str, attempts: int, counts: list, prepared, inp, gmsh_points, failed=(),
             mean_edge: float | None = None, retry: dict | None = None, same_as: int | None = None,
             counted: bool = True) -> MeshingFailed:
    """The error for a retry that cannot help. `why` is `input_crosses_itself` or `no_progress`.

    `retry` is the step of the retry of the surface preparation that made the surface (or None): the text names it (F1-5 review, MJ1).
    `same_as` is the attempt whose prepared surface is equal to this one: the refinement did not change the surface (F1-5 review, MJ1).
    `counted` is True when the crossings of the prepared surface did not become fewer (the P4 rule); else only `same_as` stops the retries,
    and the surface may have no crossing that the search finds (gmsh failed on a defect that the search does not see).
    """
    from .intersect import describe_places
    name = Path(path).name
    flags = "`--mesh-size` of `topopt verify`, MCP `mesh_size_mm`"
    made = "" if retry is None else _retry_sentence(retry, attempts) + ". "
    if why == "input_crosses_itself":
        text = (f"gmsh stopped on {name} at size {size:g} mm after {attempts} surface attempt(s). "
                f"The STL itself crosses itself at {inp.total} place(s), near where the surface stopped: {describe_places(inp.points)} (mm). "
                "The surface preparation keeps the shape of the STL, so it cannot remove a crossing of the input. "
                "More attempts would fail the same way. "
                f"{made}"
                f"Options: repair the STL (for example a boolean union of the bodies that overlap), or try another mesh size ({flags})")
    else:
        was = _counts_text(counts)
        edge = f" ({mean_edge:.3g} mm)" if mean_edge else ""
        same = (f"The refinement points did not change the prepared surface: the surface of attempt {attempts} is the same as the "
                f"surface of attempt {same_as}, so the next attempt would repeat this one. ")
        if counted:
            itself = ("was not searched for crossings (the search failed)" if inp is None else
                      "does not cross itself, so the surface preparation made the crossings at this size"
                      if not inp.total else f"also crosses itself at {inp.total} place(s)")
            text = (f"gmsh stopped on {name} at size {size:g} mm. "
                    f"The prepared surface crosses itself at {prepared.total} place(s) after {attempts} surface attempts "
                    f"(crossings in each attempt: {was}). "
                    "The local refinement around the places of the earlier attempt did not make them fewer, so more attempts would repeat it. "
                    f"{same if same_as is not None else ''}"
                    f"The STL itself {itself}. Places (mm): {describe_places(prepared.points)}. "
                    f"{made}"
                    f"Option: try a mesh size nearer to the mean edge of the STL{edge}. Then the surface preparation collapses fewer edges. "
                    f"Set it with {flags}")
        else:
            seen = ("The prepared surface was not searched for crossings (the search failed)" if prepared is None else
                    f"The prepared surface crosses itself at {prepared.total} place(s)" if prepared.total else
                    "The search finds no place where the prepared surface crosses itself (it does not test faces that share a vertex)")
            places = (f"Places (mm): {describe_places(prepared.points)}" if prepared is not None and prepared.total else
                      f"gmsh named (mm): {describe_places(np.array(gmsh_points, float))}")
            itself = ("was not searched for crossings (the search failed)" if inp is None else "does not cross itself" if not inp.total
                      else f"crosses itself at {inp.total} place(s), none of them within {size:g} mm of the places of this failure")
            text = (f"gmsh stopped on {name} at size {size:g} mm after {attempts} surface attempts. {same}"
                    f"{seen}. Crossings found in the prepared surface in each attempt: {was}. {places}. The STL itself {itself}. "
                    f"{made}"
                    f"Option: try a mesh size nearer to the mean edge of the STL{edge}. Then the surface preparation collapses fewer edges. "
                    f"Set it with {flags}")
    if failed:
        text += ". " + _search_note(failed)
    return MeshingFailed(text, gmsh_points, crossings=_crossing_facts(counts, prepared, inp, why, failed, retry, same_as))


def _crash_text(path, size, attempt, crashes, tried) -> str:
    sigs = ", ".join(sorted({c["signal"] for c in crashes}))
    return (f"gmsh crashed ({sigs}) on {Path(path).name} at size {size:g} mm, attempt {attempt}, with optimize "
            f"{', '.join(repr(o) for o in tried)}. A crash in gmsh cannot be caught, so it ran in a child process. "
            f"Options: use a larger size, or another optimize setting ('Gmsh', 'Gmsh+Netgen', 'Relocate3D'; "
            f"None leaves slivers).")


def mesh_stl(path, size: float | None = None, threads: int = 1, optimize: str | None = "Netgen",
             eps_factor: float = 0.1, max_attempts: int = 10, size_field=None,
             min_edge_mm: float = 0.05, isolate: bool = True,
             fallback_optimize=("Gmsh+Netgen", "Gmsh"), log=None) -> TetMesh:
    """Mesh a closed STL (for example `result/final.stl`) to linear tetrahedra.

    By default (`isolate=True`) gmsh runs in a child Python, so this process never loads libgmsh and
    a crash of gmsh (a segmentation fault was seen in the Netgen optimizer, on the hook at sizes 0.6 to 0.75 mm)
    cannot kill it. With `isolate=False` gmsh runs in this process: faster, but a crash kills the process.
    In-process, calls from several threads run one after the other, and the stdout and stderr of the whole
    process are captured while gmsh runs.

    Arguments:

    - `path`: a closed, oriented, manifold STL (ASCII or binary), in mm. An open STL raises
      `topopt.tet.surface.SurfaceRejected`.
    - `size`: the target edge length in mm. Default: `default_size(path)` (see there). gmsh may use
      edges up to 1.5 times `size` inside the part.
    - `threads`: the number of threads of gmsh. Default 1. A mesh is repeatable only at 1 thread
      (same machine, same gmsh version: the same hash). With more threads, two runs on the same
      input can differ, in the tet count too (0.1 to 0.3 % on large meshes), and the gmsh step did
      not get faster in the hook test. There is no hard cap.
    - `optimize`: the gmsh optimizer after meshing: "Netgen" (default), or None for none.
    - `eps_factor`: the shape error bound of the surface preparation is `eps_factor * size`.
    - `max_attempts`: how often to prepare the surface again around self-intersections
      that gmsh reports. A prepared surface can have a self-intersection that no check sees.
      The retries end earlier when the refinement does not change the prepared surface (the surface of an attempt is equal to the
      surface of the attempt before: same `V` and `F`), because the next attempt would repeat this one. The error has the reason
      `no_progress` and `crossings["same_surface_as_attempt"]`.
      A mesh with a tet of no volume or that the stress check cannot invert (the determinant of its 3 x 3 matrix of edge vectors or of its
      4 x 4 matrix `[1 x y z]` is exactly 0, as for four nodes in one plane) is a failed attempt too. The error names the count and the first place, and the next attempt refines there
      (F4-27: f6a at 0.973 mm, attempt 2). The places are not crossings of the STL, so they do not stop the retries as such.
    - `size_field`: optional function `f(P) -> sizes` for points `P` (n, 3), in mm. It sets a smaller
      local edge length of the surface (for example near a load or a fixed region). It is combined with
      the retry patches by `np.minimum`. Values above `size` have little effect. The tets inside follow
      the surface.
    - `min_edge_mm`: absolute minimum edge length of the prepared surface, in mm (not scaled with `size`).
      Lower it for a part smaller than about 5 mm.
    - `isolate`: run gmsh in a child process (default). The child start costs about 1 s per attempt
      (`facts["seconds_isolation"]`). The meshes are the same as in-process (same hash).
    - `fallback_optimize`: only with `isolate=True`. If the child crashes, try these `optimize`
      settings in order on the same surface, and record it in the facts (`optimize_used`, `crashes`).
      `()` or None turns it off. `optimize` is None, "Netgen", "Relocate3D", "Gmsh" (the default optimizer
      of gmsh during meshing) or steps joined by "+", for example "Gmsh+Netgen".
    - `log`: optional function that takes one text line (for progress).

    Return a `TetMesh` with `nodes` (n, 3) float64, `tets` (m, 4) int64 and `facts`. The facts hold:

    - `tets`, `nodes`: the counts.
    - `quality_min`, `quality_mean`, `quality_p01` (the 0.1 percentile), `below_0p1`, `below_0p05`,
      `volume_fraction_below_0p05`: mean-ratio quality, 1 for a regular tet. The cut 0.05 is the one of
      the P2 report.
    - `volume_ratio`: tet volume divided by the volume of the prepared surface that gmsh got.
      It is 1 to within about 1e-12 if the mesh fills the surface. The acceptance limit is 1e-3
      (`MeshingFailed` if it is larger).
    - `volume_ratio_stl`: tet volume divided by the volume of the STL. It differs from 1 by the
      shape change of the surface preparation (about 1 to 3 % on a marching-cubes STL).
    - `deviation_from_input_mm`: distance of the input vertices to the prepared surface
      (`mean_mm`, `p99_mm`, `max_mm`, `samples`).
    - `stl_sha256`: hash of the input file.
    - `inverted_fixed`: how many tets had a negative volume and were reordered (0 is normal).
    - `threads`, `repeatable_expected` (True if threads is 1; a prediction, not a measurement),
      `sha256` (hash of nodes and tets).
    - `size`, `size_given`, `size_field_given`, `algorithm`, `attempts`, `failed_attempts` (one entry for each failed attempt: `attempt`,
      `kind` (`crossing`: gmsh named places; `zero_volume`: a tet that the stress check cannot use; `other`), `points`, `error`),
      `gmsh_version`, `surface_faces`.
    - `surface_weld_retry`: present only when the first surface preparation was refused for an edge under `min_edge_mm` and the
      retry on the welded input (`topopt.tet.surface.weld_short_edges`) gave the surface: the step with the edges welded, the
      vertices merged, the volume change and the bodies before and after. It is absent in every other run.
    - `surface_clean_retry`: present only when the first surface preparation was refused for an edge under `min_edge_mm`, the weld
      retry did not give a surface, and the last retry (`topopt.tet.surface.clean_short_edges`: the refused surface with its edges
      under `min_edge_mm` collapsed again and the edges that stay welded) did: the step with the edges before and after the collapse,
      the edges welded, the faces and vertices before and after, the volume change, the vertices removed and the largest distance
      of one of them to the cleaned surface (exact), the fold edges that the collapse or the weld made and the fold repair removed
      (`fold_repairs`, F4-27), and the first refusal and the reason of the weld. It is absent in every other run.
    - `surface_shell_retry` (F4-45): present only when gmsh failed on the prepared surface, the surface has more than one closed shell,
      and a shell lies in the material of another shell (a nested shell: it adds no boundary to the solid, the volume of the surface
      counts that material twice, and HXT does not take a second shell). The mesh is then made without that shell. The value has `step`
      (`drop_nested_shells`), `shells_in_surface`, `volume_before_mm3` and `volume_after_mm3` of the prepared surface, `left_out` (for each
      shell that was left out: `shell`, `volume_mm3`, `faces_count`, `bbox_min_mm`, `bbox_max_mm`) and `whole_surface_errors` (what gmsh
      said on the whole surface). It is absent in every other run, also for a surface of several shells that one try meshes.
    - `optimize` (asked for), `optimize_used` (the setting of the mesh), `crashes` (a list; one entry for each
      crash of the child: `attempt`, `signal`, `size`, `optimize`), `isolated`.
    - `seconds` (total), `seconds_surface`, `seconds_gmsh`, `seconds_isolation` (child start and file
      transfer, summed over the attempts; 0 if not isolated).

    Raise `TetUnavailable` if gmsh is not `ok` (the text has the probe state and the fix),
    `SurfaceRejected` if the STL is not a closed manifold (also when the surface preparation of attempt 2 or later is refused: the text
    then starts with what gmsh did on the surface of the attempt before, F4-27), `MeshingFailed` if gmsh fails (when the surface that
    gmsh got came from a retry of the surface preparation, the text names the retry and the first refusal, and
    `crossings["surface_retry"]` has the step),
    `MeshingCrashed` (a `MeshingFailed`) if gmsh crashes with every `optimize` setting, and
    `UserError` for a bad argument or a file that is not an STL. When gmsh fails on a surface of more than one closed shell and gmsh
    named no crossing, the text of the `MeshingFailed` names the shells that are nested in another shell or are a cavity (F4-45).
    """
    t_start = time.perf_counter()
    state = _probe.probe_subprocess(cache=True) if isolate else _probe.probe()
    if not state.ok:
        raise _probe.TetUnavailable(f"The tet mesher needs the optional package gmsh. {state.line()}")
    if threads < 1:
        raise UserError(f"threads must be 1 or more, got {threads}")
    if max_attempts < 1:
        raise UserError(f"max_attempts must be 1 or more, got {max_attempts}")
    from scipy.spatial import cKDTree

    from .surface import SurfaceRejected, prepare_surface
    say = log or (lambda _msg: None)
    path = Path(path)
    V0, F0 = _load_surface(path)
    mean_edge = _mean_edge(V0, F0)
    size_given = size is not None
    if size is None:
        size = _default_size(V0, F0)
    if not size > 0:
        raise UserError(f"size must be positive, got {size}")
    vol_stl = surface_volume(V0, F0)
    stl_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    bad_points: list = []
    failures = []
    seconds_surface = 0.0
    h_patch = max(0.6, 0.3 * size)
    result = None
    crashes: list = []
    seconds_isolation = 0.0
    optimize_now = optimize
    tried = [optimize]
    prepared_counts: list = []           # crossings of the prepared surface, one count for each failed attempt (P4)
    last_cross = None
    input_cross: list = []               # the crossings of the STL itself, found once and only after a failure (None: the search failed)
    search_failed: list = []             # the searches for crossings that could not run (P4 review, L3)
    last_key = None                      # the hash of the prepared surface of the failed attempt before (F1-5 review, MJ1)
    retry = None                         # the step of the retry of the surface preparation behind the surface of the last attempt
    with tempfile.TemporaryDirectory(prefix="topopt_tet_") as workdir:
        for k in range(max_attempts):
            field_fn = size_field
            if bad_points:
                tree = cKDTree(np.array(bad_points))

                def field_fn(P, tree=tree):
                    hp = h_patch + (size - h_patch) * np.minimum(tree.query(P)[0] / 6.0, 1.0)
                    return hp if size_field is None else np.minimum(size_field(P), hp)
            say(f"surface preparation for edge length {size:g} mm, attempt {k + 1}, "
                f"{len(bad_points)} local refinement point(s)")
            t0 = time.perf_counter()
            try:
                prep = prepare_surface(str(path), size, method="collapse", eps_factor=eps_factor,
                                       verbose=False, size_field=field_fn, min_edge_mm=min_edge_mm)
            except SurfaceRejected as refused:
                if k == 0:
                    raise
                # F4-27: a later attempt is refused. The reader sees that an earlier attempt had a surface, and what gmsh did with it.
                flat_tets = failures[-1]["kind"] == "zero_volume"
                done = (f"gmsh made {len(failures[-1]['points']):,} tet(s) of zero volume on the prepared surface of attempt {k}" if flat_tets
                        else f"gmsh failed on the prepared surface of attempt {k}")
                crossing = f" (it crosses itself at {prepared_counts[-1]} place(s))" if prepared_counts and prepared_counts[-1] else ""
                late = SurfaceRejected(f"{done}{crossing}, and the surface preparation for attempt {k + 1} "
                                       f"(local refinement at {len(bad_points)} point(s)) was refused: {refused}")
                # The facts of the attempts before (review MN5): `topopt verify` names the regions near the crossings with them.
                late.crossings = _crossing_facts(prepared_counts, last_cross, input_cross[0] if input_cross else None, None,
                                                 search_failed, retry)
                raise late from None
            seconds_surface += time.perf_counter() - t0
            say(f"tet meshing with gmsh ({threads} thread(s))")
            try:
                while True:
                    try:
                        if isolate:
                            p, t, info = mesh_surface_isolated(prep.V, prep.F, size, threads=threads,
                                                               optimize=optimize_now, workdir=workdir)
                            seconds_isolation += info["seconds_isolation"]
                        else:
                            p, t, info = mesh_surface(prep.V, prep.F, size, threads=threads,
                                                      optimize=optimize_now, workdir=workdir)
                        break
                    except MeshingCrashed as crash:
                        crashes.append(dict(attempt=k + 1, **crash.crash))
                        say(f"gmsh crashed ({crash.crash.get('signal')}) with optimize={optimize_now!r}")
                        nxt = next((o for o in (fallback_optimize or ()) if o not in tried), None)
                        if nxt is None:
                            retry = _surface_retry(prep.steps)
                            crash_text = _crash_text(path, size, k + 1, crashes, tried)
                            if retry is None:
                                raise MeshingCrashed(crash_text, crashes[-1], crashes) from None
                            err = MeshingCrashed(f"{crash_text} {_retry_sentence(retry, k + 1)}.", crashes[-1], crashes)
                            err.crossings = _crossing_facts([], None, None, None, retry=retry)
                            raise err from None
                        tried.append(nxt)
                        optimize_now = nxt
                        say(f"trying optimize={nxt!r} on the same surface")
                flat = zero_volume_centres(p, t)
                if len(flat):                # the contract of this function: the stress check can use every tet (F4-27)
                    raise _FlatTets(f"gmsh made {len(flat):,} tet(s) of zero volume, the first at "
                                    f"({flat[0, 0]:.4f}, {flat[0, 1]:.4f}, {flat[0, 2]:.4f}) mm", flat.tolist())
                result = (p, t, info)
                break
            except MeshingFailed as exc:
                kind = "zero_volume" if isinstance(exc, _FlatTets) else "crossing" if exc.points else "other"
                failures.append(dict(attempt=k + 1, kind=kind, points=exc.points, error=str(exc)[:300]))
                say(f"{str(exc)[:160]}" if kind == "zero_volume" else
                    f"gmsh failed ({len(exc.points)} intersection point(s) reported): {str(exc)[:160]}")
                if isinstance(exc, MeshingCrashed):
                    raise
                # gmsh names one crossing of the prepared surface for each call. Look for all of them (a fraction of a second
                # for the prepared surface), so that the next attempt refines around every place, and so that a retry that
                # cannot help is not made (P4, F2-6: 10 attempts of 2.7 minutes each on a surface with 11 crossings).
                found = _search(prep.V, prep.F, "prepared", k + 1, search_failed, say)
                retry = _surface_retry(prep.steps)             # the retry of the surface preparation that made this surface, or None
                key = _surface_key(prep.V, prep.F)
                same_as = k if key == last_key else None         # the attempt before got the same surface (F1-5 review, MJ1)
                last_key = key
                no_progress = False
                if found is None:
                    prepared_counts.append(None)                      # the one-point refinement stays
                else:
                    last_cross = found
                    earlier = [c for c in prepared_counts if c]      # attempts in which the crossings were found
                    # No progress: not fewer crossings than before, and all of them near a place that an earlier attempt refined.
                    # The same number at other places is a refinement that moves the crossings, so the retries go on (P4 review, L2).
                    no_progress = (bool(earlier) and found.total >= min(earlier)
                                   and _near(found.points, np.array(bad_points, float).reshape(-1, 3), size, every=True))
                    prepared_counts.append(found.total)
                    if found.total:
                        say(f"the prepared surface crosses itself at {found.total} place(s)")
                if not exc.points and (found is None or not found.total):
                    if not search_failed and retry is None:
                        raise
                    notes = ([_search_note(search_failed)] if search_failed else []) + ([_retry_sentence(retry, k + 1)] if retry else [])
                    raise MeshingFailed(f"{exc}. " + ". ".join(notes), exc.points,
                                        crossings=_crossing_facts(prepared_counts, last_cross, None, None, search_failed, retry)) from exc
                places = np.array(list(exc.points) + ([] if found is None else found.points.tolist()), float).reshape(-1, 3)
                # The centres of flat tets are places to refine, not crossings of the STL: the stop for a crossing of the input
                # does not use them (review MN4).
                crossing_places = places if kind != "zero_volume" else np.array([] if found is None else found.points, float).reshape(-1, 3)
                if not input_cross:
                    input_cross.append(_search(V0, F0, "input", k + 1, search_failed, say))
                if input_cross[0] is not None and input_cross[0].total and _near(input_cross[0].points, crossing_places, size):
                    raise _stopped(path, size, "input_crosses_itself", k + 1, prepared_counts, found, input_cross[0],
                                   exc.points, search_failed, mean_edge, retry)
                if no_progress or same_as is not None:
                    raise _stopped(path, size, "no_progress", k + 1, prepared_counts, found, input_cross[0], exc.points,
                                   search_failed, mean_edge, retry, same_as, counted=no_progress)
                bad_points.extend(places.tolist())
        if result is None:
            text = (f"gmsh failed on every surface attempt ({max_attempts}): " + " | ".join(f["error"][:100] for f in failures))
            if prepared_counts:
                text += f". Crossings found in the prepared surface in each attempt: {_counts_text(prepared_counts)}"
            if search_failed:
                text += ". " + _search_note(search_failed)
            if retry is not None:
                text += ". " + _retry_sentence(retry, max_attempts)
            raise MeshingFailed(text, crossings=_crossing_facts(prepared_counts, last_cross, input_cross[0] if input_cross else None,
                                                                None, search_failed, retry))
    p, t, info = result
    nodes = np.ascontiguousarray(p.T)
    tets = np.ascontiguousarray(t.T)
    vol = signed_volumes(nodes, tets)
    neg = vol < 0
    if neg.any():
        tets[neg] = tets[neg][:, [0, 2, 1, 3]]               # swap two nodes: the volume becomes positive
        vol = signed_volumes(nodes, tets)
    q = tet_quality(nodes.T, tets.T)
    low = q < SLIVER_QUALITY
    elapsed = time.perf_counter() - t_start
    facts = dict(
        tets=int(len(tets)), nodes=int(len(nodes)),
        quality_min=float(q.min()), quality_mean=float(q.mean()), quality_p01=float(np.percentile(q, 0.1)),
        below_0p1=int((q < LOW_QUALITY).sum()), below_0p05=int(low.sum()),
        volume_fraction_below_0p05=float(vol[low].sum() / vol.sum()),
        volume_mm3=float(vol.sum()), volume_stl_mm3=float(vol_stl),
        volume_ratio=float(info["volume_ratio"]), volume_ratio_stl=float(vol.sum() / vol_stl),
        deviation_from_input_mm=prep.prepared["deviation_from_input_mm"],
        stl_sha256=stl_sha256,
        inverted_fixed=int(neg.sum()),
        threads=int(threads), repeatable_expected=bool(threads == 1),
        sha256=mesh_hash(nodes, tets),
        size=float(size), size_given=bool(size_given), size_field_given=size_field is not None,
        algorithm=info["algorithm"],
        attempts=len(failures) + 1, failed_attempts=failures, optimize=optimize,
        gmsh_version=state.version,
        optimize_used=optimize_now, crashes=crashes, isolated=bool(isolate),
        seconds_isolation=float(seconds_isolation),
        seconds=float(elapsed), seconds_surface=float(seconds_surface), seconds_gmsh=float(info["seconds"]),
        surface_faces=int(len(prep.F)),
    )
    weld = next((st for st in prep.steps if st["step"] == "weld_short_edges"), None)
    if weld is not None:
        facts["surface_weld_retry"] = weld
    clean = next((st for st in prep.steps if st["step"] == "clean_short_edges"), None)
    if clean is not None:
        facts["surface_clean_retry"] = clean
    if info.get("shell_retry") is not None:
        facts["surface_shell_retry"] = info["shell_retry"]
    return TetMesh(nodes, tets, facts)
