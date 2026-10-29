"""Helpers for the ratchet tests (CONTRACTS.md section 36). Import them with `from ratchet_helpers import ...`.

- `BoxCutSource`: a TEST-ONLY candidate source. It cuts boxes out of the current parent with a boolean (manifold3d), so the
  candidate is a clean closed solid that `topopt verify` can mesh at any size. (The sewn `apply_marks` output is meshed by verify
  since the C3a fix: test T6 in `tests/test_ratchet_e2e.py`.) It is not part of the engine: the ratchet has no part-specific code.
- `fake_judge(table)`: a judge that returns the facts of a table, keyed by the candidate id (`c001`), for the runner tests.
- `stl_folder(path, n)`: a folder of `n` small box STL files, for a `StlFolderSource`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from topopt.ratchet_sources import Candidate, tried_refs  # noqa: E402


def load_mesh(path) -> Any:
    return trimesh.load(str(path), force="mesh")


def box_stl(path, size=(1.0, 1.0, 1.0), at=(0.0, 0.0, 0.0)) -> Path:
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(at)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return Path(path)


def stl_folder(folder, n: int, name="part_{}.stl") -> Path:
    """`n` box STL files with different sizes: part_1.stl, part_2.stl, ..."""
    folder = Path(folder)
    for i in range(1, n + 1):
        box_stl(folder / name.format(i), size=(2.0 + 0.01 * i, 1.0, 1.0))
    return folder


def stl_volume(path) -> float:
    return float(load_mesh(path).volume)


def cut_box(stl_in, stl_out, centre, size) -> Path:
    """Subtract an axis-aligned box from a closed solid (a boolean). Return the new STL path."""
    import manifold3d as m3d
    mesh = load_mesh(stl_in)
    man = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                                tri_verts=np.asarray(mesh.faces, np.uint32)))
    sx, sy, sz = (float(x) for x in np.broadcast_to(np.asarray(size, float), (3,)))
    cx, cy, cz = (float(x) for x in centre)
    cutter = m3d.Manifold.cube((sx, sy, sz), center=True).translate((cx, cy, cz))
    out = (man - cutter).to_mesh()
    res = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=False)
    Path(stl_out).parent.mkdir(parents=True, exist_ok=True)
    res.export(str(stl_out))
    return Path(stl_out)


def far_vertex(stl, region_stls) -> np.ndarray:
    """The vertex of the part that is farthest from the vertices of the region STLs (the fixed, load and keep-out regions)."""
    from scipy.spatial import KDTree
    mesh = load_mesh(stl)
    pts = np.vstack([load_mesh(p).vertices for p in region_stls])
    d = KDTree(pts).query(mesh.vertices)[0]
    return mesh.vertices[int(np.argmax(d))]


class BoxCutSource:
    """A test-only source: each entry of `cuts` is `(ref, centre, size)`; the box is cut out of the CURRENT parent.

    `centre` may be a callable `centre(parent_stl) -> (x, y, z)` (for example `lambda p: far_vertex(p, regions)`).
    """
    kind = "boxcut"
    needs_labels = False

    def __init__(self, cuts: list[tuple[str, Any, Any]]):
        self.cuts = list(cuts)

    def describe(self) -> dict:
        return {"kind": self.kind, "count": len(self.cuts)}

    def propose(self, parent_dir, out_dir, history):
        done = tried_refs(history)
        for ref, centre, size in self.cuts:
            if ref in done:
                continue
            parent_stl = Path(parent_dir) / "result" / "final.stl"
            c: Any = centre(parent_stl) if callable(centre) else centre
            out = cut_box(parent_stl, Path(out_dir) / "edited.stl", c, size)
            return Candidate(self.kind, ref, out, Path(out_dir), facts={"centre": [float(x) for x in c], "size": np.broadcast_to(np.asarray(size, float), (3,)).tolist()})
        return None


def fake_judge(table: dict, default=None, calls=None):
    """A judge for the runner tests: the facts come from `table[candidate id]`, the id from `run.json` of the shadow folder.

    The baseline has the id `base`. `calls` (a list) gets `(id, parent folder name, mesh_size_mm of the context)` for each call.
    """
    def judge(shadow_dir, parent_dir, ctx):
        info = json.loads((Path(shadow_dir) / "run.json").read_text()).get("ratchet", {})
        cid = "base" if info.get("kind") == "baseline" else info.get("id")
        if calls is not None:
            calls.append((cid, None if parent_dir is None else Path(parent_dir).name, ctx.mesh_size_mm))
        got = table.get(cid, default)
        if isinstance(got, BaseException):
            raise got
        return json.loads(json.dumps(got)) if got is not None else {"errors": {}}
    return judge
