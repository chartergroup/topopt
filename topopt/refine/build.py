"""Build the refine page of a finished run (`result/refine.html`).

Adapted from the markup tool (build_markup.py) of an earlier project of the same author. The page
shows `result/final.stl` with a von Mises stress overlay. The owner places add,
remove and smooth marks and exports them as JSON. This module does not change
the mesh.

Called after the run is done (`topopt run`, which passes on the stress result of the post step) or by `topopt refine <run_dir>`.
The optimizer loop and `topopt.post.finish` do not call `build_refine`. `finish` reaches this module only through
`stress_report`: `stress_section` (a run with a stress row) and `run_load_path_facts` (every run, `metrics.load_paths`) import
`problem_from_run`, and call it when `finish` has no `Problem` of the run. The buckling command and the voxel buckling check
also use `problem_from_run`.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import html
import json
import os
import re
from pathlib import Path

import numpy as np

from topopt.config import UserError, load_config

from .page_scripts import VENDOR_SCRIPTS, script_text   # three.js r128 and OrbitControls, inline in the page

HERE = Path(__file__).resolve().parent
TARGET_FACES = 150_000   # decimation acts only above this face count
STRESS_PERCENTILE = 99.0  # the colour scale ends at this percentile of the surface values


def quadric_cluster(v, f, target, iters=14):
    """Vertex-clustering decimation with per-cluster quadric (QEM-style) vertex placement."""
    lo = v.min(0)
    ext = v.max(0) - lo

    def run(h):
        ijk = np.floor((v - lo) / h).astype(np.int64)
        dims = ijk.max(0) + 1
        key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
        uk, inv = np.unique(key, return_inverse=True)
        nf = inv[f]
        ok = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
        return uk, inv, nf[ok]

    lo_h, hi_h = ext.max() / 2000.0, ext.max() / 4.0
    best = None
    for _ in range(iters):
        h = np.sqrt(lo_h * hi_h)
        r = run(h)
        if len(r[2]) > target:
            lo_h = h
        else:
            hi_h = h
            best = (h, r)
    if best is None:
        best = (hi_h, run(hi_h))
    h, (uk, inv, nf) = best
    n = len(uk)
    # quadrics from face planes (area weighted)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nn = np.cross(b - a, c - a)
    area = np.linalg.norm(nn, axis=1) / 2
    nu = nn / (2 * area[:, None] + 1e-30)
    d = -(nu * a).sum(1)
    P = np.concatenate([nu, d[:, None]], 1)
    Q = (P[:, :, None] * P[:, None, :]) * area[:, None, None]
    Qv = np.zeros((n, 4, 4))
    for k in range(3):
        np.add.at(Qv, inv[f[:, k]], Q)
    cnt = np.bincount(inv, minlength=n).astype(float)
    mean = np.zeros((n, 3))
    for k in range(3):
        np.add.at(mean[:, k], inv, v[:, k])
    mean /= cnt[:, None]
    A = Qv[:, :3, :3]
    bvec = -Qv[:, :3, 3]
    lam = 1e-3 * (np.trace(A, axis1=1, axis2=2) / 3 + 1e-12)
    A = A + lam[:, None, None] * np.eye(3)
    rhs = bvec + lam[:, None] * mean
    x = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    # keep within a cell-sized neighbourhood of the mean
    dx = np.clip(x - mean, -h, h)
    x = mean + dx
    # drop unused vertices
    used = np.unique(nf)
    remap = -np.ones(n, np.int64)
    remap[used] = np.arange(len(used))
    return x[used], remap[nf]


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def safe_stem(name: str) -> str:
    """A run name that is safe in a file name and in the page."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return s or "part"


def _grid_text(run: dict, problem, labels) -> str:
    """The grid of the run and the grid of the YAML, as facts (shape, voxel size, origin, labels that differ)."""
    def vec(v):
        return "[" + ", ".join(f"{float(x):g}" for x in v) + "]"
    text = (f"The run has the grid {vec(run['shape'])} at {float(run['h']):g} mm, origin {vec(run['origin'])}; "
            f"the YAML gives the grid {vec(problem.shape)} at {problem.h:g} mm, origin {vec(problem.origin)}")
    if labels.size == problem.labels.size:
        n = int(np.count_nonzero(labels != problem.labels.reshape(-1)))
        if n:
            text += f"; {n} of {labels.size} voxel labels differ"
    return text


def problem_from_run(run_dir: Path, run: dict, yaml_path=None):
    """Rebuild the `Problem` from the YAML named in run.json.

    A relative `yaml_path` of run.json starts at the run folder (`runinfo.run_yaml_path`). `yaml_path` (the argument) gives
    the YAML at its new place (a run folder that was moved, or a recording made on another machine): it wins.
    Stop if the voxel grid differs from the grid of the run. The error gives the path once (absolute) and the grid of the
    run and of the YAML. Without `yaml_path` it says that the YAML or the STL files changed after the run; with it, that
    the given file does not fit the run.
    """
    from topopt.problem import build_problem
    from topopt.runinfo import missing_yaml_text, run_yaml_path
    from topopt.verify.names import opt

    given = Path(yaml_path).expanduser() if yaml_path else None
    yaml_path = given if given is not None else run_yaml_path(run, run_dir)
    if yaml_path is None or not yaml_path.is_file():
        raise UserError(missing_yaml_text(yaml_path, run, opt("yaml"), given is not None))
    problem = build_problem(load_config(yaml_path))
    shape = tuple(int(s) for s in run["shape"])
    labels = np.frombuffer((run_dir / "labels.bin").read_bytes(), np.uint8)
    same = (tuple(problem.shape) == shape
            and abs(problem.h - float(run["h"])) < 1e-9
            and np.allclose(problem.origin, run["origin"])
            and labels.size == problem.labels.size
            and np.array_equal(labels, problem.labels.reshape(-1)))
    if not same:
        shown = os.path.realpath(yaml_path)
        facts = _grid_text(run, problem, labels)
        if given is not None:
            raise UserError(f"the YAML given with {opt('yaml')} does not fit the grid of the run: {shown}. {facts}. "
                            f"Option: give the YAML that made the run with {opt('yaml')}")
        raise UserError(f"the YAML or the STL files of the run changed after the run: {shown}. {facts}. "
                        f"Option: give another YAML with {opt('yaml')}, or run again")
    return problem


def write_stress(result_dir: Path, stress: dict, problem) -> Path:
    """Write result/stress.npz (voxels with rho >= 0.5, centres in the STL mm frame)."""
    arrays = {
        "centers": stress["centers"],
        "rho": stress["rho"].astype(np.float32),
        "vm_max": stress["vm_max"],
        "case_names": np.array(stress["case_names"]),
        "compliance_cases": stress["compliance_cases"],
        "E": np.float64(problem.config.material.E),
        "nu": np.float64(problem.config.material.nu),
        "penal": np.float64(problem.config.settings.penal),
        "h": np.float64(problem.h),
    }
    for i, vm in enumerate(stress["vm_cases"]):
        arrays[f"vm_case_{i}"] = vm
    path = result_dir / "stress.npz"
    tmp = result_dir / "stress.tmp.npz"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return path


def build_refine(run_dir, problem=None, target_faces: int = TARGET_FACES, stress=None, yaml_path=None, say=None) -> Path:
    """Write `<run_dir>/result/refine.html` and `result/stress.npz`. Return the HTML path.

    `problem` is the `Problem` of the run. If it is None, it is rebuilt from the
    YAML named in run.json, or from `yaml_path` (the YAML at its new place). Then the YAML is compared with `run.json` (the
    rule of `topopt verify`: case names, weights, forces, region files). A difference is a fact: `say(line)` gets each
    `note:` line, and the page data (`META.yaml_vs_run`) holds the differences. It is not a stop. `stress` is the result of
    `compute_stress` for the last snapshot of the same
    problem, if the caller has it already (the post-process of a stress run does): it saves one solve.
    """
    import trimesh
    from scipy.spatial import cKDTree

    from topopt.post import _last_snapshot

    from .stress import compute_stress

    run_dir = Path(run_dir)
    result = run_dir / "result"
    stl = result / "final.stl"
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        raise UserError(f"no run.json in {run_dir}")
    if not stl.is_file():
        raise UserError(f"no final mesh {stl}. The run must finish first")
    run = json.loads(run_json.read_text())
    yaml_vs_run = None
    if problem is None:
        from topopt.verify.report import yaml_note_lines
        from topopt.verify.run import compare_yaml_problem
        problem = problem_from_run(run_dir, run, yaml_path)
        yaml_vs_run = compare_yaml_problem(run, problem)
        for line in yaml_note_lines(yaml_vs_run):
            if say is not None:
                say(line)

    # -- stress on the voxels of the final design
    snap = np.frombuffer(_last_snapshot(run_dir).read_bytes(), np.uint8)
    if stress is None:
        stress = compute_stress(problem, snap, log=(lambda line: say("note: " + line)) if say is not None else None, solver=run.get("solver"))
    write_stress(result, stress, problem)

    # -- final mesh
    sha = hashlib.sha256(stl.read_bytes()).hexdigest()
    m = trimesh.load(str(stl), process=False)
    nfaces_src = len(m.faces)
    vol = abs(float(m.volume))
    density = float(problem.config.material.density)  # g/cm3
    mass = vol * density / 1000.0                     # g
    v = np.asarray(m.vertices, np.float64)
    f = np.asarray(m.faces, np.int64)
    uv, inv = np.unique(np.round(v, 5), axis=0, return_inverse=True)  # merge duplicate STL vertices
    v, f = uv, inv.reshape(-1)[f]
    del m
    if len(f) > target_faces:
        v, f = quadric_cluster(v, f, target_faces)

    # -- stress on the surface vertices: nearest voxel centre
    tree = cKDTree(stress["centers"])
    s = stress["vm_max"][tree.query(v)[1]]
    smax = float(f"{np.percentile(s, STRESS_PERCENTILE):.3g}")
    if not np.isfinite(smax) or smax <= 0.0:
        smax = float(f"{max(float(s.max()), 1e-6):.3g}")
    s16 = np.clip(s / smax, 0.0, 1.0) * 65535
    stress_b64 = base64.b64encode(s16.astype("<u2").tobytes()).decode()

    ctr = (v.min(0) + v.max(0)) / 2
    vc = (v - ctr).astype(np.float32)  # centred, in mm; the page adds ctr back on export
    f32 = f.astype(np.uint32)
    stem = safe_stem(run.get("name", "part"))
    meta = dict(
        source_stl=str(stl.resolve()), source_sha256=sha, stem=stem,
        faces_source=int(nfaces_src), faces_embedded=int(len(f)), verts=int(len(v)),
        volume_mm3=vol, mass_g=mass, density_g_cm3=density,
        center=[float(x) for x in ctr],
        stress_source=str((result / "stress.npz").resolve()), stress_max_mpa=smax,
        stress_percentile=STRESS_PERCENTILE, stress_cases=list(stress["case_names"]),
        built=datetime.datetime.now().isoformat(timespec="seconds"))
    if yaml_vs_run is not None:   # a standalone rebuild: what the YAML says against run.json (facts for the person who marks)
        meta["yaml_vs_run"] = yaml_vs_run
    # "</" would end the script block early in the page.
    meta_js = json.dumps(meta).replace("</", "<\\/")
    page = (HERE / "template.html").read_text(encoding="utf-8")
    for mark in VENDOR_SCRIPTS:   # first: these files hold no other mark (tests/test_refine.py)
        page = page.replace(mark, script_text(mark))
    page = (page
            .replace("%%VERTS%%", base64.b64encode(vc.tobytes()).decode())
            .replace("%%FACES%%", base64.b64encode(f32.astype("<u4").tobytes()).decode())
            .replace("%%STRESS%%", stress_b64)
            .replace("%%TITLE%%", html.escape(stem))
            .replace("%%META%%", meta_js))  # last: the other text is not scanned again
    out = result / "refine.html"
    _atomic_write(out, page.encode("utf-8"))
    return out
