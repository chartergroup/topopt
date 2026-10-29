"""Write a fake run directory in real time (CONTRACTS.md section 5).

Use it to test the dashboard without the real optimizer.

    .venv/bin/python tools/fake_run.py runs/fake1 --spi 2.5 --iterations 80
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from topopt.dashboard_server import density_to_mesh, pack_mesh  # noqa: E402

SHAPE = (80, 40, 50)
H = 1.0
ORIGIN = (0.0, 0.0, 0.0)
EMPTY, DESIGN, FIXED, LOAD = 0, 1, 2, 3
# Share of the time of one iteration for each step.
LOAD_CENTER = [77.0, 20.0, 4.0]  # mean of the load voxel centres, mm
STEP_SHARE = [
    ("assemble", 0.10),
    ("solve", 0.45),
    ("sensitivity", 0.15),
    ("filter", 0.10),
    ("update", 0.10),
    ("snapshot", 0.10),
]


# -- atomic writes --
def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    write_atomic(path, json.dumps(obj).encode())


# -- model --
def make_labels() -> np.ndarray:
    nx, ny, nz = SHAPE
    lab = np.zeros(SHAPE, np.uint8)
    lab[3:, :, :] = DESIGN
    lab[0:3, 4:36, 4:46] = FIXED  # wall plate at x = 0
    lab[nx - 5:nx - 1, 15:25, 0:8] = LOAD  # load box at the far end, low
    return lab


def make_target(labels: np.ndarray) -> np.ndarray:
    """A beam-like shape: two arms from the plate to the load box."""
    nx, ny, nz = SHAPE
    x, y, z = np.meshgrid(
        np.arange(nx) + 0.5, np.arange(ny) + 0.5, np.arange(nz) + 0.5, indexing="ij"
    )

    def seg_dist(p0, p1):
        p0 = np.array(p0, float)
        d = np.array(p1, float) - p0
        pts = np.stack([x, y, z], -1) - p0
        t = np.clip((pts @ d) / (d @ d), 0, 1)
        return np.linalg.norm(pts - t[..., None] * d, axis=-1)

    arm_top = seg_dist((3, 20, 38), (74, 20, 6))
    arm_low = seg_dist((3, 20, 12), (74, 20, 4))
    dist = np.minimum(arm_top, arm_low)
    web = np.abs(y - 20)
    # Round bars, with a flat web in the y direction.
    t = np.clip(1.0 - (dist / 11.0) ** 2 - (web / 15.0) ** 4, 0, 1)
    t = 1.0 / (1.0 + np.exp(-12.0 * (t - 0.35)))
    t[labels == EMPTY] = 0.0
    return t.astype(np.float32)


def density_at(it: int, n_it: int, labels, target, noise) -> np.ndarray:
    p = it / n_it
    s = min(1.0, p / 0.75)
    s = s * s * (3 - 2 * s)  # smooth step
    gray = 0.25 + 0.02 * noise  # uniform gray with a little noise
    rho = (1 - s) * gray + s * target
    # Sharpen at the end: gray values go to 0 or 1.
    k = 1.0 + 6.0 * max(0.0, (p - 0.5) / 0.5)
    rho = 1.0 / (1.0 + np.exp(-k * 8.0 * (rho - 0.5) / 2.0)) if p > 0.5 else rho
    rho = np.clip(rho, 0, 1)
    rho[labels == EMPTY] = 0.0
    rho[(labels == FIXED) | (labels == LOAD)] = 1.0
    return (rho * 255.0 + 0.5).astype(np.uint8)


def mesh_volume(verts, faces) -> float:
    v0, v1, v2 = (verts[faces[:, i]].astype(np.float64) for i in range(3))
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def write_stl(path: Path, verts, faces) -> None:
    tri = verts[faces].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    write_atomic(path, b"fake topopt".ljust(80, b" ") + np.uint32(len(tri)).tobytes() + rec.tobytes())


# -- main loop --
def run(out: Path, spi: float, n_it: int) -> None:
    # Delete the files of an old run (only the files of section 5).
    import shutil

    for name in ("run.json", "labels.bin", "status.json", "history.jsonl", "run.log"):
        (out / name).unlink(missing_ok=True)
    for name in ("snapshots", "result"):
        shutil.rmtree(out / name, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(exist_ok=True)
    (out / "result").mkdir(exist_ok=True)
    for name in ("history.jsonl", "run.log"):
        (out / name).write_bytes(b"")
    t0 = time.monotonic()
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        write_atomic(out / "run.log", ("\n".join(log_lines) + "\n").encode())
        print(msg, flush=True)

    def status(state, it, step, message=""):
        write_json(out / "status.json", {
            "state": state, "iteration": it, "step": step,
            "elapsed_s": round(time.monotonic() - t0, 2), "message": message,
        })

    labels = make_labels()
    target = make_target(labels)
    rng = np.random.default_rng(1)
    from scipy.ndimage import gaussian_filter as _gf
    noise = _gf(rng.standard_normal(SHAPE).astype(np.float32), 2.0)
    noise /= np.abs(noise).max()
    n_active = int((labels > 0).sum())
    vf = 0.25
    mass_ds = n_active * H ** 3 / 1000 * 1.24

    status("starting", 0, "setup")
    write_json(out / "run.json", {
        "format": 1, "name": "fake_hook", "yaml_path": "fake.yaml",
        "started_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "shape": list(SHAPE), "origin": list(ORIGIN), "h": H,
        "n_active": n_active, "n_free_dofs": n_active * 3, "iterations": n_it,
        "volume_fraction": vf,
        "load_cases": [
            {"name": "hang", "weight": 1.0, "forces": [
                {"region": "tip", "force": [0.0, 0.0, -1.96], "center": LOAD_CENTER}]},
            {"name": "pull_off", "weight": 1.0, "forces": [
                {"region": "tip", "force": [20.9, 5.2, -20.9], "center": LOAD_CENTER}]},
        ],
        "regions": {"design": "design.stl", "fixed": ["wall_plate.stl"],
                    "keep_out": [], "load": {"tip": "hook_tip.stl"}},
        "solver": "cholesky", "mass_design_space_g": round(mass_ds, 2),
    })
    write_atomic(out / "labels.bin", labels.tobytes())
    log("fake run: setup")
    time.sleep(min(2.0, spi))
    status("running", 0, "setup")

    design_mask = labels == DESIGN
    hist_rows = []

    def history(it, rho):
        # Fake numbers: compliance falls as the shape forms.
        p = it / n_it
        c0 = 400.0
        c = 95.0 + (c0 - 95.0) * math.exp(-5.0 * p) + 4.0 * math.sin(it * 1.7) * (1 - p)
        phys = rho[design_mask].astype(np.float64) / 255.0
        row = {
            "iteration": it, "elapsed_s": round(time.monotonic() - t0, 2),
            "compliance": round(c, 3),
            "compliance_cases": [round(c * 0.3, 3), round(c * 0.7, 3)],
            "volume_fraction": round(float(phys.mean()), 4),
            "beta": 1 if p < 0.375 else 2 if p < 0.625 else 4 if p < 0.8125 else 8,
            "change": round(0.2 * math.exp(-4 * p), 4),
            "grayness": round(float((4 * phys * (1 - phys)).mean()), 4),
            "t_iter_s": spi,
        }
        hist_rows.append(row)
        write_atomic(out / "history.jsonl",
                     ("\n".join(json.dumps(r) for r in hist_rows) + "\n").encode())
        log(f"it {it:3d}  c={row['compliance']:.2f}  vf={row['volume_fraction']:.3f}")

    # Iteration 0 is the start field.
    rho0 = density_at(0, n_it, labels, target, noise)
    write_atomic(out / "snapshots" / "it_0000.bin", rho0.tobytes())
    history(0, rho0)

    rho = rho0
    for it in range(1, n_it + 1):
        for step, share in STEP_SHARE:
            status("running", it, step)
            if step == "snapshot":
                rho = density_at(it, n_it, labels, target, noise)
                write_atomic(out / "snapshots" / f"it_{it:04d}.bin", rho.tobytes())
                history(it, rho)
            time.sleep(spi * share)

    # Post-process.
    status("postprocess", n_it, "postprocess")
    log("post-process: smooth mesh")
    time.sleep(min(3.0, spi * 2))
    from scipy.ndimage import gaussian_filter

    smooth = gaussian_filter(rho.astype(np.float32) / 255.0, 0.8)
    verts, faces = density_to_mesh(smooth, ORIGIN, H)
    write_atomic(out / "result" / "preview.bin", pack_mesh(verts, faces))
    write_stl(out / "result" / "final.stl", verts, faces)
    vol = mesh_volume(verts, faces)
    run_time = time.monotonic() - t0
    comp0 = hist_rows[0]["compliance"]
    comp1 = hist_rows[-1]["compliance"]
    mass_final = vol / 1000 * 1.24
    write_json(out / "result" / "metrics.json", {
        "mass_design_space_g": round(mass_ds, 2),
        "mass_final_g": round(mass_final, 2),
        "mass_saved_pct": round(100 * (1 - mass_final / mass_ds), 1),
        "volume_final_mm3": round(vol, 1),
        "compliance_initial": comp0, "compliance_final": comp1,
        "stiffness_cases": [{"name": "hang", "N_per_mm": 812.0},
                            {"name": "pull_off", "N_per_mm": 640.0}],
        "run_time_s": round(run_time, 1), "iterations": n_it,
        "watertight": True, "triangles": int(len(faces)),
    })
    log("done")
    status("done", n_it, "done")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--spi", type=float, default=2.5, help="seconds per iteration")
    ap.add_argument("--iterations", type=int, default=80)
    args = ap.parse_args(argv)
    run(Path(args.out), args.spi, args.iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
