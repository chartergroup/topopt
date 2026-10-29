#!/usr/bin/env python3
"""The short hook run of the CI legs, and the compare with the golden file (CONTRACTS.md section 27).

    .venv/bin/python scripts/ci/short_run.py run --leg linux-x86_64
    .venv/bin/python scripts/ci/short_run.py make-golden          (writes ci/golden/hook_short.json)

`run` starts `topopt run examples/hook/hook_short.yaml --solver <expect_solver of the leg>` with one thread (the general YAML path: the
same command as for any other part, with the CPU solver of the leg named, because `auto` can pick the GPU solver gpu-mg on a machine
with an OpenCL GPU, and the golden file is a CPU result). It compares compliance, volume and mass with the golden file in the
tolerance tier of the leg, and writes the deviation of each metric to the job summary as a fact.
It fails when a metric is outside its tier, when the inputs differ from those of the golden file,
or when the run takes longer than `short_run_limit_s` of the leg. The run folder, without the snapshots,
goes to `<artifacts>/short-run/`.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, get_leg, get_tier, loadavg, md_table, run_main, write_summary  # noqa: E402

DEFAULT_YAML = "examples/hook/hook_short.yaml"
DEFAULT_GOLDEN = "ci/golden/hook_short.json"
METRICS = ("compliance_final", "volume_fraction_final", "volume_final_mm3", "mass_final_g")
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
PINNED = ("numpy", "scipy", "pypardiso", "mkl", "pyamg", "numba", "scikit-image", "trimesh", "manifold3d")


# --- reading a run folder ---------------------------------------------------------------------

def read_metrics(run_dir: Path) -> tuple[dict, dict]:
    """Return (the four compared metrics, other facts of the run) of a finished run folder."""
    m = json.loads((run_dir / "result" / "metrics.json").read_text())
    last = None
    for line in (run_dir / "history.jsonl").read_text().splitlines():
        if line.strip():
            last = json.loads(line)
    if last is None:
        raise CiError(f"{run_dir / 'history.jsonl'} has no row")
    metrics = {"compliance_final": m["compliance_final"], "volume_fraction_final": last["volume_fraction"],
               "volume_final_mm3": m["volume_final_mm3"], "mass_final_g": m["mass_final_g"]}
    info = {k: m.get(k) for k in ("iterations", "triangles", "solver", "watertight")}
    return metrics, info


def _stl_names(node) -> list[str]:
    """Every string in a parsed YAML value that names an STL file, in file order (a region kind that a later version adds is found too)."""
    if isinstance(node, str):
        return [node] if node.lower().endswith(".stl") else []
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        return [n for v in node for n in _stl_names(v)]
    return []


def input_hashes(yaml_path: Path) -> dict[str, str]:
    """sha256 of the YAML and of every STL file that it names, wherever in the YAML. The keys are paths relative to the repo."""
    import yaml
    names = list(dict.fromkeys(_stl_names(yaml.safe_load(yaml_path.read_text()))))
    out = {}
    for f in [yaml_path, *[yaml_path.parent / n for n in names]]:
        f = f.resolve()
        if not f.is_file():
            raise CiError(f"{yaml_path.name} names {f.name}, and the file is not at {f}")
        try:
            key = f.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            key = f.name
        out[key] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


# --- the compare ----------------------------------------------------------------------------------

def compare(golden: dict, measured: dict, tier: dict) -> list[dict]:
    """One row for each metric: the numbers, the deviation, the tolerance and `ok`.

    A metric passes when |measured - golden| <= atol + rtol * |golden|. A missing or non-finite number fails.
    """
    rows = []
    for name in METRICS:
        g, x = golden.get(name), measured.get(name)
        tol = tier[name]
        allowed = tol["atol"] + tol["rtol"] * abs(g) if isinstance(g, (int, float)) else None
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (g, x)):
            rows.append({"metric": name, "golden": g, "measured": x, "abs_dev": None, "rel_dev": None,
                         "rtol": tol["rtol"], "atol": tol["atol"], "allowed": allowed, "ok": False})
            continue
        dev = abs(x - g)
        rows.append({"metric": name, "golden": g, "measured": x, "abs_dev": dev,
                     "rel_dev": dev / abs(g) if g else (0.0 if dev == 0 else math.inf),
                     "rtol": tol["rtol"], "atol": tol["atol"], "allowed": allowed, "ok": dev <= allowed})
    return rows


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.12g}" if isinstance(v, float) else str(v))


def deviation_markdown(rows: list[dict], tier_name: str) -> str:
    return md_table(["metric", "golden", "measured", "abs deviation", "rel deviation", f"allowed (tier {tier_name})", "result"],
                    [[r["metric"], _fmt(r["golden"]), _fmt(r["measured"]), _fmt(r["abs_dev"]), _fmt(r["rel_dev"]),
                      _fmt(r["allowed"]), "ok" if r["ok"] else "OUTSIDE"] for r in rows])


# --- running -------------------------------------------------------------------------------------

def one_thread_env() -> dict:
    env = dict(os.environ)
    env.update({v: "1" for v in THREAD_VARS})
    return env


def run_command(yaml_path: Path, run_dir: Path, solver: str) -> list[str]:
    """The command of the short run. `--solver` names the CPU solver of the leg: the golden file is a CPU result, and `auto` can pick gpu-mg."""
    return [sys.executable, "-m", "topopt", "run", str(yaml_path), "--out", str(run_dir), "--solver", solver, "--quiet"]


def run_topopt(yaml_path: Path, run_dir: Path, solver: str) -> tuple[float, str, str]:
    """Run `topopt run` with one thread and the solver `solver`. Return (wall seconds, load average before, load average after)."""
    cmd = run_command(yaml_path, run_dir, solver)
    print("+ " + " ".join(cmd) + "   (" + " ".join(f"{v}=1" for v in THREAD_VARS) + ")", flush=True)
    load0, t0 = loadavg(), time.monotonic()
    p = subprocess.run(cmd, cwd=ROOT, env=one_thread_env(), capture_output=True, text=True)
    wall = time.monotonic() - t0
    sys.stdout.write(p.stdout + p.stderr)
    if p.returncode != 0:
        log = run_dir / "run.log"
        tail = "\n".join(log.read_text().splitlines()[-15:]) if log.is_file() else ""
        raise CiError(f"topopt run exited with code {p.returncode} after {wall:.1f} s\n{p.stdout[-600:]}{p.stderr[-600:]}\n{tail}")
    return wall, load0, loadavg()


def keep_without_snapshots(run_dir: Path) -> None:
    """Remove the snapshot files of a run folder (the artifact keeps the rest)."""
    shutil.rmtree(run_dir / "snapshots", ignore_errors=True)


def cmd_run(args) -> int:
    leg = get_leg(args.leg)
    tier_name = args.tier or leg["tier"]
    tier = get_tier(tier_name)
    yaml_path = (ROOT / args.yaml).resolve()
    golden = json.loads((ROOT / args.golden).read_text())
    artifacts = Path(args.artifacts).resolve()      # `topopt run` starts in the repo folder
    out = artifacts / "short-run"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        wall, load0, load1 = run_topopt(yaml_path, out, leg["expect_solver"])
        measured, info = read_metrics(out)
    finally:
        keep_without_snapshots(out)
    rows = compare(golden["metrics"], measured, tier)
    changed = [k for k, v in input_hashes(yaml_path).items() if golden.get("inputs_sha256", {}).get(k) != v]
    limit = leg["short_run_limit_s"]
    result = {"leg": args.leg, "tier": tier_name, "wall_s": round(wall, 2), "limit_s": limit,
              "load_average_before": load0, "load_average_after": load1, "info": info,
              "golden_made_by": golden.get("made_by"), "inputs_changed": changed, "metrics": rows}
    (artifacts / "short-run-deviation.json").write_text(json.dumps(result, indent=2) + "\n")
    table = deviation_markdown(rows, tier_name)
    gm = golden.get("made_by", {})
    write_summary(
        f"### Short hook run ({args.leg}): {wall:.1f} s at 1 thread, limit {limit} s, load average {load0}\n\n"
        f"Solver {info['solver']}, {info['iterations']} iterations, {info['triangles']} triangles. Golden file made on "
        f"{gm.get('platform')} with {gm.get('solver')} ({gm.get('date')}). Deviation from the golden file:\n\n{table}")
    print(table)
    print(f"wall {wall:.1f} s (limit {limit} s), load average before {load0}, after {load1}")
    if changed:
        raise CiError(f"the inputs differ from those of the golden file: {', '.join(changed)}. "
                      "Make a new golden file: python scripts/ci/short_run.py make-golden")
    bad = [r["metric"] for r in rows if not r["ok"]]
    if bad:
        raise CiError(f"outside tier {tier_name}: {', '.join(bad)} (the table above gives each deviation and its limit)")
    if wall > limit:
        raise CiError(f"the short run took {wall:.1f} s; the limit of the leg is {limit} s (legs.toml short_run_limit_s)")
    print(f"OK: every metric is inside tier {tier_name}")
    return 0


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _versions() -> dict:
    out = {}
    for n in PINNED:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return out


def cmd_make_golden(args) -> int:
    yaml_path = (ROOT / args.yaml).resolve()
    work = Path(args.artifacts).resolve() / "short-run-golden"
    work.parent.mkdir(parents=True, exist_ok=True)
    wall, load0, _ = run_topopt(yaml_path, work, args.solver)
    measured, info = read_metrics(work)
    keep_without_snapshots(work)
    golden = {
        "format": 1,
        "case": yaml_path.relative_to(ROOT.resolve()).as_posix(),
        "inputs_sha256": input_hashes(yaml_path),
        "metrics": measured,
        "info": info,
        "made_by": {
            "platform": f"{sys.platform}-{platform.machine()}", "cpu": _cpu_model(), "solver": info["solver"],
            "threads": 1, "python": platform.python_version(), "packages": _versions(),
            "topopt_base": args.base, "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
            "wall_s": round(wall, 1), "load_average": load0,
        },
    }
    dest = ROOT / args.golden
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote {dest}\n" + json.dumps(measured, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("make-golden", cmd_make_golden)):
        p = sub.add_parser(name)
        p.add_argument("--yaml", default=DEFAULT_YAML)
        p.add_argument("--golden", default=DEFAULT_GOLDEN)
        p.add_argument("--artifacts", default="artifacts")
        p.set_defaults(fn=fn)
        if name == "run":
            p.add_argument("--leg", required=True)
            p.add_argument("--tier", default=None, help="a tier of legs.toml (default: the tier of the leg)")
        else:
            p.add_argument("--base", default="7555b19", help="the commit of the optimizer code (for the golden file text)")
            p.add_argument("--solver", default="pardiso", help="the CPU solver of the golden file (default pardiso, Linux x86_64; "
                                                              "cholesky on an Apple silicon Mac). Never auto: it can pick the GPU")
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    run_main(main)
