"""Measure the solver gpu-mg against the default solver on the hook (CONTRACTS.md section 20).

    python tools/bench_gpu_mg.py --out runs/bench_gpu [--sizes 1.2,1.0,0.8,0.6] [--iterations 80]

For each voxel size it runs `topopt run` on the hook with the default solver (`settings.solver: auto`: Accelerate
Cholesky on a Mac) and with `gpu-mg`. It runs ONE JOB AT A TIME, in a child process, and it reads the peak memory
of each child. It writes, for each run, a run folder and a JSON file, and for all runs `bench_gpu_mg.json` and
`bench_gpu_mg.md` (a markdown table).

A default-solver run that the memory estimate of `topopt check` says does not fit (the estimate is above the RAM of
the machine) is skipped. The script says so as a fact, with the numbers. Use `--run-default always` to run it anyway.

Each run has a time limit: 4 times the time estimate of `topopt check`, and at least 600 s (`--timeout-s` sets
another limit, 0 means no limit). A run that reaches the limit is stopped and has the status "timed_out".
The gpu-mg time is summed for each optimizer iteration over all solves of the iteration (the solve of the
design, trial designs of a step acceptance, and adjoint solves). The report holds no absolute path.

The sizes 1.2, 1.0, 0.8 and 0.6 mm are the sizes of the G1b research on a Linux machine with an Intel Arc GPU. The
guard `max_active_voxels` is raised explicitly (`--max-active-voxels`, default 400000), and the value is in the report.

Steps for a Mac (Apple silicon):
    uv pip install -e '.[gpu]'            # pyopencl. With it, pytest skips the tests that make an OpenCL context
                                          # on a Mac (one reason line); TOPOPT_TEST_REAL_GPU=1 runs them on the GPU
    python tools/gpu_mg_parity.py         # first: one solve against a direct solver, a few seconds
    python tools/bench_gpu_mg.py --out runs/bench_gpu --sizes 1.2,1.0 --iterations 20   # short first
    python tools/bench_gpu_mg.py --out runs/bench_gpu                                   # the full set

The first run on a real GPU is the first one ever: this solver was only run on a CPU (pocl) before.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import __version__  # noqa: E402
from topopt.cli import check_data, gib, physical_ram_bytes  # noqa: E402
from topopt.config import UserError  # noqa: E402

DEFAULT_HOOK = REPO / "examples" / "hook" / "hook.yaml"
SIZES = "1.2,1.0,0.8,0.6"
TIMEOUT_FACTOR, TIMEOUT_FLOOR_S = 4.0, 600.0     # default time limit: 4 times the check estimate, at least 600 s


def git_state():
    """`(short revision, dirty)` of the repo. `dirty` is True when `git status --porcelain` lists anything."""
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=10)
    try:
        rev = git("rev-parse", "--short", "HEAD").stdout.strip() or None
        status = git("status", "--porcelain")
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None, None
    return rev, dirty


def machine_facts() -> dict:
    """Facts about this machine, for the report."""
    ram = physical_ram_bytes()
    mac = platform.mac_ver()[0]
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=5).stdout.strip() or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    rev, dirty = git_state()
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu": cpu,
            "macos": mac or None, "cores": os.cpu_count(), "ram_bytes": ram, "ram": gib(ram) if ram else None,
            "python": platform.python_version(), "topopt": __version__, "git_rev": rev, "git_dirty": dirty,
            "load_average_start": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                       "NUMBA_NUM_THREADS") if k in os.environ}}


def opencl_facts(device: str) -> dict:
    """The device that gpu-mg takes, with the fp64 flag and the case 'auto' gives. Never raises."""
    try:
        from topopt.gpu_mg import device as dev
        d, case = dev.select(device, "auto")
        return {"device": d.facts(), "case_auto": case}
    except UserError as e:
        return {"device": None, "error": str(e)}


def write_yaml(hook: Path, path: Path, h: float, iterations: int, solver: str, max_active: int,
               gpu_case: str, gpu_device: str) -> None:
    """A copy of the hook YAML with absolute region paths and the settings of this run."""
    doc = yaml.safe_load(hook.read_text())
    base = hook.parent

    def absolute(p):
        return str((base / p).resolve())

    reg = doc["regions"]
    reg["design"] = absolute(reg["design"])
    for key in ("fixed", "keep_out"):
        if key in reg:
            reg[key] = [absolute(p) for p in reg[key]]
    reg["load"] = {k: absolute(v) for k, v in reg["load"].items()}
    st = doc["settings"]
    st.update({"voxel_size": h, "iterations": iterations, "solver": solver, "max_active_voxels": max_active})
    if solver == "gpu-mg":
        st.update({"gpu_case": gpu_case, "gpu_device": gpu_device})
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def peak_rss_bytes(ru) -> int:
    """Peak resident memory of a finished child in bytes. `ru_maxrss` is in bytes on macOS and in KiB on Linux."""
    return int(ru.ru_maxrss) * (1 if sys.platform == "darwin" else 1024)


def child_peak_rss_mb(ru) -> float:
    """Peak resident memory of a finished child in MB (10^6 bytes), from `peak_rss_bytes`."""
    return peak_rss_bytes(ru) / 1e6


def scrub(text, out: Path):
    """Replace the absolute paths of this machine in a text: the Python of this run, the run folder, the repo and
    the home folder (the install hint of a missing pyopencl names the Python and the repo)."""
    if not isinstance(text, str):
        return text
    for path, label in ((sys.executable, "<python>"), (out.resolve(), "<out>"), (REPO, "<repo>"), (Path.home(), "~")):
        text = text.replace(str(path), label)
    return text


def scrub_tree(obj, out: Path):
    """`scrub` for every text in a JSON-like value (dict, list, str). The report goes through this before it is written."""
    if isinstance(obj, dict):
        return {k: scrub_tree(v, out) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_tree(v, out) for v in obj]
    return scrub(obj, out)


def hook_label(hook: Path) -> str:
    """The hook file for the report: the path relative to the repo, else the file name. Never an absolute path."""
    try:
        return hook.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return hook.name


def run_job(yaml_path: Path, run_dir: Path, timeout_s: float) -> dict:
    """Run `topopt run` in a child process and wait for it. Return exit code, wall time and peak memory."""
    cmd = [sys.executable, "-m", "topopt", "run", str(yaml_path), "--out", str(run_dir), "--quiet"]
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    load0 = os.getloadavg() if hasattr(os, "getloadavg") else None
    with open(run_dir / "bench_stderr.txt", "w") as err:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
        timed_out = False
        status, ru = 0, None
        deadline = time.time() + timeout_s if timeout_s else None
        while True:
            try:
                pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            except ChildProcessError:
                break
            if pid:
                break
            if deadline and time.time() > deadline:
                p.kill()
                timed_out = True
                deadline = None
            time.sleep(0.2)
    p.returncode = os.waitstatus_to_exitcode(status)
    return {"returncode": p.returncode, "timed_out": timed_out, "wall_s": round(time.time() - t0, 2),
            "peak_rss_mb": round(child_peak_rss_mb(ru), 1) if ru is not None else None,
            "load_average": {"start": list(load0) if load0 else None,
                             "end": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}}


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def device_iterations(its: dict) -> int:
    """Iterations of the device PCG in one block of load cases: the first pass, and the restart or refinement steps."""
    return int(its.get("executed", 0)) + int(sum(its.get("inner", [])))


def gpu_summary(lines: list) -> dict:
    """The facts from `gpu_mg.jsonl`: the lines are summed for each OPTIMIZER iteration.

    A line is one solve: kind "main" (the design of the loop), "trial" (a trial design of a step acceptance)
    or "adjoint". A line before version R1 of the file has no `iteration`: the line number counts then.
    """
    parts = ("setup_s", "upload_s", "solve_s", "map_s", "total_s")
    by_it: dict = {}
    for n, r in enumerate(lines):
        row = by_it.setdefault(r.get("iteration", n), {"kinds": {}, **{k: 0.0 for k in parts}, "device_iterations": 0})
        row["kinds"][r.get("kind", "main")] = row["kinds"].get(r.get("kind", "main"), 0) + 1
        for k in parts:
            row[k] += r[k]
        row["device_iterations"] += sum(device_iterations(its) for its in r["iterations"])
    per_iteration = [{"iteration": it, **{k: (round(v, 5) if k in parts else v) for k, v in row.items()}}
                     for it, row in sorted(by_it.items(), key=lambda kv: (kv[0] is None, kv[0]))]

    def med(key):
        return round(statistics.median(r[key] for r in per_iteration), 5)

    kinds: dict = {}
    for r in lines:
        kinds[r.get("kind", "main")] = kinds.get(r.get("kind", "main"), 0) + 1
    return {"case": lines[0]["case"], "per": "optimizer iteration (all solves of the iteration summed)",
            "records": len(lines), "records_by_kind": kinds, "optimizer_iterations": len(per_iteration),
            "median_s": {k: med(k) for k in parts},
            "median_device_iterations": statistics.median(r["device_iterations"] for r in per_iteration),
            "per_iteration": per_iteration}


def summarize_run(run_dir: Path, job: dict) -> dict:
    """Facts from the run folder of a finished child."""
    out = dict(job)
    hist = read_jsonl(run_dir / "history.jsonl")
    out["iterations_done"] = len(hist)
    status = {}
    if (run_dir / "status.json").exists():
        status = json.loads((run_dir / "status.json").read_text())
    out["state"] = status.get("state")
    out["message"] = status.get("message") or None
    if (run_dir / "run.json").exists():
        rj = json.loads((run_dir / "run.json").read_text())
        out.update({"solver_run": rj.get("solver"), "free_dofs": rj.get("n_free_dofs"), "n_active": rj.get("n_active"),
                    "n_load_cases": len(rj.get("load_cases") or [])})      # `topopt gpu-check` needs it: the columns for each launch
    if hist:
        t_it = [r["t_iter_s"] for r in hist]
        out.update({"final_compliance": hist[-1]["compliance"], "final_volume_fraction": hist[-1]["volume_fraction"],
                    "median_iter_s": round(statistics.median(t_it), 4), "loop_s": round(sum(t_it), 2)})
    gpu = read_jsonl(run_dir / "gpu_mg.jsonl")
    if gpu:
        out["gpu_mg"] = gpu_summary(gpu)
    log = (run_dir / "run.log").read_text() if (run_dir / "run.log").exists() else ""
    out["log_gpu_line"] = next((ln for ln in log.splitlines() if ln.startswith("gpu-mg: device")), None)
    return out


def memory_fact(yaml_path: Path) -> dict:
    """The memory estimate of `topopt check` against the RAM of this machine."""
    d = check_data(str(yaml_path), solver_setting="auto")
    me = d["memory_estimate"]
    return {"free_dofs": d["free_dofs"], "active_voxels": d["active_voxels"], "estimate_bytes": me["bytes"],
            "ram_bytes": me["ram_bytes"],
            "fits": (me["ram_bytes"] is None or me["bytes"] <= me["ram_bytes"])}


def default_timeout_s(yaml_path: Path) -> float:
    """The time limit of a run: `TIMEOUT_FACTOR` times the time estimate of `topopt check`, at least `TIMEOUT_FLOOR_S`."""
    try:
        estimate = check_data(str(yaml_path), solver_setting="auto")["time_estimate"]["total_s"]
    except UserError:
        estimate = 0.0
    return max(TIMEOUT_FLOOR_S, TIMEOUT_FACTOR * float(estimate))


def markdown(report: dict) -> str:
    m, oc = report["machine"], report["opencl"]
    d = oc.get("device") or {}
    lines = ["# gpu-mg benchmark", "",
             f"- machine: {m['platform']}, {m['cpu']}, {m['cores']} cores, RAM {m['ram']}",
             f"- macOS: {m['macos'] or 'not a Mac'}; python {m['python']}; topopt {m['topopt']} "
             f"(git {m['git_rev']}{', dirty: the working tree had uncommitted changes' if m.get('git_dirty') else ''})",
             f"- OpenCL device: {d.get('name', 'none')} ({d.get('kind', '-')}, platform {d.get('platform', '-')}, "
             f"{d.get('version', '-')}); 64-bit float: {d.get('fp64', '-')}; case 'auto' gives: {oc.get('case_auto', '-')}",
             f"- max_active_voxels used: {report['max_active_voxels']}; iterations: {report['iterations']}",
             f"- note: {report['note']}", "",
             "| voxel mm | solver | status | free dofs | case | total s | median s/iter | peak RSS MB | final compliance "
             "| rel. diff vs default | speed-up |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    by = {}
    for r in report["runs"]:
        by[(r["h"], r["solver"])] = r
    for r in report["runs"]:
        base = by.get((r["h"], "default"))
        diff = speed = "-"
        if r["solver"] == "gpu-mg" and base and base.get("final_compliance") and r.get("final_compliance"):
            diff = f"{abs(r['final_compliance'] - base['final_compliance']) / abs(base['final_compliance']):.2e}"
            if base.get("median_iter_s") and r.get("median_iter_s"):
                speed = f"{base['median_iter_s'] / r['median_iter_s']:.2f}x"
        status = r["status"] if r["status"] != "skipped" else "skipped (" + r["skipped_reason"] + ")"
        case = (r.get("gpu_mg") or {}).get("case", "-")
        lines.append(f"| {r['h']} | {r['solver']} | {status} | {r.get('free_dofs', '-')} | {case} | "
                     f"{r.get('wall_s', '-')} | {r.get('median_iter_s', '-')} | {r.get('peak_rss_mb', '-')} | "
                     f"{r.get('final_compliance', '-')} | {diff} | {speed} |")
    lines += ["", "## gpu-mg: median time of one OPTIMIZER iteration, by part (s)", "",
              "Each value is the sum over all solves of the iteration (the solve of the design, trial designs of a "
              "step acceptance, adjoint solves), then the median over the optimizer iterations.", "",
              "| voxel mm | setup (CPU) | upload | solve | map u | total | median device iterations (all solves) "
              "| solves (main/trial/adjoint) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in report["runs"]:
        g = r.get("gpu_mg")
        if g:
            s = g["median_s"]
            k = g["records_by_kind"]
            lines.append(f"| {r['h']} | {s['setup_s']} | {s['upload_s']} | {s['solve_s']} | {s['map_s']} | "
                         f"{s['total_s']} | {g['median_device_iterations']} | "
                         f"{k.get('main', 0)}/{k.get('trial', 0)}/{k.get('adjoint', 0)} |")
    lines += ["", "Values for each optimizer iteration (setup, upload, solve, map, device iterations, solves by kind) "
              "are in the JSON file, key `runs[].gpu_mg.per_iteration`."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="folder for the run folders and the report")
    ap.add_argument("--hook", default=str(DEFAULT_HOOK), help="the problem YAML (default: examples/hook/hook.yaml)")
    ap.add_argument("--sizes", default=SIZES, help=f"voxel sizes in mm, comma separated (default {SIZES})")
    ap.add_argument("--iterations", type=int, default=80)
    ap.add_argument("--solvers", default="default,gpu-mg", help="default, gpu-mg or both")
    ap.add_argument("--run-default", choices=["fits", "always"], default="fits",
                    help="'fits' skips a default-solver run whose memory estimate is above the RAM")
    ap.add_argument("--max-active-voxels", type=int, default=400000, help="settings.max_active_voxels (explicit)")
    ap.add_argument("--device", default="auto", help="settings.gpu_device")
    ap.add_argument("--gpu-case", default="auto", choices=["auto", "B", "C"], help="settings.gpu_case")
    ap.add_argument("--timeout-s", type=float, default=None,
                    help=f"stop a run after this many seconds (status timed_out). Default: {TIMEOUT_FACTOR:g} times the "
                         f"time estimate of `topopt check`, at least {TIMEOUT_FLOOR_S:g} s. 0: no limit")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the memory facts, run nothing")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hook = Path(args.hook).resolve()
    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    report = {"machine": machine_facts(), "opencl": opencl_facts(args.device), "hook": hook_label(hook),
              "iterations": args.iterations, "max_active_voxels": args.max_active_voxels, "runs": [],
              "note": ("these are measurements of one machine; the Linux numbers of the G1b research (Intel Arc GPU, "
                       "PARDISO) are not the numbers of a Mac")}
    if report["opencl"].get("device") and report["opencl"]["device"]["kind"] == "CPU":
        report["note"] = "OpenCL CPU device (pocl or similar), NOT a GPU timing. " + report["note"]
    print(f"machine: {report['machine']['platform']}, RAM {report['machine']['ram']}")
    print(f"OpenCL: {report['opencl']}")

    for h in sizes:
        for solver in solvers:
            tag = f"h{h:g}_{solver}"
            run = {"h": h, "solver": solver, "tag": tag, "iterations": args.iterations,
                   "max_active_voxels": args.max_active_voxels}
            path = out / f"{tag}.yaml"
            write_yaml(hook, path, h, args.iterations, "auto" if solver == "default" else "gpu-mg",
                       args.max_active_voxels, args.gpu_case, args.device)
            if solver == "default":
                try:
                    mem = memory_fact(path)
                except UserError as e:    # for example the voxel guard: a fact, the run would stop the same way
                    run.update({"status": "error", "message": scrub(str(e), out)})
                    report["runs"].append(run)
                    print(f"{tag}: not run. {e}")
                    continue
                run["memory_fact"] = mem
                if not mem["fits"] and args.run_default == "fits":
                    run.update({"status": "skipped",
                                "skipped_reason": (f"memory estimate {gib(mem['estimate_bytes'])} is above the RAM "
                                                   f"{gib(mem['ram_bytes'])}")})
                    report["runs"].append(run)
                    print(f"{tag}: SKIPPED. The memory estimate of topopt check for the default solver is "
                          f"{gib(mem['estimate_bytes'])} at {mem['free_dofs']} free dofs; this machine has "
                          f"{gib(mem['ram_bytes'])}. Use --run-default always to run it anyway.")
                    continue
            if args.dry_run:
                run.update({"status": "dry-run"})
                report["runs"].append(run)
                print(f"{tag}: would run ({path})")
                continue
            timeout_s = args.timeout_s if args.timeout_s is not None else default_timeout_s(path)
            run["timeout_s"] = timeout_s or None
            print(f"{tag}: running ({args.iterations} iterations, limit "
                  f"{f'{timeout_s:.0f} s' if timeout_s else 'none'}) ...", flush=True)
            job = run_job(path, out / tag, timeout_s)
            summary = summarize_run(out / tag, job)
            run.update(summary)
            run["status"] = "ok" if (job["returncode"] == 0 and summary.get("state") == "done") else "error"
            if job["timed_out"]:
                run["status"] = "timed_out"
                run["message"] = f"stopped after the time limit of {timeout_s:.0f} s ({summary.get('iterations_done', 0)} iterations done)"
            elif run["status"] == "error":
                err = (out / tag / "bench_stderr.txt").read_text().strip().splitlines()
                run["message"] = run.get("message") or (err[-1] if err else f"exit code {job['returncode']}")
            if run.get("message"):
                run["message"] = scrub(run["message"], out)
            if run.get("log_gpu_line"):
                run["log_gpu_line"] = scrub(run["log_gpu_line"], out)
            report["runs"].append(run)
            (out / f"{tag}.json").write_text(json.dumps(run, indent=1))
            print(f"{tag}: {run['status']}, {run.get('wall_s')} s, peak RSS {run.get('peak_rss_mb')} MB, "
                  f"final compliance {run.get('final_compliance')}", flush=True)
    report["machine"]["load_average_end"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    report = scrub_tree(report, out)       # no absolute path of this machine in the files
    (out / "bench_gpu_mg.json").write_text(json.dumps(report, indent=1))
    (out / "bench_gpu_mg.md").write_text(markdown(report))
    print(f"report: {out / 'bench_gpu_mg.md'}")
    return 0 if all(r["status"] in ("ok", "skipped", "dry-run") for r in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
