"""`topopt gpu-check`: check the GPU solver gpu-mg on this machine with ONE command and get ONE report.

    topopt gpu-check [--out DIR] [--device D] [--skip-tests] [--skip-bench] [--step-timeout-s S]

Four steps, in this order (CONTRACTS.md section 20):

1. facts: the checkout, the machine, pyopencl and the OpenCL devices (the facts of `topopt doctor`, no import test);
2. parity: `tools/gpu_mg_parity.py`, one solve of a small cantilever against the direct solvers;
3. tests: pytest on the four gpu-mg test files with `TOPOPT_TEST_REAL_GPU=1` (they run on the real device);
4. bench: `tools/bench_gpu_mg.py --sizes 1.2,1.0 --iterations 20`, a short run of the hook (minutes). When it passes, the
   bench step also saves the constants of the device (the seconds for each free dof and PCG iteration, and the host part for
   each free dof) in the per-user state folder, `$TOPOPT_STATE_DIR` or `~/.topopt`, file `gpu-constants.json` (not the report file `gpu-check.json` of
   the same name). `topopt check` uses them for the time model of gpu-mg and says so (`gpu_mg/check.py`, CONTRACTS.md section 20).

Steps 2 to 4 run in child processes, so a driver crash stops one step, not the report. A step that runs and fails stops
the flow (the later steps depend on it) and the report says what to run next. A step that cannot run because a tool is
missing (pytest, the `tools/` folder of a wheel install) is "not run", with the command that fixes it.
The files `gpu-check.md` and `gpu-check.json` go in the report folder, and the markdown is printed at the end: it is
one block that the user can paste in a chat. A command in the report has no trailing comment: zsh does not read `#`
in an interactive shell.

The module is light at import (standard library only). It reads no device itself: `environment_data()` does.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .doctor import REPO_URL, checkout_code_line, checkout_origin_line, utc_text
from .gpu_mg import check as gpu_time
from .uv_command import uv_command

BENCH_SIZES = "1.2,1.0"         # the short bench (GOAL-L1 decision 7); the first run on the M4 used these
BENCH_ITERATIONS = 20
TEST_FILES = ("tests/test_gpu_mg.py", "tests/test_gpu_mg_device.py", "tests/test_gpu_mg_x_merge.py",
              "tests/test_bench_gpu_mg.py")
TOOL_FILES = ("tools/gpu_mg_parity.py", "tools/bench_gpu_mg.py")
STEP_TITLES = {"facts": "facts", "parity": "parity", "tests": "tests on the real device", "bench": "short bench"}
# How long a step took on the first real GPU (MacBook Air M4, 2026-10-21). Shown before the step starts.
STEP_HINTS = {"parity": "a few seconds on the M4", "tests": "27 s on the M4",
              "bench": "about 2 minutes on the M4"}


def _q(text) -> str:
    return shlex.quote(str(text))


def _cmd(parts) -> str:
    return " ".join(_q(p) for p in parts)


def _say(text: str = "") -> None:
    print(text, flush=True)


def _kill_group(proc) -> None:
    """Stop a child and everything it started: the child leads its own process group (`start_new_session`)."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:     # it ended already
        pass


def _reap(proc):
    """The output that is left after a kill. A process outside the group that keeps the pipes open does not block us."""
    try:
        return proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        return "", ""


def _joined(stdout: str, stderr: str) -> str:
    return stdout + (("\n--- stderr ---\n" + stderr) if stderr.strip() else "")


def run_child(cmd, env=None, cwd=None, timeout=None) -> dict:
    """Run one step in a child process. Return `returncode` (None if it did not start or timed out), `output`, `seconds`.

    The child runs in its own process group. On a time limit, on Ctrl-C and on any other exception, the whole group is
    killed: `tools/bench_gpu_mg.py` starts `topopt run` as a child, and that job must not go on alone (it holds
    gigabytes). Output that is not UTF-8 is decoded with replacement characters.
    """
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", env=env,
                                cwd=cwd, stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        return {"returncode": None, "output": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - t0,
                "timed_out": False}
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        stdout, stderr = _reap(proc)
        return {"returncode": None, "output": _joined(stdout or "", stderr or "") + f"\nstopped after {timeout:g} s (--step-timeout-s)",
                "seconds": time.perf_counter() - t0, "timed_out": True}
    except BaseException:       # Ctrl-C, SIGTERM of this process: no child group is left behind
        _kill_group(proc)
        _reap(proc)
        raise
    return {"returncode": proc.returncode, "output": _joined(stdout, stderr), "seconds": time.perf_counter() - t0,
            "timed_out": False}


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _device_infos(devices: list) -> list:
    """`DeviceInfo` objects from the dicts of `opencl.devices`, so that `pick_device` can choose by number or by name."""
    from .gpu_mg.device import DeviceInfo
    names = {f.name for f in dataclasses.fields(DeviceInfo)} - {"obj"}
    return [DeviceInfo(**{k: v for k, v in d.items() if k in names}) for d in devices]


def _same_device(a: dict, b: dict) -> bool:
    """The same device: the same index, or one is the second listing of the other (pocl lists a CPU device twice)."""
    return (a["index"] == b["index"] or a.get("duplicate_of") == b["index"] or b.get("duplicate_of") == a["index"]
            or (a.get("duplicate_of") is not None and a.get("duplicate_of") == b.get("duplicate_of")))


def device_label(d: dict) -> str:
    return (f"{d['index']}: {d['name']} ({d['kind']}, platform {d['platform']}, {d.get('version') or 'OpenCL ?'}, "
            f"64-bit float {'yes' if d.get('fp64') else 'no'})")


def device_short(d: dict) -> str:
    """`0: Apple M4 (GPU)`; a device that the driver lists twice says which one it repeats."""
    dup = d.get("duplicate_of")
    return f"{d['index']}: {d['name']} ({d['kind']}" + (f", the same device as {dup}" if dup is not None else "") + ")"


def plan_facts(data: dict, root: Path, device, options: str = "") -> dict:
    """Decide from the facts what can run: `blockers` (each has `message` and `next`), `device` and `auto` (dicts or None).

    `options` is the text of the options that the user gave (without `--device`); the commands to run again carry it.
    """
    oc = data.get("opencl") or {}
    blockers = []
    py = sys.executable
    again = f"{_q(py)} -m topopt gpu-check" + (f" {options}" if options else "")
    again_device = again + (f" --device {_q(device)}" if device is not None else "")
    missing = [f for f in TOOL_FILES if not (root / f).is_file()]
    if missing:
        blockers.append({
            "summary": f"gpu-check needs a checkout: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing",
            "message": f"gpu-check needs a checkout: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not in "
                       f"{root} (the tools/ folder is not in a wheel install). Clone the repository, make an environment "
                       f"with the gpu extra there, and run gpu-check from it.",
            "next": [f"git clone {REPO_URL}",
                     f"cd topopt && {uv_command()} venv --python 3.13 && {uv_command()} pip install -e '.[dev,mcp,gpu]'",
                     ".venv/bin/python -m topopt gpu-check"]})
    chosen = auto = None
    devices = oc.get("devices") or []
    if oc.get("pyopencl") is None:
        command = oc.get("install_command")
        blockers.append({
            "summary": f"pyopencl is not usable: {oc.get('problem') or 'not known'}",
            "message": f"pyopencl is not usable here: {oc.get('problem') or 'not known'}. Install the gpu extra, then run "
                       f"gpu-check again.",
            "next": [command, again_device] if command else [again_device]})
    elif not devices:
        blockers.append({
            "summary": f"pyopencl {oc['pyopencl']} loads, but OpenCL shows no device",
            "message": f"pyopencl {oc['pyopencl']} loads, but OpenCL shows no device ({oc.get('problem') or 'no device'}). "
                       f"gpu-mg needs an OpenCL device. The other solvers work without one: `settings.solver: auto`.",
            "next": []})
    else:
        auto = next((d for d in devices if d["index"] == oc.get("auto_device")), None)
        if device is not None:
            from .config import UserError
            from .gpu_mg.device import pick_device
            try:
                picked = pick_device(device, _device_infos(devices))
                chosen = next(d for d in devices if d["index"] == picked.index)
            except UserError as exc:
                blockers.append({"summary": str(exc), "message": str(exc), "next": [f"{again} --device <number>"]})
        elif auto is not None and auto["kind"] != "CPU":
            chosen = auto
        else:
            listing = "; ".join(device_short(d) for d in devices)
            blockers.append({
                "summary": f"no OpenCL device that is not a CPU is visible (CPU devices only: {listing})",
                "message": f"No OpenCL device that is not a CPU is visible (devices: {listing}). gpu-check tests a GPU. "
                           "To run the steps on a CPU device as a smoke test, name the device with --device.",
                "next": [] if missing else [f"{again} --device {devices[0]['index']}"]})
    return {"blockers": blockers, "device": chosen, "auto": auto}


def parse_parity(output: str) -> dict:
    """The verdict and the largest relative differences from the output of `tools/gpu_mg_parity.py`."""
    du, dc, rows = 0.0, 0.0, 0
    for line in output.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and len(cells) >= 4:
            try:
                du, dc = max(du, float(cells[2])), max(dc, float(cells[3]))
                rows += 1
            except ValueError:
                pass
    verdict = next((ln.strip() for ln in reversed(output.splitlines()) if ln.strip().startswith(("PASS", "FAIL"))), None)
    return {"verdict": verdict, "max_du": du if rows else None, "max_dc": dc if rows else None, "rows": rows}


def parse_pytest(output: str) -> dict:
    """The summary line of pytest (`98 passed, 10 skipped in 27.10s`) and the FAILED or ERROR lines."""
    lines = [ln.strip().strip("=").strip() for ln in output.splitlines() if ln.strip()]
    summary = next((ln for ln in reversed(lines) if re.search(r"\b(passed|failed|errors?|skipped|deselected)\b", ln)
                    and re.search(r"\bin [\d.]+s\b", ln)), None)
    if summary is None:
        summary = next((ln for ln in reversed(lines) if "no tests ran" in ln), None)
    return {"summary": summary, "failed": [ln for ln in lines if ln.startswith(("FAILED ", "ERROR "))][:20]}


def _mm(h: float) -> str:
    return f"{h:g}" if "." in f"{h:g}" else f"{h:.1f}"


def bench_facts(out_dir: Path) -> dict:
    """From the bench report: one summary line (median seconds per iteration and the speed-up) and the markdown table."""
    result = {"summary": None, "markdown": None}
    try:
        report = json.loads((out_dir / "bench_gpu_mg.json").read_text())
        md = (out_dir / "bench_gpu_mg.md").read_text()
    except (OSError, ValueError):
        return result
    by = {(r["h"], r["solver"]): r for r in report.get("runs", [])}
    parts = []
    for (h, solver), r in sorted(by.items(), key=lambda kv: -kv[0][0]):
        if solver != "gpu-mg":
            continue
        base = by.get((h, "default"), {})
        if r.get("status") != "ok":
            parts.append(f"{_mm(h)} mm gpu-mg {r.get('status')}")
        elif base.get("median_iter_s") and r.get("median_iter_s"):
            parts.append(f"{_mm(h)} mm {r['median_iter_s']:g} s per iteration, {base['median_iter_s'] / r['median_iter_s']:.2f}x "
                         f"the default solver ({base['median_iter_s']:g} s)")
        else:
            parts.append(f"{_mm(h)} mm {r.get('median_iter_s')} s per iteration (no default run to compare)")
    result["summary"] = "; ".join(parts) or None
    result["markdown"] = "\n".join(md.splitlines()[1:]).strip()      # without the first heading of the bench file
    return result


def calibration_facts(out_dir: Path) -> dict:
    """The constants of this device for the time model of `topopt check`, from the bench report in `out_dir`.

    Return `record` (the calibration, or None), `file` (where it was saved, or None) and `text` (one line for the report: the numbers
    and where they are saved, or why there is no calibration). A folder that cannot be written is not an error: the text says so.
    `topopt check` reads the file (`gpu_mg.check.constants_for`).
    """
    try:
        report = json.loads((out_dir / "bench_gpu_mg.json").read_text())
    except (OSError, ValueError):
        return {"record": None, "file": None, "text": "no calibration for `topopt check`: the bench report cannot be read"}
    record, why = gpu_time.calibration_from_bench(report, __version__,
                                                  f"topopt gpu-check, bench step: hook at {BENCH_SIZES} mm, {BENCH_ITERATIONS} iterations")
    if record is None:
        return {"record": None, "file": None, "text": f"no calibration for `topopt check`: {why}"}
    numbers = (f"c = {record['c_s_per_dof_iteration'] * 1e9:.1f} ns for each free dof and PCG iteration of one block of "
               f"{record['columns_per_launch']} column{'s' if record['columns_per_launch'] != 1 else ''}, host part "
               f"{record['host_s_per_dof'] * 1e6:.2f} us for each free dof, device {record['device']['name']}, case {record['case']}")
    try:
        path = gpu_time.save_calibration(record)
    except OSError as exc:
        return {"record": record, "file": None,
                "text": f"calibration for `topopt check`: {numbers}. Not saved ({gpu_time.calibration_path()}: {exc.strerror or exc}), so `check` does not use it"}
    return {"record": record, "file": str(path), "text": f"calibration for `topopt check`: {numbers}. Saved in {path}; `topopt check` uses it for this device"}


def _step(step_id: str, state: str, summary: str, seconds=None, detail: str = "", next_=None, command: str = None,
          log: str = None) -> dict:
    return {"id": step_id, "title": STEP_TITLES[step_id], "state": state, "summary": summary,
            "seconds": None if seconds is None else round(seconds, 1), "detail": detail, "next": list(next_ or []),
            "command": command, "log": log}


def run(out=None, device=None, skip_tests: bool = False, skip_bench: bool = False, step_timeout_s=None,
        root=None, runner=run_child, data: dict = None, say=_say, have_module=_has_module) -> int:
    """Run the four steps, write `gpu-check.md` and `gpu-check.json`, print the markdown. Return the exit code.

    Exit code 0: every step that was due ran and passed (a step skipped by an option or by its rule does not count against
    it). 1: a step failed, or a step could not run (the report says what to run next). An internal error is exit code 2.
    Ctrl-C stops the running step with its child processes and still writes the report ("stopped by the user").
    `runner`, `data`, `say` and `have_module` are there for the tests.
    """
    given_out = out
    out = Path(out or "runs/gpu-check").resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(root).resolve() if root else Path(__file__).resolve().parent.parent
    py = sys.executable
    # The options the user gave, for the commands that run gpu-check again (`--device` is added where it is needed).
    options = " ".join(filter(None, [f"--out {_q(out)}" if given_out is not None else "", "--skip-tests" if skip_tests else "",
                                     "--skip-bench" if skip_bench else "",
                                     f"--step-timeout-s {step_timeout_s:g}" if step_timeout_s is not None else ""]))
    steps = []
    calibration = {"record": None, "file": None, "text": None}     # the time constants of the bench step (it sets them)
    plan = {"blockers": [], "device": None, "auto": None}
    current = [None]            # the id of the step that has started

    def begin(n, step_id):
        current[0] = step_id
        due = step_id == "facts" or not (steps[0]["state"] != "passed" or any(s["state"] == "failed" for s in steps))
        hint = f" ({STEP_HINTS[step_id]})" if due and step_id in STEP_HINTS else ""
        say(f"step {n} of 4: {STEP_TITLES[step_id]}{hint} ...")

    def finish(step):
        steps.append(step)
        secs = f" ({step['seconds']:g} s)" if step["seconds"] is not None else ""
        say(f"  {step['state']}: {step['summary']}{secs}")

    try:
        # 1. facts
        begin(1, "facts")
        t0 = time.perf_counter()
        if data is None:
            try:
                from .environment import environment_data
                data = environment_data()
            except Exception as exc:
                data = {}
                finish(_step("facts", "not run", f"environment_data() failed: {type(exc).__name__}: {exc}",
                             time.perf_counter() - t0, next_=[f"{_q(py)} -m topopt doctor"]))
        if data:
            plan = plan_facts(data, root, device, options)
        if not steps:
            dev = plan["device"]
            if plan["blockers"]:
                finish(_step("facts", "not run", plan["blockers"][0]["summary"], time.perf_counter() - t0,
                             detail="\n".join(b["message"] for b in plan["blockers"]),
                             next_=[c for b in plan["blockers"] for c in b["next"]]))
            else:
                from .gpu_mg.device import choose_case
                detail = [f"device {device_label(dev)}; the case that auto gives: {choose_case(_device_infos([dev])[0])}"]
                finish(_step("facts", "passed", f"device {device_label(dev)}", time.perf_counter() - t0, detail="\n".join(detail)))
                if dev["kind"] == "CPU":
                    say("  note: this is a CPU device (named with --device): the times are not GPU times.")
        blocked = steps[0]["state"] != "passed"
        stopped = None      # the id of the step that failed: it stops the flow
        dev, auto = plan["device"], plan["auto"]

        def not_run(step_id):
            return _step(step_id, "not run", "the facts step could not pass" if blocked else f"the step {stopped} failed")

        # 2. parity
        begin(2, "parity")
        if blocked or stopped:
            finish(not_run("parity"))
        else:
            cmd = [py, str(root / "tools" / "gpu_mg_parity.py"), "--device", str(dev["index"])]
            res = runner(cmd, env=dict(os.environ), cwd=str(root), timeout=step_timeout_s)
            (out / "parity.log").write_text(res["output"])
            info = parse_parity(res["output"])
            if res["returncode"] == 0:
                summary = (f"{info['verdict'] or 'PASS'}: largest relative difference of u {info['max_du']:.2e}, of compliance "
                           f"{info['max_dc']:.2e} (limit 1e-6)" if info["rows"] else info["verdict"] or "PASS")
                finish(_step("parity", "passed", summary, res["seconds"], detail=res["output"].strip(),
                             command=_cmd(cmd), log="parity.log"))
            else:
                why = info["verdict"] or ("stopped by the step time limit" if res["timed_out"] else
                                          next((ln for ln in reversed(res["output"].splitlines()) if ln.strip()), "no output"))
                finish(_step("parity", "failed", why, res["seconds"], detail=res["output"].strip(), next_=[_cmd(cmd)],
                             command=_cmd(cmd), log="parity.log"))
                stopped = "parity"

        # 3. tests
        begin(3, "tests")
        if skip_tests:
            finish(_step("tests", "skipped", "skipped by --skip-tests"))
        elif blocked or stopped:
            finish(not_run("tests"))
        elif auto is not None and not _same_device(dev, auto):
            finish(_step("tests", "skipped", f"the tests take the device that auto takes ({device_label(auto)}), not the one you "
                         f"named, so they would not test it. Run them yourself if you want that device."))
        elif not have_module("pytest"):
            finish(_step("tests", "not run", "pytest is not installed in this Python",
                         next_=[f"{uv_command()} pip install --python {_q(py)} -e {_q(str(root) + '[dev]')}"]))
        elif not all((root / f).is_file() for f in TEST_FILES):
            finish(_step("tests", "not run", f"the test files are not in {root} (a checkout has them)",
                         next_=[f"git clone {REPO_URL}"]))
        else:
            cmd = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TEST_FILES]
            res = runner(cmd, env=dict(os.environ, TOPOPT_TEST_REAL_GPU="1"), cwd=str(root), timeout=step_timeout_s)
            (out / "tests.log").write_text(res["output"])
            info = parse_pytest(res["output"])
            here = f"cd {_q(root)} && TOPOPT_TEST_REAL_GPU=1 "        # the test paths are relative to the code folder
            command = here + _cmd(cmd)
            if res["returncode"] == 0:
                finish(_step("tests", "passed", info["summary"] or "pytest passed", res["seconds"], command=command,
                             log="tests.log"))
            else:
                why = info["summary"] or ("stopped by the step time limit" if res["timed_out"] else
                                          f"pytest exit code {res['returncode']}")
                first = here + _cmd([py, "-m", "pytest", "-x", "-q", "-rf", "-p", "no:cacheprovider", *TEST_FILES])
                finish(_step("tests", "failed", why, res["seconds"], detail="\n".join(info["failed"]) or res["output"][-2000:],
                             next_=[first], command=command, log="tests.log"))
                stopped = "tests"

        # 4. bench
        begin(4, "bench")
        if skip_bench:
            finish(_step("bench", "skipped", "skipped by --skip-bench"))
        elif blocked or stopped:
            finish(not_run("bench"))
        else:
            bench_dir = out / "bench"
            for stale in ("bench_gpu_mg.json", "bench_gpu_mg.md"):      # the table of an earlier run is not this run's result
                (bench_dir / stale).unlink(missing_ok=True)
            cmd = [py, str(root / "tools" / "bench_gpu_mg.py"), "--out", str(bench_dir), "--sizes", BENCH_SIZES,
                   "--iterations", str(BENCH_ITERATIONS), "--device", str(dev["index"])]
            res = runner(cmd, env=dict(os.environ), cwd=str(root), timeout=step_timeout_s)
            (out / "bench.log").write_text(res["output"])
            info = bench_facts(bench_dir)
            if res["returncode"] == 0:
                calibration = calibration_facts(bench_dir)
                finish(_step("bench", "passed", info["summary"] or "the bench ran", res["seconds"],
                             detail=(info["markdown"] or "") + "\n\n" + calibration["text"] + "\n",
                             command=_cmd(cmd), log="bench.log"))
                say(f"  {calibration['text']}")
            else:
                why = ("stopped by the step time limit" if res["timed_out"] else
                       info["summary"] or next((ln for ln in reversed(res["output"].splitlines()) if ln.strip()), "no output"))
                finish(_step("bench", "failed", why, res["seconds"], detail=info["markdown"] or res["output"][-2000:],
                             next_=[_cmd(cmd)], command=_cmd(cmd), log="bench.log"))
                stopped = "bench"
    except KeyboardInterrupt:
        say("\nstopped by the user (Ctrl-C). The child processes of the step are stopped.")
        again = f"{_q(py)} -m topopt gpu-check" + (f" {options}" if options else "")
        if current[0] and current[0] not in {s["id"] for s in steps}:
            finish(_step(current[0], "failed", "stopped by the user (Ctrl-C)", next_=[again]))
        for step_id in STEP_TITLES:
            if step_id not in {s["id"] for s in steps}:
                finish(_step(step_id, "not run", "stopped by the user (Ctrl-C)"))

    report = build_report(data, plan, steps, device, out, calibration)
    (out / "gpu-check.md").write_text(report["markdown"])
    (out / "gpu-check.json").write_text(json.dumps(report["json"], indent=1, default=str))
    say(f"\nreport files: {out / 'gpu-check.md'} and {out / 'gpu-check.json'}")
    say("Paste the text below in a chat:\n")
    say(report["markdown"])
    return report["json"]["exit_code"]


def result_of(steps: list) -> tuple:
    """`(result text, exit code)` from the steps."""
    failed = next((s for s in steps if s["state"] == "failed"), None)
    not_run = [s for s in steps if s["state"] == "not run"]
    skipped = [s["id"] for s in steps if s["state"] == "skipped"]
    if failed:
        return f"FAIL at step {failed['id']}: {failed['summary']}", 1
    if not_run:
        return f"NOT RUN: {not_run[0]['id']}: {not_run[0]['summary']}", 1
    if skipped:
        return f"PASS (skipped: {', '.join(skipped)})", 0
    return "PASS", 0


def build_report(data: dict, plan: dict, steps: list, device, out: Path, calibration: dict = None) -> dict:
    """The markdown and the JSON of the report. `calibration` is the result of `calibration_facts` (key `calibration` of the JSON)."""
    result, code = result_of(steps)
    env = data or {}
    c = env.get("checkout") or {}
    platform, ram, oc = env.get("platform", {}), env.get("ram", {}), env.get("opencl") or {}
    dev = plan.get("device")
    next_ = [cmd for s in steps for cmd in s["next"]]
    time_utc = utc_text(time.time())
    lines = ["# topopt gpu-check report", "", f"- time: {time_utc}",
             f"- topopt {env.get('topopt_version', '?')}: "
             + (checkout_code_line(c) if c.get("git") else c.get("kind", "no checkout facts"))]
    origin = checkout_origin_line(c) if c.get("git") else None
    if origin:
        lines.append(f"- {origin}")
    lines += [f"- machine: {platform.get('name', '?')}, Python {env.get('python', {}).get('version', '?')}, "
              f"RAM {(ram.get('total_bytes') or 0) / 2**30:.1f} GiB, {env.get('cores', {}).get('usable', '?')} usable core(s)",
              f"- OpenCL: pyopencl {oc.get('pyopencl') or 'not usable'}; devices: "
              + ("; ".join(device_short(d) for d in oc.get("devices", [])) or oc.get("problem") or "none"),
              f"- tested device: {device_label(dev) if dev else 'none'}"
              + (f" (named with --device {device})" if device else ""),
              "", f"**Result: {result}**", "",
              "| step | result | seconds | summary |", "|---|---|---|---|"]
    for n, s in enumerate(steps, 1):
        summary = s["summary"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {n} {s['title']} | {s['state']} | {s['seconds'] if s['seconds'] is not None else '-'} | {summary} |")
    for n, s in enumerate(steps, 1):
        if not (s["detail"] or s["command"]):
            continue
        lines += ["", f"## {n}. {s['title']}", ""]
        if s["command"]:
            lines += ["command:", "", "```", s["command"], "```", ""]
        if s["detail"]:
            lines += ([s["detail"]] if s["id"] == "bench" else ["```", s["detail"], "```"])   # the bench detail is a table
    if next_:
        lines += ["", "## Run next", ""] + [f"{i}. `{cmd}`" for i, cmd in enumerate(next_, 1)]
    lines += ["", "The numbers are those of this machine. This report has folder names and no secret."]
    payload = {"time_utc": time_utc, "result": result, "exit_code": code, "topopt_version": env.get("topopt_version"),
               "checkout": c, "platform": platform, "python": env.get("python"), "ram": ram, "opencl": oc,
               "device": dev, "device_option": device, "steps": steps, "next": next_, "out": str(out),
               "calibration": (calibration or {}).get("record"), "calibration_file": (calibration or {}).get("file")}
    return {"markdown": "\n".join(lines) + "\n", "json": payload}
