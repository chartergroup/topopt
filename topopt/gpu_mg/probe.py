"""The OpenCL probe of `settings.solver: auto`: the device list and the GPU self-test in a child process with a time limit (CONTRACTS.md section 20).

An OpenCL driver call can hang (a GPU in a bad state, a driver that waits for a device). A hang in the process of `topopt run` or of the MCP
server would stop the run or the server with no message. So the parent makes no OpenCL call for these steps and does not import pyopencl:
a child process (`python -P -m topopt.gpu_mg.probe`) lists the devices, makes the pick of `auto` and runs the self-test, and it prints one JSON
answer. The parent waits `PROBE_TIMEOUT_S` seconds. At the limit the child is killed, and `auto` takes the CPU rule with a reason that names the limit.

Parent: `probe(device, case, pick)` gives the answer (one child for each `gpu_device`, `gpu_case` and `pick` in a process; `forget()` drops it).
`run_probe` is the one seam: it runs the child, and a test replaces it with `probe_here`, the same probe in this process.

Not in the child: an explicit `settings.solver: gpu-mg` (the user asked for the GPU), the gpu-mg solve of a run, and `topopt gpu-check`.

The answer (a dict): `facts` (the dict of `device.facts`), `device` and `case` (the pick of `auto`: the facts of the `DeviceInfo` and the
precision case, else None), `reason` (why there is no pick, else None), `self_test` (the dict of `device.self_test`, or None) and `timed_out`.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

# A constant, about 6 times the first self-test on an Intel Arc GPU (9.8 s). Not a settings field.
PROBE_TIMEOUT_S = 60

PREFIX = "TOPOPT-PROBE "      # the answer is the last line of the child's stdout that starts with this; driver text before it is ignored
MODULE = "topopt.gpu_mg.probe"

# The `problem` text of the OpenCL facts when the child gave no list starts with one of these (the environment report gives its own hint for them).
PROBLEM_STARTS = ("the OpenCL device list did not finish", "the child process of the OpenCL probe")

_ANSWERS: dict = {}


def forget() -> None:
    """Drop the answers of the children of this process (`device.forget_devices` calls it)."""
    _ANSWERS.clear()


def _text(value) -> str:
    return "auto" if value is None else str(value)


def child_command(device, case, pick: bool) -> list[str]:
    """The command of the child. `-P` keeps a file of the work folder from hiding a module."""
    return [sys.executable, "-P", "-m", MODULE, "--device", _text(device), "--case", _text(case)] + ([] if pick else ["--no-pick"])


def pyopencl_installed() -> bool:
    """True when pyopencl can be found. It does not import pyopencl and makes no driver call. A test replaces it."""
    return importlib.util.find_spec("pyopencl") is not None


def pyopencl_version() -> str | None:
    """The version of the installed pyopencl, or None. It reads the package record; it does not import pyopencl."""
    if importlib.util.find_spec("pyopencl") is None:
        return None
    try:
        return importlib.metadata.version("pyopencl")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _empty_facts() -> dict:
    return {"platforms": [], "devices": [], "auto_device": None, "auto_case": None, "chosen_device": None}


def _failed(problem: str, reason: str, timed_out: bool) -> dict:
    """The answer when the child gave none: facts with `problem`, and no pick (`reason`)."""
    from .device import install_command
    facts = {"pyopencl": pyopencl_version(), "problem": problem, "install_command": install_command(), **_empty_facts()}
    return {"facts": facts, "device": None, "case": None, "reason": reason, "self_test": None, "timed_out": timed_out}


def timeout_texts() -> tuple[str, str]:
    """`(problem, reason)` for a child that did not finish: the text of the OpenCL facts and the reason of `auto`."""
    limit = f"{PROBE_TIMEOUT_S:g} s"
    return (f"the OpenCL device list did not finish in {limit}, and the child process was stopped (a new process asks again; the MCP server asks again after a restart)",
            f"the OpenCL device list or the GPU self-test did not finish in {limit}, and the child process was stopped. "
            f"To skip the wait, set settings.solver to a solver name, or set TOPOPT_AUTO_GPU=0. A new process tests again, and the MCP server tests again after a restart")


def _reap(proc: subprocess.Popen) -> None:
    """Wait for a killed child in a thread, so the caller does not wait: a child in a driver call can be in an uninterruptible wait."""
    threading.Thread(target=proc.wait, name="topopt-probe-reaper", daemon=True).start()


def child_environment() -> dict:
    """The environment of the child: the folder that holds the `topopt` package of this process comes first on `PYTHONPATH`, then the old value.

    The child runs with `-P`, so the work folder is not on its path; without this a checkout that is not installed, or another checkout than
    the installed one, would give a child that imports another topopt, or none. When the folder is a site-packages folder (an installed wheel), the child
    finds it by itself, after the standard library, so nothing is added.
    """
    import topopt
    env = dict(os.environ)
    folder = Path(topopt.__file__).resolve().parent.parent
    old = env.get("PYTHONPATH")
    if folder not in _site_folders():       # a site-packages folder is on the path of the child already, after the standard library
        env["PYTHONPATH"] = str(folder) + (os.pathsep + old if old else "")
    return env


def _site_folders() -> set:
    """The resolved site-packages folders of the running Python."""
    import site
    found = []
    try:
        found += list(site.getsitepackages())
    except Exception:       # a virtualenv of an old kind has no such function
        pass
    try:
        found.append(site.getusersitepackages())
    except Exception:
        pass
    return {Path(f).resolve() for f in found}


def run_child(device, case, pick: bool) -> dict:
    """Start the child, wait `PROBE_TIMEOUT_S` seconds for its answer. Never raises.

    Without pyopencl (`find_spec`, no import and no driver call) no child starts: the answer is the one that the child would give.
    """
    if not pyopencl_installed():
        from .device import install_command
        text = "No module named 'pyopencl'"
        facts = {"pyopencl": None, "problem": f"missing: {text}", "install_command": install_command(), **_empty_facts()}
        return {"facts": facts, "device": None, "case": None, "reason": f"pyopencl is not usable here (missing: {text})",
                "self_test": None, "timed_out": False}
    try:
        proc = subprocess.Popen(child_command(device, case, pick), env=child_environment(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, errors="replace")
    except OSError as e:
        why = f"the child process of the OpenCL probe did not start ({type(e).__name__}: {e})"
        return _failed(why, why, False)
    try:
        out, err = proc.communicate(timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        _reap(proc)
        for pipe in (proc.stdout, proc.stderr):
            try:
                pipe.close()
            except Exception:
                pass
        problem, reason = timeout_texts()
        return _failed(problem, reason, True)
    answer = None
    for line in out.splitlines():       # the last line with the prefix that parses as a complete answer, whatever the exit code
        if line.startswith(PREFIX):
            try:
                got = json.loads(line[len(PREFIX):])
            except ValueError:
                continue
            if isinstance(got, dict) and "facts" in got:
                answer = got
    if answer is not None:
        out_answer = {"device": None, "case": None, "reason": None, "self_test": None, **answer, "timed_out": False}
        if proc.returncode != 0:      # a fact: the child gave its answer and then ended with an error (a driver can fail at exit)
            out_answer["child_exit_code"] = proc.returncode
            if isinstance(out_answer.get("facts"), dict):
                out_answer["facts"] = {**out_answer["facts"], "child_exit_code": proc.returncode}
        return out_answer
    last = (err.strip().splitlines() or [""])[-1][:300]
    why = (f"the child process of the OpenCL probe ended with exit code {proc.returncode} and no answer that can be read"
           + (f" ({last})" if last else ""))
    return _failed(why, why, False)


def probe_here(device, case, pick: bool) -> dict:
    """The probe in this process: the child calls it. A test replaces `run_probe` with it, with fake devices."""
    from . import device as devmod
    facts = devmod.facts(device)
    out = {"facts": facts, "device": None, "case": None, "reason": None, "self_test": None, "timed_out": False}
    if not pick:
        return out
    chosen, reason = devmod.auto_gpu(device, case)
    out["reason"] = reason
    if chosen is not None:
        dev, precision = chosen
        out.update(device=dev.facts(), case=precision, self_test=devmod.self_test(dev, precision))
    return out


run_probe = run_child       # the seam: `probe` calls it through this name


def disabled_reason() -> str | None:
    """The reason when `TOPOPT_AUTO_GPU=0` keeps `auto` on the CPU rule: then no child is needed for the pick, and nothing waits."""
    from .device import AUTO_GPU_VARIABLE
    if os.environ.get(AUTO_GPU_VARIABLE) == "0":
        return f"{AUTO_GPU_VARIABLE}=0 is set in the environment"
    return None


def probe(device="auto", case="auto", pick: bool = True) -> dict:
    """The answer of the probe for `settings.gpu_device` and `settings.gpu_case`. One child for each key in a process.

    `pick` False asks for the facts only (no pick, no self-test). An answer with a pick also serves a request for facts only.
    """
    key = (_text(device), _text(case), bool(pick))
    for k in ((key[0], key[1], True), key):
        if k in _ANSWERS:
            return dict(_ANSWERS[k])
    _ANSWERS[key] = run_probe(device, case, pick)
    return dict(_ANSWERS[key])


def not_asked_problem() -> str:
    return f"not asked: {disabled_reason()}"


def facts(device="auto", case="auto") -> dict:
    """The OpenCL facts of `device.facts`, from the probe. Never raises. The parent makes no driver call.

    With `TOPOPT_AUTO_GPU=0` no child is started: the facts have the pyopencl version from the package record, no device list, and `problem` says that
    the list was not asked. (An explicit `settings.solver: gpu-mg` does not come here: `cli.opencl_facts` asks this process.)
    """
    if disabled_reason() is not None:
        from .device import install_command
        return {"pyopencl": pyopencl_version(), "problem": not_asked_problem(), "install_command": install_command(), **_empty_facts()}
    return probe(device, case, pick=False)["facts"]


def auto_pick(device="auto", case="auto") -> dict:
    """The pick of `auto`: the probe answer, or the answer of `TOPOPT_AUTO_GPU=0` with no child."""
    reason = disabled_reason()
    if reason is not None:
        return {"facts": None, "device": None, "case": None, "reason": reason, "self_test": None, "timed_out": False}
    return probe(device, case, pick=True)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog=MODULE, description="The OpenCL probe of settings.solver auto (the child process).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--case", default="auto")
    ap.add_argument("--no-pick", action="store_true")
    args = ap.parse_args(argv)
    answer = probe_here(args.device, args.case, not args.no_pick)
    # A new line first: a driver can leave a part of a line on stdout. The parent takes the last line with the prefix.
    sys.stdout.write("\n" + PREFIX + json.dumps(answer) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
