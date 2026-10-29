"""Report what this machine and this Python are, and which topopt solver they give. Read-only.

    python3 tools/platform_probe.py            # text
    python3 tools/platform_probe.py --json     # one JSON object
    python3 tools/platform_probe.py --no-import   # do not try the imports (faster)

It needs only the Python standard library (Python 3.8 or newer), so it runs in a Python where the topopt packages are not
installed (an x86_64 Python on a Mac, for example). It reports:

- the machine: system, machine type, macOS version, and (on a Mac) the Rosetta flag
  `sysctl.proc_translated` and the arm64 flag of the CPU;
- the Python: version, build, the platform tag of its wheels, the path;
- for each package of topopt: the installed version, and if it imports (each import runs in a child
  Python, so a crash in one library does not stop the probe);
- the solver that `settings.solver: auto` gives here, and the hint of topopt (from
  `topopt/solver_choice.py`; the code is read from the folder above this script).

It uses no network. It writes no file outside one temporary folder (the child Pythons run in it, and
they do not write bytecode). It starts only `sysctl` (macOS) and this Python. The exit code is 0.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (distribution name, module name, where it is used)
PACKAGES = [
    ("numpy", "numpy", "core"), ("scipy", "scipy", "core"), ("numba", "numba", "core"),
    ("llvmlite", "llvmlite", "core (under numba)"), ("manifold3d", "manifold3d", "core"),
    ("pyamg", "pyamg", "core"), ("scikit-image", "skimage", "core"),
    ("trimesh", "trimesh", "core"), ("PyYAML", "yaml", "core"), ("rtree", "rtree", "core"),
    ("networkx", "networkx", "core"), ("Pillow", "PIL", "core"), ("mcp", "mcp", "extra mcp"),
    ("cryptography", "cryptography", "under mcp"), ("pypardiso", "pypardiso", "extra pardiso (Linux x86_64)"),
    ("gmsh", "gmsh", "extra tet"), ("cholespy", "cholespy", "extra verify"),
    ("pyopencl", "pyopencl", "extra gpu"),
]


def sysctl(key: str) -> str | None:
    """The text of `sysctl -n <key>` (read-only), or None if the key does not exist or the command fails."""
    for exe in ("/usr/sbin/sysctl", "sysctl"):   # the full path first: a short PATH has no /usr/sbin
        try:
            done = subprocess.run([exe, "-n", key], capture_output=True, text=True, timeout=5)
        except OSError:
            continue
        except subprocess.SubprocessError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None
    return None


def machine_facts() -> dict:
    out = {
        "system": platform.system(), "sys_platform": sys.platform, "machine": platform.machine(),
        "uname_machine": os.uname().machine if hasattr(os, "uname") else None,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        out.update({
            "macos_version": platform.mac_ver()[0],
            "rosetta_flag_proc_translated": sysctl("sysctl.proc_translated"),   # "1": translated; "0": native; missing: old macOS
            "cpu_is_arm64_hw_optional_arm64": sysctl("hw.optional.arm64"),       # "1" on Apple silicon
            "cpu_brand": sysctl("machdep.cpu.brand_string"), "model": sysctl("hw.model"),
        })
        if platform.machine() == "arm64":
            kind = "native arm64 Python"
        elif out["rosetta_flag_proc_translated"] == "1":
            kind = "x86_64 Python under Rosetta on Apple silicon"
        elif (out["rosetta_flag_proc_translated"] in (None, "0")
              and out["cpu_is_arm64_hw_optional_arm64"] != "1"):
            kind = "x86_64 Python on an Intel Mac (not translated, and the CPU is not arm64)"
        else:
            kind = "x86_64 Python on a Mac, kind not clear"
        out["python_kind"] = kind
    return out


def _config(read):
    """`read()` for a value of `sysconfig`, or the text "unknown" if this Python cannot give it."""
    try:
        return read()
    except Exception as exc:   # a Python without its build data (a faked platform in a test, a broken install)
        return f"unknown ({type(exc).__name__})"


def python_facts() -> dict:
    return {
        "version": platform.python_version(), "implementation": platform.python_implementation(),
        "build": " ".join(platform.python_build()), "compiler": platform.python_compiler(),
        "executable": sys.executable, "prefix": sys.prefix,
        "wheel_platform_tag": _config(sysconfig.get_platform),
        "macos_deployment_target": _config(lambda: sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")),
        "pointer_bits": 64 if sys.maxsize > 2**32 else 32,
    }


def try_import(module: str, cwd: str) -> dict:
    """Import `module` in a child Python. Return `{"status": "ok" | "failed" | "crashed" | "timeout", "detail"}`."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        done = subprocess.run([sys.executable, "-B", "-c", f"import {module}"], capture_output=True, text=True,
                              timeout=90, cwd=cwd, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "detail": "the import took more than 90 s"}
    if done.returncode == 0:
        return {"status": "ok", "detail": ""}
    lines = [ln for ln in done.stderr.strip().splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    return {"status": "crashed" if done.returncode < 0 else "failed",
            "detail": f"exit code {done.returncode}" + (f": {last}" if last else "")}


def package_facts(do_import: bool, cwd: str) -> list[dict]:
    rows = []
    for dist, module, use in PACKAGES:
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            version = None
        row = {"package": dist, "module": module, "use": use, "version": version}
        if version is None:
            row.update(status="not installed", detail="")
        elif do_import:
            row.update(try_import(module, cwd))
        else:
            row.update(status="installed (not imported)", detail="")
        rows.append(row)
    return rows


def solver_facts() -> dict:
    """What topopt's own rule says here. It reads `topopt/solver_choice.py` from the folder above this script."""
    sys.dont_write_bytecode = True
    if not (ROOT / "topopt" / "solver_choice.py").is_file():
        return {"available": False, "detail": f"no topopt folder next to this script ({ROOT})"}
    sys.path.insert(0, str(ROOT))
    try:
        import topopt.config  # noqa: F401  (needs PyYAML)
    except ImportError:   # a Python without PyYAML: solver_choice needs only the class UserError from it
        stub = types.ModuleType("topopt.config")
        stub.UserError = type("UserError", (Exception,), {})
        sys.modules["topopt.config"] = stub
    try:
        from topopt import solver_choice
    except Exception as exc:   # report it, do not stop
        return {"available": False, "detail": f"{type(exc).__name__}: {exc}"}
    out = {"available": True, "platform_name": solver_choice.platform_name(),
           "mac_python_kind": solver_choice.mac_python_kind(), "hint": solver_choice.platform_hint(), "settings": {}}
    for setting in solver_choice.SOLVER_SETTINGS:
        try:
            out["settings"][setting] = {"resolved": solver_choice.resolve_solver(setting), "error": None}
        except Exception as exc:   # UserError: the platform cannot run this solver
            out["settings"][setting] = {"resolved": None, "error": str(exc)}
    return out


def collect(do_import: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="topopt-probe-") as tmp:
        return {"machine": machine_facts(), "python": python_facts(),
                "packages": package_facts(do_import, tmp), "solver": solver_facts()}


def render(d: dict) -> str:
    m, p, s = d["machine"], d["python"], d["solver"]
    lines = ["topopt platform probe (read-only, no network, writes only in a temporary folder)", "", "machine:"]
    lines += [f"  {k}: {v}" for k, v in m.items()]
    lines += ["", "python:"] + [f"  {k}: {v}" for k, v in p.items()]
    lines += ["", "packages (version, import test in a child Python):"]
    for r in d["packages"]:
        lines.append(f"  {r['package']:<13} {str(r['version'] or '-'):<12} {r['status']:<22} {r['use']}"
                     + (f"  [{r['detail']}]" if r["detail"] else ""))
    lines += ["", "topopt solver rule:"]
    if not s["available"]:
        lines.append(f"  not available: {s['detail']}")
    else:
        lines.append(f"  platform: {s['platform_name']}, mac_python_kind: {s['mac_python_kind']}")
        for name, r in s["settings"].items():
            lines.append(f"  settings.solver {name}: " + (r["resolved"] if r["resolved"] else f"user error: {r['error']}"))
        lines.append(f"  hint: {s['hint']}")
    lines += ["", "Send this text back. It has no secret except folder names (of the Python and of the repo)."]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="print one JSON object")
    ap.add_argument("--no-import", action="store_true", help="do not try the imports")
    args = ap.parse_args(argv)
    d = collect(not args.no_import)
    print(json.dumps(d, indent=2) if args.json else render(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
