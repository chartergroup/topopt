"""The child process of the isolated mesher (CONTRACTS.md section 14).

`python -m topopt.tet.worker <in.npz> <out.npz>` reads a prepared surface (`V`, `F`) and the settings
(`params`, JSON text), runs `mesher.mesh_surface` and writes `p`, `t` and `info`, or `error` if gmsh
reports a failure. gmsh can crash with a signal (a segmentation fault in the Netgen optimizer was seen).
A crash kills only this process: the parent sees no output file and a negative exit code.

Core files. A core file of a crashed gmsh holds the memory of the process: many GB for a large mesh.
The mesher does not read that file, so the child starts with the soft limit RLIMIT_CORE at 0 (the hard
limit does not change). It sets the limit at the start of `main`, before gmsh is imported. On Linux the
child also calls `prctl(PR_SET_DUMPABLE, 0)` through ctypes, so the kernel does not write a core file or
start a core-dump helper for this process (a `core_pattern` that is a pipe, for example
systemd-coredump, receives the soft limit and stores no core when it is 0). macOS has no `prctl`: the
limit is the only step there. Only the child changes. The limits of the parent process do not change, and
`mesh_surface(isolate=False)` does not use this module.
If a call fails, the failure goes into the result of `limit_core_dumps` and the meshing runs anyway.

To get a core file of the child for a debug session, set the environment variable
`TOPOPT_TET_KEEP_CORE=1` and raise the limit in the shell that starts the parent (`ulimit -c unlimited`).
The child then keeps the limits of the parent and the dumpable flag. The parent passes its environment on.
The same variable lets a debugger attach to the child: a process with the dumpable flag 0 refuses a
ptrace attach from a process that has no CAP_SYS_PTRACE.

`python -m topopt.tet.worker --limits` applies the same steps, prints one JSON line with the core limits
and the dumpable flag of the child, and exits. It does not mesh and does not import gmsh.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

KEEP_CORE_ENV = "TOPOPT_TET_KEEP_CORE"  # "1" (or any value except "" and "0"): keep the core limits and flags
PR_SET_DUMPABLE = 4  # linux/prctl.h
PR_GET_DUMPABLE = 3


def keep_core_requested(environ=None) -> bool:
    value = (os.environ if environ is None else environ).get(KEEP_CORE_ENV, "")
    return value not in ("", "0")


def _load_libc():
    import ctypes
    return ctypes.CDLL(None, use_errno=True)


def _prctl(libc, option: int, value: int = 0) -> int:
    """Call prctl(option, value, 0, 0, 0). Return the result, or raise OSError with errno if it is -1."""
    import ctypes
    result = libc.prctl(ctypes.c_int(option), ctypes.c_ulong(value), ctypes.c_ulong(0),
                        ctypes.c_ulong(0), ctypes.c_ulong(0))
    if result == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result


def limit_core_dumps(environ=None, platform=None, resource_module=None, libc=None) -> dict:
    """Set the soft RLIMIT_CORE of this process to 0, and on Linux clear the dumpable flag.

    Return a dict with one text for each step: `rlimit_core` and `dumpable`. A text starts with `set`,
    `skipped` or `failed`. No step raises. The four arguments are for the tests (fake environment,
    platform, `resource` module and libc); the child calls the function without arguments.
    """
    platform = sys.platform if platform is None else platform
    status = {}
    if keep_core_requested(environ):
        return {"rlimit_core": f"skipped ({KEEP_CORE_ENV} is set)",
                "dumpable": f"skipped ({KEEP_CORE_ENV} is set)"}
    try:
        res = resource_module
        if res is None:
            import resource as res
        _soft, hard = res.getrlimit(res.RLIMIT_CORE)
        res.setrlimit(res.RLIMIT_CORE, (0, hard))
        status["rlimit_core"] = "set (soft limit 0)"
    except Exception as exc:  # the module is missing (no resource on this platform), or the call fails
        status["rlimit_core"] = f"failed ({type(exc).__name__}: {exc})"
    if not platform.startswith("linux"):
        status["dumpable"] = f"skipped (no prctl on {platform})"
        return status
    try:
        _prctl(libc if libc is not None else _load_libc(), PR_SET_DUMPABLE, 0)
        status["dumpable"] = "set (0)"
    except Exception as exc:
        status["dumpable"] = f"failed ({type(exc).__name__}: {exc})"
    return status


def core_facts(platform=None, resource_module=None, libc=None) -> dict:
    """The core limits and the dumpable flag of this process now. A value that cannot be read is None."""
    platform = sys.platform if platform is None else platform
    facts = {"core_soft": None, "core_hard": None, "dumpable": None}
    try:
        res = resource_module
        if res is None:
            import resource as res
        facts["core_soft"], facts["core_hard"] = res.getrlimit(res.RLIMIT_CORE)
    except Exception:
        pass
    if platform.startswith("linux"):
        try:
            facts["dumpable"] = _prctl(libc if libc is not None else _load_libc(), PR_GET_DUMPABLE)
        except Exception:
            pass
    return facts


def run(in_path: str, out_path: str) -> None:
    from .mesher import MeshingFailed, mesh_surface
    with np.load(in_path) as z:
        V, F, params = np.asarray(z["V"]), np.asarray(z["F"]), json.loads(str(z["params"]))
    with tempfile.TemporaryDirectory(prefix="topopt_tet_") as workdir:
        try:
            p, t, info = mesh_surface(V, F, params["size"], threads=params["threads"],
                                      max_size_factor=params["max_size_factor"], optimize=params["optimize"],
                                      algorithms=tuple(params["algorithms"]), workdir=workdir)
        except MeshingFailed as exc:
            np.savez(out_path, error=np.array(json.dumps(dict(message=str(exc), points=exc.points))))
            return
    np.savez(out_path, p=p, t=t, info=np.array(json.dumps(info)))


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    status = limit_core_dumps()  # first: gmsh is imported later, inside `run`
    if argv == ["--limits"]:
        print(json.dumps(dict(core_facts(), steps=status, platform=sys.platform)))
        return
    run(argv[0], argv[1])


if __name__ == "__main__":
    main()
