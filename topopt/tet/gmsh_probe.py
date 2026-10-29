"""The probe for the optional tet mesher gmsh (CONTRACTS.md section 14).

Same shape as `topopt/solver_choice.py` (`import_pypardiso`, `pypardiso_problem`,
`install_command`). The voxel engine never imports this module, and nothing in
this module imports gmsh at import time. gmsh is GPL-2.0-or-later. The user
installs it with the `tet` extra, and the core of topopt does not import it.

`probe()` gives one of three states and never raises:

- `ok`: gmsh imports. `version` has the version.
- `missing`: the package is not installed. On a platform that the `tet` extra does
  not cover (for example Linux arm64), the text says so and lists other options.
- `load_failed`: the package is installed, but it does not load. The usual cause
  is a missing system library. The wheel from PyPI is built with a GUI and needs
  `libGLU`, `libXft`, `libXinerama` and `libOpenGL`. On Linux, `detail` lists ALL
  missing libraries at once (from `ldd`), with the Debian package names. It is also
  `load_failed` if the gmsh package imports but cannot find its library (libgmsh).

`probe_subprocess()` does the same in a child Python, so the calling process never loads libgmsh.
`GMSH_LOCK` serializes the calls that initialize gmsh (gmsh state is global to the process).
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import UserError
from ..uv_command import uv_command

GMSH_PIN = "gmsh==4.15.2"
GMSH_LICENSE = "GPL-2.0-or-later"
NOX_INDEX = "https://gmsh.info/python-packages-dev-nox"
NOX_PIN = "gmsh==4.15.0.dev1+nox"

# Shared library -> Debian/Ubuntu package. The wheel from PyPI is built with the FLTK GUI.
DEB_PACKAGE = {
    "libGLU.so.1": "libglu1-mesa", "libGL.so.1": "libgl1", "libOpenGL.so.0": "libopengl0",
    "libXft.so.2": "libxft2", "libXinerama.so.1": "libxinerama1", "libXrender.so.1": "libxrender1",
    "libXcursor.so.1": "libxcursor1", "libXfixes.so.3": "libxfixes3", "libXext.so.6": "libxext6",
    "libX11.so.6": "libx11-6", "libfontconfig.so.1": "libfontconfig1",
    "libfreetype.so.6": "libfreetype6", "libgomp.so.1": "libgomp1",
}

STATES = ("ok", "missing", "load_failed")

# gmsh keeps one global state per process. Every use of gmsh (the mesher, the version query)
# takes this lock, so two threads never share one gmsh session.
GMSH_LOCK = threading.RLock()
INIT_ARGS = dict(readConfigFiles=False, interruptible=False)   # no ~/.gmshrc; no signal() (main thread only)


@dataclass(frozen=True)
class TetProbe:
    """The result of `probe()`. `state` is one of `STATES`."""
    state: str
    version: str | None = None      # only for "ok"
    detail: str = ""                # the cause, for "missing" and "load_failed"
    fix: str = ""                   # what to do, for "missing" and "load_failed"

    @property
    def ok(self) -> bool:
        return self.state == "ok"

    def line(self) -> str:
        """One line for a log or a tool reply."""
        if self.state == "ok":
            return f"tet extra: ok (gmsh {self.version})"
        return f"tet extra: {self.state} ({self.detail}). {self.fix}"

    def as_dict(self) -> dict:
        return asdict(self)


def import_gmsh():
    """Import and return the module gmsh. The only import of it in the package."""
    return importlib.import_module("gmsh")


def platform_name() -> str:
    return f"{sys.platform} {platform.machine()}"


def has_wheel_platform() -> bool:
    """True if the `tet` extra covers this platform: macOS, and Linux x86_64 (the marker of the extra)."""
    machine = platform.machine()
    return sys.platform == "darwin" or (sys.platform == "linux" and machine in ("x86_64", "AMD64"))


def install_command() -> str:
    """The command to install the `tet` extra into the Python that runs topopt now."""
    py = shlex.quote(sys.executable)
    root = Path(__file__).resolve().parent.parent.parent
    if (root / "pyproject.toml").is_file():
        return f"{uv_command()} pip install --python {py} -e {shlex.quote(str(root) + '[tet]')}"
    return f"{uv_command()} pip install --python {py} {shlex.quote(GMSH_PIN)}"


def nox_install_command() -> str:
    """The GUI-free build of gmsh. It needs no system library. It is a dev build, not the pin."""
    py = shlex.quote(sys.executable)
    return (f"{uv_command()} pip install --python {py} --prerelease allow --index-strategy unsafe-best-match "
            f"--extra-index-url {NOX_INDEX} '{NOX_PIN}'")


def _libgmsh_candidates() -> list[Path]:
    """The places where gmsh.py looks for libgmsh: the module folder and two parents, each with lib/ and bin/."""
    spec = importlib.util.find_spec("gmsh")
    if spec is None or not spec.origin:
        return []
    mod = Path(spec.origin).resolve().parent
    dirs = [d / sub for d in (mod, mod.parent, mod.parent.parent) for sub in ("", "lib", "bin")]
    found: list[Path] = []
    for d in dirs:
        found.extend(sorted(d.glob("libgmsh.so*")))
    return found


def missing_libraries() -> list[str]:
    """All shared libraries that libgmsh needs and the system lacks (Linux only, best effort).

    It runs `ldd` once on libgmsh, so the user sees every missing library at once.
    It returns an empty list if `ldd` or libgmsh is not there.
    """
    if sys.platform != "linux" or shutil.which("ldd") is None:
        return []
    try:
        for cand in _libgmsh_candidates():
            out = subprocess.run(["ldd", str(cand)], capture_output=True, text=True, timeout=20).stdout
            return re.findall(r"^\s*(\S+) => not found", out, flags=re.M)
    except Exception:      # the probe never raises
        return []
    return []


def _load_failed_parts(error: BaseException) -> tuple[str, str]:
    """(detail, fix) for an installed gmsh that does not load."""
    detail = f"{type(error).__name__}: {error}"
    if sys.platform == "darwin":
        return detail, (
            "Run `otool -L` on libgmsh (next to the gmsh package) to see what it links. The macOS arm64 wheel "
            "needs macOS 12 or newer, links only system libraries and has a static libomp. "
            "The macOS x86_64 wheel (an Intel Mac, or an x86_64 Python under Rosetta) needs macOS 10.15 or newer. "
            f"Check the macOS version and the architecture of Python ({platform.machine()}), "
            f"or install the extra again with: {install_command()}")
    if sys.platform != "linux":
        return detail, f"This platform is not covered by the tet extra. Install again with: {install_command()}"
    libs = missing_libraries()
    m = re.match(r"(\S+): cannot open shared object file", str(error))
    if m and m.group(1) not in libs:
        libs.append(m.group(1))
    if not libs:
        return detail, ("The cause is unknown. Run `ldd` on libgmsh to see what is missing"
                        + (f", or install the GUI-free build: {nox_install_command()}" if is_linux_x86_64() else ""))
    pkgs = sorted({DEB_PACKAGE.get(x, x) for x in libs})
    detail += f". Missing system libraries: {', '.join(libs)}"
    # libGLU needs libOpenGL. `ldd` on libgmsh does not show it while libGLU itself is missing.
    # `apt install` pulls it in by itself; a manual unpack does not.
    pkgs_unpacked = sorted(set(pkgs) | ({"libopengl0"} if "libglu1-mesa" in pkgs else set()))
    fix = (f"With root on Debian or Ubuntu: apt install {' '.join(pkgs)}. "
           f"Without root: download {' '.join(pkgs_unpacked)} with `apt-get download`, unpack them with "
           f"`dpkg-deb -x`, and put the lib folder in LD_LIBRARY_PATH before Python starts.")
    if is_linux_x86_64():
        fix += (f" Or install the GUI-free build of gmsh, which needs no system library "
                f"(a dev build, not the pin {GMSH_PIN}; Linux x86_64 only): {nox_install_command()}")
    return detail, fix


def is_linux_x86_64() -> bool:
    return sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")


def _build_version(gmsh) -> str:
    """The version string. `gmsh.__version__` hides the build (for example "-nox"), so ask gmsh."""
    version = str(getattr(gmsh, "__version__", "unknown"))
    try:
        with GMSH_LOCK:
            if not gmsh.isInitialized():
                gmsh.initialize(**INIT_ARGS)
                try:
                    version = gmsh.option.getString("General.Version") or version
                finally:
                    if gmsh.isInitialized():
                        gmsh.finalize()
    except Exception:
        pass
    return version


def probe() -> TetProbe:
    """Try to import gmsh and say what happened. It never raises.

    It imports gmsh into this process, so the first call can take a second. Use `probe_subprocess()`
    if this process must not load libgmsh.
    """
    try:
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):      # gmsh.py prints a warning if it finds no libgmsh
                gmsh = import_gmsh()
        except ModuleNotFoundError as e:
            if e.name in (None, "gmsh"):
                if not has_wheel_platform():
                    return TetProbe(
                        "missing",
                        detail=f"gmsh is not installed, and the tet extra does not cover {platform_name()}",
                        fix=("Other options: gmsh from conda-forge (another OCC build, so its meshes differ from "
                             "the pinned wheel), or a build of gmsh from source. If gmsh imports in this Python "
                             "in another way, this probe says ok"))
                return TetProbe("missing", detail="gmsh is not installed",
                                fix=f"Install it with: {install_command()} "
                                    f"(gmsh is {GMSH_LICENSE}, the core of topopt does not import it)")
            return TetProbe("load_failed", detail=f"{type(e).__name__}: {e}",
                            fix=f"A module that gmsh needs is missing. Install the extra again with: {install_command()}")
        except OSError as e:     # the wheel loads libgmsh with ctypes; a missing system library ends here
            detail, fix = _load_failed_parts(e)
            return TetProbe("load_failed", detail=detail, fix=fix)
        except Exception as e:
            return TetProbe("load_failed", detail=f"{type(e).__name__}: {e}",
                            fix=f"Install the extra again with: {install_command()}")
        if not hasattr(getattr(gmsh, "lib", None), "gmshInitialize"):
            # gmsh.py did not find libgmsh, printed a warning and loaded the C library of the process instead
            sys.modules.pop("gmsh", None)
            warn = " ".join(out.getvalue().split())[:200]
            return TetProbe("load_failed", detail="libgmsh not found" + (f" ({warn})" if warn else ""),
                            fix=f"The gmsh package has no libgmsh next to it. Install it again with: {install_command()}")
        return TetProbe("ok", version=_build_version(gmsh))
    except Exception as e:       # last guard: the probe never raises
        return TetProbe("load_failed", detail=f"{type(e).__name__}: {e}",
                        fix=f"Install the extra again with: {install_command()}")


_SUBPROCESS_CODE = ("import json; from topopt.tet.gmsh_probe import probe; "
                    "print('PROBE ' + json.dumps(probe().as_dict()))")


_OK_CACHE: dict = {}


def probe_subprocess(timeout: float = 60.0, cache: bool = False) -> TetProbe:
    """Like `probe()`, but in a child Python, so this process never loads libgmsh. It never raises.

    With `cache=True`, an `ok` answer is kept for the next call (same Python, same LD_LIBRARY_PATH);
    a failed answer is never kept, so the user can fix the problem and ask again.
    """
    key = (sys.executable, os.environ.get("LD_LIBRARY_PATH"), os.environ.get("DYLD_LIBRARY_PATH"))
    if cache and key in _OK_CACHE:
        return _OK_CACHE[key]
    result = _probe_subprocess(timeout)
    if cache and result.ok:
        _OK_CACHE[key] = result
    return result


def _probe_subprocess(timeout: float) -> TetProbe:
    try:
        env = dict(os.environ)
        root = str(Path(__file__).resolve().parent.parent.parent)
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        out = subprocess.run([sys.executable, "-c", _SUBPROCESS_CODE], capture_output=True, text=True,
                             timeout=timeout, env=env)
        for line in reversed(out.stdout.splitlines()):
            if line.startswith("PROBE "):
                return TetProbe(**json.loads(line[6:]))
        tail = (out.stderr.strip().splitlines() or ["no output"])[-1]
        return TetProbe("load_failed", detail=f"the probe process failed: {tail[:200]}",
                        fix=f"Install the extra again with: {install_command()}")
    except Exception as e:
        return TetProbe("load_failed", detail=f"the probe process failed: {type(e).__name__}: {e}",
                        fix=f"Install the extra again with: {install_command()}")


class TetUnavailable(UserError):
    """gmsh is not usable. The text has the probe state and the fix."""


def require_gmsh():
    """Return the gmsh module, or raise `TetUnavailable` with the probe state and the fix."""
    state = probe()
    if state.ok:
        return import_gmsh()
    raise TetUnavailable(f"The tet mesher needs the optional package gmsh. {state.line()}")
