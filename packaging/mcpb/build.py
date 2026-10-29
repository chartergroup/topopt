#!/usr/bin/env python3
"""Build the Claude Desktop extension (`.mcpb`, a zip) of topopt. CONTRACTS.md section 25.

    uv run --no-project --python 3.13 packaging/mcpb/build.py [--out DIR]

It writes `<DIR>/topopt-<version>.mcpb`. The default DIR is `dist/mcpb` (git-ignored: some tools cannot
publish a binary file, so nobody commits a `.mcpb`).

The script needs Python 3.11 or newer and the standard library. A clean Mac has Python 3.9 as `python3`, so the
command above lets `uv` fetch a Python for it. With a Python below 3.11, the script prints one line with that
command and stops. It reads the repo, makes the zip itself, and does not need the `mcpb` CLI. If the CLI
(`npm install -g @anthropic-ai/mcpb`, MIT) is on PATH, the script also runs `mcpb validate` on the manifest.

What goes in the zip:

- `manifest.json`: the template `packaging/mcpb/manifest.json`, with the version taken from `pyproject.toml`, the
  `tools` read from `topopt_mcp/server.py` and the `prompts` read from `topopt_mcp/roles/`.
- The launcher `scripts/topopt-env.sh` and the lock files `requirements.lock` and `requirements-macos-x86_64.lock`. The
  host starts `/bin/sh <folder>/scripts/topopt-env.sh mcp` (the server type is `binary`): the launcher answers at once from
  the snapshot of the lists, sets up its Python environment in `<folder>/.plugin-env` from the pins of the lock file, and
  then starts the server, the same first start as the Claude Code plugin. The build stops when one of the three is missing.
- `pyproject.toml`: made from the one of the repo, with the pins of the extra `mcp` moved into the main
  dependencies, so an install of the folder without the extra also gets the server. The extras of the repo stay
  (nothing installs an extra unless someone asks for it, for example the offer of `environment`). The pins of
  `requirements.lock` go in as constraints (V5: see `lock_pins`), for a `uv run` in the folder.
- The code folder: `topopt/`, `topopt_mcp/`, `examples/`, `tools/check_part.py`, `CONTRACTS.md` and the license files.
  (`environment` tells the agent that this folder is the `code_folder`.)
  In a git checkout, only the files that git tracks go in: a local file or an ignored run folder stays out, and
  the script prints the names that it left out. Without `.git` (a tarball), every file in those folders goes in.

The zip is reproducible: sorted names, fixed times, fixed modes. The same files give the same bytes on the same
Python build (another zlib can give other compressed bytes).
"""
from __future__ import annotations

import sys

MIN_PYTHON = (3, 11)
UV_COMMAND = "uv run --no-project --python 3.13 packaging/mcpb/build.py"


def python_problem(version=None):
    """None if this Python can run the script, else one line with the fact and the command that fixes it."""
    version = tuple(sys.version_info[:3]) if version is None else tuple(version)
    if version >= MIN_PYTHON:
        return None
    found = ".".join(str(part) for part in version)
    return (f"build.py needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer, and this is Python {found}. "
            f"Run it with uv, which fetches a Python: {UV_COMMAND}")


if python_problem():  # before `import tomllib`: Python 3.9 has no tomllib, and its traceback does not say what to do
    print(python_problem(), file=sys.stderr)
    raise SystemExit(1)

import argparse  # noqa: E402
import ast
import fnmatch
import inspect
import json
import re
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "packaging" / "mcpb" / "manifest.json"
DEFAULT_OUT = ROOT / "dist" / "mcpb"

# Folders and files of the repo that go in the zip (relative to ROOT).
CODE_TREES = ("topopt", "topopt_mcp", "examples")
# From `tools/` only the files that the role texts and the server name (the other files are for developers).
TOOLS_ALLOWED = ("tools/check_part.py",)
# The launcher that the manifest starts, and the lock files whose pins it installs.
LAUNCHER_FILES = ("scripts/topopt-env.sh", "requirements.lock", "requirements-macos-x86_64.lock")
CODE_FILES = ("CONTRACTS.md",) + TOOLS_ALLOWED + LAUNCHER_FILES
LICENSE_GLOBS = ("LICENSE*", "THIRD_PARTY_NOTICES*")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".dylib", ".so", ".o"}  # a built shim is for the machine that built it
SKIP_NAMES = {".DS_Store"}
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


# --- pyproject.toml of the bundle --------------------------------------------------------------------------------

def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[\n" + "".join(f"  {_toml_value(v)},\n" for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} as TOML")


def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def toml_dumps(data: dict) -> str:
    """Write nested tables of strings, numbers, booleans and lists (the shape of the bundle pyproject)."""
    out: list[str] = []

    def walk(path: tuple[str, ...], table: dict) -> None:
        plain = {k: v for k, v in table.items() if not isinstance(v, dict)}
        if path and plain:
            out.append(f"\n[{'.'.join(_toml_key(p) for p in path)}]")
        for key, value in plain.items():
            out.append(f"{_toml_key(key)} = {_toml_value(value)}")
        for key, value in table.items():
            if isinstance(value, dict):
                walk(path + (key,), value)

    walk((), data)
    return "\n".join(out).lstrip("\n") + "\n"


def bundle_pyproject(meta: dict, names: list[str] | None = None) -> str:
    """The pyproject.toml of the bundle. Same name, version, license, pins, extras, scripts and build rules as the repo.

    `names` are the files of the zip. If it is given, each pattern of `license-files` must match one of them
    (setuptools stops on a pattern with no file, and the first start on a Mac is a bad place to learn it).
    """
    project = meta["project"]
    extras = project.get("optional-dependencies", {})
    if "mcp" not in extras:
        raise SystemExit("pyproject.toml has no extra `mcp`: the bundle cannot run the MCP server")
    dependencies = list(project["dependencies"]) + [d for d in extras["mcp"] if d not in project["dependencies"]]
    out = {
        "project": {
            "name": project["name"],
            "version": project["version"],
            "description": project["description"],
            "requires-python": project["requires-python"],
            "dependencies": dependencies,
            "scripts": project.get("scripts", {}),
            # Nothing installs an extra unless someone asks for it, so the extras cost nothing. The offers of
            # `environment` (`uv pip install -e '.[pardiso]'`) need them.
            "optional-dependencies": extras,
        },
        "build-system": meta["build-system"],
        "tool": {"setuptools": meta["tool"]["setuptools"]},
    }
    for key in ("license", "license-files"):  # the metadata of the license, when the repo has it
        if key in project:
            out["project"][key] = project[key]
    if names is not None:
        for pattern in project.get("license-files", []):
            if not any(fnmatch.fnmatchcase(n, pattern) for n in names):
                raise SystemExit(f"license-files `{pattern}` of pyproject.toml matches no file of the bundle")
    pins = lock_pins()
    if pins:  # uv keeps every package that the bundle needs at the version of requirements.lock
        out["tool"]["uv"] = {"constraint-dependencies": pins}
    head = ("# Made by packaging/mcpb/build.py from the pyproject.toml of the repo. Do not edit.\n"
            "# The pins of the extra `mcp` are in `dependencies`: an install of this folder without the extra gets the server.\n"
            "# The extras of the repo stay. Nothing installs an extra unless someone asks for it.\n"
            "# `constraint-dependencies` holds the pins of requirements.lock. They fix the versions of the\n"
            "# sub-packages and add no package. On an x86_64 Mac (best effort) the pins of\n"
            "# requirements-macos-x86_64.lock replace the four that have no wheel there.\n")
    return head + toml_dumps(out)


# The markers of pyproject.toml for the Intel pins (V5): an x86_64 Python on a Mac, an Intel Mac or Rosetta.
INTEL_MAC_MARKER = "platform_system == 'Darwin' and platform_machine == 'x86_64'"
OTHER_MARKER = "platform_system != 'Darwin' or platform_machine != 'x86_64'"


def _read_pins(lock: Path) -> list[str]:
    lines = (line.split("#")[0].strip() for line in lock.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if "==" in line]


def lock_pins() -> list[str]:
    """The `name==version` lines of requirements.lock (the versions that were tested).

    A package that requirements-macos-x86_64.lock pins at another version gets two lines with markers: the pin of
    requirements.lock for every platform but an x86_64 Mac, and the pin of the x86_64 lock for an x86_64 Mac. Without
    this, `uv` finds the constraints unsatisfiable on an Intel Mac (numba 0.67.0 has no wheel there). The pins without
    a difference stay plain lines.
    """
    lock = ROOT / "requirements.lock"
    if not lock.is_file():
        return []
    pins = _read_pins(lock)
    intel_lock = ROOT / "requirements-macos-x86_64.lock"
    if not intel_lock.is_file():
        return pins
    intel = {pin.split("==")[0].lower(): pin for pin in _read_pins(intel_lock)}
    out: list[str] = []
    for pin in pins:
        other = intel.get(pin.split("==")[0].lower())
        if other is not None and other != pin:
            out += [f"{pin}; {OTHER_MARKER}", f"{other}; {INTEL_MAC_MARKER}"]
        else:
            out.append(pin)
    return out


# --- manifest.json ------------------------------------------------------------------------------------------------

def server_tools(server_py: Path) -> list[dict]:
    """`name` and first docstring line of each function that `@mcp.tool(...)` registers, in file order."""
    tree = ast.parse(server_py.read_text(encoding="utf-8"))
    tools = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            call = dec.func if isinstance(dec, ast.Call) else dec
            if (isinstance(call, ast.Attribute) and call.attr == "tool"
                    and isinstance(call.value, ast.Name) and call.value.id == "mcp"):
                doc = inspect.cleandoc(ast.get_docstring(node, clean=False) or "")
                tools.append({"name": node.name, "description": doc.splitlines()[0] if doc else node.name})
    return tools


def role_prompts() -> list[dict]:
    """One prompt for each role, from topopt_mcp/roles/. The text is the same as the server serves."""
    sys.path.insert(0, str(ROOT))  # topopt_mcp/team.py imports only the standard library
    try:
        from topopt_mcp import team
        roles = team.read_roles(ROOT / "topopt_mcp" / "roles")
    finally:
        sys.path.remove(str(ROOT))
    return [{"name": r.id, "description": r.summary, "text": r.text} for r in roles.values()]


def build_manifest(meta: dict) -> dict:
    manifest = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    manifest["version"] = meta["project"]["version"]
    manifest["tools"] = server_tools(ROOT / "topopt_mcp" / "server.py")
    manifest["prompts"] = role_prompts()
    return manifest


# --- the file list --------------------------------------------------------------------------------------------------

def _tree_files(rel: str) -> list[tuple[str, Path]]:
    found = []
    for path in sorted((ROOT / rel).rglob("*")):
        parts = path.relative_to(ROOT).parts
        if (not path.is_file() or SKIP_DIRS & set(parts) or path.suffix in SKIP_SUFFIXES
                or path.name in SKIP_NAMES or path.is_symlink()):
            continue
        found.append(("/".join(parts), path))
    return found


def _candidates() -> list[tuple[str, Path]]:
    """Every file of the code folders in the working tree (tracked or not), with the skip rules applied."""
    files: list[tuple[str, Path]] = []
    for tree in CODE_TREES:
        files += _tree_files(tree)
    for name in CODE_FILES:
        if (ROOT / name).is_file():
            files.append((name, ROOT / name))
    for pattern in LICENSE_GLOBS:
        files += [(p.name, p) for p in sorted(ROOT.glob(pattern)) if p.is_file()]
    return files


def _git_tracked() -> set[str] | None:
    """The names that git tracks, or None when ROOT is not a git checkout (a tarball) or git fails."""
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        return None
    result = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True)
    if result.returncode != 0:
        return None
    return {name for name in result.stdout.decode("utf-8", "surrogateescape").split("\0") if name}


def collect_files() -> tuple[list[tuple[str, Path]], list[str]]:
    """(the files of the zip, the names that were left out). In a git checkout only tracked files go in. Without
    `.git` every file of the code folders goes in, and the second list is empty."""
    candidates = _candidates()
    tracked = _git_tracked()
    if tracked is None:
        return candidates, []
    return ([c for c in candidates if c[0] in tracked], [name for name, _ in candidates if name not in tracked])


def _entry(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    mode = 0o755 if name.endswith(".sh") else 0o644  # topopt/native/build.sh is a script that the user can run
    info.external_attr = mode << 16
    zf.writestr(info, data)


def build(out_dir: Path) -> Path:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = meta["project"]["version"]
    manifest = build_manifest(meta)
    entry_point = manifest["server"]["entry_point"]
    files, left_out = collect_files()
    names = [name for name, _ in files]
    if entry_point not in names:
        raise SystemExit(f"entry point {entry_point} is not in the file list")
    absent = [name for name in LAUNCHER_FILES if name not in names]
    if absent:
        raise SystemExit(f"the launcher needs {', '.join(absent)}, and the file list does not have it")
    if left_out:
        shown = ", ".join(left_out[:5]) + (f" and {len(left_out) - 5} more" if len(left_out) > 5 else "")
        print(f"note: left out {len(left_out)} file(s) that git does not track (a local or ignored file): {shown}",
              file=sys.stderr)
    elif _git_tracked() is None:
        print("note: no git checkout here, so every file in the code folders went in the zip.", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"topopt-{version}.mcpb"
    entries = [("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode()),
               ("pyproject.toml", bundle_pyproject(meta, names).encode())]
    entries += [(name, path.read_bytes()) for name, path in files]
    tmp = target.with_suffix(".mcpb.part")
    with zipfile.ZipFile(tmp, "w") as zf:
        for name, data in sorted(entries):
            _entry(zf, name, data)
    tmp.replace(target)
    validate(manifest)
    return target


def validate(manifest: dict) -> None:
    """Run `mcpb validate` if the CLI is on PATH. Without it, say so once."""
    cli = shutil.which("mcpb")
    if cli is None:
        print("note: the mcpb CLI is not on PATH, so the manifest was not checked against its schema "
              "(npm install -g @anthropic-ai/mcpb).", file=sys.stderr)
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        result = subprocess.run([cli, "validate", str(path)], capture_output=True, text=True)
    text = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise SystemExit(f"mcpb validate failed:\n{text}")
    print(f"mcpb validate: {text.splitlines()[-1] if text else 'ok'}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Claude Desktop extension (.mcpb) of topopt.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default: {DEFAULT_OUT})")
    args = parser.parse_args(argv)
    target = build(args.out)
    with zipfile.ZipFile(target) as zf:
        count = len(zf.namelist())
    print(f"{target} ({target.stat().st_size / 1024:.0f} KiB, {count} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
