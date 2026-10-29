#!/usr/bin/env python3
"""The wheel job: build the wheel, install it in a clean venv, run the packaging test outside the source tree.

    python3 scripts/ci/wheel_check.py [--dist dist] [--python 3.13]

Steps 1 and 2 run once. Steps 3 to 6 run for each supported Python version (SUPPORTED_PYTHONS in _common.py, the one place that
names them: 3.13 only since #469), each in its own clean venv. With `--python V` they run for that version only.
1. `uv build --wheel`.
2. The wheel holds every data file that tests/test_packaging.py lists (DATA_FILES), and the licence files
   that `license-files` of pyproject.toml names (they come with PR #27; without the key nothing is checked).
3. A new venv (outside the repo) gets the wheel with the `dev` extra, with the versions of requirements.lock.
4. `tests/test_packaging.py` is copied to an empty folder. There, `import topopt` must come from the venv
   and not from the source tree, and the test file runs against the installed wheel. The test that builds a
   wheel from the source tree (`test_wheel_lists_data_files`) cannot do that outside the tree, so it is
   deselected: step 2 does the same check on the wheel of this run.
5. `topopt --version` and `topopt check examples/hook/hook.yaml` run from that folder with the installed wheel.
6. `pytest --collect-only` on `tests/` in that venv, which has the `dev` extra only (no `mcp`, no `tet`). A test
   module that imports `mcp` with no `importorskip` gives a collection error there (the fix of these modules is still open). This step
   reports the errors in the job summary and does not fail the job.
Standard library only.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, SUPPORTED_PYTHONS, CiError, run_main, write_summary  # noqa: E402

DESELECT = "tests/test_packaging.py::test_wheel_lists_data_files"


def data_files_of_test(test_file: Path) -> list[str]:
    """The list `DATA_FILES` of tests/test_packaging.py (read with `ast`; the test module is not imported)."""
    for node in ast.parse(test_file.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DATA_FILES" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise CiError(f"{test_file} has no DATA_FILES list")


def wheel_problems(names: list[str], data_files: list[str], license_patterns: list[str]) -> list[str]:
    """Texts of what the wheel lacks: a data file, or a licence file under `<name>.dist-info/licenses/`."""
    problems = [f"the wheel has no {f}" for f in data_files if f not in names]
    lic = [n.split("/licenses/", 1)[1] for n in names if "/licenses/" in n and ".dist-info/" in n]
    for pat in license_patterns:
        if not any(fnmatch.fnmatchcase(n, pat) or fnmatch.fnmatchcase(n.split("/")[-1], pat) for n in lic):
            problems.append(f"the wheel has no licence file for license-files entry {pat!r} (dist-info/licenses/ holds: {lic})")
    return problems


def collection_errors(output: str) -> list[tuple[str, str]]:
    """(file, reason) of each collection error in the output of `pytest --collect-only -q`.

    Pytest writes one block for each: a header `____ ERROR collecting tests/test_mcp.py ____`, the traceback, and
    a last line `E   ModuleNotFoundError: No module named 'mcp'`. The reason is that last `E` line of the block.
    """
    found: dict[str, str] = {}
    current = None
    for line in output.splitlines():
        m = re.match(r"_+ ERROR collecting (\S+) _+$", line)
        if m:
            current = m.group(1)
            found[current] = ""
        elif current and line.startswith("E "):
            found[current] = line[1:].strip()
        elif line.startswith("====") and "short test summary" in line:
            current = None
    return list(found.items())


def sh(cmd: list[str], *, cwd: Path, env: dict | None = None, log: list[str]) -> subprocess.CompletedProcess:
    print("+ " + shlex.join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    sys.stdout.write(p.stdout + p.stderr)
    log.append(f"$ {shlex.join(cmd)}\n{p.stdout}{p.stderr}(exit code {p.returncode})\n")
    return p


def check_installed_wheel(version: str, wheel: Path, log: list[str], steps: list[list[str]]) -> None:
    """Steps 3 to 6 for one Python version, in a clean venv that is removed at the end. Each finished step adds a row to `steps` (the job summary)."""
    work = Path(tempfile.mkdtemp(prefix=f"wheel-check-{version}-"))
    try:
        venv, run = work / "venv", work / "run"
        py = venv / "bin" / "python"
        if sh(["uv", "venv", "--python", version, str(venv)], cwd=work, log=log).returncode != 0:
            raise CiError("uv venv failed")
        inst = sh(["uv", "pip", "install", "--python", str(py), "-c", str(ROOT / "requirements.lock"), f"{wheel}[dev]"], cwd=work, log=log)
        if inst.returncode != 0:
            raise CiError("installing the wheel failed")
        steps.append([f"clean venv (Python {version})", f"{venv} (outside the repo)"])

        (run / "tests").mkdir(parents=True)
        shutil.copy(ROOT / "tests" / "test_packaging.py", run / "tests" / "test_packaging.py")
        where = sh([str(py), "-I", "-c", "import topopt; print(topopt.__file__)"], cwd=run, log=log)
        loc = where.stdout.strip()
        if where.returncode != 0 or not Path(loc).resolve().is_relative_to(venv.resolve()):
            raise CiError(f"import topopt does not come from the clean venv: {loc!r} {where.stderr.strip()}")
        steps.append([f"import location (Python {version})", loc])

        t = sh([str(py), "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider", "tests/test_packaging.py", "--deselect", DESELECT],
               cwd=run, log=log)
        if t.returncode != 0:
            raise CiError("the packaging test fails against the installed wheel (output above)")
        steps.append([f"packaging test (Python {version})", t.stdout.strip().splitlines()[-1] if t.stdout.strip() else "passed"])

        for cmd in ([str(venv / "bin" / "topopt"), "--version"],
                    [str(venv / "bin" / "topopt"), "check", str(ROOT / "examples" / "hook" / "hook.yaml")]):
            r = sh(cmd, cwd=run, log=log)
            if r.returncode != 0:
                raise CiError(f"{shlex.join(cmd)} exited with code {r.returncode}")
        steps.append([f"installed command (Python {version})", "topopt --version and topopt check examples/hook/hook.yaml exit 0"])

        col = sh([str(py), "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "--rootdir", str(ROOT), "tests"],
                 cwd=ROOT, log=log)
        errors = collection_errors(col.stdout)
        if col.returncode == 0:
            steps.append([f"minimal environment, dev extra only (Python {version})", "pytest --collect-only collects every test module"])
        else:
            steps.append([f"minimal environment, dev extra only (Python {version})",
                          f"pytest --collect-only exits with code {col.returncode}: {len(errors)} module(s) do not collect "
                          "(report only, the job does not fail; the missing `importorskip(\"mcp\")` is open): "
                          + "; ".join(f"{f} ({r[:80]})" for f, r in errors)])
            print(f"REPORT: Python {version}: {len(errors)} test module(s) do not collect in the dev-only environment: "
                  + ", ".join(f for f, _ in errors))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dist", default="dist", help="folder for the wheel")
    ap.add_argument("--python", choices=SUPPORTED_PYTHONS, help="run steps 3 to 6 for this Python version only (default: every supported version)")
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args(argv)
    if not shutil.which("uv"):
        raise CiError("uv is not on PATH")
    versions = (args.python,) if args.python else SUPPORTED_PYTHONS
    dist = Path(args.dist).resolve()
    shutil.rmtree(dist, ignore_errors=True)
    log: list[str] = []
    steps: list[list[str]] = []
    try:
        if sh(["uv", "build", "--wheel", "--out-dir", str(dist), "."], cwd=ROOT, log=log).returncode != 0:
            raise CiError("uv build failed")
        wheels = sorted(dist.glob("topopt-*.whl"))
        if len(wheels) != 1:
            raise CiError(f"expected one wheel in {dist}, found {[w.name for w in wheels]}")
        wheel = wheels[0]
        steps.append(["build", wheel.name])

        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        names = zipfile.ZipFile(wheel).namelist()
        problems = wheel_problems(names, data_files_of_test(ROOT / "tests" / "test_packaging.py"), cfg.get("license-files") or [])
        if problems:
            raise CiError("; ".join(problems))
        steps.append(["wheel contents", f"{len(names)} entries; data files and licence files are in it"])

        for version in versions:
            check_installed_wheel(version, wheel, log, steps)
    finally:
        out = Path(args.artifacts).resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "wheel-check.log").write_text("\n".join(log))
        write_summary("### Wheel job\n\n" + "\n".join(f"- {a}: {b}" for a, b in steps))
    print(f"OK: wheel built, installed in a clean venv, packaging test passed outside the source tree (Python {', '.join(versions)})")
    return 0


if __name__ == "__main__":
    run_main(main)
