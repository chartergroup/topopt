"""PM-2: the modules of the catalogs type-check. Needs pyright (the check is skipped where it is not installed)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYRIGHT = shutil.which("pyright")
MODULES = ("topopt/materials_catalog.py", "topopt/catalogs.py", "topopt/compat.py", "topopt/printers.py")


@pytest.mark.skipif(PYRIGHT is None, reason="pyright is not installed")
def test_the_modules_of_the_catalogs_have_no_pyright_error():
    """The shared `Hit` of `topopt.catalogs` gave `object | None` and `str | None` to the two resolvers, which asked for a profile and a str."""
    done = subprocess.run([PYRIGHT or "pyright", "--outputjson", "--pythonpath", sys.executable, *MODULES], cwd=ROOT,
                          capture_output=True, text=True, timeout=600)
    report = json.loads(done.stdout)
    errors = [f"{Path(d['file']).name}:{d['range']['start']['line'] + 1}: {d['message'].splitlines()[0]}"
              for d in report["generalDiagnostics"] if d["severity"] == "error"]
    assert not errors, "\n".join(errors)
