"""Test that a wheel carries the non-Python files and that the package finds them."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_FILES = [
    "topopt/dashboard/index.html",
    "topopt/dashboard/css/style.css",
    "topopt/dashboard/js/app.js",
    "topopt/dashboard/js/talk.js",
    "topopt/dashboard/vendor/three.module.min.js",
    "topopt/dashboard/fake_optimize.py",
    "topopt/refine/template.html",
    "topopt/refine/vendor/three.min.js",
    "topopt/refine/vendor/OrbitControls.js",
    "topopt/native/accel_chol.c",
    "topopt/native/build.sh",
    "topopt/data/printers.yaml",          # the printer catalog (PM-1)
    "topopt/data/materials.yaml",         # the filament catalog (PM-2)
    # the snapshot that scripts/topopt-env.sh serves before the Python setup is ready (#370; tests/test_launcher_snapshot.py)
    "topopt_mcp/launcher/modern.discover.json",
    "topopt_mcp/launcher/modern.tools.json",
    "topopt_mcp/launcher/modern.prompts.json",
    "topopt_mcp/launcher/modern.resources.json",
    "topopt_mcp/launcher/modern.templates.json",
    "topopt_mcp/launcher/legacy.initialize.json",
    "topopt_mcp/launcher/legacy.tools.json",
    "topopt_mcp/launcher/legacy.prompts.json",
    "topopt_mcp/launcher/legacy.resources.json",
    "topopt_mcp/launcher/legacy.templates.json",
]


def test_data_files_exist_from_package_location():
    from topopt import dashboard_server as ds
    from topopt.refine import build as refine_build  # noqa: F401

    pkg = Path(ds.__file__).resolve().parent
    for rel in DATA_FILES:
        assert (pkg.parent / rel).is_file(), rel
    assert (ds.STATIC_DIR / "index.html").is_file()
    assert ds.SOURCE_REAL.is_file() and ds.SOURCE_FAKE.is_file()
    assert ds.source_path() == ds.SOURCE_REAL


def test_source_payload_names_package_path():
    from topopt import dashboard_server as ds

    data = ds.parse_source(ds.source_path())
    assert data["path"].replace("\\", "/") == "topopt/optimize.py"
    assert data["steps"]


def test_missing_static_dir_message_names_folder(tmp_path, monkeypatch):
    from topopt import dashboard_server as ds

    monkeypatch.setattr(ds, "STATIC_DIR", tmp_path / "nodash")
    msg = ds.static_missing_message()
    assert "\n" not in msg and str(tmp_path / "nodash") in msg


def _build_wheel(out):
    """Build with uv if there is one, else with `python -m build`. Return a wheel path or None."""
    src = out.parent / "src"  # build from a copy: no stale build/ folder, no files written into the checkout
    shutil.copytree(ROOT, src, ignore=shutil.ignore_patterns(
        ".git", "build", "*.egg-info", "recordings", "__pycache__", ".venv", "runs"))
    cmds = []
    if shutil.which("uv"):
        cmds.append(["uv", "build", "--wheel", "-o", str(out), str(src)])
    cmds.append([sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(src)])
    for cmd in cmds:
        if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
            wheels = list(out.glob("topopt-*.whl"))
            if wheels:
                return wheels[0]
    return None


def test_wheel_lists_data_files(tmp_path):
    wheel = _build_wheel(tmp_path / "dist")
    if wheel is None:
        pytest.skip("neither `uv build` nor `python -m build` could build a wheel here")
    names = set(zipfile.ZipFile(wheel).namelist())
    missing = [f for f in DATA_FILES if f not in names]
    assert not missing, missing
