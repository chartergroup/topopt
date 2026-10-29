"""Tests for the optional tet mesher (topopt/tet/, CONTRACTS.md section 14).

The probe tests fake gmsh, so every state runs on every platform. The tests that call
gmsh run only when the probe says "ok" (marker `needs_gmsh`) and report SKIPPED if not.
"""
import json
import re
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest
import trimesh

from helpers import needs_gmsh, recorded_hook_run
from topopt.config import UserError
from topopt.tet import MeshingCrashed, MeshingFailed, TetMesh, TetUnavailable, mesh_stl, tet_quality
from topopt.tet import gmsh_probe as probe
from topopt.tet import mesher

probe_fn = probe.probe

REPO = Path(__file__).resolve().parent.parent

# The keys of `TetMesh.facts`. README.md and CONTRACTS.md section 14 name every one of them.
FACT_KEYS = (
    "tets", "nodes", "quality_min", "quality_mean", "quality_p01", "below_0p1", "below_0p05",
    "volume_fraction_below_0p05", "volume_mm3", "volume_stl_mm3", "volume_ratio", "volume_ratio_stl",
    "deviation_from_input_mm", "stl_sha256", "inverted_fixed", "threads", "repeatable_expected", "sha256",
    "size", "size_given", "size_field_given", "algorithm", "attempts", "failed_attempts", "optimize",
    "gmsh_version", "seconds", "seconds_surface", "seconds_gmsh", "surface_faces",
    "optimize_used", "crashes", "isolated", "seconds_isolation")


def _ok_probe(monkeypatch):
    """Make the probe say ok, so the mesher runs its Python code with a faked `mesh_surface`."""
    ok = probe.TetProbe("ok", version="fake")
    monkeypatch.setattr(probe, "probe", lambda: ok)
    monkeypatch.setattr(probe, "probe_subprocess", lambda *a, **k: ok)


# --------------------------------------------------------------------------- packaging and imports

def test_extra_and_license_note():
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    extra = cfg["project"]["optional-dependencies"]["tet"]
    assert extra == ["gmsh==4.15.2; platform_system == 'Darwin' or "
                     "(platform_system == 'Linux' and platform_machine == 'x86_64')"]
    assert not any("gmsh" in d for d in cfg["project"]["dependencies"])
    assert "gmsh" not in (REPO / "requirements.lock").read_text()
    readme = (REPO / "docs" / "reference" / "print-step-tet.md").read_text()
    assert "GPL-2.0-or-later" in readme and "tet" in readme
    contracts = (REPO / "CONTRACTS.md").read_text()
    section = contracts[contracts.index("## 14. Tet mesher"):]
    for key in FACT_KEYS:
        assert f"`{key}`" in readme, key
        assert f"`{key}`" in section, key


def test_gmsh_is_not_imported_by_the_core():
    """`import topopt`, the CLI, the MCP server and `topopt.tet` do not import gmsh."""
    code = (
        "import sys\n"
        "import topopt, topopt.cli, topopt.optimize, topopt.post, topopt.refine\n"
        "import topopt.tet\n"
        "mods = ['topopt.cli']\n"
        "import importlib.util\n"
        "if importlib.util.find_spec('mcp') is not None:\n"
        "    import topopt_mcp.server\n"
        "    mods.append('topopt_mcp.server')\n"
        "bad = [m for m in sys.modules if m == 'gmsh' or m.startswith('gmsh.')]\n"
        "print('IMPORTED' if bad else 'CLEAN', mods)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.startswith("CLEAN"), out.stdout


# --------------------------------------------------------------------------- the probe

def _fake_import(exc):
    def f():
        raise exc
    return f


def test_probe_ok_has_version(monkeypatch):
    fake = types.SimpleNamespace(__version__="9.9.9", isInitialized=lambda: True,
                                 lib=types.SimpleNamespace(gmshInitialize=1))
    monkeypatch.setattr(probe, "import_gmsh", lambda: fake)
    p = probe_fn()
    assert (p.state, p.version, p.ok) == ("ok", "9.9.9", True)
    assert "ok" in p.line() and "9.9.9" in p.line()


def test_probe_missing_on_a_platform_with_a_wheel(monkeypatch):
    monkeypatch.setattr(probe, "import_gmsh", _fake_import(ModuleNotFoundError("No module named 'gmsh'", name="gmsh")))
    monkeypatch.setattr(probe, "has_wheel_platform", lambda: True)
    p = probe_fn()
    assert p.state == "missing"
    assert "[tet]" in p.fix or "gmsh==4.15.2" in p.fix
    assert "GPL-2.0-or-later" in p.fix


def test_probe_missing_where_there_is_no_wheel(monkeypatch):
    monkeypatch.setattr(probe, "import_gmsh", _fake_import(ModuleNotFoundError("No module named 'gmsh'", name="gmsh")))
    monkeypatch.setattr(probe.sys, "platform", "linux")
    monkeypatch.setattr(probe.platform, "machine", lambda: "aarch64")
    p = probe_fn()
    assert p.state == "missing"
    assert "does not cover" in p.detail and "aarch64" in p.detail
    assert "conda-forge" in p.fix and "OCC" in p.fix and "source" in p.fix


def test_has_wheel_platform(monkeypatch):
    for plat, machine, expect in [("darwin", "arm64", True), ("darwin", "x86_64", True),
                                  ("linux", "x86_64", True), ("linux", "aarch64", False),
                                  ("win32", "AMD64", False)]:
        monkeypatch.setattr(probe.sys, "platform", plat)
        monkeypatch.setattr(probe.platform, "machine", lambda m=machine: m)
        assert probe.has_wheel_platform() is expect, (plat, machine)


def test_probe_load_failed_lists_all_missing_libraries(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "linux")      # the libraries and the apt line are the text for Linux; macOS has its own text
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")   # the nox build is for Linux x86_64 only; an arm64 runner gives no nox line
    err = OSError("libGLU.so.1: cannot open shared object file: No such file or directory")
    monkeypatch.setattr(probe, "import_gmsh", _fake_import(err))
    monkeypatch.setattr(probe, "missing_libraries",
                        lambda: ["libGLU.so.1", "libXft.so.2", "libXinerama.so.1"])
    p = probe_fn()
    assert p.state == "load_failed"
    text = p.detail + " " + p.fix
    for lib in ("libGLU.so.1", "libXft.so.2", "libXinerama.so.1"):
        assert lib in text
    assert "apt install libglu1-mesa libxft2 libxinerama1" in text
    assert "libopengl0" in text                       # needed for the unpack without root
    assert "python-packages-dev-nox" in text          # the nox build as an alternative
    assert "\n" not in p.line()


def test_probe_load_failed_adds_the_library_of_the_error(monkeypatch):
    err = OSError("libfoo.so.9: cannot open shared object file")
    monkeypatch.setattr(probe, "import_gmsh", _fake_import(err))
    monkeypatch.setattr(probe, "missing_libraries", lambda: [])
    p = probe_fn()
    assert p.state == "load_failed" and "libfoo.so.9" in p.detail


@pytest.mark.parametrize("exc", [
    ModuleNotFoundError("No module named 'numpy'", name="numpy"),
    RuntimeError("boom"),
    OSError("something else"),
    Exception("Could not find Gmsh shared library"),
])
def test_probe_never_raises(monkeypatch, exc):
    monkeypatch.setattr(probe, "import_gmsh", _fake_import(exc))
    monkeypatch.setattr(probe, "missing_libraries", lambda: (_ for _ in ()).throw(RuntimeError("ldd broke")))
    p = probe_fn()
    assert p.state in ("load_failed", "missing")
    assert p.state in probe.STATES and p.fix


def test_missing_libraries_reads_ldd_output(monkeypatch):
    ldd = ("\tlinux-vdso.so.1 (0x00007ffc)\n"
           "\tlibGLU.so.1 => not found\n"
           "\tlibm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f)\n"
           "\tlibXft.so.2 => not found\n"
           "\tlibXinerama.so.1 => not found\n")
    monkeypatch.setattr(probe.sys, "platform", "linux")
    monkeypatch.setattr(probe.shutil, "which", lambda name: "/usr/bin/ldd")
    monkeypatch.setattr(probe, "_libgmsh_candidates", lambda: [Path("/x/libgmsh.so.4.15")])
    monkeypatch.setattr(probe.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=ldd, returncode=0))
    assert probe.missing_libraries() == ["libGLU.so.1", "libXft.so.2", "libXinerama.so.1"]


def test_missing_libraries_is_empty_without_ldd(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "linux")
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)
    assert probe.missing_libraries() == []
    monkeypatch.setattr(probe.sys, "platform", "darwin")
    assert probe.missing_libraries() == []


@pytest.mark.parametrize("state,text", [
    ("missing", "gmsh is not installed"),
    ("load_failed", "Missing system libraries: libGLU.so.1"),
])
def test_mesher_raises_one_clear_error_when_gmsh_is_not_ok(monkeypatch, tmp_path, state, text):
    bad = probe.TetProbe(state, detail=text, fix="FIX TEXT")
    monkeypatch.setattr(probe, "probe", lambda: bad)
    monkeypatch.setattr(probe, "probe_subprocess", lambda *a, **k: bad)
    with pytest.raises(TetUnavailable) as info:
        mesh_stl(tmp_path / "nothing.stl")
    with pytest.raises(TetUnavailable):
        mesh_stl(tmp_path / "nothing.stl", isolate=False)
    msg = str(info.value)
    assert isinstance(info.value, UserError) and not isinstance(info.value, OSError)
    assert state in msg and text in msg and "FIX TEXT" in msg
    with pytest.raises(TetUnavailable):
        probe.require_gmsh()


def test_real_probe_in_a_child_python():
    """The probe state is one of the three, on this machine, and this process does not load libgmsh for it."""
    p = probe.probe_subprocess()
    assert p.state in probe.STATES
    assert (p.version is not None) == p.ok


def test_probe_subprocess_never_raises(monkeypatch):
    monkeypatch.setattr(probe.sys, "executable", "/nonexistent/python")
    p = probe.probe_subprocess()
    assert p.state == "load_failed" and "probe process failed" in p.detail


def test_probe_load_failed_when_libgmsh_is_not_found(monkeypatch):
    """gmsh.py prints a warning and loads the C library of the process: the probe must not say ok."""
    def fake_import():
        print("Warning: could not find Gmsh shared library libgmsh.so.4.15")
        return types.SimpleNamespace(__version__="4.15.2", lib=types.SimpleNamespace())   # no gmshInitialize
    monkeypatch.setattr(probe, "import_gmsh", fake_import)
    monkeypatch.setitem(sys.modules, "gmsh", types.SimpleNamespace())
    p = probe_fn()
    assert p.state == "load_failed" and "libgmsh not found" in p.detail
    assert "gmsh" not in sys.modules                 # a later import tries again


def test_probe_stray_stdout_does_not_reach_the_caller(monkeypatch, capsys):
    def fake_import():
        print("stray text")
        return types.SimpleNamespace(__version__="1", lib=types.SimpleNamespace(gmshInitialize=1),
                                     isInitialized=lambda: True)
    monkeypatch.setattr(probe, "import_gmsh", fake_import)
    assert probe_fn().ok
    assert "stray text" not in capsys.readouterr().out


def test_load_failed_hint_on_a_mac(monkeypatch):
    monkeypatch.setattr(probe, "import_gmsh", _fake_import(OSError("dlopen failed: libgmsh.dylib")))
    monkeypatch.setattr(probe.sys, "platform", "darwin")
    p = probe_fn()
    assert p.state == "load_failed"
    text = p.detail + p.fix
    assert "otool -L" in text and "macOS 12" in text and "static libomp" in text
    assert "apt" not in text and "nox" not in text and "LD_LIBRARY_PATH" not in text


def test_load_failed_hint_on_an_intel_mac_names_the_x86_64_wheel(monkeypatch):
    """E4: the old text named only the arm64 wheel (macOS 12). The x86_64 wheel needs macOS 10.15 (wheel tag macosx_10_15_x86_64)."""
    monkeypatch.setattr(probe, "import_gmsh", _fake_import(OSError("dlopen failed: libgmsh.dylib")))
    monkeypatch.setattr(probe.sys, "platform", "darwin")
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")
    p = probe_fn()
    assert p.state == "load_failed"
    text = p.detail + p.fix
    assert "The macOS x86_64 wheel" in text and "macOS 10.15 or newer" in text and "architecture of Python (x86_64)" in text
    assert "otool -L" in text and "apt" not in text and "LD_LIBRARY_PATH" not in text


def test_nox_build_is_offered_on_linux_x86_64_only(monkeypatch):
    monkeypatch.setattr(probe, "import_gmsh", _fake_import(OSError("libGLU.so.1: cannot open shared object file")))
    monkeypatch.setattr(probe, "missing_libraries", lambda: ["libGLU.so.1"])
    monkeypatch.setattr(probe.sys, "platform", "linux")
    monkeypatch.setattr(probe.platform, "machine", lambda: "aarch64")
    assert "nox" not in probe_fn().fix
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")
    assert "nox" in probe_fn().fix


# --------------------------------------------------------------------------- numpy helpers

def test_quality_of_a_regular_tet_is_one():
    p = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float).T   # (3, 4)
    t = np.array([[0], [1], [2], [3]])
    assert tet_quality(p, t)[0] == pytest.approx(1.0)


def test_signed_volumes_and_hash():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    v = mesher.signed_volumes(nodes, np.array([[0, 1, 2, 3], [0, 2, 1, 3]]))
    assert v == pytest.approx([1 / 6, -1 / 6])
    h1 = mesher.mesh_hash(nodes, np.array([[0, 1, 2, 3]]))
    assert h1 == mesher.mesh_hash(nodes.copy(), np.array([[0, 1, 2, 3]]))
    assert h1 != mesher.mesh_hash(nodes, np.array([[0, 2, 1, 3]]))


# --------------------------------------------------------------------------- surface (no gmsh)

def _sphere_stl(path, radius=10.0, subdiv=3):
    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=radius)
    m.export(str(path))
    return m


def test_surface_preparation_keeps_a_closed_manifold(tmp_path):
    from topopt.tet.surface import prepare_surface, surface_is_closed_manifold
    m = _sphere_stl(tmp_path / "sphere.stl")
    prep = prepare_surface(str(tmp_path / "sphere.stl"), 3.0, verbose=False)
    assert surface_is_closed_manifold(prep.V, prep.F)
    assert len(prep.F) < len(m.faces)
    assert mesher.surface_volume(prep.V, prep.F) == pytest.approx(m.volume, rel=0.05)


def test_open_surface_is_rejected(tmp_path):
    from topopt.tet.surface import SurfaceRejected, prepare_surface
    m = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    m.faces = m.faces[1:]                       # one hole
    m.export(str(tmp_path / "open.stl"))
    with pytest.raises(SurfaceRejected):
        prepare_surface(str(tmp_path / "open.stl"), 3.0, verbose=False)
    assert issubclass(SurfaceRejected, UserError)


def test_unknown_surface_method_raises(tmp_path):
    from topopt.tet.surface import prepare_surface
    _sphere_stl(tmp_path / "s.stl", subdiv=2)
    with pytest.raises(ValueError):
        prepare_surface(str(tmp_path / "s.stl"), 3.0, method="vtk", verbose=False)


# --------------------------------------------------------------------------- gmsh

@needs_gmsh
def test_mesh_sphere_facts(tmp_path):
    stl = tmp_path / "sphere.stl"
    m = _sphere_stl(stl)
    mesh = mesh_stl(stl, size=3.0)
    f = mesh.facts
    assert mesh.nodes.shape[1] == 3 and mesh.tets.shape[1] == 4
    assert f["tets"] == len(mesh.tets) and f["nodes"] == len(mesh.nodes)
    assert (mesh.volumes() > 0).all() and f["inverted_fixed"] == 0
    assert abs(f["volume_ratio"] - 1.0) <= 1e-6
    assert f["volume_ratio_stl"] == pytest.approx(1.0, abs=0.05)
    assert f["volume_stl_mm3"] == pytest.approx(m.volume, rel=1e-6)
    assert f["quality_min"] >= 0.1 and f["below_0p1"] == 0
    assert 0.5 < f["quality_mean"] <= 1.0
    assert f["threads"] == 1 and f["repeatable_expected"] is True
    assert set(f) == set(FACT_KEYS)
    assert f["below_0p05"] == 0 and f["volume_fraction_below_0p05"] == 0.0
    assert f["isolated"] is True and f["crashes"] == [] and f["optimize_used"] == f["optimize"] == "Netgen"
    assert 0 < f["seconds_isolation"] < 30
    assert set(f["deviation_from_input_mm"]) >= {"mean_mm", "p99_mm", "max_mm"}
    import hashlib
    assert f["stl_sha256"] == hashlib.sha256(stl.read_bytes()).hexdigest()
    assert f["gmsh_version"] and f["seconds"] > 0 and f["size"] == 3.0 and f["size_given"] is True
    assert f["sha256"] == mesh.sha256()
    # every node is used and indices are in range
    assert set(np.unique(mesh.tets)) == set(range(len(mesh.nodes)))
    saved = mesh.save(tmp_path / "m")                 # the suffix .npz is added
    assert saved.name == "m.npz" and saved.is_file()
    again = TetMesh.load(saved)
    assert again.nodes.dtype == np.float64 and again.tets.dtype == np.int64
    assert again.sha256() == mesh.sha256() == again.facts["sha256"]
    assert again.facts == json.loads(json.dumps(mesh.facts))
    assert mesh.save(tmp_path / "n.npz").name == "n.npz"


@needs_gmsh
def test_one_thread_is_repeatable(tmp_path):
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    a = mesh_stl(stl, size=3.0, threads=1)
    b = mesh_stl(stl, size=3.0, threads=1)
    c = mesh_stl(stl, size=3.0, threads=1, isolate=False)          # in this process
    assert a.facts["isolated"] is True and c.facts["isolated"] is False
    assert a.facts["sha256"] == b.facts["sha256"] == c.facts["sha256"]
    assert np.array_equal(a.tets, b.tets) and np.array_equal(a.nodes, b.nodes)
    assert np.array_equal(a.tets, c.tets) and np.array_equal(a.nodes, c.nodes)


@needs_gmsh
def test_more_threads_are_allowed(tmp_path):
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    mesh = mesh_stl(stl, size=3.0, threads=3)
    assert mesh.facts["threads"] == 3 and mesh.facts["repeatable_expected"] is False
    assert abs(mesh.facts["volume_ratio"] - 1.0) <= 1e-6


@needs_gmsh
def test_default_size_and_log(tmp_path):
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)
    lines = []
    mesh = mesh_stl(stl, log=lines.append)
    assert mesh.facts["size_given"] is False and mesh.facts["size"] == pytest.approx(mesher.default_size(stl))
    assert lines and "gmsh" in " ".join(lines)


@needs_gmsh
def test_open_stl_is_a_user_error(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    m.faces = m.faces[1:]
    m.export(str(tmp_path / "open.stl"))
    with pytest.raises(UserError, match="closed"):
        mesh_stl(tmp_path / "open.stl", size=3.0)
    with pytest.raises(UserError, match="not found"):
        mesh_stl(tmp_path / "absent.stl", size=3.0)


@needs_gmsh
def test_meshing_failure_is_a_user_error(tmp_path, monkeypatch):
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)

    def fail(*a, **k):
        raise MeshingFailed("algorithm 10: nope", [])
    monkeypatch.setattr(mesher, "mesh_surface", fail)
    with pytest.raises(MeshingFailed, match="nope"):
        mesh_stl(stl, isolate=False, size=3.0)


# --------------------------------------------------------------------------- the port: retry, orientation, field (no gmsh needed)

_ONE_TET = (np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float), np.array([[0], [1], [2], [3]]))


def _fake_info():
    return dict(algorithm=10, seconds=0.0, volume_ratio=1.0)


def test_retry_loop_refines_around_reported_points(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    calls = []

    def fake_mesh_surface(V, F, size, **kw):
        calls.append(len(F))
        if len(calls) == 1:
            raise MeshingFailed("algorithm 10: HXT 3D mesh failed", [(0.0, 0.0, 10.0)])
        return (*_ONE_TET, _fake_info())
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    lines = []
    mesh = mesh_stl(stl, isolate=False, size=3.0, log=lines.append)
    assert mesh.facts["attempts"] == 2
    assert mesh.facts["failed_attempts"][0]["points"] == [(0.0, 0.0, 10.0)]
    assert calls[1] > calls[0]                       # the second surface is finer near the point
    assert any("1 local refinement point" in ln for ln in lines)


def test_retry_stops_at_once_without_points(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)
    calls = []

    def fake_mesh_surface(V, F, size, **kw):
        calls.append(1)
        raise MeshingFailed("algorithm 10: nope", [])
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    with pytest.raises(MeshingFailed, match="nope"):
        mesh_stl(stl, isolate=False, size=3.0)
    assert len(calls) == 1


def test_retry_gives_up_after_max_attempts(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)

    def fake_mesh_surface(V, F, size, **kw):
        raise MeshingFailed("algorithm 10: nope", [(0.0, 0.0, 10.0)])
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    with pytest.raises(MeshingFailed, match=r"every surface attempt \(2\)"):
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=2)


def test_inverted_tet_is_reordered(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)
    p, t = _ONE_TET
    swapped = t[[0, 2, 1, 3]]                         # node order 0,2,1,3: a negative volume
    assert mesher.signed_volumes(p.T, swapped.T)[0] < 0
    monkeypatch.setattr(mesher, "mesh_surface", lambda V, F, size, **kw: (p, swapped, _fake_info()))
    mesh = mesh_stl(stl, isolate=False, size=3.0)
    assert mesh.facts["inverted_fixed"] == 1
    assert (mesh.volumes() > 0).all()
    assert sorted(mesh.tets[0]) == [0, 1, 2, 3]


def test_size_field_makes_a_finer_surface(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    calls = []

    def fake_mesh_surface(V, F, size, **kw):
        calls.append(len(F))
        return (*_ONE_TET, _fake_info())
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    mesh_stl(stl, isolate=False, size=4.0)
    seen = []

    def field(P):
        seen.append(len(P))
        return np.where(P[:, 2] > 0, 1.5, 4.0)        # fine on the upper half
    mesh = mesh_stl(stl, isolate=False, size=4.0, size_field=field)
    assert seen and calls[1] > calls[0]
    assert mesh.facts["size_field_given"] is True


def test_bad_arguments_are_user_errors(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)
    with pytest.raises(UserError, match="threads"):
        mesh_stl(stl, size=3.0, threads=0)
    with pytest.raises(UserError, match="size"):
        mesh_stl(stl, size=-1.0)
    with pytest.raises(UserError, match="max_attempts"):
        mesh_stl(stl, size=3.0, max_attempts=0)
    bad = tmp_path / "text.stl"
    bad.write_text("this is not an STL file")
    with pytest.raises(UserError, match="no triangles|cannot read"):
        mesh_stl(bad, size=3.0)
    broken = tmp_path / "broken.stl"
    broken.write_bytes(b"solid x\nfacet normal 0 0 1\n" + b"\x00" * 7)
    with pytest.raises(UserError):
        mesh_stl(broken, size=3.0)
    with pytest.raises(UserError, match="not found"):
        mesh_stl(tmp_path / "absent.stl", size=3.0)


def test_environment_reports_the_tet_state():
    from topopt.environment import environment_data
    d = environment_data()
    assert d["tet"]["state"] in probe.STATES
    assert set(d["tet"]) == {"state", "version", "detail", "fix"}
    json.dumps(d)


# --------------------------------------------------------------------------- gmsh: threads and self-intersection

@needs_gmsh
def test_mesh_stl_works_off_the_main_thread(tmp_path):
    """gmsh.initialize() with interruptible=True raises ValueError in a thread. Two threads run one after the other.

    The threads run in a child Python. On macOS arm64 (gmsh 4.15.2) a process whose first gmsh.initialize() ran in a thread
    crashes (SIGSEGV) at a later initialize in the main thread: the next in-process gmsh test of the same pytest worker died."""
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    code = (
        "import json, threading, gmsh\n"
        "from topopt.tet import mesh_stl\n"
        "results, errors = [], []\n"
        "def work():\n"
        "    try:\n"
        f"        results.append(mesh_stl({str(stl)!r}, isolate=False, size=3.0).facts['sha256'])\n"
        "    except BaseException as e:\n"
        "        errors.append(repr(e))\n"
        "threads = [threading.Thread(target=work) for _ in range(2)]\n"
        "for t in threads:\n"
        "    t.start()\n"
        "for t in threads:\n"
        "    t.join(120)\n"
        "print(json.dumps(dict(results=results, errors=errors, initialized=gmsh.isInitialized())))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    r = json.loads(out.stdout.splitlines()[-1])
    assert not r["errors"], r["errors"]
    assert len(r["results"]) == 2 and r["results"][0] == r["results"][1]
    assert not r["initialized"]


@needs_gmsh
def test_gmsh_is_finalized_after_a_failed_init(monkeypatch, tmp_path):
    import gmsh
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)
    real_init = gmsh.initialize

    def bad_init(*a, **k):
        real_init(*a, **k)
        raise ValueError("signal only works in main thread")   # the C init has run, then Python fails
    monkeypatch.setattr(gmsh, "initialize", bad_init)
    with pytest.raises(MeshingFailed):
        mesh_stl(stl, isolate=False, size=3.0)
    assert not gmsh.isInitialized()


@needs_gmsh
def test_self_intersecting_surface_reports_a_point_and_stops_at_once(tmp_path):
    """One vertex of an icosphere pushed through the other side: gmsh reports the intersection point. The STL itself crosses
    itself there, so the surface preparation cannot remove the crossing and the mesher stops after the first attempt (P4, F2-6:
    it made all `max_attempts` before, 2.7 minutes each on the case 2 STL)."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    V = np.array(m.vertices)
    V[0] *= -1.3
    with pytest.raises(MeshingFailed) as info:
        mesher.mesh_surface(V, np.array(m.faces), 3.0, algorithms=(10,))
    assert len(info.value.points) >= 1 and len(info.value.points[0]) == 3
    stl = tmp_path / "pushed.stl"
    trimesh.Trimesh(V, m.faces, process=False).export(str(stl))
    lines = []
    with pytest.raises(MeshingFailed, match="The STL itself crosses itself") as stop:
        mesh_stl(stl, size=3.0, max_attempts=3, log=lines.append)
    assert stop.value.crossings["stopped"] == "input_crosses_itself" and stop.value.crossings["input"]["count"] > 0
    assert sum("surface preparation" in ln for ln in lines) == 1        # one attempt, not three


# --------------------------------------------------------------------------- P4, F2-6: the crossings of the prepared surface

def _pushed(push):
    """An icosphere with one vertex pushed through the other side (a surface that crosses itself), as (V, F)."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    V = np.array(m.vertices)
    V[0] *= push
    return V, np.array(m.faces)


PUSHED_CROSSINGS = 10      # the icosphere (subdivisions 3) with vertex 0 pushed to -1.3 x: 10 pairs of faces cross (test_tet_intersect.py)
CUBES_CROSSINGS = 18       # two unit cubes, the second moved by (0.5, 0.5, 0.5): 18 pairs of faces cross


def _fake_preparation(monkeypatch, surfaces):
    """`prepare_surface` that returns the given (V, F) surfaces one after the other (the last one again and again).

    It records the `size_field` of each call in the list that it returns."""
    from topopt.tet import surface as surf
    fields, n = [], [0]

    def fake(path, h, **kw):
        fields.append(kw.get("size_field"))
        V, F = surfaces[min(n[0], len(surfaces) - 1)]
        n[0] += 1
        return surf.PreparedSurface(V, F, {}, {"deviation_from_input_mm": {"mean_mm": 0.0, "p99_mm": 0.0, "max_mm": 0.0,
                                                                            "samples": 0}}, [])
    monkeypatch.setattr(surf, "prepare_surface", fake)
    return fields


def test_a_prepared_surface_that_crosses_itself_is_refined_at_every_crossing_in_one_retry(monkeypatch, tmp_path):
    """gmsh names one place for each call. The mesher finds all the crossings of the prepared surface and refines around all of
    them in the next attempt (before: one place for each attempt, 10 attempts for 11 places)."""
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)                                              # the STL is clean
    n = PUSHED_CROSSINGS
    fields = _fake_preparation(monkeypatch, [_pushed(-1.3), _pushed(1.0)])
    calls = []

    def fake_mesh_surface(V, F, size, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise MeshingFailed("algorithm 10: HXT 3D mesh failed", [(0.0, 0.0, 10.0)])      # one place only
        return (*_ONE_TET, _fake_info())
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    lines = []
    mesh = mesh_stl(stl, isolate=False, size=3.0, log=lines.append)
    assert mesh.facts["attempts"] == 2 and len(calls) == 2
    assert any(f"the prepared surface crosses itself at {n} place(s)" in ln for ln in lines)
    assert any(f"attempt 2, {n + 1} local refinement point(s)" in ln for ln in lines)      # the gmsh point and every crossing
    assert fields[0] is None and fields[1] is not None


def test_retries_stop_when_the_refinement_does_not_make_the_crossings_fewer(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    n = PUSHED_CROSSINGS
    _fake_preparation(monkeypatch, [_pushed(-1.3)])               # the same crossings, at the same places, in every attempt
    calls = []

    def fake_mesh_surface(V, F, size, **kw):
        calls.append(1)
        raise MeshingFailed("algorithm 10: HXT 3D mesh failed", [(0.0, 0.0, 10.0)])
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    with pytest.raises(MeshingFailed) as stop:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=10)
    assert len(calls) == 2                                        # not 10
    msg = str(stop.value)
    assert "did not make them fewer" in msg and "does not cross itself" in msg and f"{n}, {n}" in msg
    # P4 review, nit: the text gives a direction and the names of the options
    assert re.search(r"try a mesh size nearer to the mean edge of the STL \(1\.\d+ mm\)", msg)
    assert "`--mesh-size` of `topopt verify`" in msg and "MCP `mesh_size_mm`" in msg
    assert stop.value.crossings["stopped"] == "no_progress"
    assert stop.value.crossings["prepared"]["count"] == n and stop.value.crossings["prepared"]["attempts"] == [n, n]
    assert stop.value.crossings["input"]["count"] == 0
    assert len(stop.value.crossings["prepared"]["points"]) == n and len(stop.value.crossings["prepared"]["points"][0]) == 3


def _moved(surface, dx):
    V, F = surface
    return V + np.array([dx, 0.0, 0.0]), F


def _two_of(surface, dx):
    """The surface and a copy of it moved by `dx` in x: the crossings of both, far from each other."""
    V, F = surface
    return np.vstack([V, V + np.array([dx, 0.0, 0.0])]), np.vstack([F, F + len(V)])


@pytest.mark.parametrize("second", ["moved", "added"])
def test_retries_go_on_when_the_crossings_are_at_other_places(monkeypatch, tmp_path, second):
    """The same number of crossings (or more) is no stop if they are at places that no earlier attempt refined: the refinement
    moved them, and the next attempt refines there (P4 review, L2: before, the count alone ended the retries)."""
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    first = _pushed(-1.3)
    other = _moved(first, 50.0) if second == "moved" else _two_of(first, 50.0)       # 10 crossings at new places / 10 old + 10 new
    fields = _fake_preparation(monkeypatch, [first, other, _pushed(1.0)])
    calls = []

    def fake_mesh_surface(V, F, size, **kw):
        calls.append(1)
        if len(calls) < 3:
            raise MeshingFailed("algorithm 10: HXT 3D mesh failed", [(0.0, 0.0, 10.0)])
        return (*_ONE_TET, _fake_info())
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    mesh = mesh_stl(stl, isolate=False, size=3.0, max_attempts=5)
    assert mesh.facts["attempts"] == 3 and len(calls) == 3 and len(fields) == 3


def test_retries_go_on_while_the_crossings_become_fewer(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    boxes = [trimesh.creation.box(extents=(1, 1, 1)) for _ in range(2)]
    boxes[1].apply_translation((0.5, 0.5, 0.5))                       # two overlapping cubes: 18 crossings
    big = (np.vstack([b.vertices for b in boxes]), np.vstack([boxes[0].faces, boxes[1].faces + len(boxes[0].vertices)]))
    small = _pushed(-1.3)                                              # 10 crossings
    assert 0 < PUSHED_CROSSINGS < CUBES_CROSSINGS
    _fake_preparation(monkeypatch, [big, small, _pushed(1.0)])
    calls = []

    def fake_mesh_surface(V, F, size, **kw):
        calls.append(1)
        if len(calls) < 3:
            raise MeshingFailed("algorithm 10: HXT 3D mesh failed", [(0.0, 0.0, 10.0)])
        return (*_ONE_TET, _fake_info())
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    mesh = mesh_stl(stl, isolate=False, size=3.0, max_attempts=5)
    assert mesh.facts["attempts"] == 3 and len(calls) == 3


def test_an_stl_that_crosses_itself_stops_after_the_first_attempt_with_the_places(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "pushed.stl"
    V, F = _pushed(-1.3)
    trimesh.Trimesh(V, F, process=False).export(str(stl))
    calls = []

    def fake_mesh_surface(P, Q, size, **kw):
        calls.append(1)
        raise MeshingFailed("algorithm 10: HXT 3D mesh failed", [(0.0, 0.0, 10.0)])
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    with pytest.raises(MeshingFailed) as stop:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=10)
    assert len(calls) == 1
    msg = str(stop.value)
    assert "The STL itself crosses itself at" in msg and "cannot remove a crossing of the input" in msg
    assert "`--mesh-size` of `topopt verify`" in msg and "MCP `mesh_size_mm`" in msg            # P4 review, nit
    assert stop.value.crossings["stopped"] == "input_crosses_itself" and stop.value.crossings["input"]["count"] > 0
    assert stop.value.points == [(0.0, 0.0, 10.0)]


def test_a_crossing_of_the_stl_far_from_the_failure_does_not_stop_the_retries(monkeypatch, tmp_path):
    """The input crosses itself, but the prepared surface fails somewhere else (more than `size` away): the retry may help."""
    _ok_probe(monkeypatch)
    stl = tmp_path / "pushed.stl"
    V, F = _pushed(-1.3)
    trimesh.Trimesh(V, F, process=False).export(str(stl))
    clean = _sphere_stl(tmp_path / "clean.stl")
    _fake_preparation(monkeypatch, [(np.array(clean.vertices), np.array(clean.faces))])       # a clean prepared surface
    calls = []

    def fake_mesh_surface(P, Q, size, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise MeshingFailed("algorithm 10: HXT 3D mesh failed", [(500.0, 500.0, 500.0)])
        return (*_ONE_TET, _fake_info())
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    mesh = mesh_stl(stl, isolate=False, size=3.0)
    assert mesh.facts["attempts"] == 2


def test_the_attempt_limit_still_ends_the_retries_and_the_error_has_the_crossing_facts(monkeypatch, tmp_path):
    """The last prepared surface had no crossing that the check can see (gmsh named a place): the old rule applies. Each attempt
    gets another surface (a moved copy), as a refinement that changes the surface does; with the same surface in two attempts in a
    row the retries end earlier (F1-5 review, MJ1: `test_retries_stop_when_the_prepared_surface_does_not_change`)."""
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    m = _sphere_stl(stl, subdiv=2)
    _fake_preparation(monkeypatch, [_moved((np.array(m.vertices), np.array(m.faces)), dx) for dx in (0.0, 0.5, 1.0)])

    def fake_mesh_surface(V, F, size, **kw):
        raise MeshingFailed("algorithm 10: nope", [(0.0, 0.0, 10.0)])
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    with pytest.raises(MeshingFailed, match=r"every surface attempt \(3\)") as stop:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=3)
    assert stop.value.crossings["prepared"]["count"] == 0 and stop.value.crossings["prepared"]["attempts"] == [0, 0, 0]
    assert "Crossings found in the prepared surface in each attempt: 0, 0, 0" in str(stop.value)      # P4 review, nit
    assert "stopped" not in stop.value.crossings


def _search_that_fails(monkeypatch, on_calls=None):
    """`find_crossings` that raises MemoryError (on every call, or on the calls with the given numbers, counted from 1).
    Returns the list of the calls."""
    from topopt.tet import intersect
    real, calls = intersect.find_crossings, []

    def fake(V, F):
        calls.append(len(F))
        if on_calls is None or len(calls) in on_calls:
            raise MemoryError("Unable to allocate 2.4 GiB for an array")
        return real(V, F)
    monkeypatch.setattr(intersect, "find_crossings", fake)
    return calls


def test_a_search_for_crossings_that_fails_keeps_the_one_point_refinement(monkeypatch, tmp_path):
    """A MemoryError in the search must not replace the failure of gmsh: the retry refines around the place that gmsh named, as
    before P4, and the log has a note (P4 review, L3)."""
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    fields = _fake_preparation(monkeypatch, [_pushed(-1.3), _pushed(1.0)])
    _search_that_fails(monkeypatch)
    calls = []

    def fake_mesh_surface(V, F, size, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise MeshingFailed("algorithm 10: HXT 3D mesh failed", [(0.0, 0.0, 10.0)])
        return (*_ONE_TET, _fake_info())
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    lines = []
    mesh = mesh_stl(stl, isolate=False, size=3.0, log=lines.append)
    assert mesh.facts["attempts"] == 2 and len(calls) == 2
    assert any("attempt 2, 1 local refinement point(s)" in ln for ln in lines)            # the gmsh point only
    assert any("search for crossings of the prepared surface failed: MemoryError" in ln for ln in lines)
    assert fields[1] is not None


def test_when_the_searches_fail_every_error_has_the_note_and_the_facts(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)
    _search_that_fails(monkeypatch)

    def fake_mesh_surface(V, F, size, **kw):
        raise MeshingFailed("algorithm 10: nope", [(0.0, 0.0, 10.0)])
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    with pytest.raises(MeshingFailed, match=r"every surface attempt \(2\)") as stop:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=2)
    assert "could not run" in str(stop.value) and "MemoryError" in str(stop.value)
    assert "in each attempt: not searched, not searched" in str(stop.value)
    failed = stop.value.crossings["search_failed"]
    assert [(f["attempt"], f["surface"]) for f in failed] == [(1, "prepared"), (1, "input"), (2, "prepared")]
    assert all(f["error"].startswith("MemoryError") for f in failed)
    assert "prepared" not in stop.value.crossings                                      # no search worked


def test_a_failed_search_and_no_place_from_gmsh_gives_the_gmsh_error_with_the_note(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)
    _search_that_fails(monkeypatch)

    def fake_mesh_surface(V, F, size, **kw):
        raise MeshingFailed("algorithm 10: HXT 3D mesh failed", [])
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    with pytest.raises(MeshingFailed, match="HXT 3D mesh failed") as stop:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=4)
    assert "could not run" in str(stop.value) and stop.value.crossings["search_failed"][0]["surface"] == "prepared"


def test_no_progress_is_said_when_only_the_search_of_the_stl_failed(monkeypatch, tmp_path):
    """The prepared surface was searched (the same 10 crossings twice), the STL was not (call 2 fails): the stop is
    `no_progress`, and the text says that the STL was not searched instead of that it does not cross itself."""
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    _fake_preparation(monkeypatch, [_pushed(-1.3)])
    _search_that_fails(monkeypatch, on_calls={2})

    def fake_mesh_surface(V, F, size, **kw):
        raise MeshingFailed("algorithm 10: HXT 3D mesh failed", [(0.0, 0.0, 10.0)])
    monkeypatch.setattr(mesher, "mesh_surface", fake_mesh_surface)
    with pytest.raises(MeshingFailed) as stop:
        mesh_stl(stl, isolate=False, size=3.0, max_attempts=10)
    msg = str(stop.value)
    assert stop.value.crossings["stopped"] == "no_progress" and "was not searched for crossings" in msg
    assert "does not cross itself" not in msg and "could not run" in msg
    assert "input" not in stop.value.crossings
    assert [f["surface"] for f in stop.value.crossings["search_failed"]] == ["input"]


# --------------------------------------------------------------------------- isolation: a crash of gmsh must not kill the caller

SELF_KILL = [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"]


def test_parse_optimize():
    assert mesher.parse_optimize(None) == [] and mesher.parse_optimize("") == []
    assert mesher.parse_optimize("Netgen") == ["Netgen"]
    assert mesher.parse_optimize("Gmsh+Netgen") == ["Gmsh", "Netgen"]


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")
def test_child_that_dies_on_sigsegv_gives_meshingcrashed(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)
    monkeypatch.setattr(mesher, "_worker_command", lambda i, o: SELF_KILL)
    with pytest.raises(MeshingCrashed) as info:
        mesh_stl(stl, size=3.0)                       # default: isolated, with the fallbacks
    e = info.value
    assert isinstance(e, MeshingFailed) and isinstance(e, UserError)
    msg = str(e)
    assert "SIGSEGV" in msg and "sphere.stl" in msg and "3 mm" in msg and "attempt 1" in msg
    assert "'Netgen'" in msg and "Options:" in msg and "larger size" in msg
    assert [c["optimize"] for c in e.crashes] == ["Netgen", "Gmsh+Netgen", "Gmsh"]
    assert all(c["signal"] == "SIGSEGV" and c["size"] == 3.0 and c["attempt"] == 1 for c in e.crashes)
    # no fallback: one crash
    with pytest.raises(MeshingCrashed) as info:
        mesh_stl(stl, size=3.0, fallback_optimize=())
    assert len(info.value.crashes) == 1


def test_child_failure_without_a_signal_is_meshingfailed(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)
    monkeypatch.setattr(mesher, "_worker_command",
                        lambda i, o: [sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"])
    with pytest.raises(MeshingFailed, match=r"exit code 3.*boom") as info:
        mesh_stl(stl, size=3.0)
    assert not isinstance(info.value, MeshingCrashed)


def test_crash_then_fallback_optimizer_is_recorded(monkeypatch, tmp_path):
    _ok_probe(monkeypatch)
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl, subdiv=2)
    seen = []

    def fake(V, F, size, **kw):
        seen.append(kw["optimize"])
        if kw["optimize"] == "Netgen":
            raise MeshingCrashed("died", dict(signal="SIGSEGV", returncode=-11, size=size, optimize="Netgen"))
        return (*_ONE_TET, dict(_fake_info(), seconds_isolation=0.5))
    monkeypatch.setattr(mesher, "mesh_surface_isolated", fake)
    lines = []
    mesh = mesh_stl(stl, size=3.0, log=lines.append)
    f = mesh.facts
    assert seen == ["Netgen", "Gmsh+Netgen"]
    assert f["optimize"] == "Netgen" and f["optimize_used"] == "Gmsh+Netgen"
    assert f["crashes"] == [dict(attempt=1, signal="SIGSEGV", returncode=-11, size=3.0, optimize="Netgen")]
    assert f["isolated"] is True and f["seconds_isolation"] == 0.5
    assert any("trying optimize='Gmsh+Netgen'" in ln for ln in lines)


@needs_gmsh
def test_isolated_run_does_not_load_libgmsh_in_the_caller(tmp_path):
    stl = tmp_path / "sphere.stl"
    _sphere_stl(stl)
    code = (f"import sys; from topopt.tet import mesh_stl; m = mesh_stl({str(stl)!r}, size=3.0); "
            "print('LOADED' if 'gmsh' in sys.modules else 'CLEAN', m.facts['isolated'], m.facts['tets'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.split()[:2] == ["CLEAN", "True"], out.stdout


@needs_gmsh
@pytest.mark.slow
def test_hook_at_size_0p75_does_not_kill_the_process():
    """Netgen crashes (SIGSEGV) in gmsh on the hook at size 0.75 mm. The child dies, this process lives."""
    stl = recorded_hook_run() / "result" / "final.stl"
    if not stl.is_file():
        pytest.skip("the recorded hook STL is not here")
    mesh = mesh_stl(stl, size=0.75)
    f = mesh.facts
    assert abs(f["volume_ratio"] - 1.0) <= 1e-6 and f["inverted_fixed"] == 0
    assert f["below_0p05"] == 0 and f["quality_min"] >= 0.1
    if f["crashes"]:                                   # the known crash (gmsh 4.15.2, Linux x86_64)
        assert f["crashes"][0]["signal"] == "SIGSEGV" and f["optimize_used"] != "Netgen"
