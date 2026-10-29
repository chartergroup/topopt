"""B-013: setup failures end as one-line user errors (exit 1) and a real error status."""

import json
import platform
import sys

import pytest
import yaml

from topopt import cli, solver
from topopt.cli import main
from helpers import fake_shim
from test_optimize import make_case
from test_solver_choice import fake_platform  # noqa: F401  (the fixture that fakes the platform)


def edit(p, fn):
    d = yaml.safe_load(p.read_text())
    fn(d)
    p.write_text(yaml.safe_dump(d))
    return p


def one_line_error(capsys):
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and "Traceback" not in err
    assert err.startswith("error:")
    return err


def fake_mac(monkeypatch):
    """Pretend this is an Apple silicon Mac (the rule for `auto` depends on the platform)."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")


@pytest.fixture
def as_mac(monkeypatch):
    fake_mac(monkeypatch)


def no_dylib(monkeypatch):
    """Pretend the Cholesky library is missing on a Mac; pyamg stays as it is."""
    fake_mac(monkeypatch)
    real = cli.backend_availability

    def fake():
        b = real()
        b["cholesky"] = {"available": False, "note": "libaccel_chol.dylib not found"}
        b["amg"] = {"available": True, "note": "PyAMG-preconditioned CG"}
        return b
    monkeypatch.setattr(cli, "backend_availability", fake)


def test_huge_int_force(tmp_path, capsys, monkeypatch):
    no_dylib(monkeypatch)
    p = make_case(tmp_path, h=2.0)
    p.write_text(p.read_text().replace("-2.0", str(10 ** 400)))
    assert main(["run", str(p), "--out", str(tmp_path / "r")]) == 1
    err = one_line_error(capsys)
    assert "finite number" in err and "(file:" in err


def test_tiny_voxel_size_grid_guard(tmp_path, capsys):
    p = edit(make_case(tmp_path, h=2.0), lambda d: d["settings"].update(voxel_size=1.0e-18, solver="amg"))
    for cmd in (["check", str(p)], ["run", str(p), "--out", str(tmp_path / "r")]):
        assert main(cmd) == 1
        err = one_line_error(capsys)
        assert "max_grid_voxels" in err and "no voxel" not in err and "(file:" in err


def test_missing_backend_stops_run(tmp_path, capsys, monkeypatch, fake_platform):
    """An explicit `solver: cholesky` on a Mac without the library stops. (`auto` runs amg instead.)"""
    fake_platform("darwin", "arm64")
    no_dylib(monkeypatch)
    p = edit(make_case(tmp_path, h=2.0), lambda d: d["settings"].update(solver="cholesky"))
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out)]) == 1
    err = one_line_error(capsys)
    assert "settings.solver: amg" in err and "build.sh" in err and "(file:" in err
    st = json.loads((out / "status.json").read_text())
    assert st["state"] == "error" and "settings.solver: amg" in st["message"]


def test_check_shows_backend_problem(tmp_path, capsys, monkeypatch, fake_platform):
    fake_platform("darwin", "arm64")
    no_dylib(monkeypatch)
    p = edit(make_case(tmp_path, h=2.0), lambda d: d["settings"].update(solver="cholesky"))
    assert main(["check", str(p)]) == 0
    assert "warning: settings.solver is 'cholesky'" in capsys.readouterr().out


def test_explicit_amg_is_not_a_backend_problem(monkeypatch, fake_platform):
    fake_platform("darwin", "arm64")
    no_dylib(monkeypatch)
    assert cli.backend_problem("amg") is None
    assert "amg" in cli.backend_problem("cholesky")


def test_file_suffix_on_voxelize_and_loads_errors(tmp_path, capsys):
    # voxelize: a region with no voxel at this voxel size
    p = make_case(tmp_path, h=2.0)
    big = edit(p, lambda d: d["settings"].update(voxel_size=15.0))
    assert main(["check", str(big)]) == 1
    err = one_line_error(capsys)
    assert err.rstrip().endswith(f"(file: {p.resolve()})")
    assert err.count("(file:") == 1
    # loads: a load region that is not a mapped region name is caught in config; use build_forces directly
    from topopt.loads import UserError
    e = cli.with_file(UserError("load region 'x' has no node"), p)
    assert str(e).endswith(f"(file: {p.resolve()})")
    assert cli.with_file(e, "other.yaml") is e


def test_keyboard_interrupt_status(tmp_path, monkeypatch):
    no_dylib(monkeypatch)
    p = make_case(tmp_path, h=2.0)
    edit(p, lambda d: d["settings"].update(solver="amg"))
    import topopt.optimize as opt

    def boom(problem, out):
        out.step(7, "solve")
        raise KeyboardInterrupt
    monkeypatch.setattr(opt, "run", boom)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out)]) == 130
    st = json.loads((out / "status.json").read_text())
    assert st["state"] == "error" and st["iteration"] == 7 and st["step"] == "solve"


def test_error_status_has_real_step(tmp_path, monkeypatch):
    no_dylib(monkeypatch)
    p = edit(make_case(tmp_path, h=2.0), lambda d: d["settings"].update(solver="amg"))
    import topopt.optimize as opt

    def boom(problem, out):
        out.step(3, "assemble")
        raise RuntimeError("boom")
    monkeypatch.setattr(opt, "run", boom)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out)]) == 2
    st = json.loads((out / "status.json").read_text())
    assert (st["state"], st["iteration"], st["step"]) == ("error", 3, "assemble")
    assert "internal error: boom" in st["message"]


def test_check_bad_yaml_gives_a_one_line_error(tmp_path, capsys):
    p = edit(make_case(tmp_path, h=2.0), lambda d: d["settings"].update(bogus=1))
    assert main(["check", str(p)]) == 1
    err = one_line_error(capsys)
    assert "bogus" in err


def fake_backends(chol, amg):
    return {"cholesky": {"available": chol, "note": "libaccel_chol.dylib not found"},
            "amg": {"available": amg, "note": "pyamg is not installed"}}


def test_backend_problem_table(as_mac, monkeypatch, tmp_path):
    fake_shim(monkeypatch, tmp_path, "ok")
    assert cli.backend_problem("auto", fake_backends(True, False)) is None
    assert cli.backend_problem("cholesky", fake_backends(True, True)) is None
    assert cli.backend_problem("amg", fake_backends(False, True)) is None
    msg = cli.backend_problem("cholesky", fake_backends(False, True))
    assert "settings.solver: amg" in msg and "build.sh" in msg and "\n" not in msg
    assert "install pyamg" not in msg
    # Without the shim, `auto` means amg: no problem while pyamg is there.
    fake_shim(monkeypatch, tmp_path / "no-shim", "missing")
    assert cli.backend_problem("auto", fake_backends(False, True)) is None


def test_backend_problem_pyamg_missing(as_mac, monkeypatch, tmp_path):
    msg = cli.backend_problem("amg", fake_backends(True, False))
    assert "install pyamg" in msg and "settings.solver: cholesky" in msg
    both = cli.backend_problem("cholesky", fake_backends(False, False))
    assert "install pyamg" in both and "build.sh" in both
    fake_shim(monkeypatch, tmp_path, "missing")     # `auto` means amg here, and pyamg is missing
    assert "install pyamg" in cli.backend_problem("auto", fake_backends(False, False))


def test_backend_problem_unknown_names_do_not_raise(as_mac):
    assert cli.backend_problem("mkl", fake_backends(False, False)) is None
    assert "Linux x86_64" in cli.backend_problem("pardiso", fake_backends(False, False))
    assert cli.backend_problem("auto", {}) is None


def test_backend_problem_gives_the_platform_rule_first(fake_platform):
    """The platform rule (`resolve_solver`) comes before the backend check, and the text is its text."""
    fake_platform("linux", "x86_64", pypardiso=None)
    msg = cli.backend_problem("cholesky", fake_backends(True, True))
    assert "Apple Accelerate" in msg and "pardiso" in msg and "amg" in msg and "\n" not in msg
    msg = cli.backend_problem("pardiso", fake_backends(False, True))
    assert "pypardiso" in msg and "uv pip install" in msg and "\n" not in msg
    # auto on Linux without pypardiso is amg: only pyamg matters
    assert cli.backend_problem("auto", fake_backends(False, True)) is None
    assert "install pyamg" in cli.backend_problem("auto", fake_backends(False, False))


def test_check_with_a_platform_error_prints_the_facts_and_exits_1(tmp_path, capsys, fake_platform):
    """The platform error ends `check` with exit 1 and the file, and it is not printed twice."""
    fake_platform("linux", "x86_64", pypardiso=None)
    p = edit(make_case(tmp_path, h=2.0), lambda d: d["settings"].update(solver="cholesky"))
    assert main(["check", str(p)]) == 1
    cap = capsys.readouterr()
    assert "free dofs:" in cap.out and "warning: settings.solver is" not in cap.out
    assert cap.err.startswith("error:") and "(file:" in cap.err and len(cap.err.strip().splitlines()) == 1


def test_run_passes_check_when_backend_present(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "backend_availability", lambda: fake_backends(True, True))
    import topopt.optimize as opt
    reached = []
    monkeypatch.setattr(opt, "run", lambda problem, out: reached.append(1) or None)
    monkeypatch.setattr(cli, "finish", lambda out, d: None)
    monkeypatch.setattr(cli, "run_refine", lambda *a: None)
    assert main(["run", str(make_case(tmp_path, h=2.0)), "--out", str(tmp_path / "r"), "--quiet"]) == 0
    assert reached


def test_loads_error_through_run_has_suffix(tmp_path, capsys, monkeypatch):
    import topopt.loads as loads
    from topopt.config import UserError

    def bad(*a, **k):
        raise UserError("load region 'tip' has no node")
    monkeypatch.setattr(loads, "build_forces", bad)
    import topopt.problem as problem
    monkeypatch.setattr(problem, "build_forces", bad, raising=False)
    p = make_case(tmp_path, h=2.0)
    assert main(["run", str(p), "--out", str(tmp_path / "r")]) == 1
    err = one_line_error(capsys)
    assert err.rstrip().endswith(f"(file: {p.resolve()})")


def test_huge_force_creates_no_run_folder(tmp_path, capsys):
    p = make_case(tmp_path, h=2.0)
    p.write_text(p.read_text().replace("-2.0", str(10 ** 400)))
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out)]) == 1
    assert not out.exists()


def test_grid_guard_keeps_retry_substrings(tmp_path):
    from topopt.config import UserError
    from topopt.voxelize import grid_from_bounds
    import numpy as np
    with pytest.raises(UserError) as e:
        grid_from_bounds(np.zeros(3), np.full(3, 100.0), 0.1, 1000)
    msg = str(e.value)
    assert "limit is" in msg and "Try voxel_size" in msg and "max_grid_voxels" in msg


def fake_linux(monkeypatch, pypardiso=True):
    import types
    from topopt import solver_choice
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    def imp():
        if not pypardiso:
            raise ImportError("no pypardiso")
        return types.ModuleType("pypardiso")
    monkeypatch.setattr(solver_choice, "import_pypardiso", imp)


def linux_backends(pardiso, amg):
    return {"cholesky": {"available": False, "note": "Apple silicon only"},
            "pardiso": {"available": pardiso, "note": "pypardiso is not installed"},
            "amg": {"available": amg, "note": "pyamg is not installed"}}


def test_backend_problem_linux_auto_pardiso_available(monkeypatch):
    fake_linux(monkeypatch, pypardiso=True)
    assert cli.backend_problem("auto", linux_backends(True, True)) is None


def test_backend_problem_linux_auto_amg_with_pyamg_missing(monkeypatch):
    fake_linux(monkeypatch, pypardiso=False)
    msg = cli.backend_problem("auto", linux_backends(False, False))
    assert "amg backend cannot run" in msg and "install pyamg" in msg and "\n" not in msg
    assert cli.backend_problem("auto", linux_backends(False, True)) is None


def test_backend_problem_linux_explicit_pardiso(monkeypatch):
    fake_linux(monkeypatch, pypardiso=False)
    msg = cli.backend_problem("pardiso", linux_backends(False, True))
    assert "pypardiso" in msg and "uv pip install" in msg
    fake_linux(monkeypatch, pypardiso=True)
    assert cli.backend_problem("pardiso", linux_backends(True, True)) is None
    assert "Apple Accelerate" in cli.backend_problem("cholesky", linux_backends(True, True))


def test_a_part_spec_for_build_regions_is_named_as_one_when_check_reads_it(capsys):
    """Q1-10: `regions.design` is a list of shapes in a part spec. The error says so and names build_regions."""
    from pathlib import Path
    spec = Path(__file__).resolve().parent.parent / "examples" / "ladle" / "ladle_spec.yaml"
    assert main(["check", str(spec)]) == 1
    err = one_line_error(capsys)
    assert "'regions.design' is a list of shapes" in err and "part spec for build_regions" in err
    assert "python -m topopt.regions_builder SPEC.yaml --out DIR" in err
    assert spec.read_text().startswith("# This file is a PART SPEC for build_regions")      # the example says it in its first line
    assert "must be a non-empty string" not in err


def test_a_part_spec_with_a_spec_only_top_level_key_is_named_as_one(tmp_path, capsys):
    """Q1 R1 n3: `fixed_axes` is a top-level key of the part spec only. The part-spec text comes before the unknown-key text."""
    from pathlib import Path
    spec = Path(__file__).resolve().parent.parent / "examples" / "ladle" / "ladle_spec.yaml"
    p = tmp_path / "spec.yaml"
    p.write_text(spec.read_text() + "\nfixed_axes: {grip: [y]}\n")
    assert main(["check", str(p)]) == 1
    err = one_line_error(capsys)
    assert "'regions.design' is a list of shapes" in err and "part spec for build_regions" in err and "unknown key" not in err
    q = tmp_path / "plain.yaml"                                               # a problem YAML with the same stray key keeps the key error
    q.write_text("name: x\nregions: {design: a.stl, fixed: {f: b.stl}, load: {l: c.stl}}\nload_cases: []\nsettings: {}\nfixed_axes: {f: [y]}\n")
    assert main(["check", str(q)]) == 1
    assert "unknown key 'fixed_axes' in 'top level'" in one_line_error(capsys)


def test_a_design_list_that_is_not_a_list_of_shapes_keeps_the_plain_error(tmp_path, capsys):
    p = tmp_path / "x.yaml"
    p.write_text("name: x\nregions: {design: [a.stl], fixed: {f: [b.stl]}, load: {l: c.stl}}\nload_cases: []\nsettings: {}\n")
    assert main(["check", str(p)]) == 1
    err = one_line_error(capsys)
    assert "regions.design" in err and "part spec" not in err
