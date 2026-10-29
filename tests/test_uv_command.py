"""Tests for `topopt.uv_command` (REL-9): the uv of a repair command.

The launcher exports TOPOPT_UV=<full path> when it uses a uv that it downloaded. Every repair command that topopt prints then names that
file, quoted, and not a bare `uv` (which is not on PATH). With no TOPOPT_UV the texts are the old ones.
"""
import os
import re
import stat
import sys
from pathlib import Path

import pytest

from topopt import doctor, environment, solver_choice, step
from topopt.gpu_mg import device
from topopt.tet import gmsh_probe
from topopt.uv_command import uv_command

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    """An executable file with a space in its path, named by TOPOPT_UV. Returns the shell-quoted form."""
    folder = tmp_path / "my env" / "uv" / "0.0.0"
    folder.mkdir(parents=True)
    path = folder / "uv"
    path.write_text("#!/bin/sh\necho uv 0.0.0\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("TOPOPT_UV", str(path))
    return f"'{path}'"


@pytest.fixture(autouse=True)
def no_topopt_uv(monkeypatch):
    monkeypatch.delenv("TOPOPT_UV", raising=False)


def test_without_the_variable_the_command_is_uv():
    assert uv_command() == "uv"


@pytest.mark.parametrize("value", ["", "uv", "relative/uv", "/no/such/file/uv"])
def test_a_value_that_is_no_executable_full_path_gives_uv(monkeypatch, value):
    monkeypatch.setenv("TOPOPT_UV", value)
    assert uv_command() == "uv"


def test_a_folder_or_a_file_that_does_not_run_gives_uv(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPOPT_UV", str(tmp_path))
    assert uv_command() == "uv"
    plain = tmp_path / "uv"
    plain.write_text("x")
    plain.chmod(0o644)
    monkeypatch.setenv("TOPOPT_UV", str(plain))
    if os.geteuid() != 0:           # root can run any file that has an execute bit; without the bit os.access says no for everyone
        assert uv_command() == "uv"


def test_a_full_path_to_an_executable_is_quoted(fake_uv):
    assert uv_command() == fake_uv
    assert " " in fake_uv and fake_uv.startswith("'") and fake_uv.endswith("'")


def test_a_plain_path_is_not_quoted_when_it_needs_no_quote(monkeypatch, tmp_path):
    path = tmp_path / "uv"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    monkeypatch.setenv("TOPOPT_UV", str(path))
    assert uv_command() == str(path)


def test_the_old_texts_stay_without_the_variable():
    py = re.escape(sys.executable)
    assert re.fullmatch(rf"uv pip install --python .*{py}.* pyamg.*", environment._pyamg_offer()["command"])
    assert solver_choice.install_command().startswith("uv pip install --python ")
    assert device.install_command().startswith("uv pip install --python ")
    assert gmsh_probe.install_command().startswith("uv pip install --python ")
    assert gmsh_probe.nox_install_command().startswith("uv pip install --python ")
    assert "    uv pip install -e " in step.install_text() and "    uv pip install \"" in step.install_text()


def test_every_repair_command_names_the_downloaded_uv(fake_uv):
    assert environment._pyamg_offer()["command"].startswith(f"{fake_uv} pip install --python ")
    assert solver_choice.install_command().startswith(f"{fake_uv} pip install --python ")
    assert device.install_command().startswith(f"{fake_uv} pip install --python ")
    assert gmsh_probe.install_command().startswith(f"{fake_uv} pip install --python ")
    assert gmsh_probe.nox_install_command().startswith(f"{fake_uv} pip install --python ")
    text = step.install_text()
    assert f"    {fake_uv} pip install -e " in text and f"    {fake_uv} pip install \"" in text
    assert "    uv pip" not in text


def test_the_verify_texts_name_the_downloaded_uv(fake_uv):
    from topopt.verify import factor
    assert factor.verify_install_command().startswith(f"{fake_uv} pip install --python ")
    note = factor.cholespy_note(("load_failed", "boom"))
    assert f"Reinstall it with: {fake_uv} pip install --reinstall " in note
    assert factor.cholespy_note(("missing", "x")).count(fake_uv) == 1


def test_the_rosetta_steps_check_the_downloaded_uv_by_its_path(fake_uv, monkeypatch):
    monkeypatch.setattr(solver_choice, "mac_python_kind", lambda: "rosetta")
    text = solver_choice.platform_hint()
    assert f"check uv with: file {fake_uv} (it must say arm64" in text and "command -v uv" not in text
    monkeypatch.delenv("TOPOPT_UV")
    assert 'check uv with: file "$(command -v uv)" (it must say arm64' in solver_choice.platform_hint()


def base_data(**kw):
    data = {"checkout": {"code_folder": "/code dir", "git": True, "has_pyproject": True, "behind": 3, "default_branch_ref": "origin/main",
                         "update_command": "git pull", "fetch_command": "git fetch"},
            "imports": [], "offers": [], "known": {"environments": [{"is_this": True, "role": ".venv", "dir": "/code dir/.venv"}]}}
    data.update(kw)
    return data


def test_doctor_next_commands_name_the_downloaded_uv(fake_uv):
    cut = {"name": "scipy", "use": "core", "status": "cut", "detail": "x"}
    cmds = doctor.next_commands(base_data(imports=[cut]))
    commands = [c["command"] for c in cmds if c.get("command")]
    assert commands[0].startswith(f"{fake_uv} pip install --reinstall --python ")             # the repair of the cut package
    assert any(c.startswith(f"{fake_uv} pip install --python ") for c in commands[1:])         # the refresh after the update
    assert not any(re.match(r"uv ", c) for c in commands)
    wheel = doctor.next_commands(base_data(imports=[cut], checkout={"has_pyproject": False, "git": False}))
    assert f"`{fake_uv} pip install --reinstall --python " in wheel[0]["why"]


def test_no_python_text_of_topopt_holds_a_bare_uv_command():
    """A new repair command must use `uv_command()`. A bare `uv pip`, `uv venv`, `uv run` or `uv sync` in code is a defect (REL-9)."""
    found = []
    for folder in ("topopt", "topopt_mcp"):
        for path in sorted((REPO / folder).rglob("*.py")):
            if path.name == "uv_command.py":
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if re.search(r"(?<![\w{}`.-])uv (pip|venv|run|sync)\b", line):
                    found.append(f"{path.relative_to(REPO)}:{n}: {line.strip()[:120]}")
    assert not found, "\n".join(found)
