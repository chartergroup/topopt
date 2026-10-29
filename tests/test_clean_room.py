"""Tests for scripts/dev/clean-room.sh (issue #337): the helper that puts a computer into the state of a new user.

Every test runs the script with a temporary HOME and a scrubbed environment. Nothing here touches the real home.
The tests run on Linux and macOS, and none is skipped.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "dev" / "clean-room.sh"
SYSTEM_PATH = "/usr/bin:/bin"

CURSOR = {"mcpServers": {"alpha": {"command": "a", "args": ["x,y", "}"]}, "topopt": {"command": "sh", "args": ["{", 'q"}']}}}
CODEX = (
    'model = "m"\n\n[mcp_servers.alpha]\ncommand = "a"\n\n[mcp_servers.topopt]\ncommand = "sh"\n'
    '[mcp_servers.topopt.env]\nA = "1"\n\n[profiles.p]\nk = 1\n'
)


def snapshot(root: Path, skip=()) -> dict:
    """Relative path -> (kind, content or link target) for the whole tree. Folders are listed too."""
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if any(rel == s or rel.startswith(s + "/") for s in skip):
            continue
        if path.is_symlink():
            out[rel] = ("link", os.readlink(path))
        elif path.is_dir():
            out[rel] = ("dir", None)
        else:
            out[rel] = ("file", path.read_bytes())
    return out


@pytest.fixture
def box(tmp_path):
    """A fake home with a used state: uv, caches, a uv Python, topopt state and client configs with another server."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "uv").write_text("#!/bin/sh\necho fake uv\n")
    (home / ".local" / "bin" / "uv").chmod(0o755)
    (home / ".local" / "bin" / "uvx").write_text("#!/bin/sh\n")
    (home / ".local" / "bin" / "keep-me").write_text("not uv\n")
    py = home / ".local" / "share" / "uv" / "python" / "cpython-3.12-fake" / "bin"
    py.mkdir(parents=True)
    (py / "python3.12").write_text("python\n")
    (home / ".local" / "bin" / "python3.12").symlink_to(py / "python3.12")
    (home / ".local" / "share" / "uv" / "tools" / "ruff").mkdir(parents=True)
    (home / ".cache" / "uv" / "wheels").mkdir(parents=True)
    (home / ".cache" / "uv" / "wheels" / "a.whl").write_bytes(b"\x00\x01wheel")
    (home / ".cache" / "other").mkdir()
    (home / ".cache" / "other" / "keep").write_text("keep\n")
    (home / ".local" / "state" / "topopt" / "servers").mkdir(parents=True)
    (home / ".local" / "state" / "topopt" / "servers" / "server-1.json").write_text("{}\n")
    plug = home / ".claude" / "plugins" / "data" / "topopt-topopt" / "venv"
    plug.mkdir(parents=True)
    (plug / "marker").write_text("env\n")
    (home / ".claude" / "settings.json").write_text('{"keep": true}\n')
    (home / ".cursor").mkdir()
    (home / ".cursor" / "mcp.json").write_text(json.dumps(CURSOR, indent=2) + "\n")
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text(CODEX)
    (home / ".gemini").mkdir()
    (home / ".gemini" / "settings.json").write_text('{"theme": "x"}\n')     # no topopt entry
    (tmp_path / "tmp").mkdir()
    return home


def run(home: Path, *args, path_extra="", env_extra=None, check=True, path=None):
    """Run the script with HOME set to `home` and nothing else from the real environment."""
    env = {
        "HOME": str(home),
        "PATH": (path_extra + ":" if path_extra else "") + str(home / ".local" / "bin") + ":" + SYSTEM_PATH,
        "TMPDIR": str(home.parent / "tmp"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
        "TOPOPT_STATE_DIR": str(home / ".local" / "state" / "topopt"),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "CODEX_HOME": str(home / ".codex"),
        "UV_CACHE_DIR": str(home / ".cache" / "uv"),
        "UV_PYTHON_INSTALL_DIR": str(home / ".local" / "share" / "uv" / "python"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_CONFIG_HOME": str(home / ".config"),
    }
    if path is not None:
        env["PATH"] = path
    env.update(env_extra or {})
    done = subprocess.run(["sh", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=120)
    if check:
        assert done.returncode == 0, done.stdout + done.stderr
    return done


def backup_of(home: Path) -> Path:
    folders = sorted((home / ".topopt-clean-room").iterdir())
    assert len(folders) == 1
    return folders[0]


def test_status_lists_the_items_and_exits_with_1(box):
    done = run(box, "status", check=False)
    assert done.returncode == 1
    for needle in (".local/bin/uv", "uv-managed Pythons", "uv cache", "registry of topopt", "plugin data", ".cursor/mcp.json",
                   ".codex/config.toml", "link to a uv-managed Python", "uv tools"):
        assert needle in done.stdout, needle
    assert ".gemini/settings.json (no topopt entry)" in done.stdout


def test_status_changes_nothing(box):
    before = snapshot(box)
    run(box, "status", check=False)
    assert snapshot(box) == before


def test_dry_run_changes_nothing(box):
    before = snapshot(box)
    done = run(box, "hide")
    assert "dry run" in done.stdout and "would move" in done.stdout
    assert snapshot(box) == before
    assert not (box / ".topopt-clean-room").exists()
    done = run(box, "restore", check=False)
    assert snapshot(box) == before          # a restore dry run with no backup changes nothing either


def test_hide_then_restore_gives_the_same_files_byte_for_byte(box):
    before = snapshot(box)
    run(box, "hide", "--apply")
    hidden = snapshot(box, skip=(".topopt-clean-room",))
    assert not (box / ".local" / "bin" / "uv").exists()
    assert not (box / ".cache" / "uv").exists()
    assert not (box / ".local" / "share" / "uv" / "python").exists()
    assert not (box / ".local" / "bin" / "python3.12").is_symlink()
    assert not (box / ".local" / "state" / "topopt").exists()
    assert not (box / ".claude" / "plugins" / "data" / "topopt-topopt").exists()
    # What is not a new-user item stays.
    assert (box / ".local" / "bin" / "keep-me").read_text() == "not uv\n"
    assert (box / ".cache" / "other" / "keep").read_text() == "keep\n"
    assert (box / ".claude" / "settings.json").read_text() == '{"keep": true}\n'
    assert (box / ".local" / "share" / "uv" / "tools" / "ruff").is_dir()      # reported only
    assert len(hidden) < len(before)
    clean = run(box, "status", check=False)
    assert clean.returncode == 0 and "clean: nothing" in clean.stdout
    run(box, "restore", "--apply")
    assert snapshot(box, skip=(".topopt-clean-room",)) == {k: v for k, v in before.items() if not k.startswith(".topopt-clean-room")}


def test_configs_lose_only_the_topopt_entry_and_come_back_exactly(box):
    original_cursor = (box / ".cursor" / "mcp.json").read_bytes()
    original_codex = (box / ".codex" / "config.toml").read_bytes()
    run(box, "hide", "--apply")
    cursor = json.loads((box / ".cursor" / "mcp.json").read_text())
    assert cursor == {"mcpServers": {"alpha": CURSOR["mcpServers"]["alpha"]}}
    codex = (box / ".codex" / "config.toml").read_text()
    assert "topopt" not in codex
    assert '[mcp_servers.alpha]' in codex and '[profiles.p]' in codex and 'model = "m"' in codex
    assert (box / ".gemini" / "settings.json").read_text() == '{"theme": "x"}\n'
    # The whole original is in the backup folder.
    manifest = (backup_of(box) / "manifest.tsv").read_text()
    assert "edited-config" in manifest
    run(box, "restore", "--apply")
    assert (box / ".cursor" / "mcp.json").read_bytes() == original_cursor
    assert (box / ".codex" / "config.toml").read_bytes() == original_codex


@pytest.mark.parametrize("text, check", [
    ('{"mcpServers":{"topopt":{"command":"sh"},"b":{"command":"x"}}}', lambda d: d == {"mcpServers": {"b": {"command": "x"}}}),
    ('{"mcpServers":{"b":{"command":"x"},"topopt":{"command":"sh"}}}', lambda d: d == {"mcpServers": {"b": {"command": "x"}}}),
    ('{"mcpServers":{"topopt":{"command":"sh"}}, "other": 1}', lambda d: d == {"mcpServers": {}, "other": 1}),
    ('{\n  "topopt": 1,\n  "mcpServers": {\n    "topopt": {\n      "command": "sh"\n    }\n  }\n}\n', lambda d: d == {"topopt": 1, "mcpServers": {}}),
    ('{"servers": {"topopt": {"type": "stdio", "args": ["// not a comment", "/* x */"]}, "b": {}}}', lambda d: d == {"servers": {"b": {}}}),
])
def test_json_edit_variants(box, text, check):
    (box / ".cursor" / "mcp.json").write_text(text)
    run(box, "hide", "--apply")
    assert check(json.loads((box / ".cursor" / "mcp.json").read_text()))


def test_jsonc_comments_are_kept(box):
    text = '{\n  // my servers\n  "servers": {\n    /* first */ "b": {"command": "x"},\n    "topopt": {"command": "sh"} // ours\n  }\n}\n'
    (box / ".cursor" / "mcp.json").write_text(text)
    run(box, "hide", "--apply")
    new = (box / ".cursor" / "mcp.json").read_text()
    assert "// my servers" in new and "/* first */" in new and '"b"' in new and "topopt" not in new.replace("// ours", "")


def test_a_config_without_a_topopt_entry_is_not_touched(box):
    before = (box / ".gemini" / "settings.json").stat()
    run(box, "hide", "--apply")
    assert (box / ".gemini" / "settings.json").stat().st_mtime_ns == before.st_mtime_ns
    assert "settings.json" not in (backup_of(box) / "manifest.tsv").read_text()


def test_shell_leaves_the_home_unchanged_and_has_no_uv(box, tmp_path):
    before = snapshot(box)
    target = tmp_path / "newuser"
    done = run(box, "shell", "--dir", str(target), "--", "sh", "-c",
               'echo "home=$HOME"; echo "xdg=$XDG_CONFIG_HOME"; echo "uv=$(command -v uv)"; echo "uvx=$(command -v uvx)"; '
               'echo "state=$TOPOPT_STATE_DIR"; echo "secret=${SECRET_KEY:-none}"',
               env_extra={"SECRET_KEY": "abc"})
    assert snapshot(box) == before
    assert f"home={target}/home" in done.stdout
    assert f"xdg={target}/home/.config" in done.stdout
    assert "uv=\n" in done.stdout and "uvx=\n" in done.stdout
    assert f"state={target}/home/.local/state/topopt" in done.stdout
    assert "secret=none" in done.stdout                      # env -i: no variable of yours goes in
    assert str(box / ".local" / "bin") in done.stdout        # the folder is named as left out
    assert not (target / "home" / ".codex").exists()         # no auth file without the flag


def test_shell_passes_a_named_variable_and_one_auth_file_only_on_request(box, tmp_path):
    (box / ".codex" / "auth.json").write_text('{"token": "t"}\n')
    target = tmp_path / "newuser"
    done = run(box, "shell", "--dir", str(target), "--auth-file", ".codex/auth.json", "--env", "SECRET_KEY", "--",
               "sh", "-c", 'echo "secret=$SECRET_KEY"', env_extra={"SECRET_KEY": "abc"})
    assert "secret=abc" in done.stdout
    assert (target / "home" / ".codex" / "auth.json").read_text() == '{"token": "t"}\n'
    assert not (target / "home" / ".codex" / "config.toml").exists()       # only the one file
    bad = run(box, "shell", "--dir", str(tmp_path / "x"), "--auth-file", "../etc/passwd", check=False)
    assert bad.returncode != 0 and not (tmp_path / "x").exists()


def test_shell_refuses_a_folder_that_is_not_empty(box, tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "f").write_text("x")
    done = run(box, "shell", "--dir", str(tmp_path / "full"), "--", "true", check=False)
    assert done.returncode != 0 and "not empty" in done.stderr


def test_a_uv_outside_home_is_only_reported(box, tmp_path):
    outside = tmp_path / "opt" / "bin"
    outside.mkdir(parents=True)
    (outside / "uv").write_text("#!/bin/sh\n")
    (outside / "uv").chmod(0o755)
    done = run(box, "hide", "--apply", path_extra=str(outside))
    assert f"{outside}/uv" in done.stdout and "report-only" in done.stdout
    assert (outside / "uv").is_file()
    assert "opt/bin" not in (backup_of(box) / "manifest.tsv").read_text()
    # The shell leaves that folder out of PATH, and uv is not found there.
    shell = run(box, "shell", "--dir", str(tmp_path / "nu"), "--", "sh", "-c", 'echo "uv=$(command -v uv)"', path_extra=str(outside))
    assert "uv=\n" in shell.stdout and str(outside) in shell.stdout


def test_a_state_folder_outside_home_is_only_reported(box, tmp_path):
    state = tmp_path / "elsewhere"
    (state / "servers").mkdir(parents=True)
    done = run(box, "hide", "--apply", env_extra={"TOPOPT_STATE_DIR": str(state)})
    assert (state / "servers").is_dir()
    assert "outside your home" in done.stdout


def test_the_gpu_mg_state_folder_in_home_is_moved_when_topopt_state_dir_is_not_set(box):
    """`gpu_mg.check.state_dir()` is `$TOPOPT_STATE_DIR`, else `~/.topopt` (the calibration of `topopt gpu-check`). An empty value counts as not set."""
    (box / ".topopt").mkdir()
    (box / ".topopt" / "gpu-constants.json").write_text('{"schema": 1, "calibrations": []}\n')
    unset = {"TOPOPT_STATE_DIR": ""}
    status = run(box, "status", check=False, env_extra=unset)
    assert "per-user state of gpu-mg" in status.stdout
    before = snapshot(box)
    run(box, "hide", "--apply", env_extra=unset)
    assert not (box / ".topopt").exists()
    assert not (box / ".local" / "state" / "topopt").exists()       # the registry folder of the XDG rule moves too
    run(box, "restore", "--apply", env_extra=unset)
    assert snapshot(box, skip=(".topopt-clean-room",)) == {k: v for k, v in before.items() if not k.startswith(".topopt-clean-room")}


def test_the_home_topopt_folder_stays_when_topopt_state_dir_is_set(box):
    """With the variable set, topopt keeps its gpu-mg state in that folder, so `~/.topopt` is not a topopt item."""
    (box / ".topopt").mkdir()
    (box / ".topopt" / "gpu-constants.json").write_text("{}\n")
    status = run(box, "status", check=False)
    assert "per-user state of gpu-mg" not in status.stdout
    run(box, "hide", "--apply")
    assert (box / ".topopt" / "gpu-constants.json").read_text() == "{}\n"



def test_the_kernel_caches_of_pyopencl_are_moved_and_restored(box):
    """pyopencl keeps compiled kernels in `$XDG_CACHE_HOME/pyopencl` and its invokers in `$XDG_CACHE_HOME/pytools` (else `~/.cache` on Linux)."""
    kernels = box / ".cache" / "pyopencl" / "pyopencl-compiler-cache-v2-py3.12.0"
    kernels.mkdir(parents=True)
    (kernels / "abc").write_bytes(b"\x00kernel")
    (box / ".cache" / "pytools").mkdir()
    (box / ".cache" / "pytools" / "pdict-v5-pyopencl-invoker").write_bytes(b"\x01invoker")
    status = run(box, "status", check=False)
    assert "kernel cache of pyopencl" in status.stdout and "cache of pytools" in status.stdout
    before = snapshot(box)
    run(box, "hide", "--apply")
    assert not (box / ".cache" / "pyopencl").exists() and not (box / ".cache" / "pytools").exists()
    assert (box / ".cache" / "other" / "keep").read_text() == "keep\n"
    run(box, "restore", "--apply")
    assert snapshot(box, skip=(".topopt-clean-room",)) == {k: v for k, v in before.items() if not k.startswith(".topopt-clean-room")}


def test_a_state_folder_given_as_tilde_text_is_the_folder_in_home(box):
    """A client config can pass `~/...` as plain text; topopt reads it as a folder in the home, and so does the helper."""
    (box / "x").mkdir()
    (box / "x" / "gpu-constants.json").write_text("{}\n")
    done = run(box, "hide", "--apply", env_extra={"TOPOPT_STATE_DIR": "~/x"})
    assert not (box / "x").exists(), done.stdout
    run(box, "restore", "--apply", env_extra={"TOPOPT_STATE_DIR": "~/x"})
    assert (box / "x" / "gpu-constants.json").read_text() == "{}\n"

def make_brew(tmp_path: Path, prefix: Path) -> Path:
    """A fake brew that only logs its arguments. It keeps the link prefix/bin/uv in step with unlink and link."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "uv").write_text("brew uv\n")
    (prefix / "Cellar").mkdir()
    (prefix / "Cellar" / "uv").write_text("brew uv\n")
    log = tmp_path / "brew.log"
    script = f"""#!/bin/sh
echo "$*" >> {log}
case "$1" in
  list) echo "uv 9.9.9" ;;
  --prefix) echo {prefix} ;;
  unlink) rm -f {prefix}/bin/uv ;;
  link) cp {prefix}/Cellar/uv {prefix}/bin/uv ;;
esac
"""
    (bindir / "brew").write_text(script)
    (bindir / "brew").chmod(0o755)
    return bindir


def test_homebrew_uv_is_unlinked_and_linked_again(box, tmp_path):
    prefix = tmp_path / "brewprefix"
    bindir = make_brew(tmp_path, prefix)
    # Without the home uv, the Homebrew uv is the one on PATH.
    run(box, "hide", "--apply", path_extra=f"{bindir}:{prefix}/bin")
    assert not (prefix / "bin" / "uv").exists()
    log = (tmp_path / "brew.log").read_text().splitlines()
    assert "unlink uv" in log and "link uv" not in log
    run(box, "restore", "--apply", path_extra=f"{bindir}")
    assert (prefix / "bin" / "uv").exists()
    assert (tmp_path / "brew.log").read_text().splitlines().count("link uv") == 1
    assert "Cellar" not in (box / ".topopt-clean-room").as_posix()
    assert (prefix / "Cellar" / "uv").exists()       # never uninstalled


def test_the_brew_dry_run_does_not_call_unlink(box, tmp_path):
    prefix = tmp_path / "brewprefix"
    bindir = make_brew(tmp_path, prefix)
    done = run(box, "hide", path_extra=f"{bindir}:{prefix}/bin")
    assert "brew unlink uv" in done.stdout
    assert "unlink" not in (tmp_path / "brew.log").read_text()


def test_restore_does_not_overwrite_a_new_file(box):
    run(box, "hide", "--apply")
    (box / ".cache" / "uv").mkdir()
    (box / ".cache" / "uv" / "new").write_text("new state of the test\n")
    (box / ".local" / "bin" / "uv").write_text("a newer uv\n")
    done = run(box, "restore", "--apply", check=False)
    assert done.returncode == 1 and "INCOMPLETE" in done.stdout.splitlines()[-1]
    assert "SKIP" in done.stdout and "not overwritten" in done.stdout
    assert (box / ".cache" / "uv" / "new").read_text() == "new state of the test\n"
    assert not (box / ".cache" / "uv" / "wheels").exists()
    assert (box / ".local" / "bin" / "uv").read_text() == "a newer uv\n"
    # The rest came back, and the skipped items stay in the backup folder.
    assert (box / ".local" / "share" / "uv" / "python" / "cpython-3.12-fake" / "bin" / "python3.12").is_file()
    assert (backup_of(box) / "files").is_dir()
    assert any((backup_of(box) / "files").iterdir())


def test_restore_leaves_a_config_that_changed_since_hide(box):
    run(box, "hide", "--apply")
    cfg = box / ".cursor" / "mcp.json"
    cfg.write_text('{"mcpServers": {"topopt": {"command": "again"}}}\n')     # the client wrote it again
    done = run(box, "restore", "--apply", check=False)
    assert done.returncode == 1 and "changed since hide" in done.stdout
    assert cfg.read_text() == '{"mcpServers": {"topopt": {"command": "again"}}}\n'
    assert (box / ".codex" / "config.toml").read_text() == CODEX         # the other config came back
    original = [p for p in (backup_of(box) / "files").iterdir() if p.is_file() and "alpha" in p.read_text() and "mcpServers" in p.read_text()]
    assert original, "the original config stays in the backup folder"


def test_a_second_restore_is_harmless(box):
    run(box, "hide", "--apply")
    run(box, "restore", "--apply")
    after = snapshot(box, skip=(".topopt-clean-room",))
    again = run(box, "restore", "--apply")
    assert "restore complete" in again.stdout and "SKIP" not in again.stdout and "FAIL" not in again.stdout
    assert snapshot(box, skip=(".topopt-clean-room",)) == after


def test_a_backup_folder_on_the_wrong_place_is_refused(box, tmp_path):
    before = snapshot(box)
    done = run(box, "hide", "--apply", "--backup-dir", str(tmp_path / "outside-backup"), check=False)
    assert done.returncode != 0
    assert snapshot(box) == before


def test_a_home_of_slash_is_refused():
    done = subprocess.run(["sh", str(SCRIPT), "status"], env={"HOME": "/", "PATH": SYSTEM_PATH}, capture_output=True, text=True)
    assert done.returncode == 1 and "HOME is not set or is /" in done.stderr
    done = subprocess.run(["sh", str(SCRIPT), "status"], env={"HOME": "//", "PATH": SYSTEM_PATH}, capture_output=True, text=True)
    assert done.returncode == 1 and "HOME is not set or is /" in done.stderr


def test_the_script_has_no_bash_features():
    text = SCRIPT.read_text()
    assert text.startswith("#!/bin/sh\n")
    for banned in ("\nlocal ", "declare ", "<<<", "${BASH", "echo -e", "readarray", "source "):
        assert banned not in text, banned
    assert " ]] " not in text and " ]];" not in text
    assert "function " not in text.replace("function isws", "").replace("function skipws", "").replace("function matchend", "")


# --- the Docker helper: tested with a fake docker command that only logs (the agent host has no daemon) ---------------

DOCKER_SCRIPT = ROOT / "scripts" / "dev" / "clean-room-docker.sh"
DOCKERFILE = ROOT / "docker" / "clean-room" / "Dockerfile"


def fake_docker(tmp_path: Path, daemon=True, image=False) -> tuple[Path, Path]:
    bindir = tmp_path / "dockerbin"
    bindir.mkdir()
    log = tmp_path / "docker.log"
    (bindir / "docker").write_text(f"""#!/bin/sh
echo "$*" >> {log}
[ "$1" != build ] || env | grep -i '^[a-z]*_proxy=' | sort > {tmp_path / "docker-build.env"}
case "$1" in
  info) {'exit 0' if daemon else 'exit 1'} ;;
  image) {'exit 0' if image else 'exit 1'} ;;
  run) echo cid123 ;;
  exec) case " $* " in *" -i "*) cat >/dev/null ;; esac ;;
esac
exit 0
""")
    (bindir / "docker").chmod(0o755)
    return bindir, log


def make_source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    env = {"PATH": SYSTEM_PATH, "HOME": str(tmp_path), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    (src / "README").write_text("x\n")
    for cmd in (["git", "init", "-q"], ["git", "add", "README"], ["git", "commit", "-q", "-m", "m"]):
        subprocess.run(cmd, cwd=src, env=env, check=True, capture_output=True)
    return src


def run_docker(tmp_path, *args, daemon=True, image=False, env_extra=None):
    bindir, log = fake_docker(tmp_path, daemon=daemon, image=image)
    env = {"PATH": f"{bindir}:{SYSTEM_PATH}", "HOME": str(tmp_path), "TMPDIR": str(tmp_path)}
    env.update(env_extra or {})
    done = subprocess.run(["sh", str(DOCKER_SCRIPT), "--source", str(make_source(tmp_path)), *args], env=env, capture_output=True, text=True, timeout=120)
    return done, (log.read_text().splitlines() if log.exists() else [])


def test_docker_helper_builds_runs_copies_a_clone_and_removes_the_container(tmp_path):
    done, log = run_docker(tmp_path, "--env", "SECRET_KEY", "--", "true", env_extra={"SECRET_KEY": "s3cret"})
    assert done.returncode == 0, done.stderr
    build = [line for line in log if line.startswith("build ")]
    assert len(build) == 1 and "--platform linux/amd64" in build[0] and "-t topopt-clean-room" in build[0]
    assert build[0].endswith("docker/clean-room")                                  # the context is the Dockerfile folder only
    run_line = [line for line in log if line.startswith("run ")][0]
    assert "-e SECRET_KEY" in run_line and "s3cret" not in "\n".join(log)          # the name goes in, the value is not on a command line
    assert not any(" -v " in f" {line} " or "--mount" in line for line in log)     # no mount of your folders
    assert any(line.startswith("exec -i cid123 tar -xf - -C /home/tester") for line in log)
    assert any(line.startswith("exec -w /home/tester/topopt cid123 true") for line in log)
    assert log[-1] == "rm -f cid123"
    assert not list(tmp_path.glob("clean-room-docker.*"))                          # the temporary clone is gone



def test_docker_helper_passes_the_proxy_of_this_computer_by_name_only(tmp_path):
    """A host with no direct route needs its proxy in the build (apt-get) and in the container (npm, git). Only the names go on a command line."""
    proxy = {"HTTPS_PROXY": "http://user:pw@proxy.example:3128", "http_proxy": "http://proxy.example:3128", "NO_PROXY": "localhost,.internal"}
    done, log = run_docker(tmp_path, "--", "true", env_extra=proxy)
    assert done.returncode == 0, done.stderr
    build = [line for line in log if line.startswith("build ")][0]
    run_line = [line for line in log if line.startswith("run ")][0]
    for name in proxy:
        assert f"--build-arg {name}" in build and f"-e {name}" in run_line, name
    for value in proxy.values():
        assert value not in "\n".join(log) and value not in done.stdout + done.stderr
    assert "HTTPS_PROXY" in done.stderr and "values are not shown" in done.stderr
    assert "localhost" not in done.stderr                       # NO_PROXY names localhost, and that is no proxy on this computer


def test_docker_helper_without_a_proxy_passes_none_and_says_nothing(tmp_path):
    done, log = run_docker(tmp_path, "--", "true")
    assert done.returncode == 0, done.stderr
    assert not any("--build-arg" in line for line in log) and "proxy" not in done.stderr


def test_docker_helper_sets_the_other_case_of_a_proxy_name(tmp_path):
    """apt reads only http_proxy and https_proxy. A host that sets only the upper-case names must still give apt-get its proxy."""
    proxy = {"HTTP_PROXY": "http://proxy.example:3128", "HTTPS_PROXY": "http://proxy.example:3129", "no_proxy": ".internal"}
    done, log = run_docker(tmp_path, "--", "true", env_extra=proxy)
    assert done.returncode == 0, done.stderr
    build = [line for line in log if line.startswith("build ")][0]
    run_line = [line for line in log if line.startswith("run ")][0]
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        assert f"--build-arg {name}" in build and f"-e {name}" in run_line, name
    assert "ALL_PROXY" not in build and "all_proxy" not in build                     # a name that is not set goes in no case
    seen = dict(line.split("=", 1) for line in (tmp_path / "docker-build.env").read_text().splitlines())
    assert seen["http_proxy"] == seen["HTTP_PROXY"] == proxy["HTTP_PROXY"]          # docker gets the value from the environment
    assert seen["https_proxy"] == seen["HTTPS_PROXY"] == proxy["HTTPS_PROXY"]
    assert seen["NO_PROXY"] == seen["no_proxy"] == proxy["no_proxy"]
    for value in proxy.values():
        assert value not in "\n".join(log) and value not in done.stdout + done.stderr


def test_docker_helper_keeps_two_different_values_of_a_proxy_name(tmp_path):
    proxy = {"HTTP_PROXY": "http://one.example:3128", "http_proxy": "http://two.example:3128"}
    done, log = run_docker(tmp_path, "--", "true", env_extra=proxy)
    assert done.returncode == 0, done.stderr
    seen = dict(line.split("=", 1) for line in (tmp_path / "docker-build.env").read_text().splitlines())
    assert seen["HTTP_PROXY"] == proxy["HTTP_PROXY"] and seen["http_proxy"] == proxy["http_proxy"]


@pytest.mark.parametrize("value", ["http://127.0.0.1:3128", "http://u:pw@127.0.0.1:3128", "0.0.0.0:3128", "http://[::1]:3128/",
                                   "localhost:3128", "http://localhost"])
def test_docker_helper_notes_a_proxy_on_localhost(tmp_path, value):
    done, log = run_docker(tmp_path, "--", "true", env_extra={"HTTP_PROXY": value})
    assert done.returncode == 0, done.stderr
    assert "HTTP_PROXY http_proxy names an address on this computer" in done.stderr and "host.docker.internal" in done.stderr
    assert "Docker Engine on Linux" in done.stderr
    assert value not in done.stderr and "pw" not in done.stderr


@pytest.mark.parametrize("value", ["http://localhost.example.com:3128", "http://proxy.example:3128/127.0.0.1", "http://10.0.0.1:3128"])
def test_docker_helper_gives_no_localhost_note_for_another_host(tmp_path, value):
    done, log = run_docker(tmp_path, "--", "true", env_extra={"HTTP_PROXY": value})
    assert done.returncode == 0, done.stderr
    assert "HTTP_PROXY" in done.stderr and "localhost" not in done.stderr


def test_docker_helper_does_not_build_when_the_image_is_there_and_keeps_on_request(tmp_path):
    done, log = run_docker(tmp_path, "--keep", "--", "true", image=True)
    assert done.returncode == 0, done.stderr
    assert not any(line.startswith("build ") for line in log)
    assert not any(line.startswith("rm ") for line in log) and "docker rm -f cid123" in done.stdout


def test_docker_helper_stops_with_a_clear_text_without_a_daemon(tmp_path):
    done, log = run_docker(tmp_path, daemon=False)
    assert done.returncode == 1 and "no Docker daemon" in done.stderr
    assert not any(line.startswith(("build", "run")) for line in log)


def test_docker_helper_refuses_a_variable_that_is_not_set(tmp_path):
    done, log = run_docker(tmp_path, "--env", "NOT_SET_ANYWHERE_X")
    assert done.returncode == 1 and "not set" in done.stderr
    assert not any(line.startswith(("build", "run")) for line in log)


def test_dockerfile_has_no_uv_no_python_and_no_key():
    text = DOCKERFILE.read_text()
    assert "ubuntu:24.04" in text and "git" in text and "curl" in text and "nodejs" in text
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for banned in ("uv", "python", "pip", "API_KEY", "TOKEN", "COPY ", "ADD "):
        assert banned not in code.replace("--no-install-recommends", ""), banned


# --- the review round (F1 to F14) --------------------------------------------------------------------------------------

def shim(tmp_path: Path, name: str, body: str) -> str:
    """A command in a folder that goes first on PATH. Returns the folder."""
    folder = tmp_path / "shims"
    folder.mkdir(exist_ok=True)
    (folder / name).write_text("#!/bin/sh\n" + body)
    (folder / name).chmod(0o755)
    return str(folder)


def restored_tree(box: Path) -> dict:
    return snapshot(box, skip=(".topopt-clean-room",))


@pytest.mark.parametrize("when", ["after", "before"])
@pytest.mark.parametrize("at", range(1, 10))
def test_f1_an_interrupt_at_any_move_leaves_a_backup_that_restore_can_bring_back(box, tmp_path, when, at):
    """SIGINT right after (or right before) the Nth mv of hide: the row is in the manifest already, and restore gives everything back."""
    before = restored_tree(box)
    counter = tmp_path / "count"
    kill = f'kill -INT $PPID; exit 130' if when == "before" else None
    body = f'''n=$(cat {counter} 2>/dev/null || echo 0); n=$((n + 1)); echo $n > {counter}
if [ "$n" = {at} ]; then
  {kill or ":"}
fi
/bin/mv "$@"; rc=$?
if [ "$n" = {at} ]; then kill -INT $PPID; fi
exit $rc
'''
    folder = shim(tmp_path, "mv", body)
    hide = run(box, "hide", "--apply", path_extra=folder, check=False)
    assert hide.returncode == 130 or hide.returncode == 0 and at > 9
    assert run(box, "restore", "--apply").returncode == 0
    assert restored_tree(box) == before


def test_f1_a_file_in_the_backup_folder_with_no_row_is_reported(box):
    run(box, "hide", "--apply")
    backup = backup_of(box)
    lines = (backup / "manifest.tsv").read_text().splitlines()
    (backup / "manifest.tsv").write_text("\n".join(line for line in lines if "files/001" not in line) + "\n")
    done = run(box, "restore", "--apply", check=False)
    assert done.returncode == 1 and "no row of the manifest names it" in done.stdout


def test_f2_a_second_hide_is_refused_until_the_first_backup_is_restored(box, tmp_path):
    import shutil
    before = restored_tree(box)
    run(box, "hide", "--apply")
    (box / ".cache" / "uv").mkdir()                                           # a leftover of the test
    (box / ".cache" / "uv" / "new").write_text("test\n")
    (box / ".local" / "bin" / "uv").write_text("downloaded by the test\n")
    refused = run(box, "hide", "--apply", check=False)
    assert refused.returncode == 1
    assert "not restored" in refused.stderr and str(backup_of(box)) in refused.stderr and "restore --apply --from" in refused.stderr
    assert len(list((box / ".topopt-clean-room").iterdir())) == 1             # no second backup
    first = run(box, "restore", "--apply", check=False)
    assert first.returncode == 1 and "INCOMPLETE" in first.stdout
    assert not (backup_of(box) / "restored").exists()
    assert run(box, "hide", "--apply", check=False).returncode == 1           # still refused
    shutil.rmtree(box / ".cache" / "uv")
    (box / ".local" / "bin" / "uv").unlink()
    assert run(box, "restore", "--apply").returncode == 0
    assert (backup_of(box) / "restored").exists()
    assert restored_tree(box) == {k: v for k, v in before.items() if not k.startswith(".topopt-clean-room")}
    assert run(box, "hide", "--apply").returncode == 0                        # now the next hide is allowed


def test_f10_two_backups_in_the_same_second_restore_takes_the_newer_one(box, tmp_path):
    stamp = "20261026T120000Z"
    folder = shim(tmp_path, "date", f'case "$*" in *%Y%m%dT%H%M%SZ*) echo {stamp} ;; *) exec /bin/date "$@" ;; esac\n')
    before = restored_tree(box)
    run(box, "hide", "--apply", path_extra=folder)
    run(box, "restore", "--apply")
    run(box, "hide", "--apply", path_extra=folder)
    names = sorted(p.name for p in (box / ".topopt-clean-room").iterdir())
    assert names == [stamp, stamp + "_2"]
    assert run(box, "restore", "--apply").returncode == 0                     # the unfinished one is the newer one
    assert restored_tree(box) == {k: v for k, v in before.items() if not k.startswith(".topopt-clean-room")}


def test_f3_restore_takes_the_original_back_when_the_client_only_added_topopt_again(box):
    original = (box / ".cursor" / "mcp.json").read_bytes()
    run(box, "hide", "--apply")
    cfg = box / ".cursor" / "mcp.json"
    text = cfg.read_text()
    assert text.endswith("    }\n  }\n}\n")
    cfg.write_text(text[: -len("    }\n  }\n}\n")] + '    },\n    "topopt": {"command": "again"}\n  }\n}\n')
    done = run(box, "restore", "--apply")
    assert "the client added topopt again" in done.stdout
    assert cfg.read_bytes() == original


@pytest.mark.parametrize("name, text", [
    ("inline_table", 'model = "m"\n\n[mcp_servers]\ntopopt = { command = "sh" }\nalpha = { command = "a" }\n'),
    ("dotted_root", 'mcp_servers.topopt.command = "sh"\nmcp_servers.alpha.command = "a"\n'),
    ("dotted_under_table", '[mcp_servers]\ntopopt.command = "sh"\n'),
])
def test_f4_a_topopt_entry_in_a_toml_form_that_is_not_handled_is_reported_and_never_clean(box, name, text):
    (box / ".codex" / "config.toml").write_text(text)
    status = run(box, "status", check=False)
    codex_lines = [i for i, line in enumerate(status.stdout.splitlines()) if line.strip().startswith("reported-only") and "config.toml" in line]
    assert codex_lines, status.stdout
    assert "config.toml (no topopt entry)" not in status.stdout
    run(box, "hide", "--apply")
    assert (box / ".codex" / "config.toml").read_text() == text            # hide did not touch it


def test_f4_claude_json_with_a_topopt_server_is_reported(box):
    (box / ".claude.json").write_text('{"mcpServers": {"topopt": {"command": "sh"}}}\n')
    status = run(box, "status", check=False)
    assert ".claude.json" in status.stdout and "claude mcp remove topopt" in status.stdout
    before = (box / ".claude.json").read_bytes()
    run(box, "hide", "--apply")
    assert (box / ".claude.json").read_bytes() == before


@pytest.mark.parametrize("text", [
    '{"servers": {"topopt": {"command": "sh"} /* t */ , "b": {}}}',
    '{\n  "servers": {\n    "b": {}, // keep b\n    "topopt": {"command": "sh"}\n  }\n}\n',
    '{"servers": {"b": {}, "topopt": {"command": "sh"} // ours\n  , "c": {}}}',
])
def test_f5_a_jsonc_edit_that_would_leave_a_bad_comma_is_not_written(box, text):
    (box / ".cursor" / "mcp.json").write_text(text)
    run(box, "hide", "--apply")
    new = (box / ".cursor" / "mcp.json").read_text()
    stripped = new
    # Either the file is untouched (reported only), or it has no comma next to a brace.
    import re
    cleaned = re.sub(r"//[^\n]*|/\*.*?\*/", "", stripped, flags=re.S)
    assert not re.search(r"[{\[]\s*,|,\s*[}\]]|,\s*,", cleaned), new


def test_f5_a_toml_array_line_that_starts_with_a_bracket_does_not_end_the_table(box):
    import tomllib
    text = ('[mcp_servers.alpha]\ncommand = "a"\n\n[mcp_servers.topopt]\ncommand = "sh"\nargs = [\n["a"],\n"b"\n]\n'
            '[mcp_servers.topopt.env]\nA = "1"\n\n# b is my work server, keep it\n[mcp_servers.b]\ncommand = "b"\n')
    (box / ".codex" / "config.toml").write_text(text)
    run(box, "hide", "--apply")
    new = (box / ".codex" / "config.toml").read_text()
    data = tomllib.loads(new)
    assert set(data["mcp_servers"]) == {"alpha", "b"}
    assert "# b is my work server, keep it" in new


def test_f6_shell_stops_when_dropping_the_uv_folders_leaves_no_sh(box, tmp_path):
    system = tmp_path / "allbin"
    system.mkdir()
    for folder in ("/usr/bin", "/bin"):
        for item in Path(folder).iterdir():
            if not (system / item.name).exists():
                (system / item.name).symlink_to(item)
    (system / "uv").unlink(missing_ok=True)
    (system / "uv").write_text("#!/bin/sh\n")
    (system / "uv").chmod(0o755)
    before = snapshot(box)
    done = run(box, "shell", "--dir", str(tmp_path / "nu"), "--", "true", path=str(system), check=False)
    assert done.returncode == 1 and "has no" in done.stderr and "--path" in done.stderr and str(system) in done.stderr
    assert snapshot(box) == before and not (tmp_path / "nu").exists()


def test_f7_the_dockerfile_sets_the_npm_prefix_with_env_and_the_script_uses_a_tty_only_on_a_tty(tmp_path):
    text = DOCKERFILE.read_text()
    assert "ENV NPM_CONFIG_PREFIX=/home/tester/.npm-global" in text and "ENV PATH=/home/tester/.npm-global/bin:$PATH" in text
    done, log = run_docker(tmp_path)                                  # no command, stdin is not a terminal
    assert done.returncode == 0, done.stderr
    assert any(line.startswith("exec -i -w /home/tester/topopt cid123 bash -l") for line in log)
    assert not any(" -it " in f" {line} " for line in log)


def test_f12_a_bad_variable_name_is_refused_before_eval(box, tmp_path):
    bad = 'HOME+x};echo${IFS}$((6*7))>&2;:${HOME'
    done = run(box, "shell", "--dir", str(tmp_path / "x"), "--env", bad, "--", "true", check=False)
    assert done.returncode == 1 and "not a variable name" in done.stderr and "42" not in done.stderr
    docker, log = run_docker(tmp_path, "--env", bad)
    assert docker.returncode == 1 and "not a variable name" in docker.stderr and "42" not in docker.stderr
    assert not any(line.startswith(("build", "run")) for line in log)


def test_f8_a_backup_folder_on_another_file_system_moves_nothing(box, tmp_path):
    folder = shim(tmp_path, "df", 'echo "Filesystem 1024-blocks Used Available Capacity Mounted on"\n'
                  'case "$*" in *topopt-clean-room*) echo "dev 1 1 1 1% /mnt/backup" ;; *) echo "dev 1 1 1 1% /" ;; esac\n')
    before = snapshot(box)
    done = run(box, "hide", "--apply", path_extra=folder, check=False)
    assert done.returncode == 1 and "same file system" in done.stderr
    assert (box / ".cache" / "uv" / "wheels" / "a.whl").is_file() and (box / ".local" / "bin" / "uv").is_file()    # no move happened
    assert (box / ".local" / "share" / "uv" / "python").is_dir() and (box / ".local" / "state" / "topopt").is_dir()


def test_f13_a_df_that_prints_nothing_moves_nothing(box, tmp_path):
    folder = shim(tmp_path, "df", "exit 1\n")
    before = snapshot(box)
    done = run(box, "hide", "--apply", path_extra=folder, check=False)
    assert done.returncode == 1 and "df printed nothing" in done.stderr
    assert snapshot(box, skip=(".topopt-clean-room",)) == {k: v for k, v in before.items() if not k.startswith(".topopt-clean-room")}


def test_f8_a_name_with_a_tab_is_skipped_and_stays(box):
    odd = box / ".claude" / "plugins" / "data" / "topopt-a\tb"
    odd.mkdir()
    (odd / "f").write_text("x")
    done = run(box, "hide", "--apply")
    assert "tab or a newline" in done.stderr
    assert (odd / "f").read_text() == "x"
    assert not (box / ".cache" / "uv").exists()                  # the rest was hidden


def test_f8_a_symlinked_config_is_reported_and_not_edited(box, tmp_path):
    real = tmp_path / "dotfiles-mcp.json"
    real.write_text(json.dumps(CURSOR) + "\n")
    link = box / ".cursor" / "mcp.json"
    link.unlink()
    link.symlink_to(real)
    status = run(box, "status", check=False)
    assert "symbolic link" in status.stdout
    run(box, "hide", "--apply")
    assert link.is_symlink() and json.loads(real.read_text()) == CURSOR


def test_f8_hide_and_restore_rename_the_items(box):
    """The same inode before and after: the item was renamed (mv), not copied and deleted."""
    items = [box / ".cache" / "uv", box / ".local" / "bin" / "uv", box / ".local" / "share" / "uv" / "python"]
    inodes = [p.stat().st_ino for p in items]
    run(box, "hide", "--apply")
    backup = backup_of(box)
    moved = {p.stat().st_ino for p in (backup / "files").iterdir() if not p.is_symlink()}
    assert set(inodes) <= moved
    run(box, "restore", "--apply")
    assert [p.stat().st_ino for p in items] == inodes


def test_f8_a_path_with_dotdot_is_not_moved(box, tmp_path):
    outside = tmp_path / "outside-env"
    outside.mkdir()
    (outside / "keep").write_text("x")
    sneaky = f"{box}/.local/../../outside-env"
    done = run(box, "hide", "--apply", env_extra={"TOPOPT_ENV_DIR": sneaky})
    assert (outside / "keep").read_text() == "x" and "outside your home" in done.stdout


def test_status_lists_the_glob_checks_that_found_nothing(box):
    """A tester sees that the Claude Code plugin data and the plugin install were checked, also when nothing was found."""
    import shutil
    shutil.rmtree(box / ".claude" / "plugins")
    done = run(box, "status", check=False)
    checked = done.stdout.split("Checked and not there:")[1]
    assert "plugins/data/topopt-* (no match)" in checked
    assert "plugins/installed_plugins.json" in checked
    (box / ".claude" / "plugins").mkdir()
    (box / ".claude" / "plugins" / "installed_plugins.json").write_text('{"plugins": {"other@x": {}}}\n')
    done = run(box, "status", check=False)
    assert "installed_plugins.json (no topopt plugin)" in done.stdout.split("Checked and not there:")[1]


def test_hide_says_that_brew_unlink_takes_uv_from_all_projects(box, tmp_path):
    prefix = tmp_path / "brewprefix"
    bindir = make_brew(tmp_path, prefix)
    dry = run(box, "hide", path_extra=f"{bindir}:{prefix}/bin")
    assert "off PATH for all your projects" in dry.stdout
    done = run(box, "hide", "--apply", path_extra=f"{bindir}:{prefix}/bin")
    assert "unlinked      brew uv" in done.stdout and "off PATH for all your projects" in done.stdout


# --- round 3 (N1 to N5) -------------------------------------------------------------------------------------------------

def client_rewrites_cursor_config(box: Path) -> str:
    """The client writes the whole file again: other formatting, a topopt entry, a new key."""
    text = json.dumps({"mcpServers": {"alpha": {"command": "a"}, "topopt": {"command": "again"}}, "newKey": 1}, indent=4) + "\n"
    (box / ".cursor" / "mcp.json").write_text(text)
    return text


def test_n1_keep_changed_configs_closes_the_backup(box):
    original = (box / ".cursor" / "mcp.json").read_text()
    run(box, "hide", "--apply")
    text = client_rewrites_cursor_config(box)
    first = run(box, "restore", "--apply", check=False)
    assert first.returncode == 1 and not (backup_of(box) / "restored").exists()
    bpath = next(line.split("\t")[3] for line in (backup_of(box) / "manifest.tsv").read_text().splitlines() if "mcp.json" in line)
    assert "SKIP" in first.stdout and bpath in first.stdout
    assert f"cp '{bpath}' '{box}/.cursor/mcp.json'" in first.stdout and "--keep-changed-configs" in first.stdout
    assert run(box, "hide", "--apply", check=False).returncode == 1                       # still refused
    dry = run(box, "restore", "--keep-changed-configs")
    assert "would keep" in dry.stdout and "kept " not in dry.stdout.replace("would keep", "")
    assert not (backup_of(box) / "restored").exists()
    done = run(box, "restore", "--apply", "--keep-changed-configs", check=False)
    assert done.returncode == 0, done.stdout
    assert "kept" in done.stdout and bpath in done.stdout
    assert "1 config(s) kept" in done.stdout.splitlines()[-1]
    assert (backup_of(box) / "restored").exists()
    assert (box / ".cursor" / "mcp.json").read_text() == text                             # the current file stays
    assert Path(bpath).read_text() == original                                           # the original stays in the backup
    assert run(box, "hide", "--apply", check=False).returncode == 0                       # the next hide is allowed


def test_n1_putting_the_original_back_by_hand_also_closes_the_backup(box):
    original = (box / ".cursor" / "mcp.json").read_text()
    run(box, "hide", "--apply")
    client_rewrites_cursor_config(box)
    first = run(box, "restore", "--apply", check=False)
    bpath = next(line.split("\t")[3] for line in (backup_of(box) / "manifest.tsv").read_text().splitlines() if "mcp.json" in line)
    (box / ".cursor" / "mcp.json").write_text(Path(bpath).read_text())                    # what the hint (a) says
    assert first.returncode == 1
    assert run(box, "restore", "--apply").returncode == 0
    assert (box / ".cursor" / "mcp.json").read_text() == original
    assert (backup_of(box) / "restored").exists()


def test_n1_the_flag_does_not_unblock_a_skipped_moved_item(box):
    run(box, "hide", "--apply")
    client_rewrites_cursor_config(box)
    (box / ".local" / "bin" / "uv").write_text("downloaded by the test\n")
    done = run(box, "restore", "--apply", "--keep-changed-configs", check=False)
    assert done.returncode == 1 and "INCOMPLETE" in done.stdout.splitlines()[-1]
    assert "kept" in done.stdout and "SKIP          " + str(box / ".local" / "bin" / "uv") in done.stdout
    assert not (backup_of(box) / "restored").exists()
    assert (box / ".local" / "bin" / "uv").read_text() == "downloaded by the test\n"


def test_n2_from_with_a_final_slash_or_a_relative_path_gives_no_false_fail(box):
    before = restored_tree(box)
    run(box, "hide", "--apply")
    done = run(box, "restore", "--apply", "--from", str(backup_of(box)) + "/")
    assert "FAIL" not in done.stdout and (backup_of(box) / "restored").exists()
    assert restored_tree(box) == before
    run(box, "hide", "--apply")
    newest = sorted((box / ".topopt-clean-room").iterdir())[-1]
    done = run(box, "restore", "--apply", "--from", os.path.relpath(newest, os.getcwd()))
    assert "FAIL" not in done.stdout and (newest / "restored").exists()
    assert restored_tree(box) == before


def test_n3_restore_after_an_interrupted_hide_says_that_hide_did_not_finish(box, tmp_path):
    body = f'/bin/mv "$@"; rc=$?; n=$(cat {tmp_path}/count 2>/dev/null || echo 0); n=$((n + 1)); echo $n > {tmp_path}/count\nif [ "$n" = 1 ]; then kill -INT $PPID; fi\nexit $rc\n'
    folder = shim(tmp_path, "mv", body)
    hide = run(box, "hide", "--apply", path_extra=folder, check=False)
    assert hide.returncode == 130
    assert "# hide finished" not in (backup_of(box) / "manifest.tsv").read_text()
    done = run(box, "restore", "--apply")
    header = next(line for line in done.stdout.splitlines() if line.startswith("New since hide"))
    assert "hide did not finish" in header and "Do not delete" in header
    # A hide that ran to the end writes the line, and the header stays as it was.
    (backup_of(box) / "restored").unlink()
    for p in (box / ".topopt-clean-room").rglob("manifest.tsv"):
        p.unlink()
    import shutil
    shutil.rmtree(box / ".topopt-clean-room")
    run(box, "hide", "--apply")
    assert (backup_of(box) / "manifest.tsv").read_text().splitlines()[-1] == "# hide finished"
    done = run(box, "restore", "--apply")
    assert "hide did not finish" not in done.stdout


def test_n4_the_refusal_also_looks_in_the_default_backup_root(box):
    run(box, "hide", "--apply")                                                       # the default root
    (box / ".local" / "bin" / "uv").write_text("downloaded by the test\n")            # something to hide again
    refused = run(box, "hide", "--apply", "--backup-dir", str(box / "bk1"), check=False)
    assert refused.returncode == 1 and "not restored" in refused.stderr and str(backup_of(box)) in refused.stderr
    assert not (box / "bk1").exists()
    # The root in use is the default root: the folder is listed once.
    refused = run(box, "hide", "--apply", check=False)
    assert refused.returncode == 1 and refused.stderr.count("not restored:") == 1
    # The other direction: a hide in another root, then a plain hide.
    (box / ".local" / "bin" / "uv").unlink()
    run(box, "restore", "--apply")
    run(box, "hide", "--apply", "--backup-dir", str(box / "bk1"))
    (box / ".local" / "bin" / "uv").write_text("again\n")
    refused = run(box, "hide", "--apply", check=False)
    assert refused.returncode == 1 and str(box / "bk1") in refused.stderr and refused.stderr.count("not restored:") == 1
    refused = run(box, "hide", "--apply", "--backup-dir", str(box / "bk1"), check=False)
    assert refused.returncode == 1 and refused.stderr.count("not restored:") == 1


def test_n5_the_claude_desktop_globs_run_only_on_a_mac_or_where_the_folder_exists(box, tmp_path):
    needle = "Claude Extensions"
    linux = run(box, "status", check=False, path_extra=shim(tmp_path, "uname", 'echo Linux\n'))
    assert needle not in linux.stdout
    (box / "Library" / "Application Support" / "Claude").mkdir(parents=True)
    assert needle in run(box, "status", check=False).stdout.split("Checked and not there:")[1]
    import shutil
    shutil.rmtree(box / "Library")
    mac = run(box, "status", check=False, path_extra=shim(tmp_path, "uname", 'echo Darwin\n'))
    assert needle in mac.stdout.split("Checked and not there:")[1]


# --- review of round 3 (R1 to R3) ------------------------------------------------------------------------------------

def test_r1_a_hide_with_a_skipped_item_does_not_call_that_item_new(box, tmp_path):
    folder = shim(tmp_path, "df", 'echo "Filesystem 1024-blocks Used Available Capacity Mounted on"\n'
                  'case "$*" in *.cache*) echo "dev 1 1 1 1% /mnt/other" ;; *) echo "dev 1 1 1 1% /" ;; esac\n')
    hide = run(box, "hide", "--apply", path_extra=folder, check=False)
    assert hide.returncode == 1 and (box / ".cache" / "uv" / "wheels" / "a.whl").is_file()      # the cache stayed in place
    assert (backup_of(box) / "manifest.tsv").read_text().splitlines()[-1] == "# hide finished with 1 failed item(s)"
    done = run(box, "restore", "--apply")
    header = next(line for line in done.stdout.splitlines() if line.startswith("New since hide"))
    assert "skipped items" in header and "Do not delete" in header


def test_r2_a_config_that_hide_did_not_reach_is_left_with_no_skip(box, tmp_path):
    folder = shim(tmp_path, "cp", 'kill -INT $PPID; exit 130\n')                  # stop at the first copy of a config
    hide = run(box, "hide", "--apply", path_extra=folder, check=False)
    assert hide.returncode == 130
    cfg = box / ".cursor" / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"alpha": {"command": "b"}}}, indent=4) + "\n")    # the client wrote it again
    done = run(box, "restore", "--apply")
    assert "hide did not change this file" in done.stdout and "kept" not in done.stdout
    assert json.loads(cfg.read_text()) == {"mcpServers": {"alpha": {"command": "b"}}}


def test_r3_a_backup_dir_with_a_final_slash_is_the_default_root(box):
    run(box, "hide", "--apply", "--backup-dir", str(box / ".topopt-clean-room") + "/")
    assert not (box / ".topopt-clean-room" / "roots").exists()
    (box / ".local" / "bin" / "uv").write_text("downloaded by the test\n")
    refused = run(box, "hide", "--apply", check=False)
    assert refused.returncode == 1 and refused.stderr.count("not restored:") == 1 and "//" not in refused.stderr
