"""Tests for scripts/topopt-env.sh, the launcher of the Claude Code plugin.

A fake `uv` stands in for the real one, so the tests need no network. The fake `uv` writes each call
into a log and makes a fake `bin/python` (a shell script) in the new environment.
One test (`test_every_flag_that_the_x86_mac_install_passes_to_uv_exists_in_the_real_uv`) also runs `uv pip install --help` of the
real uv (the variable UV, else `uv` on PATH; it skips if there is none). It needs no network: a fake `uv` takes any flag.

Knobs of the fake `uv` (environment variables of the launcher run):
  FAKE_UV_BREAK_LOCK=<s>  the install of the lock file "breaks" the environment (a flag file, FAKE_UV_BROKEN) and sleeps <s> s,
                          so a test can kill the setup there (a cut in step 2). The fake `python` fails an `import` while
                          the flag is there, like a package that has its .dist-info and no files (it fails a call that
                          names scipy, as the import check of the launcher does).
  FAKE_UV_BREAK_LINK=1    the link of step 3 sets the flag and ends well (so the import check of the launcher fails).
  FAKE_UV_HOLD=<file>     the install of the lock file waits until <file> exists (at most 60 s): a setup that runs as long as
                          the test needs, with no timing guess. The fake uv writes <file>.held when it starts to wait.
  FAKE_UV_HOLD_VENV=<file>  `uv venv` waits until <file> exists (at most 60 s): a setup that stays in step 1. It writes <file>.held when it starts to wait.
  FAKE_UV_FAIL_VENV=<text>  `uv venv` (step 1) writes <text> to stderr and fails: a setup that stops before the marker of step 2.
  FAKE_UV_PROGRESS=<file>  the install of the lock file first writes <file> to stderr (the `Downloading`/` Downloaded` lines of a real uv).
  Any `uv pip install --reinstall` removes the flag (uv writes the files again).
Knobs of the fake `python`: FAKE_PY_LOG (a file: each call is written there with its folder), FAKE_PY_NO_DOCTOR=1
(`-m topopt doctor` fails like the argparse error of a topopt that has no such command), FAKE_PY_ENV_LOG (a file: each call writes
the line `TOPOPT_UV=<value>` there, `unset` if the launcher did not export it).

REL-9 (the launcher downloads a pinned uv when the user has none): see the section at the end of this file. The launcher has no
switch for tests. A fake `curl` on PATH copies a local archive, and the tests rewrite the pinned SHA256 constants in the box copy of the
script (or keep the real ones, to test a wrong sum). A test with a PATH that has no uv must never reach the real network: use `no_uv_env`.
"""
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The Python that the launcher makes: the one source is the variable LAUNCHER_PYTHON in the launcher (#422).
LAUNCHER_PY = re.search(r"^LAUNCHER_PYTHON=(\d+\.\d+)$", (REPO / "scripts" / "topopt-env.sh").read_text(), re.M).group(1)

# The shell that runs the launcher. Default: `sh` of this machine (dash on Debian, bash in POSIX mode on a Mac).
# To test another one: TOPOPT_TEST_SH="bash --posix" pytest tests/test_plugin_launcher.py
SH = shlex.split(os.environ.get("TOPOPT_TEST_SH", "sh"))
# The same shell with an absolute path: for a wrapper named `sh` (it must not call itself) and for a run with a PATH that
# has no shell.
SH_ABS = [shutil.which(SH[0]) or SH[0], *SH[1:]]

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the launcher is a POSIX sh script")

FAKE_UV = """#!/bin/sh
echo "$@" >> "$FAKE_UV_LOG"
echo "$PWD" >> "$FAKE_UV_LOG.pwd"
[ "$1" = "--version" ] && { echo "uv 99.0.0-fake"; exit 0; }
[ -n "$FAKE_UV_SLEEP" ] && sleep "$FAKE_UV_SLEEP"
case "$1 $2" in
  "venv --python")
    [ -n "$FAKE_UV_FAIL_VENV" ] && { echo "$FAKE_UV_FAIL_VENV" >&2; exit 1; }
    [ -n "$FAKE_UV_HOLD_VENV" ] && { : > "$FAKE_UV_HOLD_VENV.held"; n=0; while [ ! -e "$FAKE_UV_HOLD_VENV" ] && [ "$n" -lt 1200 ]; do sleep 0.05; n=$((n + 1)); done; }
    for last; do :; done                      # the target is the last argument
    case "$*" in
      *" --clear "*) rm -rf "$last" ;;
      *) [ -e "$last" ] || [ -L "$last/bin/python" ] && {
           echo "error: Failed to create virtual environment. A virtual environment already exists at: $last" >&2
           exit 2; } ;;
    esac
    mkdir -p "$last/bin"
    if [ -n "$FAKE_UV_VENV_MACHINE" ]; then   # the machine type of the Python that this uv makes
      printf '#!/bin/sh\\necho "%s"\\n' "$FAKE_UV_VENV_MACHINE" > "$last/bin/python"
    else
      cat > "$last/bin/python" <<'PYTHON_END'
#!/bin/sh
[ -n "$FAKE_PY_LOG" ] && echo "$PWD $*" >> "$FAKE_PY_LOG"
[ -n "$FAKE_PY_ENV_LOG" ] && echo "TOPOPT_UV=${TOPOPT_UV-unset}" >> "$FAKE_PY_ENV_LOG"
case "$*" in
  *scipy*) [ -f "$FAKE_UV_BROKEN" ] && { echo "ImportError: cannot import name 'ndimage' from 'scipy'" >&2; exit 1; } ;;
  "-m topopt doctor"*) [ -n "$FAKE_PY_NO_DOCTOR" ] && {
      echo "topopt: error: argument cmd: invalid choice: 'doctor' (choose from 'run', 'check')" >&2; exit 2; } ;;
esac
case "$*" in
  "-P -m topopt_mcp.server"*) [ -n "$FAKE_PY_REAL" ] && exec env PYTHONPATH="$FAKE_PY_REAL_PATH" "$FAKE_PY_REAL" "$@"   # the real server (an end-to-end test)
    [ -n "$FAKE_PY_HANDOFF_LOG" ] && {   # the hand-off of the status server: log what this Python got, then end
      { echo "argv: $*"
        echo "file: ${TOPOPT_MCP_HANDOFF-unset}"
        echo "drop: ${TOPOPT_MCP_HANDOFF_DROP_ID-unset}"
        echo "era: ${TOPOPT_MCP_HANDOFF_ERA-unset}"
        echo "listen: ${TOPOPT_MCP_HANDOFF_LISTEN_ID-unset}"
        echo "umask: $(umask)"
        echo "mode: $(find "$TOPOPT_MCP_HANDOFF" -perm 600 2>/dev/null | wc -l | tr -d ' ')"   # 1 when the file is rw for the owner only
        [ -n "$TOPOPT_MCP_HANDOFF" ] && { sed 's/^/replay: /' "$TOPOPT_MCP_HANDOFF"; rm -f "$TOPOPT_MCP_HANDOFF"; }
        sed 's/^/stdin: /'
        echo "end"; } >> "$FAKE_PY_HANDOFF_LOG"
      exit 0; } ;;
esac
case "$*" in *version_info*) echo "${FAKE_PY_VERSION:-@LAUNCHER_PY@}"; exit 0 ;; esac   # the launcher asks for major.minor
echo "${FAKE_PY_MACHINE:-fake python: $@}"
PYTHON_END
    fi
    chmod +x "$last/bin/python" ;;
  "pip install")
    case "$*" in *--reinstall*) rm -f "$FAKE_UV_BROKEN" ;; esac
    case "$*" in
      *" -r "*) [ -n "$FAKE_UV_PROGRESS" ] && cat "$FAKE_UV_PROGRESS" >&2
                [ -n "$FAKE_UV_HOLD" ] && { : > "$FAKE_UV_HOLD.held"; n=0; while [ ! -e "$FAKE_UV_HOLD" ] && [ "$n" -lt 1200 ]; do sleep 0.05; n=$((n + 1)); done; }
                [ -n "$FAKE_UV_FAIL_LOCK" ] && { echo "$FAKE_UV_FAIL_LOCK" >&2; exit 1; }
                [ -n "$FAKE_UV_BREAK_LOCK" ] && { : > "$FAKE_UV_BROKEN"; sleep "$FAKE_UV_BREAK_LOCK"; } ;;
      *--no-deps*) [ -n "$FAKE_UV_FAIL_LINK" ] && { echo "error: link failed" >&2; exit 1; }
                   [ -n "$FAKE_UV_BREAK_LINK" ] && : > "$FAKE_UV_BROKEN" ;;
      *"[mcp]"*) [ -n "$FAKE_UV_FAIL_PYPROJECT" ] && { echo "$FAKE_UV_FAIL_PYPROJECT" >&2; exit 1; } ;;
    esac ;;
esac
exit 0
""".replace("@LAUNCHER_PY@", LAUNCHER_PY)


@pytest.fixture
def box(tmp_path):
    """A copy of the launcher in a fake plugin root, a fake uv on PATH and a private env dir."""
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    for name in ("requirements.lock", "requirements-macos-x86_64.lock", "pyproject.toml"):
        shutil.copy(REPO / name, root / name)
    script = root / "scripts" / "topopt-env.sh"
    shutil.copy(REPO / "scripts" / "topopt-env.sh", script)
    shutil.copytree(REPO / "topopt_mcp" / "launcher", root / "topopt_mcp" / "launcher")   # the snapshot that the status server serves
    bindir = tmp_path / "bin"
    bindir.mkdir()
    if os.environ.get("TOPOPT_TEST_SH"):
        # The launcher starts `sh` itself (a background setup). Make that `sh` the shell under test.
        wrapper = bindir / "sh"
        wrapper.write_text("#!/bin/sh\nexec " + " ".join(shlex.quote(a) for a in SH_ABS) + ' "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    uv = bindir / "uv"
    uv.write_text(FAKE_UV)
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    env_dir = tmp_path / "data"
    log = tmp_path / "uv.log"
    broken = tmp_path / "broken-flag"
    env = dict(os.environ, PATH=f"{bindir}:/usr/bin:/bin", FAKE_UV_LOG=str(log),
               TOPOPT_ENV_DIR=str(env_dir), HOME=str(tmp_path / "home"), FAKE_UV_BROKEN=str(broken))
    for k in ("CLAUDE_PLUGIN_DATA", "EVAL_TOPOPT_ENV_DIR", "FAKE_UV_FAIL_LOCK", "FAKE_UV_FAIL_LINK", "FAKE_UV_SLEEP", "TOPOPT_MCP_WAIT",
              "FAKE_PY_MACHINE", "FAKE_UV_FAIL_PYPROJECT", "FAKE_UV_VENV_MACHINE", "TOPOPT_ALLOW_SOURCE_BUILD",
              "FAKE_UV_BREAK_LOCK", "FAKE_UV_BREAK_LINK", "FAKE_PY_LOG", "FAKE_PY_NO_DOCTOR", "TOPOPT_DETACH", "FAKE_UV_HOLD",
              "TOPOPT_SYSCTL", "FAKE_UV_FAIL_VENV", "FAKE_UV_HOLD_VENV", "TOPOPT_UV", "FAKE_PY_ENV_LOG", "FAKE_PY_HANDOFF_LOG", "FAKE_PY_REAL", "FAKE_PY_REAL_PATH",
              "FAKE_UV_PROGRESS"):
        env.pop(k, None)

    class Box:
        pass

    b = Box()
    b.root, b.script, b.env_dir, b.log, b.env, b.bindir = root, script, env_dir, log, env, bindir
    b.broken = broken
    b.hold = tmp_path / "hold"
    b.held = tmp_path / "hold.held"
    b.marker = env_dir / "venv" / ".topopt-step2"
    b.stamp = env_dir / "venv" / ".topopt-plugin-stamp"

    def run(*args, stdin="", **extra):
        # stdin is a closed pipe by default: a `mcp` start that is not ready reads its stdin until the client closes it (P9-2b).
        return subprocess.run([*SH, str(script), *args], capture_output=True, text=True, input=stdin,
                              env=dict(env, **extra), timeout=60)

    b.run = run
    # What the fake uv was asked to do. The launcher asks every uv that it could take for its version (#390: a uv on PATH that is
    # too old is skipped), also on a start that needs no uv: those calls are counted apart, in version_calls().
    b.all_calls = lambda: log.read_text().splitlines() if log.exists() else []
    b.calls = lambda: [c for c in b.all_calls() if c != "--version"]
    b.version_calls = lambda: b.all_calls().count("--version")
    yield b
    b.hold.touch()   # a setup that a test left on hold (`not_ready_run`) ends now, as it did before the hold


def test_script_has_valid_sh_syntax():
    assert subprocess.run([*SH, "-n", str(REPO / "scripts" / "topopt-env.sh")]).returncode == 0


def test_usage_error():
    r = subprocess.run([*SH, str(REPO / "scripts" / "topopt-env.sh"), "nonsense"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "usage" in r.stderr and r.stdout == ""
    assert "doctor" in r.stderr and "exit 0" in r.stderr and "exit 1" in r.stderr   # the exit code rule of doctor


def test_status_before_setup(box):
    r = box.run("status")
    assert r.returncode == 1 and "not ready" in r.stderr and "sync" in r.stderr
    assert box.calls() == []


def test_sync_makes_the_environment_in_three_steps(box):
    r = box.run("sync")
    assert r.returncode == 0, r.stderr
    py = box.env_dir / "venv" / "bin" / "python"
    assert r.stdout.strip() == str(py)  # stdout: the path and nothing else
    calls = box.calls()
    # A uv-managed Python only (no Homebrew or system Python), and --clear for a broken venv.
    assert calls[0] == f"venv --python {LAUNCHER_PY} --managed-python --clear {box.env_dir / 'venv'}"
    assert calls[1] == f"pip install --python {py} -r {box.root}/requirements.lock"
    assert calls[2] == f"pip install --python {py} --no-deps --no-build-isolation -e {box.root}"   # the setuptools of step 2 (#397)
    assert len(calls) == 3
    stamp = (box.env_dir / "venv" / ".topopt-plugin-stamp").read_text()
    assert f"root={box.root}" in stamp
    assert not (box.env_dir / ".setup-lock").exists()
    assert box.run("status").returncode == 0
    assert not box.marker.exists()                       # the marker of step 2 is gone after the stamp
    assert not any("--reinstall" in c for c in calls)    # a clean first run installs once, with no --reinstall


def test_second_start_installs_nothing_and_only_asks_the_uv_its_version(box):
    assert box.run("sync").returncode == 0
    box.log.write_text("")
    r = box.run("sync")
    assert r.returncode == 0 and box.calls() == []
    assert box.version_calls() == 1          # #390: the uv on PATH is asked its version once (a uv older than MIN_UV_VERSION is skipped)


def test_changed_lock_file_installs_again_in_the_same_environment(box):
    assert box.run("sync").returncode == 0
    box.log.write_text("")
    with open(box.root / "requirements.lock", "a") as f:
        f.write("six==1.17.0\n")
    assert box.run("status").returncode == 1
    assert box.run("sync").returncode == 0
    calls = box.calls()
    assert not any(c.startswith("venv") for c in calls)  # the environment stays; packages are added
    assert len(calls) == 2


def test_moved_plugin_root_links_the_code_again(box, tmp_path):
    """A plugin update gives a new root folder: the stamp names the root."""
    assert box.run("sync").returncode == 0
    new_root = tmp_path / "root2"
    shutil.copytree(box.root, new_root)
    box.log.write_text("")
    r = subprocess.run([*SH, str(new_root / "scripts" / "topopt-env.sh"), "sync"],
                       capture_output=True, text=True, env=box.env)
    assert r.returncode == 0, r.stderr
    assert box.calls()[-1].endswith(f"--no-deps --no-build-isolation -e {new_root}")


def test_lock_file_that_does_not_install_here_falls_back_to_pyproject(box):
    r = box.run("sync", FAKE_UV_FAIL_LOCK="error: no wheel with a matching platform tag for numpy")
    assert r.returncode == 0, r.stderr
    assert "pins of pyproject.toml" in r.stderr and "matching platform tag" in r.stderr
    assert any(f"{box.root}[mcp]" in c for c in box.calls())
    assert (box.env_dir / "venv" / ".topopt-plugin-stamp").read_text().splitlines()[1] == "pins=pyproject.toml"
    assert "pyproject.toml" in box.run("status").stderr   # status says which pins are in


def lock_pin(name, lock="requirements.lock"):
    """The `name==version` pin of a lock file of the repo (without a comment or a hash after it)."""
    return next(m.group(1) for ln in (REPO / lock).read_text().splitlines() if (m := re.match(rf"({name}==[^\s;#\\]+)", ln)))


def test_the_pyproject_fallback_installs_the_build_backend_of_the_lock(box):
    """#397: step 3 links the code with the setuptools of the environment (--no-build-isolation). The pins of pyproject.toml do not name it,
    so the fallback of step 2 installs the pin of the lock with them."""
    r = box.run("sync", FAKE_UV_FAIL_LOCK="error: no wheel with a matching platform tag for numpy")
    assert r.returncode == 0, r.stderr
    py = box.env_dir / "venv" / "bin" / "python"
    calls = box.calls()
    assert calls[2] == f"pip install --python {py} {box.root}[mcp] {lock_pin('setuptools')}"
    assert calls[3] == f"pip install --python {py} --no-deps --no-build-isolation -e {box.root}" and len(calls) == 4


@pytest.mark.parametrize("form", ["comment", "hash", "case", "crlf"])
def test_the_fallback_gives_uv_the_bare_build_backend_pin_of_any_lock_line(box, form):
    """Review F1 of #397: `uv pip install -r` takes a lock line with a comment, a `\\` and a `--hash` line, another case of the name or
    a CR at the end; as an argument, uv refuses the first two ("Trailing ... is not allowed", uv 0.12.1). The fallback gets the bare pin."""
    version = lock_pin("setuptools").split("==", 1)[1]
    line = {"comment": f"setuptools=={version}  # the build backend",
            "hash": f"setuptools=={version} \\\n    --hash=sha256:{'0' * 64}",
            "case": f"Setuptools=={version}",
            "crlf": f"setuptools=={version}\r"}[form]
    lock = box.root / "requirements.lock"
    text = lock.read_text()
    assert text.count(f"\nsetuptools=={version}\n") == 1
    lock.write_text(text.replace(f"\nsetuptools=={version}\n", f"\n{line}\n"))
    r = box.run("sync", FAKE_UV_FAIL_LOCK="error: no wheel with a matching platform tag for numpy")
    assert r.returncode == 0, r.stderr
    assert box.calls()[2] == f"pip install --python {box.env_dir / 'venv' / 'bin' / 'python'} {box.root}[mcp] setuptools=={version}"


def test_a_lock_without_the_build_backend_gives_the_requirement_of_build_system(box):
    """A lock with no setuptools line (tests/test_locks.py keeps it in both locks): the fallback installs what `[build-system]` requires."""
    lock = box.root / "requirements.lock"
    lock.write_text("".join(ln + "\n" for ln in lock.read_text().splitlines() if not ln.startswith("setuptools==")))
    r = box.run("sync", FAKE_UV_FAIL_LOCK="error: no wheel with a matching platform tag for numpy")
    assert r.returncode == 0, r.stderr
    requires = tomllib.loads((REPO / "pyproject.toml").read_text())["build-system"]["requires"]
    assert len(requires) == 1 and box.calls()[2].endswith(f"[mcp] {requires[0]}")


def test_network_error_does_not_fall_back_and_the_next_start_resumes(box):
    r = box.run("sync", FAKE_UV_FAIL_LOCK="error: Failed to download `numpy`\n  Caused by: dns error: no such host")
    assert r.returncode == 1 and "network" in r.stderr and "resume" in r.stderr
    assert not any("[mcp]" in c for c in box.calls())          # no silent fallback to loose pins
    assert not (box.env_dir / "venv" / ".topopt-plugin-stamp").exists()
    assert not (box.env_dir / ".setup-lock").exists()
    box.log.write_text("")
    assert box.run("sync").returncode == 0                      # the network is back: the lock installs
    assert not any(c.startswith("venv") for c in box.calls())   # the venv stays
    assert (box.env_dir / "venv" / ".topopt-plugin-stamp").read_text().splitlines()[1] == "pins=lock"


def no_uv_env(box, **extra):
    """The environment of a launcher run with no uv on PATH and a `curl` that cannot connect (REL-9: the launcher downloads a uv when it
    finds none, so a test with a PATH that has no uv must never reach the real network)."""
    tools = box.bindir.parent / "offline-tools"
    tools.mkdir(exist_ok=True)
    curl = tools / "curl"
    curl.write_text('#!/bin/sh\necho "curl: (6) Could not resolve host: github.com" >&2\nexit 6\n')
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    return dict(box.env, PATH=f"{tools}:/usr/bin:/bin", HOME=str(box.root), **extra)    # HOME: no ~/.local/bin of the real user


def test_missing_uv_gives_the_install_command(box):
    # The script adds these folders to the end of PATH, so a real uv there would be found.
    if shutil.which("uv", path="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"):
        pytest.skip("a real uv is in a standard folder")
    r = subprocess.run([*SH, str(box.script), "sync"], capture_output=True, text=True, env=no_uv_env(box))   # no uv, and the download fails
    assert r.returncode == 1
    assert "uv is not installed" in r.stderr and "astral.sh/uv/install.sh" in r.stderr
    assert "The automatic download of uv did not work" in r.stderr and "Could not resolve host" in r.stderr
    assert r.stdout == ""


def test_mcp_starts_the_server_module_and_keeps_stdout_clean(box):
    assert box.run("sync").returncode == 0
    r = box.run("mcp")
    assert r.returncode == 0
    assert r.stdout == "fake python: -P -m topopt_mcp.server\n"  # nothing from the launcher
    assert r.stderr == ""


def test_python_and_topopt_subcommands(box):
    assert box.run("sync").returncode == 0
    assert box.run("python", "-c", "x").stdout == "fake python: -c x\n"
    assert box.run("topopt", "check", "a.yaml").stdout == "fake python: -P -m topopt check a.yaml\n"


def test_eval_env_dir_comes_before_the_plugin_data_folder(box, tmp_path):
    """`claude plugin eval --mocks off` gives the plugin an empty temporary CLAUDE_PLUGIN_DATA and passes EVAL_* variables."""
    env = {k: v for k, v in box.env.items() if k != "TOPOPT_ENV_DIR"}
    env["EVAL_TOPOPT_ENV_DIR"] = str(tmp_path / "eval-env")
    env["CLAUDE_PLUGIN_DATA"] = str(tmp_path / "plugin-data")
    r = subprocess.run([*SH, str(box.script), "sync"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "eval-env" / "venv" / ".topopt-plugin-stamp").is_file()
    assert not (tmp_path / "plugin-data").exists()


def test_plugin_data_folder_is_the_default_env_dir(box, tmp_path):
    data = tmp_path / "plugin-data"
    env = {k: v for k, v in box.env.items() if k != "TOPOPT_ENV_DIR"}
    env["CLAUDE_PLUGIN_DATA"] = str(data)
    r = subprocess.run([*SH, str(box.script), "sync"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert (data / "venv" / ".topopt-plugin-stamp").is_file()


def test_first_mcp_start_sets_up_in_the_background_and_exits_before_the_client_timeout(box):
    """The setup is slower than the wait: mcp exits with 1, and the setup finishes in the background after the test releases it.
    The fake uv holds the setup until the test writes `box.hold`, so it cannot end inside the wait on a slow machine (see `not_ready_run`)."""
    r = box.run("mcp", FAKE_UV_SLEEP="1", TOPOPT_MCP_WAIT="1", FAKE_UV_HOLD=str(box.hold))
    assert r.returncode == 1
    assert "not ready yet" in r.stderr and "status" in r.stderr
    assert r.stdout == ""
    box.hold.touch()
    stamp = box.env_dir / "venv" / ".topopt-plugin-stamp"
    setup_log = box.env_dir / "setup.log"

    def setup_done():   # the setup writes the stamp first and the "ready after" line just after it
        return stamp.exists() and setup_log.exists() and "ready after" in setup_log.read_text()

    end = time.time() + 30
    while time.time() < end and not setup_done():
        time.sleep(0.2)
    assert stamp.exists(), setup_log.read_text()
    assert "ready after" in setup_log.read_text()
    # The next start is fast and runs the server.
    again = box.run("mcp")
    assert again.returncode == 0 and "topopt_mcp.server" in again.stdout


def test_the_not_ready_message_says_what_goes_on_how_long_and_what_the_server_does_meanwhile(box):
    """P9-2a, changed by #370. When `mcp` stops waiting, it writes (stderr, the log of the client) that the setup is not ready, how long the
    first start takes, that the setup goes on, and that a tool call gives the setup status until the setup is done.
    It does not say "failed" or "reconnect": the client is not told of a failure. The wait (TOPOPT_MCP_WAIT) is not changed."""
    r = box.run("mcp", FAKE_UV_SLEEP="1", TOPOPT_MCP_WAIT="1", FAKE_UV_HOLD=str(box.hold))
    assert r.returncode == 1 and r.stdout == ""
    lines = r.stderr.splitlines()
    assert all(line.startswith("topopt: ") for line in lines), r.stderr
    start = next(n for n, line in enumerate(lines) if "not ready yet" in line)
    message = "\n".join(lines[start:])
    assert "1 to 3 minutes" in message and "about 200 MB" in message           # the numbers of the README and of the manifest
    assert "tool call gives the setup status" in message and "goes on in the background" in message and "work in the same session" in message
    assert "failed" not in message and "reconnect" not in message
    assert f"sh '{box.script}' status" in message                              # the check for a person with a terminal stays
    finish_the_setup(box)


def test_first_mcp_start_is_ready_in_the_wait_when_the_setup_is_fast(box):
    r = box.run("mcp", TOPOPT_MCP_WAIT="20")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "fake python: -P -m topopt_mcp.server\n"


# --- a broken venv, messages, paths, concurrency (review of M1) --------------------------------

def test_dangling_python_is_made_again(box):
    """A Homebrew upgrade or a killed setup can leave bin/python as a link to nothing.
    uv refuses to make a venv over it, unless --clear is given."""
    assert box.run("sync").returncode == 0
    py = box.env_dir / "venv" / "bin" / "python"
    py.unlink()
    py.symlink_to(box.env_dir / "gone" / "python3.12")
    assert box.run("status").returncode == 1
    box.log.write_text("")
    r = box.run("sync")
    assert r.returncode == 0, r.stderr
    assert box.calls()[0].startswith("venv ") and " --clear " in box.calls()[0]
    assert not py.is_symlink() and box.run("status").returncode == 0


def test_python_that_does_not_run_is_not_ready(box):
    assert box.run("sync").returncode == 0
    (box.env_dir / "venv" / "bin" / "python").write_text("#!/nonexistent/interpreter\n")
    assert box.run("status").returncode == 1


def test_messages_give_the_command_for_the_same_env_folder(box):
    """Run as printed, the command must make the environment that the server uses."""
    r = box.run("status")
    printed = [x for x in r.stderr.splitlines() if "Run:" in x][0].split("Run: ", 1)[1]
    assert printed == f"TOPOPT_ENV_DIR='{box.env_dir}' sh '{box.script}' sync"
    done = subprocess.run(printed, shell=True, capture_output=True, text=True, env=box.env, timeout=60)
    assert done.returncode == 0, done.stderr
    assert box.run("status").returncode == 0                  # the same environment is ready now
    assert not (box.root / ".plugin-env").exists()            # no second environment


def test_message_has_no_env_prefix_for_the_default_folder(box):
    env = {k: v for k, v in box.env.items() if k != "TOPOPT_ENV_DIR"}
    r = subprocess.run([*SH, str(box.script), "status"], capture_output=True, text=True, env=env)
    assert f"Run: sh '{box.script}' sync" in r.stderr and "TOPOPT_ENV_DIR" not in r.stderr


def test_status_says_that_the_setup_downloads_uv_when_uv_is_missing(box):
    if shutil.which("uv", path="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"):
        pytest.skip("a real uv is in a standard folder")
    env = no_uv_env(box)
    r = subprocess.run([*SH, str(box.script), "status"], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "uv is not installed" in r.stderr and "The setup downloads uv" in r.stderr
    assert "Run: " in r.stderr and " sync" in r.stderr                               # the usual line follows
    assert not box.env_dir.exists()                                                  # status changes nothing


def test_paths_with_spaces_and_a_quote(tmp_path, box):
    """A folder of a user can have spaces: "my 3d prints". Quote everything."""
    root = tmp_path / "my 3d prints" / "plugin root"
    shutil.copytree(box.root, root)
    env_dir = tmp_path / "data with 'quote'"
    env = dict(box.env, TOPOPT_ENV_DIR=str(env_dir))
    script = root / "scripts" / "topopt-env.sh"
    r = subprocess.run([*SH, str(script), "sync"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(env_dir / "venv" / "bin" / "python")
    assert f"--no-deps --no-build-isolation -e {root}" in box.calls()[-1]
    m = subprocess.run([*SH, str(script), "mcp"], capture_output=True, text=True, env=env)
    assert m.stdout == "fake python: -P -m topopt_mcp.server\n"
    broken = dict(env, TOPOPT_ENV_DIR=str(tmp_path / "other 'dir'"))
    printed = subprocess.run([*SH, str(script), "status"], capture_output=True, text=True, env=broken).stderr
    cmd = [x for x in printed.splitlines() if "Run:" in x][0].split("Run: ", 1)[1]
    assert subprocess.run(cmd, shell=True, capture_output=True, text=True, env=broken, timeout=60).returncode == 0


def test_uv_runs_in_the_plugin_folder_and_the_env_dir_is_made_absolute(box, tmp_path):
    """A uv.toml or [tool.uv] in the folder of the user must not change the install."""
    user = tmp_path / "user project"
    user.mkdir()
    r = subprocess.run([*SH, str(box.script), "sync"], capture_output=True, text=True, cwd=user,
                       env=dict(box.env, TOPOPT_ENV_DIR="relative-env"))
    assert r.returncode == 0, r.stderr
    pwds = Path(str(box.log) + ".pwd").read_text().splitlines()
    assert pwds and set(pwds) == {str(box.root)}
    assert (user / "relative-env" / "venv" / ".topopt-plugin-stamp").is_file()
    assert r.stdout.strip() == str(user / "relative-env" / "venv" / "bin" / "python")


def test_two_starts_at_once_make_one_environment(box):
    procs = [subprocess.Popen([*SH, str(box.script), "mcp"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=dict(box.env, FAKE_UV_SLEEP="1", TOPOPT_MCP_WAIT="25"))
             for _ in range(2)]
    outs = [p.communicate(timeout=60) for p in procs]
    assert [p.returncode for p in procs] == [0, 0], outs
    assert all(o[0] == "fake python: -P -m topopt_mcp.server\n" for o in outs)
    assert len(box.calls()) == 3                              # one setup, not two
    log = (box.env_dir / "setup.log").read_text()
    assert log.count("step 1 of 3") == 1 and "ready after" in log   # the second start did not erase the log


def test_a_failed_step_leaves_no_stamp_and_the_next_start_resumes(box):
    r = box.run("sync", FAKE_UV_FAIL_LINK="1")
    assert r.returncode == 1 and "Run it again to resume" in r.stderr
    assert f"TOPOPT_ENV_DIR='{box.env_dir}'" in r.stderr
    assert not (box.env_dir / "venv" / ".topopt-plugin-stamp").exists()
    assert not (box.env_dir / ".setup-lock").exists()
    box.log.write_text("")
    assert box.run("sync").returncode == 0
    calls = box.calls()
    assert not any(c.startswith("venv") for c in calls) and len(calls) == 2   # steps 2 and 3 only


def test_a_stale_lock_of_a_dead_process_is_taken(box):
    dead = subprocess.run([*SH, "-c", "echo $$"], capture_output=True, text=True).stdout.strip()
    lock = box.env_dir / ".setup-lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(dead + "\n")
    r = box.run("sync")
    assert r.returncode == 0, r.stderr
    assert not lock.exists()
    assert not (box.env_dir / ".setup-lock.break").exists()   # the mutex of the removal is gone too


# A fake `cat` that counts the reads of the pid in the lock folder and stops chosen reads (FAKE_CAT_HOLD_AT="2 3"), after it
# printed the pid. The test lets a read go by writing the file go.<n> in FAKE_CAT_DIR (held.<n> says that the read waits). So the
# test makes the interleavings of the race of the L1a review (m-4) by hand, and it does not depend on timing. Only the process
# that has FAKE_CAT_DIR counts and stops. The reads of a waiter that finds a dead pid, in order:
#   1 the owner (outer check)   2 the same pid again (outer check, after `kill -0` failed)
#   3 the owner (inside the mutex)   4 the same pid again (inside the mutex, after `kill -0` failed)
FAKE_CAT = """#!/bin/sh
case "$1" in
  */.setup-lock/pid)
    if [ -n "$FAKE_CAT_DIR" ]; then
      n=$(( $(/bin/cat "$FAKE_CAT_DIR/count" 2>/dev/null || echo 0) + 1 ))
      echo "$n" > "$FAKE_CAT_DIR/count"
      /bin/cat "$@"
      case " $FAKE_CAT_HOLD_AT " in *" $n "*)
        : > "$FAKE_CAT_DIR/held.$n"
        i=0; while [ ! -e "$FAKE_CAT_DIR/go.$n" ] && [ "$i" -lt 1200 ]; do sleep 0.05; i=$((i + 1)); done ;;
      esac
      exit 0
    fi ;;
esac
exec /bin/cat "$@"
"""


def lock_installs(box):
    return [c for c in box.calls() if c.startswith("pip install") and " -r " in c]


class Race:
    """A stale lock (the pid of a process that ended), a fake `cat` that can hold reads of the pid, and waiters (`sync`)."""

    def __init__(self, box, tmp_path):
        self.box = box
        self.tmp = tmp_path
        self.ctl = tmp_path / "ctl"
        self.ctl.mkdir()
        fake = box.bindir / "cat"
        fake.write_text(FAKE_CAT)
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        self.dead = subprocess.run([*SH, "-c", "echo $$"], capture_output=True, text=True).stdout.strip()
        self.lock = box.env_dir / ".setup-lock"
        self.mutex = box.env_dir / ".setup-lock.break"
        self.lock.mkdir(parents=True)
        (self.lock / "pid").write_text(self.dead + "\n")
        self.procs = []

    def start(self, name, hold_at=None, hold_install=None, **extra):
        """A `sync` that stops at the reads `hold_at` ("2 3") and/or in its lock install (a file: it waits until the file exists)."""
        env = dict(self.box.env, **extra)
        if hold_at:
            env.update(FAKE_CAT_DIR=str(self.ctl), FAKE_CAT_HOLD_AT=hold_at)
        if hold_install:
            env.update(FAKE_UV_HOLD=str(hold_install))
        p = subprocess.Popen([*SH, str(self.box.script), "sync"], env=env, stdout=subprocess.DEVNULL,
                             stderr=open(self.tmp / f"{name}.err", "w"), text=True)
        self.procs.append(p)
        return p

    def err(self, name):
        return (self.tmp / f"{name}.err").read_text()

    def pid_in_lock(self):
        f = self.lock / "pid"
        return f.read_text().strip() if f.exists() else None

    def held(self, n):
        return wait_for((self.ctl / f"held.{n}").exists, 20)

    def release(self, n):
        (self.ctl / f"go.{n}").write_text("")

    def finish(self):
        """Let everything go and wait for the processes (also when a test failed, so that nothing stays behind)."""
        for f in (self.box.hold, self.tmp / "holdY"):
            f.write_text("")
        for n in range(1, 9):
            self.release(n)
        for p in self.procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


@pytest.fixture
def race(box, tmp_path):
    r = Race(box, tmp_path)
    yield r
    r.finish()


def test_a_late_waiter_does_not_remove_the_new_lock_of_the_waiter_that_broke_the_stale_one(box, race):
    """Waiters A and B start on a stale lock. B has checked it (reads 1 and 2: the dead pid, twice) and stops. A removes the
    stale lock, takes the lock and runs its install (held by the fake uv). Then B goes on. It used to remove the NEW lock of A
    (`rm -rf` on the name) and take the lock itself: two setups at once. Now B checks the owner again inside the mutex (read 3),
    finds that A is alive and waits."""
    b = race.start("b", hold_at="2")
    assert race.held(2), "B did not check the lock"
    a = race.start("a", hold_install=box.hold)
    assert wait_for(box.held.exists, 20), "A did not reach its install"
    assert race.pid_in_lock() == str(a.pid)                       # A holds the lock, and its install is held by the fake uv
    race.release(2)                                               # B goes on with the pid that it read before
    assert wait_for(lambda: "another setup is running" in race.err("b") or b.poll() is not None, 20)
    assert len(lock_installs(box)) == 1, f"two setups at once: {box.calls()}"
    assert b.poll() is None and race.pid_in_lock() == str(a.pid)  # B waits, the lock is still A's
    race.finish()
    assert a.returncode == 0 and b.returncode == 0
    assert len(lock_installs(box)) == 1 and len(venv_calls(box)) == 1
    assert not race.lock.exists() and not race.mutex.exists()


def test_a_waiter_does_not_remove_the_lock_that_a_new_setup_took_after_the_owner_ended(box, race):
    """The window inside the mutex (F3 review m1). A holds the lock. B goes into the mutex and reads pid A (alive: read 3), and
    stops before `kill -0`. Now A ends (its install fails, it removes its own lock), and a new setup Y takes the lock and holds
    its install. B goes on: `kill -0 A` fails. It used to call the lock stale and remove the lock of Y. Now it reads the pid
    again (read 4), finds Y, and waits."""
    b = race.start("b", hold_at="2 3")
    assert race.held(2), "B did not check the lock"                # reads 1 and 2: the dead pid; B is not in the mutex yet
    a = race.start("a", hold_install=box.hold, FAKE_UV_FAIL_LOCK="error: network (fake)")
    assert wait_for(box.held.exists, 20), "A did not reach its install"
    assert race.pid_in_lock() == str(a.pid)
    race.release(2)
    assert race.held(3), "B did not read the pid inside the mutex"  # it printed pid A, which is alive
    assert race.mutex.exists()
    box.hold.write_text("")                                       # A's install goes on and fails: A removes its lock and ends
    assert a.wait(timeout=60) == 1 and not race.lock.exists()
    y = race.start("y", hold_install=race.tmp / "holdY")
    assert wait_for((race.tmp / "holdY.held").exists, 20), "Y did not reach its install"
    assert race.pid_in_lock() == str(y.pid)
    race.release(3)                                               # B goes on: `kill -0 A` fails
    assert wait_for(lambda: "another setup is running" in race.err("b") or b.poll() is not None, 20)
    assert len(lock_installs(box)) == 2, f"B ran a setup while Y held the lock: {box.calls()}"     # the installs of A and Y
    assert b.poll() is None and race.pid_in_lock() == str(y.pid)
    race.finish()
    assert y.returncode == 0 and b.returncode == 0
    assert len(lock_installs(box)) == 2 and not race.lock.exists() and not race.mutex.exists()


def test_only_one_waiter_at_a_time_checks_a_stale_lock(box, race):
    """The mutual exclusion. B is inside the mutex (its read 3 stops). A starts on the same stale lock. A must not check it
    and take the lock while B is in the mutex: it waits. Without the mutex A would remove the lock, take it and run its
    install while B still holds the pid that it read, and B would then remove the lock of A."""
    b = race.start("b", hold_at="3")
    assert race.held(3) and race.mutex.exists(), "B is not inside the mutex"
    a = race.start("a", hold_install=box.hold)
    assert wait_for(lambda: "Waiting for it" in race.err("a") or box.held.exists(), 20)
    assert "Waiting for it" in race.err("a") and not box.held.exists(), "A took the lock while B was inside the mutex"
    assert race.pid_in_lock() == race.dead and lock_installs(box) == []
    race.release(3)                                               # B finishes the removal and takes the lock
    race.finish()
    assert a.returncode == 0 and b.returncode == 0
    assert len(lock_installs(box)) == 1 and len(venv_calls(box)) == 1
    assert not race.lock.exists() and not race.mutex.exists()


# A fake `rm` for the trap tests: it does the real `rm`, and after a removal of the mutex (the name ends in .setup-lock.break) it
# counts the call (removed.<n> in FAKE_RM_DIR) and stops until the test writes go.<n>. Only a process that has FAKE_RM_DIR stops.
FAKE_RM = """#!/bin/sh
/bin/rm "$@"
rc=$?
case "$*" in
  *.setup-lock.break)
    if [ -n "$FAKE_RM_DIR" ]; then
      n=$(( $(/bin/cat "$FAKE_RM_DIR/count" 2>/dev/null || echo 0) + 1 ))
      echo "$n" > "$FAKE_RM_DIR/count"
      : > "$FAKE_RM_DIR/removed.$n"
      i=0; while [ ! -e "$FAKE_RM_DIR/go.$n" ] && [ "$i" -lt 1200 ]; do sleep 0.05; i=$((i + 1)); done
    fi ;;
esac
exit $rc
"""


def test_a_signal_inside_the_mutex_removes_the_mutex_and_ends_the_waiter(box, race):
    """The trap of break_stale_lock (F3 review n1). B is inside the mutex (its read 3 stops) and gets SIGTERM, as a client ends a
    server. The mutex must not stay: a mutex that stays blocks every later removal of a stale lock. B ends with 143 and does not
    remove the lock. The next sync then removes the stale lock and sets up."""
    b = race.start("b", hold_at="3")
    assert race.held(3) and race.mutex.exists(), "B is not inside the mutex"
    b.send_signal(signal.SIGTERM)                                 # the shell runs its trap when the stopped `cat` ends
    race.release(3)
    assert b.wait(timeout=30) == 143, race.err("b")
    assert not race.mutex.exists(), "the mutex stayed after the signal"
    assert race.pid_in_lock() == race.dead and lock_installs(box) == []     # the removal did not happen; no setup ran
    r = box.run("sync")
    assert r.returncode == 0, r.stderr
    assert len(lock_installs(box)) == 1 and not race.lock.exists() and not race.mutex.exists()


def test_a_signal_after_the_removal_of_the_mutex_does_not_remove_the_mutex_of_the_next_waiter(box, race):
    """F3 review n1. B removes its mutex, and the next waiter makes its own at once. If B gets a signal now, a trap that is still
    set would remove that new mutex (the one of the next waiter, which is then not protected). The trap is cleared BEFORE the
    removal, so a signal in this gap removes nothing. A fake `rm` stops B right after it removed the mutex."""
    rm = box.bindir / "rm"
    rm.write_text(FAKE_RM)
    rm.chmod(rm.stat().st_mode | stat.S_IEXEC)
    b = race.start("b", FAKE_RM_DIR=str(race.ctl))
    assert wait_for((race.ctl / "removed.1").exists, 20), "B did not remove its mutex"
    assert not race.mutex.exists()
    race.mutex.mkdir()                                            # the mutex of the next waiter
    (race.mutex / "next-waiter").write_text("")
    b.send_signal(signal.SIGTERM)
    for n in (1, 2):                                              # 2: the removal that a trap would do
        (race.ctl / f"go.{n}").write_text("")
    b.wait(timeout=30)
    assert (race.mutex / "next-waiter").exists(), "the signal of B removed the mutex of the next waiter"
    assert lock_installs(box) == []


def test_a_mutex_that_stays_blocks_the_removal_and_the_doctor_gives_the_way_out(box, tmp_path):
    """A kill that cannot be trapped inside the removal leaves the mutex. The waiter then waits, with no time limit (no healing
    here, by decision). The waiter says what it waits for, the doctor says that the mutex blocks the removal of the stale lock
    and gives the command that removes the mutex first, and that command lets the next sync go on (F3 review m2)."""
    dead = subprocess.run([*SH, "-c", "echo $$"], capture_output=True, text=True).stdout.strip()
    lock, mutex = box.env_dir / ".setup-lock", box.env_dir / ".setup-lock.break"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(dead + "\n")
    mutex.mkdir()
    err = tmp_path / "waiter.err"
    p = subprocess.Popen([*SH, str(box.script), "sync"], env=box.env, stdout=subprocess.DEVNULL, stderr=open(err, "w"), text=True)
    try:
        assert wait_for(lambda: "is gone" in err.read_text(), 20), err.read_text()
        said = err.read_text()
        assert f"the process that made the lock (pid {dead}) is gone" in said and f"the mutex '{mutex}'" in said
        assert f"rm -rf '{mutex}'" in said and "another setup is running" not in said     # it does not call a dead pid "running"
        r = box.run("doctor")
        assert r.returncode == 1
        rows = doctor_rows(r.stdout)
        assert rows["setup lock"].startswith(f"pid {dead} is NOT running (a stale lock, but the mutex .setup-lock.break is there:")
        assert "the next sync removes it" not in r.stdout
        assert rows["lock mutex"].startswith(f"{mutex} is there") and f"rm -rf '{mutex}'" in rows["lock mutex"]
        last = r.stdout.rstrip().splitlines()[-1].strip()
        assert last == f"rm -rf '{mutex}' && TOPOPT_ENV_DIR='{box.env_dir}' sh '{box.script}' sync"      # the mutex first, not sync alone
        assert "left the mutex" in r.stdout and "would wait with no end" in r.stdout
        assert p.poll() is None and lock.exists() and box.calls() == []     # it waits; no install
        done = subprocess.run(last, shell=True, capture_output=True, text=True, env=box.env, timeout=60)    # the printed command
        assert done.returncode == 0, done.stderr
        assert p.wait(timeout=30) == 0
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()
    assert not lock.exists() and not mutex.exists() and len(lock_installs(box)) == 1


def test_the_mutex_is_in_the_next_command_only_when_it_blocks_a_stale_lock(box):
    mutex = box.env_dir / ".setup-lock.break"
    sync = f"TOPOPT_ENV_DIR='{box.env_dir}' sh '{box.script}' sync"
    mutex.mkdir(parents=True)                                                      # a mutex and no lock: sync does not need it
    r = box.run("doctor")
    assert "lock mutex" in doctor_rows(r.stdout) and r.stdout.rstrip().splitlines()[-1].strip() == sync
    other = subprocess.Popen(["sleep", "300"])                                    # a mutex and a lock of a live pid
    try:
        lock = box.env_dir / ".setup-lock"
        lock.mkdir()
        (lock / "pid").write_text(f"{other.pid}\n")
        r = box.run("doctor")
        assert r.stdout.rstrip().splitlines()[-1].strip() == sync and "rm -rf" not in r.stdout.rstrip().splitlines()[-1]
    finally:
        other.kill()
        other.wait()
    mutex.rmdir()                                                                  # a stale lock and no mutex: sync removes it
    (lock / "pid").write_text("999999999\n")
    assert box.run("doctor").stdout.rstrip().splitlines()[-1].strip() == sync
    mutex.mkdir()                                                                  # a stale lock and a mutex: the mutex goes first
    r = box.run("doctor")
    assert r.returncode == 1 and r.stdout.rstrip().splitlines()[-1].strip() == f"rm -rf '{mutex}' && {sync}"
    shutil.rmtree(lock)
    assert box.run("sync").returncode == 0
    lock.mkdir()                                                                   # the same two, and a READY environment
    (lock / "pid").write_text("999999999\n")
    r = box.run("doctor")
    assert r.returncode == 0 and r.stdout.rstrip().endswith("Result: ready. Nothing to run.")     # nothing is waiting for the lock


def test_windows_gets_one_clear_line(box):
    uname = box.bindir / "uname"
    uname.write_text("#!/bin/sh\necho MINGW64_NT-10.0-19045\n")
    uname.chmod(uname.stat().st_mode | stat.S_IEXEC)
    for cmd in ("status", "sync", "mcp", "doctor"):
        r = box.run(cmd)
        assert r.returncode == 1 and r.stdout == ""
        lines = r.stderr.strip().splitlines()
        assert len(lines) == 1 and "Windows is not supported" in lines[0] and "Linux x86_64" in lines[0]
    assert box.calls() == []


# --- an x86_64 Python on a Mac (an Intel Mac, or Rosetta on Apple silicon): best effort, V5 --------------------

NO_BUILD = "--only-binary numba --only-binary llvmlite --only-binary cryptography"


def fake_mac_tools(box, machine="x86_64", sysctl=None):
    """Put a fake `uname` (Darwin, `machine`) and a fake `sysctl` on PATH. `sysctl` maps a key to its text."""
    uname = box.bindir / "uname"
    uname.write_text(f'#!/bin/sh\ncase "$*" in "-m") echo {machine};; "-s -m") echo Darwin {machine};; *) echo Darwin;; esac\n')
    cases = "".join(f'  {k}) echo {v};;\n' for k, v in (sysctl or {}).items())
    sc = box.bindir / "sysctl"
    sc.write_text(f'#!/bin/sh\ncase "$2" in\n{cases}  *) exit 1;;\nesac\n')
    for f in (uname, sc):
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    box.env["TOPOPT_SYSCTL"] = str(sc)   # the launcher calls /usr/sbin/sysctl by its full path, not the one on PATH


def lock_install_call(box, last=False):
    calls = [c for c in box.calls() if c.startswith("pip install") and " -r " in c]
    return calls[-1] if last else calls[0]


def venv_calls(box):
    return [c for c in box.calls() if c.startswith("venv")]


def test_the_x86_lock_exists_and_differs_from_the_main_lock_in_four_pins():
    def pins(name):
        lines = (REPO / name).read_text().splitlines()
        return dict(ln.split("==") for ln in lines if ln and not ln.startswith("#"))
    main, x86 = pins("requirements.lock"), pins("requirements-macos-x86_64.lock")
    # pyopencl has no requirement for an Intel Mac (the marker in pyproject.toml), so the Intel lock lacks it and the three
    # packages that only it needs; the main lock is flat (it has no markers) and holds them
    assert set(main) - set(x86) == {"pyopencl", "pytools", "platformdirs", "siphash24"} and set(x86) <= set(main)
    assert {k for k in x86 if main[k] != x86[k]} == {"numpy", "numba", "llvmlite", "cryptography"}


def test_intel_mac_uses_its_own_lock_and_gets_the_best_effort_facts(box):
    fake_mac_tools(box, sysctl={"hw.optional.arm64": 0})   # an Intel Mac: not translated, the CPU is not arm64
    r = box.run("sync", FAKE_PY_MACHINE="x86_64")
    assert r.returncode == 0, r.stderr
    assert "runs Python as x86_64 (intel). This is best effort" in r.stderr and "not on a Mac" in r.stderr
    assert "lock-macos-x86_64" in r.stderr and "numba==0.62.1" in r.stderr and "numpy==2.3.5" in r.stderr   # read from the file
    assert "Solver auto gives amg" in r.stderr and "TOPOPT_ALLOW_SOURCE_BUILD=1" in r.stderr
    assert "It has no native arm64 Python" in r.stderr and "rm -rf" not in r.stderr
    assert lock_install_call(box).endswith(f"{NO_BUILD} -r {box.root}/requirements-macos-x86_64.lock")
    assert "pins=lock-macos-x86_64" in (box.env_dir / "venv" / ".topopt-plugin-stamp").read_text()
    assert r.stdout.strip().endswith("python")   # stdout stays the path only
    assert venv_calls(box) == [f"venv --python {LAUNCHER_PY} --managed-python --clear {box.env_dir / 'venv'}"]


def test_rosetta_gets_the_native_python_commands(box):
    fake_mac_tools(box, sysctl={"sysctl.proc_translated": "1"})
    r = box.run("sync", FAKE_PY_MACHINE="x86_64")
    assert "(rosetta)" in r.stderr and "native arm64 Python is the fast path" in r.stderr
    assert "Open using Rosetta" in r.stderr and f"Check uv: file '{box.bindir}/uv'" in r.stderr     # the uv in use, by its full path
    assert f"rm -rf '{box.env_dir / 'venv'}'" in r.stderr or f"rm -rf {box.env_dir / 'venv'}" in r.stderr
    assert f"Run: TOPOPT_ENV_DIR='{box.env_dir}' sh '{box.script}' sync" in r.stderr
    assert "-requirements-macos-x86_64.lock" not in r.stderr and lock_install_call(box).endswith("requirements-macos-x86_64.lock")


def test_python_decides_first_a_rosetta_shell_with_an_arm64_python_is_not_an_x86_mac(box):
    fake_mac_tools(box, machine="x86_64", sysctl={"sysctl.proc_translated": "1"})   # a Rosetta shell
    r = box.run("sync", FAKE_PY_MACHINE="arm64")                                      # and a native Python
    assert r.returncode == 0, r.stderr
    assert "best effort" not in r.stderr and "x86_64 (" not in r.stderr
    assert lock_install_call(box) == f"pip install --python {box.env_dir / 'venv' / 'bin' / 'python'} -r {box.root}/requirements.lock"


def test_the_shell_decides_if_the_python_does_not_run(box):
    fake_mac_tools(box, sysctl={})   # the fake Python prints no machine type
    r = box.run("sync")
    assert "this Mac runs Python as x86_64" in r.stderr
    assert lock_install_call(box).endswith("requirements-macos-x86_64.lock")


def test_an_old_x86_environment_is_made_again_by_a_native_shell(box):
    """The Rosetta steps do not loop: run 1 had an x86_64 Python; in run 2 the shell is native, so the venv is made again."""
    fake_mac_tools(box, machine="x86_64", sysctl={"sysctl.proc_translated": "1"})
    assert box.run("sync", FAKE_UV_VENV_MACHINE="x86_64").returncode == 0
    assert "pins=lock-macos-x86_64" in (box.env_dir / "venv" / ".topopt-plugin-stamp").read_text()
    (box.env_dir / "venv" / ".topopt-plugin-stamp").unlink()   # the user follows the steps: stamp and uv are fresh
    box.log.write_text("")
    fake_mac_tools(box, machine="arm64", sysctl={"sysctl.proc_translated": "0", "hw.optional.arm64": "1"})
    r = box.run("sync", FAKE_UV_VENV_MACHINE="arm64")
    assert r.returncode == 0, r.stderr
    assert "step 1 of 3: make the Python" in r.stderr and "environment again (the environment has an x86_64 Python" in r.stderr and "best effort" not in r.stderr
    assert len(venv_calls(box)) == 1 and "--clear" in venv_calls(box)[0]   # run 2 only (the log was cleared)
    assert lock_install_call(box) == f"pip install --python {box.env_dir / 'venv' / 'bin' / 'python'} -r {box.root}/requirements.lock"
    assert "pins=lock\n" in (box.env_dir / "venv" / ".topopt-plugin-stamp").read_text()


def test_a_native_shell_with_an_x86_uv_makes_the_environment_once_and_says_what_to_do(box):
    fake_mac_tools(box, machine="x86_64", sysctl={"sysctl.proc_translated": "1"})
    assert box.run("sync", FAKE_UV_VENV_MACHINE="x86_64").returncode == 0
    (box.env_dir / "venv" / ".topopt-plugin-stamp").unlink()
    box.log.write_text("")
    fake_mac_tools(box, machine="arm64", sysctl={"sysctl.proc_translated": "0", "hw.optional.arm64": "1"})
    r = box.run("sync", FAKE_UV_VENV_MACHINE="x86_64")   # the uv on PATH is x86_64: the new Python is x86_64 too
    assert r.returncode == 0, r.stderr
    assert len(venv_calls(box)) == 1                      # once, no loop in one run
    assert "(rosetta)" in r.stderr and f"Check uv: file '{box.bindir}/uv'" in r.stderr
    assert lock_install_call(box).endswith("requirements-macos-x86_64.lock")


def test_source_build_override(box):
    fake_mac_tools(box, sysctl={})
    r = box.run("sync", FAKE_PY_MACHINE="x86_64", TOPOPT_ALLOW_SOURCE_BUILD="1")
    assert r.returncode == 0, r.stderr
    assert lock_install_call(box) == f"pip install --python {box.env_dir / 'venv' / 'bin' / 'python'} -r {box.root}/requirements-macos-x86_64.lock"


def test_the_failed_install_on_an_x86_mac_says_so_and_the_fallback_keeps_the_no_build_flags(box):
    fake_mac_tools(box, sysctl={})
    r = box.run("sync", FAKE_PY_MACHINE="x86_64", FAKE_UV_FAIL_LOCK="error: no matching wheel",
                FAKE_UV_FAIL_PYPROJECT="error: no matching wheel")
    assert r.returncode == 1
    assert "On this x86_64 Mac the install failed (best effort)" in r.stderr
    fallback = [c for c in box.calls() if f"{box.root}[mcp] " in c]
    assert len(fallback) == 1 and NO_BUILD in fallback[0]
    assert fallback[0].endswith(f"[mcp] {lock_pin('setuptools', 'requirements-macos-x86_64.lock')}")   # the build backend of the Intel lock


def test_the_fallback_on_an_x86_mac_takes_the_build_backend_pin_of_the_intel_lock(box):
    """#397: the pin comes from the lock of this platform. The two real locks pin the same setuptools, so the box copy of the Intel lock
    gets another version (the fake uv takes any)."""
    intel = box.root / "requirements-macos-x86_64.lock"
    text = intel.read_text()
    assert text.count("setuptools==") == 1
    intel.write_text(re.sub(r"^setuptools==\S+$", "setuptools==99.0.1", text, flags=re.M))
    fake_mac_tools(box, sysctl={})
    r = box.run("sync", FAKE_PY_MACHINE="x86_64", FAKE_UV_FAIL_LOCK="error: no matching wheel")
    assert r.returncode == 0, r.stderr
    fallback = [c for c in box.calls() if f"{box.root}[mcp] " in c]
    assert len(fallback) == 1 and fallback[0].endswith("[mcp] setuptools==99.0.1")


def real_uv():
    """The real uv: the file in the environment variable UV, else the first `uv` on PATH. None if there is none."""
    return os.environ.get("UV") or shutil.which("uv")


def flag_words(call):
    """The `--flag` words of one logged uv call."""
    return {w for w in call.split() if w.startswith("--")}


def test_every_flag_that_the_x86_mac_install_passes_to_uv_exists_in_the_real_uv(box):
    """A fake uv takes any flag, so the tests above cannot see a flag that the real uv rejects. This is the defect of
    `--no-build-package`: `uv pip install` does not have it. Read the flags that the launcher passes to `uv pip install`
    on an x86_64 Mac, and check each of them against `uv pip install --help` of the real uv."""
    uv = real_uv()
    if uv is None:
        pytest.skip("the real uv is not installed (put it on PATH or set the variable UV): it is the source of the flag list")
    fake_mac_tools(box, sysctl={})
    r = box.run("sync", FAKE_PY_MACHINE="x86_64", FAKE_UV_FAIL_LOCK="error: no matching wheel",
                FAKE_UV_FAIL_PYPROJECT="error: no matching wheel")   # both calls (the lock and the pyproject fallback) are logged
    assert r.returncode == 1
    installs = [c for c in box.calls() if c.startswith("pip install")]
    assert len(installs) == 2
    box.log.write_text("")
    assert box.run("sync", FAKE_PY_MACHINE="x86_64", TOPOPT_ALLOW_SOURCE_BUILD="1").returncode == 0
    with_override = set().union(*(flag_words(c) for c in box.calls() if c.startswith("pip install")))
    printed = set().union(*(flag_words(c) for c in installs)) - with_override   # the words of `no_build_flags`
    assert printed, "no_build_flags prints no flag on an x86_64 Mac: the test would check nothing"
    help_text = subprocess.run([uv, "pip", "install", "--help"], capture_output=True, text=True,
                               env=dict(os.environ, NO_COLOR="1")).stdout
    assert "--python" in help_text, "the help text of uv has no flag list: " + help_text[:200]
    missing = sorted(f for f in printed | with_override | set().union(*(flag_words(c) for c in installs))
                     if not re.search(r"(?<![-\w])" + re.escape(f) + r"(?![-\w])", help_text))
    assert not missing, f"`uv pip install --help` does not list: {missing} (flags that no_build_flags prints: {sorted(printed)})"


@pytest.mark.parametrize("sysname, machine", [("Darwin", "arm64"), ("Linux", "x86_64"), ("Linux", "aarch64")])
def test_other_platforms_get_no_note_and_the_same_install_call(box, sysname, machine):
    uname = box.bindir / "uname"
    uname.write_text(f'#!/bin/sh\ncase "$1" in -m) echo {machine};; *) echo {sysname};; esac\n')
    uname.chmod(uname.stat().st_mode | stat.S_IEXEC)
    r = box.run("sync")
    assert r.returncode == 0, r.stderr
    assert "x86_64 (" not in r.stderr and "best effort" not in r.stderr and "Rosetta" not in r.stderr
    calls = box.calls()
    assert calls[0] == f"venv --python {LAUNCHER_PY} --managed-python --clear {box.env_dir / 'venv'}"
    assert calls[1] == f"pip install --python {box.env_dir / 'venv' / 'bin' / 'python'} -r {box.root}/requirements.lock"
    assert calls[2] == f"pip install --python {box.env_dir / 'venv' / 'bin' / 'python'} --no-deps --no-build-isolation -e {box.root}"
    assert len(calls) == 3


# --- the stamp holds the checksum of the lock file that the platform uses (V5 round 1b) -------------------------

def _cksum(path):
    out = subprocess.run(["cksum"], stdin=open(path, "rb"), capture_output=True, text=True, check=True).stdout
    return " ".join(out.split()[:2])


def test_a_change_of_the_intel_lock_installs_again_on_an_x86_mac(box):
    fake_mac_tools(box, sysctl={})
    assert box.run("sync", FAKE_PY_MACHINE="x86_64").returncode == 0
    stamp = (box.env_dir / "venv" / ".topopt-plugin-stamp").read_text().splitlines()[0]
    assert f"lock={_cksum(box.root / 'requirements-macos-x86_64.lock')} " in stamp
    box.log.write_text("")
    assert box.run("sync", FAKE_PY_MACHINE="x86_64").returncode == 0 and box.calls() == []   # same files: ready
    with open(box.root / "requirements-macos-x86_64.lock", "a") as f:
        f.write("# a change of the Intel lock\n")
    assert box.run("status", FAKE_PY_MACHINE="x86_64").returncode == 1
    r = box.run("sync", FAKE_PY_MACHINE="x86_64")
    assert r.returncode == 0, r.stderr
    assert lock_install_call(box).endswith("requirements-macos-x86_64.lock")
    assert box.run("status", FAKE_PY_MACHINE="x86_64").returncode == 0
    # a change of the main lock alone does not matter on this platform
    box.log.write_text("")
    with open(box.root / "requirements.lock", "a") as f:
        f.write("# a change of the main lock\n")
    assert box.run("status", FAKE_PY_MACHINE="x86_64").returncode == 0 and box.calls() == []


def test_a_change_of_the_intel_lock_does_not_install_again_on_arm64(box):
    fake_mac_tools(box, machine="arm64", sysctl={"sysctl.proc_translated": "0", "hw.optional.arm64": "1"})
    assert box.run("sync", FAKE_PY_MACHINE="arm64").returncode == 0
    box.log.write_text("")
    with open(box.root / "requirements-macos-x86_64.lock", "a") as f:
        f.write("# a change of the Intel lock\n")
    assert box.run("status", FAKE_PY_MACHINE="arm64").returncode == 0
    assert box.run("sync", FAKE_PY_MACHINE="arm64").returncode == 0 and box.calls() == []
    with open(box.root / "requirements.lock", "a") as f:   # the main lock still counts here
        f.write("# a change of the main lock\n")
    assert box.run("status", FAKE_PY_MACHINE="arm64").returncode == 1


@pytest.mark.parametrize("darwin_arm64", [False, True])
def test_the_stamp_text_on_linux_and_arm64(box, darwin_arm64):
    """The first line is `root=<root> lock=<cksum of requirements.lock> pyproject=<cksum of pyproject.toml> python=<launcher Python>`."""
    if darwin_arm64:
        fake_mac_tools(box, machine="arm64", sysctl={"sysctl.proc_translated": "0", "hw.optional.arm64": "1"})
        assert box.run("sync", FAKE_PY_MACHINE="arm64").returncode == 0
    else:
        assert box.run("sync").returncode == 0
    first = (box.env_dir / "venv" / ".topopt-plugin-stamp").read_text().splitlines()[0]
    assert first == f"root={box.root} lock={_cksum(box.root / 'requirements.lock')} pyproject={_cksum(box.root / 'pyproject.toml')} python={LAUNCHER_PY}"


# --- #422: the launcher makes Python LAUNCHER_PYTHON; an environment of another Python is made again once -------------

def make_old_environment(box, old="3.12"):
    """An environment of an older launcher: its Python says `old`, and its stamp has no Python field."""
    assert box.run("sync").returncode == 0
    py = box.env_dir / "venv" / "bin" / "python"
    py.write_text(py.read_text().replace(f'echo "${{FAKE_PY_VERSION:-{LAUNCHER_PY}}}"', f'echo "${{FAKE_PY_VERSION:-{old}}}"'))
    lines = box.stamp.read_text().splitlines()
    lines[0] = lines[0].rsplit(" python=", 1)[0]
    box.stamp.write_text("\n".join(lines) + "\n")
    box.log.unlink()


def test_the_launcher_python_is_one_variable_in_the_launcher():
    text = (REPO / "scripts" / "topopt-env.sh").read_text()
    assert len(re.findall(r"^LAUNCHER_PYTHON=", text, re.M)) == 1
    assert not re.search(r"--python 3\.\d+", text)                      # no second place of the version in a uv call
    assert re.fullmatch(r"3\.\d+", LAUNCHER_PY)


def test_the_stamp_holds_the_python_of_the_launcher(box):
    assert box.run("sync").returncode == 0
    assert box.stamp.read_text().splitlines()[0].endswith(f" python={LAUNCHER_PY}")


def test_an_environment_of_another_python_is_made_again_once(box):
    make_old_environment(box)
    r = box.run("status")                                                  # the old stamp does not match: not ready, and the text says why
    assert r.returncode == 1 and "has Python 3.12" in r.stderr and f"makes Python {LAUNCHER_PY}" in r.stderr
    r = box.run("sync")
    assert r.returncode == 0, r.stderr
    assert (f"step 1 of 3: make the Python {LAUNCHER_PY} environment again (the environment has Python 3.12; "
            "the packages download again, this needs the network)") in r.stderr
    calls = box.calls()
    assert calls[0] == f"venv --python {LAUNCHER_PY} --managed-python --clear {box.env_dir / 'venv'}"
    assert len(calls) == 3 and "--reinstall" not in calls[1]               # a venv made in this call: step 2 and 3 as on a first start
    assert box.stamp.read_text().splitlines()[0].endswith(f" python={LAUNCHER_PY}") and not box.marker.exists()
    # A later start: ready, and no second remake.
    box.log.unlink()
    assert box.run("status").returncode == 0
    r = box.run("sync")
    assert r.returncode == 0 and box.calls() == [] and "environment again" not in r.stderr


def test_an_environment_of_the_launcher_python_with_a_current_stamp_is_not_made_again(box):
    assert box.run("sync").returncode == 0
    box.log.unlink()
    r = box.run("sync")
    assert r.returncode == 0 and box.calls() == []


def test_a_cut_remake_resumes_and_ends_ready(box):
    make_old_environment(box)
    r = box.run("sync", FAKE_UV_FAIL_LOCK="error: Request failed after 3 retries: dns error")
    assert r.returncode == 1 and "network problem" in r.stderr
    r = box.run("sync")                                                    # online again: the venv has the new Python, the marker makes step 2 again
    assert r.returncode == 0 and box.run("status").returncode == 0


def test_doctor_names_the_python_when_the_stamp_does_not_match(box):
    make_old_environment(box)
    r = box.run("doctor")
    rows = doctor_rows(r.stdout)
    assert r.returncode == 1
    assert "older launcher" in rows["stamp"] and f"makes Python {LAUNCHER_PY}" in rows["stamp"]
    assert f"this launcher makes Python {LAUNCHER_PY}" in rows["venv Python"]
    # A stamp with another Python field names that Python.
    lines = box.stamp.read_text().splitlines()
    box.stamp.write_text(lines[0] + " python=3.12\n" + lines[1] + "\n")
    assert "made for Python 3.12" in doctor_rows(box.run("doctor").stdout)["stamp"]


def test_a_tool_call_during_the_remake_names_step_1_and_the_wait_of_step_1(box):
    """Review F1. The status server reads the last `step N of 3` line of setup.log, and setup.log is appended to. The remake writes a step 1
    line, so a call during the download of the new Python gets step 1 and SETUP_RECHECK_S (30 s), not the step 3 line of the old setup."""
    make_old_environment(box)
    old = "topopt: step 1 of 3: make the Python 3.12 environment\ntopopt: step 2 of 3: install the packages\ntopopt: step 3 of 3: link the topopt code in /old/root\ntopopt: ready after 3 s (pins=lock): x\n"
    (box.env_dir / "setup.log").write_text(old)
    vhold = box.root.parent / "vhold"
    p = subprocess.Popen([*SH, str(box.script), "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=dict(box.env, FAKE_UV_HOLD_VENV=str(vhold), TOPOPT_MCP_WAIT="0"))
    try:
        assert wait_for(lambda: Path(str(vhold) + ".held").exists(), 60), "the remake did not reach uv venv"
        out, err = p.communicate("".join(line + "\n" for line in CALL), timeout=60)
        text = text_of(json.loads(out.splitlines()[0]))
        assert "Now: step 1 of 3: make the Python " in text and "step 3 of 3" not in text, text
        assert "about 30 s more" in text, text                                # SETUP_RECHECK_S, not SETUP_TAIL_S (10 s)
    finally:
        vhold.touch()
        log = box.env_dir / "setup.log"
        wait_for(lambda: log.exists() and log.read_text().count("ready after") >= 2, 60)
    assert box.run("status").returncode == 0


def real_python_wrapper(box):
    """Make `venv/bin/python` run the real Python of this test run, so that the launcher's own version query runs."""
    py = box.env_dir / "venv" / "bin" / "python"
    py.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    return f"{sys.version_info[0]}.{sys.version_info[1]}"


def test_the_version_query_of_the_launcher_gives_exactly_major_minor_with_the_real_python(box):
    """Review F3. The fake Python answered with a bare version; this runs the expression of `env_python_mm` in a real Python."""
    src = box.script.read_text()
    script = f"PY={shlex.quote(sys.executable)}\n" + shell_function(src, "env_python_mm") + "\nenv_python_mm"
    out = subprocess.run([*SH, "-c", script], capture_output=True, text=True).stdout
    assert out == f"{sys.version_info[0]}.{sys.version_info[1]}\n"
    # And in the launcher: a real Python of the launcher's version is not made again; another one is named by status and doctor.
    assert box.run("sync").returncode == 0
    mm = real_python_wrapper(box)
    box.log.unlink()
    if mm == LAUNCHER_PY:
        assert box.run("status").returncode == 0
        assert "this launcher makes" not in doctor_rows(box.run("doctor").stdout)["venv Python"]
        box.stamp.unlink()
        assert box.run("sync").returncode == 0 and not [c for c in box.calls() if c.startswith("venv ")]   # no remake for its own version
    else:
        box.stamp.unlink()                                                 # a current stamp means ready; the Python is read when the setup runs
        r = box.run("status")
        assert r.returncode == 1 and f"has Python {mm}," in r.stderr
        assert f"this launcher makes Python {LAUNCHER_PY}" in doctor_rows(box.run("doctor").stdout)["venv Python"]


def test_a_venv_python_that_runs_but_gives_no_version_is_not_made_again(box):
    """Review F4. The branch today: no remake, `status` says `not ready: <python>` (the stamp is gone), doctor adds nothing."""
    assert box.run("sync").returncode == 0
    box.stamp.unlink()
    box.log.unlink()
    r = box.run("status", FAKE_PY_VERSION="garbage")
    assert r.returncode == 1 and f"not ready: {box.env_dir / 'venv' / 'bin' / 'python'}." in r.stderr and "has Python" not in r.stderr
    r = box.run("sync", FAKE_PY_VERSION="garbage")
    assert r.returncode == 0, r.stderr
    assert not [c for c in box.calls() if c.startswith("venv ")] and "environment again" not in r.stderr
    assert "this launcher makes" not in doctor_rows(box.run("doctor", FAKE_PY_VERSION="garbage").stdout)["venv Python"]


# --- L1a: a cut setup is repaired (review M-2) --------------------------------------------------------------------

def wait_for(cond, timeout=20.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return bool(cond())


def cut_in_step_2(box, **extra):
    """Start `sync` in its own process group, wait until the fake uv is inside the install of the lock file (it has set the
    flag of a broken environment), and kill the whole group with SIGKILL (nothing can trap it, as in a kill or a power loss)."""
    env = dict(box.env, FAKE_UV_BREAK_LOCK="60", **extra)
    p = subprocess.Popen([*SH, str(box.script), "sync"], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE, text=True, start_new_session=True)
    for line in p.stderr:
        if "step 2 of 3" in line:
            break
    assert wait_for(box.broken.exists), "the fake uv did not start the install of the lock file"
    os.killpg(p.pid, signal.SIGKILL)
    p.wait(timeout=30)
    p.stderr.close()


def test_a_setup_cut_in_step_2_is_not_ready_and_the_next_sync_installs_again(box):
    """Review M-2. Before the fix, the next `sync` skipped the packages (uv saw their .dist-info), wrote a stamp, and a tool
    failed with an ImportError. Now a marker tells the next setup to use --reinstall."""
    cut_in_step_2(box)
    assert box.broken.exists() and not box.stamp.exists()             # the cut left a broken environment
    status = box.run("status")
    box.log.write_text("")
    r = box.run("sync")
    # The defect of the review: the packages must be installed again, and the broken environment must be whole.
    py = box.env_dir / "venv" / "bin" / "python"
    assert r.returncode == 0, r.stderr
    assert not any(c.startswith("venv") for c in box.calls())          # no new venv: only the install steps run again
    assert box.calls() == [f"pip install --python {py} --reinstall -r {box.root}/requirements.lock",
                           f"pip install --python {py} --reinstall --no-deps --no-build-isolation -e {box.root}"]
    assert not box.broken.exists()                                     # the fake uv wrote the files again
    assert "installing the packages again" in r.stderr and r.stdout.strip() == str(py)
    assert box.stamp.exists() and not box.marker.exists()
    assert box.run("status").returncode == 0
    box.log.write_text("")
    assert box.run("sync").returncode == 0 and box.calls() == []       # and the next start is the fast one
    # `status` right after the cut: not ready, it names the cut, and its command is the one for the same env folder.
    assert status.returncode == 1 and "cut" in status.stderr and str(box.marker) in status.stderr
    printed = [x for x in status.stderr.splitlines() if "Run:" in x][0].split("Run: ", 1)[1]
    assert printed == f"TOPOPT_ENV_DIR='{box.env_dir}' sh '{box.script}' sync"


def test_a_cut_setup_is_repaired_by_the_mcp_start_too(box):
    cut_in_step_2(box)
    r = box.run("mcp", TOPOPT_MCP_WAIT="25")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "fake python: -P -m topopt_mcp.server\n"
    assert any("--reinstall" in c and " -r " in c for c in box.calls())
    assert not box.broken.exists() and box.stamp.exists() and not box.marker.exists()


def test_a_marker_next_to_a_current_stamp_is_not_ready(box):
    """The marker goes away after the stamp. A cut between the two leaves both: the environment is not ready."""
    assert box.run("sync").returncode == 0
    box.marker.write_text("")
    r = box.run("status")
    assert r.returncode == 1 and "cut" in r.stderr
    box.log.write_text("")
    assert box.run("sync").returncode == 0
    assert all("--reinstall" in c for c in box.calls()) and len(box.calls()) == 2
    assert not box.marker.exists() and box.run("status").returncode == 0


def test_the_marker_stays_after_a_clean_uv_error_and_the_next_sync_installs_again(box):
    """A re-install from the uv cache is cheap, a missed repair is not."""
    r = box.run("sync", FAKE_UV_FAIL_LOCK="error: Failed to download `numpy`\n  Caused by: dns error: no such host")
    assert r.returncode == 1 and box.marker.exists() and not box.stamp.exists()
    box.log.write_text("")
    assert box.run("sync").returncode == 0
    calls = box.calls()
    assert not any(c.startswith("venv") for c in calls)
    assert all("--reinstall" in c for c in calls) and len(calls) == 2
    assert not box.marker.exists()


def test_the_pyproject_fallback_also_installs_again_after_a_cut(box):
    assert box.run("sync", FAKE_UV_FAIL_LINK="1").returncode == 1                     # a marker is left
    box.log.write_text("")
    r = box.run("sync", FAKE_UV_FAIL_LOCK="error: no wheel with a matching platform tag for numpy")
    assert r.returncode == 0, r.stderr
    fallback = [c for c in box.calls() if f"{box.root}[mcp] " in c]
    assert len(fallback) == 1 and "--reinstall" in fallback[0]


def test_a_reinstall_keeps_the_no_build_flags_and_the_intel_lock_on_an_x86_mac(box):
    fake_mac_tools(box, sysctl={})
    assert box.run("sync", FAKE_PY_MACHINE="x86_64", FAKE_UV_FAIL_LINK="1").returncode == 1   # a marker is left
    box.log.write_text("")
    r = box.run("sync", FAKE_PY_MACHINE="x86_64")
    assert r.returncode == 0, r.stderr
    py = box.env_dir / "venv" / "bin" / "python"
    assert lock_install_call(box) == f"pip install --python {py} --reinstall {NO_BUILD} -r {box.root}/requirements-macos-x86_64.lock"
    assert not box.marker.exists()


def test_a_new_venv_needs_no_reinstall_even_if_a_marker_was_there(box):
    """uv venv --clear deletes the folder with the marker, so a new venv installs as usual."""
    assert box.run("sync", FAKE_UV_FAIL_LINK="1").returncode == 1
    py = box.env_dir / "venv" / "bin" / "python"
    py.unlink()                                                                        # the Python is gone
    box.log.write_text("")
    assert box.run("sync").returncode == 0
    calls = box.calls()
    assert calls[0].startswith("venv ") and not any("--reinstall" in c for c in calls)


# --- L1a: the import check before the stamp ---------------------------------------------------------------------------

def test_import_check_failure_keeps_the_marker_writes_no_stamp_and_keeps_stdout_clean(box):
    r = box.run("sync", FAKE_UV_BREAK_LINK="1")
    assert r.returncode == 1
    assert r.stdout == ""
    assert "an import fails" in r.stderr and "ImportError" in r.stderr and "ndimage" in r.stderr
    assert "Run it again to resume" in r.stderr and "installs the packages again" in r.stderr
    assert "ImportError" in (box.env_dir / "last-step.log").read_text()
    assert box.marker.exists() and not box.stamp.exists() and not (box.env_dir / ".setup-lock").exists()
    assert box.run("status").returncode == 1
    # The next setup repairs it: --reinstall in step 3 writes the files again, the check passes, the stamp comes.
    box.log.write_text("")
    r = box.run("sync")
    assert r.returncode == 0, r.stderr
    assert box.calls()[-1].split("--python ", 1)[1].split(" ", 1)[1].startswith("--reinstall --no-deps --no-build-isolation -e")
    assert box.stamp.exists() and not box.marker.exists() and box.run("status").returncode == 0


def test_import_check_names_the_packages_and_runs_outside_the_plugin_folder(box):
    log = box.env_dir.parent / "py.log"
    r = box.run("sync", FAKE_PY_LOG=str(log))
    assert r.returncode == 0, r.stderr
    # " -c import numpy": on a Mac the launcher also runs `-c import platform` (the machine type of the Python).
    checks = [line for line in log.read_text().splitlines() if " -c import numpy" in line]
    assert checks == ["/ -c import numpy, scipy.ndimage, scipy.sparse.linalg, numba, yaml, topopt_mcp.server"]


def test_import_check_does_not_run_when_the_environment_is_ready(box):
    assert box.run("sync").returncode == 0
    log = box.env_dir.parent / "py2.log"
    assert box.run("sync", FAKE_PY_LOG=str(log)).returncode == 0
    assert box.run("mcp", FAKE_PY_LOG=str(log)).returncode == 0
    assert not any(" -c import numpy" in line for line in log.read_text().splitlines())    # the fast start stays fast


# --- L1a: the background setup is detached (review M-1) ----------------------------------------------------------------

def tool_found(box, name):
    return shutil.which(name, path=box.env["PATH"]) is not None


def ppid_of(pid):
    """The parent process of pid: from /proc (Linux), else from `ps`; None if neither works."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        pass
    ps = shutil.which("ps")
    if ps:
        out = subprocess.run([ps, "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if out.isdigit():
            return int(out)
    return None


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"   # a zombie is dead
    except (OSError, IndexError):
        return True


def descendants(root):
    """Every process below root (Linux, /proc): what a client that ends the process TREE walks."""
    parents = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            ppid = ppid_of(int(entry.name))
            if ppid is not None:
                parents[int(entry.name)] = ppid
    found, todo = [], [root]
    while todo:
        cur = todo.pop()
        kids = [pid for pid, ppid in parents.items() if ppid == cur]
        found += kids
        todo += kids
    return found


def wait_for_the_end_of_the_background_setup(box, timeout=60):
    """The stamp comes first, then the marker goes, then the log says "ready after". Wait for the last one."""
    log = box.env_dir / "setup.log"
    assert wait_for(lambda: log.exists() and "ready after" in log.read_text(), timeout), log.read_text() if log.exists() else "no log"


def setup_pid_of(box):
    pidfile = box.env_dir / ".setup-lock" / "pid"
    assert wait_for(lambda: pidfile.exists() and pidfile.read_text().strip(), 10), "the setup wrote no pid in the lock"
    return int(pidfile.read_text())


def start_mcp_in_its_own_group(box, method, **extra):
    """A client that starts the server in a new process group and session (Codex does). The first start is slower than the wait:
    the fake uv holds the install of the lock file until the test writes `box.hold`."""
    env = dict(box.env, TOPOPT_DETACH=method, FAKE_UV_HOLD=str(box.hold), **extra)
    return subprocess.Popen([*SH, str(box.script), "mcp"], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)


METHODS = ["setsid", "perl", "plain"]


def need_method(box, method):
    if method != "plain" and not tool_found(box, method):
        pytest.skip(f"no {method} on this machine")


@pytest.mark.parametrize("method", METHODS)
def test_the_setup_has_its_own_session_and_survives_the_end_of_the_server_group(box, method):
    """Review M-1. Codex ends the process group of the server (SIGTERM) each time it closes the server, also after `mcp` has
    exited. A setup in that group dies. A setup that is detached with setsid or perl survives. With neither tool (plain)
    the setup has the same group, and it dies: the documented limit."""
    need_method(box, method)
    p = start_mcp_in_its_own_group(box, method, TOPOPT_MCP_WAIT="1")
    out, err = p.communicate(timeout=30)
    assert p.returncode == 1 and out == "" and "not ready yet" in err
    pid = setup_pid_of(box)                                         # the lock holds the pid of the setup shell itself
    assert wait_for(box.held.exists, 10) and alive(pid)             # the setup is in step 2, held by the fake uv
    assert "the setup is running" in box.run("status").stderr       # setup_running reads that pid: it is right
    if method == "plain":
        assert os.getpgid(pid) == p.pid and os.getsid(pid) == p.pid  # the group of the server (it is its own leader)
    else:
        assert os.getpgid(pid) != p.pid and os.getsid(pid) != p.pid  # its own group and session
    try:
        os.killpg(p.pid, signal.SIGTERM)                            # what Codex does when it drops the server
    except (ProcessLookupError, PermissionError):
        pass                                                        # the group is already gone, or (macOS: EPERM) only zombies are left in it
    if method == "plain":
        assert wait_for(lambda: not alive(pid), 30), "the setup should end with the group in the plain way"
        assert not box.stamp.exists()
        box.hold.write_text("")
    else:
        assert alive(pid), "the group kill reached the setup"
        box.hold.write_text("")                                     # let the fake uv go on
        wait_for_the_end_of_the_background_setup(box)
        assert box.stamp.exists() and box.run("status").returncode == 0


@pytest.mark.parametrize("method", METHODS)
def test_the_first_start_message_names_the_start_method(box, method):
    need_method(box, method)
    p = start_mcp_in_its_own_group(box, method, TOPOPT_MCP_WAIT="1")
    out, err = p.communicate(timeout=30)
    assert f"the setup runs in the background (start method: {method})" in err and out == ""
    assert wait_for(box.held.exists, 10)
    box.hold.write_text("")
    wait_for_the_end_of_the_background_setup(box)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="the process tree is read from /proc")
@pytest.mark.parametrize("method", METHODS)
def test_a_client_that_ends_the_process_tree_does_not_find_the_setup(box, method):
    """VS Code ends the process TREE of the server. While the launcher waits, the setup must not be one of its children."""
    need_method(box, method)
    p = start_mcp_in_its_own_group(box, method, TOPOPT_MCP_WAIT="20")
    try:
        pid = setup_pid_of(box)
        assert wait_for(box.held.exists, 10)                        # the setup is in step 2, held by the fake uv
        assert p.poll() is None                                     # the launcher is still waiting
        assert ppid_of(pid) != p.pid, "the setup is a child of the launcher"
        tree = descendants(p.pid)
        assert pid not in tree
        for child in tree:                                          # end the tree: the children, then the launcher
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
        p.terminate()
        p.communicate(timeout=30)
        assert alive(pid), "the end of the tree reached the setup"
        box.hold.write_text("")
        wait_for_the_end_of_the_background_setup(box)
        assert box.stamp.exists()
    finally:
        box.hold.write_text("")
        if p.poll() is None:
            p.kill()
            p.communicate()


@pytest.mark.parametrize("method", ["setsid", "perl"])
def test_the_setup_survives_a_group_kill_while_the_launcher_still_waits(box, method):
    """The generic SDK client and a Codex that quits during the wait end the process group while the server process is alive.
    The group kill reaches the launcher (and its `sleep`), and not the setup."""
    need_method(box, method)
    p = start_mcp_in_its_own_group(box, method, TOPOPT_MCP_WAIT="20")
    try:
        pid = setup_pid_of(box)
        assert wait_for(box.held.exists, 10) and p.poll() is None   # the setup is in step 2, the launcher waits
        os.killpg(p.pid, signal.SIGTERM)
        p.communicate(timeout=30)                                   # the launcher ends with its group
        try:
            os.killpg(p.pid, signal.SIGKILL)                        # as Codex does after 2 s
        except (ProcessLookupError, PermissionError):
            pass                                                    # macOS gives EPERM for a group with only zombies left (the `sleep` of the launcher)
        assert alive(pid), "the group kill reached the setup"
        box.hold.write_text("")
        wait_for_the_end_of_the_background_setup(box)
        assert box.stamp.exists() and box.run("status").returncode == 0
    finally:
        box.hold.write_text("")
        if p.poll() is None:
            p.kill()
            p.communicate()


def test_a_failed_setsid_of_the_perl_way_is_written_to_the_setup_log(box):
    """POSIX::setsid() can fail (a process that leads its group). The setup then runs in the session of the launcher, and the
    log says why. A fake perl makes POSIX::setsid fail; every other part is the real perl."""
    real = shutil.which("perl", path="/usr/bin:/bin")
    if not real:
        pytest.skip("no perl on this machine")
    fake = box.bindir / "perl"
    fake.write_text("#!/bin/sh\nexec " + shlex.quote(real) + " -MPOSIX -e "
                    + shlex.quote('BEGIN { no warnings "redefine"; *POSIX::setsid = sub { $! = 1; return undef } }') + ' "$@"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    p = start_mcp_in_its_own_group(box, "perl", TOPOPT_MCP_WAIT="1")
    try:
        p.communicate(timeout=30)
        setup_pid_of(box)
        assert wait_for(box.held.exists, 10)
        assert "topopt: setsid failed: Operation not permitted" in (box.env_dir / "setup.log").read_text()
    finally:
        box.hold.write_text("")
    wait_for_the_end_of_the_background_setup(box)                   # the setup went on in spite of the failure
    assert box.stamp.exists()


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="the process tree is read from /proc")
def test_the_setup_of_the_old_form_is_a_child_of_the_launcher(box):
    """The measured reason for the subshell form: a background job (`nohup setsid sh ... &`, no subshell) keeps the launcher as
    its parent while the launcher waits, so a tree kill finds it. This test runs the old form by hand."""
    if not tool_found(box, "setsid"):
        pytest.skip("no setsid on this machine")
    script = box.bindir / "old-form.sh"
    script.write_text('#!/bin/sh\nnohup setsid sleep 5 >/dev/null 2>&1 </dev/null &\necho $!\nsleep 2\n')
    p = subprocess.Popen([*SH, str(script)], stdout=subprocess.PIPE, text=True, start_new_session=True)
    sleeper = int(p.stdout.readline())
    assert wait_for(lambda: ppid_of(sleeper) is not None)
    assert ppid_of(sleeper) == p.pid
    p.communicate(timeout=30)
    os.kill(sleeper, signal.SIGKILL)


def test_the_detach_method_follows_the_tools_and_the_knob(box):
    """doctor says which way this shell will use. The default is setsid, then perl, then plain."""
    def method_of(env_extra=None, env=None):
        r = subprocess.run([*SH, str(box.script), "doctor"], capture_output=True, text=True, timeout=60,
                           env=dict(env or box.env, **(env_extra or {})))
        line = [x for x in r.stdout.splitlines() if x.strip().startswith("start method")][0]
        return line.split("start method", 1)[1].split()[0]
    for method in METHODS:
        if method == "plain" or tool_found(box, method):
            assert method_of({"TOPOPT_DETACH": method}) == method
    default = "setsid" if tool_found(box, "setsid") else "perl" if tool_found(box, "perl") else "plain"
    assert method_of() == default


def tools_farm(box, names):
    """A PATH folder with links to the named tools only (and the fake uv), so a tool can be missing."""
    farm = box.bindir.parent / "farm"
    farm.mkdir(exist_ok=True)
    for name in names:
        target = shutil.which(name, path="/usr/bin:/bin")
        if target:
            (farm / name).symlink_to(target)
    (farm / "uv").symlink_to(box.bindir / "uv")
    return farm


FARM_TOOLS = ["dirname", "uname", "sed", "cat", "cut", "cksum", "head", "tail", "basename", "grep", "tr"]


def test_without_setsid_the_launcher_uses_perl_and_without_both_it_uses_plain(box):
    for folder in ("/usr/local/bin", "/opt/homebrew/bin"):     # the launcher adds these folders to PATH
        if Path(folder).is_dir() and any(shutil.which(t, path=folder) for t in ("setsid", "perl")):
            pytest.skip(f"setsid or perl in {folder}")
    cases = [(FARM_TOOLS + ["perl"], "perl"), (FARM_TOOLS, "plain")]
    if tool_found(box, "setsid"):
        cases.insert(0, (FARM_TOOLS + ["setsid", "perl"], "setsid"))
    for names, expected in cases:
        if expected == "perl" and not shutil.which("perl", path="/usr/bin:/bin"):
            continue
        farm = tools_farm(box, names)
        env = dict(box.env, PATH=str(farm), HOME=str(box.root))
        r = subprocess.run([*SH_ABS, str(box.script), "doctor"], capture_output=True, text=True, timeout=60, env=env)
        line = [x for x in r.stdout.splitlines() if x.strip().startswith("start method")][0]
        assert line.split("start method", 1)[1].split()[0] == expected, r.stdout
        shutil.rmtree(farm)


def test_a_knob_for_a_missing_tool_is_ignored(box):
    for folder in ("/usr/local/bin", "/opt/homebrew/bin"):     # the launcher adds these folders to PATH
        if Path(folder).is_dir() and shutil.which("setsid", path=folder):
            pytest.skip(f"setsid in {folder}")
    farm = tools_farm(box, FARM_TOOLS)
    env = dict(box.env, PATH=str(farm), HOME=str(box.root), TOPOPT_DETACH="setsid")
    r = subprocess.run([*SH_ABS, str(box.script), "doctor"], capture_output=True, text=True, timeout=60, env=env)
    line = [x for x in r.stdout.splitlines() if x.strip().startswith("start method")][0]
    # The knob is ignored: the launcher uses its own choice. A perl in those folders (Homebrew) is found.
    expected = "perl" if shutil.which("perl", path="/usr/local/bin:/opt/homebrew/bin") else "plain"
    assert line.split("start method", 1)[1].split()[0] == expected


# --- L1a: doctor ------------------------------------------------------------------------------------------------------

def doctor_rows(out):
    """The label -> text of the report rows ('  label   text')."""
    rows = {}
    for line in out.splitlines():
        if line.startswith("  ") and not line.startswith("      "):
            label = line[2:16].strip()
            if label:
                rows[label] = line[17:]
    return rows


def test_doctor_on_a_ready_environment_says_ready_and_changes_nothing(box):
    assert box.run("sync").returncode == 0
    (box.env_dir / "setup.log").write_text("".join(f"line {i} of the log\n" for i in range(1, 8)))
    before = sorted(str(f.relative_to(box.env_dir)) for f in box.env_dir.rglob("*"))
    box.log.write_text("")
    r = box.run("doctor")
    assert r.returncode == 0, r.stdout
    assert r.stderr == ""
    rows = doctor_rows(r.stdout)
    assert rows["script"] == str(box.script)
    assert rows["uv"] == f"system: {box.bindir}/uv (uv 99.0.0-fake)"
    assert rows["env dir"] == f"{box.env_dir} (chosen by: TOPOPT_ENV_DIR)"
    assert rows["venv Python"].startswith(f"{box.env_dir}/venv/bin/python: runs")
    assert rows["stamp"] == "present, matches this checkout (pins=lock)"
    assert rows["setup marker"] == "none" and rows["setup running"] == "no"
    assert rows["imports"].startswith("ok (numpy, scipy.ndimage,") and "topopt_mcp.server" in rows["imports"]
    assert rows["lock file"] == "requirements.lock (stamp name: lock)"
    assert rows["system"] == subprocess.run(["uname", "-s", "-m"], capture_output=True, text=True).stdout.strip()
    assert "line 3 of the log" in r.stdout and "line 1 of the log" not in r.stdout   # the last 5 lines of the setup log
    assert "fake python: -m topopt doctor" in r.stdout                    # the Python side is called when the Python runs
    assert r.stdout.rstrip().endswith("Result: ready. Nothing to run.")
    assert box.calls() == [] and box.version_calls() >= 1                 # no install, no network: uv only gives its version
    assert sorted(str(f.relative_to(box.env_dir)) for f in box.env_dir.rglob("*")) == before   # it writes nothing


def test_doctor_with_no_environment_gives_the_sync_command_and_makes_no_folder(box):
    r = box.run("doctor")
    assert r.returncode == 1 and r.stderr == ""
    rows = doctor_rows(r.stdout)
    assert "not there" in rows["venv Python"] and rows["stamp"].startswith("not there")
    assert rows["imports"].startswith("not checked")
    assert "topopt doctor (Python): not run" in r.stdout
    assert r.stdout.rstrip().endswith(f"Next command:\n  TOPOPT_ENV_DIR='{box.env_dir}' sh '{box.script}' sync")
    assert "Result: action needed" in r.stdout
    assert not box.env_dir.exists() and box.calls() == []
    done = subprocess.run(r.stdout.rstrip().splitlines()[-1], shell=True, capture_output=True, text=True, env=box.env, timeout=60)
    assert done.returncode == 0, done.stderr                              # the printed command works as printed
    assert box.run("doctor").returncode == 0


def test_doctor_finds_a_python_that_does_not_run(box):
    assert box.run("sync").returncode == 0
    (box.env_dir / "venv" / "bin" / "python").write_text("#!/nonexistent/interpreter\n")
    r = box.run("doctor")
    assert r.returncode == 1
    assert "does not run" in doctor_rows(r.stdout)["venv Python"]
    assert "topopt doctor (Python): not run, the venv Python does not run" in r.stdout
    assert r.stdout.rstrip().endswith(f"sh '{box.script}' sync")


def test_doctor_finds_a_dangling_python_link(box):
    assert box.run("sync").returncode == 0
    py = box.env_dir / "venv" / "bin" / "python"
    py.unlink()
    py.symlink_to(box.env_dir / "gone" / "python3.12")
    r = box.run("doctor")
    assert r.returncode == 1 and "a link to a file that is not there" in doctor_rows(r.stdout)["venv Python"]


def test_doctor_names_a_cut_setup(box):
    cut_in_step_2(box)
    r = box.run("doctor")
    rows = doctor_rows(r.stdout)
    assert r.returncode == 1
    assert rows["setup marker"].startswith(f"present ({box.marker})") and "cut or failed" in rows["setup marker"]
    assert rows["stamp"].startswith("not there")
    assert rows["imports"].startswith("FAILED: ImportError") and "ndimage" in rows["imports"]
    # The marker is there already, so the next command is `sync` alone (it installs again because of the marker).
    assert r.stdout.rstrip().endswith("Next command:\n  " + f"TOPOPT_ENV_DIR='{box.env_dir}' sh '{box.script}' sync")
    assert "earlier setup was cut or failed" in r.stdout
    assert "needs the network" in r.stdout and "UV_OFFLINE=1" in r.stdout      # --reinstall makes uv check the index
    # The printed command repairs it, as printed.
    done = subprocess.run(r.stdout.rstrip().splitlines()[-1], shell=True, capture_output=True, text=True, env=box.env, timeout=60)
    assert done.returncode == 0, done.stderr
    assert box.run("doctor").returncode == 0


def test_doctor_finds_a_broken_package_behind_a_current_stamp(box):
    """An environment that an older launcher cut and then called ready: the stamp is there and a package does not import.
    `sync` alone would say ready, so the next command makes the marker too."""
    assert box.run("sync").returncode == 0
    box.broken.write_text("")                       # the fake python now fails every import
    assert box.run("status").returncode == 0        # the launcher cannot see it (the import check runs in the setup only)
    r = box.run("doctor")
    assert r.returncode == 1
    assert doctor_rows(r.stdout)["imports"].startswith("FAILED: ImportError")
    assert "touch " in r.stdout.rstrip().splitlines()[-1] and ".topopt-step2" in r.stdout.rstrip().splitlines()[-1]
    done = subprocess.run(r.stdout.rstrip().splitlines()[-1], shell=True, capture_output=True, text=True, env=box.env, timeout=60)
    assert done.returncode == 0, done.stderr
    assert not box.broken.exists() and box.run("doctor").returncode == 0


def test_doctor_with_a_running_setup_says_wait_and_does_not_touch_the_packages(box):
    p = start_mcp_in_its_own_group(box, "plain", TOPOPT_MCP_WAIT="1")
    p.communicate(timeout=30)
    pid = setup_pid_of(box)
    assert wait_for(box.held.exists, 10)                    # step 1 is done: the Python runs, step 2 is held
    r = box.run("doctor")
    assert r.returncode == 1
    rows = doctor_rows(r.stdout)
    assert rows["setup running"] == f"yes (pid {pid})"
    assert rows["setup lock"].startswith(f"pid {pid} is alive") and "; taken " in rows["setup lock"]
    assert "; setup.log last changed " in rows["setup lock"]
    if shutil.which("ps") or Path("/proc/self/cmdline").exists():                              # ps, or /proc where there is no ps
        runs = rows["setup lock"].split("and runs: ")[1].split("; taken ")[0]                    # what the pid runs (cut at 120 characters)
        assert runs.split()[0].endswith(("sh", "bash", "dash")) and len(runs) <= 120
    else:
        assert "the command of the pid is not known" in rows["setup lock"]
    assert "(only a background setup writes it: a sync in a terminal writes to its terminal)" in rows["setup lock"]
    assert f"rm -rf '{box.env_dir}/.setup-lock'" in r.stdout                                  # the way out if the pid is not a setup
    assert rows["imports"] == "not checked (a setup is running)"
    assert "topopt doctor (Python): not run, a setup is running" in r.stdout
    assert r.stdout.rstrip().splitlines()[-1] == f"  TOPOPT_ENV_DIR='{box.env_dir}' sh '{box.script}' sync"   # sync waits; no tail -f
    assert "tail -f" not in r.stdout and "sync waits for it" in r.stdout
    box.hold.write_text("")
    wait_for_the_end_of_the_background_setup(box)


def age_seconds(text):
    """The number of seconds in a doctor age text ('125 s ago (2 min)')."""
    m = re.search(r"(\d+) s ago \((\d+) min\)", text)
    assert m, text
    assert int(m.group(2)) == int(m.group(1)) // 60
    return int(m.group(1))


def test_doctor_shows_the_lock_of_a_live_pid_with_its_command_and_the_ages(box):
    """A pid that another program has taken looks like a running setup: the row gives the facts that show it."""
    other = subprocess.Popen(["sleep", "300"])
    ps = box.bindir / "ps"      # a fake `ps` (this host may have none): the launcher must call it as `ps -p PID -o command=`
    ps.write_text('#!/bin/sh\n[ "$1 $3 $4" = "-p -o command=" ] && { printf "%0150d\\nsecond line\\n" 0 | tr 0 A; exit 0; }\nexit 1\n')
    ps.chmod(ps.stat().st_mode | stat.S_IEXEC)
    try:
        lock = box.env_dir / ".setup-lock"
        lock.mkdir(parents=True)
        (lock / "pid").write_text(f"{other.pid}\n")
        (box.env_dir / "setup.log").write_text("step 1 of 3\n")
        now = time.time()
        os.utime(lock, (now - 725, now - 725))
        os.utime(box.env_dir / "setup.log", (now - 3700, now - 3700))
        before = sorted(str(f.relative_to(box.env_dir)) for f in box.env_dir.rglob("*"))
        r = box.run("doctor")
        rows = doctor_rows(r.stdout)
        text = rows["setup lock"]
        assert text.startswith(f"pid {other.pid} is alive")
        assert f"and runs: {'A' * 120}; taken " in text and "second line" not in text      # the first line only, cut at 120 characters
        taken, changed = text.split("; taken ")[1].split("; setup.log last changed ")
        assert 725 <= age_seconds(taken) <= 760 and "(12 min)" in taken
        assert 3700 <= age_seconds(changed) <= 3735 and "(61 min)" in changed
        assert rows["setup running"] == f"yes (pid {other.pid})"
        assert sorted(str(f.relative_to(box.env_dir)) for f in box.env_dir.rglob("*")) == before   # it writes nothing
    finally:
        other.kill()
        other.wait()


needs_proc = pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="no /proc on this system")


@needs_proc
def test_doctor_reads_the_command_of_a_live_pid_from_proc_when_there_is_no_ps(box):
    """F3 review m4: a Linux host with no `ps` still shows what the pid runs."""
    other = subprocess.Popen(["sleep", "300"])
    try:
        lock = box.env_dir / ".setup-lock"
        lock.mkdir(parents=True)
        (lock / "pid").write_text(f"{other.pid}\n")
        farm = tools_farm(box, FARM_TOOLS)                              # no ps in it
        assert not (farm / "ps").exists()
        env = dict(box.env, PATH=str(farm), HOME=str(box.root))
        r = subprocess.run([*SH_ABS, str(box.script), "doctor"], capture_output=True, text=True, timeout=60, env=env)
        assert f"pid {other.pid} is alive and runs: sleep 300; taken " in doctor_rows(r.stdout)["setup lock"]
    finally:
        other.kill()
        other.wait()


@needs_proc
def test_doctor_says_that_the_command_is_not_known_when_there_is_no_ps_and_no_cmdline(box):
    """A zombie is alive for `kill -0` and has an empty /proc/<pid>/cmdline: the row says that the command is not known."""
    z = subprocess.Popen(["true"])
    try:
        assert wait_for(lambda: Path(f"/proc/{z.pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z", 10), "no zombie"
        lock = box.env_dir / ".setup-lock"
        lock.mkdir(parents=True)
        (lock / "pid").write_text(f"{z.pid}\n")
        farm = tools_farm(box, FARM_TOOLS)
        env = dict(box.env, PATH=str(farm), HOME=str(box.root))
        r = subprocess.run([*SH_ABS, str(box.script), "doctor"], capture_output=True, text=True, timeout=60, env=env)
        text = doctor_rows(r.stdout)["setup lock"]
        assert text.startswith(f"pid {z.pid} is alive; the command of the pid is not known (ps gave none, and /proc/{z.pid}/cmdline is not there or empty); taken ")
        assert " and runs: " not in text
    finally:
        z.wait()


def test_doctor_shows_a_stale_lock_a_lock_without_a_pid_and_no_lock(box):
    assert doctor_rows(box.run("doctor").stdout)["setup lock"] == "none"
    dead = subprocess.run([*SH, "-c", "echo $$"], capture_output=True, text=True).stdout.strip()
    lock = box.env_dir / ".setup-lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(dead + "\n")
    text = doctor_rows(box.run("doctor").stdout)["setup lock"]
    assert text.startswith(f"pid {dead} is NOT running (a stale lock: the next sync removes it); taken ")
    assert "setup.log" not in text                                    # no log yet: no row part for it
    assert "lock mutex" not in doctor_rows(box.run("doctor").stdout)
    (lock / "pid").unlink()
    assert doctor_rows(box.run("doctor").stdout)["setup lock"].startswith("the folder has no pid in it; taken ")


def test_doctor_says_not_known_when_the_system_has_no_date_command(box):
    lock = box.env_dir / ".setup-lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text("999999999\n")
    farm = tools_farm(box, FARM_TOOLS)                                # the farm has no date, no ps
    env = dict(box.env, PATH=str(farm), HOME=str(box.root))
    r = subprocess.run([*SH_ABS, str(box.script), "doctor"], capture_output=True, text=True, timeout=60, env=env)
    assert doctor_rows(r.stdout)["setup lock"] == "pid 999999999 is NOT running (a stale lock: the next sync removes it); taken not known"


def test_doctor_with_a_stamp_that_does_not_match(box):
    assert box.run("sync").returncode == 0
    with open(box.root / "requirements.lock", "a") as f:
        f.write("six==1.17.0\n")
    r = box.run("doctor")
    assert r.returncode == 1 and "does not match" in doctor_rows(r.stdout)["stamp"]
    assert r.stdout.rstrip().endswith("sync")


def test_doctor_says_when_this_topopt_has_no_doctor_command(box):
    assert box.run("sync").returncode == 0
    r = box.run("doctor", FAKE_PY_NO_DOCTOR="1")
    assert r.returncode == 0, r.stdout                 # the shell checks are all fine: the missing command is a fact only
    assert "has no such command (older code)" in r.stdout
    # The lines of the old code come as they are (the output is streamed); the sentence of the launcher follows them.
    assert r.stdout.index("invalid choice: 'doctor'") < r.stdout.index("has no such command (older code)")


def test_doctor_shows_a_failure_of_the_python_command_without_changing_its_own_exit_code(box):
    assert box.run("sync").returncode == 0
    broken = box.env_dir / "venv" / "bin" / "python"
    text = broken.read_text().replace('echo "${FAKE_PY_MACHINE:-fake python: $@}"', 'case "$*" in "-m topopt doctor"*) echo "boom" >&2; exit 3;; esac\necho "fake python: $@"')
    broken.write_text(text)
    r = box.run("doctor")
    assert r.returncode == 0 and "boom" in r.stdout and "ended with exit code 3" in r.stdout
    # The last line must not say only "ready": a reader of the last line has to learn about the Python report.
    assert r.stdout.rstrip().splitlines()[-1].startswith("Result: the launcher environment is ready, but the Python report above lists a problem (exit code 3)")


def slow_python_doctor(box, pause, rc=0, tail="second line of the Python report"):
    """Make the fake Python print one line, wait `pause` s, print `tail`, and end with `rc` for `-m topopt doctor`."""
    py = box.env_dir / "venv" / "bin" / "python"
    line = 'echo "${FAKE_PY_MACHINE:-fake python: $@}"'
    report = (f'case "$*" in "-m topopt doctor"*) echo "first line of the Python report"; sleep {pause}; '
              f'echo "{tail}"; exit {rc};; esac\n' + line)
    py.write_text(py.read_text().replace(line, report))


def test_doctor_says_what_the_python_step_does_before_it_runs(box):
    assert box.run("sync").returncode == 0
    r = box.run("doctor")
    lines = r.stdout.splitlines()
    said = next(i for i, ln in enumerate(lines) if ln.startswith("topopt doctor (Python): reading the machine"))
    assert "up to 90 s each" in lines[said] and "up to 60 s" in lines[said]
    assert "a minute" not in lines[said] and "usually a few seconds" in lines[said]
    assert lines[said + 1].startswith("fake python: -m topopt doctor")      # the Python output follows the line


def test_doctor_shows_the_python_lines_while_the_python_step_still_runs(box, tmp_path):
    assert box.run("sync").returncode == 0
    slow_python_doctor(box, 3, rc=3)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    t0 = time.monotonic()
    proc = subprocess.Popen([*SH, str(box.script), "doctor"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=dict(box.env, TMPDIR=str(tmp)))
    seen = {}
    for line in proc.stdout:
        seen.setdefault(line.strip(), time.monotonic() - t0)
    proc.wait(timeout=60)
    total = time.monotonic() - t0
    first, second = seen["first line of the Python report"], seen["second line of the Python report"]
    assert second - first > 2, seen                      # the first line came about 3 s before the second: it was not held
    assert first < total - 2
    assert proc.returncode == 0                          # the exit code of the Python report does not change the exit code
    assert any("ended with exit code 3" in k for k in seen)
    assert list(tmp.iterdir()) == []                     # no temp folder is left


def test_doctor_with_the_old_code_leaves_no_temp_folder(box, tmp_path):
    assert box.run("sync").returncode == 0
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    r = box.run("doctor", FAKE_PY_NO_DOCTOR="1", TMPDIR=str(tmp))
    assert "has no such command" in r.stdout and list(tmp.iterdir()) == []


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_during_the_python_step_removes_the_folder_and_ends_the_launcher_by_that_signal(box, tmp_path, sig):
    """As before the streaming: the launcher is killed by the signal (a calling shell sees 130 for Ctrl+C, 143 for TERM), not
    ended with a status of its own. The trap removes the temp folder first."""
    assert box.run("sync").returncode == 0
    slow_python_doctor(box, 2)
    py = box.env_dir / "venv" / "bin" / "python"
    pidfile = tmp_path / "py.pid"
    py.write_text(py.read_text().replace('echo "first line of the Python report"', f'echo $$ >{pidfile}; echo "first line of the Python report"'))
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    proc = subprocess.Popen([*SH, str(box.script), "doctor"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            env=dict(box.env, TMPDIR=str(tmp)))
    for line in proc.stdout:
        if line.startswith("first line of the Python report"):
            assert len(list(tmp.iterdir())) == 1         # the folder exists while the step runs
            proc.send_signal(sig)
            break
    proc.stdout.read()
    proc.wait(timeout=60)
    assert list(tmp.iterdir()) == []                     # the trap removed it
    assert proc.returncode == -sig                       # killed by the signal, as without the trap
    with pytest.raises(ProcessLookupError):              # no child is left
        os.kill(int(pidfile.read_text()), 0)


def test_doctor_without_a_usable_temp_folder_still_prints_the_report(box, tmp_path):
    assert box.run("sync").returncode == 0
    r = box.run("doctor", TMPDIR=str(tmp_path / "does-not-exist"))
    assert r.returncode == 0 and "fake python: -m topopt doctor" in r.stdout


def test_doctor_does_not_take_a_failing_report_for_a_missing_command(box):
    """Only the argparse error for the word `doctor` means that this topopt has no such command. A report that has the words
    "invalid choice" for something else is a report."""
    assert box.run("sync").returncode == 0
    py = box.env_dir / "venv" / "bin" / "python"
    line = 'echo "${FAKE_PY_MACHINE:-fake python: $@}"'
    report = 'case "$*" in "-m topopt doctor"*) echo "problem: invalid choice: \'beta\' in the settings" >&2; exit 1;; esac\n' + line
    py.write_text(py.read_text().replace(line, report))
    r = box.run("doctor")
    assert "invalid choice: 'beta' in the settings" in r.stdout and "has no such command" not in r.stdout
    assert "ended with exit code 1" in r.stdout and r.returncode == 0


def test_doctor_runs_the_python_report_of_the_environment_code_from_any_folder(box, tmp_path):
    """`python -m` puts the current folder first on sys.path. A folder `topopt` in the folder of the user (an old clone) must
    not take the place of the code of the environment. This test uses a real Python: a fake one does not read sys.path."""
    assert box.run("sync").returncode == 0
    py = box.env_dir / "venv" / "bin" / "python"
    py.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    for folder, text in (("env-code", "ENV CODE REPORT"), ("other", "OTHER CHECKOUT REPORT")):
        package = tmp_path / folder / "topopt"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "__main__.py").write_text(f"print({text!r})\n")
    r = subprocess.run([*SH, str(box.script), "doctor"], capture_output=True, text=True, timeout=60, cwd=tmp_path / "other",
                       env=dict(box.env, PYTHONPATH=str(tmp_path / "env-code")))
    assert "ENV CODE REPORT" in r.stdout, r.stdout
    assert "OTHER CHECKOUT REPORT" not in r.stdout, r.stdout


@pytest.mark.parametrize("variable, rule", [("CLAUDE_PLUGIN_DATA", "CLAUDE_PLUGIN_DATA"), ("EVAL_TOPOPT_ENV_DIR", "EVAL_TOPOPT_ENV_DIR"),
                                            (None, "default: <repo>/.plugin-env")])
def test_doctor_names_the_rule_that_chose_the_env_dir(box, tmp_path, variable, rule):
    env = {k: v for k, v in box.env.items() if k != "TOPOPT_ENV_DIR"}
    if variable:
        env[variable] = str(tmp_path / "plugin-data")
    r = subprocess.run([*SH, str(box.script), "doctor"], capture_output=True, text=True, env=env, timeout=60)
    folder = tmp_path / "plugin-data" if variable else box.root / ".plugin-env"
    assert doctor_rows(r.stdout)["env dir"] == f"{folder} (chosen by: {rule})"
    assert not folder.exists()


def test_doctor_says_when_uv_is_missing(box):
    if shutil.which("uv", path="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"):
        pytest.skip("a real uv is in a standard folder")
    r = subprocess.run([*SH, str(box.script), "doctor"], capture_output=True, text=True, env=no_uv_env(box), timeout=60)
    assert r.returncode == 1
    assert doctor_rows(r.stdout)["uv"].startswith("not found: the setup downloads uv ")
    assert "uv is not installed: the setup downloads uv " in r.stdout
    assert not box.env_dir.exists()                                                  # doctor changes nothing


def test_doctor_shows_the_lock_of_an_x86_64_python_on_a_mac(box):
    fake_mac_tools(box, sysctl={"hw.optional.arm64": 0})
    r = box.run("doctor", FAKE_PY_MACHINE="x86_64")
    assert r.returncode == 1
    assert doctor_rows(r.stdout)["lock file"] == "requirements-macos-x86_64.lock (stamp name: lock-macos-x86_64)"
    assert doctor_rows(r.stdout)["system"] == "Darwin x86_64"


def test_the_mcp_start_never_prints_the_doctor_report(box):
    """doctor is a command of its own; `mcp` keeps stdout for the protocol."""
    assert box.run("sync").returncode == 0
    box.log.write_text("")
    r = box.run("mcp")
    assert r.stdout == "fake python: -P -m topopt_mcp.server\n" and r.stderr == "" and box.calls() == []


# --- #370: a start that is not ready runs the status server --------------------------------------------------------------------
# Claude Code gives a server 30 s to connect and marks a server that fails as failed (and skips it for 15 minutes). So a start that is not
# ready answers like an MCP server, from a snapshot of the lists of the real server (topopt_mcp/launcher/), and a tool call gives the setup
# status. When the setup is ready, it hands the open stdin to the Python server. The fake uv holds the setup (`box.hold`), so the tests need no timing guess.

MODERN_META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
               "io.modelcontextprotocol/clientInfo": {"name": "test-client", "version": "1"},
               "io.modelcontextprotocol/clientCapabilities": {}}
LISTEN_PARAMS = {"notifications": {"toolsListChanged": True, "promptsListChanged": True, "resourcesListChanged": True}}


def modern(method, ident=None, **params):
    """A request of the modern era (protocol 2026-07-28): every request has params._meta with the protocol version."""
    msg = {"jsonrpc": "2.0", "method": method, "params": dict(params, _meta=MODERN_META)}
    if ident is not None:
        msg["id"] = ident
    return json.dumps(msg, separators=(",", ":"))


def legacy(method, ident=None, **params):
    """A request of the legacy era (the handshake): no _meta."""
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    if ident is not None:
        msg["id"] = ident
    return json.dumps(msg, separators=(",", ":"))


def legacy_initialize(version="2025-11-25", ident=0):
    return legacy("initialize", ident, protocolVersion=version, capabilities={}, clientInfo={"name": "test-client", "version": "1"})


INITIALIZED = '{"jsonrpc":"2.0","method":"notifications/initialized"}'
MODERN_LINES = [
    modern("server/discover", "server-discover-probe-1"),
    modern("subscriptions/listen", "listen:0", **LISTEN_PARAMS),
    modern("prompts/list", 1), modern("resources/list", 2), modern("tools/list", 3),
    '{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":1}}',
    modern("resources/templates/list", 4),
    modern("tools/call", 5, name="environment", arguments={}),
    modern("ping", 6), modern("bogus/method", 7),
    modern("prompts/get", 8, name="x"), modern("resources/read", 9, uri="topopt://x"),
]


def not_ready_run(box, lines=(), wait="0", **extra):
    """`mcp` with an environment that is not ready, the client lines on stdin, which then ends. Returns the result and the lines of
    stdout as objects. The fake uv holds the install of the lock file until the test writes `box.hold` (`finish_the_setup`, or the end of
    the test), so the environment cannot become ready, however slow the machine; the lines go when it holds. A setup whose install
    fails (`FAKE_UV_FAIL_LOCK`) is not held: it stops during the wait, and that stop is what the test checks. Nor is one that stops
    before step 2 (`FAKE_UV_FAIL_VENV`, a `TOPOPT_ENV_DIR` that cannot be made): their lines go at once."""
    env = {"FAKE_UV_SLEEP": "1", "TOPOPT_MCP_WAIT": wait}
    if "FAKE_UV_FAIL_LOCK" not in extra:
        env["FAKE_UV_HOLD"] = str(box.hold)
    env.update(extra)
    # A setup that stops before step 2 (no venv, no folder) never holds: its lines go at once.
    if "FAKE_UV_HOLD" not in env or "FAKE_UV_FAIL_VENV" in extra or "TOPOPT_ENV_DIR" in extra:
        r = box.run("mcp", stdin="".join(line + "\n" for line in lines), **env)
        return r, [json.loads(x) for x in r.stdout.splitlines()]
    # The client lines go only when the setup holds in step 2: before that a call can come in the first 3 s, when the setup shell has
    # no pid in the lock folder yet, and the answer is "the setup is starting" (seen under load: the timing flake noted on #399).
    p = subprocess.Popen([*SH, str(box.script), "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=dict(box.env, **env))
    deadline = time.time() + 60
    while not box.held.exists() and p.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    out, err = p.communicate("".join(line + "\n" for line in lines), timeout=60)
    r = subprocess.CompletedProcess(p.args, p.returncode, out, err)
    return r, [json.loads(x) for x in r.stdout.splitlines()]


def finish_the_setup(box):
    """The setup that `mcp` started goes on in the background: let it end, so that the test leaves no process behind."""
    box.hold.touch()
    wait_for_the_end_of_the_background_setup(box)


def text_of(answer):
    return answer["result"]["content"][0]["text"]


SERVER_INFO = {"io.modelcontextprotocol/serverInfo": {"name": "topopt", "version": ""}}


SNAPSHOT_DIR = REPO / "topopt_mcp" / "launcher"


def snapshot(name):
    """The `result` of an answer in the committed snapshot of the launcher (topopt_mcp/launcher/NAME.json)."""
    return json.loads((SNAPSHOT_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_a_modern_client_gets_the_real_lists_from_the_first_answer_and_no_error(box):
    r, answers = not_ready_run(box, MODERN_LINES)
    assert r.returncode == 1
    assert "not ready yet" in r.stderr and "tool call gives the setup status" in r.stderr
    # the ack has no id; the notification gets none; the list calls get results; prompts/get and resources/read get a JSON-RPC error
    assert [a.get("id") for a in answers] == ["server-discover-probe-1", None, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    discover, ack, prompts, resources, tools, templates, call, ping, bogus, prompt_get, resource_read = answers
    assert not [a for a in (discover, ack, prompts, resources, tools, templates, call) if "error" in a]
    assert discover["result"] == snapshot("modern.discover") and prompts["result"] == snapshot("modern.prompts")
    assert resources["result"] == snapshot("modern.resources") and templates["result"] == snapshot("modern.templates")
    assert tools["result"] == snapshot("modern.tools") and len(tools["result"]["tools"]) > 20            # the real tools, not one status tool
    assert "setup_status" not in json.dumps(answers)
    assert ack == {"jsonrpc": "2.0", "method": "notifications/subscriptions/acknowledged",
                   "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": "listen:0"}, "notifications": LISTEN_PARAMS["notifications"]}}
    assert call["result"]["isError"] is True and call["result"]["resultType"] == "complete" and call["result"]["_meta"] == SERVER_INFO
    assert "still setting up" in text_of(call) and "Do not call it again at once" in text_of(call)
    assert ping["error"] == {"code": -32601, "message": "Method not found"}              # modern: no ping, as the Python server
    assert bogus["error"] == {"code": -32601, "message": "Method not found"}
    for bad in (prompt_get, resource_read):
        assert bad["error"]["code"] == -32603 and "still setting up" in bad["error"]["message"]
    finish_the_setup(box)


def test_the_listen_request_gets_no_result_and_a_string_or_number_id_is_kept(box):
    lines = [modern("subscriptions/listen", 41, **LISTEN_PARAMS)]
    r, answers = not_ready_run(box, lines)
    assert len(answers) == 1 and "id" not in answers[0] and "result" not in answers[0]
    assert answers[0]["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"] == 41
    finish_the_setup(box)


def test_a_legacy_client_gets_the_handshake_answer_and_the_real_lists_without_the_modern_members(box):
    lines = [legacy_initialize("2025-06-18", 0), INITIALIZED, legacy("tools/list", 1), legacy("prompts/list", 2), legacy("resources/list", 3),
             legacy("resources/templates/list", 4), legacy("tools/call", 5, name="environment", arguments={}), legacy("ping", 6)]
    r, answers = not_ready_run(box, lines)
    assert r.returncode == 1 and [a["id"] for a in answers] == [0, 1, 2, 3, 4, 5, 6]
    init, tools, prompts, resources, templates, call, ping = answers
    expected = snapshot("legacy.initialize")
    expected["protocolVersion"] = "2025-06-18"                         # the placeholder of the file is replaced by the version of the request
    assert init["result"] == expected and "@PROTOCOL_VERSION@" not in json.dumps(answers)
    assert tools["result"] == snapshot("legacy.tools") and "resultType" not in tools["result"] and len(tools["result"]["tools"]) > 20
    assert prompts["result"] == snapshot("legacy.prompts") and resources["result"] == snapshot("legacy.resources")
    assert templates["result"] == snapshot("legacy.templates")
    assert call["result"]["isError"] is True and set(call["result"]) == {"content", "isError"} and "still setting up" in text_of(call)
    assert ping["result"] == {}
    finish_the_setup(box)


def handshake_versions():
    types_version = pytest.importorskip("mcp_types.version")
    return list(types_version.HANDSHAKE_PROTOCOL_VERSIONS), types_version.LATEST_HANDSHAKE_VERSION


def sdk_negotiated(requested):
    """What the SDK (mcp 2.3.0) answers to `initialize`: the requested version if it is a handshake version, else the latest one."""
    versions, latest = handshake_versions()
    return requested if requested in versions else latest


def test_the_handshake_versions_of_the_launcher_are_those_of_the_sdk(box):
    versions, latest = handshake_versions()
    assert script_value(box, "HANDSHAKE_LATEST") == latest
    text = box.script.read_text()
    assert re.search(r'^HANDSHAKE_VERSIONS="([^"]*)"$', text, re.M).group(1).split() == versions


@pytest.mark.parametrize("requested", ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25", "2026-07-28", "2099-01-01", "x", ""])
def test_the_version_in_the_initialize_answer_is_the_one_the_sdk_picks(box, requested):
    r, answers = not_ready_run(box, [legacy_initialize(requested)])
    assert answers[0]["result"]["protocolVersion"] == sdk_negotiated(requested)
    finish_the_setup(box)


def test_initialize_without_a_protocol_version_gets_the_latest_handshake_version(box):
    r, answers = not_ready_run(box, ['{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"capabilities":{}}}'])
    assert answers[0]["result"]["protocolVersion"] == sdk_negotiated(None)
    finish_the_setup(box)


def test_the_era_comes_from_the_first_request(box):
    """The modern era: server/discover (also without _meta), or any request with the protocolVersion key in params._meta.
    Else the legacy era. The members of the results show the era."""
    r, a = not_ready_run(box, [legacy("server/discover", 1)])
    assert "resultType" in a[0]["result"]                                       # server/discover alone means modern
    r, a = not_ready_run(box, [modern("tools/list", 1)])
    assert a[0]["result"]["resultType"] == "complete"                           # the key in _meta alone means modern
    r, a = not_ready_run(box, [legacy("tools/list", 1)])
    assert "resultType" not in a[0]["result"]                                   # neither: legacy
    r, a = not_ready_run(box, [legacy("ping", 1), modern("tools/list", 2)])
    assert "resultType" not in a[0]["result"] and "resultType" not in a[1]["result"]    # the first request decides
    r, a = not_ready_run(box, [modern("server/discover", 1), legacy_initialize(), legacy("tools/list", 2)])
    assert "resultType" in a[0]["result"] and "resultType" not in a[1]["result"] and "resultType" not in a[2]["result"]   # an initialize makes it legacy
    finish_the_setup(box)


def test_the_real_client_lines_get_an_answer_each(box):
    """The lines of Claude Code 2.1.293 (a probe of 2026-10-25): server/discover, initialize, server/discover, a notification, initialize."""
    lines = (REPO / "tests" / "data" / "mcp_client_lines_claude_code_2.1.293.jsonl").read_text().splitlines()
    r, answers = not_ready_run(box, lines)
    assert [a["id"] for a in answers] == ["server-discover-probe-1", 0, 1, 0]
    assert all("error" not in a for a in answers)
    finish_the_setup(box)


def test_a_last_line_without_a_line_end_is_answered_too(box):
    r = box.run("mcp", stdin=modern("tools/list", 5), TOPOPT_MCP_WAIT="0", FAKE_UV_SLEEP="1", FAKE_UV_HOLD=str(box.hold))
    assert [json.loads(x)["id"] for x in r.stdout.splitlines()] == [5]
    finish_the_setup(box)


def test_no_line_on_stdin_ends_the_status_server_at_once_with_empty_stdout(box):
    r, answers = not_ready_run(box, [])
    assert r.returncode == 1 and r.stdout == "" and "not ready yet" in r.stderr
    finish_the_setup(box)


def test_the_first_answer_comes_at_once_not_after_a_wait_or_an_error(box):
    """The wait is TOPOPT_MCP_WAIT (24 s by default). With 0, the server answers within the time of the start of the shell and of the setup."""
    start = time.monotonic()
    r, answers = not_ready_run(box, [modern("server/discover", 0)], wait="0")
    assert answers and time.monotonic() - start < 3
    finish_the_setup(box)


# One line each: the five fields of the awk (method, id as in the line, 1 if params._meta has the protocolVersion key, the protocolVersion in
# params, the name in params), or None for a line that is no request or notification.
PARSER_CASES = [
    ('{"jsonrpc":"2.0","method":"tools/call","params":{"id":99,"method":"initialize"},"id":7}', ("tools/call", "7", "0", "-", "-")),   # inner keys are not read
    ('{"params":{"id":5,"method":"initialize"},"method":"ping","id":8}', ("ping", "8", "0", "-", "-")),                      # inner keys first
    ('{"method":"notifications/initialized","params":{"id":3}}', ("notifications/initialized", "-", "0", "-", "-")),         # a notification: no top-level id
    ('{"method":"initialize","params":{"text":"\\"id\\":9 and \\"method\\":\\"x\\""},"id":2}', ("initialize", "2", "0", "-", "-")),   # key-like text in a string
    ('{"method":"initialize","params":{"list":[{"id":1},{"id":2}]},"id":3}', ("initialize", "3", "0", "-", "-")),             # ids in an array of objects
    ('{ "id" : 12 , "method" : "initialize" }', ("initialize", "12", "0", "-", "-")),                                          # spaces
    ('{"id":1.5e3,"method":"ping"}', ("ping", "1.5e3", "0", "-", "-")),                                                       # a number with a fraction and an exponent
    ('{"id":"é-id","method":"ping"}', ("ping", '"é-id"', "0", "-", "-")),                                                    # UTF-8 in the id
    ('{"id":"a \\"q\\" b","method":"ping"}', ("ping", '"a \\"q\\" b"', "0", "-", "-")),                                       # an id with an escaped quote stays as it is
    ('{"id":null,"method":"initialize"}', None),                                                                               # MCP does not allow a null id
    ('{"id":{"a":1},"method":"initialize"}', None),
    ('{"id":4,"method":5}', None),
    ('{"jsonrpc":"2.0","id":4,"result":{}}', None),                                                                            # a response, not a request
    ('{"jsonrpc":"2.0","id":4,"error":{"code":-1,"message":"x","method":"initialize"}}', None),
    ('[{"id":1,"method":"initialize"}]', None),                                                                                # a batch: not MCP
    ('not json', None),
    ('{"id":1,"method":"initialize', None),                                                                                    # a string without its end
    ('', None),
    ('{"jsonrpc":"2.0","id":3,"method":"ping"}\r', ("ping", "3", "0", "-", "-")),                                              # a CRLF line end
    # the era key: only params._meta counts
    (modern("tools/list", 1), ("tools/list", "1", "1", "-", "-")),
    ('{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"cursor":"io.modelcontextprotocol/protocolVersion"}}', ("tools/list", "1", "0", "-", "-")),
    ('{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{"_meta":{"io.modelcontextprotocol/protocolVersion":"1"}}}}', ("tools/call", "1", "0", "-", "x")),
    ('{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"_meta":{"x":{"io.modelcontextprotocol/protocolVersion":"1"}}}}', ("tools/call", "1", "0", "-", "-")),
    ('{"io.modelcontextprotocol/protocolVersion":"1","id":1,"method":"ping"}', ("ping", "1", "0", "-", "-")),
    # the protocolVersion and the name: only directly in params
    ('{"id":0,"method":"initialize","params":{"capabilities":{"protocolVersion":"x"},"protocolVersion":"2025-06-18"}}', ("initialize", "0", "0", "2025-06-18", "-")),
    ('{"id":0,"method":"initialize","params":{"clientInfo":{"name":"c","protocolVersion":"x"}}}', ("initialize", "0", "0", "-", "-")),
    ('{"id":0,"method":"tools/call","params":{"arguments":{"name":"evil"},"name":"setup_status"}}', ("tools/call", "0", "0", "-", "setup_status")),
    ('{"id":0,"method":"tools/call","name":"top","params":{"name":"setup_status"}}', ("tools/call", "0", "0", "-", "setup_status")),
    ('{"id":0,"method":"tools/call","other":{"name":"top"}}', ("tools/call", "0", "0", "-", "-")),
]


def awk_program():
    return re.search(r"^JSON_REQUEST_AWK='(.*?\n\})'$", (REPO / "scripts" / "topopt-env.sh").read_text(), re.M | re.S).group(1)


@pytest.mark.parametrize("line, expected", PARSER_CASES, ids=[str(i) for i in range(len(PARSER_CASES))])
def test_the_awk_reads_the_top_level_keys_whatever_the_other_keys_and_texts_hold(line, expected):
    out = subprocess.run(["awk", awk_program()], input=line + "\n", capture_output=True, text=True, env=dict(os.environ, LC_ALL="C")).stdout
    assert (out.rstrip("\n").split("\x1f") if out else None) == (list(expected) if expected else None)


def test_the_awk_reads_the_real_client_lines():
    lines = (REPO / "tests" / "data" / "mcp_client_lines_claude_code_2.1.293.jsonl").read_text().splitlines()
    got = [subprocess.run(["awk", awk_program()], input=x + "\n", capture_output=True, text=True, env=dict(os.environ, LC_ALL="C")).stdout.rstrip("\n").split("\x1f")
           for x in lines]
    assert [g[:3] for g in got] == [["server/discover", '"server-discover-probe-1"', "1"], ["initialize", "0", "0"], ["server/discover", "1", "1"],
                                    ["notifications/cancelled", "-", "1"], ["initialize", "0", "0"]]
    assert got[1][3] == "2025-11-25"


def test_json_escape_makes_a_string_for_json_of_any_text(box):
    """The function of the launcher, run by itself on texts with a double quote, a backslash and control characters."""
    source = (REPO / "scripts" / "topopt-env.sh").read_text()
    line = next(x for x in source.splitlines() if x.startswith("json_escape()"))
    cases = ['plain text', 'say "hi"', 'a\\b', 'a\\"b', 'tab\there', 'two\nlines', 'cr\rhere', 'bell\x07gone', 'unit\x1fsep',
             '100% of %s and %d', "it's", 'ünï', '', '\\\\']
    for text in cases:
        out = subprocess.run([*SH, "-c", line + '\njson_escape "$1"', "sh", text], capture_output=True, text=True).stdout
        decoded = json.loads('"' + out + '"')                                        # it is a valid JSON string
        expected = re.sub(r"[\t\r\n]", " ", text)
        expected = re.sub(r"[\x00-\x1f]", "", expected)
        assert decoded == expected, (text, out)


# The text of a tool call during the setup, in each state. It gives the step, the time, and ONE next step.

CALL = [modern("tools/call", 1, name="environment", arguments={})]
NETWORK_ERROR = "error: Failed to download `numpy`\n  Caused by: dns error: no such host"   # a clean uv error that does not fall back


def status_text(box, wait="2", **extra):
    r, answers = not_ready_run(box, CALL, wait=wait, **extra)
    assert r.returncode == 1 and len(answers) == 1 and answers[0]["result"]["isError"] is True and "error" not in answers[0], r
    return text_of(answers[0])


def test_the_text_while_the_setup_runs_gives_the_step_the_time_and_one_next_step(box):
    text = status_text(box)
    assert text.startswith("topopt is still setting up its Python environment. Now: step 2 of 3: install the packages of ")
    assert re.search(r"Now: step [123] of 3: \S", text)                                  # the step from setup.log, without the prefix of the line
    assert re.search(r"It has run \d+ s\.", text)
    assert "The whole setup takes 1 to 3 minutes (about 200 MB to download)" in text    # the numbers of the README
    assert "large packages" not in text                                                 # the fake uv writes no progress: the recheck time
    assert text.endswith("Next step: tell the user that topopt is still setting up, about 30 s more. "
                         "Wait that long before you call a topopt tool again. Do not call it again at once.")
    assert text.count("Next step:") == 1 and "\n" not in text
    finish_the_setup(box)


def test_the_text_gives_the_progress_of_step_2_and_the_wait_from_it(box):
    """The fake uv writes the lines of a real uv (one of three large packages downloaded) and holds: the text counts them."""
    progress = box.root.parent / "progress.txt"
    progress.write_text("Resolved 54 packages in 1.31s\nDownloading numpy (15.9MiB)\nDownloading scipy (33.7MiB)\n"
                        "Downloading pygments (1.2MiB)\n Downloaded pygments\n")
    text = status_text(box, FAKE_UV_PROGRESS=str(progress))
    assert " Downloaded: 1 of the 3 large packages (53 MB in all). " in text          # 50.8 MiB
    wait = re.search(r"still setting up, about (\d+) s more\. Wait that long", text)
    assert wait and int(wait.group(1)) >= 10, text                 # the numbers: test_the_wait_comes_from_the_download_progress_of_step_2
    assert text.endswith("Do not call it again at once.")
    finish_the_setup(box)


NOW = 1800000000   # the clock of the shell in wait_of: `date` is a function that prints it, so a new second cannot change a result


def wait_of(box, step, steplog=None, since=None):
    """mcp_wait of the launcher: the step text, the lines of last-step.log, the seconds since step 2 began (the lock folder)."""
    lock = box.env_dir / "lk"
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "step2").unlink(missing_ok=True)
    if since is not None:
        (lock / "step2").write_text(f"{NOW - since}\n")
    steplog_file = box.env_dir / "last-step.log"
    steplog_file.unlink(missing_ok=True)
    if steplog is not None:
        steplog_file.write_text(steplog)
    src = box.script.read_text()
    consts = "; ".join(ln for ln in src.splitlines() if re.match(r"SETUP_(RECHECK|TAIL|MAX_WAIT)_S=", ln))
    script = (f"{consts}; LOCK={shlex.quote(str(lock))}; STEPLOG={shlex.quote(str(steplog_file))}; date() {{ echo {NOW}; }}\n" +
              shell_function(src, "mcp_wait") + f"\nmcp_wait {shlex.quote(step)}")
    return subprocess.run([*SH, "-c", script], capture_output=True, text=True).stdout.strip()


STEP2 = "step 2 of 3: install the packages of requirements.lock"


def test_the_wait_without_progress_is_the_recheck_time_and_in_step_3_the_tail(box):
    src = box.script.read_text()
    assert "\nSETUP_RECHECK_S=30\n" in src and "\nSETUP_TAIL_S=10\n" in src and "\nSETUP_MAX_WAIT_S=180\n" in src   # the pinned values
    assert wait_of(box, f"step 1 of 3: make the Python {LAUNCHER_PY} environment") == "30"
    assert wait_of(box, "") == "30"                                                    # no step line yet
    assert wait_of(box, STEP2) == "30"                                                 # no step-2 time, no log
    assert wait_of(box, STEP2, "", since=20) == "30"                                   # a cache that has every package: no line
    assert wait_of(box, STEP2, "Downloading numpy (15.9MiB)\nDownloading scipy (33.7MiB)\n", since=20) == "30"   # none done yet
    assert wait_of(box, "step 3 of 3: link the topopt code in /x") == "10"
    # A new setup over one that was cut in step 2: the old log has progress, and the new setup has not written its step-2 time yet.
    old = "".join(f"Downloading p{i} ({i}.0MiB)\n" for i in range(1, 6)) + "".join(f" Downloaded p{i}\n" for i in range(1, 4))
    assert wait_of(box, STEP2, old) == "30"
    (box.env_dir / "lk" / "step2").write_text("x\n")
    assert wait_of(box, STEP2, old) == "30"                                            # not a number


def test_the_setup_empties_last_step_log_before_it_writes_the_step_2_time(box):
    """An old last-step.log (a setup cut in step 2) with a new step-2 time would give the old progress. The window is short: from the
    step-2 time to the start of uv, which empties the file too (run_logged), so no run can show it. The order of the code is checked."""
    body = shell_function(box.script.read_text(), "install_all")
    assert body.index(': >"$STEPLOG"') < body.index('date +%s >"$LOCK/step2"') < body.index('run_logged "$STEPLOG"')
    old = "".join(f"Downloading p{i} ({i}.0MiB)\n" for i in range(1, 4)) + "".join(f" Downloaded p{i}\n" for i in range(1, 4))
    assert wait_of(box, STEP2, old, since=1) == "10 3 3 6"                            # what the old file would say: all done


def test_the_wait_is_the_recheck_time_when_uv_does_not_start_the_downloads_at_once(box):
    """UV_CONCURRENT_DOWNLOADS=1 (or concurrent-downloads in a uv.toml): uv writes each Downloading line when that download starts, so
    the lines that are there do not give the number of packages (the review of #399: "1 of the 2" while 12 came)."""
    one_by_one = "Downloading llvmlite (57.1MiB)\n Downloaded llvmlite\nDownloading scipy (33.7MiB)\n"
    assert wait_of(box, STEP2, one_by_one, since=40) == "30"
    assert wait_of(box, STEP2, one_by_one + " Downloaded scipy\n", since=50) == "30"


def test_the_wait_is_at_most_the_longest_wait(box):
    """A pause (a stall, a laptop that sleeps) makes the rate low: the wait stays at SETUP_MAX_WAIT_S, and the next call computes it again."""
    lines = "Downloading a (1.0MiB)\nDownloading b (2.0MiB)\nDownloading c (200.0MiB)\n Downloaded a\n"
    assert wait_of(box, STEP2, lines, since=10) == "180 1 3 213"                       # 5 of 203 MiB in 10 s: 406 s computed
    assert wait_of(box, STEP2, lines, since=3600) == "180 1 3 213"
    assert wait_of(box, STEP2, lines.replace("200.0", "4.0"), since=30) == "22 1 3 7"  # 5 of 7 MiB in 30 s: 12 s + 10 s, under the limit


def test_the_wait_comes_from_the_download_progress_of_step_2(box):
    """10, 20 and 30 MiB; the 10 is done after 10 s. Bytes so far: 10 + 20 for each of the two others (the 20 ends next) = 50 of 60,
    at 5 MiB/s: 2 s for the rest, plus the tail of 10 s. Units KiB and GiB; all done: the tail."""
    lines = "Resolved 3 packages in 0.5s\nDownloading a (10.0MiB)\nDownloading b (20.0MiB)\nDownloading c (30.0MiB)\n Downloaded a\n"
    assert wait_of(box, STEP2, lines, since=10) == "12 1 3 63"                         # 60 MiB = 63 MB
    assert wait_of(box, STEP2, lines, since=20) == "14 1 3 63"                         # slower: more time for the rest
    units = ("Downloading k (256KiB)\nDownloading m (512.0MiB)\nDownloading g (1.0GiB)\nDownloading x (2.0GiB)\n"
             " Downloaded k\n Downloaded m\n")
    # 0.25 + 512 + 1024 for g and for x = 2560.25 of 3584.25 MiB after 10 s: 4.0 s for the rest, plus 10 s, rounded up
    assert wait_of(box, STEP2, units, since=10) == "14 2 4 3758"
    assert wait_of(box, STEP2, lines + " Downloaded b\n Downloaded c\n", since=30) == "10 3 3 63"
    assert wait_of(box, STEP2, " Downloaded x\nPrepared 1 package in 2s\n", since=30) == "30"   # a package without a size line


@pytest.mark.parametrize("seconds, words", [(1, "about 5 s"), (10, "about 10 s"), (12, "about 15 s"), (56, "about 60 s"),
                                            (95, "about 95 s"), (115, "about 115 s"), (116, "about 2 minutes"),
                                            (149, "about 2 minutes"), (150, "about 3 minutes"), (700, "about 12 minutes")])
def test_the_wait_in_words(box, seconds, words):
    script = shell_function(box.script.read_text(), "mcp_wait_words") + f"; mcp_wait_words {seconds}"
    assert subprocess.run([*SH, "-c", script], capture_output=True, text=True).stdout.strip() == words


def measured_setup(name):
    """A first setup of 2026-10-27, from tests/data: (events (seconds, source, line), start of step 2, start of step 3, ready)."""
    events = []
    for ln in (REPO / "tests" / "data" / name).read_text().splitlines():
        if ln.startswith("#"):
            continue
        t, rest = ln.split(" ", 1)
        source, line = rest.split("| ", 1)
        events.append((float(t), source, line))
    at = lambda text: next(t for t, s, x in events if s == "err" and text in x)
    return events, at("step 2 of 3"), at("step 3 of 3"), at("ready after")


@pytest.mark.parametrize("name", ["setup_progress_2026-10-27_uv_0.12.1.txt", "setup_progress_2026-10-27_uv_0.13.5.txt"])
def test_the_wait_of_a_measured_setup_is_at_most_about_twice_the_time_that_is_left(box, name):
    """#399, acceptance 2. Replay of a real first setup (59 s with uv 0.13.5, the "fast link" of the issue: about 60 s; 112 s with
    uv 0.12.1): at each second of steps 2 and 3, the wait of the launcher (in words) against the time that was left. At most twice that
    time while 10 s or more are left, plus the 5 s step of the words (the worst point: 85 s for 41.9 s left, 18 s into the 59 s setup);
    then the shortest wait (10 s). Before the first package is done it is the recheck time (30 s)."""
    src = box.script.read_text()
    events, step2, step3, ready = measured_setup(name)
    words_of = shell_function(src, "mcp_wait_words")
    worst = 0.0
    for t in range(int(step2) + 1, int(ready)):
        step = STEP2 if t < step3 else "step 3 of 3: link the topopt code in /x"
        steplog = "".join(x + "\n" for te, s, x in events if s == "step" and te <= t)
        numbers = wait_of(box, step, steplog, since=round(t - step2))
        words = subprocess.run([*SH, "-c", f"{words_of}; mcp_wait_words {numbers.split()[0]}"], capture_output=True, text=True).stdout
        m = re.match(r"about (\d+) (s|minutes)$", words.strip())
        wait = int(m.group(1)) * (1 if m.group(2) == "s" else 60)
        left = ready - t
        if left >= 10:
            assert wait <= 2 * left + 5, (name, t, left, words, numbers)
            worst = max(worst, wait / left)
        else:
            assert wait == 10, (name, t, words)
    assert worst > 1                                                                    # the replay compares something


def test_every_text_of_a_running_setup_names_a_wait_and_says_not_to_call_again_at_once(box):
    """#399, acceptance 3: with and without the time, early and late, without progress, with progress, in step 3."""
    fn = "; ".join(shell_function(box.script.read_text(), n) for n in ("mcp_wait_words", "mcp_running_text", "mcp_cause_text"))
    for ran, wait in (("", ""), ("5", "30"), ("40", "17 10 12 143"), ("90", "10"), ("181", "30"), ("400", "250 3 12 143")):
        script = f"SETUP_RECHECK_S=30; {fn}; mcp_cause_text running 'step 2 of 3: x' '{ran}' '{wait}'"
        text = subprocess.run([*SH, "-c", script], capture_output=True, text=True).stdout
        assert re.search(r"Next step: tell the user that topopt is still setting up, about \d+ (s|minutes) more\. "
                         r"Wait that long before you call a topopt tool again\. Do not call it again at once\.$", text), (ran, wait, text)
        assert text.count("Next step:") == 1
        assert ("It has run" in text) == bool(ran)
        assert ("large packages" in text) == (len(wait.split()) == 4)
        late = ran and int(ran) > 180
        assert ("takes longer than usual (a slow link)" in text) == bool(late)
        assert ("The whole setup takes 1 to 3 minutes" in text) == (not late)
    assert "about 4 minutes more" in subprocess.run(
        [*SH, "-c", f"{fn}; mcp_cause_text running 'step 2 of 3: x' 400 '250 3 12 143'"], capture_output=True, text=True).stdout


def test_the_text_in_the_first_seconds_says_that_the_setup_is_starting(box):
    """The setup shell may not have written its pid yet: no "about 3 more minutes" for a setup that can fail at once."""
    script = shell_function(box.script.read_text(), "mcp_cause_text") + '; mcp_cause_text starting "" ""'
    text = subprocess.run([*SH, "-c", script], capture_output=True, text=True).stdout
    assert text == ("topopt: the setup is starting. Next step: tell the user that the setup is starting, "
                    "and call a topopt tool again in about 10 s. Do not call it again at once.")


def test_the_cause_is_starting_when_the_setup_began_now_and_has_no_pid_yet(box):
    """The review case: a call at 0.0 s of a setup that fails at once. No lock pid yet: "starting", not "running"."""
    src = box.script.read_text()
    def cause(started_ago, running):
        script = (f'STARTED={box.env_dir}.started; echo $(( $(date +%s) - {started_ago} )) > "$STARTED"; '
                  f"setup_running() {{ {'true' if running else 'false'}; }}; " +
                  "; ".join(shell_function(src, n) for n in ("started_just_now", "mcp_cause")) + "; mcp_cause")
        return subprocess.run([*SH, "-c", script], capture_output=True, text=True).stdout.strip()
    box.env_dir.parent.mkdir(parents=True, exist_ok=True)
    assert cause(0, False) == "starting"
    assert cause(0, True) == "running"                                                  # a live pid wins
    assert cause(10, False) in ("stopped", "cut", "uv", "uvvar", "folder")             # after 3 s: the old rules


def test_the_text_when_an_earlier_setup_was_cut_or_failed_gives_the_command(box):
    text = status_text(box, wait="6", FAKE_UV_FAIL_LOCK=NETWORK_ERROR)   # the early stop needs more than 3 s of wait
    assert box.marker.exists()
    assert "stopped before it was ready" in text and "cut or failed" in text and "installs the packages again" in text
    assert text.endswith(f"Next step: show the user this command: TOPOPT_ENV_DIR='{box.env_dir}' sh '{box.script}' sync  "
                         "After that the topopt tools work in this session.")
    assert "Do not call it again" not in text                                           # no waiting: a command fixes it


def test_the_text_when_the_setup_stopped_before_the_marker(box):
    text = status_text(box, wait="6", FAKE_UV_FAIL_VENV="error: no python")
    assert not box.marker.exists()
    assert "stopped before it was ready" in text and "setup.log" in text and "cut or failed" not in text and "' sync  After that" in text
    assert "error: no python" in (box.env_dir / "setup.log").read_text()


def test_the_text_when_uv_is_missing(box):
    if shutil.which("uv", path="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"):
        pytest.skip("a real uv is in a standard folder")
    r = subprocess.run([*SH, str(box.script), "mcp"], capture_output=True, text=True, env=no_uv_env(box, TOPOPT_MCP_WAIT="6"), input=CALL[0] + "\n")   # a wait: the early stop is seen
    assert r.returncode == 1
    text = text_of(json.loads(r.stdout.splitlines()[0]))
    assert "uv is missing" in text and "automatic download failed" in text and "setup.log" in text
    assert "show the user this command and ask if they agree" in text and "curl -LsSf https://astral.sh/uv/install.sh | sh" in text
    assert "' sync  After that the topopt tools work in this session." in text
    assert "did not work: the download failed" in (box.env_dir / "setup.log").read_text()    # the reason is in the log
    wait_for(lambda: not (box.env_dir / ".setup-lock").exists(), 30)


def test_the_text_when_the_environment_folder_cannot_be_made(box, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    text = status_text(box, TOPOPT_ENV_DIR=str(blocker / "env"))      # a folder cannot be made below a file
    assert "cannot be made" in text and "plugin data folder" in text and text.endswith("start the server again (/mcp).")


@pytest.mark.parametrize("case", ["running", "cut", "stopped", "folder"])
def test_every_text_is_one_line_that_gives_facts_and_one_next_step(box, tmp_path, case):
    """No must, never or always. One line of plain text, one next step."""
    extra = {"running": {}, "cut": {"FAKE_UV_FAIL_LOCK": NETWORK_ERROR}, "stopped": {"FAKE_UV_FAIL_VENV": "error: no python"},
             "folder": {"TOPOPT_ENV_DIR": str(tmp_path / "a-file" / "env")}}[case]
    if case == "folder":
        (tmp_path / "a-file").write_text("")
    text = status_text(box, wait="6" if case in ("cut", "stopped") else "2", **extra)
    assert "\n" not in text and 60 < len(text) <= 1000, (len(text), text)
    assert not re.search(r"\b(must|never|always)\b", text, re.I)
    assert text.count("Next step:") == 1
    box.hold.touch()
    assert wait_for(lambda: not (box.env_dir / ".setup-lock").exists(), 30)


def test_the_start_time_is_written_in_the_lock_folder_by_the_setup(box):
    r = box.run("sync")
    assert r.returncode == 0                                            # the lock is removed at the end: check the code, not the folder
    assert 'date +%s >"$LOCK/started"' in box.script.read_text()


def test_a_terminal_on_stdin_means_a_person_and_nothing_is_read(box):
    """Run by hand in a terminal, `mcp` prints its stderr lines and exits with 1. It does not wait for typed lines."""
    import pty   # POSIX only, like the rest of this file
    master, slave = pty.openpty()
    try:
        r = subprocess.run([*SH, str(box.script), "mcp"], stdin=slave, capture_output=True, text=True, timeout=30,
                           env=dict(box.env, FAKE_UV_SLEEP="1", TOPOPT_MCP_WAIT="1", FAKE_UV_HOLD=str(box.hold)))
    finally:
        os.close(master)
        os.close(slave)
    assert r.returncode == 1 and r.stdout == "" and "not ready yet" in r.stderr
    finish_the_setup(box)


def shell_function(source, name):
    """The text of a function of the script: one line (`name() { ...; }`), or up to the line that is only `}`."""
    return re.search(rf"^{name}\(\) \{{(?:[^\n]*\}}$|.*?^\}})", source, re.M | re.S).group(0)


# --- #370: the hand-off ---------------------------------------------------------------------------------------------------------

import queue
import threading


class Session:
    """A client of the launcher: it starts `mcp`, sends lines, and reads the lines of stdout as they come."""

    def __init__(self, box, **extra):
        env = dict(box.env, FAKE_UV_SLEEP="1", TOPOPT_MCP_WAIT="0")
        if "FAKE_UV_FAIL_LOCK" not in extra:
            env["FAKE_UV_HOLD"] = str(box.hold)                       # as in `not_ready_run`
        env.update(extra)
        env = {k: v for k, v in env.items() if v is not None}         # TOPOPT_MCP_WAIT=None: the variable is not set
        self.p = subprocess.Popen([*SH, str(box.script), "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, env=env)
        self.lines, self.err = queue.Queue(), []
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=lambda: self.err.extend(self.p.stderr), daemon=True).start()

    def _pump_stdout(self):
        for line in self.p.stdout:
            self.lines.put(line)
        self.lines.put(None)                                          # the end of stdout

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def recv(self, timeout=20):
        line = self.lines.get(timeout=timeout)
        assert line is not None, "stdout ended"
        return json.loads(line)

    def quiet(self, seconds):
        try:
            line = self.lines.get(timeout=seconds)
        except queue.Empty:
            return
        if line is None:                                              # stdout ended
            return
        raise AssertionError(f"a line came while none was expected: {line!r}")

    def end(self, timeout=30):
        """The client closes stdin. Returns the exit code."""
        self.p.stdin.close()
        return self.p.wait(timeout)


@pytest.fixture
def session(box):
    made = []

    def start(**extra):
        made.append(Session(box, **extra))
        return made[-1]

    yield start
    for s in made:
        if s.p.poll() is None:
            s.p.kill()
            s.p.wait()
    box.hold.touch()


def handoff_log(path):
    """What the fake Python wrote: the lines by kind."""
    lines = path.read_text().splitlines()
    kinds = ("argv", "file", "drop", "era", "listen", "replay", "stdin", "umask", "mode")
    return {k: [x[len(k) + 2:] for x in lines if x.startswith(k + ": ")] for k in kinds}, lines[-1:]


def test_modern_the_request_after_ready_goes_to_python_with_the_era_and_the_listen_id(box, session, tmp_path):
    log = tmp_path / "handoff.log"
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    s.send(modern("server/discover", "server-discover-probe-1"))
    assert s.recv()["id"] == "server-discover-probe-1"
    s.send(modern("subscriptions/listen", "listen:0", **LISTEN_PARAMS))
    assert s.recv()["method"] == "notifications/subscriptions/acknowledged"
    s.send(modern("tools/list", 2))
    assert len(s.recv()["result"]["tools"]) > 20                      # the snapshot, while the setup runs
    s.quiet(1.5)                                                      # the shell writes nothing by itself: no watcher, no notification
    finish_the_setup(box)
    s.quiet(1.5)
    pending = modern("tools/list", 3)
    s.send(pending)
    assert s.end() == 0
    got, last = handoff_log(log)
    assert got["argv"] == ["-P -m topopt_mcp.server"] and got["replay"] == [pending] and got["stdin"] == [] and last == ["end"]
    assert got["drop"] == ["unset"]                                   # the modern era drops no answer
    assert got["era"] == ["modern"] and got["listen"] == ['"listen:0"']    # the id as JSON text
    assert got["file"][0].startswith(str(box.env_dir)) and not Path(got["file"][0]).exists()   # the file is in the env dir; the Python takes and deletes it


def test_a_client_start_does_not_wait_for_the_setup_when_TOPOPT_MCP_WAIT_is_not_set(box, session):
    """The default wait is 0 when stdin is not a terminal: the snapshot answers at once. No TOPOPT_MCP_WAIT, a held setup."""
    s = session(TOPOPT_MCP_WAIT=None)
    s.send(modern("server/discover", "server-discover-probe-1"))
    assert s.recv(timeout=2)["result"]["supportedVersions"] == ["2026-07-28"]
    s.send(modern("tools/list", 1))
    assert len(s.recv(timeout=2)["result"]["tools"]) > 20
    assert s.end() == 1
    s2 = session(TOPOPT_MCP_WAIT=None)
    s2.send(legacy_initialize("2025-11-25", 0))
    assert s2.recv(timeout=2)["result"]["protocolVersion"] == "2025-11-25"
    assert s2.end() == 1


def test_a_second_client_start_during_the_setup_answers_at_once_too(box, session):
    first = session(TOPOPT_MCP_WAIT=None)
    first.send(modern("tools/list", 1))
    first.recv(timeout=2)
    second = session(TOPOPT_MCP_WAIT=None)                           # the setup of the first start still runs
    second.send(modern("tools/list", 1))
    assert len(second.recv(timeout=2)["result"]["tools"]) > 20
    first.end()
    second.end()


def test_the_wait_variable_wins_over_the_default_for_a_client(box, session):
    s = session(TOPOPT_MCP_WAIT="3")
    s.send(modern("tools/list", 1))
    started = time.monotonic()
    s.recv(timeout=10)
    assert time.monotonic() - started >= 1.5                          # it waited (the answer comes after the wait of 3 s from the start)
    s.end()


def test_a_person_in_a_terminal_waits_24_s_by_default_and_the_variable_wins(box):
    import pty
    master, slave = pty.openpty()
    env = {k: v for k, v in dict(box.env, FAKE_UV_SLEEP="1", FAKE_UV_HOLD=str(box.hold)).items() if k != "TOPOPT_MCP_WAIT"}
    p = subprocess.Popen([*SH, str(box.script), "mcp"], stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        time.sleep(4)
        assert p.poll() is None                                       # still in the wait of 24 s
    finally:
        p.kill()
        p.wait()
    r = subprocess.run([*SH, str(box.script), "mcp"], stdin=slave, capture_output=True, text=True, timeout=30,
                       env=dict(env, TOPOPT_MCP_WAIT="0"))
    os.close(master)
    os.close(slave)
    assert r.returncode == 1 and "not ready yet" in r.stderr           # the variable (0) wins
    finish_the_setup(box)


def test_a_setup_that_stops_at_once_is_running_for_3_s_and_then_stopped_in_the_text_of_a_tool_call(box, session):
    """With no wait, the launcher does not see the early stop. The text of a tool call does: the first 3 s count as running (the setup shell
    may not have written its lock yet), later the cause is the real one."""
    s = session(TOPOPT_MCP_WAIT=None, FAKE_UV_FAIL_VENV="error: no python")
    call = modern("tools/call", 1, name="environment", arguments={})
    s.send(call)
    first = text_of(s.recv(timeout=2))
    assert first.startswith(("topopt is still setting up", "topopt: the setup is starting"))   # "starting" until the lock has a pid
    assert wait_for(lambda: "setup failed" in (box.env_dir / "setup.log").read_text() if (box.env_dir / "setup.log").exists() else False, 30)
    time.sleep(3.5)
    s.send(modern("tools/call", 2, name="environment", arguments={}))
    text = text_of(s.recv(timeout=5))
    assert text.startswith("The Python setup of topopt stopped before it was ready.") and "' sync  After that" in text
    s.end()


def test_modern_without_a_listen_request_the_hand_off_has_no_listen_id(box, session, tmp_path):
    log = tmp_path / "handoff.log"
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    s.send(modern("tools/list", 1))
    s.recv()
    finish_the_setup(box)
    s.send(modern("tools/list", 2))
    assert s.end() == 0
    got, _ = handoff_log(log)
    assert got["era"] == ["modern"] and got["listen"] == ["unset"]


def test_modern_a_late_listen_after_ready_goes_to_python(box, session, tmp_path):
    """The client may send its listen request after the setup is ready. It is a request with an id: the hand-off, no ack from the shell."""
    log = tmp_path / "handoff.log"
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    finish_the_setup(box)
    listen = modern("subscriptions/listen", "listen:0", **LISTEN_PARAMS)
    s.send(listen)
    assert s.end() == 0
    got, _ = handoff_log(log)
    assert got["replay"] == [listen] and got["listen"] == ["unset"]
    s.quiet(0.2)                                                      # the shell wrote nothing


def test_modern_the_lines_after_the_pending_line_go_to_python_as_stdin(box, session, tmp_path):
    log = tmp_path / "handoff.log"
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    finish_the_setup(box)
    s.send(modern("tools/list", 8))
    s.send(modern("tools/list", 9))
    assert s.end() == 0
    got, _ = handoff_log(log)
    assert got["replay"] == [modern("tools/list", 8)]
    assert got["stdin"] in ([modern("tools/list", 9)], [])           # the fake reads the file first; stdin has what the shell did not read


@pytest.mark.parametrize("ident, raw", [(0, "0"), ("abc-1", '"abc-1"')])
def test_legacy_the_hand_off_replays_the_handshake_and_names_the_id_to_drop(box, session, tmp_path, ident, raw):
    log = tmp_path / "handoff.log"
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    init = legacy_initialize("2025-11-25", ident)
    s.send(init)
    assert s.recv()["id"] == ident
    s.send(INITIALIZED)
    s.send(legacy("tools/list", 1))
    assert len(s.recv()["result"]["tools"]) > 20
    finish_the_setup(box)
    pending = legacy("tools/list", 2)
    s.send(pending)
    assert s.end() == 0
    got, last = handoff_log(log)
    assert got["replay"] == [init, INITIALIZED, pending] and got["drop"] == [raw] and got["era"] == ["legacy"] and last == ["end"]


def test_legacy_the_clients_own_initialized_line_is_replayed_exactly(box, session, tmp_path):
    """The client may send params in notifications/initialized (a _meta): the replay keeps the line, it does not make its own."""
    log = tmp_path / "handoff.log"
    own = '{"jsonrpc":"2.0","method":"notifications/initialized","params":{"_meta":{"client":"own"}}}'
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    s.send(legacy_initialize())
    s.recv()
    s.send(own)
    finish_the_setup(box)
    pending = legacy("tools/list", 1)
    s.send(pending)
    assert s.end() == 0
    got, _ = handoff_log(log)
    assert got["replay"] == [legacy_initialize(), own, pending]


def test_the_hand_off_file_is_for_the_owner_only_and_the_umask_is_restored(box, session, tmp_path):
    log = tmp_path / "handoff.log"
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    finish_the_setup(box)
    s.send(modern("tools/call", 1, name="environment", arguments={"secret": "x"}))
    assert s.end() == 0
    got, _ = handoff_log(log)
    assert got["mode"] == ["1"]
    assert got["umask"] != ["0077"]                                   # Python does not inherit the tight umask


def test_legacy_a_request_before_notifications_initialized_gets_a_synthetic_initialized(box, session, tmp_path):
    log = tmp_path / "handoff.log"
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    s.send(legacy_initialize())
    s.recv()
    finish_the_setup(box)
    pending = legacy("tools/list", 1)
    s.send(pending)
    assert s.end() == 0
    got, _ = handoff_log(log)
    assert got["replay"] == [legacy_initialize(), INITIALIZED, pending] and got["drop"] == ["0"]


def test_legacy_ready_at_the_first_line_replays_only_that_line_and_drops_nothing(box, session, tmp_path):
    log = tmp_path / "handoff.log"
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    finish_the_setup(box)                                             # ready before the client says anything
    init = legacy_initialize()
    s.send(init)
    assert s.end() == 0
    got, _ = handoff_log(log)
    assert got["replay"] == [init] and got["drop"] == ["unset"]       # Python answers this initialize itself


def test_a_tool_call_is_handed_off_when_ready_and_answered_by_the_shell_before(box, session, tmp_path):
    log = tmp_path / "handoff.log"
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    s.send(modern("tools/call", 1, name="environment", arguments={}))
    assert s.recv()["result"]["isError"] is True
    assert not log.exists()
    finish_the_setup(box)
    call = modern("tools/call", 2, name="environment", arguments={})
    s.send(call)
    assert s.end() == 0
    assert handoff_log(log)[0]["replay"] == [call]


def test_no_background_job_runs_in_the_status_server(box, session):
    """The shell is the only writer: it has no watcher. So a quiet status server has no child shell and no sleep."""
    if not Path("/proc").is_dir():
        pytest.skip("needs /proc to list the processes")
    s = session()
    s.send(modern("subscriptions/listen", "listen:0", **LISTEN_PARAMS))
    s.recv()
    s.send(modern("tools/list", 1))
    s.recv()
    kids = [p for p in descendants(s.p.pid) if comm_of(p) in ("sh", "dash", "bash", "sleep")]
    assert kids == [], kids
    assert s.end() == 1
    assert not list(box.env_dir.glob(".mcp-*"))                       # no folder of the status server is left
    finish_the_setup(box)


def comm_of(pid):
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def test_a_pending_line_longer_than_an_environment_string_reaches_python_whole(box, session, tmp_path):
    log = tmp_path / "handoff.log"
    s = session(FAKE_PY_HANDOFF_LOG=str(log))
    finish_the_setup(box)
    big = modern("tools/call", 1, name="run_topopt", arguments={"yaml": "x" * 300000})
    s.send(big)
    assert s.end() == 0
    assert handoff_log(log)[0]["replay"] == [big]


def test_a_ready_start_from_the_beginning_is_not_changed_by_the_status_server(box, tmp_path):
    assert box.run("sync").returncode == 0
    log = tmp_path / "handoff.log"
    r = box.run("mcp", stdin=modern("tools/list", 1) + "\n", FAKE_PY_HANDOFF_LOG=str(log))
    assert r.returncode == 0 and r.stdout == "" and r.stderr == ""
    got, _ = handoff_log(log)
    assert got["file"] == ["unset"] and got["drop"] == ["unset"] and got["era"] == ["unset"] and got["replay"] == []
    assert got["stdin"] == [modern("tools/list", 1)]
    assert not list(box.env_dir.glob(".mcp-*"))


# End to end: the real launcher, and the real Python server (the fake `python` of the box runs the python of the test for `-m topopt_mcp.server`).

def real_server_session(box, session, tmp_path, **extra):
    return session(FAKE_PY_REAL=sys.executable, FAKE_PY_REAL_PATH=str(REPO), TOPOPT_WORKSPACE=str(tmp_path / "ws"),
                   TOPOPT_STATE_DIR=str(tmp_path / "state"), HOME=str(tmp_path / "home"), **extra)


MODERN_REQUESTS = [("server/discover", {}), ("prompts/list", {}), ("resources/list", {}), ("resources/templates/list", {}), ("tools/list", {})]


def test_end_to_end_modern_the_answers_of_the_shell_are_those_of_the_real_server_and_nothing_else_is_sent(box, session, tmp_path):
    s = real_server_session(box, session, tmp_path)
    shell, real = {}, {}
    s.send(modern("subscriptions/listen", "listen:0", **LISTEN_PARAMS))
    assert s.recv()["method"] == "notifications/subscriptions/acknowledged"
    for n, (method, params) in enumerate(MODERN_REQUESTS):
        s.send(modern(method, n, **params))
        shell[method] = s.recv()
    finish_the_setup(box)
    for n, (method, params) in enumerate(MODERN_REQUESTS):
        s.send(modern(method, 100 + n, **params))
        real[method] = s.recv(timeout=60)
    for method, _ in MODERN_REQUESTS:
        assert shell[method]["result"] == real[method]["result"], method     # the same bytes, as JSON
    s.quiet(1.0)                                                             # the safety net is silent: the snapshot is current
    assert s.end(timeout=60) == 0


def test_end_to_end_legacy_one_initialize_answer_then_the_real_server_and_nothing_else_is_sent(box, session, tmp_path):
    s = real_server_session(box, session, tmp_path)
    s.send(legacy_initialize("2025-06-18", 0))
    shell_init = s.recv()
    s.send(INITIALIZED)
    s.send(legacy("tools/list", 1))
    shell_tools = s.recv()
    finish_the_setup(box)
    s.send(legacy("tools/list", 2))
    s.send(legacy("initialize", 3, protocolVersion="2025-06-18", capabilities={}, clientInfo={"name": "t", "version": "1"}))
    answers = [s.recv(timeout=60) for _ in range(2)]
    assert [a["id"] for a in answers] == [2, 3]                              # no second answer with the id 0: it was dropped
    assert answers[0]["result"] == shell_tools["result"]                     # the same list from the shell and from Python
    assert answers[1]["result"] == shell_init["result"]                      # and the same initialize result
    s.quiet(1.0)
    assert s.end(timeout=60) == 0


# --- REL-9: the launcher gets a pinned uv when the user has none --------------------------------------------------------------
#
# How these tests download. The launcher has no switch for tests: a test cannot lower the check by an environment variable. A fake `curl`
# (and `wget`) on PATH logs its arguments, so the tests check the real command line (HTTPS only, the URL of the release asset, the time
# limit), and copies a local archive to the file that the launcher names. The pinned SHA256 values are constants of the script. A test
# works on the copy of the script in the box and rewrites those constants to the sum of its fake archive (`pin_sums`), or keeps the
# real ones, to test a wrong sum with the real check. A fake `uname` (Linux x86_64 by default) makes the platform the same on every host.

import hashlib
import io
import tarfile
from types import SimpleNamespace

FAKE_CURL = """#!/bin/sh
# Logs the arguments, one on each line, and "--" after the call, in $FAKE_DL_LOG. Copies $FAKE_DL_FILE to the file after --output.
{ printf '%s\\n' "$@"; echo --; } >> "$FAKE_DL_LOG"
[ -n "$FAKE_DL_FAIL" ] && { echo "$FAKE_DL_FAIL" >&2; exit "${FAKE_DL_RC:-6}"; }
out=
while [ $# -gt 0 ]; do
  case "$1" in --output) out=$2; shift ;; esac
  shift
done
cp "$FAKE_DL_FILE" "$out"
"""
FAKE_WGET = FAKE_CURL.replace("--output) out=$2; shift ;;", "--output-document=*) out=${1#*=} ;;")
FAKE_UNAME = """#!/bin/sh
case "$*" in
  "-s") echo "${FAKE_UNAME_S:-Linux}" ;;
  "-m") echo "${FAKE_UNAME_M:-x86_64}" ;;
  "-s -m") echo "${FAKE_UNAME_S:-Linux} ${FAKE_UNAME_M:-x86_64}" ;;
  *) exec REAL_UNAME "$@" ;;
esac
"""
FAKE_SYSCTL = """#!/bin/sh
case "$2" in
  hw.optional.arm64) echo "${FAKE_SYSCTL_ARM64:-0}" ;;
  *) exit 1 ;;
esac
"""
FAKE_TAR = """#!/bin/sh
echo "$*" >> "$FAKE_TAR_LOG"
exec REAL_TAR "$@"
"""
UV_ASSET = "x86_64-unknown-linux-musl"     # what the launcher takes on the fake Linux x86_64
UV_INSTALL_LINE = "curl -LsSf https://astral.sh/uv/install.sh | sh"


def script_value(box, name):
    m = re.search(rf"^{name}=(\S+)", box.script.read_text(), re.M)
    assert m, name
    return m.group(1)


def pin_sums(box, sha):
    """Rewrite the three pinned SHA256 constants in the box copy of the launcher."""
    text, n = re.subn(r"^(PINNED_UV_SHA256_[A-Z0-9_]+)=[0-9a-f]{64}", rf"\1={sha}", box.script.read_text(), flags=re.M)
    assert n == 3
    box.script.write_text(text)


def fake_uv_text(version):
    """The fake uv of the other tests, with a version line like a release (the launcher compares it with the pinned version)."""
    assert FAKE_UV.count('echo "uv 99.0.0-fake"') == 1
    return FAKE_UV.replace('echo "uv 99.0.0-fake"', f'echo "uv {version} (fake download)"')


def make_archive(path, asset, uv_text):
    """A tar.gz like the release asset (`uv-<asset>/uv` and `uvx`). Returns its SHA256."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data in ((f"uv-{asset}/uv", uv_text.encode()), (f"uv-{asset}/uvx", b"#!/bin/sh\nexit 0\n")):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_exec(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def dl(box, tmp_path):
    """The launcher with no uv on PATH, a fake curl that serves a fake archive, a fake uname (Linux x86_64) and a tar that logs.
    The box copy of the script has its three pinned sums set to the sum of the archive."""
    if shutil.which("uv", path="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"):
        pytest.skip("a real uv is in a standard folder")
    tools = tmp_path / "dltools"
    tools.mkdir()
    write_exec(tools / "curl", FAKE_CURL)
    write_exec(tools / "uname", FAKE_UNAME.replace("REAL_UNAME", shutil.which("uname")))
    write_exec(tools / "sysctl", FAKE_SYSCTL)
    write_exec(tools / "tar", FAKE_TAR.replace("REAL_TAR", shutil.which("tar")))
    version = script_value(box, "PINNED_UV_VERSION")
    archive = tmp_path / "uv-archive.tar.gz"
    sha = make_archive(archive, UV_ASSET, fake_uv_text(version))
    pin_sums(box, sha)
    d = SimpleNamespace(box=box, tools=tools, version=version, archive=archive, sha=sha, copy=box.env_dir / "uv" / version / "uv",
                        curl_log=tmp_path / "curl.log", tar_log=tmp_path / "tar.log")
    d.env = dict(box.env, PATH=f"{tools}:/usr/bin:/bin", HOME=str(box.root), FAKE_DL_LOG=str(d.curl_log), FAKE_DL_FILE=str(archive),
                 FAKE_TAR_LOG=str(d.tar_log), TOPOPT_SYSCTL=str(tools / "sysctl"))

    def run(*args, stdin="", **extra):
        return subprocess.run([*SH, str(box.script), *args], capture_output=True, text=True, input=stdin, env=dict(d.env, **extra), timeout=120)

    def curl_calls():
        """The arguments of each call of the fake curl (or wget)."""
        if not d.curl_log.exists():
            return []
        calls, one = [], []
        for line in d.curl_log.read_text().splitlines():
            if line == "--":
                calls.append(one)
                one = []
            else:
                one.append(line)
        return calls

    d.run, d.curl_calls, d.uv_calls = run, curl_calls, box.calls     # box.calls leaves out the calls of `--version`
    d.with_path_uv = f"{tools}:{box.bindir}:/usr/bin:/bin"       # a PATH where the fake uv of the box is found, after the fake tools
    return d


def test_a_good_download_is_unpacked_used_by_full_path_and_named_by_doctor(dl):
    r = dl.run("sync")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(dl.box.env_dir / "venv" / "bin" / "python")
    assert dl.copy.is_file() and os.access(dl.copy, os.X_OK)
    # One download: HTTPS only (also for a redirect), the URL of the release asset, fail on an HTTP error, a time limit.
    (call,) = dl.curl_calls()
    assert call[-1] == f"https://github.com/astral-sh/uv/releases/download/{dl.version}/uv-{UV_ASSET}.tar.gz"
    assert call[call.index("--proto") + 1] == "=https" and call[call.index("--proto-redir") + 1] == "=https"
    assert "--fail" in call and call[call.index("--max-time") + 1] == script_value(dl.box, "PINNED_UV_DOWNLOAD_LIMIT_S")
    assert dl.tar_log.read_text().count("\n") == 1                           # unpacked once
    # The setup ran with the downloaded uv: its three calls are in the log of the fake uv.
    calls = dl.uv_calls()
    assert len(calls) == 3 and calls[0].startswith(f"venv --python {LAUNCHER_PY} --managed-python --clear")
    assert dl.box.stamp.exists() and not dl.box.marker.exists()
    assert [p.name for p in (dl.box.env_dir / "uv").iterdir()] == [dl.version]   # no temporary folder is left
    assert "downloading uv" in r.stderr and "SHA256" not in r.stderr
    rows = doctor_rows(dl.run("doctor").stdout)
    assert rows["uv"] == f"downloaded by topopt: {dl.copy} (uv {dl.version} (fake download))"


def test_the_python_side_gets_the_downloaded_uv_in_the_variable_topopt_uv(dl, tmp_path):
    assert dl.run("sync").returncode == 0
    log = tmp_path / "env.log"
    assert dl.run("python", "-c", "x", FAKE_PY_ENV_LOG=str(log)).returncode == 0
    assert dl.run("mcp", FAKE_PY_ENV_LOG=str(log)).returncode == 0           # a ready environment: the server gets it too
    assert set(log.read_text().splitlines()) == {f"TOPOPT_UV={dl.copy}"}     # every call of the Python, also the readiness check


def test_a_uv_on_path_is_used_and_nothing_is_downloaded(dl, tmp_path):
    log = tmp_path / "env.log"
    r = dl.run("sync", PATH=dl.with_path_uv, FAKE_PY_ENV_LOG=str(log))
    assert r.returncode == 0, r.stderr
    assert dl.curl_calls() == [] and not (dl.box.env_dir / "uv").exists()
    assert len(dl.uv_calls()) == 3
    assert doctor_rows(dl.run("doctor", PATH=dl.with_path_uv).stdout)["uv"] == f"system: {dl.box.bindir}/uv (uv 99.0.0-fake)"
    assert dl.run("python", "-c", "x", PATH=dl.with_path_uv, FAKE_PY_ENV_LOG=str(log)).returncode == 0
    assert set(log.read_text().splitlines()) == {"TOPOPT_UV=unset"}          # a system uv is called by its name: the variable stays unset


def test_a_uv_on_path_wins_over_a_downloaded_copy(dl):
    assert dl.run("sync").returncode == 0                                    # a copy is there now
    shutil.rmtree(dl.box.env_dir / "venv")
    dl.box.log.write_text("")
    r = dl.run("sync", PATH=dl.with_path_uv)
    assert r.returncode == 0, r.stderr
    assert len(dl.curl_calls()) == 1                                         # the one download of the first call
    assert doctor_rows(dl.run("doctor", PATH=dl.with_path_uv).stdout)["uv"].startswith(f"system: {dl.box.bindir}/uv")


# --- #390: a uv on PATH that is older than MIN_UV_VERSION is skipped -----------------------------------------------------------
#
# The old fake uv is like uv 0.6.7 and older: it prints its version, and `uv venv --managed-python` stops with exit code 2 (the flag came
# with uv 0.6.8). It writes each call in $OLD_UV_LOG, so a test shows that the launcher only asked it for its version.

# The advice for a uv that is too old: the package managers first (`uv self update` only works for a uv from the install script).
UV_UPDATE_HOW = "with the tool that installed it (for example brew upgrade uv or pip install --upgrade uv; uv self update for a uv from the install script)"

OLD_UV = """#!/bin/sh
[ -n "$OLD_UV_LOG" ] && echo "$@" >> "$OLD_UV_LOG"
[ "$1" = "--version" ] && { echo "uv 0.5.0 (Homebrew 2024-11-07)"; exit 0; }
case "$*" in *--managed-python*) echo "error: unexpected argument '--managed-python' found" >&2; exit 2 ;; esac
exit 0
"""


@pytest.fixture
def old(dl, tmp_path):
    """`dl` plus the old fake uv in a folder of its own: `old.path` has it first on PATH (after the fake tools)."""
    folder = tmp_path / "old uv"                                            # a space in the path
    folder.mkdir()
    write_exec(folder / "uv", OLD_UV)
    dl.old_uv, dl.old_log = folder / "uv", tmp_path / "old-uv.log"
    dl.env["OLD_UV_LOG"] = str(dl.old_log)
    dl.path = f"{dl.tools}:{folder}:/usr/bin:/bin"
    dl.min = script_value(dl.box, "MIN_UV_VERSION")
    dl.old_calls = lambda: set(dl.old_log.read_text().splitlines()) if dl.old_log.exists() else set()
    dl.why = f"the uv on PATH is too old ({dl.old_uv} is uv 0.5.0; topopt needs uv {dl.min} or newer)"
    return dl


def test_an_old_uv_on_path_is_skipped_and_the_setup_downloads_and_uses_the_pinned_uv(old, tmp_path):
    before = old.old_uv.read_bytes()
    r = old.run("sync", PATH=old.path)
    assert r.returncode == 0, r.stderr
    assert f"{old.why}: downloading uv {old.version} ({UV_ASSET})" in r.stderr     # one line, with both versions and the pinned one
    assert r.stderr.count("too old") == 1
    assert len(old.curl_calls()) == 1 and old.copy.is_file()
    calls = old.uv_calls()                                                   # the downloaded uv made the environment
    assert len(calls) == 3 and calls[0].startswith(f"venv --python {LAUNCHER_PY} --managed-python --clear")
    assert old.old_calls() == {"--version"} and old.old_uv.read_bytes() == before   # the old uv was only asked, and is not changed
    rows = doctor_rows(old.run("doctor", PATH=old.path).stdout)
    assert rows["uv"] == f"downloaded by topopt: {old.copy} (uv {old.version} (fake download))"
    assert rows["uv on PATH"] == f"skipped: {old.old_uv} is uv 0.5.0, older than uv {old.min}, which topopt needs"
    log = tmp_path / "env.log"
    assert old.run("python", "-c", "x", PATH=old.path, FAKE_PY_ENV_LOG=str(log)).returncode == 0
    assert set(log.read_text().splitlines()) == {f"TOPOPT_UV={old.copy}"}   # the Python side names the downloaded uv, not the old one


def test_the_first_start_says_once_in_setup_log_that_the_old_uv_on_path_is_skipped(old):
    r = old.run("mcp", PATH=old.path, TOPOPT_MCP_WAIT="30")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "fake python: -P -m topopt_mcp.server\n"
    wait_for_the_end_of_the_background_setup(old.box)
    log = (old.box.env_dir / "setup.log").read_text()
    assert log.count(old.why) == 1 and f"{old.why}: downloading uv {old.version}" in log
    wait_for(lambda: not (old.box.env_dir / ".setup-lock").exists(), 30)


def test_status_and_doctor_name_the_old_uv_on_path_and_download_nothing(old):
    r = old.run("status", PATH=old.path)
    assert r.returncode == 1
    assert f"{old.why}. The setup downloads uv {old.version} ({UV_ASSET})" in r.stderr
    r = old.run("doctor", PATH=old.path)
    assert r.returncode == 1
    rows = doctor_rows(r.stdout)
    assert rows["uv"] == f"not found: {old.why}; the setup downloads uv {old.version} ({UV_ASSET}) into {old.box.env_dir}/uv/{old.version}"
    assert f"is not ready. The {old.why[4:]}: the setup downloads uv {old.version} first" in r.stdout    # a sentence of its own
    assert old.curl_calls() == [] and not old.box.env_dir.exists() and old.old_calls() == {"--version"}


def test_an_old_uv_on_path_gives_way_to_a_downloaded_copy(old):
    assert old.run("sync").returncode == 0                                   # no uv on PATH: a copy is downloaded
    shutil.rmtree(old.box.env_dir / "venv")
    old.box.log.write_text("")
    r = old.run("sync", PATH=old.path)
    assert r.returncode == 0, r.stderr
    assert f"{old.why}: the setup uses {old.copy} instead. The uv on PATH is not changed." in r.stderr
    assert len(old.curl_calls()) == 1 and len(old.uv_calls()) == 3          # no second download; the copy made the environment
    assert old.old_calls() == {"--version"}
    rows = doctor_rows(old.run("doctor", PATH=old.path).stdout)
    assert rows["uv"].startswith(f"downloaded by topopt: {old.copy} ") and rows["uv on PATH"].startswith(f"skipped: {old.old_uv} is uv 0.5.0")


def test_an_old_uv_on_path_and_a_failed_download_say_to_update_that_uv(old):
    r = old.run("sync", PATH=old.path, FAKE_DL_FAIL="curl: (6) Could not resolve host: github.com")
    assert r.returncode == 1 and r.stdout == ""
    assert f"The uv on PATH is too old: {old.old_uv} is uv 0.5.0, and topopt needs uv {old.min} or newer. Update it {UV_UPDATE_HOW}." in r.stderr
    assert "The automatic download of uv did not work: the download failed" in r.stderr
    assert "uv is not installed" not in r.stderr and old.old_calls() == {"--version"}


def test_doctor_says_to_update_the_old_uv_where_no_download_can_be_made(old):
    r = old.run("doctor", PATH=old.path, FAKE_UNAME_M="i686")             # a platform with no pinned uv
    assert r.returncode == 1
    assert doctor_rows(r.stdout)["uv"].startswith(f"not found: {old.why}; the setup cannot download it here (")
    assert f"is not ready. The {old.why[4:]}, and the setup cannot download a newer uv (" in r.stdout       # a sentence of its own
    assert f"first update it {UV_UPDATE_HOW}." in r.stdout and old.curl_calls() == []


def test_the_cause_text_when_the_uv_on_path_is_too_old_and_the_download_failed(old):
    r = old.run("mcp", stdin=CALL[0] + "\n", PATH=old.path, TOPOPT_MCP_WAIT="12", FAKE_DL_FAIL="curl: (6) Could not resolve host: github.com")
    assert r.returncode == 1
    text = text_of(json.loads(r.stdout.splitlines()[0]))
    assert text.startswith(f"topopt cannot set up its Python environment: {old.why}, and the automatic download of a newer uv failed")
    assert f"update that uv {UV_UPDATE_HOW}." in text and UV_INSTALL_LINE not in text
    assert text.endswith("After that the topopt tools work in this session.")
    wait_for(lambda: not (old.box.env_dir / ".setup-lock").exists(), 30)


@pytest.mark.parametrize("line, taken", [
    ("uv 0.6.8", True), ("uv 0.6.8 (Homebrew 2025-03-18)", True), ("uv 0.6.10", True), ("uv 0.10.0", True), ("uv 0.7", True),
    ("uv 1.0.0-rc1 (x86_64-unknown-linux-gnu)", True), ("uv 0.6.7 (a1b2c3d 2025-03-14)", False), ("uv 0.6", False), ("uv 0.5.31", False),
    ("uv dev", True), ("", True)])                                         # a uv that says no version is taken, as before #390
def test_the_minimum_compares_the_version_parts_as_numbers(dl, line, taken):
    text = dl.box.script.read_text()
    dl.box.script.write_text(re.sub(r"^MIN_UV_VERSION=\S+", "MIN_UV_VERSION=0.6.8", text, count=1, flags=re.M))   # this table needs 0.6.8
    write_exec(dl.box.bindir / "uv", FAKE_UV.replace('echo "uv 99.0.0-fake"', f'echo "{line}"' if line else ":"))
    rows = doctor_rows(dl.run("doctor", PATH=dl.with_path_uv).stdout)
    if taken:
        assert rows["uv"].startswith(f"system: {dl.box.bindir}/uv (") and "uv on PATH" not in rows
    else:
        version = line.split()[1]
        assert rows["uv"].startswith(f"not found: the uv on PATH is too old ({dl.box.bindir}/uv is uv {version}; topopt needs uv 0.6.8 or newer)")
    assert dl.curl_calls() == [] and dl.uv_calls() == []


def test_a_wrong_sum_is_refused_nothing_is_unpacked_and_the_message_has_the_install_command(dl):
    dl.box.script.write_text((REPO / "scripts" / "topopt-env.sh").read_text())     # the real pinned sums: the fake archive cannot match
    pinned = script_value(dl.box, "PINNED_UV_SHA256_LINUX_X86_64")
    assert pinned != dl.sha
    r = dl.run("sync")
    assert r.returncode == 1 and r.stdout == ""
    assert f"the SHA256 of the download is {dl.sha} and the pinned value is {pinned}" in r.stderr
    assert "The file was deleted and nothing was unpacked." in r.stderr
    assert f"uv is not installed. Install it with: {UV_INSTALL_LINE}" in r.stderr and "The automatic download of uv did not work" in r.stderr
    assert not dl.tar_log.exists()                                           # tar never ran
    assert list((dl.box.env_dir / "uv").iterdir()) == []                     # no archive, no folder, no copy
    assert dl.uv_calls() == [] and not (dl.box.env_dir / "venv").exists()


def test_no_environment_variable_changes_the_pinned_url_sum_version_or_limit(dl):
    dl.box.script.write_text((REPO / "scripts" / "topopt-env.sh").read_text())     # the real pins
    real_version = script_value(dl.box, "PINNED_UV_VERSION")
    r = dl.run("sync", PINNED_UV_SHA256_LINUX_X86_64=dl.sha, PINNED_UV_URL_BASE="http://127.0.0.1:1", PINNED_UV_DOWNLOAD_LIMIT_S="1",
               PINNED_UV_VERSION="9.9.9", TOPOPT_UV_URL_BASE="http://127.0.0.1:1", TOPOPT_UV_SHA256=dl.sha, TOPOPT_TEST="1")
    assert r.returncode == 1 and "The file was deleted and nothing was unpacked." in r.stderr
    (call,) = dl.curl_calls()
    assert call[-1] == f"https://github.com/astral-sh/uv/releases/download/{real_version}/uv-{UV_ASSET}.tar.gz"
    assert call[call.index("--max-time") + 1] == "300" and not dl.tar_log.exists()


def test_offline_gives_the_current_message_with_the_reason_and_leaves_nothing(dl):
    r = dl.run("sync", FAKE_DL_FAIL="curl: (6) Could not resolve host: github.com")
    assert r.returncode == 1 and r.stdout == ""
    assert f"uv is not installed. Install it with: {UV_INSTALL_LINE}" in r.stderr
    assert "The automatic download of uv did not work: the download failed (curl exit code 6; no network?): " in r.stderr
    assert "Could not resolve host" in r.stderr and "Then start this again." in r.stderr
    assert "prerequisite" not in r.stderr
    assert list((dl.box.env_dir / "uv").iterdir()) == [] and not dl.tar_log.exists()
    assert not (dl.box.env_dir / ".setup-lock").exists()
    r = dl.run("sync")                                                       # the network is back
    assert r.returncode == 0, r.stderr


def test_a_download_that_gives_an_empty_file_is_a_failure(dl, tmp_path):
    empty = tmp_path / "empty"
    empty.write_text("")
    r = dl.run("sync", FAKE_DL_FILE=str(empty))
    assert r.returncode == 1 and "the download failed" in r.stderr and not dl.tar_log.exists()


def test_a_downloaded_copy_is_used_again_without_a_new_download(dl):
    assert dl.run("sync").returncode == 0
    shutil.rmtree(dl.box.env_dir / "venv")                                   # the environment must be made again, the copy stays
    dl.box.log.write_text("")
    r = dl.run("sync")
    assert r.returncode == 0, r.stderr
    assert len(dl.curl_calls()) == 1 and "downloading uv" not in r.stderr
    assert dl.uv_calls()[0].startswith(f"venv --python {LAUNCHER_PY}")                 # the copy ran again
    assert doctor_rows(dl.run("doctor").stdout)["uv"].startswith(f"downloaded by topopt: {dl.copy} ")


@pytest.mark.parametrize("text", ["#!/bin/sh\nexit 1\n", '#!/bin/sh\necho "uv 0.0.1 (an old one)"\n', "", "#!/bin/sh\necho hello\n"])
def test_a_copy_that_does_not_run_or_has_another_version_is_not_taken(dl, text):
    dl.copy.parent.mkdir(parents=True)
    write_exec(dl.copy, text)
    r = dl.run("sync")
    assert r.returncode == 0, r.stderr
    assert len(dl.curl_calls()) == 1
    assert "fake download" in dl.copy.read_text()                            # the bad copy was replaced by the new one
    assert [p.name for p in (dl.box.env_dir / "uv").iterdir()] == [dl.version]


def test_what_a_cut_download_left_is_removed_and_never_taken(dl):
    left = dl.box.env_dir / "uv" / ".tmp-4242"
    left.mkdir(parents=True)
    (left / "uv.tar.gz").write_text("half a file")
    write_exec(left / "uv", FAKE_UV)                                         # a file that runs, in the temporary folder
    assert doctor_rows(dl.run("doctor").stdout)["uv"].startswith("not found: the setup downloads uv ")   # not taken
    assert dl.run("sync").returncode == 0
    assert not left.exists()


def test_status_and_doctor_never_download(dl):
    r = dl.run("status")
    assert r.returncode == 1 and "uv is not installed. The setup downloads uv " in r.stderr
    assert f"({UV_ASSET})" in r.stderr and "checks its SHA256" in r.stderr and "That needs the network." in r.stderr
    assert "Run: " in r.stderr
    r = dl.run("doctor")
    assert r.returncode == 1
    assert doctor_rows(r.stdout)["uv"] == f"not found: the setup downloads uv {dl.version} ({UV_ASSET}) into {dl.box.env_dir}/uv/{dl.version}"
    assert "uv is not installed: the setup downloads uv" in r.stdout
    assert dl.curl_calls() == [] and not dl.box.env_dir.exists()


def test_the_first_start_of_mcp_downloads_in_the_background_setup(dl):
    r = dl.run("mcp", TOPOPT_MCP_WAIT="30")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "fake python: -P -m topopt_mcp.server\n"                 # stdout stays the protocol
    assert dl.copy.is_file() and len(dl.curl_calls()) == 1
    wait_for_the_end_of_the_background_setup(dl.box)
    assert "uv is not installed: downloading uv" in (dl.box.env_dir / "setup.log").read_text()   # the download is a step of the setup
    wait_for(lambda: not (dl.box.env_dir / ".setup-lock").exists(), 30)


def test_a_failed_download_at_the_first_start_of_mcp_gives_the_cause_text(dl):
    r = dl.run("mcp", stdin=CALL[0] + "\n", TOPOPT_MCP_WAIT="12", FAKE_DL_FAIL="curl: (6) Could not resolve host: github.com")
    assert r.returncode == 1
    answer = json.loads(r.stdout.splitlines()[0])
    text = text_of(answer)
    assert answer["result"]["isError"] is True
    assert text.startswith("topopt cannot set up its Python environment: uv is missing and its automatic download failed")
    assert "show the user this command and ask if they agree" in text and UV_INSTALL_LINE in text
    assert text.endswith("After that the topopt tools work in this session.")
    log = (dl.box.env_dir / "setup.log").read_text()
    assert "Could not resolve host" in log and f"Install it with: {UV_INSTALL_LINE}" in log
    wait_for(lambda: not (dl.box.env_dir / ".setup-lock").exists(), 30)


@pytest.mark.parametrize("s, m, arm64, key, asset", [
    ("Darwin", "arm64", "0", "MACOS_ARM64", "aarch64-apple-darwin"),
    ("Darwin", "x86_64", "0", "MACOS_X86_64", "x86_64-apple-darwin"),
    ("Darwin", "x86_64", "1", "MACOS_ARM64", "aarch64-apple-darwin"),             # an Apple silicon Mac under Rosetta: like the official installer
    ("Linux", "x86_64", "0", "LINUX_X86_64", "x86_64-unknown-linux-musl")])
def test_the_asset_and_its_own_pinned_sum_follow_the_platform(dl, s, m, arm64, key, asset):
    sums = {"MACOS_ARM64": "1" * 64, "MACOS_X86_64": "2" * 64, "LINUX_X86_64": "3" * 64}
    text = dl.box.script.read_text()
    for k, v in sums.items():
        text = re.sub(rf"^(PINNED_UV_SHA256_{k})=[0-9a-f]{{64}}", rf"\1={v}", text, flags=re.M)
    dl.box.script.write_text(text)
    r = dl.run("sync", FAKE_UNAME_S=s, FAKE_UNAME_M=m, FAKE_SYSCTL_ARM64=arm64)    # the fake archive matches none of them
    assert r.returncode == 1
    (call,) = dl.curl_calls()
    assert call[-1] == f"https://github.com/astral-sh/uv/releases/download/{dl.version}/uv-{asset}.tar.gz"
    assert f"the pinned value is {sums[key]}" in r.stderr
    assert not dl.tar_log.exists()


@pytest.mark.parametrize("s, m", [("Linux", "aarch64"), ("Linux", "i686"), ("FreeBSD", "amd64"), ("Darwin", "ppc")])
def test_another_platform_gets_no_download_and_the_install_message(dl, s, m):
    r = dl.run("sync", FAKE_UNAME_S=s, FAKE_UNAME_M=m)
    assert r.returncode == 1
    assert f"uv is not installed. Install it with: {UV_INSTALL_LINE}" in r.stderr
    assert f"The automatic download of uv did not work: there is no pinned uv download for this platform ({s} {m})" in r.stderr
    assert dl.curl_calls() == []
    assert doctor_rows(dl.run("doctor", FAKE_UNAME_S=s, FAKE_UNAME_M=m).stdout)["uv"].startswith("not found: the setup cannot download it here (")
    r = dl.run("status", FAKE_UNAME_S=s, FAKE_UNAME_M=m)
    assert r.returncode == 1 and UV_INSTALL_LINE in r.stderr and "The setup downloads" not in r.stderr


def dl_farm(tmp_path, names, **scripts):
    """A folder that is the whole PATH of a run: links to the tools by name, and the given scripts (fake tools). A PATH without curl
    cannot be made from /usr/bin, which has it."""
    farm = tmp_path / "dl-farm"
    farm.mkdir()
    for n in names:
        found = shutil.which(n)
        if not found:
            pytest.skip(f"{n} is not available")
        (farm / n).symlink_to(found)
    for n, text in scripts.items():
        write_exec(farm / n, text)
    return farm


DL_FARM_TOOLS = ["dirname", "gzip", "sed", "cksum", "cut", "mkdir", "cat", "rm", "date", "grep", "tail", "head", "basename", "tar", "mv",
              "chmod", "cp", "sleep", "find", "tr"]


def sha_tool():
    return "sha256sum" if shutil.which("sha256sum") else "shasum"


def farm_run(dl, tmp_path, tools, *args, **scripts):
    """A launcher run whose whole PATH is a folder with `tools` and the given fake scripts. The launcher adds some folders to the end of
    PATH (/usr/local/bin, /opt/homebrew/bin: a curl or a sha256sum there would be found), so the box copy of the script gets an empty list
    of extra folders. That feature is tested by the tests of the start method."""
    extra = 'for d in "$HOME/.local/bin" "$HOME/.cargo/bin" /opt/homebrew/bin /usr/local/bin; do'
    text = dl.box.script.read_text()
    assert text.count(extra) == 1
    dl.box.script.write_text(text.replace(extra, "for d in; do"))
    scripts.setdefault("uname", FAKE_UNAME.replace("REAL_UNAME", shutil.which("uname")))      # the platform is Linux x86_64 on every host
    farm = dl_farm(tmp_path, tools, **scripts)
    env = dict(dl.env, PATH=str(farm))
    return subprocess.run([*SH_ABS, str(dl.box.script), *args], capture_output=True, text=True, env=env, timeout=120)


def test_wget_is_the_fallback_when_there_is_no_curl(dl, tmp_path):
    r = farm_run(dl, tmp_path, DL_FARM_TOOLS + [sha_tool()], "sync", wget=FAKE_WGET)
    assert r.returncode == 0, r.stderr
    (call,) = dl.curl_calls()
    assert call[-1] == f"https://github.com/astral-sh/uv/releases/download/{dl.version}/uv-{UV_ASSET}.tar.gz"
    assert "--https-only" in call and f"--timeout={script_value(dl.box, 'PINNED_UV_DOWNLOAD_LIMIT_S')}" in call
    assert dl.copy.is_file()


def test_no_curl_and_no_wget_gives_the_install_message(dl, tmp_path):
    r = farm_run(dl, tmp_path, DL_FARM_TOOLS + [sha_tool()], "sync")
    assert r.returncode == 1
    assert "The automatic download of uv did not work: neither curl nor wget is installed" in r.stderr and UV_INSTALL_LINE in r.stderr


def test_no_tool_for_the_sha256_gives_the_install_message_and_downloads_nothing(dl, tmp_path):
    r = farm_run(dl, tmp_path, DL_FARM_TOOLS, "sync", curl=FAKE_CURL)
    assert r.returncode == 1
    assert "neither sha256sum nor shasum is installed, so a download could not be checked" in r.stderr
    assert dl.curl_calls() == []                                             # nothing is fetched that could not be checked


def test_topopt_uv_names_the_uv_and_wins_over_path(dl, tmp_path):
    mine = tmp_path / "my folder" / "uv"                                     # a space in the path
    mine.parent.mkdir()
    write_exec(mine, FAKE_UV.replace('echo "uv 99.0.0-fake"', 'echo "uv 9.9.9-mine"'))
    log = tmp_path / "env.log"
    for path in (dl.env["PATH"], dl.with_path_uv):                          # with no other uv, and with a uv on PATH
        shutil.rmtree(dl.box.env_dir, ignore_errors=True)
        r = dl.run("sync", TOPOPT_UV=str(mine), PATH=path, FAKE_PY_ENV_LOG=str(log))
        assert r.returncode == 0, r.stderr
        assert doctor_rows(dl.run("doctor", TOPOPT_UV=str(mine), PATH=path).stdout)["uv"] == f"TOPOPT_UV: {mine} (uv 9.9.9-mine)"
        assert len(dl.uv_calls()) >= 3
    assert dl.curl_calls() == []
    dl.run("python", "-c", "x", TOPOPT_UV=str(mine), FAKE_PY_ENV_LOG=str(log))
    assert log.read_text().splitlines()[-1] == f"TOPOPT_UV={mine}"            # it reaches the Python side as it is


@pytest.mark.parametrize("value, why", [("/no/such/folder/uv", "that file is not a uv that runs"), ("uv", "it is not a full path"),
                                         ("TRUE", "that file is not a uv that runs"),     # TRUE: the `true` of this machine runs, and prints no uv line
                                         ("OLD", "it is uv 0.5.0, and topopt needs uv {min} or newer")])   # OLD: a uv that runs and is too old (#390)
def test_topopt_uv_that_does_not_work_is_an_error_and_no_other_uv_is_tried(dl, tmp_path, value, why):
    if value == "TRUE":
        value = shutil.which("true")
    if value == "OLD":
        value = str(tmp_path / "my folder" / "uv")
        Path(value).parent.mkdir()
        write_exec(Path(value), OLD_UV)
    why = why.format(min=script_value(dl.box, "MIN_UV_VERSION"))
    r = dl.run("sync", TOPOPT_UV=value, PATH=dl.with_path_uv)                # a working uv is on PATH: it must not be used
    assert r.returncode == 1 and r.stdout == ""
    assert f"TOPOPT_UV is set to '{value}', but {why}." in r.stderr
    assert "tries no other uv" in r.stderr
    assert dl.uv_calls() == [] and dl.curl_calls() == [] and not dl.box.env_dir.exists()
    assert dl.run("status", TOPOPT_UV=value, PATH=dl.with_path_uv).returncode == 1
    out = dl.run("doctor", TOPOPT_UV=value, PATH=dl.with_path_uv).stdout
    assert doctor_rows(out)["uv"].startswith(f"TOPOPT_UV is set to '{value}', but {why}")
    assert "TOPOPT_UV is set but" in out                                     # the result line names it too


def test_a_bad_topopt_uv_does_not_stop_a_ready_environment(dl):
    assert dl.run("sync", PATH=dl.with_path_uv).returncode == 0
    assert dl.run("status", TOPOPT_UV="/no/such/uv").returncode == 0
    r = dl.run("mcp", TOPOPT_UV="/no/such/uv")
    assert r.returncode == 0 and r.stdout == "fake python: -P -m topopt_mcp.server\n"


def test_the_cause_text_when_topopt_uv_does_not_work(dl):
    r = dl.run("mcp", stdin=CALL[0] + "\n", TOPOPT_UV="/no/such/uv")
    assert r.returncode == 1
    text = text_of(json.loads(r.stdout.splitlines()[0]))
    assert text.startswith("topopt cannot set up its Python environment: the variable TOPOPT_UV names a uv that does not run.")
    assert text.endswith("start the server again (/mcp).")
    assert dl.curl_calls() == [] and not (dl.box.env_dir / ".setup-lock").exists()


def test_the_cause_text_when_topopt_uv_is_too_old(old):
    r = old.run("mcp", stdin=CALL[0] + "\n", TOPOPT_UV=str(old.old_uv))
    assert r.returncode == 1
    text = text_of(json.loads(r.stdout.splitlines()[0]))
    assert text.startswith(f"topopt cannot set up its Python environment: the variable TOPOPT_UV is set, but it is uv 0.5.0, and topopt needs "
                           f"uv {old.min} or newer. Next step: tell the user to set it to the full path of a uv that runs (uv {old.min} or newer)")
    assert old.curl_calls() == [] and old.old_calls() == {"--version"}


def test_the_script_has_no_switch_that_changes_the_pinned_download():
    """The launcher reads no environment variable for the URL, the sums or the limit: only TOPOPT_UV (another uv) is read."""
    text = (REPO / "scripts" / "topopt-env.sh").read_text()
    for name in ("PINNED_UV_VERSION", "PINNED_UV_URL_BASE", "PINNED_UV_SHA256_MACOS_ARM64", "PINNED_UV_SHA256_MACOS_X86_64",
                 "PINNED_UV_SHA256_LINUX_X86_64", "PINNED_UV_DOWNLOAD_LIMIT_S", "MIN_UV_VERSION"):
        assert re.search(rf"^{name}=[^$\s]+", text, re.M), name              # a plain assignment of a literal, never ${VAR:-...}
    assert re.search(r"^PINNED_UV_URL_BASE=https://github\.com/astral-sh/uv/releases/download$", text, re.M)
    assert "TOPOPT_TEST" not in text and "TOPOPT_UV_" not in text            # no test switch
    assert not re.search(r"\$\{PINNED_UV_\w+:", text)                       # no default taken from the environment


def test_the_pinned_sums_are_three_different_sha256_values_and_the_version_is_a_release_number():
    text = (REPO / "scripts" / "topopt-env.sh").read_text()
    sums = re.findall(r"^PINNED_UV_SHA256_[A-Z0-9_]+=([0-9a-f]{64})\b", text, re.M)
    assert len(sums) == 3 and len(set(sums)) == 3
    assert re.search(r"^PINNED_UV_VERSION=\d+\.\d+\.\d+$", text, re.M)


def test_the_minimum_uv_is_a_release_number_next_to_the_pinned_version_with_the_rule_to_raise_it():
    lines = (REPO / "scripts" / "topopt-env.sh").read_text().splitlines()
    (pin,) = [i for i, line in enumerate(lines) if line.startswith("PINNED_UV_VERSION=")]
    (low,) = [i for i, line in enumerate(lines) if line.startswith("MIN_UV_VERSION=")]
    assert re.fullmatch(r"MIN_UV_VERSION=\d+\.\d+\.\d+", lines[low])
    assert 0 < low - pin <= 10 and "To raise it" in "\n".join(lines[pin:low])    # next to the pinned version, after its comment
    version = [tuple(int(x) for x in lines[i].split("=")[1].split(".")) for i in (low, pin)]
    assert version[0] <= version[1]                                          # the pinned uv that the launcher downloads is new enough


def test_every_call_of_uv_in_the_script_uses_the_full_path():
    """After REL-9 no line of the launcher runs a bare `uv` (the comments and the messages for people are not code)."""
    for n, line in enumerate((REPO / "scripts" / "topopt-env.sh").read_text().splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        assert not re.search(r"\b(run|run_logged)\b.*\buv (venv|pip)\b", line), (n, line)
        assert not re.search(r"^\s*uv\s|\$\(uv\s|\|\s*uv\s", line), (n, line)


# --- plugin-sync: the setup of the Claude Code plugin before its first start (#357) ------------------------------------------

def installed_copy(box, tmp_path, marketplace="topopt", installed=("topopt@topopt",)):
    """The box root as a copy that Claude Code installed: the two manifests (the marketplace name can differ), a config folder
    whose installed_plugins.json names the ids in `installed`, and an environment without TOPOPT_ENV_DIR (as in Claude Code)."""
    (box.root / ".claude-plugin").mkdir()
    shutil.copy(REPO / ".claude-plugin" / "plugin.json", box.root / ".claude-plugin" / "plugin.json")
    market = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    market["name"] = marketplace
    (box.root / ".claude-plugin" / "marketplace.json").write_text(json.dumps(market, indent=2))
    config = tmp_path / "claude-config"
    (config / "plugins").mkdir(parents=True)
    entries = {i: [{"scope": "user", "installPath": str(box.root), "version": "0.0.0"}] for i in installed}
    (config / "plugins" / "installed_plugins.json").write_text(json.dumps({"version": 2, "plugins": entries}, indent=2))
    env = {k: v for k, v in box.env.items() if k not in ("TOPOPT_ENV_DIR", "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_PLUGIN_CACHE_DIR")}
    env["CLAUDE_CONFIG_DIR"] = str(config)
    return config, env


def plugin_sync(box, env, **extra):
    return subprocess.run([*SH, str(box.script), "plugin-sync"], capture_output=True, text=True, env=dict(env, **extra), timeout=60)


def test_plugin_sync_makes_the_environment_that_the_first_start_of_the_plugin_server_finds_ready(box, tmp_path):
    config, env = installed_copy(box, tmp_path)
    r = plugin_sync(box, env)
    assert r.returncode == 0, r.stderr
    data = config / "plugins" / "data" / "topopt-topopt"
    assert f"the plugin data folder of topopt@topopt: {data}" in r.stderr
    assert r.stderr.rstrip().splitlines()[-1].startswith("topopt: ready (pins=lock): ")
    assert (data / "venv" / ".topopt-plugin-stamp").is_file() and not box.env_dir.exists()
    calls = len(box.calls())
    # The first start of the server, as Claude Code makes it: CLAUDE_PLUGIN_DATA is that folder. It starts the server at once.
    server_env = {k: v for k, v in env.items() if k != "CLAUDE_CONFIG_DIR"}
    first = subprocess.run([*SH, str(box.script), "mcp"], capture_output=True, text=True, timeout=60,
                           env=dict(server_env, CLAUDE_PLUGIN_DATA=str(data), TOPOPT_MCP_WAIT="0"))
    assert first.returncode == 0, first.stderr
    assert first.stdout == "fake python: -P -m topopt_mcp.server\n" and first.stderr == ""
    assert len(box.calls()) == calls                                  # no install call of uv: nothing to install at the first start
    again = plugin_sync(box, env)                                     # a second run finds it ready
    assert again.returncode == 0 and len(box.calls()) == calls


def test_plugin_sync_takes_the_config_folder_of_claude_code_from_home_without_claude_config_dir(box, tmp_path):
    config, env = installed_copy(box, tmp_path)
    home = Path(env["HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / ".claude").mkdir()
    shutil.move(str(config / "plugins"), str(home / ".claude" / "plugins"))
    del env["CLAUDE_CONFIG_DIR"]
    r = plugin_sync(box, env)
    assert r.returncode == 0, r.stderr
    assert (home / ".claude" / "plugins" / "data" / "topopt-topopt" / "venv" / ".topopt-plugin-stamp").is_file()


def test_plugin_sync_takes_the_plugin_cache_folder_of_claude_code_first(box, tmp_path):
    config, env = installed_copy(box, tmp_path)
    cache = tmp_path / "plugin-cache"
    shutil.move(str(config / "plugins"), str(cache))
    r = plugin_sync(box, env, CLAUDE_CODE_PLUGIN_CACHE_DIR=str(cache))
    assert r.returncode == 0, r.stderr
    assert (cache / "data" / "topopt-topopt" / "venv" / ".topopt-plugin-stamp").is_file()
    assert not (config / "plugins").exists()


def test_plugin_sync_reads_a_leading_tilde_in_the_plugin_cache_folder_as_the_home_folder(box, tmp_path):
    config, env = installed_copy(box, tmp_path)
    home = Path(env["HOME"])
    home.mkdir(parents=True, exist_ok=True)
    shutil.move(str(config / "plugins"), str(home / "plugin-cache"))
    r = plugin_sync(box, env, CLAUDE_CODE_PLUGIN_CACHE_DIR="~/plugin-cache")
    assert r.returncode == 0, r.stderr
    assert (home / "plugin-cache" / "data" / "topopt-topopt" / "venv" / ".topopt-plugin-stamp").is_file()


@pytest.mark.parametrize("marketplace, folder", [("topopt-demo", "topopt-topopt-demo"), ("team.tools_2", "topopt-team-tools_2")])
def test_plugin_sync_names_the_folder_as_claude_code_does(box, tmp_path, marketplace, folder):
    """Claude Code replaces each character other than a letter, a digit, "_" and "-" of <plugin>@<marketplace> by "-"."""
    config, env = installed_copy(box, tmp_path, marketplace=marketplace, installed=(f"topopt@{marketplace}",))
    r = plugin_sync(box, env)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (config / "plugins" / "data").iterdir()) == [folder]


def test_plugin_sync_stops_when_that_claude_code_has_not_installed_the_plugin(box, tmp_path):
    config, env = installed_copy(box, tmp_path, installed=("topopt@topopt-demo", "other@topopt"))
    r = plugin_sync(box, env)
    assert r.returncode == 1
    assert "Claude Code has no plugin topopt@topopt here" in r.stderr
    assert "CLAUDE_CONFIG_DIR" in r.stderr and "CLAUDE_CODE_PLUGIN_CACHE_DIR" in r.stderr
    assert box.calls() == [] and not (config / "plugins" / "data").exists()
    (config / "plugins" / "installed_plugins.json").unlink()            # no plugin installed at all
    assert plugin_sync(box, env).returncode == 1 and box.calls() == []


def test_plugin_sync_stops_in_a_copy_without_the_plugin_manifests(box, tmp_path):
    env = {k: v for k, v in box.env.items() if k != "TOPOPT_ENV_DIR"}
    r = plugin_sync(box, env, CLAUDE_CONFIG_DIR=str(tmp_path / "claude-config"))
    assert r.returncode == 1 and "plugin-sync is for a copy that Claude Code installed as a plugin" in r.stderr
    assert box.calls() == []


def test_plugin_sync_keeps_the_folder_rule_of_the_server_when_topopt_env_dir_is_set(box, tmp_path):
    config, env = installed_copy(box, tmp_path)
    r = plugin_sync(box, env, TOPOPT_ENV_DIR=str(box.env_dir))
    assert r.returncode == 0, r.stderr
    assert f"the setup uses {box.env_dir}" in r.stderr
    assert (box.env_dir / "venv" / ".topopt-plugin-stamp").is_file() and not (config / "plugins" / "data").exists()


def test_the_first_name_in_the_plugin_manifests_is_the_top_level_name():
    """plugin-sync reads the first "name" line of the two files (no JSON parser in sh): it must be the plugin and marketplace name."""
    for name in ("plugin.json", "marketplace.json"):
        path = REPO / ".claude-plugin" / name
        first = re.search(r'^\s*"name"\s*:\s*"([^"]*)"', path.read_text(), re.M)
        assert first and first.group(1) == json.loads(path.read_text())["name"], name


# --- #422: the texts that name the Python of the launcher say the version of LAUNCHER_PYTHON ------------------------------

@pytest.mark.parametrize("path, needle", [
    ("README.md", "**Python {v}** through uv"),
    ("README.md", "once when the Python of the launcher changes (the update to Python {v} downloads Python {v} and the packages again"),
    ("docs/reference/install.md", "and the setup downloads Python {v} and the packages again (about 180 MB) and needs the network"),
    ("docs/reference/install.md", "The launcher and the desktop extension make Python {v} by themselves"),
    ("docs/reference/install.md", "It downloads Python {v}\n"),
    ("docs/reference/install.md", "| Python {v} | Only if uv has no Python {v} of its own |"),
    ("docs/reference/any-mcp-client.md", "(Python {v}, and 55 packages)"),
    ("docs/reference/any-mcp-client.md", "The launcher gets Python {v} itself."),
    ("docs/reference/any-mcp-client.md", "step 1 makes the Python {v}\nenvironment"),
    ("docs/clients.md", "The launcher gets Python {v} itself."),
    ("skills/start-here/SKILL.md", "uv gets Python {v} itself."),
    ("skills/start-here/SKILL.md", "the server downloads Python {v} and the packages"),
    (".agents/skills/start-here/SKILL.md", "uv gets Python {v} itself."),
    ("packaging/mcpb/manifest.json", "downloads Python {v} and about 200 MB"),
    ("CONTRACTS.md", "extension make Python {v} by themselves"),
    ("docs/reference/install.md", "uv run --no-project --python {v} packaging/mcpb/build.py"),
    ("packaging/mcpb/build.py", "uv run --no-project --python {v} packaging/mcpb/build.py"),
])
def test_the_texts_that_name_the_python_of_the_launcher_say_its_version(path, needle):
    assert needle.format(v=LAUNCHER_PY) in (REPO / path).read_text(), path


def test_the_launcher_python_is_also_in_the_error_texts_and_the_dev_install_text():
    assert f"venv --python {LAUNCHER_PY} &&" in (REPO / "topopt" / "gpu_check.py").read_text()
    assert f"uv venv --python {LAUNCHER_PY}\n" in (REPO / "docs" / "reference" / "install.md").read_text()
    # No text that says what the launcher makes names another version.
    for path in ("README.md", "docs/clients.md", "docs/reference/any-mcp-client.md", "docs/reference/install.md", "skills/start-here/SKILL.md"):
        for line in (REPO / path).read_text().splitlines():
            if re.search(r"(launcher|setup|extension) (makes|gets|downloads) Python 3\.\d+", line):
                assert f"Python {LAUNCHER_PY}" in line, (path, line)
