#!/bin/sh
# Run topopt from its own Python environment. It needs uv: the uv of the user if there is one, else a pinned release that
# this script downloads and checks (see "The uv" below).
#
#   sh topopt-env.sh sync         make or refresh the environment (waits until it is ready)
#   sh topopt-env.sh status       say if the environment is ready (exit 0), or not (exit 1)
#   sh topopt-env.sh doctor       print the facts of the setup (stdout) and the next command to run
#                                 (exit 0: ready; exit 1: an action is needed). It changes nothing.
#   sh topopt-env.sh mcp          start the MCP server (the Claude Code plugin does this)
#   sh topopt-env.sh plugin-sync  `sync` in the folder that Claude Code gives this plugin, before the first start of the
#                                 installed plugin (see "plugin-sync" below)
#   sh topopt-env.sh python ...   run the environment Python
#   sh topopt-env.sh topopt ...   run the topopt command line
#
# The uv (REL-9). Which uv the launcher uses, in this order:
#   1. $TOPOPT_UV, a full path to a uv. If it is set and that uv does not run, or is older than MIN_UV_VERSION, the launcher says so and
#      stops (it tries no other).
#   2. a `uv` on PATH (it wins over a downloaded copy). The folders that this script adds at the end of PATH (below) count as PATH.
#      A uv on PATH that says a version older than MIN_UV_VERSION is skipped (#390): the launcher says so once in the setup (setup.log)
#      and in `doctor`, and goes on with step 3 and 4. The uv on PATH is not changed. A uv on PATH that says no version is taken, as before.
#   3. a copy that this launcher downloaded before, <env dir>/uv/<version>/uv, if it runs and is the pinned version.
#   4. a new download of that pinned version (only where the setup installs Python and the packages anyway: `sync`, `python`,
#      `topopt`, and the first start of `mcp`; never `status` or `doctor`, which change nothing).
# The download: the release asset of the pinned uv for this platform (macOS arm64, macOS x86_64, Linux x86_64), from
# https://github.com/astral-sh/uv/releases/download/ only, with curl (else wget), HTTPS only. The SHA256 of each asset is pinned
# below and is checked BEFORE the archive is unpacked; a wrong sum deletes the file and unpacks nothing. The archive is unpacked into a
# temporary folder inside <env dir>/uv and moved into place in one step, so a cut download or two setups at once leave no half copy
# that step 3 could take. Nothing is written outside the env dir; the PATH of the user, the shell profile and sudo are not touched.
# Offline, an unknown platform, a missing tool or a failed check: the message with the official install command and the reason.
# Every call of uv in this script uses the full path in $UV. When the copy comes from step 3 or 4, the launcher exports
# TOPOPT_UV=<that path>, so that the repair commands of topopt (doctor, environment, solver choice) name that uv.
#
# Folders. The environment is in <env dir>/venv. The env dir is, in this order:
# $TOPOPT_ENV_DIR, $EVAL_TOPOPT_ENV_DIR (for `claude plugin eval --mocks off`: the run gives the
# plugin an empty temporary CLAUDE_PLUGIN_DATA and passes only EVAL_* variables of the shell),
# $CLAUDE_PLUGIN_DATA (Claude Code sets it for a plugin and keeps it across updates), else
# <repo>/.plugin-env.
# Versions: requirements.lock (exact pins). If the lock cannot install on this platform (a resolution
# or wheel error, not a network error), the pins of pyproject.toml. The stamp says which one was used.
# Python: always a uv-managed Python (the version is the variable LAUNCHER_PYTHON below: 3.13). The venv never links to a Homebrew or system Python, so an
# upgrade of those cannot break it. A venv whose Python does not run, or is not that Python (an older launcher made 3.12), is made again (uv venv --clear).
#
# The first start downloads that Python, 3.13 (if uv has none) and the packages: about 200 MB. That can take
# longer than the 30 s that Claude Code gives an MCP server to connect. So `mcp` starts the setup in a
# separate background process. A client start does not wait for it (the wait is 0 when stdin is not a terminal); a person in a
# terminal waits up to 24 s; $TOPOPT_MCP_WAIT (seconds), when set, wins in both cases. If it is not ready by
# then, `mcp` does not fail and does not exit. It runs a small MCP server in this shell (serve_status, see "The status server"):
# it connects at once with all the tools (a snapshot of the lists), answers a tool call with the setup status, and, when the setup is
# ready, replaces itself with the Python server (exec). No reconnect is needed. Progress is in <env dir>/setup.log.
# A later start only compares a stamp and starts Python.
#
# The background setup is detached from the server (L1a). The command is a subshell, so that the setup
# is no child of the launcher at once (its parent is process 1, or the subreaper of the system, for example systemd
# --user or a container), and, where a tool exists, it has its own session and process group:
#   ( nohup setsid sh SELF sync ... & )                                 when `setsid` exists (Linux)
#   ( nohup perl -MPOSIX -e 'POSIX::setsid() or warn ...; exec @ARGV' sh SELF sync ... & )   else, when `perl` exists (macOS)
#   ( nohup sh SELF sync ... & )                                        else: no child of the launcher, same process group
# Why both parts:
#   - A client that ends the process GROUP of the server (Codex does this each time it closes or drops the server:
#     SIGTERM, then SIGKILL after 2 s; codex-rs/rmcp-client/src/stdio_server_launcher.rs at rust-v0.163.0, lines 284,
#     409-436, 139-140 and 510-518) does not find a setup in its own group. The generic `mcp` SDK client ends the
#     group too, while the server process is alive (mcp/client/stdio.py, `_stop_server_process`).
#   - A client that ends the process TREE of the server (VS Code does; read in its code) walks the children of the
#     launcher. A setup that is no child of the launcher (parent 1 or the subreaper) is not one of them. A background job with `setsid` alone stays a
#     child of the launcher until the launcher exits (measured).
#   - With neither setsid nor perl, the setup is in the process group of the server (a non-interactive sh makes no new
#     group), and a client that ends that group stops it. Run `sync` once in a terminal before the first start then.
# TOPOPT_DETACH=setsid|perl|plain forces one way (a knob for the tests and for diagnosis; a way whose tool is missing is
# ignored). `doctor` says which way this shell will use.
# A setup that is cut by a kill (SIGKILL cannot be trapped) or by a power loss can leave a package without its files
# (and with its .dist-info, so uv calls it installed). So step 2 writes a marker, <env dir>/venv/.topopt-step2, before it
# starts, and removes it after the stamp. If the marker is there at the next setup, the install steps use
# `uv pip install --reinstall`, and `status` and `is_ready` say "not ready". The marker also stays after a clean uv error
# (a re-install from the uv cache is cheap). `--reinstall` makes uv check the index again (it implies `--refresh`): the
# repair needs the network, but it downloads no wheel that the uv cache has. UV_OFFLINE=1 works when the cache has every
# file. After step 3 the setup imports the main packages; the stamp is written only
# if that works.
#
# Stdout is the MCP channel: this script writes only to stderr (except `sync`, `python`, `topopt` and `doctor`; `doctor`
# is never the MCP server). The one exception for `mcp` is the status server, which writes the JSON-RPC lines of the protocol
# to stdout, all of them through ss_emit.
# Nothing here uses bash features. It runs with the macOS /bin/sh and with dash. No `stat`: its flags differ on macOS.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P) || exit 1
SELF=$ROOT/scripts/topopt-env.sh
DEFAULT_ENV_DIR=$ROOT/.plugin-env
ENV_DIR=${TOPOPT_ENV_DIR:-${EVAL_TOPOPT_ENV_DIR:-${CLAUDE_PLUGIN_DATA:-$DEFAULT_ENV_DIR}}}
case "$ENV_DIR" in /*) ;; *) ENV_DIR=$PWD/$ENV_DIR ;; esac   # uv runs in another folder: use a full path
VENV=$ENV_DIR/venv
STAMP=$VENV/.topopt-plugin-stamp
MARKER=$VENV/.topopt-step2
LOCK=$ENV_DIR/.setup-lock
LOG=$ENV_DIR/setup.log
STARTED=$ENV_DIR/.setup-started-by-mcp
STEPLOG=$ENV_DIR/last-step.log
PY=$VENV/bin/python
CHILD=
PINS=lock
REINSTALL=
# The Python that this launcher makes (#422): the one place of the version. It is the Python of every CI leg (CONTRACTS.md 27.2). Step 1,
# the remake on an x86_64 Mac, the remake of an environment of another Python, the stamp and the texts read it. It is a constant, not an
# environment variable. The package accepts 3.13 only (requires-python, #469).
LAUNCHER_PYTHON=3.13
# The uv release that the launcher downloads when the user has no uv (REL-9): the newest stable release of astral-sh/uv on 2026-10-26.
# Each SHA256 is the value in the official <asset>.sha256 file of that release, equal to the sum computed from the downloaded asset and to the
# digest that the GitHub API shows for the asset. They are constants, not environment variables: a different uv is TOPOPT_UV.
# The Linux x86_64 asset is the musl build: it is a static file, so it needs no particular glibc, and uv still picks the gnu build of Python on a glibc host.
PINNED_UV_VERSION=0.13.5
# The oldest uv that the launcher can use (#390). Step 1 of the setup runs `uv venv --python $LAUNCHER_PYTHON --managed-python --clear`, and
# `--managed-python` came with uv 0.6.8 (the uv changelog, 0.6.8: "Add simpler --managed-python and --no-managed-python flags",
# astral-sh/uv#12246). Measured on 2026-10-27: uv 0.6.7 stops there ("unexpected argument '--managed-python' found", exit code 2), and a
# full `sync` with TOPOPT_UV=<uv 0.6.8> is ready; the other flags (`venv --clear`, `pip install --python --reinstall --only-binary
# --no-deps --no-build-isolation -e -r`) work in 0.6.8 too (#397 measured `--no-build-isolation` with 0.6.8: a full `sync`, and the link
# after a plugin update with the network cut). To raise it: when the launcher starts to use a newer flag of uv, set the version of the first uv
# release that has it, and run `TOPOPT_UV=<a uv of that version> sh scripts/topopt-env.sh sync` once. It is a constant, not an
# environment variable (like the pinned version).
MIN_UV_VERSION=0.6.8
PINNED_UV_URL_BASE=https://github.com/astral-sh/uv/releases/download
PINNED_UV_SHA256_MACOS_ARM64=743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73   # uv-aarch64-apple-darwin.tar.gz
PINNED_UV_SHA256_MACOS_X86_64=b0f52b0e3945b4107d89024ea38e625683011fbfc05a0d98f9fa2d6dcee28c66  # uv-x86_64-apple-darwin.tar.gz
PINNED_UV_SHA256_LINUX_X86_64=57ef9d9ccb01fb1bd998df995b0dd221b52f7d7841fbea93f2d4463bf61fb7a9  # uv-x86_64-unknown-linux-musl.tar.gz
# Seconds that the download of the uv archive (about 18 MB) may take. 300 s: a link of 0.5 Mbit/s still finishes, and a slower link
# could not take the 200 MB of packages in a useful time either; the limit only ends a connection that hangs. curl applies it to the
# whole transfer. wget has no limit for the whole transfer: for wget it is the limit of each wait (connect, read). It is not read from the environment.
PINNED_UV_DOWNLOAD_LIMIT_S=300
# The wait that the status text of a tool call during the setup names (#399; mcp_wait). Constants, not environment variables.
# Seconds to wait when the setup gives no progress yet: step 1, or step 2 before the first large package is downloaded. 30 s: a
# setup on a fast link is ready about 60 s after it starts (the host test of 2026-10-27), so a call then is at most one wait late.
SETUP_RECHECK_S=30
# Seconds after the downloads: the install of the packages, step 3 and the import check (measured 6 s on 2026-10-27, Linux x86_64,
# uv 0.12.1). It is also the shortest wait, and the wait in step 3.
SETUP_TAIL_S=10
# The longest wait. A pause of the download (a stall, a laptop that sleeps) makes the rate low and the computed wait long, and each call
# computes it again. 180 s: the upper figure of the usual setup time (README: 1 to 3 minutes).
SETUP_MAX_WAIT_S=180
# What the launcher environment must import after the install (the setup checks it before the stamp; `doctor` checks it too).
IMPORT_LIST='numpy, scipy.ndimage, scipy.sparse.linalg, numba, yaml, topopt_mcp.server'

say() { printf 'topopt: %s\n' "$*" >&2; }

case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*)
    say "Windows is not supported. topopt runs on macOS with Apple silicon and on Linux x86_64."
    exit 1 ;;
esac

# A string for a shell: in single quotes, so a space or a quote in a path does no harm.
quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# The command that runs this script again for the SAME environment folder (for the messages).
# Without the folder, a copy-and-paste run would make a second environment in the plugin folder.
again() {
  if [ "$ENV_DIR" = "$DEFAULT_ENV_DIR" ]; then
    printf 'sh %s %s' "$(quote "$SELF")" "$1"
  else
    printf 'TOPOPT_ENV_DIR=%s sh %s %s' "$(quote "$ENV_DIR")" "$(quote "$SELF")" "$1"
  fi
}

# The GUI apps of a Mac (the Claude Desktop Code tab) can start with a short PATH.
# Add the usual uv folders at the end, so a uv that the user put first on PATH still wins.
for d in "$HOME/.local/bin" "$HOME/.cargo/bin" /opt/homebrew/bin /usr/local/bin; do
  case ":$PATH:" in *":$d:"*) ;; *) [ -d "$d" ] && PATH="$PATH:$d" ;; esac
done
export PATH
export PYTHONNOUSERSITE=1

# The stamp holds the checksum of the lock file that this platform uses: requirements.lock everywhere, except an
# x86_64 Python on a Mac (an Intel Mac or Rosetta), which uses requirements-macos-x86_64.lock (see detect_x86_mac).
# It holds the Python of the launcher too (#422): an environment of an older launcher (Python 3.12, a stamp without the field) does not
# match, so the setup runs once and makes it again with the new Python. Since #422 a change of this text does start an install again,
# on purpose (before, the text was kept the same as before V5, so that no install started again because of it).
want_stamp() {
  detect_x86_mac
  printf 'root=%s lock=%s pyproject=%s python=%s\n' "$ROOT" \
    "$(cksum <"$LOCKFILE" | cut -d' ' -f1,2)" \
    "$(cksum <"$ROOT/pyproject.toml" | cut -d' ' -f1,2)" "$LAUNCHER_PYTHON"
}

# The Python of the environment as major.minor (3.13), or nothing when it does not run or does not answer with a version.
env_python_mm() {
  pm=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || return 0
  case "$pm" in
    [0-9]*.[0-9]*) case "$pm" in *[!0-9.]*) ;; *) printf '%s\n' "$pm" ;; esac ;;
  esac
}

# Ready: the Python runs (a dangling link or a missing library fails here), the stamp is current, and no marker of a
# cut setup is there (the marker is removed after the stamp, so this check only matters for an old stamp).
is_ready() {
  [ -x "$PY" ] && [ -f "$STAMP" ] && [ ! -e "$MARKER" ] && [ "$(sed -n 1p "$STAMP")" = "$(want_stamp)" ] \
    && "$PY" -c '' >/dev/null 2>&1
}

# Is a setup process alive? It writes its pid in the lock folder.
setup_running() {
  owner=$(cat "$LOCK/pid" 2>/dev/null)
  [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null
}

# --- The uv (REL-9; the order and the rules are at the top of this file) ---------------------------------------------------------

# The first line of `uv --version` of the file $1 ("uv 0.13.5 (x86_64-unknown-linux-musl)"). It prints nothing and returns 1 if the
# file does not run or does not print a line that starts with "uv ". It runs in the plugin folder, like every other call of uv.
uv_version_line() {
  uvl_out=$(cd "$ROOT" && "$1" --version 2>/dev/null) || return 1
  uvl_line=$(printf '%s\n' "$uvl_out" | sed -n 1p)
  case "$uvl_line" in "uv "*) printf '%s\n' "$uvl_line" ;; *) return 1 ;; esac
}

# The version in a line of `uv --version`: "uv 0.6.8 (Homebrew 2025-03-18)" gives 0.6.8. It prints nothing if the line has no X.Y or
# X.Y.Z after "uv " (a suffix such as -rc1 is left out).
uv_version_of() {
  printf '%s\n' "$1" | sed -n 's/^uv \([0-9][0-9]*\.[0-9][0-9]*\(\.[0-9][0-9]*\)\{0,1\}\).*/\1/p'
}

# Is the version $1 (X.Y or X.Y.Z) older than MIN_UV_VERSION? A missing part counts as 0.
uv_too_old() {
  awk -v a="$1" -v b="$MIN_UV_VERSION" 'BEGIN {
    n = split(a, x, "."); m = split(b, y, ".")
    for (i = 1; i <= 3; i++) { p = (i <= n) ? x[i] + 0 : 0; q = (i <= m) ? y[i] + 0 : 0
      if (p < q) exit 0
      if (p > q) exit 1 }
    exit 1 }'
}

# The uv on PATH (step 2). Returns 0 and sets $fn_path if there is one that is new enough, or that says no version. A uv that says an
# older version than MIN_UV_VERSION sets $UV_OLD (its path) and $UV_OLD_VER (its version) and returns 1, like no uv on PATH.
path_uv() {
  fn_path=$(command -v uv 2>/dev/null)
  [ -n "$fn_path" ] || return 1
  case "$fn_path" in /*) ;; *) fn_path=$PWD/$fn_path ;; esac   # PATH can hold a relative folder, and `run` changes the folder
  fn_ver=$(uv_version_of "$(uv_version_line "$fn_path")")
  if [ -n "$fn_ver" ] && uv_too_old "$fn_ver"; then
    UV_OLD=$fn_path
    UV_OLD_VER=$fn_ver
    return 1
  fi
  return 0
}

# Find the uv. Sets $UV (a full path) and $UV_KIND: env ($TOPOPT_UV), system (on PATH) or downloaded (a copy of this launcher).
# Returns 1 if there is none. A $TOPOPT_UV that is set but does not work, or is older than MIN_UV_VERSION, is no "none": $UV_BAD then
# holds the reason, and no other uv is tried. A uv on PATH that is older than MIN_UV_VERSION is skipped: $UV_OLD then holds its path and
# $UV_OLD_VER its version, and the search goes on. It writes nothing and fetches nothing. A downloaded copy is exported as TOPOPT_UV, so the
# Python side names the same uv.
find_uv() {
  UV=
  UV_KIND=
  UV_BAD=
  UV_OLD=
  UV_OLD_VER=
  if [ -n "${TOPOPT_UV:-}" ]; then
    case "$TOPOPT_UV" in
      /*) ;;
      *) UV_BAD="it is not a full path"; return 1 ;;
    esac
    fn_line=$(uv_version_line "$TOPOPT_UV") || { UV_BAD="that file is not a uv that runs"; return 1; }
    fn_ver=$(uv_version_of "$fn_line")
    if [ -n "$fn_ver" ] && uv_too_old "$fn_ver"; then
      UV_BAD="it is uv $fn_ver, and topopt needs uv $MIN_UV_VERSION or newer"
      return 1
    fi
    UV=$TOPOPT_UV
    UV_KIND=env
    # A launcher that exported the downloaded copy (below) to a child, or to this process earlier: it is still that copy. Then the uv
    # on PATH is looked at only to say why it is not used (an old one sets $UV_OLD for setup.log and doctor).
    if [ "$UV" = "$ENV_DIR/uv/$PINNED_UV_VERSION/uv" ]; then
      UV_KIND=downloaded
      path_uv >/dev/null 2>&1
    fi
    return 0
  fi
  if path_uv; then
    UV=$fn_path
    UV_KIND=system
    return 0
  fi
  uv_copy=$ENV_DIR/uv/$PINNED_UV_VERSION/uv
  case "$(uv_version_line "$uv_copy" 2>/dev/null)" in
    "uv $PINNED_UV_VERSION "*|"uv $PINNED_UV_VERSION")
      UV=$uv_copy
      UV_KIND=downloaded
      TOPOPT_UV=$UV
      export TOPOPT_UV
      return 0 ;;
  esac
  return 1
}

# The SHA256 (64 hex digits) of the file $1. `sha256sum` is on Linux, `shasum` on macOS. Prints nothing and returns 1 if there is no tool.
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" 2>/dev/null | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1
  else return 1
  fi
}

# What a download of the pinned uv would be on this machine. It reads facts only: nothing is fetched or written. Sets $DL_ASSET (the
# target name in the file name), $DL_SHA (its pinned SHA256) and $DL_TOOL (curl or wget). Returns 1 and sets $DL_NO (the reason, one
# line) if no download is possible: another platform, no curl and no wget, or no tool for the SHA256.
plan_uv_download() {
  DL_ASSET=
  DL_SHA=
  DL_TOOL=
  DL_NO=
  pd_os=$(uname -s 2>/dev/null)
  pd_cpu=$(uname -m 2>/dev/null)
  case "$pd_os $pd_cpu" in
    "Darwin arm64"|"Darwin aarch64")
      DL_ASSET=aarch64-apple-darwin; DL_SHA=$PINNED_UV_SHA256_MACOS_ARM64 ;;
    "Darwin x86_64")
      # An Apple silicon Mac in a Terminal that runs under Rosetta says x86_64 too. Like the official uv installer, take the arm64 uv
      # there (it runs under Rosetta, and it makes a native arm64 Python, which is the fast path: see x86_mac_note).
      if [ "$(mac_sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ]; then
        DL_ASSET=aarch64-apple-darwin; DL_SHA=$PINNED_UV_SHA256_MACOS_ARM64
      else
        DL_ASSET=x86_64-apple-darwin; DL_SHA=$PINNED_UV_SHA256_MACOS_X86_64
      fi ;;
    "Linux x86_64"|"Linux amd64")
      DL_ASSET=x86_64-unknown-linux-musl; DL_SHA=$PINNED_UV_SHA256_LINUX_X86_64 ;;
    *)
      DL_NO="there is no pinned uv download for this platform ($pd_os $pd_cpu): the launcher has one for macOS (arm64, x86_64) and Linux x86_64"
      return 1 ;;
  esac
  if command -v curl >/dev/null 2>&1; then DL_TOOL=curl
  elif command -v wget >/dev/null 2>&1; then DL_TOOL=wget
  else DL_NO="neither curl nor wget is installed"; return 1
  fi
  if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    DL_NO="neither sha256sum nor shasum is installed, so a download could not be checked"
    return 1
  fi
  return 0
}

# The message when there is no uv and none can be had. $1 is the one-line reason why the automatic download did not work.
uv_missing_message() {
  if [ -n "$UV_OLD" ]; then   # a new uv from the install script can come after the old one on PATH: the old one must be updated
    say "The uv on PATH is too old: $UV_OLD is uv $UV_OLD_VER, and topopt needs uv $MIN_UV_VERSION or newer. Update it $(uv_update_how)."
  else
    say "uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  fi
  say "The automatic download of uv did not work: $1"
  say "Then start this again. uv gets Python $LAUNCHER_PYTHON itself."
}

# The message when TOPOPT_UV is set and does not work.
uv_bad_message() {
  say "TOPOPT_UV is set to $(quote "$TOPOPT_UV"), but $UV_BAD. Set it to the full path of a uv that runs (uv $MIN_UV_VERSION or newer), or remove the variable."
  say "The launcher tries no other uv while TOPOPT_UV is set."
}

# Why the launcher has to get its own uv, for the messages: "uv is not installed", or the uv on PATH that find_uv skipped. $1 is the first
# word for the skipped uv ("The" at the start of a sentence; default "the").
uv_absent_why() {
  if [ -n "$UV_OLD" ]; then printf '%s' "${1:-the} uv on PATH is too old ($UV_OLD is uv $UV_OLD_VER; topopt needs uv $MIN_UV_VERSION or newer)"
  else printf '%s' "uv is not installed"
  fi
}

# How to update a uv that is too old. `uv self update` works only for a uv from the install script; for a uv from pip or brew it stops
# and names those (measured with uv 0.6.7 and 0.12.1), so the package managers come first.
uv_update_how() {
  printf '%s' "with the tool that installed it (for example brew upgrade uv or pip install --upgrade uv; uv self update for a uv from the install script)"
}

# Is there a uv, or can the setup get one? It changes nothing, so `status` and the first step of `mcp` use it. Without a uv and without a
# way to download one it says why and returns 1. With $1 = plan it also says that the setup will download the uv (for `status`).
need_uv() {
  find_uv && return 0
  if [ -n "$UV_BAD" ]; then uv_bad_message; return 1; fi
  if plan_uv_download; then
    [ "${1:-}" = plan ] && say "$(uv_absent_why). The setup downloads uv $PINNED_UV_VERSION ($DL_ASSET) from github.com/astral-sh/uv, checks its SHA256 and keeps it in $(quote "$ENV_DIR/uv/$PINNED_UV_VERSION"). That needs the network."
    return 0
  fi
  uv_missing_message "$DL_NO"
  return 1
}

# The folder of a download in progress (inside the env dir). on_signal removes it.
UV_TMP=
UV_FAIL=

# A download step failed: remove the temporary folder, keep the reason in $UV_FAIL, return 1.
uv_fail() {
  UV_FAIL=$1
  [ -n "$UV_TMP" ] && rm -rf "$UV_TMP"
  UV_TMP=
  return 1
}

# Download the pinned uv into <env dir>/uv/<version>/uv. The caller holds the setup lock, so no second download runs in this env dir.
# Order: download into a temporary folder; check the SHA256; only then unpack; run the new uv once; move it into place in one step (a
# rename inside the same folder). On a failure it leaves nothing but the reason in $UV_FAIL. It sets $UV when it works.
fetch_uv() {
  plan_uv_download || { UV_FAIL=$DL_NO; return 1; }
  fu_base=$ENV_DIR/uv
  fu_dest=$fu_base/$PINNED_UV_VERSION
  fu_url=$PINNED_UV_URL_BASE/$PINNED_UV_VERSION/uv-$DL_ASSET.tar.gz
  mkdir -p "$fu_base" || { UV_FAIL="cannot make the folder $fu_base"; return 1; }
  rm -rf "$fu_base"/.tmp-* 2>/dev/null      # what a download that was cut left behind (the lock is ours)
  UV_TMP=$fu_base/.tmp-$$
  mkdir "$UV_TMP" || { UV_TMP=; UV_FAIL="cannot make a folder in $fu_base"; return 1; }
  say "$(uv_absent_why): downloading uv $PINNED_UV_VERSION ($DL_ASSET) from $fu_url"
  if [ "$DL_TOOL" = curl ]; then
    run_logged "$UV_TMP/download.err" curl --proto '=https' --proto-redir '=https' --tlsv1.2 --fail --silent --show-error --location \
      --max-time "$PINNED_UV_DOWNLOAD_LIMIT_S" --output "$UV_TMP/uv.tar.gz" "$fu_url"
  else
    # wget cannot limit the protocol of a redirect; the SHA256 check below does not depend on the channel.
    run_logged "$UV_TMP/download.err" wget --https-only --quiet --tries=1 --timeout="$PINNED_UV_DOWNLOAD_LIMIT_S" \
      --output-document="$UV_TMP/uv.tar.gz" "$fu_url"
  fi
  fu_rc=$?
  if [ "$fu_rc" -ne 0 ] || [ ! -s "$UV_TMP/uv.tar.gz" ]; then
    uv_fail "the download failed ($DL_TOOL exit code $fu_rc; no network?): $(sed -n 1p "$UV_TMP/download.err" 2>/dev/null | cut -c1-200)"
    return 1
  fi
  fu_got=$(sha256_of "$UV_TMP/uv.tar.gz")
  if [ "$fu_got" != "$DL_SHA" ]; then
    rm -f "$UV_TMP/uv.tar.gz"
    uv_fail "the SHA256 of the download is ${fu_got:-not known} and the pinned value is $DL_SHA. The file was deleted and nothing was unpacked."
    return 1
  fi
  mkdir "$UV_TMP/x" && tar -xzf "$UV_TMP/uv.tar.gz" -C "$UV_TMP/x" "uv-$DL_ASSET/uv" 2>"$UV_TMP/tar.err" \
    || { uv_fail "the archive does not unpack: $(sed -n 1p "$UV_TMP/tar.err" 2>/dev/null | cut -c1-200)"; return 1; }
  fu_new=$UV_TMP/x/uv-$DL_ASSET/uv
  chmod +x "$fu_new" 2>/dev/null
  case "$(uv_version_line "$fu_new")" in
    "uv $PINNED_UV_VERSION "*|"uv $PINNED_UV_VERSION") ;;
    *) uv_fail "the unpacked uv does not run here or is not version $PINNED_UV_VERSION (is the folder $fu_base mounted without the right to run files?)"; return 1 ;;
  esac
  mkdir "$UV_TMP/stage" && mv "$fu_new" "$UV_TMP/stage/uv" || { uv_fail "cannot move the unpacked uv"; return 1; }
  rm -rf "$fu_dest"      # a copy that step 3 of find_uv did not take (it does not run, or is not the pinned version) goes away
  mv "$UV_TMP/stage" "$fu_dest" || { uv_fail "cannot move the uv to $fu_dest"; return 1; }
  rm -rf "$UV_TMP"
  UV_TMP=
  find_uv || { UV_FAIL="the uv in $fu_dest does not run"; return 1; }
  say "uv $PINNED_UV_VERSION is ready: $UV"
}

# For the setup (the lock is held): the uv that is there, else a download. Says why and returns 1 if there is none.
get_uv() {
  if find_uv; then
    [ -n "$UV_OLD" ] && say "$(uv_absent_why): the setup uses $UV instead. The uv on PATH is not changed."
    return 0
  fi
  if [ -n "$UV_BAD" ]; then uv_bad_message; return 1; fi
  fetch_uv && return 0
  uv_missing_message "$UV_FAIL"
  return 1
}

# Run a command in the background and wait for it, so a signal can stop it at once.
# It runs in the plugin folder, not in the folder of the user: a uv.toml or a [tool.uv] table
# of the user's project does not change the install.
run() {
  ( cd "$ROOT" && exec "$@" ) >&2 &
  CHILD=$!
  wait "$CHILD"
  status=$?
  CHILD=
  return "$status"
}

# Like run. The error output also goes to the file $1 (and then to stderr), so it can be read.
run_logged() {
  out=$1
  shift
  ( cd "$ROOT" && exec "$@" ) >&2 2>"$out" &
  CHILD=$!
  wait "$CHILD"
  status=$?
  CHILD=
  cat "$out" >&2
  return "$status"
}

# Import the main packages in a child Python. It runs in the folder "/", not in the plugin folder: the source tree of
# topopt must not take the place of an installed package (a broken link of step 3 shows here). The Python error goes to
# the file $STEPLOG; stdout is not used. Returns 0 if every import works.
check_imports() {
  ( cd / && exec "$PY" -c "import $IMPORT_LIST" ) >/dev/null 2>"$STEPLOG" &
  CHILD=$!
  wait "$CHILD"
  status=$?
  CHILD=
  return "$status"
}

release_lock() { rm -rf "$LOCK" 2>/dev/null; }

on_signal() {
  [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null
  [ -n "$UV_TMP" ] && rm -rf "$UV_TMP"
  release_lock
  exit 143
}

# Is the lock folder there with an owner that is gone? 0 = yes: the pid in it is not running, or there is no pid and the
# folder is more than a minute old (a process that was killed between `mkdir` and writing the pid). Sets $owner (the pid text).
# A pid that is not running is read ONE MORE TIME, and the lock is stale only if it is the same pid (F3, review m1): between
# the first read and `kill -0` the owner can end, remove its own lock and exit, and a new owner can take the lock. The pid
# is then another one, and that lock is not stale. A gone owner cannot remove or change its lock, and only the process that
# holds the mutex of break_stale_lock removes a stale lock, so a lock that still has the same dead pid stays until that process
# removes it.
lock_is_stale() {
  owner=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$owner" ]; then
    kill -0 "$owner" 2>/dev/null && return 1
    [ "$(cat "$LOCK/pid" 2>/dev/null)" = "$owner" ]
    return
  fi
  [ -n "$(find "$LOCK" -maxdepth 0 -mmin +1 2>/dev/null)" ]
}

# Remove the lock of a gone owner. Returns 0 when this process did the check (the lock may be gone now: try `mkdir` at
# once) and 1 when another waiter is doing it (go on waiting).
# Why a mutex, and not `rm -rf "$LOCK"` where the waiter saw the dead pid (the race of the L1a review, m-4): waiters A and
# B both read the dead pid. A removes the lock and makes a new one, then B removes A's new lock, and two setups run. Only one
# waiter can make the folder "$LOCK.break". It checks the owner AGAIN inside it (lock_is_stale), so a waiter that is late finds
# the new owner (alive, or a fresh folder with no pid yet) and removes nothing. Inside the mutex, the lock can change only
# in these ways: its owner ends and removes it (lock_is_stale then reads another pid or no folder, and removes nothing), or
# a new owner takes the name after that. A lock with the same dead pid at the second read is not changed by anybody before the
# `rm`.
# A signal removes the mutex (the trap). The trap is cleared BEFORE the mutex is removed, so it can never remove the mutex of
# the next waiter. Only a kill that cannot be trapped (SIGKILL, a power loss) inside these few commands, or a signal in the
# gap between `trap -` and `rm`, leaves the mutex; `doctor` shows it and gives the command to remove it. There is no time limit.
break_stale_lock() {
  mkdir "$LOCK.break" 2>/dev/null || return 1
  trap 'rm -rf "$LOCK.break"; exit 143' INT TERM HUP
  lock_is_stale && rm -rf "$LOCK"
  trap - INT TERM HUP
  rm -rf "$LOCK.break"
  return 0
}

acquire_lock() {
  mkdir -p "$ENV_DIR" || return 1
  waited=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    if lock_is_stale; then
      break_stale_lock && continue   # the process that held it is gone (checked again inside the mutex): try to take it at once
      # Another process is removing the stale lock. If it was killed in the middle, the mutex stays and this wait has no end.
      [ "$waited" -eq 0 ] && say "the process that made the lock (pid ${owner:-unknown}) is gone, and another process is removing the lock (the mutex $(quote "$LOCK.break")). Waiting for it. If this does not end, that process was killed: remove the mutex with rm -rf $(quote "$LOCK.break")"
    else
      [ "$waited" -eq 0 ] && say "another setup is running (pid ${owner:-unknown}). Waiting for it."
    fi
    sleep 2
    waited=$((waited + 2))
    is_ready && return 2
  done
  echo $$ >"$LOCK/pid"
  date +%s >"$LOCK/started" 2>/dev/null   # the status text of `mcp` says how long the setup has run
}

# An x86_64 Python on a Mac (an Intel Mac, or Rosetta on Apple silicon). Best effort (V5). Facts of 2026-10-20
# (PyPI metadata and `uv pip compile --python-platform x86_64-apple-darwin`): numba 0.67.0, llvmlite 0.49.0
# and cryptography 50.0.2 of requirements.lock have no macOS x86_64 wheel. So this platform uses
# requirements-macos-x86_64.lock (older pins for those packages and for numpy). Nothing was tested on a Mac.
# X86MAC is 1 on that platform, else empty. The Python of the environment decides (it is the Python that
# runs topopt). The machine type of the shell decides only if that Python does not run.
X86MAC=
LOCKFILE=$ROOT/requirements.lock
LOCKNAME=lock
X86LOCK=$ROOT/requirements-macos-x86_64.lock
detect_x86_mac() {
  X86MAC=
  LOCKFILE=$ROOT/requirements.lock
  LOCKNAME=lock
  [ "$(uname -s 2>/dev/null)" = Darwin ] || return 0
  machine=$("$PY" -c 'import platform; print(platform.machine())' 2>/dev/null)
  case "$machine" in x86_64|arm64) ;; *) machine=$(uname -m 2>/dev/null) ;; esac
  [ "$machine" = x86_64 ] || return 0
  X86MAC=1
  if [ -f "$X86LOCK" ]; then LOCKFILE=$X86LOCK; LOCKNAME=lock-macos-x86_64; fi
  return 0
}

# sysctl is in /usr/sbin, which a short PATH (a GUI app) can lack. TOPOPT_SYSCTL replaces that full path (a knob for the
# tests: a fake sysctl on PATH cannot replace the full path of the real one).
mac_sysctl() {
  if [ -x "${TOPOPT_SYSCTL:-/usr/sbin/sysctl}" ]; then "${TOPOPT_SYSCTL:-/usr/sbin/sysctl}" "$@"; else sysctl "$@"; fi
}

# "rosetta" if this is an Apple silicon Mac (the translation flag, or the arm64 flag of the CPU), else "intel".
x86_mac_kind() {
  if [ "$(mac_sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ] || [ "$(mac_sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ]; then
    echo rosetta
  else
    echo intel
  fi
}

# On an x86_64 Mac, do not build numba, llvmlite or cryptography from source (they need LLVM and Rust, and a
# build takes long): the install then stops at once with the name of the package. TOPOPT_ALLOW_SOURCE_BUILD=1
# lifts it. Prints nothing on every other platform. The flag is --only-binary: `uv pip install` has no
# --no-build-package (that one is a flag of `uv sync`, `uv lock` and `uv run`). The test of the flags reads the real
# `uv pip install --help`.
no_build_flags() {
  [ -n "$X86MAC" ] && [ -z "$TOPOPT_ALLOW_SOURCE_BUILD" ] &&
    printf '%s' '--only-binary numba --only-binary llvmlite --only-binary cryptography'
  return 0
}

x86_mac_note() {
  [ -n "$X86MAC" ] || return 0
  kind=$(x86_mac_kind)
  say "this Mac runs Python as x86_64 ($kind). This is best effort: topopt was tested on Linux x86_64, not on a Mac."
  say "It uses $LOCKNAME, which is older than requirements.lock: $(grep -E '^(numpy|numba|llvmlite|cryptography)==' "$LOCKFILE" 2>/dev/null | tr '\n' ' ')"
  say "Solver auto gives amg, and the time estimate is not calibrated. To build from source anyway: TOPOPT_ALLOW_SOURCE_BUILD=1."
  if [ "$kind" = rosetta ]; then
    say "This is an Apple silicon Mac: a native arm64 Python is the fast path (Apple Accelerate). To use it:"
    say "  1. Open the Terminal without Rosetta (Get Info on the app, clear 'Open using Rosetta')."
    say "  2. Check uv: file $(quote "${UV:-uv}")  It must say arm64. If it does not, install uv again from that Terminal."
    say "  3. Remove this environment: rm -rf $(quote "$VENV")"
    say "  4. Run: $(again sync)"
  else
    say "This is an Intel Mac. It has no native arm64 Python. Facts and options: docs/reference/install.md."
  fi
}

# The build backend of topopt (`[build-system]` of pyproject.toml) at the version of the lock file (#397). Step 3 links the code with
# the setuptools of the environment (--no-build-isolation), so it asks no index; step 2 installs it from the lock. The fallback of step 2
# (the pins of pyproject.toml) installs this pin too. It prints `setuptools==<version>` only: a lock line can have another case of the
# name, a comment or a `\` and a `--hash` line after it (`uv pip install -r` takes these, an argument does not). A lock without the line
# gives the requirement of `[build-system]`.
build_backend_pin() {
  bb_pin=$(grep -i -m 1 -E '^setuptools==' "$LOCKFILE" 2>/dev/null | sed -E 's/^[^=]*==([^[:space:];#\\]+).*/setuptools==\1/')
  echo "${bb_pin:-setuptools>=77}"
}

# Text in the uv output that points to a network problem, not to a problem of this platform.
NET_PATTERN='dns error|error sending request|connection (refused|reset|closed)|timed out|timeout|failed to (fetch|download)|network|could not resolve|temporary failure|tls|ssl|certificate|proxy|offline'

install_all() {
  start=$(date +%s)
  PINS=lock
  made=
  if ! "$PY" -c '' >/dev/null 2>&1; then
    # No venv yet, or a venv whose Python is gone (for example a killed setup). --clear makes it new.
    say "step 1 of 3: make the Python $LAUNCHER_PYTHON environment (uv gets Python $LAUNCHER_PYTHON if it has none)"
    run "$UV" venv --python "$LAUNCHER_PYTHON" --managed-python --clear "$VENV" || return 1
    made=1
  else
    have_py=$(env_python_mm)
    if [ -n "$have_py" ] && [ "$have_py" != "$LAUNCHER_PYTHON" ]; then
      # An environment of an older launcher (Python 3.12): make it again with the Python of this launcher (#422). The binary wheels are
      # per Python version, so the packages download again.
      # The line starts with the step: the status server reads the last "step N of 3" line of setup.log (mcp_step), and without it a tool call
      # during the remake would get the step 3 line of the old setup.
      say "step 1 of 3: make the Python $LAUNCHER_PYTHON environment again (the environment has Python $have_py; the packages download again, this needs the network)"
      run "$UV" venv --python "$LAUNCHER_PYTHON" --managed-python --clear "$VENV" || return 1
      made=1
    fi
  fi
  detect_x86_mac
  if [ -n "$X86MAC" ] && [ -z "$made" ] && [ "$(uname -m 2>/dev/null)" = arm64 ]; then
    # An old x86_64 environment, and this shell is native: make it again with the uv of this shell.
    say "step 1 of 3: make the Python $LAUNCHER_PYTHON environment again (the environment has an x86_64 Python, and this shell runs natively (arm64))"
    run "$UV" venv --python "$LAUNCHER_PYTHON" --managed-python --clear "$VENV" || return 1
    made=1
    detect_x86_mac
  fi
  [ -n "$X86MAC" ] && PINS=$LOCKNAME
  x86_mac_note
  # The marker says: step 2 or 3 started and did not end with a stamp. A kill in step 2 can leave a package with its
  # .dist-info and without its files, and uv then skips it. So a marker from an earlier call means: install again
  # (--reinstall). A venv made in this call has no marker (uv venv --clear deletes the folder).
  REINSTALL=
  if [ -f "$MARKER" ] && [ -z "$made" ]; then
    REINSTALL=--reinstall
    say "an earlier setup did not end (the marker $MARKER is there): installing the packages again (--reinstall)"
  fi
  : >"$MARKER" || { say "cannot write $MARKER"; return 1; }
  : >"$STEPLOG"                                 # the status text reads the progress of step 2 there: none from an earlier step
  date +%s >"$LOCK/step2" 2>/dev/null           # and the time since step 2 began
  say "step 2 of 3: install the packages of $(basename "$LOCKFILE")"
  # shellcheck disable=SC2046  (no_build_flags prints words or nothing; REINSTALL is one word or nothing)
  if ! run_logged "$STEPLOG" "$UV" pip install --python "$PY" $REINSTALL $(no_build_flags) -r "$LOCKFILE"; then
    if grep -q -i -E "$NET_PATTERN" "$STEPLOG" 2>/dev/null; then
      say "the packages did not download: a network problem. Check the connection. Run it again to resume."
      return 1
    fi
    say "the lock file did not install on this platform: $(grep -i -m 1 error "$STEPLOG" 2>/dev/null | cut -c1-160)"
    say "Using the pins of pyproject.toml."
    PINS=pyproject.toml
    run "$UV" pip install --python "$PY" $REINSTALL $(no_build_flags) "$ROOT[mcp]" "$(build_backend_pin)" || return 1
  fi
  say "step 3 of 3: link the topopt code in $ROOT"
  # With the setuptools of step 2, not a new one from the index in an isolated build: a plugin update (a new root) relinks offline (#397).
  run "$UV" pip install --python "$PY" $REINSTALL --no-deps --no-build-isolation -e "$ROOT" || return 1
  if ! check_imports; then
    say "the packages are installed, but an import fails: $(tail -n 1 "$STEPLOG" | cut -c1-200)"
    cat "$STEPLOG" >&2
    say "The marker stays, so the next setup installs the packages again. The Python error is also in $STEPLOG."
    return 1
  fi
  { want_stamp; echo "pins=$PINS"; } >"$STAMP" || return 1
  rm -f "$MARKER"
  say "ready after $(($(date +%s) - start)) s (pins=$PINS): $PY"
}

# Make the environment, in this process. Wait if another process makes it.
ensure() {
  is_ready && return 0
  need_uv || return 1   # nothing is downloaded yet: this only checks that a uv is there or can be had
  acquire_lock
  got=$?
  [ "$got" -eq 2 ] && return 0   # another process made the environment while we waited
  [ "$got" -eq 0 ] || { say "cannot make the folder $ENV_DIR"; return 1; }
  trap on_signal INT TERM HUP
  get_uv && install_all   # the download of uv, if the user has none, is the first step of the setup (it holds the lock)
  result=$?
  trap - INT TERM HUP
  release_lock
  [ "$result" -eq 0 ] || say "setup failed. Run it again to resume: $(again sync)"
  [ "$result" -eq 0 ] || [ -z "$X86MAC" ] || say "On this x86_64 Mac the install failed (best effort): see the notes above."
  return "$result"
}

# How the background setup is detached (see the top of this file): setsid, perl or plain. TOPOPT_DETACH forces one,
# if its tool exists (a knob for the tests and for diagnosis).
detach_method() {
  case "${TOPOPT_DETACH:-}" in
    setsid) command -v setsid >/dev/null 2>&1 && { echo setsid; return 0; } ;;
    perl) command -v perl >/dev/null 2>&1 && { echo perl; return 0; } ;;
    plain) echo plain; return 0 ;;
  esac
  if command -v setsid >/dev/null 2>&1; then echo setsid
  elif command -v perl >/dev/null 2>&1; then echo perl
  else echo plain
  fi
}

# Start `sync` in the background, so that it is not a child of this launcher and, where possible, not in its
# process group or session. The subshell ( ... & ) ends at once, and the setup then is no child of this launcher.
# The lock folder gets the pid of the setup shell itself (acquire_lock), so setup_running stays right.
start_setup_detached() {
  case "$1" in
    setsid) ( nohup setsid sh "$SELF" sync >>"$LOG" 2>&1 </dev/null & ) ;;
    perl) ( nohup perl -MPOSIX -e 'POSIX::setsid() or warn "topopt: setsid failed: $!\n"; exec @ARGV' sh "$SELF" sync >>"$LOG" 2>&1 </dev/null & ) ;;
    *) ( nohup sh "$SELF" sync >>"$LOG" 2>&1 </dev/null & ) ;;
  esac
}

# --- The status server: what `mcp` does while the environment is not ready (#370) -----------------------------------------------
# Claude Code gives an MCP server 30 s to connect, and marks a server that does not connect, or that answers with an error, as failed;
# it then skips that server for 15 minutes. So a start that is not ready does not fail. It runs a small MCP server in this shell
# (serve_status) that answers from a snapshot of the lists of the real server, and it changes to the Python server when the setup is
# ready:
#   - the era comes from the first request: `server/discover`, or a request whose params._meta has the key
#     io.modelcontextprotocol/protocolVersion, means the modern era (protocol 2026-07-28: no initialize, a subscription stream);
#     `initialize` means the legacy era (the handshake);
#   - the lists come from the snapshot in topopt_mcp/launcher/ (made by `python -P -m topopt_mcp.snapshot` from the answers of the real
#     server, one file for each answer, one line each). The client sees all the tools from its first answer: many clients (Codex CLI,
#     Cursor, Claude Desktop, the Code tab of the Claude desktop app) do not ask for the lists again after a list_changed notification.
#     The one thing that depends on the request is the version in the legacy `initialize` answer: the file holds the text
#     @PROTOCOL_VERSION@, and the shell puts the version that the SDK would pick (below);
#   - a call of a tool during the setup is answered with a tool result with isError true, and the text gives the step, the time and one
#     next step (tell the user, wait). `prompts/get` and `resources/read` have no such form: a JSON-RPC error with the same text;
#   - the hand-off: the first request with an id that comes after the setup is ready is not answered here. The shell replaces itself with
#     the Python server (exec). The open stdin goes with it: `read` takes one line, and the rest stays in the pipe. The lines that the shell
#     already read are written to a file, and the variable TOPOPT_MCP_HANDOFF names it (topopt_mcp/handoff.py reads the file and deletes it).
#     A file, not a variable: a line can be longer than the limit of one environment string (128 KB on Linux), and a file keeps the client
#     text out of the environment of the child processes. Modern: the pending line. Legacy: the initialize line, notifications/initialized
#     and the pending line; TOPOPT_MCP_HANDOFF_DROP_ID is the id of that initialize, so that Python drops its answer (the client has the
#     answer of the shell). TOPOPT_MCP_HANDOFF_ERA and TOPOPT_MCP_HANDOFF_LISTEN_ID (the id of the listen request, if there was one) let
#     Python compare the live lists with the snapshot and send list_changed if they differ (the safety net).
# The shell is the only writer to stdout until the exec. A terminal on stdin means that a person started `mcp` by hand: nothing is read,
# and the stderr lines are all there is. The server needs no feature of bash. The JSON is read by awk (JSON_REQUEST_AWK).

# The text for a JSON string: backslash and double quote get a backslash, and a control character is removed.
json_escape() { printf '%s' "$1" | LC_ALL=C tr '\t\r\n' '   ' | LC_ALL=C tr -d '\000-\037' | LC_ALL=C sed 's/\\/\\\\/g; s/"/\\"/g'; }

# True when this launcher started the setup less than 3 s ago (the same 3 s as the early-stop check in ensure_for_server): the setup
# shell may not have written its pid in the lock folder yet. A client start does not wait for the setup, so a tool call can come at once.
started_just_now() {
  sj_then=$(cat "$STARTED" 2>/dev/null)
  sj_now=$(date +%s 2>/dev/null)
  case "$sj_then$sj_now" in ''|*[!0-9]*) return 1 ;; esac
  [ "$sj_now" -ge "$sj_then" ] && [ $((sj_now - sj_then)) -le 3 ]
}

# Why the start is not ready, from the state now: starting | running | uvvar | uv | folder | cut | stopped.
mcp_cause() {
  if setup_running; then echo running
  elif started_just_now; then echo starting
  elif ! find_uv && [ -n "$UV_BAD" ]; then echo uvvar
  elif [ -z "$UV" ]; then echo uv
  elif [ ! -d "$ENV_DIR" ]; then echo folder
  elif [ -e "$MARKER" ]; then echo cut
  else echo stopped
  fi
}

# The last "step N of 3: ..." line of the setup log, without the prefix. Empty if there is none.
mcp_step() { grep 'step [0-9] of 3' "$LOG" 2>/dev/null | tail -n 1 | sed 's/^topopt: //'; }

# Seconds that the running setup has run (the lock folder holds the start time). Empty if not known.
mcp_elapsed() {
  me_then=$(cat "$LOCK/started" 2>/dev/null)
  me_now=$(date +%s 2>/dev/null)
  case "$me_then$me_now" in ''|*[!0-9]*) return 0 ;; esac
  [ "$me_now" -ge "$me_then" ] && echo $((me_now - me_then))
}

# The wait for the status text, from the step ($1, as mcp_step prints it). Prints "<seconds>", or "<seconds> <done> <of> <MB>" when
# step 2 gives its progress: the large packages that are downloaded, of how many, and their MB in all (uv gives MiB; MB as the README).
# uv writes `Downloading <name> (<size>)` for each large package when it starts it, and ` Downloaded <name>` when it has it (uv 0.6.8,
# 0.12.1 and 0.13.5, also when the output is a file; the small packages get no line). It downloads them all at once. In the first-setup
# runs of 2026-10-27 the packages came in the order of their size, at a steady rate: the bytes are shared about evenly. So with k of the
# n packages done, and s the smallest one that is not done, the bytes so far are about the sum of the done ones plus s for each of the
# n - k others (s comes next). That over the seconds since step 2 began is the rate; the rest of the bytes at that rate, plus
# SETUP_TAIL_S, is the wait, at most SETUP_MAX_WAIT_S. Without progress (no line, no package done yet, no step-2 time) the wait is
# SETUP_RECHECK_S. So it is when uv does not start them at once (UV_CONCURRENT_DOWNLOADS, or concurrent-downloads in a uv.toml, under
# the number of large packages): a `Downloading` line then comes after a ` Downloaded` line, and the count would be wrong.
mcp_wait() {
  case "$1" in
    "step 2 of 3"*) ;;
    "step 3 of 3"*) echo "$SETUP_TAIL_S"; return 0 ;;
    *) echo "$SETUP_RECHECK_S"; return 0 ;;
  esac
  mw_then=$(cat "$LOCK/step2" 2>/dev/null)
  mw_now=$(date +%s 2>/dev/null)
  case "$mw_then" in ''|*[!0-9]*) echo "$SETUP_RECHECK_S"; return 0 ;; esac   # a new setup has not written it yet
  case "$mw_now" in ''|*[!0-9]*) echo "$SETUP_RECHECK_S"; return 0 ;; esac
  mw_out=$(awk -v dt=$((mw_now - mw_then)) -v recheck="$SETUP_RECHECK_S" -v tail="$SETUP_TAIL_S" -v cap="$SETUP_MAX_WAIT_S" '
    $1 == "Downloading" && $NF ~ /^\([0-9.]+[KMG]iB\)$/ {
      if (ended) late = 1                     # this download started after another one ended: not all at once
      v = substr($NF, 2, length($NF) - 5) + 0
      u = substr($NF, length($NF) - 3, 3)
      size[$2] = (u == "KiB") ? v / 1024 : (u == "GiB") ? v * 1024 : v
    }
    $1 == "Downloaded" { got[$2] = 1; ended = 1 }
    END {
      n = 0; k = 0; tot = 0; sd = 0; s = -1
      for (p in size) {
        n++; tot += size[p]
        if (p in got) { k++; sd += size[p] } else if (s < 0 || size[p] < s) s = size[p]
      }
      d = (k == n) ? tot : sd + s * (n - k)
      if (k == 0 || late || dt < 1 || d <= 0) { print recheck; exit }
      w = (tot - d) * dt / d + tail
      if (w > cap) w = cap
      printf "%d %d %d %.0f\n", (w > int(w)) ? int(w) + 1 : w, k, n, tot * 1.048576
    }' "$STEPLOG" 2>/dev/null)
  echo "${mw_out:-$SETUP_RECHECK_S}"   # one line in every awk, also when the file is missing
}

# A wait in seconds ($1) as words: under 2 minutes in steps of 5 s, rounded up ("about 95 s"), else the nearest whole minutes.
mcp_wait_words() {
  mww_s=$(( ($1 + 4) / 5 * 5 ))
  if [ "$mww_s" -lt 120 ]; then echo "about $mww_s s"; else echo "about $(( ($1 + 30) / 60 )) minutes"; fi
}

# The text while the setup runs, from the step part of the text ($1), the seconds that it has run ($2, can be empty) and the numbers of
# mcp_wait ($3 to $6). The whole setup takes 1 to 3 minutes (the text of the README); after 3 minutes the text says that it takes
# longer than usual. It always names a wait and says not to call again at once.
mcp_running_text() {
  mrt_run=
  mrt_usual=" The whole setup takes 1 to 3 minutes (about 200 MB to download) and goes on in the background."
  if [ -n "$2" ]; then
    mrt_run=" It has run $2 s."
    [ "$2" -gt 180 ] && mrt_usual=" It takes longer than usual (a slow link); it goes on in the background."
  fi
  mrt_done=
  [ -n "$6" ] && mrt_done=" Downloaded: $4 of the $5 large packages ($6 MB in all)."
  printf '%s' "topopt is still setting up its Python environment.$1$mrt_run$mrt_done$mrt_usual Next step: tell the user that topopt is still setting up, $(mcp_wait_words "$3") more. Wait that long before you call a topopt tool again. Do not call it again at once."
}

# The text of a tool call during the setup, from the cause ($1), the step ($2), the seconds ($3) and the wait ($4, as mcp_wait prints
# it). It ends with ONE next step.
mcp_cause_text() {
  mct_step=
  [ -n "$2" ] && mct_step=" Now: $2."
  case "$1" in
    starting) printf '%s' "topopt: the setup is starting. Next step: tell the user that the setup is starting, and call a topopt tool again in about 10 s. Do not call it again at once." ;;
    # shellcheck disable=SC2086  (the wait is one to four numbers: one argument each)
    running) mcp_running_text "$mct_step" "$3" ${4:-$SETUP_RECHECK_S} ;;
    uv)
      find_uv
      if [ -n "$UV_OLD" ]; then   # a new uv from the install script can come after the old one on PATH: the old one must be updated
        printf '%s' "topopt cannot set up its Python environment: $(uv_absent_why), and the automatic download of a newer uv failed (see setup.log in the plugin data folder).$mct_step Next step: tell the user to update that uv $(uv_update_how). Then they run: $(again sync)  After that the topopt tools work in this session."
      else
        printf '%s' "topopt cannot set up its Python environment: uv is missing and its automatic download failed (see setup.log in the plugin data folder).$mct_step Next step: show the user this command and ask if they agree to run it: curl -LsSf https://astral.sh/uv/install.sh | sh  Then they run: $(again sync)  After that the topopt tools work in this session."
      fi ;;
    uvvar)
      find_uv
      mct_why="names a uv that does not run"
      case "$UV_BAD" in "it is uv "*) mct_why="is set, but $UV_BAD" ;; esac   # a uv that runs and is too old
      printf '%s' "topopt cannot set up its Python environment: the variable TOPOPT_UV $mct_why. Next step: tell the user to set it to the full path of a uv that runs (uv $MIN_UV_VERSION or newer), or to remove it, and to start the server again (/mcp)." ;;
    folder) printf '%s' "topopt cannot set up its Python environment: the folder for it cannot be made. Next step: tell the user to check that the plugin data folder can be written to, and to start the server again (/mcp)." ;;
    cut) printf '%s' "The Python setup of topopt stopped before it was ready. An earlier setup was cut or failed, and the next sync installs the packages again (it needs the network).$mct_step Next step: show the user this command: $(again sync)  After that the topopt tools work in this session." ;;
    *) printf '%s' "The Python setup of topopt stopped before it was ready.$mct_step The setup.log in the plugin data folder says why. Next step: show the user this command: $(again sync)  After that the topopt tools work in this session." ;;
  esac
}

# The status text for the state now.
ss_status() {
  sst_step=$(mcp_step)
  mcp_cause_text "$(mcp_cause)" "$sst_step" "$(mcp_elapsed)" "$(mcp_wait "$sst_step")"
}

# Read one line of JSON-RPC (stdin of awk). Print nothing, or one line of five fields, separated by the character \037 (it cannot
# be in a JSON text): the method (without quotes), the id as it is in the line (a number, or a string with its quotes; "-" for a
# notification, which has no id), 1 if params._meta has the key io.modelcontextprotocol/protocolVersion (else 0), the
# protocolVersion in params (without quotes; "-" if none), and the name in params (a tool name; "-" if none). Only the TOP level
# keys id and method, and the keys name and protocolVersion directly in the top level params, are read: the client puts the same
# keys in other places too. A response (no method) prints nothing. No apostrophe in this text: it is in single quotes.
JSON_REQUEST_AWK='
function skipws(   c) {
  while (i <= n) {
    c = substr(s, i, 1)
    if (c != " " && c != "\t" && c != "\r") break
    i++
  }
}
function str(   start, c) {
  start = i
  i++
  while (i <= n) {
    c = substr(s, i, 1)
    if (c == "\\") { i += 2; continue }
    if (c == "\"") { i++; return substr(s, start, i - start) }
    i++
  }
  return ""
}
function unq(v) { return (substr(v, 1, 1) == "\"") ? substr(v, 2, length(v) - 2) : "" }
{
  s = $0; n = length(s); i = 1; depth = 0; method = ""; id = ""; meta = 0; ver = ""; name = ""; top = ""; k2 = ""; bad = 0
  skipws()
  if (substr(s, i, 1) != "{") exit
  while (i <= n) {
    c = substr(s, i, 1)
    if (c == "\"") {
      tok = str()
      if (tok == "") exit
      skipws()
      if (substr(s, i, 1) != ":") continue
      if (depth == 1) top = tok
      else if (depth == 2) k2 = tok
      else if (depth == 3 && top == "\"params\"" && k2 == "\"_meta\"" && tok == "\"io.modelcontextprotocol/protocolVersion\"") meta = 1
      want = 0
      if (depth == 1 && (tok == "\"id\"" || tok == "\"method\"")) want = 1
      if (depth == 2 && top == "\"params\"" && (tok == "\"name\"" || tok == "\"protocolVersion\"")) want = 1
      if (want) {
        i++
        skipws()
        if (substr(s, i, 1) == "\"") {
          v = str()
        } else {
          st = i
          while (i <= n && index("+-.0123456789eE", substr(s, i, 1)) > 0) i++
          v = substr(s, st, i - st)
        }
        if (depth == 1) {
          if (tok == "\"id\"" && id == "") { if (v == "") bad = 1; else id = v }
          if (tok == "\"method\"" && method == "") method = unq(v)
        } else {
          if (tok == "\"name\"" && name == "") name = unq(v)
          if (tok == "\"protocolVersion\"" && ver == "") ver = unq(v)
        }
      }
      continue
    }
    if (c == "{" || c == "[") depth++
    else if (c == "}" || c == "]") { depth--; if (depth < 2) k2 = "" }
    i++
  }
  if (method == "" || bad) exit
  if (id == "") id = "-"
  else if (substr(id, 1, 1) != "\"" && id !~ /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/) exit
  if (ver == "") ver = "-"
  if (name == "") name = "-"
  printf "%s\037%s\037%d\037%s\037%s\n", method, id, meta, ver, name
}'

# The versions that the Python server (mcp 2.3.0) negotiates in the handshake: the version of the client if it is in this list,
# else the last one. They are HANDSHAKE_PROTOCOL_VERSIONS and LATEST_HANDSHAKE_VERSION of mcp_types (a test compares them).
HANDSHAKE_VERSIONS="2024-11-05 2025-03-26 2025-06-18 2025-11-25"
HANDSHAKE_LATEST=2025-11-25
# Never use \001 or \177 as an IFS separator: bash 3.2 (the macOS /bin/sh) uses them inside and does not split on them.
SS_SEP=$(printf '\037')
SS_META='"_meta":{"io.modelcontextprotocol/serverInfo":{"name":"topopt","version":""}}'
SS_SNAPSHOT=$ROOT/topopt_mcp/launcher

ss_emit() { printf '%s\n' "$1"; }

# The answer with a result: $1 is the id, $2 the members of the result (a tool result). The modern results carry resultType and _meta;
# the legacy results carry neither.
ss_result() {
  if [ "$ss_era" = modern ]; then
    ss_emit "{\"jsonrpc\":\"2.0\",\"id\":$1,\"result\":{${2:+$2,}\"resultType\":\"complete\",$SS_META}}"
  else
    ss_emit "{\"jsonrpc\":\"2.0\",\"id\":$1,\"result\":{$2}}"
  fi
}

ss_error() { ss_emit "{\"jsonrpc\":\"2.0\",\"id\":$1,\"error\":{\"code\":$2,\"message\":\"$3\"}}"; }

# The answer from a file of the snapshot: $1 is the id, $2 the file name (without .json), $3 the version for the placeholder (or empty).
ss_snapshot() {
  ssn_file=$SS_SNAPSHOT/$2.json
  if [ -n "$3" ]; then ssn_body=$(sed "s/@PROTOCOL_VERSION@/$3/" "$ssn_file" 2>/dev/null); else ssn_body=$(cat "$ssn_file" 2>/dev/null); fi
  if [ -z "$ssn_body" ]; then
    say "the snapshot file $ssn_file is missing or empty"
    ss_error "$1" -32603 "The launcher snapshot of the lists is missing."
    return 0
  fi
  ss_emit "{\"jsonrpc\":\"2.0\",\"id\":$1,\"result\":$ssn_body}"
}

# Replace this shell by the Python server. The lines that were read are in the file TOPOPT_MCP_HANDOFF (see the top of the section).
ss_handoff() {
  ss_hf=$ENV_DIR/.mcp-handoff.$$
  ss_umask=$(umask)
  umask 077                                  # the file holds the client text (tool arguments): the owner only
  if {
    if [ "$ss_era" = legacy ] && [ -n "$ss_init_line" ]; then
      printf '%s\n' "$ss_init_line"
      printf '%s\n' "${ss_inited_line:-$SS_INITIALIZED}"
    fi
    printf '%s\n' "$ss_line"
  } >"$ss_hf" 2>/dev/null; then
    umask "$ss_umask"
    TOPOPT_MCP_HANDOFF=$ss_hf
    TOPOPT_MCP_HANDOFF_ERA=$ss_era
    export TOPOPT_MCP_HANDOFF TOPOPT_MCP_HANDOFF_ERA
    [ -n "$ss_listen_id" ] && { TOPOPT_MCP_HANDOFF_LISTEN_ID=$ss_listen_id; export TOPOPT_MCP_HANDOFF_LISTEN_ID; }
    if [ "$ss_era" = legacy ] && [ -n "$ss_init_line" ]; then
      TOPOPT_MCP_HANDOFF_DROP_ID=$ss_init_id
      export TOPOPT_MCP_HANDOFF_DROP_ID
    fi
    exec "$PY" -P -m topopt_mcp.server "$@"
  fi
  umask "$ss_umask"
  rm -f "$ss_hf" 2>/dev/null
  say "the environment is ready, but the hand-off file cannot be written: $ss_hf"
  return 1
}

SS_INITIALIZED='{"jsonrpc":"2.0","method":"notifications/initialized"}'

# Answer the requests of the client until it closes stdin. $@ are the arguments for the Python server at the hand-off.
serve_status() {
  ss_era=; ss_listen_id=; ss_init_line=; ss_init_id=; ss_inited_line=
  while IFS= read -r ss_line || [ -n "$ss_line" ]; do
    ss_fields=$(printf '%s\n' "$ss_line" | LC_ALL=C awk "$JSON_REQUEST_AWK")
    [ -n "$ss_fields" ] || continue
    IFS=$SS_SEP read -r ss_method ss_id ss_meta ss_ver ss_name <<SS_FIELDS_END
$ss_fields
SS_FIELDS_END
    if [ "$ss_id" = - ]; then                     # a notification: no answer
      [ "$ss_method" = notifications/initialized ] && [ "$ss_era" = legacy ] && ss_inited_line=$ss_line
      continue
    fi
    if [ -z "$ss_era" ]; then
      if [ "$ss_method" = server/discover ] || [ "$ss_meta" = 1 ]; then ss_era=modern; else ss_era=legacy; fi
    fi
    if is_ready; then
      ss_handoff "$@" || { ss_error "$ss_id" -32603 "The Python environment is ready, but the server cannot start."; continue; }
    fi
    case "$ss_method" in
      initialize)
        ss_era=legacy
        ss_init_line=$ss_line
        ss_init_id=$ss_id
        ss_version=$HANDSHAKE_LATEST
        for ss_v in $HANDSHAKE_VERSIONS; do [ "$ss_v" = "$ss_ver" ] && ss_version=$ss_v; done
        ss_snapshot "$ss_id" legacy.initialize "$ss_version" ;;
      server/discover) ss_snapshot "$ss_id" modern.discover ;;
      subscriptions/listen)
        ss_listen_id=$ss_id
        ss_emit "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/subscriptions/acknowledged\",\"params\":{\"_meta\":{\"io.modelcontextprotocol/subscriptionId\":$ss_id},\"notifications\":{\"toolsListChanged\":true,\"promptsListChanged\":true,\"resourcesListChanged\":true}}}" ;;
      tools/list) ss_snapshot "$ss_id" "$ss_era.tools" ;;
      prompts/list) ss_snapshot "$ss_id" "$ss_era.prompts" ;;
      resources/list) ss_snapshot "$ss_id" "$ss_era.resources" ;;
      resources/templates/list) ss_snapshot "$ss_id" "$ss_era.templates" ;;
      ping) if [ "$ss_era" = legacy ]; then ss_result "$ss_id" ""; else ss_error "$ss_id" -32601 "Method not found"; fi ;;   # as the Python server
      tools/call) ss_result "$ss_id" "\"content\":[{\"type\":\"text\",\"text\":\"$(json_escape "$(ss_status)")\"}],\"isError\":true" ;;
      prompts/get|resources/read) ss_error "$ss_id" -32603 "$(json_escape "$(ss_status)")" ;;
      *) ss_error "$ss_id" -32601 "Method not found" ;;
    esac
  done
  return 0
}

# For the MCP server: make the environment in a background process and wait a short time for it.
# Whether the setup goes on after `mcp` exits depends on how it was detached (see the top of this file).
ensure_for_server() {
  is_ready && return 0
  need_uv || return 1
  mkdir -p "$ENV_DIR" || return 1
  if ! setup_running; then
    method=$(detach_method)
    start_setup_detached "$method"
    say "first start: the setup runs in the background (start method: $method). Log: $LOG"
    date +%s >"$STARTED" 2>/dev/null   # mcp_cause: for 3 s after this start, a setup with no lock yet is "running", not "stopped"
  fi
  i=0
  while [ "$i" -lt "$MCP_WAIT" ]; do
    is_ready && return 0
    sleep 1
    i=$((i + 1))
    if [ "$i" -gt 3 ] && ! setup_running && ! is_ready; then
      say "the setup stopped before it was ready. Last lines of $LOG:"
      tail -n 5 "$LOG" >&2 2>/dev/null
      return 1
    fi
  done
  is_ready && return 0
  say "the setup is not ready yet (the first start downloads about 200 MB: 1 to 3 minutes)."
  say "The server answers from a snapshot of its lists, and a tool call gives the setup status, until the setup is done. The setup goes on in the background, and the tools work in the same session."
  say "To see if it is done, run: $(again status). Progress is in $LOG"
  return 1
}

# The facts of the setup, on stdout, for a person (or an agent) who asks why the tools do not show. It only reads: no
# install, no lock, no write, no network. It works with a broken or a missing venv. Exit 0: ready and every import works.
# Exit 1: an action is needed, and the last lines name the next command.
doc_row() { printf '  %-14s %s\n' "$1" "$2"; }

# The time since a file or folder last changed, as "725 s ago (12 min)", or "not known". Facts only: no limit is applied to it.
# No `stat` (its flags differ on macOS): `date -r FILE` is in GNU date and in the date of macOS and BSD.
age_of() {
  a_then=$(date -r "$1" +%s 2>/dev/null)
  a_now=$(date +%s 2>/dev/null)
  case "$a_then$a_now" in ''|*[!0-9]*) printf 'not known'; return 0 ;; esac
  a_secs=$((a_now - a_then))
  [ "$a_secs" -ge 0 ] || a_secs=0
  printf '%s s ago (%s min)' "$a_secs" "$((a_secs / 60))"
}

doctor() {
  d_need=        # the next command, if an action is needed
  d_why=
  d_run=0        # the venv Python runs
  d_stamp=0      # the stamp is there and matches
  d_imp=         # ok | failed | skipped
  d_pyrc=        # the exit code of `topopt doctor`, if it is not 0
  printf 'topopt launcher doctor\n'
  doc_row "script" "$SELF"
  doc_row "system" "$(uname -s -m 2>/dev/null)"
  # The uv, as the setup would pick it (find_uv: TOPOPT_UV, PATH, a copy of this launcher). It downloads nothing.
  if find_uv; then
    case "$UV_KIND" in
      env) d_uvkind="TOPOPT_UV" ;;
      downloaded) d_uvkind="downloaded by topopt" ;;
      *) d_uvkind="system" ;;
    esac
    doc_row "uv" "$d_uvkind: $UV ($(uv_version_line "$UV"))"
    [ -n "$UV_OLD" ] && doc_row "uv on PATH" "skipped: $UV_OLD is uv $UV_OLD_VER, older than uv $MIN_UV_VERSION, which topopt needs"
  elif [ -n "$UV_BAD" ]; then
    doc_row "uv" "TOPOPT_UV is set to $(quote "$TOPOPT_UV"), but $UV_BAD (the setup stops with this message and tries no other uv)"
  else
    d_old=
    [ -n "$UV_OLD" ] && d_old="$(uv_absent_why); "
    if plan_uv_download; then
      doc_row "uv" "not found: ${d_old}the setup downloads uv $PINNED_UV_VERSION ($DL_ASSET) into $ENV_DIR/uv/$PINNED_UV_VERSION"
    else
      doc_row "uv" "not found: ${d_old}the setup cannot download it here ($DL_NO)"
    fi
  fi
  d_rule="default: <repo>/.plugin-env"
  [ -n "${CLAUDE_PLUGIN_DATA:-}" ] && d_rule=CLAUDE_PLUGIN_DATA
  [ -n "${EVAL_TOPOPT_ENV_DIR:-}" ] && d_rule=EVAL_TOPOPT_ENV_DIR
  [ -n "${TOPOPT_ENV_DIR:-}" ] && d_rule=TOPOPT_ENV_DIR
  doc_row "env dir" "$ENV_DIR (chosen by: $d_rule)"
  # The Python of the environment.
  if [ -x "$PY" ]; then
    if d_out=$("$PY" -c 'import platform; print(platform.python_version(), platform.machine())' 2>&1); then
      d_run=1
      doc_row "venv Python" "$PY: runs ($d_out)$(d_pm=$(env_python_mm); [ -n "$d_pm" ] && [ "$d_pm" != "$LAUNCHER_PYTHON" ] && printf '; this launcher makes Python %s: the next sync makes the environment again' "$LAUNCHER_PYTHON")"
    else
      doc_row "venv Python" "$PY: does not run: $(printf '%s\n' "$d_out" | tail -n 1 | cut -c1-160)"
    fi
  elif [ -L "$PY" ]; then
    doc_row "venv Python" "$PY: a link to a file that is not there"
  else
    doc_row "venv Python" "$PY: not there (no environment yet)"
  fi
  # The stamp: written after step 3 and the import check.
  if [ -f "$STAMP" ]; then
    if [ "$(sed -n 1p "$STAMP")" = "$(want_stamp)" ]; then
      d_stamp=1
      doc_row "stamp" "present, matches this checkout ($(sed -n 2p "$STAMP"))"
    else
      case "$(sed -n 1p "$STAMP")" in
        *" python=$LAUNCHER_PYTHON") doc_row "stamp" "present, but it does not match: the folder of the code, requirements.lock or pyproject.toml changed since the last setup" ;;
        *" python="*) doc_row "stamp" "present, but it does not match: it was made for Python $(sed -n 1p "$STAMP" | sed 's/.* python=//'), and this launcher makes Python $LAUNCHER_PYTHON" ;;
        *) doc_row "stamp" "present, but it does not match: it was made by an older launcher that made Python 3.12 (it holds no Python), and this launcher makes Python $LAUNCHER_PYTHON; or the folder of the code, requirements.lock or pyproject.toml changed" ;;
      esac
    fi
  else
    doc_row "stamp" "not there (the setup did not end, or it never ran)"
  fi
  # The marker: step 2 started and no stamp followed.
  if [ -e "$MARKER" ]; then
    doc_row "setup marker" "present ($MARKER): an earlier setup was cut or failed. The next sync installs the packages again."
  else
    doc_row "setup marker" "none"
  fi
  d_running=0
  if setup_running; then
    d_running=1
    d_pid=$(cat "$LOCK/pid" 2>/dev/null)
    doc_row "setup running" "yes (pid $d_pid)"
  else
    doc_row "setup running" "no"
  fi
  # The lock folder as facts. The launcher asks only `kill -0` of the pid in it, so a pid that another program has taken
  # (after a crash or a reboot) looks like a running setup. The row shows what the pid runs (`ps`, else /proc/<pid>/cmdline), when
  # the lock was taken and when setup.log last changed, so that a person (or an agent) sees a lock that does nothing. Only a
  # BACKGROUND setup writes setup.log (a sync in a terminal writes to its terminal), so an old setup.log says nothing about a
  # sync that runs in a terminal; the row says so.
  if [ -d "$LOCK" ]; then
    d_lpid=$(cat "$LOCK/pid" 2>/dev/null)
    if [ -z "$d_lpid" ]; then
      d_ltxt="the folder has no pid in it"
    elif kill -0 "$d_lpid" 2>/dev/null; then
      d_ltxt="pid $d_lpid is alive"
      d_cmd=
      command -v ps >/dev/null 2>&1 && d_cmd=$(ps -p "$d_lpid" -o command= 2>/dev/null | head -n 1 | cut -c1-120)
      if [ -z "$d_cmd" ] && [ -r "/proc/$d_lpid/cmdline" ]; then   # no ps (or no answer of it): Linux has the command line in /proc
        d_cmd=$(tr '\0' ' ' <"/proc/$d_lpid/cmdline" 2>/dev/null | sed 's/ *$//' | cut -c1-120)
      fi
      if [ -n "$d_cmd" ]; then
        d_ltxt="$d_ltxt and runs: $d_cmd"
      else
        d_ltxt="$d_ltxt; the command of the pid is not known (ps gave none, and /proc/$d_lpid/cmdline is not there or empty)"
      fi
    elif [ -d "$LOCK.break" ]; then
      d_ltxt="pid $d_lpid is NOT running (a stale lock, but the mutex $(basename "$LOCK.break") is there: no sync removes the lock while it stays, see the row lock mutex)"
    else
      d_ltxt="pid $d_lpid is NOT running (a stale lock: the next sync removes it)"
    fi
    d_ltxt="$d_ltxt; taken $(age_of "$LOCK")"
    [ -e "$LOG" ] && d_ltxt="$d_ltxt; setup.log last changed $(age_of "$LOG") (only a background setup writes it: a sync in a terminal writes to its terminal)"
    doc_row "setup lock" "$d_ltxt"
  else
    doc_row "setup lock" "none"
  fi
  if [ -d "$LOCK.break" ]; then
    doc_row "lock mutex" "$LOCK.break is there, made $(age_of "$LOCK.break"). A waiter makes it for a moment while it removes a stale lock, and while it is there no other waiter removes a stale lock. If it stays, that waiter was killed: remove it with rm -rf $(quote "$LOCK.break")"
  fi
  # The imports: only when the Python runs and no setup is changing the packages.
  if [ "$d_run" -ne 1 ]; then
    d_imp=skipped
    doc_row "imports" "not checked (the Python does not run)"
  elif [ "$d_running" -eq 1 ]; then
    d_imp=skipped
    doc_row "imports" "not checked (a setup is running)"
  elif d_out=$(cd / && "$PY" -c "import $IMPORT_LIST" 2>&1); then
    d_imp=ok
    doc_row "imports" "ok ($IMPORT_LIST)"
  else
    d_imp=failed
    doc_row "imports" "FAILED: $(printf '%s\n' "$d_out" | tail -n 1 | cut -c1-200)"
  fi
  detect_x86_mac
  doc_row "lock file" "$(basename "$LOCKFILE") (stamp name: $LOCKNAME)"
  d_method=$(detach_method)
  case "$d_method" in
    setsid) d_how="setsid ($(command -v setsid)): the background setup gets a session of its own and is no child of the launcher" ;;
    perl) d_how="perl ($(command -v perl)), POSIX::setsid: the background setup gets a session of its own and is no child of the launcher" ;;
    *) d_how="plain (no tool): the background setup is no child of the launcher and stays in the process group of the server" ;;
  esac
  [ -n "${TOPOPT_DETACH:-}" ] && d_how="$d_how [TOPOPT_DETACH=$TOPOPT_DETACH]"
  doc_row "start method" "$d_how"
  if [ -s "$LOG" ]; then
    doc_row "setup log" "$LOG, last lines:"
    tail -n 5 "$LOG" | sed 's/^/      /'
  else
    doc_row "setup log" "$LOG: empty or not there"
  fi
  # The Python side: the checkout facts and the offers. The command exists in a version of topopt that has L1b.
  printf '\n'
  if [ "$d_run" -ne 1 ]; then
    printf 'topopt doctor (Python): not run, the venv Python does not run.\n'
  elif [ "$d_running" -eq 1 ]; then
    printf 'topopt doctor (Python): not run, a setup is running.\n'
  else
    # Say what runs and show its lines as they come (a silent step looks hung). "up to 90 s each" is IMPORT_TIMEOUT_S of
    # topopt/doctor.py and "up to 60 s" is PROBE_TIMEOUT_S of topopt/gpu_mg/probe.py (a test pins both numbers).
    printf 'topopt doctor (Python): reading the machine (with a GPU: the self-test of gpu-mg, up to 60 s, until it passes once on a device and driver) and checking the imports of the packages (each in its own Python, up to 90 s each); usually a few seconds.\n'
    # The lines go to the terminal and into a file (for the check of the old code); the exit code goes into a file too (POSIX sh has
    # no pipefail). The folder is made by mktemp, and removed at the end and by a signal. With no folder, the output comes at the end.
    d_tmp=$(mktemp -d "${TMPDIR:-/tmp}/topopt-doctor.XXXXXX" 2>/dev/null) || d_tmp=
    if [ -n "$d_tmp" ]; then
      # A signal removes the folder and then ends this script by the same signal, as before (a calling shell sees 130 for Ctrl+C).
      # The shell runs the trap when the pipeline below ends: Ctrl+C reaches the whole foreground group, so it ends at once; a signal
      # sent to this pid alone waits for the end of the Python step.
      trap 'rm -rf "$d_tmp"; trap - INT; kill -INT $$' INT
      trap 'rm -rf "$d_tmp"; trap - TERM; kill -TERM $$' TERM
      trap 'rm -rf "$d_tmp"; trap - HUP; kill -HUP $$' HUP
      { (cd / && PYTHONUNBUFFERED=1 "$PY" -m topopt doctor 2>&1); echo $? >"$d_tmp/rc"; } | tee "$d_tmp/out"
      d_rc=$(cat "$d_tmp/rc" 2>/dev/null)
      case "$d_rc" in ''|*[!0-9]*) d_rc=1 ;; esac
      d_out=$(cat "$d_tmp/out" 2>/dev/null)
      trap - INT TERM HUP
      rm -rf "$d_tmp"
      d_shown=1
    else
      # In the folder "/", as above and as the import check: `python -m` puts the current folder first on sys.path, and a topopt folder
      # of the user (an old clone) would take the place of the code of the environment.
      d_out=$(cd / && "$PY" -m topopt doctor 2>&1)
      d_rc=$?
      d_shown=0
    fi
    if [ "$d_rc" -ne 0 ] && printf '%s\n' "$d_out" | grep -q "invalid choice: '\{0,1\}doctor"; then
      printf 'topopt doctor (Python): this version of topopt has no such command (older code). The lines above are all that this version can say. Update this copy of topopt to get it.\n'
    else
      [ "$d_shown" -eq 1 ] || printf '%s\n' "$d_out"
      if [ "$d_rc" -ne 0 ]; then
        d_pyrc=$d_rc
        printf '(topopt doctor ended with exit code %s. This does not change the exit code of this command.)\n' "$d_rc"
      fi
    fi
  fi
  # The next command.
  printf '\n'
  if [ "$d_running" -eq 1 ]; then
    d_need=$(again sync)
    d_why="a setup is running (pid $d_pid). sync waits for it and returns when the environment is ready (the log is in the row setup log). The launcher only checks that the pid is alive: the row setup lock shows what the pid runs (when the system can tell) and how old the lock and setup.log are. Only a background setup writes setup.log, so an old setup.log is no sign that a sync in a terminal is stuck. If the pid is not a setup, remove the lock with rm -rf $(quote "$LOCK") and run sync."
  elif [ "$d_imp" = failed ] && [ ! -e "$MARKER" ]; then
    # A package does not import and nothing says why. uv skips a package that has its .dist-info, so make the marker:
    # then sync installs the packages again.
    d_need="touch $(quote "$MARKER") && $(again sync)"
    d_why="a package does not import (a setup that an older launcher cut?). The marker file makes sync install the packages again (from the uv cache; uv still checks the index, so it needs the network, or UV_OFFLINE=1 when the cache has every file)."
  elif [ "$d_run" -ne 1 ] || [ "$d_stamp" -ne 1 ] || [ -e "$MARKER" ] || [ "$d_imp" = failed ]; then
    d_need=$(again sync)
    d_why="the environment is not ready."
    [ -e "$MARKER" ] && d_why="an earlier setup was cut or failed. sync installs the packages again (from the uv cache; uv still checks the index, so it needs the network, or UV_OFFLINE=1 when the cache has every file)."
  fi
  # A stale lock and a mutex that a killed waiter left: no sync removes the lock, so a sync waits with no end. The mutex goes first.
  if [ -n "$d_need" ] && [ "$d_running" -eq 0 ] && [ -d "$LOCK.break" ] && lock_is_stale; then
    d_need="rm -rf $(quote "$LOCK.break") && $d_need"
    d_why="a waiter that was killed left the mutex $LOCK.break, and the lock $LOCK is stale (pid ${owner:-unknown} is not running). While the mutex is there no sync removes the lock, so a sync would wait with no end. Remove the mutex first, then sync. $d_why"
  fi
  if [ -n "$d_need" ] && [ -n "$UV_BAD" ]; then
    d_why="$d_why TOPOPT_UV is set but $UV_BAD: set it to the full path of a uv that runs, or remove it, first."
  elif [ -n "$d_need" ] && [ -z "$UV" ]; then
    if plan_uv_download; then
      d_why="$d_why $(uv_absent_why The): the setup downloads uv $PINNED_UV_VERSION first (a pinned release, its SHA256 is checked)."
    else
      if [ -n "$UV_OLD" ]; then
        d_why="$d_why $(uv_absent_why The), and the setup cannot download a newer uv ($DL_NO): first update it $(uv_update_how)."
      else
        d_why="$d_why uv is not installed and the setup cannot download it ($DL_NO): first run   curl -LsSf https://astral.sh/uv/install.sh | sh"
      fi
    fi
  fi
  if [ -n "$d_need" ]; then
    printf 'Result: action needed: %s\n' "$d_why"
    printf 'Next command:\n  %s\n' "$d_need"
    return 1
  fi
  if [ -n "$d_pyrc" ]; then
    printf 'Result: the launcher environment is ready, but the Python report above lists a problem (exit code %s). The exit code of this command stays 0.\n' "$d_pyrc"
  else
    printf 'Result: ready. Nothing to run.\n'
  fi
  return 0
}

# --- plugin-sync: the setup of the Claude Code plugin before its first start (#357) ------------------------------------------------
# Claude Code starts the server of an installed plugin with CLAUDE_PLUGIN_DATA=<plugins>/data/<id> (read in Claude Code 2.1.293; its
# docs name the folder ~/.claude/plugins/data/{id}/):
#   <plugins> is $CLAUDE_CODE_PLUGIN_CACHE_DIR if it is set (a leading ~/ is the home folder), else
#   ${CLAUDE_CONFIG_DIR:-$HOME/.claude}/plugins (another mode of Claude Code, with the folder cowork_plugins, is not covered);
#   <id> is <plugin>@<marketplace>, with each character that is not a letter, a digit, "_" or "-" replaced by "-".
# plugin-sync names that folder and runs `sync` with it, under the folder rule of the server (TOPOPT_ENV_DIR and EVAL_TOPOPT_ENV_DIR
# still come first). The first start of the server then finds a ready environment, so Claude Code records no failed start (it
# skips a plugin server that failed for 15 minutes). The names are the "name" of .claude-plugin/plugin.json and of
# .claude-plugin/marketplace.json in this copy: Claude Code names the marketplace by that file. It stops (exit 1) when
# <plugins>/installed_plugins.json does not name the plugin: then the plugin is not installed for that Claude Code, or its
# CLAUDE_CONFIG_DIR is another one. Run it from the copy that Claude Code starts (`claude plugin list --json`: readFromFolder for a
# marketplace in a local folder, else installPath), because the stamp holds the root of the copy.

# The first "name" value of a JSON file of this copy (the top-level name: it comes first in both files).
first_json_name() {
  sed -n 's/^[[:space:]]*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$1" 2>/dev/null | sed -n 1p
}

# Sets PS_DATA (the plugin data folder) and PS_ID (the plugin id), or says why not and returns 1.
plugin_data_folder() {
  ps_plugin=$(first_json_name "$ROOT/.claude-plugin/plugin.json")
  ps_market=$(first_json_name "$ROOT/.claude-plugin/marketplace.json")
  if [ -z "$ps_plugin" ] || [ -z "$ps_market" ]; then
    say "no plugin name or marketplace name in $ROOT/.claude-plugin: plugin-sync is for a copy that Claude Code installed as a plugin."
    return 1
  fi
  if [ -n "${CLAUDE_CODE_PLUGIN_CACHE_DIR:-}" ]; then
    ps_plugins=$CLAUDE_CODE_PLUGIN_CACHE_DIR
    case "$ps_plugins" in '~') ps_plugins=$HOME ;; '~/'*) ps_plugins=$HOME/${ps_plugins#'~/'} ;; esac
  elif [ -n "${CLAUDE_CONFIG_DIR:-}" ]; then
    ps_plugins=$CLAUDE_CONFIG_DIR/plugins
  else
    ps_plugins=$HOME/.claude/plugins
  fi
  PS_ID=$ps_plugin@$ps_market
  if ! grep -F -q "\"$PS_ID\"" "$ps_plugins/installed_plugins.json" 2>/dev/null; then
    say "Claude Code has no plugin $PS_ID here: $ps_plugins/installed_plugins.json does not name it."
    say "Install the plugin first. If Claude Code runs with another CLAUDE_CONFIG_DIR or CLAUDE_CODE_PLUGIN_CACHE_DIR, set the same one for this command."
    return 1
  fi
  PS_DATA=$ps_plugins/data/$(printf '%s' "$PS_ID" | LC_ALL=C sed 's/[^A-Za-z0-9_-]/-/g')
}

# Look for the uv now: a copy that this launcher downloaded earlier is exported as TOPOPT_UV to every child (the Python server, the
# doctor of topopt), even when the environment is ready and no uv call follows. A TOPOPT_UV that does not work is no error here: a ready
# environment needs no uv. need_uv says it when a uv is needed.
find_uv

cmd=${1:-}
[ $# -gt 0 ] && shift
case "$cmd" in
  status)
    if is_ready; then say "ready ($(sed -n 2p "$STAMP")): $PY"; exit 0; fi
    if setup_running; then
      say "not ready: the setup is running. Last line of $LOG: $(tail -n 1 "$LOG" 2>/dev/null)"
    else
      need_uv plan || exit 1   # say it before the user agrees to a download of 200 MB (a missing uv is downloaded by the setup)
      if [ -e "$MARKER" ]; then
        say "not ready: an earlier setup was cut or failed ($MARKER). The next sync installs the packages again. Run: $(again sync)"
      else
        st_py=$(env_python_mm)
        if [ -n "$st_py" ] && [ "$st_py" != "$LAUNCHER_PYTHON" ]; then
          say "not ready: the environment has Python $st_py, and this launcher makes Python $LAUNCHER_PYTHON. The next sync makes it again (the first start after an update: it downloads Python $LAUNCHER_PYTHON and the packages, and needs the network). Run: $(again sync)"
        else
          say "not ready: $PY. Run: $(again sync)"
        fi
      fi
    fi
    exit 1 ;;
  doctor)
    doctor
    exit $? ;;
  sync)
    ensure || exit 1
    echo "$PY" ;;
  mcp)
    # -P: the client starts the server in the project folder of the user. `python -m` would put that folder first on sys.path, and a
    # user file there (timeit.py, json.py) would take the place of a standard module. The packages are installed in the venv.
    # The wait for the setup: TOPOPT_MCP_WAIT if it is set. Else 0 for a client (stdin is not a terminal): the status server answers at once
    # from the snapshot, so there is nothing to wait for. Else 24 s for a person in a terminal (the stderr lines are all there is).
    if [ -n "${TOPOPT_MCP_WAIT:-}" ]; then MCP_WAIT=$TOPOPT_MCP_WAIT; elif [ -t 0 ]; then MCP_WAIT=24; else MCP_WAIT=0; fi
    if ensure_for_server; then exec "$PY" -P -m topopt_mcp.server "$@"; fi
    [ -t 0 ] && exit 1   # a person in a terminal: the lines above are all there is
    serve_status "$@"
    exit 1 ;;   # the client closed stdin and the Python server never ran: not ready, as before
  plugin-sync)
    plugin_data_folder || exit 1
    say "the plugin data folder of $PS_ID: $PS_DATA"
    if [ -n "${TOPOPT_ENV_DIR:-}${EVAL_TOPOPT_ENV_DIR:-}" ]; then
      say "TOPOPT_ENV_DIR or EVAL_TOPOPT_ENV_DIR is set, and it comes first for the server too: the setup uses $ENV_DIR"
    fi
    CLAUDE_PLUGIN_DATA=$PS_DATA sh "$SELF" sync >/dev/null || exit 1
    CLAUDE_PLUGIN_DATA=$PS_DATA exec sh "$SELF" status ;;   # the same check as the first start of the server: "ready", exit 0
  python)
    ensure || exit 1
    exec "$PY" "$@" ;;
  topopt)
    ensure || exit 1
    exec "$PY" -P -m topopt "$@" ;;   # -P: as for mcp. The python case above gives the arguments of the user, so it has no -P.
  *)
    say "usage: sh $(quote "$SELF") sync | status | doctor | mcp | plugin-sync | python ARGS... | topopt ARGS..."
    say "status: exit 0 = ready, exit 1 = not ready. doctor: exit 0 = ready and every import works, exit 1 = an action is needed (it prints the command)."
    exit 2 ;;
esac
