#!/bin/sh
# Put a computer into the state of a new user for the install tests of topopt, and put it back (issue #337).
#
#   sh scripts/dev/clean-room.sh status                         what a new user would not have (changes nothing)
#   sh scripts/dev/clean-room.sh hide [--apply] [--backup-dir D] move it into one dated backup folder (dry run without --apply)
#   sh scripts/dev/clean-room.sh restore [--apply] [--from D] [--keep-changed-configs]
#                                                               put it back from the manifest of a backup folder
#   sh scripts/dev/clean-room.sh shell [options] [-- command]    a new-user shell in a temporary home (changes nothing else)
#
# `status` exits with 0 when there is nothing to hide, and with 1 when there is.
#
# What hide does, by kind (the table is in docs/reference/install.md, "Clean-room helper"):
#   moved          a file or folder inside your home is moved (mv) into the backup folder. Nothing is deleted.
#   brew-unlink    `brew unlink uv` (reversible; restore runs `brew link uv`). Homebrew is not uninstalled.
#   edited-config  the topopt entry is taken out of an MCP config file of a client. The whole original file is copied into the
#                  backup folder first, and a copy of the file without the entry is written. Why: the file also holds the other
#                  servers of the user, so the file is never moved away and never rewritten from a guess. Restore puts the original
#                  back byte for byte, but only if the file is still the one that hide wrote. If the client changed it since, restore
#                  leaves it and says where the original is. The edit is done with awk (no python: on macOS /usr/bin/python3 can be
#                  a stub that starts the Xcode tools install). If the awk edit does not fit the file, the file is reported only.
#   reported-only  said in the output, never changed: a uv or a Python outside your home (system, Homebrew folders, /opt/...),
#                  uv tools, the plugin install of Claude Code, and the environment variables that this script reads.
# Every path that is moved must be inside $HOME and on the file system of the backup folder (a cross-file-system move is refused,
# not copied). The default backup folder is $HOME/.topopt-clean-room/<UTC date>.
#
# The script reads these variables: HOME, PATH, TMPDIR, XDG_CONFIG_HOME, XDG_DATA_HOME, XDG_CACHE_HOME, XDG_STATE_HOME,
# UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, UV_TOOL_DIR, TOPOPT_UV, TOPOPT_ENV_DIR, EVAL_TOPOPT_ENV_DIR, TOPOPT_STATE_DIR,
# CLAUDE_PLUGIN_DATA, CLAUDE_CONFIG_DIR, CODEX_HOME, CLEAN_ROOM_BACKUP_DIR.
# Nothing here uses bash features. It runs with the macOS /bin/sh and with dash. Quit the clients (and any claude or codex session)
# before `hide --apply` and `restore --apply`.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P) || exit 1
TAB=$(printf '\t')

err() { printf 'clean-room: %s\n' "$*" >&2; }
say() { printf '%s\n' "$*"; }
usage() {
  cat >&2 <<'EOF'
usage: clean-room.sh status
       clean-room.sh hide [--apply] [--backup-dir DIR]
       clean-room.sh restore [--apply] [--from BACKUP_DIR] [--backup-dir ROOT] [--keep-changed-configs]
       clean-room.sh shell [--dir DIR] [--auth-file PATH_IN_HOME] [--env NAME]... [--path PATH] [-- COMMAND...]
EOF
  exit 2
}

# --- home and helpers -----------------------------------------------------------------------------------------------

[ -n "${HOME:-}" ] || { err "HOME is not set or is /. Stop."; exit 1; }
while [ "$HOME" != "${HOME%/}" ]; do HOME=${HOME%/}; done
[ -n "$HOME" ] || { err "HOME is not set or is /. Stop."; exit 1; }
[ -d "$HOME" ] || { err "HOME ($HOME) is not a folder. Stop."; exit 1; }
BACKUP_ROOT=${CLEAN_ROOM_BACKUP_DIR:-$HOME/.topopt-clean-room}
OS=$(uname -s 2>/dev/null)

exists() { [ -e "$1" ] || [ -L "$1" ]; }
no_final_slash() { r=$1; while [ "$r" != "${r%/}" ]; do r=${r%/}; done; printf '%s' "$r"; }
BREW_NOTE="(the Homebrew uv is then off PATH for all your projects, not only topopt, until 'restore --apply')"

# A path that hide may move: inside $HOME, not $HOME itself, no ".." part.
inside_home() {
  case "$1" in
    "$HOME"/*) case "$1" in */..|*/../*) return 1 ;; esac; return 0 ;;
  esac
  return 1
}

# The checksum of a file, "<cksum>-<bytes>".
sum_of() { cksum <"$1" 2>/dev/null | { read -r a b _; printf '%s-%s' "$a" "$b"; }; }
# The size of a path in KiB.
size_of() { du -sk "$1" 2>/dev/null | { read -r k _; printf '%s' "${k:-0}"; }; }
# The mount point of the file system that holds a path (df -P: the last field is the mount point).
mount_of() { df -P "$1" 2>/dev/null | tail -n 1 | sed 's/^.*[0-9][0-9]*% *//'; }
# Both mount points known and equal. An empty value (df failed) counts as different.
same_mount() { [ -n "$1" ] && [ -n "$2" ] && [ "$1" = "$2" ]; }
# A name of an environment variable, before it goes into eval.
valid_name() { case "$1" in ''|[0-9]*|*[!A-Za-z0-9_]*) return 1 ;; esac; return 0; }

WORK=
make_work() {
  WORK=$(mktemp -d "${TMPDIR:-/tmp}/clean-room.XXXXXX") || { err "cannot make a temporary folder"; exit 1; }
  ITEMS=$WORK/items; CHECKED=$WORK/checked; EDIT=$WORK/edit.out
  : >"$ITEMS"; : >"$CHECKED"
  trap 'cleanup_work' EXIT
  trap 'exit 130' INT TERM
}
TMPFILE=
cleanup_work() {
  [ -n "$TMPFILE" ] && rm -f "$TMPFILE"
  [ -n "$WORK" ] && [ -d "$WORK" ] || return 0
  rm -f "$WORK/items" "$WORK/checked" "$WORK/edit.out" "$WORK/rows"
  rmdir "$WORK" 2>/dev/null
}

# --- the config edits (awk; no python) ---------------------------------------------------------------------------------

# JSON and JSONC: remove the first key "topopt" whose object is the value of "mcpServers" or "servers".
# Output: the new text on stdout; exit 3 when there is no such entry; exit 4 when the entry is not an object (not edited).
JSON_AWK='
{ buf = buf $0 "\n" }
function isws(ch) { return ch == " " || ch == "\t" || ch == "\n" || ch == "\r" }
function skipws(p,   ch) { while (p <= n && isws(substr(buf, p, 1))) p++; return p }
function matchend(p,   dd, ch, q) {
  dd = 0
  while (p <= n) {
    ch = substr(buf, p, 1)
    if (ch == "\"") { p++; while (p <= n) { q = substr(buf, p, 1); if (q == "\\") p += 2; else if (q == "\"") break; else p++ } }
    else if (ch == "/" && substr(buf, p + 1, 1) == "/") { while (p <= n && substr(buf, p, 1) != "\n") p++ }
    else if (ch == "/" && substr(buf, p + 1, 1) == "*") { p += 2; while (p < n && substr(buf, p, 2) != "*/") p++; p++ }
    else if (ch == "{" || ch == "[") dd++
    else if (ch == "}" || ch == "]") { dd--; if (dd == 0) return p }
    p++
  }
  return 0
}
END {
  n = length(buf); d = 0; i = 1
  while (i <= n) {
    c = substr(buf, i, 1)
    if (c == "\"") {
      s = i; i++
      while (i <= n) { ch = substr(buf, i, 1); if (ch == "\\") i += 2; else if (ch == "\"") break; else i++ }
      str = substr(buf, s + 1, i - s - 1); i++
      if (d > 0 && kind[d] == "o" && expk[d]) {
        expk[d] = 0; lastkey[d] = str
        if (str == "topopt" && (ckey[d] == "mcpServers" || ckey[d] == "servers")) {
          j = skipws(i)
          if (substr(buf, j, 1) != ":") continue
          j = skipws(j + 1)
          if (substr(buf, j, 1) != "{") exit 4
          e = matchend(j)
          if (e == 0) exit 4
          a = s; b = e + 1
          k = skipws(b)
          if (substr(buf, k, 1) == ",") {
            b = k + 1
            ls = a; while (ls > 1 && substr(buf, ls - 1, 1) != "\n") ls--
            if (substr(buf, ls, a - ls) ~ /^[ \t]*$/) {
              le = b; while (le <= n && substr(buf, le, 1) != "\n" && (substr(buf, le, 1) == " " || substr(buf, le, 1) == "\t" || substr(buf, le, 1) == "\r")) le++
              if (le > n || substr(buf, le, 1) == "\n") { a = ls; b = le + 1 }
            }
          } else {
            p = a - 1; while (p >= 1 && isws(substr(buf, p, 1))) p--
            if (p >= 1 && substr(buf, p, 1) == ",") a = p
            else {
              ls = a; while (ls > 1 && substr(buf, ls - 1, 1) != "\n") ls--
              if (substr(buf, ls, a - ls) ~ /^[ \t]*$/) {
                le = b; while (le <= n && (substr(buf, le, 1) == " " || substr(buf, le, 1) == "\t" || substr(buf, le, 1) == "\r")) le++
                if (le > n || substr(buf, le, 1) == "\n") { a = ls; b = le + 1 }
              }
            }
          }
          printf "%s%s", substr(buf, 1, a - 1), substr(buf, b)
          exit 0
        }
      }
      continue
    }
    if (c == "/" && substr(buf, i + 1, 1) == "/") { while (i <= n && substr(buf, i, 1) != "\n") i++; continue }
    if (c == "/" && substr(buf, i + 1, 1) == "*") { i += 2; while (i < n && substr(buf, i, 2) != "*/") i++; i += 2; continue }
    if (c == "{" || c == "[") {
      parent = (d > 0 && kind[d] == "o") ? lastkey[d] : ""
      d++; kind[d] = (c == "{") ? "o" : "a"; ckey[d] = parent; expk[d] = (c == "{") ? 1 : 0
    } else if (c == "}" || c == "]") { if (d > 0) d-- }
    else if (c == "," && d > 0 && kind[d] == "o") expk[d] = 1
    i++
  }
  exit 3
}'

# TOML (Codex CLI): remove the tables [mcp_servers.topopt] and [mcp_servers.topopt.<sub>]. A line that starts with "[" inside a multi-line array
# is not a table header (the depth of the brackets is counted). Comment lines just above the next table stay with that table.
# Exit 0 (edited), 3 (no such table), 4 (the table holds a multi-line string: not edited).
TOML_AWK='
BEGIN { skip = 0; dropped = 0; depth = 0; nb = 0; bad = 0 }
{
  t = $0; sub(/^[ \t]+/, "", t)
  st = t; gsub(/"[^"]*"/, "", st); gsub(/\047[^\047]*\047/, "", st); sub(/#.*/, "", st)
  hdr = (depth == 0 && t ~ /^\[/)
  if (skip && (t ~ /"""/ || t ~ /\047\047\047/)) bad = 1
  if (hdr) {
    skip = (t ~ /^\[[ \t]*mcp_servers[ \t]*\.[ \t]*(topopt|"topopt"|\047topopt\047)[ \t]*(\.|\])/) ? 1 : 0
    if (!skip) for (i = 1; i <= nb; i++) print buf[i]
    nb = 0
  }
  if (skip) {
    dropped = 1
    if (!hdr && depth == 0 && (t == "" || t ~ /^#/)) buf[++nb] = $0; else nb = 0
  } else print $0
  depth += gsub(/\[/, "[", st) - gsub(/\]/, "]", st)
  if (depth < 0) depth = 0
}
END { if (bad) exit 4; exit (dropped ? 0 : 3) }'

# Exit 0 when the TOML file has a topopt server in a form that TOML_AWK does not handle: an inline table under [mcp_servers], a dotted key.
TOML_FORMS_AWK='
{
  t = $0; sub(/^[ \t]+/, "", t); sub(/[ \t\r]+$/, "", t)
  if (t ~ /^\[/) { inms = (t ~ /^\[[ \t]*mcp_servers[ \t]*\]/) ? 1 : 0; next }
  if (t ~ /^mcp_servers[ \t]*\.[ \t]*(topopt|"topopt"|\047topopt\047)[ \t]*[.=]/) f = 1
  if (t ~ /^mcp_servers[ \t]*=.*topopt/) f = 1
  if (inms && t ~ /^(topopt|"topopt"|\047topopt\047)[ \t]*[.=]/) f = 1
}
END { exit (f ? 0 : 1) }'

# Exit 1 when the JSON text has a comma right after "{" or "[", right before "}" or "]", or two commas (comments and strings are skipped).
JSON_CHECK_AWK='
{ buf = buf $0 "\n" }
END {
  n = length(buf); prev = ""; i = 1; bad = 0
  while (i <= n) {
    c = substr(buf, i, 1)
    if (c == "\"") { i++; while (i <= n) { ch = substr(buf, i, 1); if (ch == "\\") i += 2; else if (ch == "\"") break; else i++ }; i++; prev = "s"; continue }
    if (c == "/" && substr(buf, i + 1, 1) == "/") { while (i <= n && substr(buf, i, 1) != "\n") i++; continue }
    if (c == "/" && substr(buf, i + 1, 1) == "*") { i += 2; while (i < n && substr(buf, i, 2) != "*/") i++; i += 2; continue }
    if (c == " " || c == "\t" || c == "\n" || c == "\r") { i++; continue }
    if (c == "," && (prev == "{" || prev == "[" || prev == ",")) bad = 1
    if ((c == "}" || c == "]") && prev == ",") bad = 1
    prev = c; i++
  }
  exit (bad ? 1 : 0)
}'

# edit_config FILE: write the edited text to $EDIT.
# Returns 0 (edited), 3 (no topopt entry), 4 (the edit does not fit), 5 (a topopt entry in a form that the edit does not handle),
# 6 (the edited text looks wrong: nothing is written).
edit_config() {
  case "$1" in
    *.toml)
      awk "$TOML_AWK" <"$1" >"$EDIT"; er=$?
      if [ "$er" -eq 3 ]; then
        if awk "$TOML_FORMS_AWK" <"$1"; then return 5; fi
        return 3
      fi
      [ "$er" -eq 0 ] || return "$er"
      if awk "$TOML_FORMS_AWK" <"$EDIT"; then return 5; fi
      return 0 ;;
    *)
      awk "$JSON_AWK" <"$1" >"$EDIT"; er=$?
      [ "$er" -eq 0 ] || return "$er"
      awk "$JSON_CHECK_AWK" <"$EDIT" || return 6
      return 0 ;;
  esac
}

# --- the list of items -------------------------------------------------------------------------------------------------
# Each row of $ITEMS: kind TAB path TAB reason.

add_item() {
  case "$2" in *"$TAB"*|*'
'*) err "skipped a path with a tab or a newline in it: $2"; return ;; esac
  awk -F'\t' -v p="$2" '$2 == p { f = 1 } END { exit !f }' "$ITEMS" && return
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$ITEMS"
}
# A candidate that hide may move. Present: moved (inside home) or reported-only. Absent: only the CHECKED list.
add_movable() {
  if exists "$1"; then
    if inside_home "$1"; then add_item moved "$1" "$2"
    else add_item reported-only "$1" "$2; it is outside your home, so it is not moved"; fi
  else
    case "$1" in *'*'*) printf '%s (no match)\n' "$1" >>"$CHECKED" ;; *) printf '%s\n' "$1" >>"$CHECKED" ;; esac
  fi
}

uv_dir_of() { printf '%s' "${UV_PYTHON_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/uv/python}"; }
# A path from a variable, as topopt reads it: a leading "~" or "~/" is the home folder (a client config can pass the text as it is).
home_path() { case $1 in '~') printf '%s' "$HOME" ;; '~/'*) printf '%s/%s' "$HOME" "${1#\~/}" ;; *) printf '%s' "$1" ;; esac; }

collect() {
  : >"$ITEMS"; : >"$CHECKED"
  # 1. Homebrew uv.
  BREW_BIN=
  if command -v brew >/dev/null 2>&1; then
    if [ -n "$(brew list --versions uv 2>/dev/null)" ]; then
      prefix=$(brew --prefix 2>/dev/null)
      if [ -n "$prefix" ]; then
        BREW_BIN=$prefix/bin
        if exists "$prefix/bin/uv"; then add_item brew-unlink "$prefix/bin/uv" "uv from Homebrew; hide runs brew unlink uv, restore runs brew link uv"
        else printf '%s\n' "$prefix/bin/uv (Homebrew uv, already unlinked)" >>"$CHECKED"; fi
      fi
    fi
  fi
  # 2. The uv programs of the standalone installer and of cargo.
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    for n in uv uvx uvw; do add_movable "$d/$n" "uv program (installer or cargo)"; done
  done
  # 3. Any other uv on PATH.
  set -f
  oldifs=$IFS; IFS=:
  for d in $PATH; do
    IFS=$oldifs
    [ -n "$d" ] || { IFS=:; continue; }
    if [ "$d" != "$BREW_BIN" ] && { [ -x "$d/uv" ] && [ ! -d "$d/uv" ]; }; then
      if inside_home "$d"; then
        for n in uv uvx uvw; do exists "$d/$n" && add_item moved "$d/$n" "uv program found on PATH inside your home"; done
      else
        add_item reported-only "$d/uv" "uv on PATH outside your home (system or package manager): not moved; 'shell' runs without it"
      fi
    fi
    IFS=:
  done
  IFS=$oldifs
  set +f
  # 4. The uv-managed Pythons, the links to them in ~/.local/bin, the uv tools, the uv cache.
  pydir=$(uv_dir_of)
  add_movable "$pydir" "folder of the uv-managed Pythons"
  if [ -d "$HOME/.local/bin" ]; then
    for f in "$HOME/.local/bin"/*; do
      [ -L "$f" ] || continue
      t=$(readlink "$f" 2>/dev/null)
      case "$t" in "$pydir"/*|*/uv/python/*) add_item moved "$f" "link to a uv-managed Python" ;; esac
    done
  fi
  tools=${UV_TOOL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/uv/tools}
  exists "$tools" && add_item reported-only "$tools" "uv tools: not moved; they stop working while the uv Pythons are hidden"
  add_movable "${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}" "uv cache"
  # 5. The topopt state.
  state=$(home_path "${TOPOPT_STATE_DIR:-}")
  if [ -z "$state" ]; then
    if [ "$OS" = Darwin ]; then state="$HOME/Library/Application Support/topopt"
    else case "${XDG_STATE_HOME:-}" in /*) state=$XDG_STATE_HOME/topopt ;; *) state=$HOME/.local/state/topopt ;; esac; fi
  fi
  add_movable "$state" "live-server registry of topopt"
  # The per-user state folder of gpu-mg (gpu_mg.check.state_dir): TOPOPT_STATE_DIR when set (the folder above), else ~/.topopt.
  [ -z "${TOPOPT_STATE_DIR:-}" ] && add_movable "$HOME/.topopt" "per-user state of gpu-mg (the calibration of topopt gpu-check)"
  # The kernel caches of pyopencl, which gpu-mg uses: warm kernels make the first GPU self-test of a new user faster. The rule of
  # pyopencl (cache.py) and pytools (persistent_dict.py, the kernel invokers): $XDG_CACHE_HOME, else ~/Library/Caches on macOS, else ~/.cache.
  if [ "$OS" = Darwin ]; then kcache=${XDG_CACHE_HOME:-$HOME/Library/Caches}; else kcache=${XDG_CACHE_HOME:-$HOME/.cache}; fi
  add_movable "$kcache/pyopencl" "kernel cache of pyopencl (the GPU solver gpu-mg)"
  add_movable "$kcache/pytools" "cache of pytools (pyopencl keeps its kernel invokers there; other pytools users share it)"
  add_movable "$ROOT/.plugin-env" "environment, downloaded uv and setup.log of scripts/topopt-env.sh in this checkout"
  [ -n "${TOPOPT_ENV_DIR:-}" ] && add_movable "$TOPOPT_ENV_DIR" "environment folder named by TOPOPT_ENV_DIR"
  [ -n "${EVAL_TOPOPT_ENV_DIR:-}" ] && add_movable "$EVAL_TOPOPT_ENV_DIR" "environment folder named by EVAL_TOPOPT_ENV_DIR"
  [ -n "${CLAUDE_PLUGIN_DATA:-}" ] && add_movable "$CLAUDE_PLUGIN_DATA" "plugin data folder named by CLAUDE_PLUGIN_DATA"
  cc=${CLAUDE_CONFIG_DIR:-$HOME/.claude}
  for p in "$cc"/plugins/data/topopt-*; do add_movable "$p" "Claude Code plugin data of topopt (environment and setup.log)"; done
  if [ -f "$cc/plugins/installed_plugins.json" ] && grep -q 'topopt@' "$cc/plugins/installed_plugins.json" 2>/dev/null; then
    add_item reported-only "$cc/plugins/installed_plugins.json" "the topopt plugin is installed; hide does not edit it. For the install test run: claude plugin uninstall topopt@<marketplace>, then claude plugin marketplace remove <marketplace> (this also deletes the plugin data)"
  elif [ -f "$cc/plugins/installed_plugins.json" ]; then
    printf '%s (no topopt plugin)\n' "$cc/plugins/installed_plugins.json" >>"$CHECKED"
  else
    printf '%s\n' "$cc/plugins/installed_plugins.json" >>"$CHECKED"
  fi
  # The Claude Desktop extension folders (macOS; the folder names are from reading, not checked on a Mac).
  # Checked on a Mac, or where that folder exists; on Linux the status does not list macOS paths.
  if [ "$OS" = Darwin ] || [ -d "$HOME/Library/Application Support/Claude" ]; then
    for base in "$HOME/Library/Application Support/Claude/Claude Extensions" "$HOME/Library/Application Support/Claude/Claude Extensions Settings"; do
      for p in "$base"/*topopt*; do add_movable "$p" "Claude Desktop extension of topopt"; done
    done
  fi
  # 6. The MCP configs of the clients.
  if [ "$OS" = Darwin ]; then vsc="$HOME/Library/Application Support/Code/User/mcp.json"; else vsc="${XDG_CONFIG_HOME:-$HOME/.config}/Code/User/mcp.json"; fi
  for cfg in "$HOME/.cursor/mcp.json" "${CODEX_HOME:-$HOME/.codex}/config.toml" "$HOME/.gemini/settings.json" "$vsc" \
             "$HOME/Library/Application Support/Claude/claude_desktop_config.json"; do
    if [ -f "$cfg" ] || [ -L "$cfg" ]; then
      if [ -L "$cfg" ]; then
        grep -q 'topopt' "$cfg" 2>/dev/null && add_item reported-only "$cfg" "MCP config is a symbolic link: not edited. Take the topopt entry out by hand"
        continue
      fi
      edit_config "$cfg" 2>/dev/null; rc=$?
      case $rc in
        0) add_item edited-config "$cfg" "topopt entry in an MCP config; the other servers stay" ;;
        3) printf '%s (no topopt entry)\n' "$cfg" >>"$CHECKED" ;;
        5) add_item reported-only "$cfg" "topopt is a key in a form that the edit does not handle (an inline table or a dotted key). Take the entry out by hand; hide did not change this file" ;;
        *) add_item reported-only "$cfg" "this file has a topopt entry that the awk edit cannot take out safely (code $rc). Take it out by hand; hide did not change this file" ;;
      esac
    else
      printf '%s\n' "$cfg" >>"$CHECKED"
    fi
  done
  # Claude Code keeps user and local MCP servers in .claude.json (a big file that Claude Code writes itself): reported only.
  for cj in "$HOME/.claude.json" "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/.claude.json"; do
    if [ -f "$cj" ] && grep -Eq '"topopt"[[:space:]]*:' "$cj" 2>/dev/null; then
      add_item reported-only "$cj" "Claude Code has a topopt server here; hide does not edit this file. Remove it with: claude mcp remove topopt (add the scope, user or local, that you used)"
    fi
  done
}

count_hideable() { awk -F'\t' '$1 != "reported-only" { n++ } END { print n + 0 }' "$ITEMS"; }

# --- status ----------------------------------------------------------------------------------------------------------

cmd_status() {
  make_work; collect
  say "home:            $HOME"
  say "backup folder:   $BACKUP_ROOT"
  for v in TOPOPT_UV TOPOPT_ENV_DIR EVAL_TOPOPT_ENV_DIR TOPOPT_STATE_DIR CLAUDE_PLUGIN_DATA CLAUDE_CONFIG_DIR UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_TOOL_DIR; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && say "variable set:   $v=$val (the script uses it to find the folders)"
  done
  say ""
  say "Things that a new user would not have:"
  if [ -s "$ITEMS" ]; then
    while IFS="$TAB" read -r kind path reason <&3; do
      case "$kind" in
        moved) sz=$(size_of "$path"); printf '  %-14s %s  (%s KiB)\n                 %s\n' "$kind" "$path" "$sz" "$reason" ;;
        *) printf '  %-14s %s\n                 %s\n' "$kind" "$path" "$reason" ;;
      esac
    done 3<"$ITEMS"
  else
    say "  (none)"
  fi
  say ""
  say "Checked and not there:"
  if [ -s "$CHECKED" ]; then sed 's/^/  /' "$CHECKED"; else say "  (none)"; fi
  say ""
  n=$(count_hideable)
  if [ "$n" -eq 0 ]; then say "clean: nothing for 'hide' to do."; return 0; fi
  say "not clean: $n item(s) for 'hide' (reported-only items are never changed)."
  return 1
}

# --- hide ------------------------------------------------------------------------------------------------------------
# The manifest is written ahead: the row (state "pending") goes in BEFORE the move or the edit, and a line "done" after it. So an
# interrupt at any point leaves a row for every item that can be in the backup folder. Restore reads the "pending" rows and looks at the
# disk to see what happened. Columns of a row: pending, kind, original path, backup path, size KiB, cksum before, cksum after, note.

write_row() { printf 'pending\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >>"$BK/manifest.tsv"; }
mark_done() { printf 'done\t%s\n' "$1" >>"$BK/manifest.tsv"; }

# The backup folders (dated, in the root $1) that hold rows and have no "restored" marker. One path for each line.
unfinished_in() {
  for f in "$1"/*/manifest.tsv; do
    [ -f "$f" ] || continue
    d=$(dirname "$f")
    [ -e "$d/restored" ] && continue
    grep -q '^pending' "$f" 2>/dev/null && printf '%s\n' "$d"
  done
}
unfinished_backups() { unfinished_in "$BACKUP_ROOT"; }
# For hide: the root in use, the default root, and every other root that an earlier hide wrote in the file "roots" of the default root.
# Each folder is listed once.
unfinished_for_hide() {
  DEFAULT_ROOT=$HOME/.topopt-clean-room
  { printf '%s\n%s\n' "$BACKUP_ROOT" "$DEFAULT_ROOT"; cat "$DEFAULT_ROOT/roots" 2>/dev/null; } | awk 'NF && !s[$0]++' | while IFS= read -r r; do
    unfinished_in "$r"
  done
}

cmd_hide() {
  apply=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --apply) apply=1; shift ;;
      --backup-dir) [ $# -ge 2 ] || usage; BACKUP_ROOT=$2; shift 2 ;;
      *) usage ;;
    esac
  done
  make_work; collect
  n=$(count_hideable)
  if [ "$n" -eq 0 ]; then say "nothing to hide: this home looks clean."; show_reported; return 0; fi
  BACKUP_ROOT=$(no_final_slash "$BACKUP_ROOT")
  case "$BACKUP_ROOT" in /*) ;; *) err "the backup folder must be a full path"; return 1 ;; esac
  unf=$(unfinished_for_hide)
  if [ -n "$unf" ]; then
    err "a backup is not restored yet. Restore before the next hide, or the leftovers of the test go into a second backup and the first one stays hidden."
    printf '%s\n' "$unf" | while IFS= read -r d; do
      err "  not restored: $d"
      err "  run: sh $ROOT/scripts/dev/clean-room.sh restore --apply --from $d"
    done
    return 1
  fi
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  BK=$BACKUP_ROOT/$stamp
  k=1; while [ -e "$BK" ]; do k=$((k + 1)); BK=$BACKUP_ROOT/${stamp}_$k; done
  say "home:          $HOME"
  say "backup folder: $BK"
  if [ "$apply" -eq 0 ]; then say "dry run: nothing is changed. Add --apply to act."; fi
  if [ "$apply" -eq 1 ]; then
    inside_home "$BACKUP_ROOT" || { err "the backup folder must be inside $HOME (same file system as the items). Stop."; return 1; }
    mkdir -p "$BK/files" || { err "cannot make $BK"; return 1; }
    # A root that is not the default one is written down in the default root, so that the next hide finds this backup.
    if [ "$BACKUP_ROOT" != "$HOME/.topopt-clean-room" ]; then
      { mkdir -p "$HOME/.topopt-clean-room" && printf '%s\n' "$BACKUP_ROOT" >>"$HOME/.topopt-clean-room/roots"; } \
        || err "cannot write $HOME/.topopt-clean-room/roots: a later hide in another backup folder will not find this backup"
    fi
    {
      printf '# clean-room manifest v2\n# date\t%s\n# home\t%s\n' "$stamp" "$HOME"
      printf '# columns: pending, kind, original path, backup path, size KiB, cksum before, cksum after, note. A line "done" follows when the item is in place.\n'
    } >"$BK/manifest.tsv" || { err "cannot write the manifest. Stop."; return 1; }
    bkmount=$(mount_of "$BK")
    [ -n "$bkmount" ] || { err "cannot find the file system of $BK (df printed nothing). Stop; nothing was moved."; return 1; }
  fi
  nn=0; fails=0
  while IFS="$TAB" read -r kind path reason <&3; do
    case "$kind" in
      reported-only) printf '  report-only   %s\n                %s\n' "$path" "$reason"; continue ;;
    esac
    nn=$((nn + 1)); dest=$BK/files/$(printf '%03d' "$nn")
    case "$kind" in
      moved)
        if [ "$apply" -eq 0 ]; then printf '  would move    %s\n             -> %s\n' "$path" "$dest"; continue; fi
        if ! inside_home "$path" || ! exists "$path"; then err "skip $path (not inside home, or gone)"; fails=$((fails + 1)); continue; fi
        pm=$(mount_of "$(dirname "$path")")
        if ! same_mount "$pm" "$bkmount"; then err "skip $path: it is not on the same file system as the backup folder (or df gave no answer: '$pm', '$bkmount'). Use --backup-dir on the same one."; fails=$((fails + 1)); continue; fi
        sz=$(size_of "$path"); sm=-; [ -f "$path" ] && [ ! -L "$path" ] && sm=$(sum_of "$path")
        write_row moved "$path" "$dest" "$sz" "$sm" - "$reason" || { err "cannot write the manifest: $path is not moved"; fails=$((fails + 1)); continue; }
        if mv "$path" "$dest" && exists "$dest" && ! exists "$path"; then
          mark_done "$dest"; printf '  moved         %s\n' "$path"
        else err "move failed: $path"; fails=$((fails + 1)); fi ;;
      brew-unlink)
        if [ "$apply" -eq 0 ]; then printf '  would run     brew unlink uv   (%s)\n                %s\n' "$path" "$BREW_NOTE"; continue; fi
        write_row brew-unlink "$path" - 0 - - "$reason" || { err "cannot write the manifest: brew uv is not unlinked"; fails=$((fails + 1)); continue; }
        if brew unlink uv >/dev/null 2>&1; then mark_done -; printf '  unlinked      brew uv\n                %s\n' "$BREW_NOTE"
        else err "brew unlink uv failed"; fails=$((fails + 1)); fi ;;
      edited-config)
        if [ "$apply" -eq 0 ]; then printf '  would edit    %s\n             (copy of the original -> %s; topopt entry out)\n' "$path" "$dest"; continue; fi
        if [ -L "$path" ] || ! edit_config "$path" 2>/dev/null; then err "skip $path (cannot edit it safely)"; fails=$((fails + 1)); continue; fi
        before=$(sum_of "$path"); after=$(sum_of "$EDIT")
        tmp=$path.clean-room-tmp.$$; TMPFILE=$tmp
        write_row edited-config "$path" "$dest" "$(size_of "$path")" "$before" "$after" "$reason" || { err "cannot write the manifest: $path is not edited"; fails=$((fails + 1)); continue; }
        if cp -p "$path" "$dest" && [ "$(sum_of "$dest")" = "$before" ] \
           && cp -p "$path" "$tmp" && cat "$EDIT" >"$tmp" && mv "$tmp" "$path" && TMPFILE= && [ "$(sum_of "$path")" = "$after" ]; then
          mark_done "$dest"; printf '  edited        %s\n' "$path"
          rest=$(grep -n 'topopt' "$path" 2>/dev/null | head -n 3)
          [ -n "$rest" ] && printf '                note: the word topopt is still in this file (maybe another server). Lines:\n%s\n' "$rest"
        else rm -f "$tmp"; err "edit failed: $path (the original is untouched or in $dest)"; fails=$((fails + 1)); fi ;;
    esac
  done 3<"$ITEMS"
  if [ "$apply" -eq 0 ]; then return 0; fi
  # The last line: restore knows that hide reached the end of its loop. With a failed item, some items of the user stayed in
  # place with no row, so the line says so, and restore does not call the items that it finds "new".
  if [ "$fails" -eq 0 ]; then printf '# hide finished\n' >>"$BK/manifest.tsv"; else printf '# hide finished with %s failed item(s)\n' "$fails" >>"$BK/manifest.tsv"; fi \
    || err "cannot write the last line of the manifest"
  say "manifest:      $BK/manifest.tsv"
  say "restore with:  sh $ROOT/scripts/dev/clean-room.sh restore --apply --from $BK"
  [ "$fails" -eq 0 ] || { err "$fails item(s) failed. Run status."; return 1; }
  return 0
}
show_reported() { awk -F'\t' '$1 == "reported-only" { printf "  report-only   %s\n                %s\n", $2, $3 }' "$ITEMS"; }

# --- restore ---------------------------------------------------------------------------------------------------------

cmd_restore() {
  apply=0; from=; keep=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --apply) apply=1; shift ;;
      --keep-changed-configs) keep=1; shift ;;
      --from) [ $# -ge 2 ] || usage; from=$2; shift 2 ;;
      --backup-dir) [ $# -ge 2 ] || usage; BACKUP_ROOT=$2; shift 2 ;;
      *) usage ;;
    esac
  done
  if [ -z "$from" ]; then
    # The newest backup that is not restored; else the newest one that has a manifest.
    unf=$(unfinished_backups)
    if [ -n "$unf" ]; then
      from=$(printf '%s\n' "$unf" | tail -n 1)
      [ "$(printf '%s\n' "$unf" | wc -l)" -gt 1 ] && err "more than one backup is not restored. Restoring the newest; run restore again for the others:" && printf '%s\n' "$unf" | sed 's/^/  /' >&2
    else
      for f in "$BACKUP_ROOT"/*/manifest.tsv; do [ -f "$f" ] && from=$(dirname "$f"); done
    fi
    [ -n "$from" ] || { err "no backup with a manifest in $BACKUP_ROOT. Use --from."; return 1; }
  fi
  M=$from/manifest.tsv
  [ -f "$M" ] || { err "no manifest: $M"; return 1; }
  make_work
  say "home:          $HOME"
  say "backup folder: $from"
  [ "$apply" -eq 0 ] && say "dry run: nothing is changed. Add --apply to act."
  # Reverse order of the manifest.
  grep '^pending' "$M" | awk '{ l[NR] = $0 } END { for (i = NR; i >= 1; i--) print l[i] }' >"$WORK/rows"
  skipped=0; failed=0; done_n=0; kept=0
  while IFS="$TAB" read -r state kind path bpath sz before after note <&3; do
    case "$kind" in
      moved)
        if ! exists "$bpath"; then
          if exists "$path"; then printf '  ok            %s is in place (restored already, or the move never happened)\n' "$path"
          else printf '  FAIL          %s: neither the item nor its backup %s is there\n' "$path" "$bpath"; failed=$((failed + 1)); fi
          continue
        fi
        if exists "$path"; then printf '  SKIP          %s\n                exists now (new since hide); not overwritten. The backup stays at %s\n' "$path" "$bpath"; skipped=$((skipped + 1)); continue; fi
        if ! inside_home "$path"; then printf '  SKIP          %s\n                outside home\n' "$path"; skipped=$((skipped + 1)); continue; fi
        if [ "$apply" -eq 0 ]; then printf '  would move    %s\n             <- %s\n' "$path" "$bpath"; continue; fi
        if mkdir -p "$(dirname "$path")" && mv "$bpath" "$path" && exists "$path" && ! exists "$bpath"; then
          printf '  restored      %s\n' "$path"; done_n=$((done_n + 1))
          now=$(size_of "$path"); [ "$now" != "$sz" ] && printf '                changed since hide: %s KiB then, %s KiB now\n' "$sz" "$now"
        else printf '  FAIL          move back failed: %s\n' "$path"; failed=$((failed + 1)); fi ;;
      brew-unlink)
        if [ -e "$path" ] || [ -L "$path" ]; then printf '  ok            brew uv is linked (%s)\n' "$path"; continue; fi
        if [ "$apply" -eq 0 ]; then printf '  would run     brew link uv\n'; continue; fi
        if brew link uv >/dev/null 2>&1; then printf '  linked        brew uv\n'; done_n=$((done_n + 1)); else printf '  FAIL          brew link uv failed\n'; failed=$((failed + 1)); fi ;;
      edited-config)
        if [ -L "$path" ]; then printf '  SKIP          %s\n                is a symbolic link now; not touched\n' "$path"; skipped=$((skipped + 1)); continue; fi
        # hide copies the file before it edits it, and writes "done" after the edit. No copy and no "done": hide did not change it.
        if ! exists "$bpath" && ! grep -Fqx "done$TAB$bpath" "$M"; then
          printf '  ok            %s\n                hide did not change this file (it was stopped before the copy); left as it is\n' "$path"; continue
        fi
        why=
        if exists "$path"; then
          cs=$(sum_of "$path")
          if [ "$cs" = "$before" ]; then printf '  ok            %s is the original already\n' "$path"; continue; fi
          if [ "$cs" != "$after" ]; then
            # The client may only have added the topopt entry again: the file minus that entry is then the file that hide wrote.
            if edit_config "$path" 2>/dev/null && [ "$(sum_of "$EDIT")" = "$after" ]; then why="the client added topopt again; the rest is as hide left it"
            elif [ "$keep" -eq 1 ]; then
              if [ "$apply" -eq 0 ]; then printf '  would keep    %s\n                changed since hide; it would stay as it is. The original is at %s\n' "$path" "$bpath"
              else printf '  kept          %s\n                changed since hide; left as it is. The original stays at %s\n' "$path" "$bpath"; fi
              kept=$((kept + 1)); continue
            else
              printf '  SKIP          %s\n                changed since hide (not the file that hide wrote). Not overwritten. The original is %s. Two ways out:\n' "$path" "$bpath"
              printf "                (a) put the original back: cp '%s' '%s'  then run restore --apply again\n" "$bpath" "$path"
              printf '                (b) keep the current file: run restore --apply --keep-changed-configs\n'
              skipped=$((skipped + 1)); continue
            fi
          fi
        fi
        if ! exists "$bpath"; then printf '  FAIL          %s: the backup copy %s is gone\n' "$path" "$bpath"; failed=$((failed + 1)); continue; fi
        if [ "$apply" -eq 0 ]; then printf '  would restore %s\n             <- %s\n' "$path" "$bpath"; continue; fi
        tmp=$path.clean-room-tmp.$$; TMPFILE=$tmp
        if mkdir -p "$(dirname "$path")" && cp -p "$bpath" "$tmp" && mv "$tmp" "$path" && TMPFILE= && [ "$(sum_of "$path")" = "$before" ]; then
          printf '  restored      %s\n' "$path"; [ -n "$why" ] && printf '                (%s)\n' "$why"; done_n=$((done_n + 1))
        else rm -f "$tmp"; printf '  FAIL          restore failed: %s\n' "$path"; failed=$((failed + 1)); fi ;;
      *) ;;
    esac
  done 3<"$WORK/rows"
  rm -f "$WORK/rows"
  # A file in files/ that no row names: the manifest write failed or was cut. Restore cannot know where it belongs.
  for f in "$from"/files/*; do
    exists "$f" || continue
    # Compare the name files/NNN, not the text of the whole path (a final slash or a relative --from changes the text).
    awk -F'\t' -v p="files/${f##*/}" '$1 == "pending" { m = split($4, a, "/"); if (m >= 2 && a[m - 1] "/" a[m] == p) f = 1 } END { exit !f }' "$M" && continue
    printf '  FAIL          %s is in the backup folder and no row of the manifest names it. Look at it by hand.\n' "$f"; failed=$((failed + 1))
  done
  collect
  say ""
  if grep -qx '# hide finished' "$M"; then
    say "New since hide (a new user would not have these; they are not in the manifest):"
  else
    say "New since hide (hide did not finish, or it skipped items, so some of these can be your own items that hide did not move. Do not delete them):"
  fi
  awk -F'\t' 'NR == FNR { if ($1 == "pending") seen[$3] = 1; next } $1 != "reported-only" && !($2 in seen) { printf "  %s  %s\n", $1, $2; n++ } END { if (!n) print "  (none)" }' "$M" "$ITEMS"
  say ""
  if [ "$apply" -eq 0 ]; then
    say "dry run done: $skipped item(s) would be skipped, $failed would fail, $kept config(s) would be kept."
    return 0
  fi
  printf 'restored %s item(s), %s skipped, %s failed, on %s\n' "$done_n" "$skipped" "$failed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$from/restore.log"
  if [ "$skipped" -eq 0 ] && [ "$failed" -eq 0 ]; then
    date -u +%Y-%m-%dT%H:%M:%SZ >"$from/restored"
    if [ "$kept" -gt 0 ]; then say "restore complete: $done_n restored; nothing skipped; $kept config(s) kept as they are (the originals stay in the backup folder). This backup is marked as restored."
    else say "restore complete: $done_n restored; nothing skipped. This backup is marked as restored."; fi
    return 0
  fi
  say "restore INCOMPLETE: $done_n restored, $skipped skipped, $failed failed. Move or merge the items above by hand, then run restore again. For a config that the client changed, restore --apply --keep-changed-configs keeps the current file. The next hide stays refused until then."
  return 1
}

# --- shell -----------------------------------------------------------------------------------------------------------

cmd_shell() {
  dir=; auth=; pass=; newpath=; have_path=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --dir) [ $# -ge 2 ] || usage; dir=$2; shift 2 ;;
      --auth-file) [ $# -ge 2 ] || usage; auth=$2; shift 2 ;;
      --env) [ $# -ge 2 ] || usage; pass="$pass $2"; shift 2 ;;
      --path) [ $# -ge 2 ] || usage; newpath=$2; have_path=1; shift 2 ;;
      --) shift; break ;;
      *) usage ;;
    esac
  done
  # Check the options before anything is made.
  if [ -n "$auth" ]; then
    case "$auth" in /*|*..*) err "--auth-file is a path inside your home, without '..' (example: .codex/auth.json)"; return 1 ;; esac
    [ -f "$HOME/$auth" ] && [ ! -L "$HOME/$auth" ] || { err "no regular file $HOME/$auth"; return 1; }
  fi
  for v in $pass; do
    valid_name "$v" || { err "--env $v: not a variable name"; return 1; }
    eval "isset=\${$v+x}"
    [ -n "$isset" ] || { err "--env $v: the variable is not set"; return 1; }
  done
  if [ "$have_path" -eq 0 ]; then
    removed=; set -f; oldifs=$IFS; IFS=:
    for d in $PATH; do
      IFS=$oldifs
      if [ -z "$d" ] || inside_home "$d" || [ "$d" = "$HOME" ] || [ -x "$d/uv" ] || [ -x "$d/uvx" ]; then
        removed="$removed $d"
      else newpath=${newpath:+$newpath:}$d; fi
      IFS=:
    done
    IFS=$oldifs; set +f
  fi
  for t in sh env cat ls mkdir cp mv rm sed grep awk; do
    if [ -z "$(PATH=$newpath command -v "$t" 2>/dev/null)" ]; then
      err "the new PATH ($newpath) has no '$t'. Folders were left out because they hold a uv:${removed:- (none)}"
      err "Run 'hide --apply' first (it unlinks a Homebrew uv), or give the folders you want with --path. Stop."
      return 1
    fi
  done
  if [ -n "$(PATH=$newpath command -v uv 2>/dev/null)" ]; then err "uv is still found on the new PATH: $(PATH=$newpath command -v uv). Use --path. Stop."; return 1; fi
  if [ -n "$dir" ]; then
    if [ -e "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then err "$dir is not empty. Stop."; return 1; fi
    mkdir -p "$dir" || return 1
  else
    dir=$(mktemp -d "${TMPDIR:-/tmp}/topopt-clean-room.XXXXXX") || return 1
  fi
  case "$dir" in /*) ;; *) dir=$(CDPATH= cd -- "$dir" && pwd -P) ;; esac
  nh=$dir/home
  mkdir -p "$nh/.config" "$nh/.local/share" "$nh/.local/state" "$nh/.cache" "$nh/.npm-global/bin" "$dir/tmp" || return 1
  cat >"$nh/.bashrc" <<'EOF2'
PS1='(clean-room) \$ '
EOF2
  cp "$nh/.bashrc" "$nh/.profile"
  cat >"$nh/.zshrc" <<'EOF2'
PROMPT='(clean-room) %# '
EOF2
  if [ -n "$auth" ]; then
    mkdir -p "$(dirname "$nh/$auth")" && cp -p "$HOME/$auth" "$nh/$auth" || return 1
    say "copied one auth file: $auth (into the temporary home only)"
  fi
  say "new-user shell"
  say "  temporary home:  $nh"
  say "  PATH:            $newpath"
  [ -n "${removed:-}" ] && say "  left out of PATH:$removed"
  say "  uv:              none on PATH. topopt-env.sh downloads its pinned uv on the first sync."
  say "  npm installs:    npm install -g goes to $nh/.npm-global (NPM_CONFIG_PREFIX); its bin is on PATH."
  say "  registry:        TOPOPT_STATE_DIR=$nh/.local/state/topopt (topopt reads the account home otherwise)"
  say "  to leave:        type exit. The folder $dir stays; remove it yourself when you are done."
  prog=${SHELL:-/bin/sh}
  [ "$#" -gt 0 ] || set -- "$prog" -i
  # env -i: the new user has none of your variables. Only the ones below, and those named with --env.
  for v in $pass; do
    eval "val=\$$v"
    set -- "$v=$val" "$@"
  done
  set -- env -i HOME="$nh" PATH="$nh/.npm-global/bin:$newpath" USER="${USER:-user}" LOGNAME="${LOGNAME:-${USER:-user}}" SHELL="$prog" \
    TERM="${TERM:-dumb}" LANG="${LANG:-C}" TMPDIR="$dir/tmp" \
    XDG_CONFIG_HOME="$nh/.config" XDG_DATA_HOME="$nh/.local/share" XDG_STATE_HOME="$nh/.local/state" XDG_CACHE_HOME="$nh/.cache" \
    TOPOPT_STATE_DIR="$nh/.local/state/topopt" NPM_CONFIG_PREFIX="$nh/.npm-global" ZDOTDIR="$nh" "$@"
  "$@"
  rc=$?
  say "left the clean-room shell (exit $rc). The folder $dir is still there."
  return $rc
}

# --- main ------------------------------------------------------------------------------------------------------------

[ $# -ge 1 ] || usage
sub=$1; shift
case "$sub" in
  status) cmd_status "$@" ;;
  hide) cmd_hide "$@" ;;
  restore) cmd_restore "$@" ;;
  shell) cmd_shell "$@" ;;
  -h|--help|help) usage ;;
  *) usage ;;
esac
exit $?
