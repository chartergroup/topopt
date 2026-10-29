#!/bin/sh
# Headless end-to-end test of the topopt plugin with `claude -p` (one model session).
#
# What it does
#   1. Makes a clean Claude Code config in a temp folder (CLAUDE_CONFIG_DIR). Nothing else is installed.
#   2. Adds the checkout as a marketplace and installs the plugin topopt@topopt.
#   3. Makes the plugin Python environment BEFORE the session (uv sync of the launcher). A headless
#      session has nobody to type /mcp, so the first-start failure (30 s client wait) must not happen.
#   4. Runs ONE `claude -p --model sonnet` session in an empty temp work folder (not a git repo)
#      and asks for the hook example: check, run, result. The transcript is stream-json.
#   5. Prints the run folder and the metrics. Removes the temp config at the end.
#
# Marketplace: the script adds the CHECKOUT as a directory marketplace. Claude Code then reads the plugin in
# place from the checkout. It does not test the git-copy path that a user gets from
# `claude plugin marketplace add owner/repo` (a copy in plugins/cache/). Test that path by hand.
# The names come from the checkout: `.claude-plugin/marketplace.json` names the marketplace `topopt`, so the plugin is
# topopt@topopt and Claude Code names its data folder <plugin>-<marketplace>, which is plugins/data/topopt-topopt.
# (The tags v0.1.0 and v0.2.0 name the marketplace `topopt-demo`; this script does not fit a checkout of those tags.)
#
# Authentication: the script never prints a credential and never writes one outside the temp config.
#   - Main path: a long-lived token. Run `claude setup-token` once (on a machine with a browser) and set
#     CLAUDE_CODE_OAUTH_TOKEN in the environment. ANTHROPIC_API_KEY works too. `claude` uses it.
#   - Fallback: TOPOPT_E2E_CREDENTIALS names a copy of ~/.claude/.credentials.json. It is copied into the
#     temp config with mode 600 and deleted with the config at the end. Take care: the session can refresh
#     the token in the copy, and then the refresh token of the original can be stale.
#   - Else the session answers "Not logged in" and the script stops with an error.
#
# Usage: sh scripts/plugin-e2e.sh CHECKOUT OUT_DIR [--setup-only]
#   CHECKOUT   the repo checkout (the marketplace source)
#   OUT_DIR    where the transcript and the work folder go (outside any git repo)
# Env: MAX_TURNS (default 60), MODEL (default sonnet), PROMPT (default below), KEEP_CONFIG=1 keeps the temp config.

set -u
CHECKOUT=${1:?usage: plugin-e2e.sh CHECKOUT OUT_DIR [--setup-only]}
OUT=${2:?usage: plugin-e2e.sh CHECKOUT OUT_DIR [--setup-only]}
SETUP_ONLY=${3:-}
MAX_TURNS=${MAX_TURNS:-60}
MODEL=${MODEL:-sonnet}
PROMPT=${PROMPT:-"Use the topopt plugin to optimize the T-shirt hook example (examples/hook/hook.yaml). Check the problem first and tell me the time estimate. Then run it, wait until it is done, and give me the mass before and after, the compliance, the run time and the path of final.stl."}

CHECKOUT=$(cd "$CHECKOUT" && pwd -P) || exit 1
mkdir -p "$OUT" || exit 1
OUT=$(cd "$OUT" && pwd -P)
if git -C "$OUT" rev-parse --git-dir >/dev/null 2>&1; then
  echo "OUT_DIR is inside a git repo. Use another folder." >&2
  exit 1
fi

CFG=$(mktemp -d /tmp/topopt-e2e-cfg.XXXXXX) || exit 1
WORK=$OUT/work
rm -rf "$WORK"
mkdir -p "$WORK"
cleanup() {
  [ -n "${KEEP_CONFIG:-}" ] || rm -rf "$CFG"   # this removes a copied credentials file too
}
trap cleanup EXIT          # also runs when one of the two traps below exits
trap 'exit 130' INT
trap 'exit 143' TERM

export CLAUDE_CONFIG_DIR=$CFG
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -n "${TOPOPT_E2E_CREDENTIALS:-}" ]; then
  cp "$TOPOPT_E2E_CREDENTIALS" "$CFG/.credentials.json" && chmod 600 "$CFG/.credentials.json"
fi

echo "== install the plugin (clean config $CFG)"
claude plugin marketplace add "$CHECKOUT" || exit 1
claude plugin install topopt@topopt || exit 1
claude plugin list

echo "== make the plugin environment before the session"
DATA=$CFG/plugins/data/topopt-topopt
mkdir -p "$DATA"
START=$(date +%s)
CLAUDE_PLUGIN_DATA=$DATA sh "$CHECKOUT/scripts/topopt-env.sh" sync >/dev/null || exit 1
echo "environment ready after $(($(date +%s) - START)) s"

[ "$SETUP_ONLY" = "--setup-only" ] && { echo "setup only: stop before the session"; exit 0; }

echo "== claude -p (model $MODEL, max $MAX_TURNS turns), work folder $WORK"
cd "$WORK" || exit 1
START=$(date +%s)
claude -p "$PROMPT" \
  --model "$MODEL" \
  --max-turns "$MAX_TURNS" \
  --setting-sources user \
  --allowedTools "mcp__plugin_topopt_topopt" "Skill" "Bash(sleep:*)" "Bash(sh:*)" "Read" \
  --output-format stream-json --verbose \
  >"$OUT/transcript.jsonl" 2>"$OUT/claude.err"
CODE=$?
echo "claude exited with $CODE after $(($(date +%s) - START)) s"

# Facts for the report (no credential is in these files).
python3 - "$OUT/transcript.jsonl" "$WORK" <<'PY'
import json, sys, pathlib
tr, work = sys.argv[1], pathlib.Path(sys.argv[2])
init = result = None
tools = []
for line in open(tr):
    try:
        ev = json.loads(line)
    except ValueError:
        continue
    if ev.get("type") == "system" and ev.get("subtype") == "init":
        init = ev
    elif ev.get("type") == "assistant":
        for c in ev.get("message", {}).get("content", []):
            if c.get("type") == "tool_use":
                tools.append(c["name"])
    elif ev.get("type") == "result":
        result = ev
print("apiKeySource:", (init or {}).get("apiKeySource"))
print("mcp servers:", (init or {}).get("mcp_servers"))
print("tool calls:", len(tools), [t for t in dict.fromkeys(tools)])
if result:
    print("result:", result.get("subtype"), "turns:", result.get("num_turns"), "is_error:", result.get("is_error"))
    print("final text:", (result.get("result") or "")[:1500])
for m in sorted(work.glob("runs/*/result/metrics.json")):
    print("run folder:", m.parent.parent)
    print("metrics:", m.read_text()[:1200])
PY
exit "$CODE"
