"""Make the snapshot that the launcher serves while the Python setup is not ready (#370).

    python -P -m topopt_mcp.snapshot [--out DIR]

`scripts/topopt-env.sh mcp` answers the client itself, in sh, until the environment is ready. It must list the real tools, prompts and
resources from its first answer, because many clients do not ask for the lists again after a `list_changed` notification. So it
serves the answers of this server, recorded here: this command starts the REAL server over stdio, once for each protocol era, asks it
the list requests, and writes the `result` of each answer to one file, as one line of JSON.

Files (default folder: topopt_mcp/launcher/):
    modern.discover.json  modern.tools.json  modern.prompts.json  modern.resources.json  modern.templates.json
    legacy.initialize.json  legacy.tools.json  legacy.prompts.json  legacy.resources.json  legacy.templates.json

The modern files are the results as the server sends them (resultType, ttlMs, cacheScope and _meta included). The legacy `initialize`
result holds the text @PROTOCOL_VERSION@ in place of the version: the version depends on the request, and the shell puts it in
(CONTRACTS.md, section "The status server of the launcher"). One file for each answer, not one for each version: the rest of the
result is the same for every version.

The files are deterministic: the server runs with an empty work folder and an empty home, and nothing in a list holds a time, a path
or a random id. A tool change needs a new snapshot; `tests/test_launcher_snapshot.py` fails until the files are made again.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

LAUNCHER_DIR = Path(__file__).resolve().parent / "launcher"
PLACEHOLDER = "@PROTOCOL_VERSION@"
LEGACY_VERSION = "2025-11-25"
MODERN_META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
               "io.modelcontextprotocol/clientInfo": {"name": "topopt-snapshot", "version": "1"},
               "io.modelcontextprotocol/clientCapabilities": {}}
LISTS = (("tools", "tools/list"), ("prompts", "prompts/list"), ("resources", "resources/list"), ("templates", "resources/templates/list"))
FILE_NAMES = ("modern.discover", "legacy.initialize", *(f"{era}.{name}" for era in ("modern", "legacy") for name, _ in LISTS))


def _line(msg):
    return json.dumps(msg, separators=(",", ":")) + "\n"


def _ask(requests, notifications=()):
    """Start the real server, send the notifications and then the requests (id, method, params), and return {id: result}."""
    with tempfile.TemporaryDirectory(prefix="topopt-snapshot-") as tmp:
        env = dict(os.environ, TOPOPT_WORKSPACE=tmp, TOPOPT_STATE_DIR=os.path.join(tmp, "state"), HOME=tmp, LANG="C", TZ="UTC")
        env.pop("TOPOPT_MCP_HANDOFF", None)
        proc = subprocess.Popen([sys.executable, "-P", "-m", "topopt_mcp.server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, env=env, cwd=tmp)
        results, errors = {}, []
        try:
            for note in notifications:
                proc.stdin.write(_line(note))
            for ident, method, params in requests:
                proc.stdin.write(_line({"jsonrpc": "2.0", "id": ident, "method": method, "params": params}))
            proc.stdin.flush()
            timer = threading.Timer(120, proc.kill)
            timer.start()
            while len(results) + len(errors) < len(requests):
                line = proc.stdout.readline()
                if not line:
                    break
                msg = json.loads(line)
                if "result" in msg:
                    results[msg["id"]] = msg["result"]
                elif "error" in msg:
                    errors.append(msg)
            timer.cancel()
        finally:
            proc.stdin.close()
            proc.stdout.close()
            proc.wait(timeout=60)
    if errors or len(results) != len(requests):
        raise SystemExit(f"the server did not answer every request: {len(results)} of {len(requests)}, errors {errors}")
    return results


def make_snapshot():
    """Return {file name without .json: text of the file} for the current server."""
    out = {}
    modern = [(0, "server/discover", {"_meta": MODERN_META})] + [(n, method, {"_meta": MODERN_META}) for n, (_, method) in enumerate(LISTS, 1)]
    answers = _ask(modern)
    out["modern.discover"] = answers[0]
    for n, (name, _) in enumerate(LISTS, 1):
        out[f"modern.{name}"] = answers[n]
    init_params = {"protocolVersion": LEGACY_VERSION, "capabilities": {}, "clientInfo": {"name": "topopt-snapshot", "version": "1"}}
    legacy = [(0, "initialize", init_params)] + [(n, method, {}) for n, (_, method) in enumerate(LISTS, 1)]
    answers = _ask(legacy, notifications=[{"jsonrpc": "2.0", "method": "notifications/initialized"}])
    init = dict(answers[0])
    assert init["protocolVersion"] == LEGACY_VERSION
    init["protocolVersion"] = PLACEHOLDER
    out["legacy.initialize"] = init
    for n, (name, _) in enumerate(LISTS, 1):
        out[f"legacy.{name}"] = answers[n]
    return {name: json.dumps(value, separators=(",", ":")) + "\n" for name, value in out.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -P -m topopt_mcp.snapshot", description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=LAUNCHER_DIR, help="the folder for the files (default: topopt_mcp/launcher/)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in make_snapshot().items():
        (args.out / f"{name}.json").write_text(text, encoding="utf-8")
    print(f"snapshot written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
