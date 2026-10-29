"""The hand-off from the status server of the launcher to this server (#370).

On a first start, `scripts/topopt-env.sh mcp` answers the client itself (a small MCP server in sh) while the Python setup runs. When
the setup is ready, the shell replaces itself with `python -m topopt_mcp.server` (exec), and the open stdin goes with it. The shell has
already read some lines from that stdin. It writes them to a file and names the file in the variable TOPOPT_MCP_HANDOFF:

- modern era (protocol 2026-07-28): the one request that arrived after the setup was ready;
- legacy era: the `initialize` request, `notifications/initialized` and that request. The client has the answer of the shell to its
  `initialize` already, so TOPOPT_MCP_HANDOFF_DROP_ID names the JSON id of that request and this server drops its answer.

This module gives the SDK a stdin that yields those lines first, then the real stdin, and a stdout that drops that one answer.
Without the variable nothing here runs, and `main()` starts the server as before.

The SDK protects the protocol stream: `stdio_server()` without arguments moves fd 0 to /dev/null and fd 1 to stderr, and serves the
wire from private copies, so that a stray `print()` or a child process cannot write into the protocol. A call with its own streams skips
that, so `claim_stdio` does the same here.
"""
import io
import json
import os
import sys
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

HANDOFF_ENV = "TOPOPT_MCP_HANDOFF"
DROP_ID_ENV = "TOPOPT_MCP_HANDOFF_DROP_ID"
ERA_ENV = "TOPOPT_MCP_HANDOFF_ERA"            # modern | legacy: the era that the shell decided
LISTEN_ID_ENV = "TOPOPT_MCP_HANDOFF_LISTEN_ID"  # the JSON id of the listen request that the shell acknowledged (modern era)
SNAPSHOT_DIR = Path(__file__).resolve().parent / "launcher"
SUBSCRIPTION_KEY = "io.modelcontextprotocol/subscriptionId"


class Handoff:
    """The lines that the shell already read, and the id of the answer to drop (None: drop none; the id 0 is a real id)."""

    def __init__(self, lines, drop_id=None, has_drop_id=False, era="", listen_id=None, has_listen_id=False):
        self.lines = list(lines)
        self.drop_id = drop_id
        self.has_drop_id = has_drop_id
        self.era = era
        self.listen_id = listen_id
        self.has_listen_id = has_listen_id


def read_handoff(environ=None):
    """Take the hand-off from the environment, and delete the file. Both variables are removed from the environment, so that the
    child processes of this server (the solver, the ratchet) do not see them. None if there is no hand-off."""
    environ = os.environ if environ is None else environ
    path = environ.pop(HANDOFF_ENV, "")
    raw_id = environ.pop(DROP_ID_ENV, "")
    era = environ.pop(ERA_ENV, "")
    raw_listen = environ.pop(LISTEN_ID_ENV, "")
    if not path:
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as exc:
        print(f"topopt: the hand-off file cannot be read ({type(exc).__name__}: {exc}); the server starts with no replayed line.", file=sys.stderr)
        text = ""
    try:
        os.unlink(path)
    except OSError:
        pass
    lines = [line for line in text.split("\n") if line.strip()]
    drop_id, has_drop_id = None, False
    if raw_id:
        try:
            drop_id, has_drop_id = json.loads(raw_id), True
        except ValueError:
            pass
    listen_id, has_listen_id = None, False
    if raw_listen:
        try:
            listen_id, has_listen_id = json.loads(raw_listen), True
        except ValueError:
            pass
    return Handoff(lines, drop_id, has_drop_id, era, listen_id, has_listen_id)


class ReplayStdin:
    """An async iterator of lines: the replayed lines first, then the lines of `inner` (an async iterable of lines)."""

    def __init__(self, lines, inner):
        self.lines, self.inner = lines, inner

    def __aiter__(self):
        return self._lines()

    async def _lines(self):
        for line in self.lines:
            yield line + "\n"
        async for line in self.inner:
            yield line


def _same_id(a, b):
    # 0 == False == 0.0 in Python: an id is equal only with the same JSON type.
    return type(a) is type(b) and a == b


class DropAnswer:
    """The write side of the SDK without the first answer (a result or an error) to the request with the id `drop_id`."""

    def __init__(self, inner, drop_id=None, has_drop_id=False):
        self.inner = inner
        self.waiting = has_drop_id
        self.drop_id = drop_id

    async def write(self, text):
        if self.waiting:
            try:
                message = json.loads(text)
            except ValueError:
                message = None
            if (isinstance(message, dict) and "method" not in message and ("result" in message or "error" in message)
                    and "id" in message and _same_id(message["id"], self.drop_id)):
                self.waiting = False
                return
        await self.inner.write(text)

    async def flush(self):
        await self.inner.flush()


@contextmanager
def claim_stdio():
    """Claim fd 0 and fd 1 as `mcp.server.stdio.stdio_server()` does. Yield the private binary files of the wire (stdin, stdout).
    While the block runs, fd 0 is /dev/null and fd 1 is a copy of stderr (/dev/null if there is no stderr); the old fds come back at the end."""
    import fcntl
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    private_in = fcntl.fcntl(0, fcntl.F_DUPFD_CLOEXEC, 3)
    try:
        private_out = fcntl.fcntl(1, fcntl.F_DUPFD_CLOEXEC, 3)
    except OSError:
        os.close(private_in)
        raise
    try:
        null_in = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_in, 0)
        os.close(null_in)
        try:
            diversion = os.dup(2)
        except OSError:
            diversion = os.open(os.devnull, os.O_WRONLY)
        os.dup2(diversion, 1)
        os.close(diversion)
        # closefd=False: a worker thread can still block on a descriptor after the server ends.
        yield os.fdopen(private_in, "rb", closefd=False), os.fdopen(private_out, "wb", closefd=False)
    finally:
        os.dup2(private_out, 1)
        os.dup2(private_in, 0)


def _plain(value):
    """A JSON value without the keys whose value is None (the SDK leaves an unset field out of the wire answer)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _dump(items):
    return [_plain(item.model_dump(mode="json", by_alias=True, exclude_none=True)) for item in items]


async def live_lists(server):
    """The lists of the live server, as the snapshot holds them: {"tools": [...], "prompts": [...], "resources": [...], "templates": [...],
    "instructions": text}."""
    return {"tools": _dump(await server.list_tools()), "prompts": _dump(await server.list_prompts()),
            "resources": _dump(await server.list_resources()), "templates": _dump(await server.list_resource_templates()),
            "instructions": server.instructions}


def snapshot_lists(era):
    """The same, from the files that the launcher served (topopt_mcp/launcher/, made by `python -P -m topopt_mcp.snapshot`)."""
    first = "initialize" if era == "legacy" else "discover"
    out = {"instructions": json.loads((SNAPSHOT_DIR / f"{era}.{first}.json").read_text(encoding="utf-8")).get("instructions")}
    keys = {"tools": "tools", "prompts": "prompts", "resources": "resources", "templates": "resourceTemplates"}
    for name, key in keys.items():
        out[name] = _plain(json.loads((SNAPSHOT_DIR / f"{era}.{name}.json").read_text(encoding="utf-8"))[key])
    return out


def changed_lines(handoff, live, snapshot):
    """The list_changed notifications (raw JSON lines) that tell the client the lists are not those of the snapshot. Empty when they are the
    same, or when the client has no place to get them (modern: no listen request was acknowledged; legacy: no initialize was answered)."""
    if live == snapshot:
        return []
    if handoff.era == "modern":
        if not handoff.has_listen_id:
            return []
        params = {"_meta": {SUBSCRIPTION_KEY: handoff.listen_id}}
    elif handoff.era == "legacy":
        if not handoff.has_drop_id:
            return []
        params = None
    else:
        return []
    lines = []
    for kind in ("tools", "prompts", "resources"):
        msg = {"jsonrpc": "2.0", "method": f"notifications/{kind}/list_changed"}
        if params is not None:
            msg["params"] = params
        lines.append(json.dumps(msg, separators=(",", ":")))
    return lines


async def announce_changes(server, handoff, wire_out):
    """The safety net. The launcher answered the client from a snapshot. If the live lists differ, tell the client before the SDK starts
    to answer. Nothing is written when they are the same."""
    if handoff.era not in ("modern", "legacy"):
        return
    try:
        live, snapshot = await live_lists(server), snapshot_lists(handoff.era)
    except Exception as exc:  # a missing snapshot file must not stop the server
        print(f"topopt: the launcher snapshot cannot be compared with the live lists ({type(exc).__name__}: {exc}).", file=sys.stderr)
        return
    lines = changed_lines(handoff, live, snapshot)
    if live != snapshot:
        differing = [k for k in live if live[k] != snapshot[k]]
        print(f"topopt: the lists differ from the launcher snapshot ({', '.join(differing)}); run python -P -m topopt_mcp.snapshot. "
              f"{'Sending list_changed.' if lines else 'The client has no place for list_changed.'}", file=sys.stderr)
    if lines:
        wire_out.write("".join(line + "\n" for line in lines).encode("utf-8"))
        wire_out.flush()


def serve(server, handoff):
    """Run `server` (an MCPServer) over stdio as `server.run("stdio")` does, with the streams of the hand-off.

    `server.run("stdio")` calls `stdio_server()` of mcp.server.mcpserver.server (a name in that module). It is replaced for the time
    of the run by a function that makes the streams of the hand-off and calls the real `stdio_server` with them."""
    import anyio
    import mcp.server.mcpserver.server as sdk_server

    real = sdk_server.stdio_server

    @asynccontextmanager
    async def handoff_stdio_server():
        with claim_stdio() as (wire_in, wire_out):
            await announce_changes(server, handoff, wire_out)
            stdin = anyio.wrap_file(io.TextIOWrapper(wire_in, encoding="utf-8", errors="replace"))
            stdout = anyio.wrap_file(io.TextIOWrapper(wire_out, encoding="utf-8"))
            async with real(stdin=ReplayStdin(handoff.lines, stdin),
                            stdout=DropAnswer(stdout, handoff.drop_id, handoff.has_drop_id)) as streams:
                yield streams

    sdk_server.stdio_server = handoff_stdio_server
    try:
        server.run("stdio")
    finally:
        sdk_server.stdio_server = real

