"""Tests for topopt_mcp/handoff.py: the Python side of the hand-off from the status server of the launcher (#370).

The launcher (scripts/topopt-env.sh) answers the client while the Python setup runs. When the setup is ready it replaces itself with this
server, and the open stdin goes with it. The lines that the shell already read are in a file (TOPOPT_MCP_HANDOFF). The tests run the real
server (the real SDK) in a child process over pipes, as the launcher does.

The CI runs this file on macOS too (tests/tiers/platform_files.txt): the hand-off reads the pipe of the platform.
"""
import asyncio
import inspect
import json
import os
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from topopt_mcp import handoff, server  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"},
        "io.modelcontextprotocol/clientCapabilities": {}}


def modern(method, ident, **params):
    return json.dumps({"jsonrpc": "2.0", "id": ident, "method": method, "params": dict(params, _meta=META)}, separators=(",", ":"))


def legacy(method, ident=None, **params):
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    if ident is not None:
        msg["id"] = ident
    return json.dumps(msg, separators=(",", ":"))


INITIALIZE = legacy("initialize", 0, protocolVersion="2025-11-25", capabilities={}, clientInfo={"name": "t", "version": "1"})
INITIALIZED = legacy("notifications/initialized")


# --- read_handoff ---------------------------------------------------------------------------------------------------------------

def test_read_handoff_takes_the_lines_deletes_the_file_and_removes_the_variables(tmp_path):
    f = tmp_path / "h"
    f.write_text("one\n\ntwo\n")
    env = {handoff.HANDOFF_ENV: str(f), handoff.DROP_ID_ENV: '"abc"', "OTHER": "x"}
    h = handoff.read_handoff(env)
    assert h.lines == ["one", "two"] and h.drop_id == "abc" and h.has_drop_id
    assert not f.exists() and env == {"OTHER": "x"}                 # the child processes of the server do not see the hand-off


def test_read_handoff_without_the_variable_is_none_and_the_id_zero_is_an_id(tmp_path):
    assert handoff.read_handoff({}) is None
    f = tmp_path / "h"
    f.write_text("x\n")
    h = handoff.read_handoff({handoff.HANDOFF_ENV: str(f), handoff.DROP_ID_ENV: "0"})
    assert h.drop_id == 0 and h.has_drop_id
    f.write_text("x\n")
    h = handoff.read_handoff({handoff.HANDOFF_ENV: str(f)})
    assert not h.has_drop_id
    f.write_text("x\n")
    assert not handoff.read_handoff({handoff.HANDOFF_ENV: str(f), handoff.DROP_ID_ENV: "{bad"}).has_drop_id


def test_read_handoff_with_a_missing_file_starts_with_no_replayed_line(tmp_path, capsys):
    h = handoff.read_handoff({handoff.HANDOFF_ENV: str(tmp_path / "gone")})
    assert h.lines == [] and "cannot be read" in capsys.readouterr().err


# --- the streams -----------------------------------------------------------------------------------------------------------------

class Lines:
    def __init__(self, lines):
        self.lines = lines

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for line in self.lines:
            yield line


class Sink:
    def __init__(self):
        self.written = []

    async def write(self, text):
        self.written.append(text)

    async def flush(self):
        pass


def test_the_replayed_lines_come_before_the_real_stdin():
    async def go():
        return [x async for x in handoff.ReplayStdin(["a", "b"], Lines(["c\n", "d\n"]))]
    assert asyncio.run(go()) == ["a\n", "b\n", "c\n", "d\n"]


@pytest.mark.parametrize("drop, answer, dropped", [
    (0, '{"jsonrpc":"2.0","id":0,"result":{}}\n', True),
    ("abc", '{"jsonrpc":"2.0","id":"abc","result":{}}\n', True),
    ("é", '{"jsonrpc":"2.0","id":"\\u00e9","result":{}}\n', True),                          # the id is compared as JSON, not as text
    (0, '{"jsonrpc":"2.0","id":0,"error":{"code":-1,"message":"x"}}\n', True),             # an error answer to it is dropped too
    (0, '{"jsonrpc":"2.0","id":1,"result":{}}\n', False),
    (0, '{"jsonrpc":"2.0","id":false,"result":{}}\n', False),                                # 0 is not false
    (0, '{"jsonrpc":"2.0","id":0.0,"result":{}}\n', False),                                  # and not 0.0
    (0, '{"jsonrpc":"2.0","id":"0","result":{}}\n', False),
    (0, '{"jsonrpc":"2.0","id":0,"method":"ping"}\n', False),                                # a request of the server with that id is not an answer
    (0, '{"jsonrpc":"2.0","method":"notifications/message","params":{"id":0}}\n', False),
    (0, 'not json\n', False),
])
def test_the_answer_to_the_replayed_initialize_is_dropped_once_and_only_that_one(drop, answer, dropped):
    async def go():
        sink = Sink()
        out = handoff.DropAnswer(sink, drop, True)
        await out.write(answer)
        await out.write(answer)                                                              # the second one is kept
        return sink.written
    assert asyncio.run(go()) == ([answer] if dropped else [answer, answer])


def test_without_a_drop_id_nothing_is_dropped():
    async def go():
        sink = Sink()
        out = handoff.DropAnswer(sink)
        await out.write('{"jsonrpc":"2.0","id":0,"result":{}}\n')
        return sink.written
    assert len(asyncio.run(go())) == 1


# --- the real server over pipes --------------------------------------------------------------------------------------------------

# Runs main() of the real server. NOISY adds a tool that writes to stdout in three ways (print, a write to fd 1, a child process).
SERVER_CODE = textwrap.dedent('''
    import os, subprocess, sys
    from topopt_mcp import server
    server.TOOL_SURFACE["noisy"] = server.ToolSurface("Noisy", read_only=True, destructive=False, idempotent=True, open_world=False)
    @server.mcp.tool()
    def noisy() -> str:
        print("STRAY-PRINT")
        sys.stdout.flush()
        os.write(1, b"STRAY-FD\\n")
        subprocess.run(["echo", "STRAY-CHILD"])
        return "noisy done"
    server.main()
''')


def run_server(tmp_path, handoff_lines=(), drop_id=None, stdin_lines=(), expect=1, code=SERVER_CODE, **extra_env):
    """Start the server in a child process, as the launcher does after the hand-off. It sends the lines of `stdin_lines`, waits for
    `expect` answers (a server that sees the end of stdin first stops before it answers), then closes stdin. Returns (result, hand-off file)."""
    hf = None
    env = dict(os.environ, PYTHONPATH=str(REPO), TOPOPT_WORKSPACE=str(tmp_path / "ws"), HOME=str(tmp_path))
    env.pop(handoff.HANDOFF_ENV, None)
    env.pop(handoff.DROP_ID_ENV, None)
    if handoff_lines is not None:
        hf = tmp_path / "handoff.txt"
        hf.write_text("".join(x + "\n" for x in handoff_lines))
        env[handoff.HANDOFF_ENV] = str(hf)
        if drop_id is not None:
            env[handoff.DROP_ID_ENV] = drop_id
    env.update(extra_env)
    p = subprocess.Popen([sys.executable, "-P", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=env, cwd=tmp_path)
    out = []
    try:
        for line in stdin_lines:
            p.stdin.write(line + "\n")
            p.stdin.flush()
        timer = threading.Timer(90, p.kill)
        timer.start()
        while len(out) < expect:
            line = p.stdout.readline()
            if not line:
                break
            out.append(line)
        timer.cancel()
        p.stdin.close()
        errors = []
        reader = threading.Thread(target=lambda: errors.append(p.stderr.read()))
        reader.start()
        rest = p.stdout.read()
        p.wait(timeout=60)
        reader.join()
        err = errors[0]
    finally:
        if p.poll() is None:
            p.kill()
    return SimpleNamespace(returncode=p.returncode, stdout="".join(out) + rest, stderr=err), hf


def answers_of(r):
    return [json.loads(x) for x in r.stdout.splitlines()]


def test_modern_the_replayed_request_is_answered_first_and_then_the_next_line_of_stdin(tmp_path):
    r, hf = run_server(tmp_path, [modern("tools/list", 3)], stdin_lines=[modern("prompts/list", 4)], expect=2)
    assert r.returncode == 0, r.stderr
    answers = answers_of(r)
    assert [a["id"] for a in answers] == [3, 4]
    assert len(answers[0]["result"]["tools"]) > 10 and "setup_status" not in [t["name"] for t in answers[0]["result"]["tools"]]
    assert answers[1]["result"]["prompts"]
    assert not hf.exists()                                                                   # the server deleted the file


def test_legacy_the_answer_to_the_replayed_initialize_is_dropped_and_the_rest_is_answered(tmp_path):
    r, hf = run_server(tmp_path, [INITIALIZE, INITIALIZED, legacy("tools/list", 1)], drop_id="0", stdin_lines=[legacy("prompts/list", 2)], expect=2)
    assert r.returncode == 0, r.stderr
    answers = answers_of(r)
    assert [a["id"] for a in answers] == [1, 2]                                               # no answer with id 0: the client has the answer of the shell
    assert "tools" in answers[0]["result"] and "prompts" in answers[1]["result"]


def test_legacy_without_a_drop_id_the_initialize_is_answered(tmp_path):
    """The setup was ready at the first line: the shell did not answer `initialize`, so Python must."""
    r, _ = run_server(tmp_path, [INITIALIZE], stdin_lines=[INITIALIZED, legacy("tools/list", 1)], expect=2)
    answers = answers_of(r)
    assert [a["id"] for a in answers] == [0, 1] and answers[0]["result"]["protocolVersion"] == "2025-11-25"


def test_legacy_a_string_id_of_the_initialize_is_dropped(tmp_path):
    init = legacy("initialize", "abc-1", protocolVersion="2025-06-18", capabilities={}, clientInfo={"name": "t", "version": "1"})
    r, _ = run_server(tmp_path, [init, INITIALIZED, legacy("tools/list", 1)], drop_id='"abc-1"', expect=1)
    assert [a["id"] for a in answers_of(r)] == [1]


def test_a_stray_print_a_write_to_fd_1_and_a_child_process_do_not_reach_the_protocol_stream(tmp_path):
    call = modern("tools/call", 5, name="noisy", arguments={})
    r, _ = run_server(tmp_path, [modern("tools/list", 3)], stdin_lines=[call], expect=2)
    assert r.returncode == 0, r.stderr
    for line in r.stdout.splitlines():
        json.loads(line)                                                                      # every line of stdout is JSON
    answers = answers_of(r)
    assert [a["id"] for a in answers] == [3, 5] and answers[1]["result"]["content"][0]["text"] == "noisy done"
    for stray in ("STRAY-PRINT", "STRAY-FD", "STRAY-CHILD"):
        assert stray not in r.stdout and stray in r.stderr                                     # they went to stderr


def test_the_same_stray_output_is_kept_off_the_stream_without_a_hand_off_too(tmp_path):
    """The start with no hand-off variable is the start of today (mcp.run), which has the same protection from the SDK."""
    r, _ = run_server(tmp_path, None, stdin_lines=[modern("tools/call", 5, name="noisy", arguments={})], expect=1)
    assert [a["id"] for a in answers_of(r)] == [5] and "STRAY" not in r.stdout


def test_the_fds_are_given_back_after_the_run():
    """claim_stdio: fd 0 reads /dev/null and fd 1 writes to stderr inside the block; both are the old ones after it."""
    code = textwrap.dedent('''
        import os, sys
        from topopt_mcp import handoff
        with handoff.claim_stdio() as (wire_in, wire_out):
            print("inside")
            sys.stdout.flush()
            inside_stdin = os.read(0, 10)
            wire_out.write(b"wire\\n")
            wire_out.flush()
            first = wire_in.readline()
        print("after")
        sys.stdout.flush()
        sys.stderr.write(repr((inside_stdin, first)) + "\\n")
    ''')
    r = subprocess.run([sys.executable, "-P", "-c", code], input="line1\nline2\n", capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(REPO)),
                       timeout=60)
    assert r.stdout == "wire\nafter\n"                                                         # the wire, then the restored stdout
    assert "inside" in r.stderr and "(b'', b'line1\\n')" in r.stderr


def test_main_without_the_hand_off_runs_the_server_as_before(monkeypatch):
    calls = []
    monkeypatch.delenv(handoff.HANDOFF_ENV, raising=False)
    monkeypatch.setattr(server.mcp, "run", lambda transport: calls.append(("run", transport)))
    monkeypatch.setattr(server, "serve_handoff", lambda *a: calls.append(("handoff",)))
    monkeypatch.setenv("TOPOPT_STARTED_BY", "test")    # main() sets it with setdefault: keep the value out of the other tests of this process
    monkeypatch.setattr(server, "_register_server", lambda folder: None)
    server.main()
    assert calls == [("run", "stdio")]


def test_main_with_the_hand_off_serves_with_the_streams_of_the_hand_off(monkeypatch, tmp_path):
    f = tmp_path / "h"
    f.write_text("x\n")
    monkeypatch.setenv(handoff.HANDOFF_ENV, str(f))
    calls = []
    monkeypatch.setattr(server.mcp, "run", lambda transport: calls.append(("run", transport)))
    monkeypatch.setattr(server, "serve_handoff", lambda srv, h: calls.append(("handoff", srv is server.mcp, h.lines)))
    monkeypatch.setenv("TOPOPT_STARTED_BY", "test")    # main() sets it with setdefault: keep the value out of the other tests of this process
    monkeypatch.setattr(server, "_register_server", lambda folder: None)
    server.main()
    assert calls == [("handoff", True, ["x"])]
    assert handoff.HANDOFF_ENV not in os.environ


def test_the_name_in_the_sdk_that_the_hand_off_replaces_is_still_there():
    """`serve` replaces `stdio_server` in mcp.server.mcpserver.server for the time of the run, and `run_stdio_async` of the pinned SDK (mcp 2.3.0)
    calls it by that name. If a new SDK moves it, this test fails, and the hand-off needs a new hook. (The tests above fail too: the replayed
    lines would not come first.)"""
    import mcp.server.mcpserver.server as sdk_server
    assert callable(sdk_server.stdio_server)
    source = inspect.getsource(sdk_server.MCPServer.run_stdio_async)
    assert "stdio_server()" in source
    assert "stdin" in inspect.signature(sdk_server.stdio_server.__wrapped__).parameters if hasattr(sdk_server.stdio_server, "__wrapped__") else True


# --- the safety net: the launcher served a snapshot; if the live lists differ, the client is told -----------------------------------

import shutil  # noqa: E402

SNAPSHOT_CODE = textwrap.dedent('''
    import os
    from pathlib import Path
    from topopt_mcp import handoff, server
    handoff.SNAPSHOT_DIR = Path(os.environ["TEST_SNAPSHOT_DIR"])
    server.main()
''')
CHANGED = ["notifications/tools/list_changed", "notifications/prompts/list_changed", "notifications/resources/list_changed"]


def tampered_snapshot(tmp_path, name=None):
    folder = tmp_path / "snap"
    shutil.copytree(handoff.SNAPSHOT_DIR, folder)
    if name:
        path = folder / f"{name}.json"
        data = json.loads(path.read_text())
        key = next(k for k in data if isinstance(data[k], list))
        data[key] = data[key][1:]                                   # the live server has one more entry than this snapshot
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    return folder


def test_changed_lines_are_empty_when_the_lists_are_the_same_or_the_client_has_no_place_for_them():
    live = {"tools": [1]}
    same = handoff.Handoff([], era="modern", listen_id="l", has_listen_id=True)
    assert handoff.changed_lines(same, live, {"tools": [1]}) == []
    assert handoff.changed_lines(handoff.Handoff([], era="modern"), live, {"tools": []}) == []                          # no listen request
    assert handoff.changed_lines(handoff.Handoff([], era="legacy"), live, {"tools": []}) == []                          # no initialize answered
    assert handoff.changed_lines(handoff.Handoff([], era=""), live, {"tools": []}) == []


@pytest.mark.parametrize("listen_id", ["listen:0", 41])
def test_changed_lines_name_the_listen_id_in_the_modern_era_and_nothing_in_the_legacy_era(listen_id):
    modern_lines = handoff.changed_lines(handoff.Handoff([], era="modern", listen_id=listen_id, has_listen_id=True), {"a": 1}, {"a": 2})
    assert [json.loads(x) for x in modern_lines] == [
        {"jsonrpc": "2.0", "method": m, "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": listen_id}}} for m in CHANGED]
    legacy_lines = handoff.changed_lines(handoff.Handoff([], drop_id=0, has_drop_id=True, era="legacy"), {"a": 1}, {"a": 2})
    assert [json.loads(x) for x in legacy_lines] == [{"jsonrpc": "2.0", "method": m} for m in CHANGED]


@pytest.mark.parametrize("era", ["modern", "legacy"])
def test_the_committed_snapshot_equals_the_live_lists(era):
    """The safety net stays silent on a current snapshot. (tests/test_launcher_snapshot.py checks the files against the real server.)"""
    live = asyncio.run(handoff.live_lists(server.mcp))
    assert live == handoff.snapshot_lists(era)


def test_modern_a_snapshot_that_differs_gives_the_list_changed_lines_with_the_listen_id_first(tmp_path):
    folder = tampered_snapshot(tmp_path, "modern.tools")
    r, _ = run_server(tmp_path, [modern("tools/list", 3)], expect=4, code=SNAPSHOT_CODE, TEST_SNAPSHOT_DIR=str(folder),
                      **{handoff.ERA_ENV: "modern", handoff.LISTEN_ID_ENV: '"listen:0"'})
    assert r.returncode == 0, r.stderr
    lines = [json.loads(x) for x in r.stdout.splitlines()]
    assert [x.get("method") for x in lines[:3]] == CHANGED
    assert all(x["params"] == {"_meta": {"io.modelcontextprotocol/subscriptionId": "listen:0"}} for x in lines[:3])
    assert lines[3]["id"] == 3 and len(lines[3]["result"]["tools"]) > 20                    # the replayed request is still answered
    assert "differ from the launcher snapshot (tools)" in r.stderr


def test_legacy_a_snapshot_that_differs_gives_plain_list_changed_lines_and_the_replay_still_works(tmp_path):
    folder = tampered_snapshot(tmp_path, "legacy.resources")
    r, _ = run_server(tmp_path, [INITIALIZE, INITIALIZED, legacy("tools/list", 1)], drop_id="0", expect=4, code=SNAPSHOT_CODE,
                      TEST_SNAPSHOT_DIR=str(folder), **{handoff.ERA_ENV: "legacy"})
    lines = [json.loads(x) for x in r.stdout.splitlines()]
    assert [x.get("method") for x in lines[:3]] == CHANGED and all("params" not in x for x in lines[:3])
    assert lines[3]["id"] == 1 and "tools" in lines[3]["result"]                            # and no answer with the id 0
    assert "differ from the launcher snapshot (resources)" in r.stderr


@pytest.mark.parametrize("era, handoff_lines, extra", [
    ("modern", [modern("tools/list", 3)], {handoff.LISTEN_ID_ENV: '"listen:0"'}),
    ("legacy", [INITIALIZE, INITIALIZED, legacy("tools/list", 1)], {}),
])
def test_a_snapshot_that_is_the_same_gives_no_extra_line(tmp_path, era, handoff_lines, extra):
    r, _ = run_server(tmp_path, handoff_lines, drop_id="0" if era == "legacy" else None, expect=1, code=SNAPSHOT_CODE,
                      TEST_SNAPSHOT_DIR=str(handoff.SNAPSHOT_DIR), **{handoff.ERA_ENV: era}, **extra)
    lines = [json.loads(x) for x in r.stdout.splitlines()]
    assert len(lines) == 1 and "result" in lines[0]                                          # the answer to the replayed request, nothing before it
    assert "differ from" not in r.stderr


def test_a_missing_snapshot_file_does_not_stop_the_server(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    r, _ = run_server(tmp_path, [modern("tools/list", 3)], expect=1, code=SNAPSHOT_CODE, TEST_SNAPSHOT_DIR=str(empty),
                      **{handoff.ERA_ENV: "modern", handoff.LISTEN_ID_ENV: '"listen:0"'})
    assert [json.loads(x).get("id") for x in r.stdout.splitlines()] == [3] and "cannot be compared" in r.stderr


def test_main_without_the_hand_off_runs_no_comparison(monkeypatch):
    called = []
    monkeypatch.delenv(handoff.HANDOFF_ENV, raising=False)
    monkeypatch.setattr(handoff, "announce_changes", lambda *a: called.append(a))
    monkeypatch.setattr(server.mcp, "run", lambda transport: None)
    monkeypatch.setenv("TOPOPT_STARTED_BY", "test")    # main() sets it with setdefault: keep the value out of the other tests of this process
    monkeypatch.setattr(server, "_register_server", lambda folder: None)
    server.main()
    assert called == []
