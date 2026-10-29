"""Tests for the problem view: the tool show_problem and its MCP Apps page, the resource ui://topopt/problem-view.html (topopt_mcp/problem_view.py)."""
import asyncio
import base64
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt_mcp import problem_view, server  # noqa: E402

HOOK = ROOT / "examples" / "hook" / "hook.yaml"

# The keys that the view reads, trimmed from a real check_problem reply of examples/hook/hook.yaml (Linux, solver auto -> pardiso).
CHECK = {
    "name": "tshirt_hook",
    "grid": {"shape": [68, 30, 55], "voxel_mm": 1.2, "origin_mm": [-1.0, -17.5, -30.0], "voxels": 112200},
    "free_dofs": 115836, "iterations": 80, "volume_fraction": 0.2,
    "solver": {"setting": "auto", "resolved": "pardiso", "choice": {"line": "solver: auto -> pardiso"}},
    "time_estimate": {"solver": "pardiso", "s_per_iteration": 1.486, "total_s": 145.39, "limit_s": None, "amg_total_s": 518.6, "stress": None},
    "memory_estimate": {"bytes": 1847290413, "ram_bytes": 68719476736},
    "warnings": ["region planes off the voxel grid: ..."],
}


@pytest.fixture
def fast_check(monkeypatch):
    import topopt.cli
    monkeypatch.setattr(topopt.cli, "check_data", lambda yaml: dict(CHECK))


def call(name, args):
    return asyncio.run(server.mcp.call_tool(name, args))


def test_the_tool_names_the_page_in_both_meta_forms():
    tool = {t.name: t for t in asyncio.run(server.mcp.list_tools())}["show_problem"]
    assert tool.meta["ui"]["resourceUri"] == problem_view.URI
    assert tool.meta["ui/resourceUri"] == problem_view.URI
    assert tool.annotations.read_only_hint is True


def test_the_page_is_an_mcp_app_resource_with_three_inside():
    listed = {str(r.uri): r for r in asyncio.run(server.mcp.list_resources())}
    assert listed[problem_view.URI].mime_type == "text/html;profile=mcp-app"
    contents = list(asyncio.run(server.mcp.read_resource(problem_view.URI)))
    html = contents[0].content
    assert contents[0].mime_type == problem_view.MIME_TYPE
    assert html.startswith("<!doctype html>")
    assert "/*__THREE__*/" not in html and "const THREE={" in html and "WebGLRenderer:" in html
    assert "ui/initialize" in html and "ui/notifications/tool-result" in html
    assert "export{" not in html[-5000:]          # the module export list became the object


def test_three_export_list_becomes_an_object():
    text = 'const a=1,b=2;function c(){}\nexport{a as Alpha,b as Beta,c};\n'
    assert problem_view._three_as_object(text).endswith("const THREE={Alpha:a,Beta:b,c:c};")
    with pytest.raises(ValueError):
        problem_view._three_as_object("const a=1;")


def test_show_problem_gives_text_facts_and_the_meshes_in_meta(fast_check):
    import trimesh

    result = call("show_problem", {"yaml_path": str(HOOK)})
    facts = result.structured_content
    assert facts["ok"] is True and facts["name"] == "tshirt_hook"
    assert [r["kind"] for r in facts["regions"]] == ["design", "fixed", "keep_out", "keep_out", "load"]
    assert facts["load_cases"] == ["hang", "pull_off"]
    assert facts["start_run_dir_rel"] == server._rel(server.default_out())          # the folder that run_topopt uses without out_dir
    assert facts["estimate"]["time_s"] == pytest.approx(145.39) and facts["estimate"]["solver"] == "pardiso"
    text = result.content[0].text
    assert "about 2 min with pardiso" in text and "2 load case(s) (hang, pull_off)" in text

    geo = result.meta[problem_view.GEOMETRY_KEY]
    assert geo["format"] == problem_view.FORMAT
    design = geo["regions"][0]
    mesh = trimesh.load(str(HOOK.parent / "design.stl"), force="mesh")
    positions = np.frombuffer(base64.b64decode(design["positions"]), dtype="<f4").reshape(-1, 3)
    indices = np.frombuffer(base64.b64decode(design["indices"]), dtype="<u4").reshape(-1, 3)
    assert np.allclose(positions, mesh.vertices) and (indices == mesh.faces).all()
    tip = next(r for r in geo["regions"] if r["kind"] == "load")
    hang = geo["load_cases"][0]
    assert hang["forces"][0]["force"] == [0.0, 0.0, -1.96] and hang["forces"][0]["center"] == tip["center"]
    # The meshes stay out of the text that the model reads.
    assert design["positions"] not in text and len(text) < 2000


def test_show_problem_refuses_a_missing_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    result = call("show_problem", {"yaml_path": "nope.yaml"})
    assert result.structured_content["ok"] is False
    assert "YAML file not found" in result.structured_content["error"]
    assert not result.meta


# ---- the run button: ui/message and the host capabilities (MCP Apps spec 2026-01-26; #376 comment 6461985990) ----
PAGE = ROOT / "topopt_mcp" / "problem_view.html"
LOGIC_START = "// ------------------------------------------------------------ run button logic"
LOGIC_END = "// ------------------------------------------------------------ end of the run button logic"


def run_logic(tmp_path, script):
    """Run the run-button block of the page in node with `script` after it; `script` prints one JSON value."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed: the run button logic is a JavaScript block of the page")
    html = PAGE.read_text(encoding="utf-8")
    js = tmp_path / "run_button.js"
    js.write_text(html[html.index(LOGIC_START):html.index(LOGIC_END)] + "\n" + script, encoding="utf-8")
    out = subprocess.run([node, str(js)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_the_host_capability_message_decides_between_the_button_and_the_line(tmp_path):
    got = run_logic(tmp_path, """
const caps = [{message: {text: {}}}, {message: {}}, {message: {image: {}}}, {message: true}, {}, null, undefined];
const f = {yaml_rel: "runs/hook.yaml"};
console.log(JSON.stringify({
  offered: caps.map(messageOffered),
  no_capability: runState(false, f), no_capability_no_part: runState(false, null),
  capability: runState(true, f), capability_no_part: runState(true, null)}));
""")
    assert got["offered"] == [True, True, False, False, False, False, False]
    assert got["no_capability"] == {"button": "hidden", "line": "Type in the chat: start the run of runs/hook.yaml"}
    assert got["no_capability_no_part"] == {"button": "hidden", "line": None}
    assert got["capability"] == {"button": "enabled", "line": None}
    assert got["capability_no_part"] == {"button": "disabled", "line": None}


def test_after_ui_message_the_page_says_what_happened_and_what_to_do(tmp_path):
    got = run_logic(tmp_path, """
const y = "runs/hook.yaml";
console.log(JSON.stringify({
  ok: afterMessage({}, null, y),
  is_error: afterMessage({isError: true}, null, y),
  is_error_text: afterMessage({isError: true, content: [{type: "text", text: "The user declined."}]}, null, y),
  denied: afterMessage(null, "Message sending denied", y)}));
""")
    line = "Type in the chat: start the run of runs/hook.yaml"
    # Success: a host may only put the text into the chat input (the spec lets it ask the user first), so the status is neutral.
    assert got["ok"] == {"status": "Passed to the chat. If the text is still in the input field, press Enter to send it.",
                         "enable": False, "line": None}
    assert got["is_error"] == {"status": "The chat did not take the message: the host gave no reason.", "enable": True, "line": line}
    assert got["is_error_text"]["status"] == "The chat did not take the message: The user declined." and got["is_error_text"]["enable"]
    # A JSON-RPC error of the host (-32000 "Message sending denied") rejects the request: the same next step.
    assert got["denied"] == {"status": "The chat did not take the message: Message sending denied", "enable": True, "line": line}


def test_the_page_wires_the_capability_and_the_answer_and_sends_the_spec_params():
    html = PAGE.read_text(encoding="utf-8")
    assert "canMessage = messageOffered(r && r.hostCapabilities);" in html     # ui/initialize result -> hostCapabilities.message
    assert 'request("ui/message", { role: "user", content: [{ type: "text", text }] })' in html   # McpUiMessageRequest params
    assert "show(afterMessage(r, null, facts.yaml_rel))" in html and "show(afterMessage(null, e.message, facts.yaml_rel, e.noAnswer))" in html
    assert '$("run").disabled = !o.enable;' in html and 'id="typeline"' in html
    assert "Sent to the chat" not in html


# ---- the direct start: tools/call through the host, the model context, the watch of the run (MCP Apps serverTools, updateModelContext) ----
# The page runs in node with a fake host. The host answers each request of the page; the test reads what the page sent and what it shows.
HOST_JS = r"""
const fs = require("fs"), vm = require("vm");
const cfg = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const html = fs.readFileSync(process.argv[2], "utf8");
const script = html.slice(html.indexOf('<script type="module">') + 22, html.lastIndexOf("</script>"));
const els = {};
function el(id) {
  if (els[id]) return els[id];
  const hidden = new RegExp('id="' + id + '"[^>]*\\shidden').test(html);   // the attribute of the markup
  return (els[id] = { id, hidden, disabled: false, textContent: "", style: { setProperty() {} }, dataset: {}, listeners: {},
    addEventListener(t, f) { this.listeners[t] = f; }, replaceChildren() {}, appendChild() {}, getBoundingClientRect: () => ({ height: 500 }),
    clientWidth: 600, clientHeight: 340, getContext: () => ({}), setPointerCapture() {} });
}
const sent = [], delays = [], guards = new Set();
let maxGuards = 0, inflight = 0;
// A fake clock: the timers of the page go into a queue and fire in time order when the driver advances the clock. No wall-clock time decides
// what the page does, so the result is the same at any load of the machine.
let now = 0;
const queue = [];
let listener = null, statusCalls = 0;
const winEvents = {};
const win = { parent: { postMessage(m) { sent.push(m); inflight++; setImmediate(() => { try { host(m); } finally { inflight--; } }); } }, innerWidth: 600, innerHeight: 400, devicePixelRatio: 1,
              addEventListener(t, f) { winEvents[t] = f; if (t === "message") listener = f; } };
const ctx = vm.createContext({ window: win, console, atob, Uint8Array, Float32Array, Uint32Array, Math, Map, Promise, Object, Array, Error, JSON,
  Number, String, isFinite, requestAnimationFrame: () => 0, ResizeObserver: class { observe() {} },
  // The poll wait is 1 ms of the fake clock; the 30 s guard fires (20 ms) only when the test asks (short_guard). `guards` holds the guards still waiting.
  clearTimeout: (h) => { guards.delete(h); const i = queue.indexOf(h); if (i >= 0) queue.splice(i, 1); },
  setTimeout: (f, ms) => {
    delays.push(ms);
    const guard = ms === 30000;
    const h = { f, due: now + (guard ? (cfg.short_guard ? 20 : Infinity) : 1) };
    if (h.due !== Infinity) queue.push(h);
    if (guard) { guards.add(h); maxGuards = Math.max(maxGuards, guards.size); }
    return h;
  },
  document: { getElementById: el, documentElement: { dataset: {}, style: { setProperty() {} }, clientWidth: 600, getBoundingClientRect: () => ({ height: 500 }) },
              body: {}, createElement: () => el("x") } });
const asText = (obj) => ({ content: [{ type: "text", text: JSON.stringify(obj) }] });   // the shape of the real server: the JSON as text only
const answer = (id, result) => listener({ source: win.parent, data: { jsonrpc: "2.0", id, result } });
const fail = (id, message) => listener({ source: win.parent, data: { jsonrpc: "2.0", id, error: { code: -32000, message } } });
function host(m) {
  if (m.id === undefined) return;
  if (m.method === "ui/initialize") return answer(m.id, { hostCapabilities: cfg.caps, hostContext: {} });
  if (m.method === "tools/call" && m.params.name === "run_topopt") {
    seenStart = true;
    if (cfg.start_error) return fail(m.id, cfg.start_error);
    if (cfg.start_hang) return;
    return answer(m.id, cfg.text_only ? asText(cfg.start.structuredContent) : cfg.start);
  }
  if (m.method === "tools/call" && m.params.name === "run_status" && !seenStart) {      // the look at the run folder before the start
    if (cfg.pre_error) return fail(m.id, cfg.pre_error);
    if (cfg.pre_is_error) return answer(m.id, { isError: true, content: [{ type: "text", text: cfg.pre_is_error }] });
    const pre = cfg.pre_seq ? cfg.pre_seq[Math.min(preCalls++, cfg.pre_seq.length - 1)] : cfg.pre || { state: "none", process_alive: false, checkpoint: null, summary: "no run found" };
    return answer(m.id, cfg.text_only ? asText(pre) : { structuredContent: pre, content: [{ type: "text", text: JSON.stringify(pre) }] });
  }
  if (m.method === "tools/call" && m.params.name === "run_status") {
    const n = statusCalls++;
    const item = cfg.statuses[Math.min(n, cfg.statuses.length - 1)];
    if (item.rpc_error) return fail(m.id, item.rpc_error);
    if (cfg.teardown_at === n) listener({ source: win.parent, data: { jsonrpc: "2.0", id: 900, method: "ui/resource-teardown", params: {} } });
    return answer(m.id, cfg.text_only ? asText(item) : { structuredContent: item, content: [{ type: "text", text: JSON.stringify(item) }] });
  }
  if (m.method === "ui/message" && cfg.message_hang) return;      // a host that holds ui/message (it asks the user first)
  answer(m.id, cfg.other_result || {});
}
vm.runInContext(script.replace(/^import .*$/mg, ""), ctx);
// Let the host answers and the promises of the page run until nothing is in flight (turns of the event loop, not milliseconds).
async function drain() {
  let calm = 0, last = -1;
  for (let i = 0; i < 1e6 && calm < 3; i++) {
    await new Promise((r) => setImmediate(r));
    if (inflight === 0 && sent.length === last) calm++; else calm = 0;
    last = sent.length;
  }
}
// Advance the fake clock by ms: fire the timers that fall due, in time order, and drain after each one.
async function wait(ms) {
  const target = now + ms;
  await drain();
  for (let n = 0; n < 1e5; n++) {
    queue.sort((a, b) => a.due - b.due);
    if (!queue.length || queue[0].due > target) break;
    const t = queue.shift();
    now = Math.max(now, t.due);
    guards.delete(t);
    t.f();
    await drain();
  }
  now = target;
}
const toolResult = { structuredContent: { ok: true, name: "hook", yaml_rel: "runs/hook.yaml", start_run_dir_rel: "runs/live", estimate: { iterations: 80, warnings: 0 } },
                     _meta: { "topopt/problem-view": { regions: [], load_cases: [] } } };
let preCalls = 0, seenStart = false, pageshowSent = null, redeliveredDisabled = null;
(async () => {
  await wait(30);
  listener({ source: win.parent, data: { jsonrpc: "2.0", method: "ui/notifications/tool-result", params: toolResult } });
  await wait(30);
  const snap = () => ({ run: { hidden: el("run").hidden, disabled: el("run").disabled }, typeline: el("typeline").hidden ? null : el("typeline").textContent });
  const before = snap();
  if (!before.run.hidden) el("run").listeners.click();   // a hidden button cannot be clicked
  await wait(cfg.settle || 300);
  const after = snap();
  let afterFirstClick = { label: el("run").textContent, runline: el("runline").hidden ? null : el("runline").textContent };
  if (cfg.second_click && !el("run").disabled) { el("run").listeners.click(); await wait(cfg.settle || 300); }
  let again = null;
  if (cfg.click_check && !el("check").hidden) { el("check").listeners.click(); await wait(300); again = true; }
  if (cfg.click_ask && !el("ask").hidden) { el("ask").listeners.click ? el("ask").listeners.click() : el("ask").onclick(); await wait(100); }
  if (cfg.pageshow) { winEvents.pagehide(); await wait(50); const calls = sent.length; winEvents.pageshow({ persisted: true }); await wait(100); pageshowSent = sent.length - calls; }
  if (cfg.redeliver) { listener({ source: win.parent, data: { jsonrpc: "2.0", method: "ui/notifications/tool-result", params: toolResult } }); await wait(50); redeliveredDisabled = el("run").disabled; }
  if (cfg.click_run_again) { el("run").listeners.click(); await wait(300); }
  if (cfg.click_discuss && !el("discuss").hidden) { el("discuss").onclick(); await wait(100); }
  console.log(JSON.stringify({
    sent: sent.filter((m) => m.method && !m.method.startsWith("ui/notifications/")), before, after, runline: el("runline").hidden ? null : el("runline").textContent,
    status: el("status").textContent, caps: el("capstext").textContent, check_hidden: el("check").hidden, discuss_hidden: el("discuss").hidden, delays, again, guards_left: guards.size, max_guards: maxGuards,
    start_line: el("startline").hidden ? null : el("startline").textContent, ask_hidden: el("ask").hidden, pageshow_sent: pageshowSent,
    redelivered_disabled: redeliveredDisabled, first: afterFirstClick, typeline_now: el("typeline").hidden ? null : el("typeline").textContent,
    discuss_disabled: el("discuss").disabled, run_now: { hidden: el("run").hidden, disabled: el("run").disabled, label: el("run").textContent } }));
  process.exit(0);
})();
"""

ALL = {"message": {"text": {}}, "serverTools": {}, "updateModelContext": {"text": {}, "structuredContent": {}}}
STARTED = {"structuredContent": {"run_dir_rel": "runs/live", "yaml_rel": "runs/hook.yaml", "pid": 4242}, "content": [{"type": "text", "text": "{}"}]}


def running(n):
    return {"state": "running", "process_alive": True, "status": {"state": "running", "iteration": n * 10, "elapsed_s": n * 20},
            "last_history": {"iteration": n * 10, "volume_fraction": 0.5 - n * 0.1}}


DONE = {"state": "done", "process_alive": True, "status": {"state": "done", "iteration": 80, "elapsed_s": 130.0},
        "last_history": {"iteration": 80, "volume_fraction": 0.2}, "job": {"state": "done", "wall_s": 131.4},
        "verdict": {"state": "converged", "facts": {"material": {"mass_g": 12.3}}}}


def drive(tmp_path, **cfg):
    """Load the page in node with a fake host, click the run button once, and return what the page sent and showed."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed: the run button is a JavaScript block of the page")
    cfg.setdefault("caps", ALL)
    cfg.setdefault("start", STARTED)
    cfg.setdefault("statuses", [running(1), running(2), DONE])
    (tmp_path / "page.html").write_text(problem_view.page(), encoding="utf-8")
    (tmp_path / "host.js").write_text(HOST_JS, encoding="utf-8")
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    out = subprocess.run([node, str(tmp_path / "host.js"), str(tmp_path / "page.html"), str(tmp_path / "cfg.json")],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def raw_methods(got):
    return [m["method"] + (":" + m["params"]["name"] if m["method"] == "tools/call" else "") for m in got["sent"]]


def sent(got):
    """The requests of the page without its look at the run folder (the run_status call before run_topopt); the tests of that look read `got["sent"]`."""
    out = list(got["sent"])
    names = [m["params"]["name"] if m["method"] == "tools/call" else None for m in out]
    if "run_topopt" in names and "run_status" in names[:names.index("run_topopt")]:
        del out[names.index("run_status")]
    return out


def methods(got):
    return [m["method"] + (":" + m["params"]["name"] if m["method"] == "tools/call" else "") for m in sent(got)]


def test_with_serverTools_the_click_calls_run_topopt_then_tells_the_model_and_watches_the_run(tmp_path):
    got = drive(tmp_path)
    assert methods(got) == ["ui/initialize", "tools/call:run_topopt", "ui/update-model-context", "tools/call:run_status", "tools/call:run_status",
                            "tools/call:run_status", "ui/update-model-context"]
    start = sent(got)[1]["params"]
    assert start == {"name": "run_topopt", "arguments": {"yaml_path": "runs/hook.yaml", "out_dir": "runs/live"}}
    first, last = sent(got)[2]["params"], sent(got)[-1]["params"]
    assert "started the run of runs/hook.yaml from the problem view" in first["content"][0]["text"] and "run_status" in first["content"][0]["text"]
    assert first["structuredContent"]["problem_view_run"] == {"event": "started", "yaml": "runs/hook.yaml", "run_dir": "runs/live", "pid": 4242,
                                                              "next": "run_status", "replaced": None, "earlier_checkpoint_note": None, "other_server_jobs": None}
    end = last["structuredContent"]["problem_view_run"]
    assert end["event"] == "ended" and end["state"] == "done" and end["verdict"] == "converged" and end["mass_g"] == 12.3
    assert end["volume_fraction"] == 0.2 and end["run_time_s"] == 131.4 and end["passes"] == 80
    assert "run time 2 min" in last["content"][0]["text"] and "mass of the part 12.3 g" in last["content"][0]["text"]
    assert all(m["params"]["arguments"] == {"run_dir": "runs/live"} for m in sent(got) if m["method"] == "tools/call" and m["params"]["name"] == "run_status")
    # The end is shown: the state and the numbers, the button to discuss. There is no ui/message.
    assert got["runline"] == "Done: pass 80 of 80 · volume 20.0 % · 2 min · verdict converged · mass of the part 12.3 g"
    assert got["discuss_hidden"] is False and "ui/message" not in methods(got)
    assert got["after"]["run"]["disabled"] is True


def test_the_watch_shows_the_pass_and_the_volume_while_the_run_goes(tmp_path):
    got = drive(tmp_path, statuses=[running(1), running(2), running(3)], settle=60)
    assert got["runline"].startswith("Running: pass ") and " of 80 · volume " in got["runline"] and got["discuss_hidden"] is True
    assert methods(got).count("tools/call:run_status") >= 2


def test_without_updateModelContext_the_page_sends_no_context(tmp_path):
    caps = {"message": {"text": {}}, "serverTools": {}}
    got = drive(tmp_path, caps=caps)
    assert "ui/update-model-context" not in methods(got) and methods(got).count("tools/call:run_status") == 3
    assert got["caps"] == "Host offers: message yes · serverTools yes · updateModelContext no"


def test_without_serverTools_the_click_sends_ui_message_as_before(tmp_path):
    got = drive(tmp_path, caps={"message": {"text": {}}, "updateModelContext": {"text": {}}})
    assert methods(got) == ["ui/initialize", "ui/message"]
    assert sent(got)[1]["params"] == {"role": "user", "content": [{"type": "text", "text": "Start the run of runs/hook.yaml (from the problem view)."}]}
    assert got["status"] == "Passed to the chat. If the text is still in the input field, press Enter to send it."
    assert got["runline"] is None


def test_with_neither_the_page_shows_the_line_to_type_and_sends_nothing(tmp_path):
    got = drive(tmp_path, caps={})
    assert methods(got) == ["ui/initialize"]
    assert got["before"]["run"]["hidden"] is True and got["before"]["typeline"] == "Type in the chat: start the run of runs/hook.yaml"
    assert got["runline"] is None and got["after"] == got["before"]


def test_a_refused_run_topopt_shows_its_text_and_starts_no_watch(tmp_path):
    lock = {"structuredContent": {"error": "A run is still running (pid 77, folder runs/live). Wait for it to finish or give override_lock=true.",
                                  "run_dir": "runs/live", "pid": 77}, "content": [{"type": "text", "text": "{}"}]}
    got = drive(tmp_path, start=lock)
    assert methods(got) == ["ui/initialize", "tools/call:run_topopt"]
    assert got["runline"] == "The run did not start: A run is still running (pid 77, folder runs/live)."      # the first sentence: the rest is for the agent
    ask_lock = drive(tmp_path, start=lock, click_ask=True)
    assert sent(ask_lock)[-1]["params"]["content"][0]["text"] == (
        "The run of runs/hook.yaml did not start (from the problem view). run_topopt said: A run is still running (pid 77, folder runs/live). "
        "Wait for it to finish or give override_lock=true.")
    assert got["after"]["run"] == {"hidden": False, "disabled": False} and got["ask_hidden"] is False   # the user may click again; the page does not retry
    ask = {"structuredContent": {"state": "awaiting_user", "options": ["Call record_brief first.", "Or run without a brief."], "override": "x"},
           "content": [{"type": "text", "text": "{}"}]}
    got = drive(tmp_path, start=ask)
    assert methods(got) == ["ui/initialize", "tools/call:run_topopt"]
    assert got["runline"] == "The run did not start: the case needs a decision first."          # short: the options are for the agent
    got = drive(tmp_path, start=ask, click_ask=True)             # "Ask the agent": the refusal and the options go in a message
    assert methods(got) == ["ui/initialize", "tools/call:run_topopt", "ui/message"]
    assert sent(got)[-1]["params"]["content"][0]["text"] == (
        "The run of runs/hook.yaml did not start (from the problem view). run_topopt said: the case needs a decision first.\n"
        "Options in its reply:\n- Call record_brief first.\n- Or run without a brief.")
    got = drive(tmp_path, start=ask, caps={"serverTools": {}})   # no message: the line to type
    assert got["ask_hidden"] is True and got["typeline_now"] == "Type in the chat: start the run of runs/hook.yaml"


def test_a_failed_tools_call_shows_the_reason_and_the_old_ways_are_the_fallback(tmp_path):
    got = drive(tmp_path, start_error="tools/call is not allowed")
    assert methods(got) == ["ui/initialize", "tools/call:run_topopt"]
    assert got["runline"] == "The host did not run the tool: tools/call is not allowed. Use the button to send a chat message instead."
    assert got["after"]["run"] == {"hidden": False, "disabled": False}
    got = drive(tmp_path, start_error="tools/call is not allowed", click_run_again=True)       # the second click goes by ui/message
    assert methods(got) == ["ui/initialize", "tools/call:run_topopt", "ui/message"]
    got = drive(tmp_path, caps={"serverTools": {}}, start={"isError": True, "content": [{"type": "text", "text": "boom"}]})
    assert got["runline"] == "The host did not run the tool: boom. Type the line in the chat instead."
    assert got["after"]["run"]["hidden"] is True and got["after"]["typeline"] == "Type in the chat: start the run of runs/hook.yaml"
    got = drive(tmp_path, start={"structuredContent": {"ok": True}, "content": []})
    assert got["runline"].startswith("The host did not run the tool: the reply names no run folder.")


@pytest.mark.parametrize("final, text", [
    (DONE, "Done: pass 80 of 80"),
    ({**DONE, "state": "error", "status": {"state": "error", "iteration": 12, "elapsed_s": 5, "message": "out of memory"}, "verdict": None, "job": None,
      "last_history": {"iteration": 12, "volume_fraction": 0.4}}, "Error: pass 12 of 80 · volume 40.0 % · 5 s · out of memory"),
    ({**DONE, "state": "stopped", "status": {"state": "stopped", "iteration": 30, "elapsed_s": 40, "message": "stop requested"}, "verdict": None, "job": None,
      "last_history": {"iteration": 30, "volume_fraction": 0.3}}, "Stopped: pass 30 of 80 · volume 30.0 % · 40 s · stop requested"),
    ({**running(2), "process_alive": False}, "The run process ended (state running): pass 20 of 80"),
])
def test_each_end_stops_the_poll_and_shows_the_end_numbers(tmp_path, final, text):
    got = drive(tmp_path, statuses=[running(1), final])
    assert methods(got).count("tools/call:run_status") == 2          # the repeat of the last answer would be a third call
    assert got["runline"].startswith(text) and got["discuss_hidden"] is False and got["check_hidden"] is True
    assert methods(got)[-1] == "ui/update-model-context"


def test_a_job_refused_over_an_earlier_run_ends_by_its_own_state(tmp_path):
    # topopt/earlier_run.py: the status is the refused job's own (error, pass 0, step setup); only the data keys moved into `earlier_run`.
    refused = {"state": "error", "process_alive": False, "status": {"state": "error", "iteration": 0, "step": "setup", "elapsed_s": 1.0,
                                                                   "message": "bad input. The earlier run in this folder is kept."},
               "last_history": None, "verdict": None, "checkpoint": None, "earlier_run": {"summary": "earlier"}}
    got = drive(tmp_path, statuses=[refused, refused])
    assert methods(got).count("tools/call:run_status") == 1 and got["runline"].startswith("Error: ") and "volume" not in got["runline"]
    still = {**refused, "process_alive": True}
    got = drive(tmp_path, statuses=[still, still])               # the state alone ends it, as for any error
    assert methods(got).count("tools/call:run_status") == 1


def test_a_poll_error_is_shown_and_the_poll_does_not_loop(tmp_path):
    got = drive(tmp_path, statuses=[running(1), {"rpc_error": "Server not reachable"}])
    assert methods(got).count("tools/call:run_status") == 2
    assert got["runline"] == "Could not read the state of the run: Server not reachable." and got["check_hidden"] is False
    got = drive(tmp_path, statuses=[running(1), {"rpc_error": "Server not reachable"}, DONE], click_check=True)    # "Check again": one more call
    assert methods(got).count("tools/call:run_status") == 3 and got["runline"].startswith("Done: ") and got["check_hidden"] is True


def test_a_teardown_of_the_view_stops_the_poll(tmp_path):
    got = drive(tmp_path, statuses=[running(1), running(2), running(3), running(4)], teardown_at=1)
    assert methods(got).count("tools/call:run_status") == 2


def test_discuss_the_result_sends_ui_message_only_when_the_host_takes_messages(tmp_path):
    got = drive(tmp_path, click_discuss=True)
    assert methods(got)[-1] == "ui/message"
    assert sent(got)[-1]["params"]["content"][0]["text"] == ("Let us discuss the result of the run of runs/hook.yaml (runs/live). It ended: done, verdict converged.")
    got = drive(tmp_path, caps={"serverTools": {}})
    assert got["discuss_hidden"] is True and "ui/message" not in methods(got)


def test_the_poll_has_one_interval_the_named_constant(tmp_path):
    html = PAGE.read_text(encoding="utf-8")
    assert html.count("const RUN_STATUS_POLL_MS = 5000;") == 1 and html.index("RUN_STATUS_POLL_MS") < html.index("function request(")
    assert html.count("const HOST_ANSWER_WAIT_MS = 30000;") == 1 and html.index("HOST_ANSWER_WAIT_MS") < html.index("function request(")
    view = html[html.index("(function () {"):]                           # the view script, without three.js
    timers = [line for line in view.splitlines() if re.search(r"\bset(Timeout|Interval)\(", line)]
    assert len(timers) == 2 and all(("RUN_STATUS_POLL_MS" in line) != ("HOST_ANSWER_WAIT_MS" in line) for line in timers)   # each timer uses one named constant
    assert not re.search(r"\b30000\b|\b5000\b", view.replace("const RUN_STATUS_POLL_MS = 5000;", "").replace("const HOST_ANSWER_WAIT_MS = 30000;", ""))
    poll = html[html.index("  function poll() {"):html.index("  function pollError(")]
    assert re.findall(r"setTimeout\([^)]*\)", poll) == ["setTimeout(poll, RUN_STATUS_POLL_MS)"]
    assert not re.search(r"\d{2,}", poll)                                  # no other number in the poll code
    assert [d for d in drive(tmp_path)["delays"] if d != 30000] == [5000, 5000]     # the waits of the page: two polls go on, the third ends the run


def test_an_answered_request_leaves_no_guard_timer_and_the_poll_does_not_pile_them_up(tmp_path):
    got = drive(tmp_path)                                      # initialize, run_status (the look), run_topopt, 3 x run_status, 2 x update-model-context: all answered
    assert got["delays"].count(30000) == 8 and got["guards_left"] == 0 and got["max_guards"] <= 2    # the look at the run folder is one of the eight
    long_run = drive(tmp_path, statuses=[running(i) for i in range(1, 9)] + [DONE])      # nine polls: still no guard is left
    assert long_run["delays"].count(30000) >= 12 and long_run["guards_left"] == 0 and long_run["max_guards"] <= 2
    failed = drive(tmp_path, start_error="not allowed")        # a rejected request clears its guard too
    assert failed["guards_left"] == 0


@pytest.mark.parametrize("caps, line", [
    ({"message": {"text": {}}, "serverTools": {}, "updateModelContext": {"text": {}}}, "Host offers: message yes · serverTools yes · updateModelContext yes"),
    ({"message": {}}, "Host offers: message yes · serverTools no · updateModelContext no"),
    ({"serverTools": {"listChanged": True}}, "Host offers: message no · serverTools yes · updateModelContext no"),
    ({"updateModelContext": {"image": {}}}, "Host offers: message no · serverTools no · updateModelContext no"),
    ({}, "Host offers: message no · serverTools no · updateModelContext no"),
    (None, "The host sent no capabilities."),
])
def test_the_capability_line_and_the_route_for_each_combination(tmp_path, caps, line):
    got = run_logic(tmp_path, "const c = %s;\nconsole.log(JSON.stringify({line: capabilityLine(c), route: startRoute(c, false), failed: startRoute(c, true)}));" % json.dumps(caps))
    assert got["line"] == line
    has_tools = bool(caps) and "serverTools" in caps
    has_message = bool(caps) and caps.get("message") is not None
    assert got["route"] == ("tools" if has_tools else "message" if has_message else "line")
    assert got["failed"] == ("message" if has_message else "line")


def test_the_status_view_and_the_replies_of_run_topopt_in_pure_form(tmp_path):
    got = run_logic(tmp_path, """
console.log(JSON.stringify({
  starting: statusView({state: "starting", process_alive: true}, 80),
  post: statusView({state: "postprocess", process_alive: true, status: {iteration: 80, elapsed_s: 200}, last_history: {volume_fraction: 0.25}}, 80),
  gone: statusView({state: "none", process_alive: false, status: null}, 80).ended,
  not_json: toolReply({content: [{type: "text", text: "plain words"}]}),
  from_text: toolReply({content: [{type: "text", text: "{\\"run_dir\\": \\"/w/runs/a\\"}"}]}).kind,
  info: startInfo({run_dir: "/w/runs/a", earlier_checkpoint: {note: "n"}}, {yaml_rel: "p.yaml"})}));
""")
    assert got["starting"] == {"text": "Starting…", "ended": False, "end": None}
    assert got["post"]["text"] == "Post-processing: pass 80 of 80 · volume 25.0 % · 3 min" and got["post"]["ended"] is False
    assert got["gone"] is True
    assert got["not_json"]["kind"] == "failed" and got["from_text"] == "ok"
    assert got["info"] == {"run_dir": "/w/runs/a", "yaml": "p.yaml", "pid": None, "note": "n", "others": None, "replaced": None}


def test_the_start_facts_stay_on_the_screen_and_go_to_the_model(tmp_path):
    reply = {"run_dir_rel": "runs/live", "yaml_rel": "runs/hook.yaml", "pid": 4242,
             "earlier_checkpoint": {"iteration": 40, "note": "The run folder has a checkpoint at pass 40 of 80. This run deletes it."},
             "other_server_jobs": {"fact": "A job that another topopt server started is running now: pid 9.", "jobs": []}}
    got = drive(tmp_path, start={"structuredContent": reply})
    assert got["start_line"] == ("Run started in runs/live.\nThe run folder has a checkpoint at pass 40 of 80. This run deletes it.\n"
                                 "A job that another topopt server started is running now: pid 9.")
    assert got["runline"].startswith("Done: ")                    # the poll has replaced its own line only
    context = sent(got)[2]["params"]
    assert "This run deletes it." in context["content"][0]["text"] and "another topopt server started" in context["content"][0]["text"]
    assert context["structuredContent"]["problem_view_run"]["earlier_checkpoint_note"].endswith("deletes it.")


def test_a_start_with_no_answer_says_the_run_may_have_started_and_offers_check_again(tmp_path):
    got = drive(tmp_path, start_hang=True, short_guard=True, settle=200, click_check=True, statuses=[running(1), DONE])
    assert methods(got) == ["ui/initialize", "tools/call:run_topopt", "tools/call:run_status", "tools/call:run_status", "ui/update-model-context"]
    assert sent(got)[2]["params"] == {"name": "run_status", "arguments": {"run_dir": "runs/live"}}      # the folder of the facts, not the last run
    assert got["runline"].startswith("Done: ") and got["after"]["run"]["disabled"] is True      # the route did not change to the message button
    got = drive(tmp_path, start_hang=True, short_guard=True, settle=200)
    assert got["runline"] == "No answer from the host: the run may have started." and got["check_hidden"] is False
    assert got["after"]["run"]["disabled"] is True and "ui/message" not in methods(got)


def test_structuredContent_goes_in_the_context_only_when_the_host_lists_it(tmp_path):
    only_text = drive(tmp_path, caps={"serverTools": {}, "updateModelContext": {"text": {}}})
    updates = [m["params"] for m in sent(only_text) if m["method"] == "ui/update-model-context"]
    assert len(updates) == 2 and all("structuredContent" not in u for u in updates)
    for caps in ({"text": {}, "structuredContent": {}}, {}):
        both = drive(tmp_path, caps={"serverTools": {}, "updateModelContext": caps})
        updates = [m["params"] for m in sent(both) if m["method"] == "ui/update-model-context"]
        assert len(updates) == 2 and all("structuredContent" in u for u in updates)


def test_the_real_server_shape_json_as_text_only_works_end_to_end(tmp_path):
    got = drive(tmp_path, text_only=True)
    assert methods(got).count("tools/call:run_status") == 3 and got["runline"].startswith("Done: pass 80 of 80")
    assert got["start_line"] == "Run started in runs/live."


def test_the_start_button_stays_off_while_a_run_is_watched(tmp_path):
    got = drive(tmp_path, statuses=[running(1)] * 40, settle=60, redeliver=True)
    assert got["redelivered_disabled"] is True
    done = drive(tmp_path, redeliver=True)                       # the same after the end
    assert done["redelivered_disabled"] is True
    refused = drive(tmp_path, start={"structuredContent": {"error": "busy"}}, redeliver=True)    # a refusal gives the button back
    assert refused["redelivered_disabled"] is False


def test_a_page_back_from_the_cache_offers_check_again_and_does_not_poll(tmp_path):
    got = drive(tmp_path, pageshow=True, statuses=[running(1)] * 40, settle=60)           # the run goes on
    assert got["pageshow_sent"] == 0 and got["check_hidden"] is False
    assert got["runline"] == "The view came back. Press Check again to see the run."
    ended = drive(tmp_path, pageshow=True)                                                 # the run had ended: the end view stays
    assert ended["pageshow_sent"] == 0 and ended["check_hidden"] is True and ended["discuss_hidden"] is False
    assert ended["runline"].startswith("Done: pass 80 of 80") and methods(ended).count("ui/update-model-context") == 2   # the start note and one end note


PRE_FOLDER = "runs/live"


def old_run(**kw):
    """A run_status reply of a folder that holds a finished run with a checkpoint (the keys of the real reply that the page reads)."""
    return {"state": "done", "process_alive": False, "status": {"state": "done", "iteration": 80, "elapsed_s": 130.0}, "summary": "done: iteration 80/80",
            "checkpoint": {"iteration": 80, "iterations": 80, "complete": True, "yaml_path": "/work/runs/old/part.yaml"}, **kw}


def test_an_empty_run_folder_starts_at_once_and_both_calls_name_the_folder_of_the_facts(tmp_path):
    got = drive(tmp_path)
    calls = [m["params"] for m in got["sent"] if m["method"] == "tools/call"]
    assert calls[0] == {"name": "run_status", "arguments": {"run_dir": PRE_FOLDER}}                   # the look: the folder of the facts, not the last run
    assert calls[1]["name"] == "run_topopt" and calls[1]["arguments"]["out_dir"] == PRE_FOLDER      # the start: the same folder
    assert raw_methods(got)[:3] == ["ui/initialize", "tools/call:run_status", "tools/call:run_topopt"]


def test_a_folder_with_a_run_shows_what_the_start_replaces_and_the_second_click_starts(tmp_path):
    got = drive(tmp_path, pre=old_run())
    assert "tools/call:run_topopt" not in raw_methods(got)
    assert got["runline"] == ("The run folder runs/live holds a run (done, pass 80 of 80, the run of /work/runs/old/part.yaml). "
                              "The start deletes its result and its checkpoints. Press \"Replace and start\" to go on.")
    assert got["first"]["label"] == "Replace and start" and got["after"]["run"] == {"hidden": False, "disabled": False}
    got = drive(tmp_path, pre=old_run(), second_click=True)
    assert raw_methods(got).count("tools/call:run_topopt") == 1 and raw_methods(got).index("tools/call:run_topopt") == 3          # init, look, look again, start
    assert got["start_line"].startswith("Run started in runs/live.\nReplaced the run in runs/live (done, pass 80 of 80, the run of /work/runs/old/part.yaml).")
    assert "Replaced the run in runs/live" in sent(got)[3]["params"]["content"][0]["text"]               # the model is told what was replaced
    assert sent(got)[3]["params"]["structuredContent"]["problem_view_run"]["replaced"].startswith("Replaced the run in runs/live")


def test_the_warning_has_no_yaml_words_when_the_checkpoint_has_no_yaml_path(tmp_path):
    pre = old_run(checkpoint={"iteration": 40, "iterations": 80, "complete": False, "yaml_path": None})
    got = drive(tmp_path, pre=pre)
    assert got["runline"].startswith("The run folder runs/live holds a run (done, pass 40 of 80). The start deletes")
    assert "the run of" not in got["runline"]
    no_checkpoint = old_run(checkpoint=None, state="error", status={"state": "error", "iteration": 3})
    assert drive(tmp_path, pre=no_checkpoint)["runline"].startswith("The run folder runs/live holds a run (error, pass 3). ")


def test_a_run_that_works_in_the_folder_gets_no_start(tmp_path):
    working = {"state": "running", "process_alive": True, "summary": "iteration 12/80, 40 s", "checkpoint": None,
               "status": {"state": "running", "iteration": 12}}
    got = drive(tmp_path, pre=working, second_click=True)
    assert "tools/call:run_topopt" not in raw_methods(got) and got["after"]["run"]["disabled"] is True and got["check_hidden"] is False
    assert got["runline"] == "A run is working in runs/live: iteration 12/80, 40 s."
    # A done run whose process only serves the dashboard is not a working run: run_topopt stops that process and goes on.
    serving = old_run(process_alive=True)
    assert drive(tmp_path, pre=serving)["runline"].startswith("The run folder runs/live holds a run (done")


def test_a_failed_look_at_the_run_folder_starts_nothing_and_shows_the_fallback(tmp_path):
    for cfg in ({"pre_error": "tools/call is not allowed"}, {"pre_is_error": "boom"}):
        got = drive(tmp_path, **cfg)
        assert "tools/call:run_topopt" not in raw_methods(got)
        assert got["runline"].startswith("Could not look at the run folder: ") and got["runline"].endswith("Use the button to send a chat message instead.")
        assert got["after"]["run"] == {"hidden": False, "disabled": False}
    got = drive(tmp_path, caps={"serverTools": {}}, pre_error="no")
    assert got["runline"].endswith("Type the line in the chat instead.") and got["after"]["typeline"] == "Type in the chat: start the run of runs/hook.yaml"


def with_job(reply, started_at, pid):
    return {**reply, "job": {"state": reply.get("state"), "started_at": started_at, "pid": pid}}


def test_a_stopped_run_with_a_checkpoint_gets_the_warning_and_no_start(tmp_path):
    stopped = old_run(state="stopped", status={"state": "stopped", "iteration": 40}, checkpoint={"iteration": 40, "iterations": 80, "complete": False,
                                                                                               "yaml_path": "/work/runs/old/part.yaml"})
    got = drive(tmp_path, pre=stopped)
    assert "tools/call:run_topopt" not in raw_methods(got)
    assert got["runline"] == ("The run folder runs/live holds a run (stopped, pass 40 of 80, the run of /work/runs/old/part.yaml). "
                              "The start deletes its checkpoints. Press \"Replace and start\" to go on.")      # a stopped run has no result


def test_a_job_that_holds_the_lock_gets_no_start_even_without_a_process_of_this_server(tmp_path):
    cli = old_run(state="running", process_alive=False, job={"state": "running", "pid": 777, "started_at": "t"}, summary="iteration 12/80",
                  status={"state": "running", "iteration": 12})
    got = drive(tmp_path, pre=cli, second_click=True)
    assert "tools/call:run_topopt" not in raw_methods(got) and got["after"]["run"]["disabled"] is True and got["check_hidden"] is False
    assert got["runline"] == "A run is working in runs/live: iteration 12/80."
    ended_job = old_run(job={"state": "stopped", "pid": 777, "started_at": "t"})           # any other job state is replaceable
    assert drive(tmp_path, pre=ended_job)["runline"].startswith("The run folder runs/live holds a run")


def test_the_warning_names_only_what_the_folder_holds(tmp_path):
    no_files = old_run(state="error", checkpoint=None, status={"state": "error", "iteration": 0})
    assert drive(tmp_path, pre=no_files)["runline"].endswith("(error, pass 0). The start deletes its files. Press \"Replace and start\" to go on.")
    no_checkpoint = old_run(checkpoint=None)
    assert "deletes its result. Press" in drive(tmp_path, pre=no_checkpoint)["runline"]


def test_check_again_after_a_held_start_does_not_take_the_old_run_for_this_one(tmp_path):
    old = with_job(old_run(), "2026-10-28T10:00:00", 5)
    got = drive(tmp_path, pre=old, second_click=True, start_hang=True, short_guard=True, settle=200, click_check=True, statuses=[old])
    assert got["runline"] == "No run has started in runs/live yet. You can start it again." and got["check_hidden"] is False
    assert got["run_now"] == {"hidden": False, "disabled": False, "label": "Start the run"}   # the held call did not run: the start is free again
    assert methods(got).count("ui/update-model-context") == 0 and got["discuss_hidden"] is True
    assert sent(got)[-1]["params"] == {"name": "run_status", "arguments": {"run_dir": "runs/live"}}
    none = drive(tmp_path, start_hang=True, short_guard=True, settle=200, click_check=True,
                 statuses=[{"state": "none", "process_alive": False, "checkpoint": None, "summary": "no run found"}])
    assert none["runline"] == "No run has started in runs/live yet. You can start it again." and none["discuss_hidden"] is True
    assert none["run_now"]["disabled"] is False
    # A different job is the new run: the page watches it and ends with its own end.
    new = with_job(running(1), "2026-10-28T11:00:00", 6)
    finished = with_job(DONE, "2026-10-28T11:00:00", 6)
    got = drive(tmp_path, pre=old, second_click=True, start_hang=True, short_guard=True, settle=200, click_check=True, statuses=[new, finished])
    assert got["runline"].startswith("Done: pass 80 of 80") and got["discuss_hidden"] is False
    assert methods(got).count("ui/update-model-context") == 1


def test_the_second_click_looks_again_and_a_changed_folder_gets_a_new_warning(tmp_path):
    first = with_job(old_run(), "t1", 5)
    other = with_job(old_run(checkpoint={"iteration": 10, "iterations": 80, "complete": False, "yaml_path": None}), "t2", 9)
    got = drive(tmp_path, pre_seq=[first, other], second_click=True)
    assert "tools/call:run_topopt" not in raw_methods(got)
    assert got["runline"].startswith("The run folder changed since the last warning. The run folder runs/live holds a run (done, pass 10 of 80). ")
    assert got["after"]["run"]["disabled"] is False
    same = drive(tmp_path, pre_seq=[first, first], second_click=True)                       # the same job: the start goes on
    assert raw_methods(same).count("tools/call:run_topopt") == 1


def test_a_run_with_a_fine_pass_to_come_is_working_and_gets_no_start(tmp_path):
    # The real reply of run_status for the first folder of run_topopt(fine_pass=true): the first job is done and its process goes on to the fine folder.
    waiting = old_run(process_alive=True, job={"state": "done", "pid": 1111, "started_at": "t1", "wall_s": 130.0}, summary="done: iteration 80/80",
                      fine_run={"run_dir_rel": "runs/live-fine", "state": "not started"})
    for state in ("not started", "starting", "running", "postprocess"):
        got = drive(tmp_path, pre={**waiting, "fine_run": {**waiting["fine_run"], "state": state}}, second_click=True)
        assert "tools/call:run_topopt" not in raw_methods(got) and got["runline"].startswith("A run is working in runs/live: ")
        assert got["after"]["run"]["disabled"] is True and got["check_hidden"] is False
    ended = {**waiting, "fine_run": {**waiting["fine_run"], "state": "done"}}             # the fine pass has ended: replaceable
    assert drive(tmp_path, pre=ended)["runline"].startswith("The run folder runs/live holds a run (done")
    not_alive = {**waiting, "process_alive": False}                                       # no process: nothing goes on
    assert drive(tmp_path, pre=not_alive)["runline"].startswith("The run folder runs/live holds a run (done")


def test_a_new_process_with_the_old_job_record_is_not_the_old_run(tmp_path):
    old = with_job(old_run(), "t1", 5)
    starting = {"state": "starting", "process_alive": True, "checkpoint": None, "summary": "starting, no data yet",
                "job": {"state": "done", "pid": 5, "started_at": "t1", "earlier_run": True}}       # the old job record, the new process
    got = drive(tmp_path, pre=old, second_click=True, start_hang=True, short_guard=True, settle=200, click_check=True, statuses=[starting])
    assert got["runline"].startswith("Starting") and "No run has started" not in got["runline"] and got["check_hidden"] is True
    only_state = {**starting, "job": {"state": "done", "pid": 5, "started_at": "t1"}}              # state starting with a live process is enough
    got = drive(tmp_path, pre=old, second_click=True, start_hang=True, short_guard=True, settle=200, click_check=True, statuses=[only_state])
    assert got["runline"].startswith("Starting")


def test_a_new_process_that_died_before_its_job_is_shown_with_its_reason(tmp_path):
    died = {"state": "error", "process_alive": False, "checkpoint": None, "status": None, "last_history": None,
            "job": {"state": "no_record", "exit_code": 1}, "summary": "error: the run process ended with exit 1 before it wrote any run data"}
    got = drive(tmp_path, pre=old_run(), second_click=True, start_hang=True, short_guard=True, settle=200, click_check=True, statuses=[died])
    assert "No run has started" not in got["runline"]
    assert got["runline"] == "Error: error: the run process ended with exit 1 before it wrote any run data"
    note = [m for m in sent(got) if m["method"] == "ui/update-model-context"][-1]["params"]["content"][0]["text"]
    assert "reason: error: the run process ended with exit 1 before it wrote any run data" in note
    # A status message wins over the summary.
    with_message = {**died, "status": {"state": "error", "iteration": 3, "message": "out of memory"}}
    assert drive(tmp_path, statuses=[with_message])["runline"].startswith("Error: pass 3 of 80") and "out of memory" in drive(tmp_path, statuses=[with_message])["runline"]


def test_the_held_check_in_pure_form(tmp_path):
    got = run_logic(tmp_path, """
const old = {empty: false, key: "t1|5"};
const job = (extra) => ({state: "done", pid: 5, started_at: "t1", ...extra});
console.log(JSON.stringify({
  same: heldCheck({state: "done", job: job({})}, old),
  none: heldCheck({state: "none", job: null}, old),
  other_job: heldCheck({state: "done", job: {state: "done", pid: 6, started_at: "t2"}}, old),
  earlier_run: heldCheck({state: "done", job: job({earlier_run: true})}, old),
  no_record: heldCheck({state: "error", job: {state: "no_record", exit_code: 1}}, {empty: false, key: null}),
  starting: heldCheck({state: "starting", process_alive: true, job: job({})}, old),
  starting_dead: heldCheck({state: "starting", process_alive: false, job: job({})}, old),
  no_job_both: heldCheck({state: "done", job: null}, {empty: false, key: null}),
  empty_then_run: heldCheck({state: "running", job: null}, {empty: true, key: null})}));
""")
    assert got == {"same": False, "none": False, "other_job": True, "earlier_run": True, "no_record": True, "starting": True,
                   "starting_dead": False, "no_job_both": False, "empty_then_run": True}


def test_a_held_start_that_never_ran_gives_the_start_back_and_a_new_click_looks_first(tmp_path):
    old = with_job(old_run(), "2026-10-28T10:00:00", 5)
    got = drive(tmp_path, pre=old, second_click=True, start_hang=True, short_guard=True, settle=200, click_check=True, statuses=[old],
                click_run_again=True)
    # The click after "No run has started" looks at the folder again (run_status before any new run_topopt), as the first click does.
    names = [m["params"]["name"] for m in got["sent"] if m["method"] == "tools/call"]
    last_start = len(names) - 1 - names[::-1].index("run_topopt")
    assert names[last_start + 1:] == ["run_status", "run_status"] and names.count("run_topopt") == 1   # "Check again", then the look of the new click
    assert got["runline"].startswith("The run folder runs/live holds a run (done")


def test_the_look_takes_a_live_job_lock_as_a_working_run_before_an_empty_folder(tmp_path):
    got = run_logic(tmp_path, """
console.log(JSON.stringify({
  locked: replaceCheck({state: "none", checkpoint: null, job: {state: "running", pid: 77}, summary: "no run found"}, "runs/live").action,
  empty: replaceCheck({state: "none", checkpoint: null, job: null}, "runs/live").action,
  ended_job: replaceCheck({state: "none", checkpoint: null, job: {state: "done"}}, "runs/live").action}));
""")
    assert got == {"locked": "alive", "empty": "start", "ended_job": "start"}
    locked = {"state": "none", "process_alive": False, "checkpoint": None, "job": {"state": "running", "pid": 77}, "summary": "no run found"}
    got = drive(tmp_path, pre=locked)
    assert "tools/call:run_topopt" not in raw_methods(got) and got["runline"].startswith("A run is working in runs/live")


def test_a_held_ui_message_may_have_been_sent_and_the_page_says_so(tmp_path):
    got = drive(tmp_path, caps={"message": {"text": {}}}, message_hang=True, short_guard=True, settle=200)
    assert methods(got) == ["ui/initialize", "ui/message"]
    assert got["status"] == "No answer from the host: the message may have been sent. Look in the chat before you press the button again."
    assert got["run_now"]["disabled"] is False
