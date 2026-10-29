"""MCP server for topopt (CONTRACTS.md section 10). Stdio transport.

Tools: run_topopt, run_status, run_result, stop_run, list_runs, run_history, compare_runs, marks_to_hints, apply_marks,
check_problem, build_regions, environment, render_part, verify_run, verify_status, print_check, export_step, record_brief,
case_status, ratchet_run, ratchet_status, ratchet_approve, cleanup_list, team_guide, save_preferences, find_printer,
save_printer, find_material, save_material, printer_materials, find_features, show_problem.
Stdout is the MCP channel. Log to stderr only.
The role playbooks are served by the tool team_guide, as MCP prompts and as MCP resources (CONTRACTS.md section 25).
The list of tools for each client is in docs/clients.md.

Env var TOPOPT_PORT sets the first dashboard port to try (default 8765).
If another process holds it (for example an old dashboard), the next free
port is used, and run_topopt returns that URL.

Folders: the work folder is the env var TOPOPT_WORKSPACE if it is set and usable (a leading ~ and the variables
${HOME}, ${DOCUMENTS}, ${DESKTOP} and ${DOWNLOADS} are replaced), else TOPOPT_DEFAULT_WORKSPACE (the Desktop extension
sets it), else the current directory (topopt/workfolder.py has the rule). Relative YAML paths and the default run folder
(<work folder>/runs/live) start there. A relative YAML path that is not in the work
folder is also looked up in the code folder (the examples of a checkout or of the plugin).
Every dict reply has `work_folder_note` while the work folder setting could not be used or the folder cannot be written;
environment, list_runs and run_topopt also say what the server did (a variable replaced, runs in an old place).
A tool that starts a child process makes the work folder just before the start. run_topopt and build_regions refuse only when
they cannot create or write the folder that they write (the run folder, or runs/).
The child process runs in the Python of this server (sys.executable); the env var
TOPOPT_PYTHON names another one.

The registry: each server writes one small file in a folder of the user (topopt/registry.py, CONTRACTS.md section 43). `environment` lists the
other servers with the job state of their run folders (`other_servers`; a server that has ended is listed while its run goes on), and `run_topopt`
says when another server has a running job.

Test hook: set TOPOPT_MCP_CMD to a command template. The words {yaml},
{out} and {port} are replaced by the YAML path, the run folder and the port.
"""
from __future__ import annotations

import atexit
import datetime
import functools
import inspect
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

from mcp.server.mcpserver import Image, MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations

from topopt import registry, workfolder
from topopt_mcp.handoff import read_handoff, serve as serve_handoff

REPO = Path(__file__).resolve().parent.parent  # code folder (a checkout, or site-packages' parent)


def workspace() -> Path:
    """Work folder: topopt.workfolder.resolve() (TOPOPT_WORKSPACE, else TOPOPT_DEFAULT_WORKSPACE, else the current directory).
    Read at each call. It creates nothing."""
    return workfolder.resolve(check=False).path


def default_out() -> Path:
    return workspace() / "runs" / "live"


def python_exe() -> str:
    """Python for the child process: TOPOPT_PYTHON, else the Python of this server."""
    return os.environ.get("TOPOPT_PYTHON") or sys.executable


def _port() -> int:
    """Dashboard port: env var TOPOPT_PORT, default 8765. Read at each call."""
    try:
        return int(os.environ.get("TOPOPT_PORT", "8765"))
    except ValueError:
        return 8765


def _free_port(start: int, tries: int = 20) -> int:
    """Return the first port from start that nothing listens on (127.0.0.1)."""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as http.server
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    return start  # all busy: the run prints the problem in its log


BUSY_STATES = ("starting", "running", "postprocess")

# Limit: 2048 characters in all (OLD_CLIENT_CUT of tests/test_plugin_manifest.py; Claude Code before v2.1.296 cuts the instructions there, #380).
# A rule for one tool lives in the description of that tool, or in the start-here skill; the last line only names those tools.
INSTRUCTIONS = (
    "3D topology optimization: YAML + STL regions in, printable STL out. Units: mm, N, MPa, g.\n"
    "A problem: design space (where material may go), fixed regions (where the part is held), loads (a region with a force in N; one or more cases) and keep-out regions, if any (no material).\n"
    "Core tools: environment, find_features, build_regions, record_brief, check_problem (own or edited STLs and YAML), render_part, run_topopt, run_status, run_result, print_check. One job at a time. run_topopt returns at once; stop_run ends it after a pass.\n"
    "Offer three ways for a request with no numbers: all the questions; the 2 or 3 that change the part most; a first design from stated guesses. Every way keeps the load question (in the third, ask the user to confirm the guessed load before the run); \"just run it\" skips the offer, not the load.\n"
    "State assumptions before the run: forces, mounting, size, volume fraction, voxel size (record_brief saves them). If an answer changes the part (load, mounting, size), ask before the run, also when the user says \"run it\". Small details: choose a sensible value, say so.\n"
    "Before the first run call environment; tell the user the device, the time estimate (check_problem, then run_status) and the choice of another solver or device. Name the work folder in the first reply. Use the user's language, plain words first, numbers on request.\n"
    "Tools give facts, estimates and options. The voxel guards (settings.max_grid_voxels, max_active_voxels) are method settings: a stop names its override. Decide with check_problem.memory_estimate against the RAM (above it build_regions stops; the machine can freeze); report what, why and the price.\n"
    "After a run say what was checked and what was not (stress, fatigue, the printed part), the files, and what to ask next (`levers` of run_result). For iteration_limit say \"the run stopped at its pass limit\", not \"not converged\".\n"
    "Descriptions say when to use: case_status, run_history, compare_runs, marks_to_hints, apply_marks, verify_run, ratchet_run, cleanup_list, team_guide, list_runs."
)

STARTED_UTC = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")   # when this server process started (once, at import)

NOTE_KEY = "work_folder_note"
INFO_NOTE_TOOLS = ("environment", "list_runs", "run_topopt")   # these replies also say what the server did with the work folder setting


def _add_work_folder_note(reply: dict, tool: str) -> None:
    """Add `work_folder_note` to a dict reply: the `fix_needed` texts of the work folder (every tool), and the `notes` (INFO_NOTE_TOOLS)."""
    text = workfolder.resolve().note(info=tool in INFO_NOTE_TOOLS, skip=str(reply.get("error") or ""))
    if text and NOTE_KEY not in reply:
        reply[NOTE_KEY] = text


def _with_work_folder_note(fn):
    """The tool `fn` with the work folder note in each dict reply. One place for all tools; the schema and the description stay."""
    if inspect.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        reply = fn(*args, **kwargs)
        if isinstance(reply, dict):
            _add_work_folder_note(reply, fn.__name__)
        return reply

    return wrapper


class ToolSurface(NamedTuple):
    """The title and the four MCP hints of one tool. `_TopoptServer.tool` gives them to the SDK (CONTRACTS.md section 10)."""
    title: str
    read_only: bool
    destructive: bool
    idempotent: bool
    open_world: bool


# The title and the four hints of each tool: the one table. A tool line `@mcp.tool(...)` does not pass `title` or `annotations`.
# A tool with no row stops the import (`_TopoptServer.tool`), and tests/test_mcp_tool_surface.py fails for a row with no tool. A hint is a claim
# about the whole tool, over all its arguments; when two values fit, the table has the safe one (not read-only, destructive). The rules:
# - read_only: the tool creates, changes and deletes no file, folder, process or lock. A cache or log file counts as a write.
# - destructive (only for a tool that is not read-only): the tool can delete or overwrite a file or folder that it did not create in the same
#   call, or stop a process. A tool that overwrites only with `overwrite=true` still counts. A tool that only adds new files does not.
#   The lock file of a job that has ended holds no data: a tool that removes it is not read-only, but it is not destructive for that.
#   A tool that only writes again its own derived output (a file that topopt makes from the facts of the folder, such as the part card
#   result/report.md) is not read-only, but it is not destructive for that: the facts stay, and the file comes back the same from them.
# - idempotent: a second call with the same arguments has no more effect on the environment than the first.
# - open_world: the tool can reach the network or a system outside the work folder and this machine. A local file or process does not.
TOOL_SURFACE: dict[str, ToolSurface] = {
    "run_topopt": ToolSurface("Start an optimization run", read_only=False, destructive=True, idempotent=False, open_world=False),
    "run_status": ToolSurface("Show run status", read_only=True, destructive=False, idempotent=True, open_world=False),
    # run_result writes the part card result/report.md again when its bytes change (P9-7a): not read-only. The card is derived output, so
    # it is not destructive (the rule above); a second call writes nothing new: idempotent.
    "run_result": ToolSurface("Get the run result", read_only=False, destructive=False, idempotent=True, open_world=False),
    "stop_run": ToolSurface("Ask a run to stop", read_only=False, destructive=True, idempotent=True, open_world=False),
    "render_part": ToolSurface("Render a part to images", read_only=False, destructive=True, idempotent=True, open_world=False),
    "list_runs": ToolSurface("List runs", read_only=True, destructive=False, idempotent=True, open_world=False),
    "run_history": ToolSurface("Show run history", read_only=True, destructive=False, idempotent=True, open_world=False),
    "compare_runs": ToolSurface("Compare runs", read_only=True, destructive=False, idempotent=True, open_world=False),
    "marks_to_hints": ToolSurface("Turn marks into hints", read_only=False, destructive=True, idempotent=False, open_world=False),
    "apply_marks": ToolSurface("Edit the part at marks", read_only=False, destructive=True, idempotent=False, open_world=False),
    "check_problem": ToolSurface("Check a problem YAML", read_only=True, destructive=False, idempotent=True, open_world=False),
    "verify_run": ToolSurface("Verify stress of a run", read_only=False, destructive=True, idempotent=False, open_world=False),
    # verify_status and ratchet_status remove the lock file of a job that has ended (`_alive_verify`, `_alive_ratchet`): not read-only,
    # not destructive (the lock holds no data; a client that asks before a destructive tool must not ask at each status poll).
    "verify_status": ToolSurface("Show verify status", read_only=False, destructive=False, idempotent=True, open_world=False),
    "ratchet_run": ToolSurface("Judge candidate edits", read_only=False, destructive=True, idempotent=False, open_world=False),
    "ratchet_status": ToolSurface("Show ratchet status", read_only=False, destructive=False, idempotent=True, open_world=False),
    "ratchet_approve": ToolSurface("Decide a waiting candidate", read_only=False, destructive=True, idempotent=True, open_world=False),
    "build_regions": ToolSurface("Build regions from a spec", read_only=False, destructive=True, idempotent=True, open_world=False),
    "record_brief": ToolSurface("Record the brief", read_only=False, destructive=True, idempotent=False, open_world=False),
    "case_status": ToolSurface("Show case status", read_only=True, destructive=False, idempotent=True, open_world=False),
    # environment removes the registry entry of a server that has ended (`registry.read_servers`, P9-14): not read-only, not destructive
    # (the entry holds no data, as the lock of an ended job).
    "environment": ToolSurface("Check this machine", read_only=False, destructive=False, idempotent=True, open_world=False),
    "save_preferences": ToolSurface("Save user preferences", read_only=False, destructive=True, idempotent=True, open_world=False),
    "find_printer": ToolSurface("Find a 3D printer", read_only=True, destructive=False, idempotent=True, open_world=False),
    "save_printer": ToolSurface("Save a printer profile", read_only=False, destructive=True, idempotent=True, open_world=False),
    "find_material": ToolSurface("Find a filament", read_only=True, destructive=False, idempotent=True, open_world=False),
    "save_material": ToolSurface("Save a filament profile", read_only=False, destructive=True, idempotent=True, open_world=False),
    "printer_materials": ToolSurface("List filaments for a printer", read_only=True, destructive=False, idempotent=True, open_world=False),
    "find_features": ToolSurface("Find holes and faces of an STL", read_only=True, destructive=False, idempotent=True, open_world=False),
    "print_check": ToolSurface("Check printability", read_only=True, destructive=False, idempotent=True, open_world=False),
    "export_step": ToolSurface("Export a STEP file", read_only=False, destructive=True, idempotent=True, open_world=False),
    "cleanup_list": ToolSurface("List cleanup candidates", read_only=False, destructive=True, idempotent=False, open_world=False),
    "team_guide": ToolSurface("Get a role guide", read_only=True, destructive=False, idempotent=True, open_world=False),
    "show_problem": ToolSurface("Show the problem in 3D", read_only=True, destructive=False, idempotent=True, open_world=False),
}


# Argument keys that a client adds to the input schema of every MCP tool for its own use, so that the model can send them in a call.
# `_TopoptServer.call_tool` drops such a key before the name check and before the tool runs, but only for a tool that has no parameter of
# that name (a tool parameter of the same name is kept and passed). The refusal does not list these keys as valid names. The rule for a new
# entry (CONTRACTS.md section 10): a client documented to add a key of its own to the schema of every tool, or to the arguments of every
# call; the source and the read date in the comment; a test.
CLIENT_CONTROL_KEYS: tuple[str, ...] = (
    # Gemini CLI 0.67.0-nightly.20261020 (https://github.com/google-gemini/gemini-cli, public source, read 2026-10-28):
    # `packages/core/src/tools/tools.ts` (`addWaitForPreviousParameter`) adds `wait_for_previous` (boolean) to the schema of every MCP
    # tool, the system prompt tells the model to set it for a call that depends on an earlier one, `mcp-tool.ts` sends the arguments to the
    # MCP server as they are, and only `utils/tool-utils.ts` (`isParallelizable`) reads it. Not checked in Gemini CLI yet: a real call.
    "wait_for_previous",
)


class _TopoptServer(MCPServer):
    """MCPServer whose `tool` decorator wraps each tool with the work folder note and gives it its title and hints from `TOOL_SURFACE`.
    `@mcp.tool()` stays the form of every tool.

    Every tool is strict about argument names (#174). The SDK ignores a name that the tool does not have, and a call that only looks right
    then does a wrong thing (P9-7b: `save_preferences` made the "keep the defaults" file from `{"preferences": {...}}`; a typo such as
    `stop_rule` for `stop_rules` gives a run with the stop rules of the YAML). For each tool, the input schema has `additionalProperties: false`, and `call_tool`
    answers an unknown name with `{"ok": false, "error": ...}` before the tool runs. The check reads the parameter names of the tool, not the
    schema keyword. A key of `CLIENT_CONTROL_KEYS` is dropped first. `@mcp.tool(strict_arguments=False)` turns the check off for one tool
    (no tool does today).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._strict_argument_tools: set[str] = set()

    def tool(self, *args, strict_arguments: bool = True, **kwargs):
        sdk_tool = super().tool
        if args and callable(args[0]):
            raise TypeError("The tool decorator must be called: use @mcp.tool(), not @mcp.tool.")   # the SDK refuses it too

        def decorator(fn):
            name = kwargs.get("name") or (args[0] if args else None) or fn.__name__
            surface = TOOL_SURFACE.get(name)
            if surface is None:
                raise ValueError(f"The tool {name!r} has no row in TOOL_SURFACE (topopt_mcp/server.py). Add its title and the four hints.")
            hints = ToolAnnotations(read_only_hint=surface.read_only, destructive_hint=surface.destructive,
                                    idempotent_hint=surface.idempotent, open_world_hint=surface.open_world)
            options = dict(kwargs)
            if "description" not in options and fn.__doc__:
                # The same text on every Python (#388): 3.13 strips the common indentation of a docstring when it compiles it, 3.12 keeps
                # it, and the launcher snapshot (#370) must equal the live list on both.
                options["description"] = inspect.cleandoc(fn.__doc__)
            register = sdk_tool(*args, title=surface.title, annotations=hints, **options)
            wrapped = _with_work_folder_note(fn)
            register(wrapped)
            if strict_arguments:
                self._strict_argument_tools.add(name)
                self._tool_manager.get_tool(name).parameters["additionalProperties"] = False
            return wrapped

        return decorator

    async def call_tool(self, name, arguments, context=None):
        if name in self._strict_argument_tools:
            tool = self._tool_manager.get_tool(name)
            names = list(tool.parameters.get("properties", {}))
            given = dict(arguments or {})
            for key in CLIENT_CONTROL_KEYS:
                if key not in names:
                    given.pop(key, None)
            arguments = given
            unknown = sorted(str(key) for key in given if key not in names)
            if unknown:
                valid = f"The argument names are: {', '.join(names)}." if names else "It takes no arguments."
                reply = {"ok": False, "error": f"{name} did nothing: it has no argument {', '.join(repr(k) for k in unknown)}. {valid}"}
                _add_work_folder_note(reply, name)
                return tool.fn_metadata.convert_result(reply)
        return await super().call_tool(name, arguments, context)


mcp = _TopoptServer(
    "topopt",
    instructions=INSTRUCTIONS,
)


def _in_work_folder(path: Path) -> bool:
    """True when `path` lies in the work folder (the work folder itself counts)."""
    try:
        path.resolve().relative_to(workspace())
    except ValueError:
        return False
    return True


def _work_folder_refusal(write: Path, anywhere: bool = False) -> str | None:
    """The one refusal: the text when the server cannot create or write `write`, the folder that the tool writes in the work folder
    (`run_topopt`: the folder that holds the run folder; `build_regions`: `runs`). Else None. Nothing is created here.

    Without `anywhere`, a folder outside the work folder is left to the tool (it gives its own error). The work folder itself is not
    checked: a work folder that cannot be written, with a `runs/` that can, works.
    """
    folder = workfolder.resolve()
    if not anywhere and not _in_work_folder(write):
        return None
    _, writable, cause = workfolder.where(write)
    if writable:
        return None
    return folder.blocked or f"The server cannot write the folder {write} in the work folder {folder.path}: {cause}. {folder.fix}"


def _how_to_change() -> str:
    """For an error text: a space and how to change the work folder, in the words of the surface that started this server
    (`workfolder.change_text`). Empty when the work folder has a fix text of its own: `work_folder_note` of the same reply holds it,
    so it is not said twice."""
    return "" if workfolder.resolve().fix_needed else " " + workfolder.change_text()


def _yaml_not_found(shown: str) -> str:
    """The error for a YAML that is not there. Two ways out: a full path (every surface), or another work folder."""
    return f"YAML file not found: {shown} (work folder: {workspace()}). Give an absolute path." + _how_to_change()


def _make_work_folder() -> str | None:
    """Make the work folder just before a child process starts with it as its current folder. None, or a text when it cannot be made.

    A new install has no work folder yet (the default is `<home>/Documents/topopt`), and a child cannot start in a folder that is not there.
    The text is the work folder text, not the raw error of the system.
    """
    folder = workfolder.resolve()
    try:
        folder.path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return folder.blocked or f"The server cannot make the work folder {folder.path}: {exc.strerror or exc}. {folder.fix}"
    return None


def _child_env(**extra) -> dict:
    """The environment of a child process that works in the work folder (a run, a verify or ratchet job, `topopt step`, the environment probe, the import probe of `TOPOPT_PYTHON`).

    `TOPOPT_WORKSPACE` is the work folder as this server resolved it, a full path. A relative value (`w`) would be read again from the
    current folder of the child, which is the work folder: `w/w`, so the child would not find the same `topopt-printers.yaml` as the
    server (CONTRACTS.md section 18.7). With a full path the server and its children read one folder, whatever the current folder is.

    `PYTHONSAFEPATH=1` keeps the current folder off the module path. `python -m` and `python -c` put it first, and a user file in the
    work folder (`timeit.py`, `json.py`, `random.py`) would then take the place of a standard or dependency module, so a run would fail
    at the start or run the code of the user. The variable is in the environment, not only in the command, so the Python processes that
    the child starts in the same folder (the gmsh worker, the gmsh probe) have the safe path too. `PYTHONPATH` still works. Python 3.11
    and later read the variable; an older Python ignores it.
    """
    return {**os.environ, **extra, workfolder.ENV: str(workspace()), "PYTHONSAFEPATH": "1"}


def _read_cwd() -> str:
    """A current folder for a child process that only reads: the work folder when it is there, else its nearest existing parent.
    It makes nothing."""
    path = workspace()
    while not path.is_dir() and path.parent != path:
        path = path.parent
    return str(path)


# The run that this server started: Popen handle, run folder, start time. `fine_run_dir` is the second folder of `fine_pass=true` (#318), or None.
_current: dict = {}
# The run processes that `_stop` has signalled, by pid: {"run_dir": ..., "at": epoch seconds}. A failed run says whether this server stopped it (P9-6).
_stopped: dict = {}


def log(msg: str) -> None:
    print(f"[topopt_mcp] {msg}", file=sys.stderr, flush=True)


def _resolve(p: str | os.PathLike) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = workspace() / path
    return path.resolve()


def _resolve_yaml(p: str) -> Path:
    """Path of a problem YAML: relative paths start at the work folder.

    If a relative path is not there but is in the code folder (the examples of a source checkout or of the
    Claude Code plugin copy), that file is used. If no file exists, return the work-folder path (for the message).
    """
    yaml = _resolve(p)
    if not yaml.is_file() and not Path(p).expanduser().is_absolute():
        alt = (REPO / p).resolve()  # a source checkout or the plugin copy, started from another folder
        if alt.is_file():
            return alt
    return yaml


def _rel(p: str | os.PathLike | None) -> str | None:
    """Path relative to the work folder if it sits inside it, else the absolute path."""
    if p is None:
        return None
    path = Path(p)
    try:
        return path.resolve().relative_to(workspace()).as_posix() or "."
    except ValueError:
        return str(path)


def _command(yaml: Path, out: Path, port: int, override_lock: bool = False, resume: bool = False,
             allow_changed_inputs: bool = False, stop_rules: bool | None = None,
             continue_from: Path | None = None, more_iterations: int | None = None, fine_from: Path | None = None,
             fine_pass: bool = False, fine_voxel=None, fine_passes=None, fine_tol=None) -> list[str]:
    template = os.environ.get("TOPOPT_MCP_CMD")
    if template:   # the test hook: the words are replaced and no run flag (resume, stop_rules, continue_from, more_iterations, ...) is added
        return [
            w.replace("{yaml}", str(yaml)).replace("{out}", str(out)).replace("{port}", str(port))
            for w in shlex.split(template)
        ]
    cmd = [python_exe(), "-m", "topopt", "run", str(yaml), "--out", str(out), "--dashboard",
           "--port", str(port)]
    # `claude plugin eval` passes only EVAL_* variables. An eval job needs no dashboard after the run: with
    # EVAL_TOPOPT_NO_WAIT=1 the job process ends when it is done (else it stays on and serves the result).
    if os.environ.get("EVAL_TOPOPT_NO_WAIT") == "1":
        cmd.append("--no-wait")
    flags = (["--override-lock"] if override_lock else []) + (["--resume"] if resume else []) \
        + (["--allow-changed-inputs"] if allow_changed_inputs else []) \
        + ([] if continue_from is None else ["--continue-from", str(continue_from), "--more-iterations", str(more_iterations)]) \
        + ([] if stop_rules is None else ["--stop-rules" if stop_rules else "--no-stop-rules"]) \
        + ([] if fine_from is None else ["--fine-from", str(fine_from)]) + (["--fine-pass"] if fine_pass else []) \
        + ([] if fine_voxel is None else ["--fine-voxel", str(fine_voxel)]) + ([] if fine_passes is None else ["--fine-passes", str(fine_passes)]) \
        + ([] if fine_tol is None else ["--fine-tol", str(fine_tol)])
    return cmd + flags


def _python_problem(exe: str) -> str | None:
    """Return a message if another Python (not this one) cannot import topopt, else None."""
    if exe == sys.executable:
        return None
    try:
        r = subprocess.run([exe, "-c", "import topopt"], capture_output=True, text=True,
                           timeout=30, stdin=subprocess.DEVNULL, env=_child_env())    # `-c` puts the current folder first: PYTHONSAFEPATH keeps it off
    except OSError as exc:
        return f"Python {exe} cannot start: {exc}"
    except subprocess.TimeoutExpired:
        return None  # slow import: not a fact about a failure
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["?"])[-1]
        return f"Python {exe} cannot import topopt ({tail})"
    return None


def _read_json(path: Path):
    """Return the JSON object in a file, or None if it is missing, unreadable or not an object."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _last_history(run_dir: Path):
    """The last row of history.jsonl that parses, or None. A part line is skipped (the shared reader)."""
    from topopt.runinfo import read_history
    try:
        rows = read_history(run_dir / "history.jsonl", strict=False).rows
    except OSError:
        return None
    return rows[-1] if rows else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_conflict(out: Path, override: bool, ignore_pid: int | None = None) -> dict | None:
    """The reply of run_topopt if a live process holds the job lock of the folder (CONTRACTS.md 24.1), else None.

    `ignore_pid` is the pid of the old run process of this server, which `run_topopt` is about to stop.
    """
    from topopt.runjob import live_holder
    held = None if override else live_holder(out)
    if held and ignore_pid is not None and held.get("pid") == ignore_pid:
        held = None
    return {"error": held["message"], "run_dir": str(out), "job": held} if held else None


def _is_current_dir(d: Path) -> bool:
    """True if `d` is a folder of the run process that this server started: its run folder, or the fine pass folder of `fine_pass=true`."""
    return d == _current.get("run_dir") or (_current.get("fine_run_dir") is not None and d == _current.get("fine_run_dir"))


def _own_exit(run_dir: Path) -> int | None:
    """Exit code of the run process that this server started for this folder, if it has ended. Else None."""
    proc = _current.get("proc")
    if proc is not None and _is_current_dir(run_dir):
        return proc.poll()
    return None


def _exit_text(code: int) -> str:
    return f"ended with exit {code}" if code >= 0 else f"was stopped by signal {-code}"


def _log_path(run_dir: Path) -> Path:
    return run_dir.parent / (run_dir.name + ".stdout")


def _job_facts(run_dir: Path, started: float | None = None, own_exit: int | None = None) -> dict | None:
    """The job facts of the folder (state, exit code, wall time, peak memory), or None if it has no job files.

    `started` is the start time of the run that this server started. A job record older than that is from an
    earlier run: the new process has not written yet. The facts say so.
    `own_exit` is the exit code of that run process if it has ended. If it ended and wrote no new job record, the
    facts are `state: no_record` with the exit code, and the old record (if any) is named as an earlier job.
    """
    from topopt.runjob import job_facts
    job = job_facts(run_dir)
    if started is not None and (job is None or job["state"] != "running"):
        try:
            older = job is not None and (run_dir / "job.json").stat().st_mtime < started - 1
        except OSError:
            older = False
        if own_exit is not None and (job is None or older):
            log = _log_path(run_dir)
            facts = {"state": "no_record", "exit_code": own_exit, "run_dir": str(run_dir), "log_file": str(log),
                     "summary": f"job: the run process {_exit_text(own_exit)} and wrote no job record (see {log})"}
            if job is not None:
                facts["earlier_job"] = {k: job.get(k) for k in ("state", "exit_code", "started_at", "ended_at")}
            return facts
        if older:
            job["earlier_run"] = True
            job["summary"] = job["summary"].replace("job:", "job (an earlier run; the new run has not written yet):", 1)
    return job


def _checkpoint_facts(run_dir: Path, run: dict) -> dict | None:
    """Facts about the checkpoint of the folder (CONTRACTS.md section 24.2), or None if it has none. No arrays are read."""
    from topopt.checkpoint import describe
    from topopt.runinfo import run_yaml_path
    cp = describe(run_dir)
    if cp is not None:
        yp = run_yaml_path(run, run_dir)   # a relative yaml_path of run.json starts at the run folder
        cp["yaml_path"] = None if yp is None else os.path.realpath(yp)   # the argument for run_topopt(resume=true): no `..`
    return cp


def _resume_refusal(yaml: Path, out: Path, allow_changed_inputs: bool, stop_rules: bool | None = None) -> dict | None:
    """The reply of run_topopt(resume=true) if a resume cannot start, else None. It writes nothing.

    It reads the checkpoint and compares the YAML, the region files and the settings. The run itself also compares the grid.
    """
    from topopt.checkpoint import ResumeRefused, config_of_continuation, config_of_fine, describe, plan_resume
    from topopt.cli import with_stop_rules
    from topopt.config import UserError, load_config
    try:   # a fine pass keeps its voxel, schedule and stop rule (#318), and a continuation its pass count (P4b-5): from the checkpoint, not from the YAML
        plan_resume(out, config_of_continuation(out, config_of_fine(out, with_stop_rules(load_config(yaml), stop_rules), stop_rules)),
                    allow_changed=allow_changed_inputs)
    except ResumeRefused as exc:
        return {"error": str(exc), "run_dir": str(out),
                "resume": {"facts": exc.facts, "options": exc.options, "checkpoint": describe(out)}}
    except UserError as exc:
        return {"error": str(exc), "run_dir": str(out)}
    return None


def _resume_pass_keys(yaml: Path, out: Path) -> dict | None:
    """The reply key `resume` of run_topopt(resume=true, allow_changed_inputs=true) on a finished run that has changed keys which act in
    the passes (CONTRACTS.md section 41.8), else None. It reads the checkpoint and writes nothing."""
    from topopt.checkpoint import ResumeRefused, plan_resume
    from topopt.config import UserError, load_config
    from topopt.resume_fact import pass_key_note
    try:
        plan = plan_resume(out, load_config(yaml), allow_changed=True)
    except (ResumeRefused, UserError):   # the refusal check before has the reply for these
        return None
    note = pass_key_note(plan.changes, bool(plan.ck.meta.get("complete")))
    return None if note is None else {"facts": [note["fact"]], "options": note["options"], "pass_keys": note["keys"]}


def _continue_plan(yaml: Path, out: Path, parent: Path, more: int, allow_changed_inputs: bool,
                   stop_rules: bool | None) -> tuple[dict | None, dict | None]:
    """The plan of run_topopt(continue_from=...) without any write: `(refusal, info)`. One of them is None.

    `refusal` is the reply if the continuation cannot start: the error with `continue` (`facts`, `options`, the `checkpoint` of the parent). A new
    folder that is the parent or inside it is a refusal too. `info` is the key `continue` of the reply of a run that starts: the pass of the
    checkpoint, the pass count of the new run, whether pass K runs again, and the facts. The parent is only read.
    """
    from topopt.checkpoint import CONTINUE_RESTART_OPTION, ResumeRefused, config_for_continuation, describe, plan_continue, read_checkpoint
    from topopt.cli import with_stop_rules
    from topopt.config import UserError, load_config
    try:
        ck = read_checkpoint(parent, CONTINUE_RESTART_OPTION)   # the one read; a parent that is a fine pass gives its voxel, schedule and stop rule (#318)
        config = config_for_continuation(ck, with_stop_rules(load_config(yaml), stop_rules), stop_rules)
        plan = plan_continue(parent, config, None, None, more, allow_changed_inputs, new_dir=out, ck=ck)
    except ResumeRefused as exc:
        return {"error": str(exc), "run_dir": str(out),
                "continue": {"facts": exc.facts, "options": exc.options, "checkpoint": describe(parent)}}, None
    except UserError as exc:
        return {"error": str(exc), "run_dir": str(out)}, None
    rec = plan.continued_from
    facts = list(plan.facts)
    job = _job_facts(parent)
    if job is not None and job["state"] == "running":
        # The run process is another process: it reads the checkpoint again when it starts, and its read decides.
        facts.append(f"continue: the numbers in this reply come from a read before the run process starts. The parent runs, and the run process "
                     f"reads its checkpoint again. Its pass K can be later than pass {rec['iteration']}, and the pass count K + "
                     f"{rec['more_iterations']} with it. run_status and the continued_from record of run.json name the pass that the run read.")
    return None, {"parent": str(parent), "iteration": rec["iteration"], "parent_iterations": rec["parent_iterations"],
                  "iterations": rec["iterations"], "more_iterations": rec["more_iterations"], "pass_runs_again": rec["replay"],
                  "step_settings": rec["step_settings"], "facts": facts}


def _continue_arguments_error(continue_from, more_iterations, resume: bool) -> dict | None:
    """The `error` reply of run_topopt for a wrong mix of `continue_from`, `more_iterations` and `resume`, else None. It writes nothing."""
    from topopt.checkpoint import check_more_iterations
    from topopt.config import UserError
    if continue_from is None or more_iterations is None:
        return {"error": "continue_from and more_iterations go together. continue_from names the run folder to continue, and more_iterations "
                         "says how many passes to add. Options: give both, for example continue_from='runs/hook', more_iterations=20."}
    if resume:
        return {"error": "resume goes on in out_dir, and continue_from starts a new run in out_dir from another run folder. Options: give "
                         "resume=true only, to go on in out_dir; or continue_from and more_iterations only, to add passes in a new folder."}
    try:
        check_more_iterations(more_iterations)
    except UserError as exc:
        return {"error": str(exc).replace("--more-iterations", "more_iterations")}
    return None


def _fine_arguments_error(fine_from, fine_pass, fine_voxel, fine_passes, fine_tol, resume, continue_from, more_iterations,
                          allow_changed_inputs) -> dict | None:
    """The `error` reply of run_topopt for a wrong mix of the fine pass arguments (#318, CONTRACTS.md section 24.6), or a wrong value. Else None.

    It writes nothing. `fine_voxel`, `fine_passes` and `fine_tol` go with `fine_from` or `fine_pass`. `fine_from` and `fine_pass` do not go with each
    other, with `resume`, `continue_from` or `more_iterations`; `fine_pass` does not go with `allow_changed_inputs` either.
    """
    from topopt.config import UserError
    from topopt.fine_pass import check_fine_passes, check_fine_tol, check_fine_voxel
    if fine_from is None and not fine_pass:
        given = [n for n, v in (("fine_voxel", fine_voxel), ("fine_passes", fine_passes), ("fine_tol", fine_tol)) if v is not None]
        if given:
            return {"error": f"{', '.join(given)} go with fine_from or fine_pass: they set the fine pass. Options: add fine_pass=true to run the first "
                             "run and then the fine pass; or fine_from=PARENT to run the fine pass from a finished run folder; or leave them out."}
        return None
    name = "fine_from" if fine_from is not None else "fine_pass"
    if fine_from is not None and fine_pass:
        return {"error": "fine_from and fine_pass do not go together. fine_pass=true makes a new first run in out_dir and then the fine pass in "
                         "<out_dir>-fine. fine_from=PARENT makes the fine pass only, from a finished run folder. Options: give one of them."}
    for flag, given in (("resume", resume), ("continue_from", continue_from), ("more_iterations", more_iterations),
                        ("allow_changed_inputs", allow_changed_inputs and fine_pass)):
        if given:
            return {"error": f"{name} does not go with {flag}. A fine pass is a new run in a new folder, on a finer voxel. Options: leave {flag} out; "
                             "resume=true alone goes on in out_dir; continue_from with more_iterations adds passes on the same grid."}
    try:
        for check, value in ((check_fine_voxel, fine_voxel), (check_fine_passes, fine_passes), (check_fine_tol, fine_tol)):
            if value is not None:
                check(value)
    except UserError as exc:
        return {"error": _mcp_words(str(exc))}
    return None


_FLAG_WORDS = re.compile(r"--(fine-voxel|fine-passes|fine-tol|fine-from|continue-from|more-iterations|allow-changed-inputs)\b(?: (PARENT|N|H)\b)?")


def _mcp_words(text: str) -> str:
    """The text of the engine with the argument names of the tool in place of the flags of the CLI (`--fine-voxel H` -> `fine_voxel=H`)."""
    return _FLAG_WORDS.sub(lambda m: m.group(1).replace("-", "_") + ("" if m.group(2) is None else "=" + m.group(2)), text)


def _fine_plan(yaml: Path, out: Path, parent: Path, allow_changed_inputs: bool, stop_rules: bool | None, fine_voxel, fine_passes,
               fine_tol) -> tuple[dict | None, dict | None]:
    """The plan of run_topopt(fine_from=...) without any write: `(refusal, info)`. One of them is None.

    `refusal` is the reply if the fine pass cannot start: the error with `fine` (`facts`, `options`). A new folder that is the parent or inside it is a
    refusal too. `info` is the key `fine` of the reply of a run that starts: the parent pass and voxel, the fine voxel and where it comes from (with the voxel
    of the part: `limiting`, `limiting_role`, `limiting_thickness_mm`, `grid_check`), the maximum of passes, the tolerance, and the facts. The parent is only read.
    """
    from topopt import fine_pass as fp
    from topopt.checkpoint import ResumeRefused, plan_fine
    from topopt.cli import with_stop_rules
    from topopt.config import UserError, load_config
    try:
        setup = fp.resolve_fine(with_stop_rules(load_config(yaml), stop_rules), parent, out, fine_voxel, fine_passes, fine_tol,
                                early_stop=stop_rules is not False)
        facts = list(setup.facts)
        if setup.parent.ck is not None:
            facts += plan_fine(setup.parent.ck, setup.config, None, None, allow_changed_inputs, parent)
    except ResumeRefused as exc:
        return {"error": _mcp_words(str(exc)), "run_dir": str(out),
                "fine": {"facts": [_mcp_words(f) for f in exc.facts], "options": [_mcp_words(o) for o in exc.options]}}, None
    except UserError as exc:
        return {"error": _mcp_words(str(exc)), "run_dir": str(out)}, None
    source, source_facts = fp.choose_source(setup.parent)
    rec = fp.fine_record(setup, source)
    keys = ("iteration", "parent_voxel_mm", "voxel_mm", "voxel_from", "max_passes", "max_from", "tolerance", "tolerance_from", "window",
            "schedule_offset", "source")
    part = ("part_voxel_mm", "limiting", "limiting_role", "limiting_thickness_mm", "grid_check")   # only when the voxel is the one that the part needs
    return None, {"parent": str(parent), **{k: rec[k] for k in keys}, **{k: rec[k] for k in part if k in rec}, "facts": [_mcp_words(f) for f in (fp.plan_line(setup), *facts, *source_facts)]}


def _earlier_checkpoint(out: Path) -> dict | None:
    """Facts for a run without `resume` in a folder that holds the checkpoint of a run that did not finish, else None.

    The new run deletes that checkpoint (a fresh run deletes the files of the old run). The reply says so and the run goes on: the
    caller decides. `run_topopt` with `resume=true` continues the old run.
    """
    from topopt.checkpoint import describe
    cp = describe(out)
    if cp is None or cp["complete"]:
        return None
    return {**cp, "note": f"The run folder has a checkpoint at pass {cp['iteration']} of {cp['iterations']} from an earlier run that did not "
                          "finish. This run deletes it, as a new run deletes the files of an old run. run_topopt with resume=true "
                          "continues the old run instead."}


def _history_facts(run_dir: Path) -> list[str]:
    """What the reader skips in history.jsonl (a partial last line), as short sentences."""
    from topopt.runinfo import read_history
    try:
        return read_history(run_dir / "history.jsonl", strict=False).facts
    except OSError:
        return []


def _pid_file(run_dir: Path) -> Path:
    return run_dir.parent / (run_dir.name + ".pid")


def _is_running(run_dir: Path) -> int | None:
    """Return the pid if a run process for this folder is alive, else None."""
    proc = _current.get("proc")
    if proc is not None and _is_current_dir(run_dir):
        return proc.pid if proc.poll() is None else None
    # No handle (the server restarted): use the pid file.
    try:
        pid = int(_pid_file(run_dir).read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def _alive_run() -> tuple[Path, int] | None:
    """Return (run folder, pid) of a run process that is alive, else None."""
    proc = _current.get("proc")
    if proc is not None and proc.poll() is None:
        return _current["run_dir"], proc.pid
    d = default_out()
    if _pid_file(d).exists():
        pid = _is_running(d)
        if pid:
            return d, pid
    return None


# Texts of the refusals "a job is still running" (run_topopt, verify_run, ratchet_run): the options that are true for each kind of job.
STOP_RUN_OPTION = ("To end the run sooner, call stop_run: it asks the run to stop after its current pass (a run of an older build ignores "
                   "the request). The checkpoint is kept, and run_topopt with resume=true continues the run.")
NO_STOP_VERIFY = "stop_run does not stop a verify job: a verify job reads no job.stop."
NO_STOP_RATCHET = "stop_run does not stop a ratchet job: a ratchet job reads no job.stop."


def _run_works(run_dir: Path) -> bool:
    """True if the run of this folder still works: new, or state starting/running/postprocess. The fine pass to come (`fine_pass=true`) is not counted."""
    status_path = run_dir / "status.json"
    status = _read_json(status_path)
    if status is None:
        return True  # the run is just starting
    started = _current.get("started") if _is_current_dir(run_dir) else None
    if started is not None:
        try:
            if status_path.stat().st_mtime < started - 1:
                return True  # old status file: the new run did not write yet
        except OSError:
            return True
    return status.get("state") in BUSY_STATES


def _is_busy(run_dir: Path) -> bool:
    """True if the run still works: new, or state starting/running/postprocess; or its fine pass (`fine_pass=true`) is still to come or runs.

    A live process with a done or error run only serves the dashboard.
    """
    return _run_works(run_dir) or _fine_pending(run_dir)


def _fine_run_block(d: Path) -> dict | None:
    """The key `fine_run` of run_status for the first folder of `fine_pass=true` (#318): the fine run folder and its state. None for any other folder.

    `state` is the status of the fine run (starting, running, postprocess, done, stopped), `not started` while the process is alive and the fine pass has
    not written its status, or, after the process has ended: `failed` (the fine run wrote the status error), `refused` (the process ended and the fine
    run never started: the guards or the plan refused it; `message` has the exit and the log) or `none` (no fine pass was needed or made: the first run
    did not end well, or the part needs no finer voxel).
    """
    fine = _current.get("fine_run_dir")
    if fine is None or _current.get("run_dir") != d:
        return None
    block = {"run_dir": str(fine), "run_dir_rel": _rel(fine)}
    voxel = (_read_json(d / "run.json") or {}).get("part_voxel")
    if voxel is not None and not _fine_follows(voxel):
        return {**block, "state": "none", "note": "the part does not need a finer voxel than the run has: there is no fine pass"}
    status = _read_json(fine / "status.json")
    try:
        old = status is not None and (fine / "status.json").stat().st_mtime < (_current.get("started") or 0) - 1
    except OSError:
        old = True
    state = "not started" if status is None or old else status.get("state", "?")
    proc = _current.get("proc")
    ended = proc is not None and proc.poll() is not None   # the process of `fine_pass=true` makes both runs: after its end nothing is left to start
    if state == "error":
        return {**block, "state": "failed", "message": (status or {}).get("message") or f"the fine run ended with an error; the log is {_log_path(d)}",
                "log": str(_log_path(d))}
    if state == "not started" and ended:
        first = (_read_json(d / "status.json") or {}).get("state")
        from topopt.runjob import STOPPED_EXIT
        if proc.poll() == STOPPED_EXIT and first == "done":   # a stop request came after the first run: the process did not start the fine pass
            return {**block, "state": "stopped", "note": "a stop request came after the first run: the fine pass did not start. The first run is finished "
                    "and keeps its result", "log": str(_log_path(d))}
        if first in ("error", "stopped"):
            return {**block, "state": "none", "note": f"the first run ended with the state {first}: there is no fine pass", "log": str(_log_path(d))}
        return {**block, "state": "refused", "message": f"the process {_exit_text(proc.poll())} before the fine run started, so the fine pass was refused "
                f"or could not start. The reason is in the log {_log_path(d)}: it is often a guard of the fine grid; the options are in the "
                "log and in check_problem `fine_pass.guards`.", "log": str(_log_path(d))}
    return {**block, "state": state}


def _fine_follows(voxel: dict) -> bool:
    """True if the fine pass of `fine_pass=true` follows the first run whose `run.json` has this `part_voxel` block (#318).

    The part needs a finer voxel (`recommend`), or the caller gave `fine_voxel` (the process checked that it is finer than the run, and it wins:
    the fine pass runs at it also when the part needs no finer voxel).
    """
    return bool(voxel.get("recommend") or _current.get("fine_voxel") is not None)


def _fine_pending(run_dir: Path) -> bool:
    """True while the process of `fine_pass=true` (#318) has a fine pass to do for this first run folder.

    The first run is done, its run.json says that the part needs a finer voxel (`part_voxel.recommend`) or the caller gave `fine_voxel`, and the fine run
    folder has no end state yet.
    So a second run_topopt does not stop the process in the middle of the fine pass. A process that has ended is not alive: `_alive_run` ignores it.
    """
    fine = _current.get("fine_run_dir")
    if fine is None or _current.get("run_dir") != run_dir:
        return False
    voxel = (_read_json(run_dir / "run.json") or {}).get("part_voxel")
    if not voxel or not _fine_follows(voxel):
        return False
    status = _read_json(fine / "status.json")
    if status is None:
        return True    # the fine pass has not written its status yet
    try:
        if (fine / "status.json").stat().st_mtime < (_current.get("started") or 0) - 1:
            return True    # the status of an older run in that folder
    except OSError:
        return True
    return status.get("state") in BUSY_STATES


def _fine_target(first: Path) -> Path | None:
    """The fine run folder when `stop_run()` with no run_dir must stop the fine pass of `fine_pass=true` (#318), else None.

    While the first run works, the first folder is the target (a stop of it also stops the fine pass: the process makes the fine pass only after
    a first run that ended with exit code 0). When the first run is done and the fine pass is to come or runs, the target is the fine folder.
    """
    fine = _current.get("fine_run_dir")
    if fine is None or _current.get("run_dir") != first or _run_works(first) or not _fine_pending(first):
        return None
    return Path(fine)


def _fine_not_started(first: Path) -> tuple[Path, str] | None:
    """`(fine folder, phase)` while a stop request still prevents the fine pass of `fine_pass=true` (#318) from starting, else None.

    The windows are: the post step and the refine step of the first run (`phase` is its status, `postprocess` or `done`, and no request is read there), and
    the time after the first run until the fine run has written its status (the process reads the first run, plans the fine grid and builds it).
    Before the post step the first folder is the target of a stop (the first run reads it, and the fine pass does not start after it); once the fine run
    has a status, the fine folder is.
    """
    fine = _current.get("fine_run_dir")
    if fine is None or _current.get("run_dir") != first or not _fine_pending(first):
        return None
    phase = (_read_json(first / "status.json") or {}).get("state")
    if phase not in ("postprocess", "done") or (_fine_run_block(first) or {}).get("state") != "not started":
        return None
    return Path(fine), phase


def _stop_before_fine(first: Path, fine: Path, phase: str, pid: int) -> dict:
    """The reply of `stop_run()` with no run_dir while the fine pass of `fine_pass=true` has not started (#318): the request goes to the fine folder.

    The process (pid) reads `job.stop` in the fine folder before it starts the fine pass, ends without starting it (exit code 5) and says so in the
    log. The reply has `state` stop_requested (or already_requested) and the fine folder in `run_dir`, not `none`.
    """
    from topopt.runjob import stop_reply
    reply = stop_reply(fine, by="mcp stop_run", starting_pid=pid)
    where = "in its post step" if phase == "postprocess" else "done (its refine step may still run)" if phase == "done" else phase
    if reply["state"] in ("stop_requested", "already_requested"):
        verb = "is written" if reply["state"] == "stop_requested" else "is in the folder already"
        reply.update(
            message=f"The first run in {first} is {where}, and the fine pass in {fine} has not started. The request to stop {verb} in {fine}. The process "
                    f"(pid {pid}) reads it before it starts the fine pass, ends without starting it (exit code 5) and says so in the log: a stop request "
                    "came after the first run, the fine pass did not start. The first run is finished and keeps its result.",
            options=["wait: run_status of the first folder shows fine_run.state `stopped` when the process has ended",
                     f"remove job.stop in {fine} to take the request back, before the process reads it",
                     f"to make the fine pass later: run_topopt(yaml_path, out_dir={str(fine)!r}, fine_from={str(first)!r})"])
    reply.update(run_dir_rel=_rel(fine), found_by="the fine pass of fine_pass=true that this server started, before it started", process_alive=True)
    return reply


def _stop(run_dir: Path, pid: int) -> None:
    """Stop a run process group: SIGTERM, wait 3 s, then SIGKILL."""
    log(f"stop old run process {pid} ({run_dir})")
    proc = _current.get("proc")
    handle = proc if proc is not None and proc.pid == pid else None

    def alive() -> bool:
        return handle.poll() is None if handle else _pid_alive(pid)

    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    _stopped[pid] = {"run_dir": str(run_dir), "at": time.time()}   # run_status names this stop as a fact (P9-6)
    end = time.time() + 3
    while time.time() < end and alive():
        time.sleep(0.05)
    if alive():
        log(f"SIGKILL process group {pid}")
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if handle:
            handle.wait()


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _yaml_assumptions(yaml: Path) -> dict:
    """`assumptions` and `derived` (the numbers that the YAML gives) of a problem YAML, to show the user."""
    try:
        from topopt.cli import derived_numbers
        from topopt.config import load_config
        cfg = load_config(yaml)
        return {"assumptions": list(cfg.assumptions), "derived": derived_numbers(cfg)}
    except Exception as exc:  # the run has started: this only adds facts
        return {"assumptions": [], "derived": None, "derived_error": str(exc)}


# The entry of this server in the registry of topopt servers (CONTRACTS.md section 43). None until main() has made it.
_registration: registry.Registration | None = None


def _register_server(folder: workfolder.WorkFolder) -> None:
    """Write the entry of this server and remove it at a clean exit. A problem is one line on stderr: the server goes on without the entry."""
    global _registration
    try:
        import topopt
        reg = registry.Registration(registry.KIND_MCP_SERVER, work_folder=folder.path, started_utc=STARTED_UTC, code_folder=REPO,
                                    python=sys.executable, topopt_version=topopt.__version__)
        _registration = reg
        problem = reg.start()
        atexit.register(reg.remove)      # also after a problem: a later write can work
        if problem:
            log(problem)
    except Exception as exc:
        log(f"The registry of topopt servers is off: the entry of this server is not written ({type(exc).__name__}: {exc}). "
            "Other servers do not see this server.")


def _note_run_dir(out: Path, run_pid: int | None = None) -> None:
    """A run started in `out`: add the folder, and the run process `run_pid`, to the entry of this server. Does nothing before main() has made the entry."""
    if _registration is None:
        return
    try:
        problem = _registration.add_run_dir(out, run_pid)
        if problem:
            log(problem)
    except Exception as exc:
        log(f"The registry of topopt servers is off: the entry of this server is not written ({type(exc).__name__}: {exc}).")


def _other_servers() -> tuple[list, str | None]:
    """The other servers (`registry.read_servers`) for the reply of `environment`, and one text for what the registry could not do.
    It never raises."""
    try:
        others = registry.read_servers()
        notes = ([_registration.problem] if _registration is not None and _registration.problem else []) + others.notes
        return others.servers, " ".join(notes) or None
    except Exception as exc:
        return [], f"The registry of topopt servers cannot be read ({type(exc).__name__}: {exc})."


def _gib(nbytes) -> str:
    return f"{nbytes / 2**30:.1f} GiB" if isinstance(nbytes, (int, float)) else "unknown"


def _other_server_jobs(out: Path) -> dict | None:
    """A fact for the reply of `run_topopt`: a listed server has a running job. None when there is none. It never raises.

    The run has started already. The fact has the job (which server, and if that server has ended; which run folder; the memory estimate if
    the run folder has one), the RAM of this machine, and options. A server that has ended is listed while its run goes on. It is not a stop,
    and it adds no limit.
    """
    try:
        jobs = registry.running_jobs(registry.read_servers().servers)
        if not jobs:
            return None
        from topopt.environment import _ram
        ram = _ram()
        total, free = ram.get("total_bytes"), ram.get("available_bytes")
        what = []
        for j in jobs:
            parts = [f"started by server {j['server_pid']}" + (" (no longer running)" if j["server_state"] in registry.ENDED_STATES else "")]
            if j["server_state"] == "unconfirmed":
                parts.append("may be stale")
            if j["memory_estimate_gib"] is not None:
                parts.append(f"memory estimate about {j['memory_estimate_gib']:g} GiB")
            what.append(f"pid {j['pid']} in {j['run_dir']} ({', '.join(parts)})")
        fact = (f"A job that another topopt server started is running now: {'; '.join(what)}. "
                f"This machine has {_gib(total)} of RAM and {_gib(free)} free now. "
                "Two heavy jobs at the same time can use all the RAM, and the machine can freeze. This run has started already.")
        return {
            "fact": fact,
            "jobs": [{k: j[k] for k in ("server_pid", "server_state", "work_folder", "run_dir", "pid", "started_at", "memory_estimate_gib")}
                     for j in jobs],
            "ram": {"total_bytes": total, "available_bytes": free},
            "options": [
                "wait: stop this run with stop_run, and start it again when the other job has ended (run_status with the run folder of "
                "the other job shows its state)",
                "stop the other job: stop_run with its run_dir asks it to stop after its current pass; its checkpoint is kept",
                f"run both: this run is going ({out}); tell the user the memory estimates and the RAM",
            ],
        }
    except Exception:
        return None


# #440: no MCP client starts a new model turn when a background job ends, and the server sends nothing that could.
# One fact, in the texts of the tools that start or show a job. `end` says when the job has ended, in the words of that tool.
RUN_END = "the state is done, error or stopped, or process_alive is false"
VERIFY_END = "the state is done, error or failed"
RATCHET_END = "the state is finished, waiting_for_owner, baseline_failed, stopped_after_approval or failed, or process_alive is false"


def no_callback(tool: str, end: str, noun: str = "job") -> str:
    """The reply line: nothing calls the agent back when the job ends, a promise of a later report cannot be kept, and the two choices."""
    return (f"Nothing calls you back when the {noun} ends. A promise of a later report cannot be kept. "
            f"You can call {tool} again in this turn until {end}. "
            "Or you can end the turn and tell the user what to ask for the result, for example \"what is the result?\".")


@mcp.tool()
def run_topopt(yaml_path: str, out_dir: str | None = None, override_lock: bool = False, resume: bool = False,
               allow_changed_inputs: bool = False, stop_rules: bool | None = None,
               proceed_without_brief: bool = False, continue_from: str | None = None, more_iterations: int | None = None,
               fine_from: str | None = None, fine_pass: bool = False, fine_voxel: float | None = None, fine_passes: int | None = None,
               fine_tol: float | None = None) -> dict:
    """Start a topology optimization run in the background. Returns at once.

    Use after check_problem and after you state your assumptions to the user. If the user gave no load (no object, no mass and no force), ask first. Then run_status to watch it, and run_result when it is done.
    Nothing calls you back when the run ends: call run_status in this turn, or end the turn and tell the user what to ask.
    It also returns `assumptions` (the strings of the YAML) and `derived` (forces in N, voxel size, minimum feature size, volume fraction), so you can show them.
    Reply keys: resource topopt://replies/run_topopt (the meaning of each key, and the fine pass: fine_from, fine_pass).

    yaml_path: the problem YAML (relative paths start at the work folder; `environment` shows it).
    out_dir: the run folder. Default: runs/live in the work folder (the dashboard shows it). Without resume, a folder that holds the checkpoint of a run that did not finish loses it: the reply then has `earlier_checkpoint` (the pass and a note), and the run goes on.
    override_lock: a live process can hold the job lock of the run folder (the reply names its pid and start time). Give true to run anyway. A lock of a stopped process is replaced without it.
    proceed_without_brief: a YAML with a case file (made by record_brief) can have no brief, or a stale brief (a changed STL file or requirement key). Then the reply is `{"state": "awaiting_user", "facts", "options", "override"}` and nothing starts. Give true to run anyway: the case log records it.
    resume: continue the run of out_dir from its last checkpoint (run_status shows `checkpoint`). The history and the snapshots after it are dropped. The YAML and the region files are read again: if they or the settings changed, the reply is an `error` with the changes in `resume.facts`, and nothing is written. A run that stop_run stopped is continued the same way, with no allow_changed_inputs.
    continue_from: add passes to a run in a NEW run folder (out_dir); this folder is the parent and is only read. more_iterations (with continue_from): the number of passes to add (a whole number of 1 or more, no upper limit). The new run starts from the last checkpoint of the parent (pass K) and has K + more_iterations passes (settings.iterations is not used). The reply has `continue`. The card run_topopt:continue_from has the rest.
    allow_changed_inputs: with resume or continue_from, go on with the changed inputs. The run log and the verdict name the changes. With resume=true the stop of proceed_without_brief names both overrides in one option (this one and allow_changed_inputs=true). On a finished run a resume runs the post step only: a key that only it reads (for example `settings.mesh_method`) takes effect in seconds; a key of the passes is accepted, the part stays, and the reply has `resume` (with pass_keys).
    stop_rules: true turns the stop rules on, false turns them off. None: as `settings.stop_rules` of the YAML says (off by default). A rule ends the run at a pass that only measures. The verdict names the rule and its numbers.
    With resume, a run that was killed gives the same result as one that was not (bit-equal with one thread and a direct solver).
    The lost force is in check_problem (before) and in run_result or run.json (after). A moment, a remote force, a pressure or a per-axis support gives `extended` in check_problem; the stiffness of such a case is null with a reason.

    Who decides a change of the YAML: a method setting (penal, beta schedule, solver, voxel size, iterations, a voxel guard: the memory estimate against the RAM first, above it the machine can freeze) you change without asking, and report what, why and the price. A user requirement (loads, supports, keep-outs, material, stress limits and safety factors, a mass target, print limits, symmetry) you may tighten without asking, with a report and its measured price; you loosen it only with the user's consent, and you bring the facts. A key that can be either follows its origin in record_brief. Cards: topopt://levers.
    """
    yaml = _resolve_yaml(yaml_path)
    if not yaml.is_file():
        return {"error": _yaml_not_found(str(yaml))}
    out = _resolve(out_dir) if out_dir else default_out()
    refusal = _work_folder_refusal(out.parent)    # the log, the pid file and the run folder go in this folder
    if refusal:
        return {"error": refusal}
    parent, more = None, None
    if continue_from is not None or more_iterations is not None:   # P4b-5: a new run folder from the checkpoint of another run folder
        wrong = _continue_arguments_error(continue_from, more_iterations, resume)
        if wrong:
            return wrong
        parent, more = _resolve(continue_from), int(more_iterations)
    wrong = _fine_arguments_error(fine_from, fine_pass, fine_voxel, fine_passes, fine_tol, resume, continue_from, more_iterations, allow_changed_inputs)
    if wrong:   # #318: a wrong mix or value of the fine pass arguments writes nothing
        return wrong
    fine_parent = None if fine_from is None else _resolve(fine_from)
    problem = None if os.environ.get("TOPOPT_MCP_CMD") else _python_problem(python_exe())
    if problem:
        return {"error": problem}

    from topopt import case
    brief_gate = case.gate(yaml, workspace(), proceed_without_brief, resume)   # reads files; a stop writes one log line, closes nothing
    if brief_gate.stop:
        return brief_gate.stop

    verifying = _alive_verify()
    if verifying:
        return {
            "error": f"A verify job is still running (pid {verifying[1]}, folder {verifying[0]}). "
            "Only one job can go at a time on this machine. Wait for it to finish (verify_status shows it). " + NO_STOP_VERIFY,
            "run_dir": str(verifying[0]),
            "pid": verifying[1],
        }
    ratcheting = _alive_ratchet()
    if ratcheting:
        return {
            "error": f"A ratchet job is still running (pid {ratcheting[1]}, folder {ratcheting[0]}). "
            "Only one job can go at a time on this machine. Wait for it to finish (ratchet_status shows it). " + NO_STOP_RATCHET,
            "run_dir": str(ratcheting[0]),
            "pid": ratcheting[1],
        }
    refusal = _resume_refusal(yaml, out, allow_changed_inputs, stop_rules) if resume else None   # reads only: no refusal closes the old dashboard
    continuing = None
    if parent is not None:   # reads only, and the parent is never written
        refusal, continuing = _continue_plan(yaml, out, parent, more, allow_changed_inputs, stop_rules)
    fining = None
    if fine_parent is not None:   # #318: reads only, and the parent is never written
        refusal, fining = _fine_plan(yaml, out, fine_parent, allow_changed_inputs, stop_rules, fine_voxel, fine_passes, fine_tol)
    alive = _alive_run()
    if alive:
        if _is_busy(alive[0]):
            return {
                "error": f"A run is still running (pid {alive[1]}, folder {alive[0]}). "
                "Only one run can go at a time on this machine. Wait for it to finish "
                "(run_status shows it; list_runs shows all runs). " + STOP_RUN_OPTION,
                "run_dir": str(alive[0]),
                "pid": alive[1],
            }
        # A refusal must not close the old dashboard: check the lock first. The old process is not a conflict.
        conflict = _lock_conflict(out, override_lock, ignore_pid=alive[1])
        if conflict:
            return conflict
        if refusal:
            return refusal
        _stop(*alive)  # old run is done: free port 8765 for the new run

    conflict = _lock_conflict(out, override_lock)
    if conflict:
        return conflict
    if refusal:
        return refusal

    port = _free_port(_port())
    if port != _port():
        log(f"port {_port()} is busy: the dashboard uses port {port}")
    cmd = _command(yaml, out, port, override_lock, resume, allow_changed_inputs, stop_rules,
                   **({} if parent is None else {"continue_from": parent, "more_iterations": more}),
                   **({} if fine_parent is None and not fine_pass else
                      {"fine_from": fine_parent, "fine_pass": fine_pass, "fine_voxel": fine_voxel, "fine_passes": fine_passes, "fine_tol": fine_tol}))
    earlier = None if resume else _earlier_checkpoint(out)   # before the new process deletes it
    resume_note = _resume_pass_keys(yaml, out) if resume and allow_changed_inputs else None   # before the process cuts the run folder
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # A run folder outside the work folder is the choice of the caller: another work folder would not help there.
        return {"error": f"Cannot create the run folder {out.parent}: {exc}. Give out_dir a folder that you can write."
                + (_how_to_change() if _in_work_folder(out.parent) else "")}
    made = _make_work_folder()
    if made:
        return {"error": made}
    stdout_path = _log_path(out)
    log(f"start: {' '.join(cmd)}")
    try:
        with open(stdout_path, "wb") as sink:
            proc = subprocess.Popen(
                cmd,
                cwd=str(workspace()),
                env=_child_env(),
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        return {"error": f"Could not start the run: {exc}"}
    from topopt.cli import fine_folder
    fine_dir = fine_folder(out) if fine_pass else None   # #318: the second folder of the one command
    _current.update(proc=proc, run_dir=out, started=time.time(), port=port, fine_run_dir=fine_dir,
                    fine_voxel=fine_voxel if fine_pass else None)   # a given fine voxel wins: the fine pass follows although the part needs no finer one
    _pid_file(out).write_text(str(proc.pid))
    _note_run_dir(out, proc.pid)
    facts = _yaml_assumptions(yaml)
    brief = case.run_started(brief_gate, yaml, workspace(), out, proc.pid, _current.get("started"))
    other_jobs = _other_server_jobs(out)
    return {
        **facts,
        **({"other_server_jobs": other_jobs} if other_jobs else {}),
        **({"earlier_checkpoint": earlier} if earlier else {}),
        **({"resume": resume_note} if resume_note else {}),
        **({"continue": continuing} if continuing else {}),
        **({"fine": fining} if fining else {}),
        **({} if fine_dir is None else _fine_pass_reply(out, fine_dir)),
        **brief,
        "run_dir": str(out),
        "dashboard_url": f"http://127.0.0.1:{port}/",
        "pid": proc.pid,
        "log_file": str(stdout_path),
        "yaml": str(yaml),
        "workspace": str(workspace()),
        "run_dir_rel": _rel(out),
        "log_file_rel": _rel(stdout_path),
        "yaml_rel": _rel(yaml),
        "message": no_callback("run_status", RUN_END),
    }


def _fine_pass_reply(out: Path, fine_dir: Path) -> dict:
    """The keys of the reply of run_topopt(fine_pass=true) (#318): the two folders, and the facts of the one command."""
    return {"fine_run_dir": str(fine_dir), "fine_run_dir_rel": _rel(fine_dir),
            "fine": {"first_run_dir": str(out), "fine_run_dir": str(fine_dir),
                     "facts": [f"fine pass: the first run goes to {out}. The fine pass goes to {fine_dir} when the part needs a finer voxel than "
                               "the run has (check_problem shows `fine_pass`) or fine_voxel is given and finer than the run; else the process ends "
                               "after the first run and says so in its log. "
                               "run_status and run_result take the folder that you want to read."]}}


def _pick_dir(run_dir: str | None) -> Path:
    if run_dir:
        return _resolve(run_dir)
    return _current.get("run_dir") or default_out()


# --------------------------------------------------------------------------- #
# A run that ended with no result: what is known, and what the user can do (P9-6, CONTRACTS.md section 10)
# --------------------------------------------------------------------------- #

DEAD_JOB_STATES = ("failed", "no_record", "stale_lock", "unfinished")   # job states of a run process that has ended, with no success
_ESTIMATE_LINE = re.compile(r"memory estimate: about ([0-9.]+) GiB(?: of ([0-9.]+) GiB RAM)?")   # the first log line of `topopt run` about memory
_ROW_LINE = re.compile(r"buckling row: on in the YAML \(.*?\): memory estimate about ([0-9.]+) GB with the row")   # the line with the buckling row
_MEMORY_WORDS = re.compile(r"MemoryError|Unable to allocate|out of memory|cannot allocate|bad_alloc", re.I)
_LOST_PEAK = "not recorded: the process did not reach its end"


def _ended_without_result(state: str, job: dict | None, pid: int | None, started: float | None) -> bool:
    """True if the run process of the folder has ended and left no result: an error, a failed job, a stale lock, or a stop of the process.

    A done run and a run that a request to stop ended (`stopped`) are not in it. A live process of this server works, or serves the
    dashboard: only the status `error` counts then. A busy status with no process and no job record counts only for a run of this server.
    """
    jstate = (job or {}).get("state")
    if state in ("done", "stopped") or jstate in ("running", "done", "stopped"):
        return False
    if state == "error":
        return True
    if pid:
        return False
    if jstate in DEAD_JOB_STATES:
        return True
    return state in BUSY_STATES and started is not None


def _signal_name(code: int | None) -> str | None:
    """The name of the signal that ended a process (`Popen.returncode` < 0), else None."""
    if code is None or code >= 0:
        return None
    try:
        return signal.Signals(-code).name
    except ValueError:
        return f"signal {-code}"


def _iso_epoch(text) -> float | None:
    """Epoch seconds of an ISO time text (`started_at` of job.json), or None."""
    import datetime
    try:
        return datetime.datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def _run_process(d: Path, job: dict | None) -> tuple[int | None, bool]:
    """`(pid, own)`: the pid of the run process of the folder, and whether this server started it. `(None, False)` if no pid is known.

    A run that this session did not start has the pid in its own job record. The pid file `<run>.pid` is the file of this server, and a
    run that the CLI started in the same folder later does not write it: it can name the pid of an older run (P9-6 review F1).
    """
    proc = _current.get("proc")
    if proc is not None and _is_current_dir(d):
        return proc.pid, True
    pid = (job or {}).get("pid")
    if isinstance(pid, int):
        return pid, False
    try:
        return int(_pid_file(d).read_text().strip()), False
    except (OSError, ValueError):
        return None, False


def _stop_mark(pid: int | None, d: Path, since: float | None) -> dict | None:
    """The mark that `_stop` set for this pid, if it is a stop of this run: the same run folder, and not before the start of the run.

    `since` is the start of the run (None if not known: then the folder and the pid are all that can be compared).
    """
    mark = _stopped.get(pid) if pid is not None else None
    if mark is None or os.path.realpath(mark["run_dir"]) != os.path.realpath(d):
        return None
    if since is not None and mark["at"] < since - 1:
        return None
    return mark


def _run_logs(d: Path, since: float | None) -> list[Path]:
    """The logs of this run, the newest first: the output file `<out>.stdout` of this server, and `run.log` of the folder.

    Each file counts only if it is not older than `since`, the start of the run. `run.log` is deleted when a run starts, so a run that
    died before that leaves the one of an earlier run. A run of the CLI does not write the output file, so a folder that a server run
    used before can hold an older one (P9-6 review F2). With `since` None nothing can be compared: the output file counts, `run.log` does not.
    If both count, the file with the later write time comes first; on a tie the output file, which also has what goes to stderr.
    """
    found = []
    for rank, path in enumerate((_log_path(d), d / "run.log")):
        try:
            mtime = path.stat().st_mtime if path.is_file() else None
        except OSError:
            mtime = None
        if mtime is not None and (mtime >= since - 1 if since is not None else rank == 0):
            found.append((-mtime, rank, path))
    return [path for _mtime, _rank, path in sorted(found)]


def _memory_estimate(logs: list[Path]) -> dict:
    """The memory estimate that `topopt run` writes to its log before it solves (no new computation), or `{"gib": None, "reason": ...}`.

    The last line of the newest log that has one: a resume does not clean `run.log`, so it can hold the lines of both runs.
    A run with the buckling row logs a second estimate, with the row: it is `with_buckling_row` (GB = 1e9 bytes, as the line says).
    """
    for path in logs:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        est = row = None
        for line in lines:
            if _ESTIMATE_LINE.match(line):
                est = line
            if _ROW_LINE.match(line):
                row = line
        if est:
            m = _ESTIMATE_LINE.match(est)
            out = {"gib": float(m.group(1)), "ram_gib": float(m.group(2)) if m.group(2) else None, "line": est, "source": str(path)}
            if row:
                out["with_buckling_row"] = {"gb": float(_ROW_LINE.match(row).group(1)), "line": row}
            return out
    if not logs:
        return {"gib": None, "reason": "no log of this run can be read"}
    return {"gib": None, "reason": "the log has no 'memory estimate' line: topopt run writes it after it has read the YAML and the regions, "
                                   "and the run ended before that"}


def _ram_now() -> dict:
    """RAM of this machine in GiB (the unit of the memory estimate line): the total, and the room now. A cheap read of the system."""
    try:
        from topopt.environment import _ram
        ram = _ram()
    except Exception as exc:   # a fact about the reply only: the run facts are valid without it
        return {"total_gib": None, "available_now_gib": None, "note": f"not read: {type(exc).__name__}: {exc}"}

    def gib(n):
        return None if n is None else round(n / 2**30, 1)
    return {"total_gib": gib(ram.get("total_bytes")), "available_now_gib": gib(ram.get("available_bytes")),
            "note": "available_now_gib is the room now, after the process ended: not the room at the time of the stop"}


def _build_failure_extras(d: Path, *, state: str, status: dict | None, hist: dict | None, job: dict | None, own: int | None,
                          pid: int | None, started: float | None, checkpoint: dict | None, summary: str,
                          earlier: bool = False) -> tuple[dict, str]:
    """`(keys, clause)` for a run that ended with no result: the keys that run_status and run_result add, and one clause for the summary.

    Facts first: `failure` (pid, exit code, signal, run data, whether this server stopped the process, peak memory, memory estimate, RAM),
    `log_tail` (the last lines of the log), `options` (only those that apply) and, for a stop by a signal (or an end with no record) with
    no message, `possible_causes`: the exit code of a stop by a signal does not say who stopped the process.
    The order of the causes follows facts of this reply only (no threshold): a stop that this server made is the cause and the only one;
    a positive memory fact (the log or the message names memory; the estimate of the run is above the RAM that the run saw) puts the
    memory cause first; without one, a stop by the user or another program comes first and the memory cause follows with its numbers.
    `summary` is the summary so far (the clause does not repeat the resume text).
    `earlier` (P9-18): the job was refused in a folder that holds an earlier run (`_refused_over_earlier_run`). `hist` and `checkpoint` are
    then the data of that earlier run, and the texts say so: `run_data`, the resume option and the dashboard note. The refused job writes
    no line to `run.log`, so that file is not a log of this job.
    """
    since = started if started is not None else _iso_epoch((job or {}).get("started_at"))
    logs = _run_logs(d, since)
    if earlier:
        logs = [path for path in logs if path != d / "run.log"]
    tail = _tail(logs[0]) if logs else None
    code = own if own is not None else ((job or {}).get("exit_code") if (job or {}).get("state") == "failed" else None)
    code = code if isinstance(code, int) else None
    sig = _signal_name(code)
    run_pid, own_handle = _run_process(d, job)
    mark = _stop_mark(run_pid, d, since)
    stopped_by_server = True if mark else (False if own_handle else None)
    message = (status or {}).get("message") if state == "error" else None
    peak = (job or {}).get("peak_rss_bytes")
    recorded = (job or {}).get("state") == "failed" and isinstance(peak, (int, float)) and peak > 0
    estimate = _memory_estimate(logs)
    ram = _ram_now()
    run_data = ("none" if status is None else f"up to pass {hist['iteration']}" if hist and "iteration" in hist and not earlier
                else "setup only: no pass is done")
    earlier_data = f"up to pass {hist['iteration']}" if earlier and hist and "iteration" in hist else None
    if earlier_data:
        run_data += f" (the passes in the folder, {earlier_data}, are of an earlier run: see earlier_run)"
    failure = {
        "pid": run_pid,
        "exit_code": code,
        "signal": sig,
        "run_data": run_data,
        "stopped_by_this_server": stopped_by_server,
        "peak_memory": (f"{peak / 1e9:.2f} GB ({peak / 2**30:.2f} GiB; job.json, {job.get('peak_rss_source')})" if recorded else
                        "not recorded: the process wrote no job record of its end" if code is not None and code >= 0 else _LOST_PEAK),
        "peak_memory_bytes": int(peak) if recorded else None,
        "memory_estimate": estimate,
        "ram": ram,
    }
    if code is None:
        failure["exit_note"] = ("the process is still alive; it ends after the error" if pid else
                                "not known: this server did not start the process in this session, and the process left no record of its end")
    if mark:
        failure["stopped_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mark["at"]))

    # A stop by a signal (or no record of the end) and no message: the exit code says nothing about the cause.
    unseen_end = code is None and not pid and (job or {}).get("state") != "failed"
    signal_stop = not message and (sig is not None or unseen_end)
    kill_stop = signal_stop and sig in (None, "SIGKILL") and stopped_by_server is not True   # the kind of end that a memory stop gives
    words = _MEMORY_WORDS.search((message or "") + "\n" + (tail or ""))
    over = None      # (what, GiB): an estimate of the run that is above the RAM that the run saw
    ram_run = estimate.get("ram_gib")
    if ram_run is not None:
        row = estimate.get("with_buckling_row")
        for what, gib in (("the memory estimate of the run", estimate["gib"]),
                          ("the memory estimate with the buckling row", row["gb"] * 1e9 / 2**30 if row else None)):
            if gib is not None and gib > ram_run:
                over = (what, round(gib, 1))
    positive = bool(words) or over is not None
    memory_possible = stopped_by_server is not True and (kill_stop or bool(words))

    resume = []
    if checkpoint and earlier:   # P9-18: the checkpoint is the earlier run's, and a resume goes on with that run
        where = "(yaml_path: earlier_run.checkpoint.yaml_path, out_dir: run_dir)"
        resume.append(f"run_topopt with resume=true {where} continues the earlier run (not the refused job) from its checkpoint at pass "
                      f"{checkpoint['iteration']} of {checkpoint['iterations']}" if not checkpoint["complete"] else
                      f"run_topopt with resume=true {where} runs the post step of the earlier run (not the refused job) only: the "
                      f"checkpoint of the earlier run at pass {checkpoint['iteration']} is complete")
    elif checkpoint:
        where = "(yaml_path: checkpoint.yaml_path, out_dir: run_dir)"
        resume.append(f"run_topopt with resume=true {where} continues the run from the checkpoint at pass {checkpoint['iteration']} of "
                      f"{checkpoint['iterations']}" if not checkpoint["complete"] else
                      f"run_topopt with resume=true {where} runs the post step only: the checkpoint at pass {checkpoint['iteration']} is complete")
    memory_options = []
    if memory_possible:
        memory_options = ["a larger voxel_size in the YAML (x 1.26 gives about half the voxels, x 2 about an eighth)",
                          "a solver with less memory (check_problem lists the solvers in `backends`; its `memory_estimate` is for the solver "
                          "of the YAML)"]
        if kill_stop and run_pid is not None and sys.platform.startswith("linux"):
            memory_options.append(f"on Linux the kernel log names a memory stop of pid {run_pid}: `journalctl -k` (or `dmesg`) has a line "
                                  f"'Killed process {run_pid}' if the kernel stopped it; this server does not read that log")
    read_log = ["read `log_tail` for the message of the run"] if tail and not signal_stop else []
    check = ["check_problem on the YAML before a new start: it shows the memory estimate against the RAM, and the warnings"]
    # With a positive memory fact the memory options come before check_problem; without one they come after it.
    options = resume + (memory_options + read_log + check if positive else read_log + check + memory_options)

    causes: list[tuple[str, str]] = []   # (short text for the summary, full text)
    if signal_stop and stopped_by_server is True:
        causes = [("a stop by this server", f"a stop by this server at {failure['stopped_at']} (run_topopt stops the process of an earlier "
                                            "run before it starts a new one)")]
    elif signal_stop:
        user = ("a stop by the user or another program", "a stop by the user or another program (a force quit, or `kill`)")
        crash = None
        memory = None
        if sig in (None, "SIGKILL"):
            facts = []
            if over:
                facts.append(f"{over[0]} ({over[1]} GiB) is above the RAM that the run saw ({ram_run} GiB)")
            elif estimate["gib"] is not None:
                facts.append(f"the memory estimate is {estimate['gib']} GiB")
            else:
                facts.append("the log has no memory estimate")
            if words:
                facts.append(f"the message or the log names memory ('{words.group(0)}')")
            total, free = ram.get("total_gib"), ram.get("available_now_gib")
            facts.append(f"this machine has {total} GiB RAM, {free} GiB available now" if total is not None and free is not None
                         else f"this machine has {total} GiB RAM" if total is not None else "the RAM is not known")
            facts.append("an estimate is a model and can be lower than the real peak")
            if words:
                short = "lack of memory (the log or the message names memory)"
            elif over:
                short = f"lack of memory ({over[0].replace('the memory estimate', 'estimate')} {over[1]} GiB, above the RAM of the run)"
            elif estimate["gib"] is not None and total is not None:
                short = f"lack of memory (estimate {estimate['gib']} GiB, RAM {total} GiB; an estimate can be low)"
            else:
                short = "lack of memory"
            memory = (short, f"the system stopped it for lack of memory ({'; '.join(facts)})")
        elif sig not in ("SIGTERM", "SIGINT", "SIGHUP"):
            crash = ("a crash in native code", f"a crash in native code (a library); {sig} is a fault of the process. "
                                               "`log_tail` can show the last output")
        causes = ([crash, user] if crash else [memory, user] if memory and positive else [user, memory] if memory else [user])

    keys: dict = {}
    if tail is not None:
        keys["log_tail"], keys["log_tail_file"] = tail, str(logs[0])
    keys["failure"] = failure
    keys["options"] = options
    if causes:
        keys["possible_causes"] = [full for _short, full in causes]
        if stopped_by_server is not True:
            keys["possible_causes_note"] = "The exit code does not say who stopped the process. The order of this list is not a ranking."
    if not pid:    # the dashboard is a thread of the run process: it ended with the process (cli.start_dashboard)
        keys["dashboard_note"] = ("the dashboard ran inside the run process, which has ended, so the link that run_topopt gave does not answer. "
                                  + (f"The run saved no pass: `topopt dashboard {d}` would show nothing." if hist is None else
                                     f"`topopt dashboard {d}` shows the passes of the earlier run in the folder ({earlier_data})."
                                     if earlier_data else f"`topopt dashboard {d}` shows the saved passes ({run_data})."))

    bits = []
    if causes:
        names = [short for short, _full in causes]
        label = "possible cause" if stopped_by_server is True else "possible cause (the exit code does not say)"
        bits.append(f"{label}: " + (", ".join(names[:-1]) + ", or " + names[-1] if len(names) > 1 else names[0]))
    elif tail:
        last = next((ln for ln in reversed(tail.splitlines()) if ln.strip()), None)
        if last:
            bits.append(f'last log line: "{last.strip()}"')
    first = options[0]
    if not (first.startswith("run_topopt with resume=true") and "resume=true" in summary) and not (bits and first.startswith("read `log_tail`")):
        bits.append("first option: " + first)
    return keys, ("; " + "; ".join(bits)) if bits else ""


def _failure_extras(d: Path, **kw) -> tuple[dict, str]:
    """`_build_failure_extras`, with a fault in it as a fact: `failure.error`. The rest of the reply of run_status is valid without it."""
    try:
        return _build_failure_extras(d, **kw)
    except Exception as exc:
        return {"failure": {"error": f"the failure facts could not be read: {type(exc).__name__}: {exc}"}}, ""


def _refused_over_earlier_run(d: Path, status: dict | None, job: dict | None) -> bool:
    """True if `status` is the status of a job that an input check refused in a folder that holds an earlier run (F4-43, P9-18).

    The rule is `topopt.earlier_run.refused_over_earlier_run` (CONTRACTS.md section 24.1): `list_runs`, `run_history` and `compare_runs` use it too (P9-19).
    """
    from topopt import earlier_run
    return earlier_run.refused_over_earlier_run(d, status, job)


def _earlier_run_block(total, hist: dict | None, verdict: dict | None, checkpoint: dict | None, history_facts: list[str]) -> dict:
    """The reply key `earlier_run` (P9-18): the facts of the run that a refused job left in the folder, and a note that says whose they are.

    The same facts are not given at the top level of the reply: there `last_history`, `verdict` and `checkpoint` are null, as for any job
    that has no data. `summary` is one line for the summary text of the reply.
    """
    bits = []
    if hist and "iteration" in hist:
        bits.append(f"pass {hist['iteration']} of {total}" if total else f"pass {hist['iteration']}")
        if "volume_fraction" in hist:
            bits.append(f"volume {hist['volume_fraction'] * 100:.1f}%")
    if verdict:
        bits.append(f"verdict {verdict.get('state', '?')}")
    if checkpoint:
        bits.append(f"checkpoint at pass {checkpoint['iteration']} of {checkpoint['iterations']} "
                    f"({'complete' if checkpoint['complete'] else 'not complete'})")
    return {
        "note": ("The input check refused this job before it wrote a run file. The folder still holds an earlier run, and this job did not "
                 "change it. The facts in this block are of that earlier run. `status`, `job` and `failure` are of the refused job. "
                 "`last_history`, `verdict` and `checkpoint` at the top of the reply are null for the refused job. "
                 "run_topopt with resume=true continues the earlier run."),
        "summary": ", ".join(bits) or "no pass data",
        "last_history": hist,
        "verdict": verdict,
        "checkpoint": checkpoint,
        "history_facts": history_facts,
    }


def _failed_run_reply(st: dict) -> dict | None:
    """The reply of run_result for a run that ended with no result: the facts of the reply `st` of run_status. None if it has none."""
    if "failure" not in st:
        return None
    note = "`failure`, `log_tail` and `options` say what is known."
    message = (f"The run process has stopped while the status of the run was `{st['state']}`: there is no result. " + note
               if st["state"] in BUSY_STATES else "The run ended with no result. " + note)
    reply = {"done": False, "state": st["state"], **({"earlier_run": st["earlier_run"]} if "earlier_run" in st else {}),   # P9-18
             "run_dir": st["run_dir"], "run_dir_rel": st["run_dir_rel"], "summary": st["summary"], "message": message}
    for key in ("log_tail", "log_tail_file", "failure", "options", "possible_causes", "possible_causes_note", "dashboard_note", "checkpoint"):
        if key in st:
            reply[key] = st[key]
    return reply


@mcp.tool()
def run_status(run_dir: str | None = None) -> dict:
    """Show the state of a run: status, last iteration, and a one-line summary.

    Call it again to watch a run. Each call returns at once. Nothing calls you back when the run ends: call it again in this turn until the state is done, error or stopped, or process_alive is false, or end the turn and tell the user what to ask for the result. A promise of a later report cannot be kept. The numbers are facts; you decide what to do next.
    run_dir: the run folder. Default: the last run that this server started.
    `job` has the facts of the run process: state (running, done, stopped, failed with its exit code, or stale_lock when the process stopped without a marker), wall time and peak memory. `history_facts` names lines that were skipped.
    `state` is `stopped` for a run that a request to stop (stop_run, `topopt stop`) ended at the end of a pass: the checkpoint is saved, there is no result/, and run_topopt with resume=true continues it. A process of this server can still be alive after that: it only serves the dashboard (`process_alive` true, `job.state` stopped). `stop_request` is the request in the run folder (null if none): while the job is running it means the run stops at the end of its current pass.
    `checkpoint` (null if the folder has none, or the state is starting, error before any data, or none) gives the pass of the last checkpoint, `complete` (the run reached its last pass or a stop rule) and `yaml_path`: run_topopt with resume=true continues a run that stopped before its last pass.
    `verdict` (null until the run ends) has the `state` (converged, iteration_limit, stopped_by_rule, stopped_by_request, unfinished), whether the constraint rows are met (`feasible`), the measures with their thresholds, the stop rule that fired, and the resume points. The same verdict is in `metrics.json` (run_result).
    `verdict.facts` has the facts of the last pass, and the verdict text ends with "Facts: ..." for each fact that the run has: the volume used (the projected design, and the thresholded part at density 0.5; only with a volume row); the load path (which load regions the thresholded part does not join to a fixed, spring or joint region; in the text only when a region is not joined); for a run with objective mass, the `material` (volume and mass of the thresholded part; only with no volume row) and, when the grayness is above `grayness_max`, the `grey` design (volume and body count of the thresholded part at the cuts 0.4, 0.5 and 0.6, and the volume spread between 0.4 and 0.6 as a share of the volume at 0.5). They are facts: no test uses them. The run log has them at the end of each beta stage. The body count of final.stl comes after the post step: `metrics.grey_design` (see run_result).
    A run that ended with no result also has `log_tail`, `failure` (facts: pid, exit code, signal, run data, whether this server stopped it,
    peak memory, estimate, RAM), `options` and, after a signal or an unseen end with no message, `possible_causes` (not a verdict).
    `fine_run` (the first folder of run_topopt with fine_pass=true): the folder and the state of the fine pass.
    `time_left` (while the run is in its passes, from pass 2 of the run process) has `left_s` and `up_to_s` (seconds left if no stop rule or request to stop ends the run earlier) and `basis`. Else it is null, with the reason in `time_left_note`.
    """
    from topopt.runjob import read_stop, stop_text
    d = _pick_dir(run_dir)
    status = _read_json(d / "status.json")
    pid = _is_running(d)
    started = _current.get("started") if _is_current_dir(d) else None
    own = _own_exit(d)             # exit code of the process that this server started here, if it has ended
    job = _job_facts(d, started, own)   # the run helper: running, done, FAILED with the exit code, or a stale lock
    stale = False
    if status is not None and started and (pid or own is not None):
        try:
            mtime = (d / "status.json").stat().st_mtime
            # An ended process that wrote no job record wrote nothing: a status file from before its start is old.
            stale = mtime < started - 1 or (job is not None and job["state"] == "no_record" and mtime < started)
        except OSError:
            pass
    if status is None or stale:
        if pid:
            return {"run_dir": str(d), "run_dir_rel": _rel(d), "state": "starting",
                    "process_alive": True, "job": job, "history_facts": [], "verdict": None, "checkpoint": None,
                    "stop_request": read_stop(d), "summary": "starting, no data yet"}
        if own is not None:    # the new run process has ended, and the folder holds no data of it: do not show the old run
            summary = f"error: the run process {_exit_text(own)} before it wrote any run data (see {_log_path(d)})"
            extras, clause = _failure_extras(d, state="error", status=None, hist=None, job=job, own=own, pid=None, started=started,
                                             checkpoint=None, summary=summary)
            return {"run_dir": str(d), "run_dir_rel": _rel(d), "state": "error",
                    "process_alive": False, "job": job, "history_facts": [], "verdict": None, "checkpoint": None,
                    "stop_request": read_stop(d), "summary": summary + clause, **extras}
        if status is None:
            summary = f"no run found in {d}" + (f"; {job['summary']}" if job else "")
            extras, clause = {}, ""
            if _ended_without_result("none", job, pid, started):   # a job that ended and left no status file
                extras, clause = _failure_extras(d, state="none", status=None, hist=None, job=job, own=None, pid=None, started=started,
                                                 checkpoint=None, summary=summary)
            return {"run_dir": str(d), "run_dir_rel": _rel(d), "state": "none",
                    "process_alive": False, "job": job, "history_facts": [], "verdict": None, "checkpoint": None,
                    "stop_request": read_stop(d), "summary": summary + clause, **extras}
    hist = _last_history(d)
    run = _read_json(d / "run.json") or {}
    total = run.get("iterations")
    state = status.get("state", "?")
    it = status.get("iteration", 0)
    refused = _refused_over_earlier_run(d, status, job)   # P9-18: hist, run.json, verdict and checkpoint are then of the earlier run
    parts = [f"iteration {it}/{total}" if total and not refused else f"iteration {it}",
             _clock(status.get("elapsed_s", 0.0))]
    if hist and "volume_fraction" in hist and not refused:
        parts.append(f"volume {hist['volume_fraction'] * 100:.1f}%")
    from topopt import eta   # E1-P2: the live time estimate; a process that has stopped has none
    # A run that this server did not start has no `started`: the job record then says that its process has ended (a kill leaves `running`).
    ended = state in BUSY_STATES and not pid and (started is not None or eta.job_ended(job))
    time_left, time_left_note = ((None, "the run process has stopped") if ended
                                 else eta.run_time_left(lambda: eta.read_rows(d), run, status))
    if time_left:
        parts.append(eta.left_text(time_left))
    summary = ", ".join(parts)
    if state == "done":
        summary = "done: " + summary
    elif state == "error":
        summary = f"error: {status.get('message', '')} ({summary})"
    elif state == "stopped":
        summary = f"stopped: {status.get('message', '')} ({summary})"
    elif state == "postprocess":
        summary = "post-process: " + summary
    # A done run with a live process (it serves the dashboard) is normal. A busy status with no process: a run of this server, or a run whose
    # job record says that its process has ended (a CLI run that ended natively in the post step leaves `postprocess`, #448).
    if ended:
        summary += " (warning: the process has stopped)"
    history_facts = _history_facts(d)
    from topopt.verdict import read_verdict
    verdict, checkpoint = read_verdict(d), _checkpoint_facts(d, run)
    earlier_run = None
    if refused:   # P9-18: the facts move to `earlier_run`; the summary names them as the earlier run's, with the verdict state only
        earlier_run = _earlier_run_block(total, hist, verdict, checkpoint, history_facts)
        summary += (f"; {job['summary']}" if job else "") + "; earlier run in this folder (this job did not change it; see `earlier_run`): "
        summary += earlier_run["summary"] + "".join(f"; {fact}" for fact in history_facts)
    else:
        summary += "".join(f"; {fact}" for fact in history_facts) + (f"; {job['summary']}" if job else "")
        if verdict:
            summary += "; verdict: " + verdict["text"]
    if state == "done":   # I7: a run that asked for a mass in grams: the final mesh against the asked mass (a fact; the same words as the run log)
        from topopt.volume_basis import mass_asked_line
        asked_line = mass_asked_line(_read_json(d / "result" / "metrics.json"))
        if asked_line:
            summary += "; " + asked_line.removeprefix("fact: ")
    running = job is not None and job.get("state") == "running"   # a CLI run is alive with no pid file of this server
    # The job state says whether the run works: a process of this server that only serves the dashboard (a done or a stopped run) is alive too.
    # A live process and no job record (a build from before the run helper) can still be a run at work.
    if checkpoint and not running and (job is not None or not pid):
        if not checkpoint["complete"]:
            summary += (f"; checkpoint at pass {checkpoint['iteration']} of {checkpoint['iterations']}: "
                        f"run_topopt with resume=true continues the {'earlier run' if refused else 'run'}")
        elif state == "stopped":
            summary += (f"; checkpoint at pass {checkpoint['iteration']} of {checkpoint['iterations']} is complete: "
                        "run_topopt with resume=true runs the post step only")
    stop_request = read_stop(d)
    if stop_request is not None and running:
        summary += (f"; a request to stop is pending ({stop_text(stop_request)}): the run stops at the end of its current pass, "
                    "after it saves the checkpoint")
    extras = {}
    if _ended_without_result(state, job, pid, started):   # P9-6: the process has ended and left no result
        extras, clause = _failure_extras(d, state=state, status=status, hist=hist, job=job, own=own, pid=pid, started=started,
                                         checkpoint=checkpoint if checkpoint and not running else None, summary=summary,
                                         earlier=refused)
        summary += clause
    port = _current.get("port") if _is_current_dir(d) else None
    return {
        "run_dir": str(d),
        "run_dir_rel": _rel(d),
        "dashboard_url": f"http://127.0.0.1:{port}/" if port else None,
        "state": state,
        **({"earlier_run": earlier_run} if earlier_run else {}),   # P9-18: the data keys below are null for the refused job
        "process_alive": bool(pid),
        "status": status,
        "last_history": None if earlier_run else hist,
        "time_left": time_left,
        "time_left_note": time_left_note,
        "job": job,
        "history_facts": [] if earlier_run else history_facts,
        "verdict": None if earlier_run else verdict,
        "checkpoint": None if earlier_run else checkpoint,
        "stop_request": stop_request,
        "summary": summary,
        **({"fine_run": _fine_run_block(d)} if _fine_run_block(d) else {}),
        **extras,
    }


@mcp.tool()
def run_result(run_dir: str | None = None, preview: bool = True) -> dict:
    """Get the final metrics, the path of final.stl and a text preview when the run is done.

    Tell the user the result: mass, stiffness, run time. Units: mm, N, MPa, g.
    A run that a request to stop ended has no result: the reply is `done: false` and `stopped: true` with the pass.

    run_dir: the run folder. Default: the last run that this server started.
    preview: add mesh facts and ASCII silhouettes of final.stl (for clients that cannot open files). render_part gives real images of the part.
    Reply keys: resource topopt://replies/run_result (the meaning of each key).
    A note names a wall thinner than the voxel size. The thinnest wall (`thinnest_wall_mm`) is the lower end of the film, not a value to use.
    The setting `settings.part_only_min_film_mm` (the wall thickness to cut off, in mm; default off) cuts off that thin film; then `part_only` also has `min_film_mm`, `film_removed_mm3` and, for each body, `thin_area_mm2`, `cut_area_mm2`, `thin_area_after_mm2` and the thinnest wall that stays. To apply it to a run that has finished: write the YAML with the value as a NEW file in the folder of the old YAML (region paths start there), then run_topopt with that YAML, the run folder as out_dir, resume=true and allow_changed_inputs=true (the post step repeats; copy the run folder first to keep the old result). Read the three areas and change the value if the film is still there.
    For a run with a stress limit, "met" of `metrics.stress.cases.<case>.relaxed` is the relaxed measure of the optimizer, not the stress of the part. `summary` and `independent_check` of the case give the P2 p99.9 of verify_run and its ratio to the limit when verify has run on the folder (read again at each call), else the option to run it.
    """
    d = _pick_dir(run_dir)
    st = run_status(str(d))   # one reading of the folder: run_status knows a status.json that is older than the run (P9-6 review F6)
    state, status = st["state"], st.get("status") or {}
    if state == "stopped":   # a request to stop ended the run: no post step, no result (a process can still serve the dashboard)
        return {"done": False, "stopped": True, "run_dir": str(d), "run_dir_rel": _rel(d),
                "message": f"The run was {status.get('message') or 'stopped by a request'}. result/ is not made. "
                           "run_topopt with resume=true continues the run (run_status shows the checkpoint).",
                "levers": _levers_block(d)}
    if _is_running(d) and state != "done":
        return {"done": False, "message": "The run is not done yet. Call run_status."}
    failed = _failed_run_reply(st)   # P9-6: a run that ended with no result returns the facts of run_status, not "Call run_status"
    if failed:
        if state == "error" and status:   # a status of this run: its message stays
            failed["message"] = f"The run failed: {status.get('message', '')}"
        return failed
    metrics = _read_json(d / "result" / "metrics.json")
    stl = d / "result" / "final.stl"
    if metrics is None or not stl.is_file():
        return {"done": False, "message": "No result yet. Call run_status."}
    from topopt.stress_report import refresh_from_verify
    refresh_from_verify(metrics, d)   # F4-29: the P2 value next to "met" of a stress row is the one of verify/verify.json now (in memory; no file is written)
    res = {"done": True, "metrics": metrics, "final_stl": str(stl.resolve()),
           "final_stl_rel": _rel(stl), "run_dir": str(d), "run_dir_rel": _rel(d),
           "workspace": str(workspace())}
    run_json = _read_json(d / "run.json") or {}
    lost = run_json.get("lost_force")
    if lost is not None:
        res["lost_force"] = lost
    if run_json.get("part_voxel") is not None:   # #318: the voxel that the part needs, and the fact when the run voxel is too coarse for it
        res["fine_pass"] = run_json["part_voxel"]
    if run_json.get("fine_from") is not None:    # #318: the run is a fine pass: the record of the parent, the voxel, the maximum and the tolerance
        res["fine_from"] = run_json["fine_from"]
    iface_warn = ((metrics.get("interfaces") or {}).get("warnings")) or []   # check C, D and E of CONTRACTS.md section 23
    if iface_warn:
        res["interface_warnings"] = list(iface_warn)
    from topopt import part_card
    fb_lines = part_card.mesh_fallback_lines(metrics)   # #445: a fallback in the mesh step, in any round of the volume calibration
    if fb_lines:
        res["mesh_fallbacks"] = fb_lines
    nominal = d / "result" / "nominal.stl"   # only with machining stock (CONTRACTS.md section 23)
    if nominal.is_file():
        res["nominal_stl"], res["nominal_stl_rel"] = str(nominal.resolve()), _rel(nominal)
    part_only = d / "result" / "part_only.stl"   # only with regions.not_printed (CONTRACTS.md section 9.1); metrics.part_only has the facts
    if part_only.is_file():
        res["part_only_stl"], res["part_only_stl_rel"] = str(part_only.resolve()), _rel(part_only)
    from topopt import printcheck   # the print check that the post step ran (CONTRACTS.md section 32): the states, the file has the report
    pc = printcheck.summary_of(_read_json(d / "result" / printcheck.REPORT_FILE))
    if pc is not None:
        res["print_check"] = {**pc, "file": str((d / "result" / printcheck.REPORT_FILE).resolve()),
                              "file_rel": _rel(d / "result" / printcheck.REPORT_FILE)}
    end_fact = _end_fact(metrics.get("verdict"), metrics)
    if end_fact:
        res["end_fact"] = end_fact
    res.update(_card_block(d))
    if preview:
        res["preview"] = _mesh_preview(stl)
    res["levers"] = _levers_block(d)
    return res


def _end_fact(verdict, metrics: dict) -> str | None:
    """The key `end_fact` of run_result (CONTRACTS.md section 10, #380): the sentence of the part card about how the run ended, for the verdict
    state `iteration_limit` only, so that a client that never read the start-here skill still says "the run stopped at its pass limit". None
    for every other state (the reply keeps its keys)."""
    if not isinstance(verdict, dict) or verdict.get("state") != "iteration_limit":
        return None
    from topopt import part_card
    return part_card.result_sentence(verdict, verdict.get("iterations", metrics.get("iterations")))


def _is_shipped_recording(d: Path) -> bool:
    """True for a run folder under `<code folder>/recordings/` (the replay backup, committed). Not for the rest of the code folder:
    the runs of a user can live inside a clone."""
    try:
        return d.resolve().is_relative_to((REPO / "recordings").resolve())
    except OSError:
        return False


def _card_block(d: Path) -> dict:
    """The keys `report_md` and `report_md_rel` of run_result (CONTRACTS.md section 9.2): the part card, written again with the facts the folder has now.

    `topopt run` writes the card at its end (after the post step and the buckling check). A fact file that comes later (verify, a later buckling, renders, STEP) shows after this call.
    The file is not touched when its bytes do not change. A write error is a note (`report_md_note`); an older card is still named.
    A run folder under `<code folder>/recordings/` (the shipped recordings, which are committed) gets no write: the note names `topopt report`.
    """
    from topopt import part_card
    path, res = part_card.card_path(d), {}
    if _is_shipped_recording(d):
        res["report_md_note"] = ("this run folder is a shipped recording (the folder `recordings/` of the code): the part card is not written here. "
                                 "Copy the folder to a folder that you can write, then run `topopt report <that folder>`")
    else:
        try:
            part_card.write(d)
        except Exception as exc:
            res["report_md_note"] = f"the part card could not be written again: {type(exc).__name__}: {exc}"
    if path.is_file():
        res["report_md"], res["report_md_rel"] = str(path.resolve()), _rel(path)
    return res


def _levers_block(d: Path) -> dict:
    """The block `levers` of run_result (CONTRACTS.md section 41.10): the facts of the run folder that the engine flags, with the cards that name them.

    A fault in it is a fact in the block (`error`); the result of the run is still valid.
    """
    try:
        from topopt import levers_run
        return levers_run.compact(levers_run.run_levers(d))
    except Exception as exc:
        return {"entries": [], "error": f"the lever cards of this run could not be read: {type(exc).__name__}: {exc}"}


def _starting_pid(d: Path) -> int | None:
    """The pid of the run process that this server started for this folder, if it is alive and has not yet written run data of its own.

    Such a process is starting: it may not hold the job lock yet, so the job files cannot show it. A live process that has written
    status.json since its start holds the lock, or has ended its job (it then only serves the dashboard): `_is_busy` tells which.
    """
    pid = _is_running(d)
    return pid if pid and _is_busy(d) else None


def _live_runs() -> list[dict]:
    """Run folders in the work folder (and runs/) where a live process holds the job lock: `[{run_dir, pid, started_at}]`.

    Only run folders count (`is_run_folder`, the rule of `list_runs`): the folder of a ratchet job also holds a job lock, and it is not a run.
    This finds a run that the CLI started (no pid file of this server). A run that has only just taken the lock and has not written
    status.json yet is not found; the next call finds it.
    """
    from topopt.runjob import is_run_folder, live_holder
    found = []
    for c in _scan_folders(lambda c: is_run_folder(c) and (c / "job.lock").is_file()):
        held = live_holder(c)
        if held:
            found.append({"run_dir": str(c), "pid": held["pid"], "started_at": held["started_at"]})
    return found


@mcp.tool()
def stop_run(run_dir: str | None = None) -> dict:
    """Ask a running run to stop at the end of its current pass. Returns at once; sends no signal; waits for nothing.

    The tool writes the file `job.stop` in the run folder. The run then ends with exit code 5. There is no post step and no result/. The history, the snapshots and the checkpoint are those of a run that goes on, so run_topopt with resume=true (no allow_changed_inputs) gives the same result as a run that was not stopped.
    run_dir: the run folder. Default: the run that this server started and that still works; else the one run in the work folder (or its runs/ folder) that a live process holds the job lock of (this finds a run that the CLI started). If several runs are live, the reply lists them and writes nothing: give run_dir.
    The reply has `state`:
    stop_requested: the request is written. `pid`, `iteration` (the pass at the last status), `message`, `options`.
    already_requested: a request is in the folder already (`stop_request` has its time); it is not written again.
    postprocess: the run is in the post step or the refine step: all passes are done, nothing is written. Options: wait, or end the process yourself (resume=true then repeats the post step only).
    nothing_to_stop: the run is done, stopped or failed already: nothing is written. `process_alive` says whether a process of this server is alive and only serves the dashboard. `options` has the resume route where a checkpoint is there.
    gone: a lock whose process is not running, or a record that says running and has no lock: nothing is written. `checkpoint` and the option resume=true (it replaces the stale lock, with a fact). Also the run process of this server that ended before it took the job lock: the message has its exit and the log file.
    none: no run in this folder: no job files and no live process, or the folder does not exist, or a live job lock of a job that is not a topopt run (the message names the marker, for example ratchet.json). Nothing is written.
    several: several live runs: the list is in `runs`.
    Every reply has `note`: a topopt build from before this change does not read job.stop, and a run of such a build ignores the request. `found_by` says how the run was found (run_dir, this server, the lock scan).
    The run reads job.stop after the checkpoint of each pass (one file read per pass, no polling). It then saves its verdict (state `stopped_by_request`), writes the status `stopped` and the job marker `job.STOPPED`, and releases the job lock. A pass that is long (a big grid, a buckling row) delays the stop by up to one pass. A solve that hangs never reads the request: the reply has the pid.
    With run_topopt fine_pass=true and no run_dir, the fine pass is stopped when it runs; while the first run works, that run is stopped and the fine pass does not start. From the post step of the first run until the fine pass has started, the request is written in the fine folder (state stop_requested, run_dir the fine folder): the process reads it before the fine pass starts, ends without starting it (exit code 5) and says so in the log.
    Only run folders count for run_dir: a folder with status.json or run.json, as in list_runs. The folder of a ratchet job holds a job lock too, and it is not a run.
    """
    from topopt.runjob import STOP_NOTE, stop_reply
    found_by = "the run_dir argument"
    if run_dir:
        d = _resolve(run_dir)
    else:
        alive = _alive_run()
        if alive and _is_busy(alive[0]):
            d, found_by = alive[0], "the run that this server started"
            before = _fine_not_started(alive[0])
            if before is not None:   # fine_pass=true: the fine pass has not started (the post step of the first run, or the gap and the plan after it)
                return _stop_before_fine(alive[0], before[0], before[1], alive[1])
            fine = _fine_target(alive[0])
            if fine is not None:   # fine_pass=true: the first run is done, the run that works now is the fine pass
                d, found_by = fine, "the fine pass of fine_pass=true that this server started"
        else:
            live = _live_runs()
            if len(live) > 1:
                return {"state": "several", "runs": live, "stop_request": None, "run_dir": None, "pid": None,
                        "message": f"{len(live)} runs hold a live job lock. Nothing is written. Give run_dir to pick the run.",
                        "options": [f"stop_run(run_dir={r['run_dir']!r}) for pid {r['pid']}" for r in live],
                        "note": STOP_NOTE}
            if live:
                d, found_by = Path(live[0]["run_dir"]), "the scan for a live job lock"
            else:
                d, found_by = (alive[0] if alive else _pick_dir(None)), "the last run of this server" if alive else "the default run folder"
    reply = stop_reply(d, by="mcp stop_run", starting_pid=_starting_pid(d))
    own = _own_exit(d)
    if reply["state"] == "none" and reply["job"] is None and own is not None:
        # the run process of this server ended and left no job files: it ended before it took the job lock
        reply.update(state="gone", message=f"The run process of this server {_exit_text(own)} before it took the job lock, and the folder has no job "
                     f"files (log: {_log_path(d)}). Nothing is written.",
                     options=[f"read the log {_log_path(d)} for the reason", "run_topopt starts a new run in this folder"])
    pid = _is_running(d)
    reply.update(run_dir_rel=_rel(d), found_by=found_by, process_alive=bool(pid))
    if pid and reply["state"] in ("nothing_to_stop", "gone"):
        reply["message"] += (f" A process of this server (pid {pid}) is alive: it only serves the dashboard and holds no job lock "
                             "(it ends with the dashboard, or when the next run_topopt replaces it).")
    return reply


def _mesh_preview(stl: Path, cols: int = 48, samples: int = 30000) -> dict:
    """Mesh facts and three ASCII silhouettes (top x-y, front x-z, side y-z).

    No renderer is used: surface points of the mesh fall into a coarse grid.
    """
    try:
        import numpy as np
        import trimesh
        mesh = trimesh.load_mesh(str(stl), file_type="stl")
        v = np.asarray(mesh.vertices, float)
        f = np.asarray(mesh.faces)
        tri = v[f]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(f), size=samples, p=area / area.sum()) if area.sum() > 0 else []
        r = rng.random((len(pick), 2))
        flip = r.sum(axis=1) > 1
        r[flip] = 1 - r[flip]
        t = tri[pick]
        pts = t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0])
        lo, hi = v.min(axis=0), v.max(axis=0)
        size = hi - lo
        scale = max(float(size.max()), 1e-9) / cols  # mm per column; a row is twice as tall

        def view(a: int, b: int) -> list[str]:
            w = max(1, int(np.ceil(size[a] / scale)))
            h = max(1, int(np.ceil(size[b] / (2 * scale))))
            ix = np.minimum(((pts[:, a] - lo[a]) / scale).astype(int), w - 1)
            iy = np.minimum(((pts[:, b] - lo[b]) / (2 * scale)).astype(int), h - 1)
            g = np.zeros((h, w), bool)
            g[iy, ix] = True
            return ["".join("#" if c else "." for c in row) for row in g[::-1]]

        return {
            "kind": "ascii-silhouettes",
            "note": "Coarse silhouettes of surface points (# = surface seen in this view, rows are "
                    "flipped so +up is first). The tool render_part gives real images.",
            "triangles": int(len(f)),
            "bbox_mm": {"min": [round(float(x), 2) for x in lo], "size": [round(float(x), 2) for x in size]},
            "watertight": bool(mesh.is_watertight),
            "volume_mm3": round(float(mesh.volume), 1) if mesh.is_watertight else None,
            "views": {"xy_top": view(0, 1), "xz_front": view(0, 2), "yz_side": view(1, 2)},
        }
    except Exception as exc:  # a preview must not hide the result
        return {"kind": "none", "note": f"No preview: {type(exc).__name__}: {exc}"}


LIST_RUNS_EARLIER_KEYS = ("iterations", "problem", "final_stl_rel")   # the keys of a `list_runs` row that run.json and result/ give (P9-19)


def _run_summary(d: Path) -> dict:
    status = _read_json(d / "status.json")
    run = _read_json(d / "run.json") or {}
    state = "unreadable" if status is None and (d / "status.json").exists() else None
    status = status or {}
    files = [d / n for n in ("status.json", "run.json", "history.jsonl")]
    mtimes = []
    for f in files:
        try:
            mtimes.append(f.stat().st_mtime)
        except OSError:
            pass
    stl = d / "result" / "final.stl"
    mt = max(mtimes) if mtimes else d.stat().st_mtime
    row = {
        "name": d.name,
        "path": _rel(d),
        "run_dir": str(d),
        "state": state or status.get("state", "unknown"),
        "iteration": status.get("iteration"),
        "iterations": run.get("iterations"),
        "problem": run.get("name"),
        "process_alive": bool(_is_running(d)),
        "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mt)),
        "mtime": mt,
        "final_stl_rel": _rel(stl) if stl.is_file() else None,
    }
    from topopt import earlier_run
    if earlier_run.refused_or_false(d, status):   # P9-19: the keys that come from run.json and result/ are of the earlier run
        row = earlier_run.mark_row(row, LIST_RUNS_EARLIER_KEYS)
    return row


@mcp.tool()
def render_part(run_dir: str | None = None, stl_paths: list[str] | None = None,
                roles: list[str] | None = None, views: list[str] | None = None,
                section: str | None = None, size: str | None = None,
                out_dir: str | None = None, regions: str | None = None, max_triangles: int | None = None,
                inline: bool = True, yaml_path: str | None = None,
                loads: bool | None = None) -> list[TextContent | ImageContent]:
    """Render a part to PNG images so you can look at it. No display or GL is needed (CONTRACTS.md section 15).

    A look before the first run and after each run finds a wrong region or load fast.

    Input: stl_paths (STL files; roles: one role for all or one per file, from part, design, fixed, load, keep-out; default part; an item may be a comma list), or run_dir (renders result/final.stl and the region STLs that run.json names; missing region files are listed in the facts), or yaml_path (a problem YAML, before a run: the region STLs and the interface shapes. Fixed, load and union shapes are solid. The design space, keep-out regions and cut shapes are ghosts. A face is its cut zone, clipped to the grid box. Look at it before a run: it shows what a cut takes). With none of them, the last run is used. A YAML render also draws the force arrows (one for each force of each load case, a color for each case, along the force, ending where it acts): you can show it to the user to check the directions before the run.
    The answer is a JSON text and, when inline is true, the montage as one image. With inline false the answer is the text only; the PNG files are on disk and a client that can open files reads them.
    A 4-view render of an 85,000-triangle part takes about 2 to 4 s; the first call with an empty numba cache adds under 1 s. Later renders overwrite PNGs with the same names; out_dir picks another folder.
    out_dir: default <run>/renders, or renders beside the first STL.
    max_triangles: refuse meshes with more triangles than this (default: no limit; the facts give a memory estimate, about 450 MB plus 450 MB per million triangles).
    Reply keys: resource topopt://replies/render_part (the meaning of each key). The facts of each mesh have watertight and winding_consistent.
    loads: false draws no force arrow in a YAML render (default: arrows). A moment and a pressure get no arrow, and the facts say so. The arrow length is a share of the part size: it does not show the force size.
    views: names from iso, front, back, right, left, top, bottom (default iso, front, right, top); NAME:dx,dy,dz gives your own direction from the part to the camera, z up (name: letters, digits, _, -).
    section: "x=30" (or "x>=30") keeps x >= 30; "x<=30" keeps x <= 30; the axis is x, y or z, in mm.
    regions: how the non-part meshes are drawn. ghost (default with a part): design and keep-out as faint ghosts, fixed and load as strong ghosts, so the part stays visible. solid (default without a part). off: the part only.
    size: one tile in pixels, WxH (default 640x480). The montage is at most 1568 px on the long edge. All tiles have one scale; a scale bar shows it.
    In views, one item may be "fl:-0.5,-1,0.25". A text like "iso,fl:-0.5,-1,0.25,top" in one item is read as three views.
    Cut faces are flat orange. With a section and no views, the views are: cut (straight at the cut face from the removed side), iso-cut, and two side views of the kept part. The facts give cut_face_visible_px for each view; they say if a mesh could not be cut (for example, it is not a closed solid), and the render is made anyway.

    Colors: part grey-blue, design space light blue, fixed dark blue, load red, keep-out brown, cut face orange; for a YAML also union shape green and cut shape (a ghost) pink-red. Back faces are magenta: they show a flipped or open surface. The same input gives the same PNG bytes.
    """
    def reply(data: dict, image: Path | None = None):
        out: list = [TextContent(type="text", text=json.dumps(data, separators=(",", ":")))]
        if image is not None:
            out.append(Image(path=image).to_image_content())
        return out

    from topopt.config import UserError
    try:
        from topopt.render_input import check_inputs   # light: a wrong input stops before numba and Pillow load
        if yaml_path:
            sources = [str(_resolve_yaml(yaml_path))]
        elif stl_paths:
            sources = [str(_resolve(p)) for p in stl_paths]
        else:
            d = _pick_dir(run_dir)
            if not run_dir and not (d / "result" / "final.stl").is_file():
                raise UserError("no run yet: give run_dir or stl_paths")
            sources = [str(d)]
        opts = dict(roles=roles or None, views=views, section=section, size=size, regions=regions,
                    max_triangles=max_triangles, loads=loads)
        check_inputs(sources, **opts)
        from topopt.render import render
        facts = render(sources, out_dir=str(_resolve(out_dir)) if out_dir else None, **opts)
    except Exception as exc:  # a user error or a failure: say it as data
        return reply({"ok": False, "error": str(exc) if isinstance(exc, UserError)
                      else f"{type(exc).__name__}: {exc}"})
    facts["out_dir_rel"] = _rel(facts["out_dir"])
    facts["montage_rel"] = _rel(facts["montage"]["path"])
    facts["inline"] = bool(inline)
    return reply(facts, Path(facts["montage"]["path"]) if inline else None)


def _scan_folders(wanted) -> list[Path]:
    """The folders in the work folder, and in its runs/ folder, for which `wanted(folder)` is true (a folder that cannot be read is skipped)."""
    root = workspace()
    found: list[Path] = []
    for parent in (root, root / "runs"):
        try:
            children = sorted(c for c in parent.iterdir() if c.is_dir() and not c.is_symlink())
        except OSError:
            continue
        for c in children:
            try:
                if wanted(c):
                    found.append(c)
            except OSError:
                continue
    return found


@mcp.tool()
def list_runs() -> dict:
    """List the run folders in the work folder (newest first), with state and iteration.

    A run folder is any folder in the work folder, or in its runs/ folder, that holds
    status.json or run.json. Paths are relative to the work folder.
    """
    root = workspace()
    from topopt.runjob import is_run_folder
    found = _scan_folders(is_run_folder)
    summaries = []
    for d in found:
        try:
            summaries.append(_run_summary(d))
        except OSError:
            continue
    runs = sorted(summaries, key=lambda r: r["mtime"], reverse=True)
    return {"workspace": str(root), "runs": runs}


@mcp.tool()
def run_history(root: str | None = None, depth: int = 2, limit: int | None = 20, detail: bool = False) -> dict:
    """Facts of the earlier runs under a folder, newest first, read from the files on disk. Writes nothing.

    Use it to see what was tried: state, the YAML of each run, key settings, main metrics. A missing or partial file
    (no metrics.json, a cut last history line, a broken status.json) is a fact in `problems`, not an error.
    CONTRACTS.md section 21.

    root: the folder to search (relative paths start at the work folder). Default: the work folder.
    depth: how many levels of folders to search (default 2: runs/<name> under the work folder; at most 5). A run
        folder holds run.json or status.json. A root that is a run folder gives that run. The scan stops after 5000
        folders and says so.
    limit: the newest runs only (default 20). `count` is the number of runs found, and a note says when older runs
        are left out. 0 gives all runs.
    detail: the full facts of each run (all settings, all metrics, which files exist).
    Each run: name, run_dir, path (relative), state, iteration, iterations, process_alive, modified, yaml_path,
    yaml_exists, yaml_newer_than_run (the YAML file is newer than the run: it may not be the one that ran), settings
    (voxel_size_mm, volume_fraction, iterations, solver, filter_radius_mm, n_active, load case names), metrics
    (compliance_initial and compliance_final, mass_final_g, volume_final_mm3, volume_fraction_final, run_time_s),
    metrics_source and problems. list_runs is the cheaper list (it reads no metrics).
    """
    from topopt.feedback import run_history as history
    base = _resolve(root) if root else workspace()
    data = history(base, depth, limit, detail)
    if data.get("ok"):
        data["root_rel"] = _rel(base)
        for r in data["runs"]:
            r["path"] = _rel(r["run_dir"])
            r["process_alive"] = bool(_is_running(Path(r["run_dir"])))
    return data


@mcp.tool()
def compare_runs(run_a: str, run_b: str | None = None, check_labels: bool | None = None,
                 variants: list[str] | None = None) -> dict:
    """Runs side by side: two runs (setup diff and metrics diff), or a base with its variants (one table). Reads files, writes nothing.

    Give one form. (run_a, run_b): A is the base and B the change; `delta` is B - A. Every value names its source: run.json (what each
    run recorded at its start: the YAML path and the region file names, no hash), labels.bin, the YAML and STL files on disk (read now),
    and result/metrics.json. So the files on disk can be newer than the run or can differ from it; `inputs` says so as facts (a newer
    YAML or STL, YAML values that differ from run.json, labels from the files on disk that differ from labels.bin). The facts are evidence, not proof.
    (run_a, variants): run_a is the base; variants is a list of run folders, any number. One row per run, the base first.

    run_a: the base run folder (relative paths start at the work folder). run_b: the changed run folder (two-run form).
    variants: the run folders of the table (table form); not together with run_b.
    check_labels: voxelize the YAML on disk and compare it with labels.bin (a few seconds on a big grid).
        null: do it when the grid has at most 10 million voxels. true: any grid size. false: skip it.
    Two-run keys: a, b, setup (run_json_diff, yaml_diff, region_files with the STL hashes, labels_diff: the voxel classes that changed,
    with the box in mm), metrics (rows with a, b, delta, delta_pct; stiffness per case; filter facts), inputs, notes (for example: the
    load cases differ, so compliance is not comparable), summary and problems.
    Table keys: base; runs (state, mass_final_g, mass_saved_pct, stiffness per load case from the optimizer, verdict, verify: done,
    not run or no result, print: a few print facts, render: the path of the newest montage PNG on disk, or null); vs_base (the setup
    diff of each variant against the base); notes; summary; problems. A fact that no file holds is null and an entry in problems.
    The table makes no image: render_part does.
    On a wrong call or folder: {"ok": false, "error": ...}.
    """
    from topopt.feedback import compare_runs as compare

    def path(x):    # a bad value goes to the library as it is, which refuses it and names the forms
        return _resolve(x) if isinstance(x, str) and x.strip() else x
    a = path(run_a)
    b = path(run_b)
    v = [path(x) for x in variants] if isinstance(variants, list) else variants
    data = compare(a, b, check_labels, v)
    if data.get("ok") and "runs" in data:
        for row in [data["base"], *data["runs"], *data["vs_base"]]:
            row["run_dir_rel"] = _rel(row["run_dir"])
    elif data.get("ok"):
        data["a"]["run_dir_rel"], data["b"]["run_dir_rel"] = _rel(a), _rel(b)
    return data


@mcp.tool()
def marks_to_hints(marks: str | dict, run_dir: str | None = None, out_dir: str | None = None,
                   options: dict | None = None, overwrite: bool = False, detail: bool = False,
                   yaml_path: str | None = None) -> dict:
    """Turn the marks of the Refine page into facts, options and a proposed YAML in a NEW folder. Edits no mesh.

    A mark is a point of emphasis, not an absolute. The report gives, for each mark, the facts (position, radius, distance to the surface of final.stl, voxels touched, overlap with the design, fixed, load and keep-out regions, whether the region touches the material of final.stl) and the options for its type, and says which option the proposed YAML uses. The agent decides: read the report, change `options` and call again, edit the YAML, or run it with run_topopt.
    The tool writes only into the new folder. It refuses an out_dir that is a run folder or inside one, one that holds an input file of the run (the folder of the YAML or of a region STL), a symbolic link, and a folder with files that is not an earlier hints folder; the reply names the fact and the options.
    After it: check the report, then run_topopt with the proposed YAML and a new out_dir, then compare_runs.
    Reply keys: resource topopt://replies/marks_to_hints (the meaning of each key).

    marks: the path of marks.json (the Refine page exports it: version 1, mm, the frame of final.stl), its JSON text ("Copy JSON" on the page) or the marks object.
    run_dir: the run the marks belong to. Default: the last run of this server. Needs run.json and the YAML it names.
    out_dir: the new folder. Default: <run folder>-hints beside the run folder. It holds the proposed YAML (the YAML of the run, with the changes), the region STLs (the originals are copied, the hints are new) and hints.json (the full report; its paths are relative to the folder).
    detail: the full report in the reply (about 1.5 KB for each mark). Default: a short reply; hints.json has the full report.
    options: remove ("keep_out" or "skip"), add ("solid" or "skip"), radius_scale (default 1.0; 0.5 gives smaller regions), protect_regions (default true: fixed and load regions are cut out of a keep-out hint), clip_to_design (default true: a solid hint is cut to the design space), per_mark ({mark id: option}).
    overwrite: write again into a folder that holds an earlier hints.json.
    yaml_path: the YAML of the run, when run.json names a file that is not there (a moved run). The labels check in the report says if it fits the run.
    The input YAML, the run folder and the region STLs of the run stay as they are; the new region STLs touch the grid only inside the box of the input regions.
    Mapping in this version: remove -> a keep-out STL (sphere, or capsule chain for a stroke) of radius radius_mm x radius_scale; add -> a forced-solid region (a load region that no load case uses); smooth -> facts and options only, because the YAML has no key that smooths one place. `mirror` repeats the mark at y -> -y. normals, amount_mm, strength and note are not used. The marks are read in file order and the first mark wins an overlap: a later hint region is cut by the earlier ones (a mark inside an earlier one has status no_effect, with the reason).
    """
    from topopt.feedback import make_hints, short_report
    d = _pick_dir(run_dir)
    marks_arg = marks if isinstance(marks, dict) or marks.lstrip().startswith("{") else str(_resolve(marks))
    data = make_hints(d, marks_arg, _resolve(out_dir) if out_dir else None, options, overwrite, True,
                      _resolve_yaml(yaml_path) if yaml_path else None)
    if data.get("out_dir"):                  # also when ok is false: the folder holds the proposal and hints.json
        data["run_dir_rel"] = _rel(d)
        data["out_dir_rel"] = _rel(data["out_dir"])
        data["yaml_rel"] = _rel(data["proposed_yaml"]["path"])
        data["report_rel"] = _rel(Path(data["out_dir"]) / "hints.json")
        if not detail:
            rels = {k: data[k] for k in ("run_dir_rel", "out_dir_rel", "yaml_rel", "report_rel")}
            data = dict(short_report(data), **rels)
    return data


@mcp.tool()
def apply_marks(marks: str | dict, run_dir: str | None = None, out_dir: str | None = None,
                options: dict | None = None, overwrite: bool = False, allow_stale: bool = False,
                detail: bool = False, yaml_path: str | None = None) -> dict:
    """Edit the finished part at the marks of the Refine page, in a NEW folder: facts and options, no verdict.

    The run folder and final.stl stay as they are. No FE run is made: the stiffness and stress of the edited part are not known. Order: remove, then smooth, then add. A step that gives a part that is not closed, or that splits it, is rolled back, and the reply says why. The fixed, load and keep-out regions are frozen: an edit within frozen_keep_mm of them is clipped, and the clip is a fact.
    Reply keys: resource topopt://replies/apply_marks (the meaning of each key).
    marks: the path of marks.json (the Refine page exports it: version 1, mm, the frame of final.stl), its JSON text, or the marks object. radius_mm, amount_mm (the neck offset of add), strength (smooth) and mirror are used; a value that the page does not record is the pipeline value, named in `defaults_used`. A mark may carry smooth_mode or remove_mode (cleanup_list writes them); the order is per_mark, the mark, then the call option.
    run_dir: the run the marks belong to. Default: the last run of this server. Needs run.json, labels.bin and result/final.stl.
    out_dir: the new folder. Default: <run folder>-marks beside the run folder. It gets edited.stl and apply_marks.json (the full report). Refused: a run folder or a folder inside one, a folder that holds an input file of the run, a symbolic link, a folder with files that is not an earlier apply_marks folder.
    allow_stale: apply the marks when source_sha256 of the marks file is not the hash of final.stl. Default: the reply states the fact (marks_match_final_stl false) and nothing changes.
    options: radius_scale (1.0), pitch_mm (node spacing of the crop box; default voxel size / 4, between 0.2 and 0.5), max_crop_voxels (8 million, a guard: above it the reply gives the facts and the options, e.g. pitch_mm).
    overwrite: write again into an earlier apply_marks folder.
    detail: the full report in the reply. Default: a short reply; apply_marks.json has the full report.
    yaml_path: the YAML of the run when run.json names a file that is not there; used for the exact frozen-region check.
    CONTRACTS.md section 29. Each mark is one local edit of result/final.stl on a signed distance field in a crop box around the mark, with the method of the pipeline stage s7_cleanup: remove = a cut that protects the parent (then the parent check), add = a closing with a ball (a neck offset when the closing adds little), smooth = an isotropic fairing for a point and a web fairing for a stroke (smooth_mode tube rebuilds a whole strut: it does not use radius_mm, and the reply gives its planned volume and reach). The new surface is spliced into the mesh in a zone around the edit; the triangles outside the zones stay bit for bit.
    More keys of options: max_total_voxels, frozen_keep_mm (3), frozen_full_mm (6), close_mm (3), neck_min_mm3, parent_tol_mm (0.2), parent_check ("report" or "enforce"), smooth_mode ("auto", "tube", "iso", "fair"), remove_mode ("cut" or "shave"), mirror_plane ({axis, at_mm}; else the plane of the mark or of run.json, else y = 0), skip ([mark ids]), per_mark ({id: {radius_mm, amount_mm, strength, mirror, smooth_mode, remove_mode, skip}}), check_frozen_exact, loose_body_mm3, max_odd_rays.
    """
    from topopt.apply_marks import apply_marks as apply, short_report
    d = _pick_dir(run_dir)
    marks_arg = marks if isinstance(marks, dict) or marks.lstrip().startswith("{") else str(_resolve(marks))
    data = apply(d, marks_arg, out_dir=_resolve(out_dir) if out_dir else None, options=options, overwrite=overwrite,
                 allow_stale=allow_stale, yaml_path=_resolve_yaml(yaml_path) if yaml_path else None)
    if data.get("out_dir"):
        data["run_dir_rel"] = _rel(d)
        data["out_dir_rel"] = _rel(data["out_dir"])
        data["report_rel"] = _rel(Path(data["out_dir"]) / "apply_marks.json")
        if (data.get("files") or {}).get("edited_stl"):
            data["edited_rel"] = _rel(Path(data["out_dir"]) / data["files"]["edited_stl"])
        if not detail:
            rels = {k: data[k] for k in ("run_dir_rel", "out_dir_rel", "report_rel", "edited_rel") if k in data}
            data = dict(short_report(data), **rels)
    return data


def _preference_facts(yaml: Path) -> list[str]:
    """The fact about a difference between a problem YAML and the preferences file (topopt.preferences). [] when there is none.
    It adds a key to the reply and changes nothing else; a failure here never breaks check_problem."""
    try:
        from topopt import preferences
        return preferences.facts_for_yaml(workspace(), yaml)
    except Exception:
        return []


@mcp.tool()
def check_problem(yaml_path: str) -> dict:
    """Load and voxelize a problem YAML (no optimizer run; with settings.solver auto it runs the short self-test of gpu-mg in a child process, at most 60 s: seconds the first time on a device and driver, then the cached pass). Returns facts, estimates and warnings.

    Use it before run_topopt, also for your own YAML and STLs. Read the facts and tell the user the numbers that matter.
    Resource topopt://levers: the key for a symptom of these facts, and the price of a try.
    yaml_path: the problem YAML (relative paths start at the work folder).
    Another solver: settings.solver (auto, cholesky, pardiso, amg, gpu-mg) or --solver of topopt run and check, not of run_topopt. For gpu-mg, settings.gpu_device (auto, a number from topopt check, or a part of the name).
    Warnings are facts; a warning that starts with "settings.solver is" (the solver cannot run here) means run_topopt will stop with the same message. A warning that starts with "fact: settings.iterations" says that the passes end before the last stage of settings.beta_schedule (a grey design): it gives the options (iterations, or the schedule scaled to the passes) and the measured price. It does not block the run. On a wrong input: {"ok": false, "error": ...}.
    Reply keys: resource topopt://replies/check_problem (the meaning of each key).
    A settings.perimeter_max_mm2 below the smallest possible perimeter gives {"ok": false, "error": ...} with the options.
    The reply key `lost_force` has, for each load case, the total force, the force on nodes that are also fixed (the solver drops it) and the lost percent.
    Key fine_pass: the voxel that the part needs (`voxel_mm`, one value; `recommend`; `limiting`: the feature that sets it, its thickness and the voxels across now and needed; and with `recommend` the `fact` line). run_topopt with fine_pass=true or fine_from runs a fine pass.
    """
    yaml = _resolve_yaml(yaml_path)
    if not yaml.is_file():
        return {"ok": False, "error": f"YAML file not found: {yaml} (work folder: {workspace()})"}
    try:
        from topopt.cli import check_data
        from topopt.config import UserError
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}"}
    try:
        data = check_data(yaml)
    except UserError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}
    data["ok"] = True
    data["yaml_rel"] = _rel(yaml)
    facts = _preference_facts(yaml)
    if facts:
        data["preferences_facts"] = facts
    return data


# --------------------------------------------------------------------------- #
# verify_run: a detached child process with a status file (like run_topopt)
# --------------------------------------------------------------------------- #

VERIFY_LOCK = ".topopt-verify.lock"   # in the work folder: {pid, run_dir, started}. One verify job at a time.
_verify: dict = {}                    # the verify job that this server started: Popen handle, run folder, start time


def _verify_lock_path() -> Path:
    return workspace() / VERIFY_LOCK


def _verify_log_path(run_dir: Path) -> Path:
    return run_dir.parent / (run_dir.name + ".verify.stdout")


def _cmdline_matches(pid: int) -> bool:
    """False if the process with this pid is not a topopt process (a reused pid). True if it is, or if we cannot tell."""
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True            # no /proc (macOS): trust the pid
    return b"topopt" in data


def _verify_terminal(run_dir: Path) -> bool:
    """True if the status file of the folder says the job ended (done or error)."""
    from topopt.verify.status import BUSY_STATES as VERIFY_BUSY, read_status
    st = read_status(run_dir / "verify")
    return st is not None and st.get("state") not in VERIFY_BUSY


def _alive_verify() -> tuple[Path, int] | None:
    """Return (run folder, pid) of a verify job that is alive, else None.

    It looks at the Popen handle of this server, and at the lock file in the work folder (a job of an earlier server).
    A lock of a dead process, of a process that is not topopt, or of a job that wrote done or error is removed.
    """
    proc = _verify.get("proc")
    if proc is not None and proc.poll() is None:
        return _verify["run_dir"], proc.pid
    lock = _read_json(_verify_lock_path())
    if lock is None:
        return None
    pid, d = lock.get("pid"), lock.get("run_dir")
    if isinstance(pid, int) and d and _pid_alive(pid) and _cmdline_matches(pid) and not _verify_terminal(Path(d)):
        return Path(d), pid
    try:
        _verify_lock_path().unlink()
    except OSError:
        pass
    return None


def _verify_command(run_dir: Path, args: list[str]) -> list[str]:
    template = os.environ.get("TOPOPT_MCP_VERIFY_CMD")
    if template:   # test hook: the words {run_dir} is replaced; no options are added
        return [w.replace("{run_dir}", str(run_dir)) for w in shlex.split(template)]
    return [python_exe(), "-m", "topopt", "verify", str(run_dir), *args]


def _verify_args(mesh_size_mm, levels_mm, region_band_mm, allowable_mpa, percentile, solver, max_gb, force, yaml_path,
                 near_region_mm, load_sensitivity, part_only: bool = False, gate: bool = False) -> list[str]:
    a: list[str] = []
    for flag, val in (("--mesh-size", mesh_size_mm), ("--region-band-mm", region_band_mm), ("--allowable-mpa", allowable_mpa),
                      ("--solver", solver), ("--max-gb", max_gb), ("--yaml", yaml_path), ("--near-mm", near_region_mm)):
        if val is not None:
            a += [flag, str(val)]
    if levels_mm:
        a += ["--levels", ",".join(f"{x:g}" for x in levels_mm)]
    if percentile != 99.9:
        a += ["--percentile", str(percentile)]
    if force:
        a.append("--force")
    if not load_sensitivity:
        a.append("--no-sensitivity")
    if part_only:
        a.append("--part-only")
    if gate:
        a.append("--gate")
    return a


def _tail(path: Path, n: int = 12) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _tight_doc(fn):
    """Remove the indent of the docstring of a tool, in `__doc__` itself (some tests read it). Since #388 the tool decorator cleans
    every description the same way, so the text that a client gets does not depend on this. With `python -OO` the docstrings are gone
    (`__doc__` is None): then there is nothing to change."""
    if fn.__doc__:
        fn.__doc__ = inspect.cleandoc(fn.__doc__)
    return fn


@mcp.tool()
@_tight_doc
def verify_run(run_dir: str | None = None, estimate_only: bool = True, mesh_size_mm: float | None = None,
               levels_mm: list[float] | None = None, region_band_mm: float | None = None,
               allowable_mpa: float | None = None, percentile: float = 99.9, solver: str | None = None,
               max_gb: float | None = None, force: bool = False, yaml_path: str | None = None,
               near_region_mm: float = 2.0, load_sensitivity: bool = True, part_only: bool = False, gate: bool = False) -> dict:
    """Check the final STL of a finished run with quadratic tets (P2): von Mises stress per load case.

    It is independent of the voxel optimizer: it meshes result/final.stl (part_only.stl with part_only) with tets (gmsh), clamps each fixed region (a joint or spring holds the part too), loads each load region with a traction (uniform per area, the resultant is the YAML force), and solves each load case alone (no case weights). Units: mm, N, MPa.

    estimate_only: default True. It returns the estimates for each mesh size (memory in bytes, time in s per measured solver), the solver that would run, the extras and the RAM, and meshes nothing. Tell the user the numbers, then call again with estimate_only=false.
    A real run starts in the background and returns at once; verify_status shows it and gives the result. One verify job or optimizer run at a time: a second start is refused.
    allowable_mpa, percentile: your allowable stress, no default (ranges: topopt://materials/reference). It adds a `limit` block. exit_code is 3 if the percentile value (by volume, default 99.9) of a case is above it. With gate it replaces the YAML stress limits (a larger value loosens them: the user's consent).
    run_dir: the run folder (default: the last run that this server started). It needs run.json and result/final.stl.
    gate: true adds the key gate (met per check and case; exit_code 3 if it fails); limits: the YAML's, see topopt://levers/verify_run:gate
    max_gb: refuse above this memory estimate in GB; it also refuses above the available RAM. force: run anyway.
    On a wrong input or a run that cannot go on: {"ok": false, "error": ..., "exit_code": 1 or 2, "facts": ...} (exit_code 1: bad input, an unsolvable part, no usable solver; exit_code 2: a missing gmsh or scikit-fem, a meshing failure, a memory refusal, an empty region).
    p99.9 is the main number; the peak can move with the mesh at sharp features. A load patch can hold a large part of the top volume: read near_regions and load_band_sensitivity.
    mesh_size_mm: one mesh size. Default: the tet mesher size, about 1.3 to 2.2 voxels for an sdf STL.
    levels_mm: several sizes; the finest is the reference level; the result gets a `levels` block.
    region_band_mm: a mesh node belongs to a region when it is inside the region STL or within this band of its surface. Default: 0.5 voxel (counts_by_band: the counts at 0, 0.5 and 1 band).
    solver: auto (default: cholesky on macOS arm64, pardiso, cholespy, superlu) or a factor name. The result names the one that ran and those auto passed.
    yaml_path: the problem YAML, if the one that run.json names is gone (moved folder).
    near_region_mm: distance from a fixed or load region for the stress numbers without the region patches.
    load_sensitivity: also solve the load at bands of 0 and 2 times the band.
    part_only: mesh result/part_only.stl (the run needs regions.not_printed); a removed body becomes a boundary condition at its contact face, see topopt://levers/verify_run:part_only. Key: part_only.

    Result keys (verify_status, when done): ok, exit_code (0 done, 3 limit failed), cases.<case>.von_mises.p99_9_by_volume_MPa. All other keys, and the keys that YAML blocks add (buckling.check, limits rows, bore_checks): topopt://levers/verify_run
    """
    try:
        from topopt.config import UserError
        from topopt.verify import names
        from topopt.verify.errors import EmptyRegion, MemoryRefused, VerifyError
        from topopt.verify.run import preflight
        from topopt.verify.run import verify_run as run_verify
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}", "exit_code": 2}
    d = _pick_dir(run_dir)
    ypath = str(_resolve(yaml_path)) if yaml_path else None
    kw = dict(mesh_size=mesh_size_mm, levels=levels_mm, region_band_mm=region_band_mm, allowable_mpa=allowable_mpa,
              percentile=percentile, solver=solver, max_gb=max_gb, force=force, yaml_path=ypath,
              near_region_mm=near_region_mm, load_sensitivity=load_sensitivity, part_only=part_only, gate=gate)
    try:
        with names.use_style("mcp"):
            if estimate_only:
                res = run_verify(d, estimate_only=True, log=log, **kw)
                res["run_dir"], res["run_dir_rel"] = str(d), _rel(d)
                return res
            plan = preflight(d, say=log, **kw)
    except UserError as exc:
        out = {"ok": False, "error": str(exc), "exit_code": 1}
        if getattr(exc, "facts", None):                  # SolverUnavailable, UnsolvableMesh: the facts behind the text
            out["facts"] = exc.facts
        return out
    except VerifyError as exc:
        out = {"ok": False, "error": str(exc), "exit_code": 2}
        if isinstance(exc, (MemoryRefused, EmptyRegion)):
            out["facts"] = exc.facts
        return out
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}", "exit_code": 2}

    # ---- a real run: one job at a time, then a detached child ----
    busy = _alive_verify()
    if busy:
        return {"ok": False, "exit_code": 2, "run_dir": str(busy[0]), "pid": busy[1],
                "error": f"A verify job is still running (pid {busy[1]}, folder {busy[0]}). Only one job goes at a time "
                         "on this machine (each one can use several GB). Call verify_status to watch it. " + NO_STOP_VERIFY}
    ratcheting = _alive_ratchet()
    if ratcheting:
        return {"ok": False, "exit_code": 2, "run_dir": str(ratcheting[0]), "pid": ratcheting[1],
                "error": f"A ratchet job is still running (pid {ratcheting[1]}, folder {ratcheting[0]}). Only one job goes at a "
                         "time on this machine (it runs verify for each candidate). Call ratchet_status to watch it. " + NO_STOP_RATCHET}
    alive = _alive_run()
    if alive and _is_busy(alive[0]):
        return {"ok": False, "exit_code": 2, "run_dir": str(alive[0]), "pid": alive[1],
                "error": f"An optimizer run is still running (pid {alive[1]}, folder {alive[0]}). Only one job goes at a "
                         "time on this machine. Wait for it to finish (run_status shows it). " + STOP_RUN_OPTION}
    problem = None if os.environ.get("TOPOPT_MCP_VERIFY_CMD") else _python_problem(python_exe())
    if problem:
        return {"ok": False, "error": problem, "exit_code": 2}
    cmd = _verify_command(d, _verify_args(mesh_size_mm, levels_mm, region_band_mm, allowable_mpa, percentile, solver,
                                          max_gb, force, ypath, near_region_mm, load_sensitivity, part_only, gate))
    log_path = _verify_log_path(d)
    made = _make_work_folder()
    if made:
        return {"ok": False, "error": made, "exit_code": 2}
    from topopt.verify.status import clear_outputs
    clear_outputs(d / "verify")        # the child does it too; this makes verify_status say "starting", not an old result
    log(f"start: {' '.join(cmd)}")
    try:
        with open(log_path, "wb") as sink:
            proc = subprocess.Popen(cmd, cwd=str(workspace()), stdin=subprocess.DEVNULL, stdout=sink,
                                    stderr=subprocess.STDOUT, start_new_session=True, env=_child_env(TOPOPT_VERIFY_STYLE="mcp"))
    except OSError as exc:
        return {"ok": False, "error": f"Could not start the verify job: {exc}", "exit_code": 2}
    _verify.update(proc=proc, run_dir=d, started=time.time())
    try:
        _verify_lock_path().write_text(json.dumps({"pid": proc.pid, "run_dir": str(d), "started": time.time()}))
    except OSError as exc:
        log(f"cannot write the lock file: {exc}")
    secs = [e["estimated_seconds"] for e in plan.estimates]
    # a job that ends at once (for example a broken Python) shows its output here
    end = time.time() + 1.0
    while time.time() < end and proc.poll() is None:
        time.sleep(0.05)
    if proc.poll() is not None and proc.returncode not in (0, 3):
        return {"ok": False, "exit_code": 2, "run_dir": str(d), "error": f"The verify job ended at once with code "
                f"{proc.returncode}: {_tail(log_path)}", "log_file": str(log_path)}
    reply = {
        "ok": True, "started": True, "run_dir": str(d), "run_dir_rel": _rel(d), "pid": proc.pid,
        "log_file": str(log_path), "log_file_rel": _rel(log_path),
        "status_file": str(d / "verify" / "status.json"), "status_file_rel": _rel(d / "verify" / "status.json"),
        "verify_dir": str(d / "verify"), "verify_dir_rel": _rel(d / "verify"),
        "solver": plan.chosen, "sizes_mm": plan.sizes, "estimates_before_meshing": plan.estimates,
        "estimated_seconds_total": None if None in secs else float(sum(secs)),
        "yaml_vs_run": plan.base["yaml_vs_run"],
        "message": "The verify job runs in the background. Call verify_status to watch it and to get the result. " + no_callback("verify_status", VERIFY_END),
    }
    if "part_only" in plan.base:          # part_only=true: the file, its sha256 and the removed bodies (the facts of the bodies come with the result)
        reply["part_only"] = plan.base["part_only"]
    if plan.notes:                        # a run that has part_only.stl, verified without the option: one fact line (not in verify.json)
        reply["notes"] = list(plan.notes)
    return reply


@mcp.tool()
def verify_status(run_dir: str | None = None, result: bool = True) -> dict:
    """Show the state of a verify job and, when it is done, its result (the content of verify.json).

    run_dir: the run folder (default: the folder of the last verify job of this server, else the last run).
    Call it again to watch the job. Nothing calls you back when the job ends: call it again in this turn until the state is `done`, `error` or `failed`, or end the turn and tell the user what to ask for the result. A promise of a later report cannot be kept. States: `starting`, `meshing`, `solving`, `writing` (the job works), `done`,
    `error`, `failed` (the process ended with no final status; `log_tail` has its last output) and `none` (no job
    and no result in the folder). It has `elapsed_s` and `estimated_seconds_total` (the estimate of verify_run).
    result: with a done job, add the result (set false to get the state only).
    `result_matches_final_stl` says whether the result is for the mesh file that the folder has now (`result.run.verified_file`:
    nominal.stl when the run has machining stock, else final.stl; part_only.stl after verify_run with part_only=true,
    and then the result has the block `part_only`). If it is false, the result belongs to an older file: run verify_run again. Tell the user the numbers; you decide what to do.
    `plain` (a done job with the result): one plain line for each load case (the value that is compared, who gave the allowable, the margin) and a line for what the check does not cover. They are a base for the answer to "will it break?".
    `partial`, `coverage` (a done job; also with result=false): `partial` is true when a check that the YAML or the call asks for failed or did not run; the other numbers still stand. `coverage` has `ran` (names), `failed` (name, reason) and `not_run` (name, reason, `asked`). A check that was not asked for is under `not_run` with `asked` false and does not make the result partial. Then `summary` and the first `plain` line say so.
    """
    try:
        from topopt.verify.status import BUSY_STATES as VERIFY_BUSY, read_status
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}", "exit_code": 2}
    d = _resolve(run_dir) if run_dir else (_verify.get("run_dir") or _pick_dir(None))
    out_root = d / "verify"
    st = read_status(out_root)
    alive = _alive_verify()
    pid = alive[1] if alive and alive[0] == d else None
    log_path = _verify_log_path(d)
    res: dict = {"run_dir": str(d), "run_dir_rel": _rel(d), "process_alive": bool(pid), "pid": pid,
                 "log_file": str(log_path) if log_path.exists() else None,
                 "log_file_rel": _rel(log_path) if log_path.exists() else None, "status": st}
    vjson = _read_json(out_root / "verify.json")
    if st is None:
        if pid:
            res.update(state="starting", summary="the job has started")
        elif vjson is not None:
            res.update(state="done", summary="a result from an earlier run (no status file)")
        else:
            res.update(state="none", summary="no verify job and no result in this folder")
    else:
        state = st.get("state")
        res.update(state=state, message=st.get("message"), level=st.get("level"), levels=st.get("levels"),
                   size_mm=st.get("size_mm"), estimated_seconds_total=st.get("estimated_seconds_total"))
        started = st.get("started_at")
        if started:
            res["elapsed_s"] = float((st.get("updated_at") if state not in VERIFY_BUSY else time.time()) - started)
        if state in VERIFY_BUSY and not pid:
            res.update(state="failed", summary="the verify process has stopped with no final status "
                       "(it can have been killed, for example for lack of memory)", log_tail=_tail(log_path))
        elif state == "error":
            res.update(summary=f"error: {st.get('error')}", error=st.get("error"), exit_code=st.get("exit_code"))
        elif state == "done":
            res.update(summary="done", exit_code=st.get("exit_code"))
        else:
            n, i = st.get("levels"), st.get("level")
            res["summary"] = f"{state}: level {i}/{n}, mesh {st.get('size_mm', 0):g} mm: {st.get('message', '')}"
    if res["state"] == "done" and vjson is not None:
        sha_now = None
        try:
            import hashlib
            sha_now = hashlib.sha256((d / "result" / ((vjson.get("run") or {}).get("verified_file") or "final.stl")).read_bytes()).hexdigest()
        except OSError:
            pass
        old = (vjson.get("run") or {}).get("final_stl_sha256")
        res["result_matches_final_stl"] = None if sha_now is None or old is None else bool(sha_now == old)
        res["exit_code"] = vjson.get("exit_code", res.get("exit_code"))
        if isinstance(vjson.get("partial"), bool) and isinstance(vjson.get("coverage"), dict):      # TC-2: the record, before the large result
            from topopt import check_coverage
            res["partial"], res["coverage"] = vjson["partial"], vjson["coverage"]
            if vjson["partial"]:
                res["summary"] = f"{res.get('summary') or 'done'}; {check_coverage.line_of(vjson)}"
        res["verify_json"] = str(out_root / "verify.json")
        res["verify_json_rel"] = _rel(out_root / "verify.json")
        res["verify_dir"] = str(out_root)
        res["verify_dir_rel"] = _rel(out_root)
        files = vjson.get("files") or {}
        res["files_abs"] = {k: str(out_root / v) for k, v in files.items() if isinstance(v, str)}
        res["files_rel"] = {k: _rel(out_root / v) for k, v in files.items() if isinstance(v, str)}
        if result:
            res["result"] = vjson
            lines = _plain_verify_lines(vjson, stale=res["result_matches_final_stl"] is False)
            if lines:
                res["plain"] = lines
    return res


def _plain_verify_lines(vjson: dict, stale: bool = False) -> list[str]:
    """The `plain` key of verify_status (P9-17): one plain line for each load case, from `topopt.verify.report.plain_lines`."""
    from topopt.verify import names
    from topopt.verify.report import plain_lines
    try:
        with names.use_style("mcp"):
            lines = plain_lines(vjson)
    except (KeyError, TypeError, ValueError):       # a result of another version: the full result is still in the reply
        return []
    if lines and stale:
        lines.insert(0, "These lines belong to an older mesh file than the folder has now (result_matches_final_stl is false): "
                        "run verify_run again before you rely on them.")
    return lines


# --------------------------------------------------------------------------- #
# ratchet_run: a detached child process with a lock file (CONTRACTS.md section 36)
# --------------------------------------------------------------------------- #

RATCHET_LOCK = ".topopt-ratchet.lock"   # in the work folder: {pid, root, run_dir, started}. One ratchet at a time.
_ratchet: dict = {}                     # the ratchet that this server started: Popen handle, root, run folder, start time


def _ratchet_lock_path() -> Path:
    return workspace() / RATCHET_LOCK


def _ratchet_log_path(root: Path) -> Path:
    """The output of the child. It sits beside the root, not in it (the root holds the files of the ratchet only)."""
    return root.parent / (root.name + ".stdout")


def _ratchet_terminal(root: Path) -> bool:
    """True if the job files of the root say that the job ended (done or failed)."""
    from topopt.runjob import job_facts
    facts = job_facts(root)
    return facts is not None and facts.get("state") in ("done", "failed")


def _alive_ratchet() -> tuple[Path, int] | None:
    """Return (ratchet root, pid) of a ratchet job that is alive, else None.

    It looks at the Popen handle of this server, and at the lock file in the work folder (a job of an earlier server).
    A lock of a dead process, of a process that is not topopt, or of a job that wrote done or failed is removed.
    """
    proc = _ratchet.get("proc")
    if proc is not None and proc.poll() is None:
        return _ratchet["root"], proc.pid
    lock = _read_json(_ratchet_lock_path())
    if lock is None:
        return None
    pid, root = lock.get("pid"), lock.get("root")
    if isinstance(pid, int) and root and _pid_alive(pid) and _cmdline_matches(pid) and not _ratchet_terminal(Path(root)):
        return Path(root), pid
    try:
        _ratchet_lock_path().unlink()
    except OSError:
        pass
    return None


def _ratchet_command(run_dir: Path, root: Path, source: str, args: list[str]) -> list[str]:
    template = os.environ.get("TOPOPT_MCP_RATCHET_CMD")
    if template:   # test hook: the words {run_dir} and {root} are replaced; no options are added
        return [w.replace("{run_dir}", str(run_dir)).replace("{root}", str(root)) for w in shlex.split(template)]
    return [python_exe(), "-m", "topopt.ratchet", "run", str(run_dir), "--source", source, "--root", str(root), *args]


def _ratchet_source(source: str | None, marks: str | None) -> str | None:
    """`marks:FILE` or `stl:FOLDER` with the path made absolute (a relative path starts at the work folder)."""
    if not source and marks:
        source = f"marks:{marks}"
    if not source:
        return None
    kind, sep, rest = source.partition(":")
    if sep and rest.strip() and kind in ("marks", "stl"):
        return f"{kind}:{_resolve(rest)}"
    return source          # the tool says what is wrong with it


@mcp.tool()
def ratchet_run(run_dir: str | None = None, source: str | None = None, marks: str | None = None, root: str | None = None,
                estimate_only: bool = True, yaml_path: str | None = None, max_candidates: int | None = None,
                mesh_size_mm: float | None = None, stop_before_promote: bool = False, overwrite: bool = False,
                rebaseline: bool = False, override_lock: bool = False) -> dict:
    """Judge candidate edits of a finished part and keep the ones that pass the rules of the YAML (the ratchet).

    The start part is judged first (the baseline). A candidate that passes every listed rule becomes the new parent (folder `acc_NN` in the ratchet root). A candidate that fails is kept in the log with the reason. The rules are the list `acceptance.rules` in the problem YAML; a rule that is not listed is off. Nothing in the start run is written.
    estimate_only: default True. It returns the rules, what each rule needs, what is missing (a rule with a missing input fails every candidate), the number of candidates and the time of verify (its own estimate), and starts nothing. Tell the user the numbers, then call again with estimate_only=false. A real start returns at once (a detached process does the work, like verify_run); call ratchet_status to watch it. Only one job goes at a time on this machine.
    Reply to a start: ok, started, run_dir, root, pid, log_file, check (the estimate), message.
    On a wrong input: {"ok": false, "error": ..., "exit_code": 1 or 2}.
    run_dir: the finished run (needs run.json, labels.bin for marks, result/final.stl). Default: the last run of this server.
    source: `marks:FILE` (a marks file of the Refine page: one candidate for each mark, made by apply_marks) or `stl:FOLDER` (each STL file of the folder is a candidate). marks: a marks file, the same as source="marks:FILE".
    root: the ratchet folder (default: beside the run folder, `<run folder>-ratchet`). A second call with the same root goes on where the first stopped. If the rules, the mesh size or the inputs (the YAML, the region files) differ from the ratchet in the root, the call stops with the facts: use a new root; or rebaseline=true (judge the current parent again under the new inputs, keep the log); or overwrite=true (it moves the old files into earlier_NN).
    override_lock: run although another process holds the job lock of the root.
    stop_before_promote: the owner gate. The job stops before it promotes an accepted candidate (ratchet_status shows `pending`); ratchet_approve promotes it or drops it. Off by default; the YAML key `acceptance.owner_gate` does the same.
    yaml_path: the problem YAML, if the path in run.json is not valid.
    max_candidates: judge at most this many in this call. mesh_size_mm: the size of the baseline and of all candidates.
    Each candidate is judged against the current parent with `topopt printcheck` and, if a rule needs it, `topopt verify` at the mesh size of the baseline. Every limit comes from the YAML and the log names its source.
    A stress or stiffness rule on a marks candidate is judged like on any other candidate. If verify refuses the surface (an apply_marks step that fell back to the plain boolean seam), the log record has `errors.verify` and the rule check is `no_input`.
    """
    try:
        from topopt import ratchet as rt
        from topopt.config import UserError
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}", "exit_code": 2}
    d = _pick_dir(run_dir)
    ypath = str(_resolve(yaml_path)) if yaml_path else None
    src = _ratchet_source(source, marks)
    if not src:
        return {"ok": False, "exit_code": 1, "error": "no candidate source. Give source=\"marks:FILE\" (or marks=FILE) or "
                "source=\"stl:FOLDER\"."}
    rt_root = _resolve(root) if root else d.parent / f"{d.name}-ratchet"
    try:
        check = rt.check_report(d, ypath, src, root=rt_root)
    except UserError as exc:
        return {"ok": False, "error": str(exc), "exit_code": 1}
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}", "exit_code": 2}
    if estimate_only:
        return {"ok": True, "estimate_only": True, "run_dir": str(d), "run_dir_rel": _rel(d), "root": str(rt_root),
                "source": src, "check": check,
                "message": "Nothing started. Tell the user the rules, what is missing and the time, then call again with "
                           "estimate_only=false."}

    # ---- a real run: one job at a time, then a detached child ----
    verifying = _alive_verify()
    if verifying:
        return {"ok": False, "exit_code": 2, "run_dir": str(verifying[0]), "pid": verifying[1],
                "error": f"A verify job is still running (pid {verifying[1]}, folder {verifying[0]}). Only one job goes at a "
                         "time on this machine. Call verify_status to watch it. " + NO_STOP_VERIFY}
    ratcheting = _alive_ratchet()
    if ratcheting:
        return {"ok": False, "exit_code": 2, "run_dir": str(ratcheting[0]), "pid": ratcheting[1],
                "error": f"A ratchet job is still running (pid {ratcheting[1]}, folder {ratcheting[0]}). Only one job goes at "
                         "a time on this machine. Call ratchet_status to watch it. " + NO_STOP_RATCHET}
    alive = _alive_run()
    if alive and _is_busy(alive[0]):
        return {"ok": False, "exit_code": 2, "run_dir": str(alive[0]), "pid": alive[1],
                "error": f"An optimizer run is still running (pid {alive[1]}, folder {alive[0]}). Only one job goes at a "
                         "time on this machine. Wait for it to finish (run_status shows it). " + STOP_RUN_OPTION}
    problem = None if os.environ.get("TOPOPT_MCP_RATCHET_CMD") else _python_problem(python_exe())
    if problem:
        return {"ok": False, "error": problem, "exit_code": 2}
    args: list[str] = []
    for flag, val in (("--yaml", ypath), ("--max-candidates", max_candidates), ("--mesh-size", mesh_size_mm)):
        if val is not None:
            args += [flag, str(val)]
    for flag, on in (("--stop-before-promote", stop_before_promote), ("--overwrite", overwrite),
                     ("--rebaseline", rebaseline), ("--override-lock", override_lock)):
        if on:
            args.append(flag)
    cmd = _ratchet_command(d, rt_root, src, args)
    log_path = _ratchet_log_path(rt_root)
    made = _make_work_folder()
    if made:
        return {"ok": False, "error": made, "exit_code": 2}
    log(f"start: {' '.join(cmd)}")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "wb") as sink:
            proc = subprocess.Popen(cmd, cwd=str(workspace()), stdin=subprocess.DEVNULL, stdout=sink,
                                    stderr=subprocess.STDOUT, start_new_session=True, env=_child_env())
    except OSError as exc:
        return {"ok": False, "error": f"Could not start the ratchet job: {exc}", "exit_code": 2}
    _ratchet.update(proc=proc, root=rt_root, run_dir=d, started=time.time())
    try:
        _ratchet_lock_path().write_text(json.dumps({"pid": proc.pid, "root": str(rt_root), "run_dir": str(d),
                                                    "started": time.time()}))
    except OSError as exc:
        log(f"cannot write the lock file: {exc}")
    # a job that ends at once (for example a bad input or a broken Python) shows its output here
    end = time.time() + 1.0
    while time.time() < end and proc.poll() is None:
        time.sleep(0.05)
    if proc.poll() is not None and proc.returncode not in (0, 3):
        return {"ok": False, "exit_code": 2 if proc.returncode != 1 else 1, "run_dir": str(d), "root": str(rt_root),
                "error": f"The ratchet job ended at once with code {proc.returncode}: {_tail(log_path)}",
                "log_file": str(log_path)}
    return {"ok": True, "started": True, "run_dir": str(d), "run_dir_rel": _rel(d), "root": str(rt_root),
            "root_rel": _rel(rt_root), "pid": proc.pid, "source": src, "log_file": str(log_path),
            "log_file_rel": _rel(log_path), "log": str(rt_root / rt.LOG_FILE), "check": check,
            "message": "The ratchet job runs in the background. Call ratchet_status to watch it. " + no_callback("ratchet_status", RATCHET_END)}


@mcp.tool()
def ratchet_status(root: str | None = None, last: int = 5) -> dict:
    """Show the state of a ratchet: counts, the current parent, the last records with the reasons, and the job.

    root: the ratchet folder (default: the folder of the last ratchet that this server started).
    last: how many of the last records to show (each: id, decision ACCEPT or REJECT, the rules that failed, the
    reasons, `promoted`, `owner_approval`). The whole log is `ratchet.jsonl` in the root: one JSON line for each record.
    States: `starting`, `running` (the job process is alive; `file_state` holds the state of the file when it is a final
    one that waits for the process to end), `finished` (every candidate was judged, or max_candidates was
    reached), `waiting_for_owner` (the opt-in gate: `pending` lists the accepted candidates, ratchet_approve decides),
    `baseline_failed` (the start part could not be judged), `stopped_after_approval` (ratchet_approve ended the job after an approval), `failed` (the last process ended with an error, for example a
    refused start: `last_error` has the exit code and the text; or it stopped with no exit code and no final state, for
    example killed: `log_tail` has its last output) and `none` (no ratchet in the folder). The final states show only after the
    process has ended, so the next call (ratchet_approve or ratchet_run) is not refused as busy.
    Nothing calls you back when the job ends: call it again in this turn until the state is finished, waiting_for_owner, baseline_failed, stopped_after_approval or failed, or process_alive is false, or end the turn and tell the user what to ask for the result. A promise of a later report cannot be kept.
    Also: `job` (the facts of the job: state, exit code, wall time, peak memory), `process_alive`, `pid`, `mesh_size_mm`
    (one size for the whole ratchet), `baseline` (its failed rules and facts), `log_facts` (a skipped partial line).
    """
    try:
        from topopt import ratchet as rt
        from topopt.config import UserError
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}", "exit_code": 2}
    r = _resolve(root) if root else _ratchet.get("root")
    if r is None:
        return {"ok": False, "exit_code": 1, "error": "no ratchet root. Give root=... (the folder of ratchet_run)."}
    r = Path(r)
    alive = _alive_ratchet()
    pid = alive[1] if alive and alive[0] == r else None
    log_path = _ratchet_log_path(r)
    try:
        res = rt.status(r, last=max(1, int(last)), starting_pid=pid)
    except UserError as exc:
        if pid:
            return {"ok": True, "root": str(r), "state": "starting", "process_alive": True, "pid": pid,
                    "summary": "the job has started; the root is not there yet"}
        return {"ok": False, "error": str(exc), "exit_code": 1}
    res.update(ok=True, log_file=str(log_path) if log_path.exists() else None)
    if res["state"] == "failed":
        res["log_tail"] = _tail(log_path)
    return res


@mcp.tool()
def ratchet_approve(root: str | None = None, id: str | None = None, decline: bool = False) -> dict:
    """The opt-in owner gate: promote the accepted candidate that waits, or drop it.

    A ratchet started with stop_before_promote (or `acceptance.owner_gate: true`) stops before it promotes an accepted
    candidate. ratchet_status shows it as `pending` with state `waiting_for_owner`. The owner decides; this tool writes the
    decision to the log (`owner_approval`: given or declined). Ask the user first. It does not start a job: after an
    approval, call ratchet_run again with the same root to judge the next candidates against the new parent.
    root: the ratchet folder (default: the last ratchet that this server started).
    id: the candidate (for example c003). Default: the one that waits.
    decline: drop the candidate instead of promoting it.
    Reply: ok, id, owner_approval, promoted (the new parent folder, or null) and parent.
    """
    try:
        from topopt import ratchet as rt
        from topopt.config import UserError
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}", "exit_code": 2}
    r = _resolve(root) if root else _ratchet.get("root")
    if r is None:
        return {"ok": False, "exit_code": 1, "error": "no ratchet root. Give root=... (the folder of ratchet_run)."}
    alive = _alive_ratchet()
    if alive and alive[0] == Path(r):
        return {"ok": False, "exit_code": 1, "root": str(r), "pid": alive[1],
                "error": f"The ratchet job is still running (pid {alive[1]}). A candidate waits only after the job has "
                         "stopped. Call ratchet_status."}
    try:
        res = rt.approve(Path(r), id, decline=bool(decline), say=log)
    except UserError as exc:
        return {"ok": False, "error": str(exc), "exit_code": 1, "root": str(r)}
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}", "exit_code": 2, "root": str(r)}
    return {"ok": True, "root": str(r), **res}


@mcp.tool()
def build_regions(spec: dict | str, name: str | None = None) -> dict:
    """Build region STLs and a problem YAML from a part spec given as data.

    Use it when the user describes a part in words. Define the design space, fixed regions, loads (force vectors in N), keep-out regions (if any) and solid bodies (`solid`, forced solid).
    It already returns `check` and `warnings`. Call check_problem only after you edit the YAML, or for your own STLs. Next: state your assumptions, then run_topopt. If the user gave no load (no object, no mass and no force), ask before the run. A guard stop names an override: read the memory estimate against the RAM (a run above it can freeze the machine), raise the guard in the spec or take a larger voxel, and report what, why and the price.
    Writes <work folder>/runs/<name>/. Reply keys: resource topopt://replies/build_regions (the meaning of each key). Use yaml_rel with run_topopt. On a wrong spec: {"ok": false, "error": one line that names the key path}. ok is false also when the builder fails (builder_ok false): problems gives the builder errors, and the files stay.
    name: folder and file name; default is spec["name"].
    spec: a dict (or YAML text) with the schema of the part-to-topopt skill (spec.yaml): regions (design, fixed, load, solid, keep_out; primitives box, cylinder, sphere, capsule, bowl, prism, stl and screw_ring (see find_features), subtract), load_cases, material (E, nu, density or filament), settings (voxel_size may be "auto"; optional min_member_mm, filter_solid_neighbours, stop_rules (as in the YAML: a mapping turns the stop rules on, and time_budget_s, the user's own time budget, ends the run at the first pass after it; enabled: false keeps the rules off; the check warns when the estimate is above the budget) and perimeter_max_mm2: a bound on the surface area of the part in mm2). Units mm, N, MPa. settings.volume_fraction (default 0.25) is the mean density of the FREE design voxels; the fixed, load and solid voxels are solid on top. The YAML keeps your value. A mass in grams that the user asks for goes to settings.target_mass_g in place of volume_fraction (one of the two keys; both is an error): the engine derives the fraction from the grid.
    The report and `derived.volume` give the free and the forced-solid voxels, the target volume in mm3 and the target mass in g (the same numbers as check_problem).
    Loads: a force entry can also have `moment` in N mm and `at` in mm, and a case can have `pressures` of `{region, value_mpa}`.
    A `fixed_axes` mapping (fixed name -> axes) makes a roller of a fixed region.
    A `springs` list of `{region, stiffness: [kx, ky, kz]}` (N/mm, total; or `stiffness_per_area`) makes an elastic support on a load region; `fixed` may then be empty.
    A `joints` list of `{region, to: [x, y, z], section: {kind: round, d: ..}}` (mm; kinds round, tube, rect, props) bonds the faces of a load region to a rigid body and joins it with a beam to the fixed point `to`; `fixed` may then be empty.
    A flat face of a region whose plane is not on a voxel face (a voxel belongs to a region when its centre is inside) moves to the nearest voxel face. When such planes change the part, the report has ONE `note:` line (also in warnings) with the voxel size, the grid origin, an item for each plane (the region, the plane, larger or smaller than your shape) and the options. check.region_planes has all planes, with offsets in mm and voxels, and `changes_part` (false when the layer is under a region of higher priority or outside the part).
    Items in one region join (union).
    Optional spec key `mirror`: a list of up to 3 axis-aligned planes with distinct axes, each {normal: x|y|z, at_mm: <mm>}, align: auto|face|centre optional. The tools return facts about it (parity, orphans, load asymmetry).
    """
    try:
        from topopt.regions_builder import build_in_workspace
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}"}
    refusal = _work_folder_refusal(workspace() / "runs", anywhere=True)
    if refusal:
        return {"ok": False, "error": refusal}
    return build_in_workspace(spec, workspace(), name)


# --- case: the brief, its status and the soft gate of run_topopt (CONTRACTS.md section 30) ---------------------------

@mcp.tool()
def record_brief(yaml_path: str, text: str, assumptions: list[dict] | None = None,
                 criteria: list[dict] | None = None) -> dict:
    """Record the brief of a problem YAML: what the user wants and the assumptions that you state. Say it, then go on.

    A case is one problem YAML with the STL files it names, one brief and many runs. This call makes the case
    (the files `<yaml stem>.case.json` and `<yaml stem>.case-log.jsonl` beside the YAML) or replaces its brief.
    The case keeps the SHA-256 of the YAML and of each region file with the brief, and the values of the method keys
    of the YAML. If a file or a requirement key changes later, `case_status` and `run_topopt` say that the brief is
    stale and name it. For a case with no brief or a stale brief, `run_topopt` returns `awaiting_user` with an override.
    A change of a method setting (for example `settings.solver` or `settings.voxel_size`) is a fact and not stale:
    `run_topopt` goes on, and `brief_facts` names the key with both values, so that you can report it.

    yaml_path: the problem YAML in the work folder. A YAML outside the work folder (an example in the code
        folder, for example) gets no case file: the reply gives the fact and the options. A YAML in the code
        folder of a source checkout is recorded, with a fact and an option (a work folder is the better place).
    text: the brief in plain words: the goal, the loads and where they come from, the mounting, the material,
        the critical sizes.
    assumptions: a list of {"text": ..., "origin": ...}. `origin` is `user` (the user said it), `inferred`
        (you concluded it) or `default` (a default value). The YAML `assumptions` list is a list of strings
        that run.json copies; the reply names the texts that differ from the brief.
    criteria: what the result has to show, in the same form (optional).
    """
    from topopt import case
    yaml = _resolve_yaml(yaml_path)
    if not yaml.is_file():
        return {"error": _yaml_not_found(str(yaml))}
    return case.record_brief(yaml, workspace(), text, assumptions, criteria, code_folder=REPO)


@mcp.tool()
def case_status(yaml_path: str) -> dict:
    """Show the case of a problem YAML: the brief, the input hashes, the earlier runs and what is stale. Changes nothing.

    `brief_stated` is true when a current brief is recorded. `stale_files` names the files that changed since the
    brief. A change of a comment, of the format or of a method setting (for example `settings.solver`) in the YAML is a
    fact, not stale. `inputs` has the SHA-256 of the YAML and of each region file now. `runs` lists the runs that
    run_topopt started for this case, newest first. Each has its state (read from the run folder), `stale` and
    `stale_files` for its own inputs (a run is stale after any change of the YAML content, a method setting too), and
    facts, for example that the folder now holds another run. `gate` says what run_topopt does now.

    yaml_path: the problem YAML (relative paths start at the work folder).
    """
    from topopt import case
    yaml = _resolve_yaml(yaml_path)
    if not yaml.is_file():
        return {"error": _yaml_not_found(str(yaml))}
    return case.status(yaml, workspace())


def _read_preferences() -> dict:
    """The key `preferences` of `environment`: the preferences file of the work folder as a fact (a broken file is a fact, not a crash)."""
    from topopt import preferences
    try:
        return preferences.read(workspace())
    except Exception as exc:     # read() does not raise; this keeps environment whole if it ever does
        return {"path": str(preferences.path_of(workspace())), "exists": None, "problems": [f"internal error: {type(exc).__name__}: {exc}"]}


@mcp.tool()
def environment(yaml_path: str | None = None) -> dict:
    """Facts about this machine and the solver backends. Changes nothing.

    Use it to answer "can this machine run it, and how fast?", or before a long run. Tell the user the numbers that matter.
    yaml_path: optional problem YAML (relative paths start at the work folder). With it, `problem` holds the time estimate for each backend (model values, not promises).
    Reply keys: resource topopt://replies/environment (the meaning of each key).
    If TOPOPT_PYTHON names another Python, the data is for that Python (run_interpreter says so).
    `recommendations` are offers as data: when a faster backend is possible but not installed (pypardiso on Linux x86_64, the Accelerate shim on a Mac), each has a reason, the command and license_note (speedup_factor when it is measured).
    This tool installs and builds nothing. If the user agrees, you run the command. Say the reason (and the speed-up) first. `gpu_solver` is a fact, not an offer. `auto` takes a GPU that passes a short self-test, and goes on with the CPU solver (`auto_choice.cpu_solver`) from the last checkpoint if gpu-mg fails during a run. `settings.solver: gpu-mg` asks for it on any OpenCL device (run_topopt has no solver argument: write it in the YAML) and has no fallback. The OpenCL device list and the self-test run in a child process (limit 60 s). At the limit `opencl.problem` says so, and `auto` takes the CPU rule; TOPOPT_AUTO_GPU=0 skips the wait.
    A gpu-mg run that stops with "did not reach the relative residual" lists its options in the error (settings.gpu_tol, settings.gpu_backward_error, settings.gpu_max_iter in case B, a stiffer start design, another solver); the residual history is in gpu_mg.jsonl of the run folder.
    gpu-mg takes springs, joints and per-axis supports (a fixed entry with axes), alone or together. In case C (no 64-bit float on the device) the joint term is computed in 32-bit float and the inner PCG removes the rigid modes of each body and 6 harmonic columns for each bonded patch, and solves them in 64-bit on the host: check prints the beam over the bond stiffness with the cases that were measured on pocl and on one Intel Arc iGPU (one host each; revision 133e5a1; a cube that only a joint holds solved at 9.9e-12 and above with the rigid modes removed, and at 3.3e-7 and above without; a part with two tube joints and no fixed region solved with the harmonic columns and stopped without them) and the memory of the columns, which the memory estimate does not count. These are facts, not a limit. A run that stops gives the ratio and the options (another solver, gpu_case B on a device with 64-bit float, a stiffer beam).
    """
    path = None
    if yaml_path:
        path = _resolve_yaml(yaml_path)
    missing = path is not None and not path.is_file()
    if missing:
        path = None
    exe = python_exe()
    data = None
    probe_error = None
    if exe != sys.executable:
        # Runs go to another Python (TOPOPT_PYTHON): ask that one, read-only.
        data, probe_error = _probe_environment(exe, path)
    if data is None:
        try:
            from topopt.environment import environment_data
        except ImportError as exc:
            return {"ok": False, "error": f"Cannot import topopt: {exc}"}
        try:
            data = environment_data(path)
        except Exception as exc:
            return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}
    data["run_interpreter"] = {"executable": exe, "server_executable": sys.executable,
                               "report_is_for": data["python"]["executable"], "probe_error": probe_error}
    if missing:
        data["problem"] = {"ok": False, "error": f"YAML file not found: {_resolve(yaml_path)} (work folder: {workspace()})"}
    data["work_folder"] = workfolder.resolve().facts()
    data["preferences"] = _read_preferences()
    data["server"] = {"pid": os.getpid(), "started_utc": STARTED_UTC}
    data["other_servers"], note = _other_servers()
    if note:
        data["other_servers_note"] = note
    data["ok"] = True
    return data


# --- preferences (CONTRACTS.md section 10.2) -----------------------------------------------------------------------------


@mcp.tool()
def save_preferences(print: dict | None = None, material: dict | None = None, quality: dict | None = None,
                     outputs: list[str] | dict | None = None, filaments: dict | None = None) -> dict:
    """Keep the answers of the user (print bed, material, filaments, quality, outputs) for the next runs, in `topopt-preferences.yaml` in the work folder. It writes only that file.

    Use it once, after the user answered. `environment` shows the file in `preferences`. The server reads the file at each call.
    A preference does not change a run by itself: `run_topopt` and `check_problem` fill nothing. `build_regions` takes a `material`, `print` or `settings.voxel_size` that the spec does not have at all, writes it into the YAML and says so in `preferences_facts`; a block that the spec gives, also a part of one, stays as it is. `check_problem` names a difference in `preferences_facts`.
    Sections (each optional; checked like the problem YAML):
    - print: printer_model (a name or id from `find_printer`; it gives the bed and type unless the YAML does), printer_type (fdm, sla, lpbf, sls), bed_mm (one number or [x, y] in mm), build_direction (+x to -z, or 3 numbers), self_supporting_angle_deg. A print limit (min_wall_mm) belongs to one part: it is refused.
    - material: E (MPa), nu, density (g/cm3), filament (from `find_material`; gives E and density), name (free text, for example "PETG").
    - filaments: available (names or ids of the filaments the user has).
    - quality: voxel_size (mm).
    - outputs: renders (render_part), step (export_step), verify (verify_run). The STL and the part card are made in every run. [] is none of the three.
    A section you do not give stays; one you give is merged key by key: a key with null is removed, and {} removes the section (outputs too). An unknown key stays in the file. With no section it makes an empty file if none exists (the user kept the defaults).
    Returns: ok, path, values, changed, created, kept_not_used, problems, note. A wrong name, key or value gives {"ok": false, "error": ...} with the key, the value and the rule; nothing is written. It refuses a file that is not valid YAML and a folder that cannot be written. There is no CLI command: edit the file.
    """
    try:
        from topopt import preferences
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}"}
    folder = workfolder.resolve()
    if folder.fix_needed:
        return {"ok": False, "error": " ".join(folder.fix_needed)}
    refusal = _work_folder_refusal(folder.path, anywhere=True)
    if refusal:
        return {"ok": False, "error": refusal}
    try:
        return preferences.save(folder.path, print, material, quality, outputs, filaments)
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


# --- printer profiles (CONTRACTS.md section 18.7) ------------------------------------------------------------------------


@mcp.tool()
def find_printer(name: str) -> dict:
    """Find a 3D printer by name in the catalog that ships with topopt and in the user printers of the work folder (`topopt-printers.yaml`). It reads; it changes nothing.

    The user only names the printer; the tool knows the print parameters. Use it when the user says which printer they have ("Bambu P1S", "my Prusa MK4S"), before you ask for a bed size or a process.
    name: the model as the user says it. Case, punctuation, the maker name and the words "my" and "printer" do not matter ("p1s", "Bambu Lab P1S" and "bambu p1 s" are one printer).
    Returns: ok, status, query, catalog_version, matched_by, origin, profile, derives, candidates, differs, suggestions, vendors, user_file, note.
    - "match": `profile` is the printer: id, name, origin (`catalog` or `user`), bed_mm, build_height_mm, nozzle, temperatures, enclosure, nozzle material, extruder, and the source of the values (source_url, opened, tag, field_sources, note). A value that the source does not give is null (unknown): say so. `derives` says what `print.printer_model: <id>` sets in a run (the bed and the printer type, when the YAML does not give them) and what the profile cannot give.
    - "ambiguous": `candidates` and `differs` (the facts on which they differ). Ask the user which one; do not pick for them. `matched_by` is "partial" when the text is only a part of a name (one candidate too: a similar model can have other numbers).
    - "none": `suggestions` ("did you mean": never taken by itself) and a `note` on how to go on: look up the maker page and `save_printer` the values, or ask the user.
    `user_file`: path, exists, entries, problems. A problem in the file is a fact; the shipped catalog still works.
    Next: show the user the derived values, then `save_preferences(print={"printer_model": id})` keeps the printer, or write `print: {printer_model: <id>}` in the YAML. A bed_mm or printer_type in the YAML wins. With a printer, the print report also compares the part height with its build height.
    """
    try:
        from topopt import printers
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}"}
    try:
        return printers.resolve(name, printers.catalog_for(workspace())).to_dict()
    except printers.CatalogError as exc:
        return {"ok": False, "error": f"The printer catalog cannot be read: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


@mcp.tool()
def save_printer(profile: dict) -> dict:
    """Keep a printer that is not in the catalog, with its facts and their sources, in `topopt-printers.yaml` in the work folder. It writes only that file.

    Use it after `find_printer` says "none": look up the maker page (web search), show the user the values with the page address, and call this tool after the user agrees. A value that you cannot find stays null (unknown): ask the user for it, and leave it null if they do not know; a guess would look like a fact. The server reads the file at each call, so the printer is live at once.
    profile: one printer as a dict. Needed: id (lower-case slug, for example "voron-trident"), vendor, model, printer_type ("fdm", "sla", "lpbf", "sls"), bed_mm ([x, y] in mm), and a source: source_url (the maker page) or source "user" (the user said the values).
    Optional (null or left out = unknown): aliases (list), build_height_mm, nozzle_mm, nozzle_options_mm (list), max_nozzle_c, max_bed_c, enclosure ("open", "enclosed", "heated"), chamber_max_c (only with "heated"), stock_nozzle_material ("brass", "stainless", "hardened_steel", "tungsten_carbide"), extruder ("direct", "bowden"), opened (YYYY-MM-DD, the day the page was read), tag ("maker doc", "third party", "user"), field_sources ({field: page address or "user"}), note.
    The entry is checked like the shipped catalog. The same id replaces a user entry; the id of a shipped printer is refused with another id to use, and a name or alias of a shipped printer with another model name or alias. Other entries stay.
    Returns: ok, path, profile, created, replaced, note, problems (of the rest of the file). A wrong argument name or value gives {"ok": false, "error": ...}; nothing is written. A file that is not valid YAML is not changed. There is no CLI command: edit the file.
    """
    try:
        from topopt import printers
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}"}
    folder = workfolder.resolve()
    if folder.fix_needed:
        return {"ok": False, "error": " ".join(folder.fix_needed)}
    refusal = _work_folder_refusal(folder.path, anywhere=True)
    if refusal:
        return {"ok": False, "error": refusal}
    try:
        return printers.save_user_printer(folder.path, profile)
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


# --- filament profiles (CONTRACTS.md section 18.8) -------------------------------------------------------------------------


@mcp.tool()
def find_material(name: str) -> dict:
    """Find a 3D printing filament by name in the catalog that ships with topopt and in the user filaments of the work folder (`topopt-filaments.yaml`). It reads; it changes nothing.

    The user only names the filament; the tool knows the numbers of the data sheet. Use it when the user says which spool they have ("Bambu PLA Basic", "Prusament PETG"), before you ask for a modulus or a density.
    name: the product as the user says it. Case, punctuation, the maker name and the word "my" do not matter ("polylite abs" and "Polymaker PolyLite ABS" are one filament). A family alone ("PLA", "PETG") names no product: the answer is "ambiguous" with every filament of the family.
    Returns: ok, status, query, catalog_version, matched_by, origin, profile, derives, candidates, differs, suggestions, makers, user_file, note.
    - "match": `profile` is the filament: id, name, family, sheet_version, the tensile modulus and strength in the layer plane (xy) and across the layers (z), density, temperatures, enclosure, hardened nozzle, drying, and the source of the values (source_url, opened, tag, field_sources, note). A value that no source states is null (unknown): say so. `nu` is null for every filament, because no data sheet gives it. `derives` says what `material.filament: <id>` sets in a run (E is the lower of the XY and Z modulus, taken from the bending modulus of the sheet when the sheet gives no tensile modulus; density is the one of the sheet, the middle of a range that the sheet gives, or else the median of the filaments of the family, labelled assumed; nu stays the assumed 0.35; a key in the YAML wins) and what the profile cannot give.
    - "ambiguous": `candidates` and `differs` (the facts on which they differ). Ask the user which one; do not pick for them.
    - "none": `suggestions` ("did you mean": never taken by itself) and a `note` on how to go on: look up the maker data sheet and `save_material` the values, or ask the user.
    The numbers of a data sheet describe printed test bars of one maker and one set of print settings, not the part of the user.
    `user_file`: path, exists, entries, problems. A problem in the file is a fact; the shipped catalog still works.
    Next: show the user the derived values. `printer_materials` shows what the printer of the user can print; `check_problem` gives the same fit for a YAML with `material.filament`, in its key `material`. `save_preferences(material={"filament": id})` keeps the filament, or write `material: {filament: <id>}` in the YAML.
    """
    try:
        from topopt import materials_catalog
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}"}
    try:
        return materials_catalog.resolve(name, materials_catalog.catalog_for(workspace())).to_dict()
    except materials_catalog.CatalogError as exc:
        return {"ok": False, "error": f"The filament catalog cannot be read: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


@mcp.tool()
def save_material(profile: dict) -> dict:
    """Keep a filament that is not in the catalog, with its facts and their sources, in `topopt-filaments.yaml` in the work folder. It writes only that file.

    Use it after `find_material` says "none": look up the data sheet of the maker (web search), show the user the values with the page address, and call this tool after the user agrees. A value that you cannot find stays null (unknown): ask the user for it, and leave it null if they do not know; a guess would look like a fact. The server reads the file at each call, so the filament is live at once.
    profile: one filament as a dict. Needed: id (lower-case slug, for example "acme-pla-plus"), maker, product, family (PLA, PETG, ABS, PA6-CF, ...), and a source: source_url (the data sheet) or source "user" (the user said the values). A run needs a modulus (the tensile E_xy_mpa or E_z_mpa, MPa; a sheet that gives only a bending modulus gives bending_modulus_xy_mpa or bending_modulus_z_mpa, and E is taken from it) and a density (density_g_cm3, g/cm3) from it, or from the YAML.
    Optional (null or left out = unknown): aliases (list), sheet_version, E_xy_mpa, E_z_mpa, bending_modulus_xy_mpa, bending_modulus_z_mpa, strength_xy_mpa, strength_z_mpa, strength_kind ("yield", "peak", "not stated"), elongation_xy_pct, density_range_g_cm3 ([min, max] of a sheet that gives a range, with density_g_cm3 null), hdt_18_c, hdt_045_c, nu, nozzle_c ([min, max] in C), bed_c ([min, max]), enclosure ("no", "recommended", "required"), chamber_c_min, hardened_nozzle (true or false), drying (text), direct_drive (true), abrasive (true for fibre), soft (true for TPU), conditioned (wet values of a polyamide: condition, strength_xy_mpa, strength_z_mpa, E_xy_mpa, E_z_mpa), opened (YYYY-MM-DD, the day the page was read), tag ("maker TDS", "maker page", "third party", "user"), field_sources ({field: page address or "user"}), enclosed_note (what the sheet says about printing in an enclosed printer), note.
    The entry is checked like the shipped catalog. The same id replaces a user entry; the id of a shipped filament is refused with another id to use, and a name or alias of a shipped filament with another product name or alias. Other entries stay.
    Returns: ok, path, profile, created, replaced, note, problems (of the rest of the file). A wrong argument name or value gives {"ok": false, "error": ...}; nothing is written. A file that is not valid YAML is not changed. There is no CLI command: edit the file.
    """
    try:
        from topopt import materials_catalog
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}"}
    folder = workfolder.resolve()
    if folder.fix_needed:
        return {"ok": False, "error": " ".join(folder.fix_needed)}
    refusal = _work_folder_refusal(folder.path, anywhere=True)
    if refusal:
        return {"ok": False, "error": refusal}
    try:
        return materials_catalog.save_user_filament(folder.path, profile)
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


@mcp.tool()
def printer_materials(printer: str | None = None) -> dict:
    """Which filaments can this printer print? Every current filament of the catalog and the user file, grouped by state, with the reasons (older sheets of a listed product are counted in `note`). It reads; it changes nothing.

    printer: a printer as the user says it (a name or id; see `find_printer`). Without it, the printer of the preferences file (`print.printer_model`) is used; if there is none, the reply is {"ok": false, "error": ...} that says how to name one.
    States (facts about the stated values, not a stop; no run is refused because of them): `ok` (every stated value fits), `care` (it can work, and the person has to do or know something: fit a hardened nozzle or a chamber heater, take the lower part of a temperature range), `no` (a stated need is above a stated limit of the printer as it ships, for example an enclosed printer, a nozzle temperature or a bed temperature; the user may have changed the printer), `unknown` (a value is missing on one side; the reason names it).
    Returns: ok, printer (id, name, origin, from), catalog_version, counts, groups, available, user_file, note.
    - groups: `ok`, `care`, `no`, `unknown`, each a list of {id, name, family, origin, available, source, reasons}. `reasons` lists the checks that are not ok (check, state, text); the text names both values that it compares. `source` is the data sheet of the filament; the source of the printer is `printer.from`. The reply is long (about 70 filaments): read the groups that matter and name the few that the user asked for.
    - available: the filaments of `filaments.available` of the preferences, each with its state; they are also marked in the groups. An id that is not found now is left out and named in `not_used`; an older sheet has its state and a note with the newer id.
    Say it as a fact with the reason ("PLA Basic: ok on a P1S", "ABS on an A1 mini: no, the maker asks for an enclosed printer"). The user decides. A filament that is not listed: `find_material`, then `save_material`.
    """
    try:
        from topopt import materials_catalog, preferences, printers
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}"}
    try:
        folder = workspace()
        saved = preferences.read(folder)
        values = saved.get("values") or {}
        named = printer.strip() if isinstance(printer, str) and printer.strip() else (values.get("print") or {}).get("printer_model")
        if not named:
            lost = next((t for t in saved.get("not_used") or [] if t.startswith("'print.printer_model'")), None)
            why = (f"the printer of the preferences file is not found now ({lost})" if lost
                   else "the preferences file has no print.printer_model")
            return {"ok": False, "error": f"No printer is named, and {why}. Give printer (a name or an id; `find_printer` finds one), or keep "
                                          "the printer with save_preferences(print={\"printer_model\": id}); `save_printer` adds a printer that is not listed."}
        res = printers.resolve(named, printers.catalog_for(folder))
        if res.status != "match":
            d = res.to_dict()
            return {"ok": False, "status": res.status, "query": res.query, "candidates": d["candidates"], "suggestions": d["suggestions"],
                    "error": f"printer {res.query!r} is not one printer of the catalog or the user printer file. Call `find_printer` with the name, "
                             "and ask the user which printer it is."}
        out = materials_catalog.printer_fit(res.profile, materials_catalog.catalog_for(folder), (values.get("filaments") or {}).get("available") or [])
        out["printer"]["named_by"] = "argument" if printer and str(printer).strip() else "preferences"
        return {"ok": True, **out}
    except (printers.CatalogError, materials_catalog.CatalogError) as exc:
        return {"ok": False, "error": f"A catalog cannot be read: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


# --- print check (CONTRACTS.md section 32) -------------------------------------------------------------


@mcp.tool()
def print_check(path: str | None = None, yaml_path: str | None = None, print_settings: dict | None = None,
                resolution_mm: float | None = None, strict: bool = False) -> dict:
    """Check a finished part for printing. Geometry only: no solve, no run. It gives the same report as `topopt printcheck`.

    path: an STL file or a run folder (it has result/final.stl); relative paths start at the work folder. With no path, the last run is used. It works on any STL (final.stl, a nominal.stl, an edited.stl): no metrics.json and no YAML are needed. About 2 to 5 s for an 85,000-triangle part.
    yaml_path: a problem YAML whose `print` block gives the settings and limits. A run folder already has the `print` block of its run. print_settings: a dict with keys of the `print` block, for example {"min_wall_mm": 1.2, "bed_mm": [250, 210]}; it wins over the YAML.
    resolution_mm: the size of the smallest member that the mesh holds: the voxel size (a run folder gives it), or 0 for exact CAD. The default is the voxel size of the run, else the median mesh edge.
    strict: opt in to the gate. Without it the tool only reports.
    `gate`: with strict, `state` is fail only when a check that has a limit fails (unknown never sets it); `exit_code` is then 4 and `override` names the way on. No run stops on it.
    On an input that cannot be read: {"ok": false, "error": ...}. A failed check is data, not an error.
    A print pass is not a structural pass. A finished run already has this report as `result/print_check.json` (run_result shows its counts): call this tool again for other limits or another direction.
    Each check has `id`, `state` (pass, fail or unknown), `reason`, `value`, `limit` (`value`, `key` and `source`: run.json, yaml, option, profile:<id>, assumed or none) and `method`. Checks: mesh_closed (the STL as a float32 file: open and non-manifold edges, winding, volume), bodies, closed_voids (count and volume), bed_fit, overhang, support_volume, plate_contact, wall_thickness. The limits are print-block keys: bed_mm, max_bodies, min_wall_mm, min_wall_area_pct, max_closed_void_mm3, max_overhang_area_mm2, max_support_volume_mm3, min_plate_contact_mm2, and the V2 keys build_direction, printer_type and self_supporting_angle_deg. printer_model (a printer of `find_printer`) gives the bed, the type (source profile:<id>) and the build height: bed_fit then also compares the part height with it. A check with no limit is `unknown` and its value is a fact. The thickness is an estimate: by the rule of this check it passes only if the limit is at least 3 x the resolution; a thin wall that is measured fails at any resolution. When it is `unknown` only for that rule, `value.pass_needs` gives the resolution that a pass needs (`resolution_mm_at_most`, and `voxel_count_factor` for a run folder: an estimate of the voxels of that grid against this one).
    Also: `summary`, `settings` (with `from`: where each source came from), `input` (path, sha256), `timing_s`, `definitions`.
    `partial`, `coverage`: `partial` is true when a check that is asked for stopped with an error (its record has state `unknown` and `error`; the other checks go on) or has a limit and could not decide. `coverage` has `ran`, `failed` (name, reason) and `not_run` (name, reason, `asked`); a check that is not asked for (the support volume of a powder-bed printer) does not make it partial.
    """
    try:
        from topopt import printcheck
        from topopt.config import UserError
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}"}
    try:
        if path:
            target = _resolve(path)
        else:
            target = _pick_dir(None)
            if not (target / "result" / "final.stl").is_file():
                raise UserError("no run yet: give path (an STL file or a run folder)")
        yaml = _resolve_yaml(yaml_path) if yaml_path else None
        return printcheck.check(target, yaml_path=yaml, print_settings=print_settings,
                                resolution_mm=resolution_mm, strict=bool(strict), contain_errors=True)
    except UserError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}


# --- STEP export (CONTRACTS.md section 42) -------------------------------------------------------------------------------------------


@mcp.tool()
def export_step(run_dir: str | None = None, part_only: bool = False, tolerance_mm: float | None = None,
                merge_planes: bool = True, schema: str = "AP214", out: str | None = None, estimate_only: bool = False) -> dict:
    """Write a faceted STEP file of a finished run (extra step: OpenCASCADE). Post-only: no pass runs and no input file changes. Same as `topopt step`.

    The call is synchronous: it returns when the file is written, and a large export holds the call for minutes. Call it with estimate_only first for a large mesh: the estimate gives the time before the work starts.
    Cost (Linux x86_64): the cost follows the number of faces after the simplification, and that follows the part. A hook gives 9,000 to 36,000 faces with the default tolerance: 10 to 40 MB, 12 to 40 s, about 1 GB. A larger part with 85,000 faces took 102 s and 1.8 GB; another with 356,000 faces took 613 s (10 minutes) and 5.8 GB, and the file was 402 MB. tolerance_mm=0 keeps every triangle: about 1.1 KB, 1 ms and 16 KB of memory for each triangle (322,000 triangles: 363 MB, 6 minutes, 5.5 GB).
    run_dir: the run folder (relative paths start at the work folder; default: the last run). It wraps result/final.stl, or result/part_only.stl with part_only. Output: result/final.step (result/part_only.step) and its report next to it, result/final.step.json (result/part_only.step.json). out is another file path or an existing folder; the report is then next to that file (<file>.json), so the run folder gets no file. An existing file is replaced only when it is a .step or .stp file; the input STL and any other existing file give an error with the options.
    estimate_only: simplify, count the faces and estimate size, time and memory; nothing is written. It works without OpenCASCADE.
    A failed check is a fact and the file is still written (exit_code 3). Without the extra the reply is {"ok": false, "missing_extra": "step", "install_command": ..., "error": ...}. Any other failure is {"ok": false, "error": ..., "exit_code": ...}.
    tolerance_mm: the simplification of the mesh (manifold3d), the largest move of the surface in mm; 0 = none. Default: the voxel size of the run / 20 (0 when the run folder holds no voxel size). A larger value gives fewer faces and a smaller, faster file; a value that removes every triangle is an error and no file is written. merge_planes: coplanar faces join into one planar face (the frozen plates). schema: AP214 (default) or AP242 (ap-242 and ap242 also work).
    The faces are the facets of the mesh, so a CAD program can cut holes in the solid and join it to other parts; it cannot edit the surfaces.
    The reply is the report: input (triangles, volume), settings (tolerance_mm and where it comes from), simplify, merge_planes, build (bodies, voids, and the check of the model in memory), model (the file read back: solids, faces, edges), volume (STEP against STL), deviation (sampled distance in both directions: max and mean, in mm), checks (read_back, topology, before_write, valid_solid, closed, units, schema: pass or fail with a reason; none has a threshold), checks_failed, options, timing_s, peak_rss_bytes.
    The reply of any other failure has the first line of stdout when there is one, and "signal" when the child process was killed.
    """
    try:
        target = _pick_dir(run_dir)
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}
    cmd = [python_exe(), "-m", "topopt", "step", str(target), "--json", "--schema", str(schema)]
    if part_only:
        cmd.append("--part-only")
    if tolerance_mm is not None:
        cmd += ["--tolerance-mm", str(tolerance_mm)]
    if not merge_planes:
        cmd.append("--no-merge-planes")
    if out:
        cmd += ["--out", str(_resolve(out))]
    if estimate_only:
        cmd.append("--estimate-only")
    made = _make_work_folder()
    if made:
        return {"ok": False, "error": made}
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=str(workspace()), env=_child_env())
    except OSError as exc:
        return {"ok": False, "error": f"Python {python_exe()} did not start: {exc}"}

    def failure(text: str) -> dict:
        """A failure reply: the text, the exit code (and the signal when the child was killed), the first line of stdout if any."""
        reply = {"ok": False, "error": text, "exit_code": done.returncode}
        if done.returncode < 0:
            try:
                reply["signal"] = signal.Signals(-done.returncode).name
            except ValueError:
                reply["signal"] = str(-done.returncode)
        first = (done.stdout.strip().splitlines() or [""])[0]
        if first:
            reply["stdout_first_line"] = first
        return reply

    last_line = (done.stderr.strip().splitlines() or ["?"])[-1]
    if done.returncode in (0, 3):
        try:
            report = json.loads(done.stdout)
        except ValueError:
            return failure("topopt step gave no JSON: " + last_line)
        report.update({"ok": True, "exit_code": done.returncode})
        for key in ("run_dir", "report_path"):
            if report.get(key):
                report[key + "_rel"] = _rel(report[key])
        if report.get("output"):
            report["output"]["path_rel"] = _rel(report["output"]["path"])
        return report
    if done.returncode == 1:      # a usage or input error: the text is the message of topopt, with its options
        text = done.stderr.strip()
        text = text[len("error: "):] if text.startswith("error: ") else text
        if "optional extra `step`" in text:
            from topopt.step import OCP_DISTRIBUTION, OCP_PIN, EXTRA
            from topopt.uv_command import uv_command
            return {"ok": False, "missing_extra": EXTRA, "install_command": f'{uv_command()} pip install -e ".[{EXTRA}]"',
                    "install_command_no_checkout": f'{uv_command()} pip install "{OCP_DISTRIBUTION}=={OCP_PIN}"', "error": text,
                    "exit_code": 1}
        return failure(text or "topopt step exited with code 1")
    # exit code 2 (an internal error: the last line of stderr is the exception), another code, or a signal: no message of topopt
    return failure(last_line if done.stderr.strip() else f"topopt step stopped with exit code {done.returncode}")


def _probe_environment(exe: str, yaml: Path | None):
    """Run `python -m topopt.environment` with the Python of the runs. Returns (data, error)."""
    cmd = [exe, "-m", "topopt.environment"] + ([str(yaml)] if yaml else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           stdin=subprocess.DEVNULL, cwd=_read_cwd(), env=_child_env())     # a read: it makes no folder
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"Python {exe} did not answer: {exc}"
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["?"])[-1]
        return None, f"Python {exe} cannot run topopt.environment ({tail})"
    try:
        return json.loads(r.stdout), None
    except ValueError:
        return None, f"Python {exe} gave no JSON"


@mcp.tool()
def cleanup_list(run_dir: str | None = None, out_dir: str | None = None, options: dict | None = None, overwrite: bool = False,
                 estimate_only: bool = False, detail: bool = False, yaml_path: str | None = None, root: str | None = None) -> dict:
    """List the geometric cleanup candidates of a finished part as a marks file, in a NEW folder: facts and options, no edit, no verdict.

    CONTRACTS.md section 29.3. The tool scans result/final.stl with the detectors of the pipeline stage s7_cleanup and writes marks.json (format 1, mm: apply_marks, the Refine page and the marks: source of ratchet_run read it) and cleanup.json (the full report). It lists loose bodies (no fixed and no load voxel), dead-end spurs, thin bumps, uneven struts (SMOOTH) and flat webs (FAIR). When a loose body exists the file holds the bodies only: judge them, then list the acc_01 of that ratchet root for the features. The list judges nothing: ratchet_run(marks=marks_rel) applies each mark and judges it with the rules of the YAML. The run folder and every input stay as they are.
    Time: about 2 s on the short hook and 8 s on the ladle. The call runs in the server and finishes if the client stops waiting (Codex waits 60 s for a tool by default); then read cleanup.json, or call again with overwrite=true. A grid above max_voxels or the memory limit gives the facts and the options (pitch_mm, max_gb), not a crash.
    A wrong input gives {"ok": false, "error": ...}.
    run_dir: a finished run, or an acc_NN folder of a ratchet root (needs run.json, labels.bin, result/final.stl). Default: the last run of this server.
    out_dir: the new folder. Default: <run folder>-cleanup beside the run folder (beside the root for an acc_NN folder). Refused, with the reason and the options: a run folder or a folder inside one, a ratchet root or a folder inside one, a folder with an input file, a symbolic link, a folder with files that is not an earlier cleanup folder.
    estimate_only: the memory and time estimate of the detector and the judge; no search, no file.
    options: pitch_mm (0.5), max_candidates (10; null lists all), kinds (body, spur, bump, member), order ("pipeline" or "stress"), max_voxels, max_gb, and the limits of the detectors (spur_*, bump_*, member_*, mild_*, web_flatness, opening_radius_mm, frozen_near_mm, frozen_max_share, near_frozen_mm); cleanup.json gives each default and its source line in the pipeline. The constants of the method are not settable.
    overwrite: write again into an earlier cleanup folder. detail: the full report in the reply.
    yaml_path: the YAML of the run when run.json names a file that is not there. root: a ratchet root; its log gives the measured judge time for the estimate.
    Each mark has the detector, the location, the operator, the thresholds used, the measured sizes and, when verify/fields.npz is there, the stress near it. The facts also say how well the detectors fit this part (applicability).
    Reply keys: ok, counts, candidates (id, type, operator, detector, confidence, centre_mm, detected_volume_mm3, length_mm, nearest_frozen_mm, stress_ratio, mirror), applicability, bodies, estimate, routes, problems, notes, out_dir_rel, marks_rel, report_rel.
    """
    from topopt.cleanup import cleanup_list as lst, short_report
    d = _pick_dir(run_dir)
    data = lst(d, out_dir=_resolve(out_dir) if out_dir else None, options=options, overwrite=overwrite,
               estimate_only=estimate_only, yaml_path=_resolve_yaml(yaml_path) if yaml_path else None,
               root=_resolve(root) if root else None)
    if data.get("out_dir"):
        data["run_dir_rel"] = _rel(d)
        data["out_dir_rel"] = _rel(data["out_dir"])
        data["marks_rel"] = _rel(Path(data["out_dir"]) / "marks.json")
        data["report_rel"] = _rel(Path(data["out_dir"]) / "cleanup.json")
        if not detail:
            rels = {k: data[k] for k in ("run_dir_rel", "out_dir_rel", "marks_rel", "report_rel") if k in data}
            data = dict(short_report(data), **rels)
    elif not detail and data.get("estimate_only"):
        data["run_dir_rel"] = _rel(d)
    return data


# --- Team guide: role playbooks as a tool, MCP prompts and MCP resources (CONTRACTS.md section 25) ------------------
# One source: topopt_mcp/roles/<id>.md, read by topopt_mcp/team.py. This block only registers it.
from topopt_mcp import team  # noqa: E402


@mcp.tool(structured_output=False)
def team_guide(role: str | None = None) -> str:
    """Playbooks for the roles of a topopt team: lead-design, cad, fea, print and finish.

    A role is a job, not a separate agent. One agent can do all five in one conversation. A client that runs
    sub-agents can give each role to one. A playbook says what the role does, which tools it can use today,
    what does not exist yet, and what to report. The text is the same as the MCP prompt of that role and the
    resource topopt://roles/<id>.

    role: a role id (lead-design, cad, fea, print, finish). Also accepted: any case, the plugin agent name
    (cad-engineer) and the resource URI (topopt://roles/cad). Without a role, or with a name that matches
    none, the answer is the list of roles with one line each. That is a fact, not an error.
    """
    return team.guide(role)


def _add_role(role: team.Role) -> None:
    """Serve one playbook as an MCP prompt (name = role id) and as a resource (topopt://roles/<id>)."""
    def prompt() -> str:
        return role.text

    def resource() -> str:
        return role.text

    mcp.prompt(name=role.id, title=role.title, description=role.summary)(prompt)
    mcp.resource(role.uri, name=role.id, title=role.title, description=role.summary,
                 mime_type="text/markdown")(resource)


try:
    for _role in team.load_roles().values():
        _add_role(_role)
except team.RoleFilesError as _exc:  # a broken install: the other tools still work
    log(f"role playbooks are not available: {_exc}")


# --- Material reference: a read-only document as an MCP resource (CONTRACTS.md section 25.7) ------------------------
# One source: topopt_mcp/materials/reference.md, read by topopt_mcp/materials.py. No tool returns a number from it, and
# the engine never reads it (decision D7, revised 2026-10-24: no default allowable; the filament catalog gives E and density only). `topopt verify` names the URI.
from topopt_mcp import materials  # noqa: E402


def _add_material_reference() -> None:
    """Serve the material reference as the resource topopt://materials/reference (text/markdown)."""
    def resource() -> str:
        return materials.load_reference()

    mcp.resource(materials.URI, name=materials.NAME, title=materials.TITLE, description=materials.DESCRIPTION,
                 mime_type="text/markdown")(resource)


try:
    materials.load_reference()
    _add_material_reference()
except materials.MaterialFileError as _exc:  # a broken install: the other tools still work
    log(f"material reference is not available: {_exc}")


# --- Levers: the cards of the YAML keys and the engine tools as an MCP resource (CONTRACTS.md section 41) ----------------
# One source: the table topopt/levers_cards.py, shown by topopt/levers.py. The text is made at each read (no file). The CLI `topopt levers`
# prints the same text.
from topopt import levers  # noqa: E402


def _add_levers_reference() -> None:
    """Serve the lever cards as the resource topopt://levers and the template topopt://levers/{query} (text/markdown)."""
    def resource() -> str:
        return levers.reference()

    mcp.resource(levers.URI, name=levers.NAME, title=levers.TITLE, description=levers.DESCRIPTION,
                 mime_type="text/markdown")(resource)


    def cards_of(query: str) -> str:
        return levers.reference(query)

    mcp.resource(levers.TEMPLATE, name=levers.TEMPLATE_NAME, title=levers.TEMPLATE_TITLE, description=levers.TEMPLATE_DESCRIPTION,
                 mime_type="text/markdown")(cards_of)


_add_levers_reference()


# --- Features of an STL: holes, bosses and flat faces (CONTRACTS.md section 45) -------------------------------------------
# A read: the file is read in this process (numpy, scipy, trimesh; no child process, nothing is written, no work folder is made).
# The detection numbers (`topopt.features.OPTION_NAMES`) are the defaults of `topopt.features.find_features`. A tool argument for one of them
# is one more name in the signature of `find_features` and one more line in `options` below.


@mcp.tool()
def find_features(stl_path: str) -> dict:
    """List the holes, bosses and flat faces of an STL file, in mm. Read only: the file is read and nothing is written (CONTRACTS.md section 45).

    stl_path: an STL file (a relative path starts at the work folder). About 4 s for a million triangles. A wrong input gives {"ok": false, "error": ...}.
    A hole's `point`, `axis`, `radius_mm` and `length_mm` are the keys of a `bore` in `interfaces` (section 23).
    holes: each cylindrical hole has `id`, `point` (the centre of the first end face), the unit `axis`, `radius_mm`, `length_mm`, `end_point`, `depth` (through, blind, closed or unknown), `ends`, `arc_deg` (the angle the wall covers), `segments` and `chord_sag_mm` (the facets lie that far inside the radius), `fit` (distance of the wall vertices to the circle) and `coaxial_with` (a stepped hole is two cylinders on one axis). Every hole is listed with its radius and fit, whatever its size.
    bosses: convex cylinders (a shaft, a post), the same keys without `depth`.
    flat_faces: planar regions, the largest first: `normal` (outward, unit), `area_mm2`, `centroid`, `offset_mm`.
    mesh: faces, `watertight`, `bounds_mm`. A mesh that is not closed still gives what can be found (no repair).
    method: `numbers` (each detection number as used) and `counts`. facts: plain sentences (an open mesh, a stepped hole, a feature near a limit, regions left out).
    A hole whose wall joins a fillet, a tapered hole and a slot are not cylinders to this method.
    Your STL as a region item of build_regions: `{stl: {path: part.stl, plug_holes: [H1] or all, screw_holes: [H2] or all, ring_mm: 2}}` (a relative path starts at the work folder; a growth box is one more item). Only a closed solid mesh is read (no repair). plug_holes fills those holes (ids stable for one file). screw_holes adds a frozen ring round each: t = 0.5 x d, at least 2 voxels, or ring_mm (a number, or {H2: 2}). `{screw_ring: {path: part.stl, hole: H2}}` is that ring as a fixed or load region. A mouth chamfer and a drill point stay.
    """
    try:
        from topopt import features
        from topopt.config import UserError
    except ImportError as exc:
        return {"ok": False, "error": f"Cannot import topopt: {exc}"}
    options: dict = {}   # name -> value for a number of topopt.features.OPTION_NAMES; empty: the defaults of the module
    try:
        path = _resolve(stl_path)
        reply = features.find_features(path, **options)
    except UserError as exc:
        text = str(exc)
        if text.startswith("file not found:"):
            text += f" (work folder: {workspace()}). Give an absolute path." + _how_to_change()
        return {"ok": False, "error": text}
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {type(exc).__name__}: {exc}"}
    reply["mesh"]["source_rel"] = _rel(path)
    return reply


# --- Reply keys: the lines that explain the keys of the reply of seven tools, and the fine pass of run_topopt, as an MCP resource template (CONTRACTS.md section 10) ---
# One source: topopt_mcp/replies.py. The text of each of these tools names its address.
from topopt_mcp import replies  # noqa: E402


def _add_reply_guides() -> None:
    """Serve the reply key guides as the resource template topopt://replies/{tool} (text/markdown)."""
    def guide(tool: str) -> str:
        return replies.guide(tool)

    mcp.resource(replies.TEMPLATE, name=replies.NAME, title=replies.TITLE, description=replies.DESCRIPTION,
                 mime_type="text/markdown")(guide)


_add_reply_guides()



# --- The problem view: the problem as a 3D view in the chat (MCP Apps; topopt_mcp/problem_view.py) ---------------------------------------------
# One tool and one ui:// resource. A client that shows MCP Apps reads the resource that the tool names in `_meta.ui.resourceUri`, shows it,
# and sends the result to it. The meshes go in the `_meta` of the result (for the page only), the facts in `structuredContent`.
from topopt_mcp import problem_view  # noqa: E402


def _view_error(text: str) -> CallToolResult:
    reply = {"ok": False, "error": text}
    _add_work_folder_note(reply, "show_problem")
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(reply))], structured_content=reply)


@mcp.tool(meta=problem_view.TOOL_META)
def show_problem(yaml_path: str) -> CallToolResult:
    """Show a problem YAML to the user as a 3D view in the chat: the design space, the fixed, keep-out, solid and load regions, an arrow for each load case, and the time and memory estimate. No optimizer run: the same checks as check_problem, in the same time.

    yaml_path: the problem YAML (relative paths start at the work folder).
    A client that shows MCP Apps (for example Claude Desktop) shows the card. The user can turn and zoom it. The view's button "Start the run" works in one of three ways. The host lets the view call tools: the click starts the run at once (run_topopt), the view watches it (run_status), and the host may tell you about the start and the end. While that run works, a second run_topopt is refused; run_status shows the run. When the run folder (runs/live) holds a run, the first click shows what the start replaces; a second click starts. The host takes only a message from the view: the button sends a chat message from the user that asks you to start the run. Neither: the view shows a line for the user to type. Another client shows only the text: the region counts and the estimate.
    Use it when the YAML is ready, so that the user sees the setup before the run. The full facts and the warnings are in check_problem.
    On a wrong input: {"ok": false, "error": ...}.
    """
    yaml = _resolve_yaml(yaml_path)
    if not yaml.is_file():
        return _view_error(f"YAML file not found: {yaml} (work folder: {workspace()})")
    try:
        from topopt.cli import check_data
        from topopt.config import UserError, load_config
    except ImportError as exc:
        return _view_error(f"Cannot import topopt: {exc}")
    try:
        cfg = load_config(yaml)
        check = check_data(yaml)
        meshes = problem_view.regions(cfg)
        cases = problem_view.load_cases(cfg, meshes)
    except UserError as exc:
        return _view_error(str(exc))
    except Exception as exc:
        return _view_error(f"internal error: {type(exc).__name__}: {exc}")
    est = problem_view.estimate(check)
    name = check.get("name") or yaml.stem
    facts = {"ok": True, "name": name, "yaml_rel": _rel(yaml), "start_run_dir_rel": _rel(default_out()), "estimate": est,
             "regions": [{k: m[k] for k in ("kind", "name", "file", "triangles")} for m in meshes],
             "load_cases": [c["name"] for c in cases]}
    _add_work_folder_note(facts, "show_problem")
    text = problem_view.summary(name, meshes, cases, est, facts["yaml_rel"])
    if NOTE_KEY in facts:
        text += "\n" + str(facts[NOTE_KEY])
    geometry = {"format": problem_view.FORMAT, "regions": meshes, "load_cases": cases}
    return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=facts,
                          meta={problem_view.GEOMETRY_KEY: geometry})


def _add_problem_view() -> None:
    """Serve the page of the card as the resource ui://topopt/problem-view.html (text/html;profile=mcp-app)."""
    def view() -> str:
        return problem_view.page()

    mcp.resource(problem_view.URI, name=problem_view.NAME, title=problem_view.TITLE, description=problem_view.DESCRIPTION,
                 mime_type=problem_view.MIME_TYPE, meta=problem_view.RESOURCE_META)(view)


_add_problem_view()


def start_line(folder: workfolder.WorkFolder) -> str:
    """The line at server start (stderr; Claude Desktop keeps it in mcp-server-*.log): the work folder, where it came from, the notes."""
    parts = [f"python: {sys.executable}", f"work folder: {folder.path} (from {folder.source})"]
    return "; ".join(parts + [f"fix needed: {t}" for t in folder.fix_needed] + [f"note: {t}" for t in folder.notes])


def main() -> None:
    # job.json `host.started_by` (topopt.hostinfo.STARTED_BY_ENV): every job that this server runs says `mcp`, the run and the ratchet in a child
    # process (the child inherits the variable) and `ratchet_approve` in this process.
    os.environ.setdefault("TOPOPT_STARTED_BY", "mcp")
    folder = workfolder.resolve()
    log(start_line(folder))
    _register_server(folder)
    handoff = read_handoff()   # the launcher answered the client itself until the setup was ready (#370)
    if handoff is None:
        mcp.run("stdio")
    else:
        serve_handoff(mcp, handoff)


if __name__ == "__main__":
    main()
