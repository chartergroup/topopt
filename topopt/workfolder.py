"""The work folder: one rule for the MCP server and for the checkpoint (CONTRACTS.md section 10.1).

This module does not import `mcp`, so the CLI and the checkpoint can use it. It imports `topopt.runjob` (the `is_run_folder` rule of
`list_runs`) only when it looks for runs in an old place. `topopt_mcp/server.py` (`workspace()`) and `topopt/runinfo.py`
(`key_root()`) both call `resolve()`, so they give the same folder.

The text of the env var `TOPOPT_WORKSPACE` is read like this:

- Empty, unset, or a whole `${user_config.<key>}` (a host gives this text for an option that the user left empty): not set.
  This is the normal case, so it has no note. Blanks are cut for this test only: a path keeps its blanks.
- A leading `~` and the MCPB variables `${HOME}`, `${DOCUMENTS}`, `${DESKTOP}`, `${DOWNLOADS}`, `${/}` and `${pathSeparator}` (exact
  case) are replaced. `${DOCUMENTS}`, `${DESKTOP}` and `${DOWNLOADS}` are folders of the home folder, on every platform.
  `expanded` lists what was replaced.
- A `${` that is left, or a `~name` that no home folder replaces: the value cannot be used. The next value is used, and the facts have
  a `fix_needed` text with the value as it was given. If both values cannot be used, one text names both and gives the fix once.
- Not set, or not usable: the env var `TOPOPT_DEFAULT_WORKSPACE` (the same rules; the Desktop extension sets it), else the
  current folder (the Claude Code plugin, the CLI and Cursor start the server in the folder to work in).
- A relative path (after the replacement) starts at the current folder of the process, as before. `notes` says so. The current folder
  is read only when a rule needs it (no usable value, a relative path, the old-runs check). If it is gone, `fix_needed` says so.
- A folder inside `.../Claude/scratch-workspaces/...` is the scratch workspace that Claude Desktop makes for a Code tab session that has
  no project folder (observed 2026-10-23, Desktop 3.6556.2). The app removes it when the session ends. `notes` says so and names the options.

The words that tell a person how to change the work folder are in this module only (`FIX_DESKTOP`, `FIX_OTHER`). `fix_text()` gives the
text of the surface that started the server, and `change_text()` the same with a lead-in for a reply that offers it as a way out. A tool
reply, a tool description and a case option do not copy these words.

`resolve()` creates no folder and changes nothing. A tool that writes creates the folder where it writes. The facts say when
the folder cannot be created or written (`writable` false) and, when `runs/` cannot be written either, a `fix_needed` text. A
work folder that the process cannot write, with a `runs/` that it can write, works: `notes` says that the lock files of verify and
ratchet cannot be written. `run_topopt` and `build_regions` refuse only when the folder that they write (the run folder, or `runs/`)
cannot be created or written; they use `where()` for it.

`fix_needed` holds texts about a value that the server could not use or a folder that it cannot write. `notes` holds texts
about what the server did (replaced a variable, resolved a relative path, found runs in an old place). The server gives
`fix_needed` in every tool reply, and `notes` in the replies of `environment`, `list_runs` and `run_topopt`.
"""
from __future__ import annotations

import errno
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ENV = "TOPOPT_WORKSPACE"
ENV_DEFAULT = "TOPOPT_DEFAULT_WORKSPACE"
CURRENT = "current folder"

# What a person does to change the work folder. There are two texts, because the server can tell two cases apart:
# - FIX_DESKTOP: the Desktop extension started the server. Its manifest (packaging/mcpb/manifest.json, mcp_config.env) is the only
#   file that sets TOPOPT_DEFAULT_WORKSPACE, so that variable is the sign (`is_desktop_extension`).
# - FIX_OTHER: every other start (the Claude Code plugin, a checkout, Cursor, VS Code, Codex CLI, Gemini CLI, a hand-made config). No
#   file of those sets a variable that names the client, so the text is true for all of them: the plugin has no env to edit (its
#   `.claude-plugin/plugin.json` gives the server no env), and it starts in the folder where Claude Code runs.
# Every reply that tells a person how to change the work folder takes its words from `fix_text` or `change_text`: no second copy.
FIX_DESKTOP = ("Open Settings > Extensions > topopt, choose the Work folder with the folder button, then quit Claude completely "
               "(Cmd+Q on a Mac) and open it again. A new chat or a new Code tab session is not enough: it keeps the old server process.")
FIX_OTHER = ("Start the client in the folder that you want to use as the work folder. Or set TOPOPT_WORKSPACE to a full path in the env "
             "of the topopt server in the settings of your client. Then restart the server.")

# macOS asks the user before an app reads a folder such as ~/Documents. A "no" shows as EPERM. The setting is a fact of macOS 15.
MAC_PRIVACY = ("macOS may block this folder: System Settings > Privacy & Security > Files and Folders shows if Claude may use it")

# The scratch workspace of a Code tab session without a project folder: <app data>/Claude/scratch-workspaces/<id>/<id>/scratch-<date>-<hex>.
SCRATCH_PARTS = ("Claude", "scratch-workspaces")

# The default of the option in version 0.1.0 of the extension. Claude Desktop did not replace the variable, so the server before this
# one made the runs in `<extension folder>/${DOCUMENTS}/topopt`, also for a person who typed another folder later.
OLD_DEFAULT = "${DOCUMENTS}/topopt"

_VARIABLE = re.compile(r"\$\{([^}]*)\}")
_EMPTY_OPTION = re.compile(r"\$\{user_config\.[^}]*\}")   # the whole text, matched with fullmatch


@dataclass
class _Value:
    """One env var after the replacement: `kind` is `unset`, `ok` or `broken`. A broken value has `problem`: `variable` or `tilde`."""
    kind: str
    text: str = ""
    expanded: list = field(default_factory=list)
    problem: str = ""


@dataclass
class WorkFolder:
    path: Path
    source: str                      # ENV, ENV_DEFAULT or CURRENT
    raw: str | None                  # the text of TOPOPT_WORKSPACE as the process got it (None: not in the environment)
    expanded: list = field(default_factory=list)
    exists: bool | None = None       # None: not checked (resolve(check=False))
    writable: bool | None = None
    fix_needed: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    old_runs: dict | None = None
    blocked: str | None = None       # the text in fix_needed about a folder that cannot be created or written
    fix: str = ""                    # what a person does about a work folder that cannot be used (the Desktop text, or the other text)

    def facts(self) -> dict:
        return {"path": str(self.path), "source": self.source, "raw": self.raw, "expanded": list(self.expanded),
                "exists": self.exists, "writable": self.writable, "fix_needed": list(self.fix_needed),
                "notes": list(self.notes), "old_runs": self.old_runs}

    def note(self, info: bool = False, skip: str = "") -> str | None:
        """The text for a tool reply: `fix_needed`, and with `info` also `notes`. A text that `skip` holds is left out.
        None when there is no text."""
        texts = [t for t in list(self.fix_needed) + (list(self.notes) if info else []) if t not in skip]
        return " ".join(texts) or None


def is_desktop_extension(env=None) -> bool:
    """True when the Desktop extension started the server: its manifest sets `TOPOPT_DEFAULT_WORKSPACE`, and no other launcher does."""
    env = os.environ if env is None else env
    return bool((env.get(ENV_DEFAULT) or "").strip())


def fix_text(env=None) -> str:
    """What a person does to change the work folder, in the words of the surface that started this server."""
    return FIX_DESKTOP if is_desktop_extension(env) else FIX_OTHER


def change_text(env=None) -> str:
    """The sentence for a reply that offers the change of the work folder as a way out (a file not found, a folder that cannot be written)."""
    return f"To change the work folder: {fix_text(env)}"


def _home() -> Path | None:
    try:
        return Path.home()
    except (RuntimeError, KeyError, OSError):   # no home folder can be found
        return None


def _expand(text: str, home: Path | None) -> tuple[str, list]:
    """Replace a leading `~` and the known `${...}` variables. Returns the new text and what was replaced."""
    expanded: list = []
    if text == "~" or text.startswith("~/"):
        if home is not None:
            text = str(home) + text[1:]
            expanded.append("~")
    elif text.startswith("~"):                    # ~name: the home folder of another user
        other = os.path.expanduser(text)
        if other != text:
            text = other
            expanded.append("~")
    known = {"/": os.sep, "pathSeparator": os.sep}
    if home is not None:
        known.update({"HOME": str(home), "DOCUMENTS": str(home / "Documents"), "DESKTOP": str(home / "Desktop"),
                      "DOWNLOADS": str(home / "Downloads")})

    def replace(m: re.Match) -> str:
        if m.group(1) not in known:
            return m.group(0)
        if m.group(0) not in expanded:
            expanded.append(m.group(0))
        return known[m.group(1)]

    return _VARIABLE.sub(replace, text), expanded


def _value(raw: str | None, home: Path | None) -> _Value:
    """Blanks are cut for the test "is it empty" and for the test "is it a whole `${user_config.<key>}`". A path keeps its blanks."""
    probe = (raw or "").strip()
    if not probe or _EMPTY_OPTION.fullmatch(probe):
        return _Value("unset")
    text, expanded = _expand(raw, home)
    if "${" in text:
        return _Value("broken", text, expanded, "variable")
    if text.startswith("~"):       # a ~name that expanduser cannot replace (no such user), or no home folder to replace ~ with
        return _Value("broken", text, expanded, "tilde")
    return _Value("ok", text, expanded)


def _is_under(path: Path, folder: Path) -> bool:
    for base in (folder, folder.resolve()):
        try:
            path.relative_to(base)
            return True
        except ValueError:
            pass
    return False


def where(path: Path, home: Path | None = None) -> tuple[bool, bool, str | None]:
    """`(exists, writable, cause)` of a folder, with no write. A folder that is not there is writable when it can be created.

    `cause` says why it is not writable (None when it is). A tool uses this for the folder that it writes."""
    try:
        if path.is_dir():
            if os.access(path, os.W_OK | os.X_OK):
                return True, True, None
            return True, False, "the server cannot write in it"
        if path.exists():
            return True, False, "it is a file, not a folder"
        up = path.parent
        while not up.is_dir():
            if up.exists():
                return False, False, f"{up} is a file, not a folder"
            if up.parent == up:
                break
            up = up.parent
        if os.access(up, os.W_OK | os.X_OK):
            return False, True, None
        return False, False, f"the server cannot create it, because it cannot write in {up}"
    except OSError as exc:
        cause = f"the server cannot read it ({exc.strerror or exc})"
        home = _home() if home is None else home
        if exc.errno == errno.EPERM and sys.platform == "darwin" and home is not None and _is_under(path, home):
            cause += f". {MAC_PRIVACY}"
        return False, False, cause


def _current_folder(cwd) -> Path | None:
    """The current folder of the process, or None when it is gone (the folder was removed: the system gives an `OSError`)."""
    if cwd is not None:
        return Path(cwd)
    try:
        return Path.cwd()
    except OSError:
        return None


def _in_scratch_workspace(path: Path) -> bool:
    """True when the folder lies inside `.../Claude/scratch-workspaces/...` (a fact from the path)."""
    parts = path.parts
    return any(parts[i:i + 2] == SCRATCH_PARTS for i in range(len(parts) - 2))


def _count_old_runs(old: Path) -> dict | None:
    from .runjob import is_run_folder   # the rule of list_runs
    count, has_runs_folder = 0, False
    for parent in (old, old / "runs"):
        try:
            with os.scandir(parent) as it:
                for entry in it:
                    if parent != old:
                        has_runs_folder = True
                    if entry.is_dir(follow_symlinks=False) and is_run_folder(entry.path):
                        count += 1
        except OSError:
            continue
    if not count and not has_runs_folder:
        return None
    return {"path": str(old), "run_folders": count}


def _old_runs(texts: list, here, path: Path) -> list:
    """Runs in the places where the server before this one put them: `[{path, run_folders}, ...]`, one item for each place that holds runs.

    That server resolved a text with `${` (for example `${DOCUMENTS}/topopt`) against its current folder, which is the extension
    folder of the host. `texts` are the texts that it may have had. The folder in use is not an old place.
    """
    found, seen = [], set()
    for text in texts:
        old = Path(os.path.expanduser(text))
        if not old.is_absolute():
            cwd = here()
            if cwd is None:                  # no current folder: there is no old place to look at
                continue
            old = cwd / old
        old = old.resolve()
        if old == path or old in seen or not old.is_dir():
            continue
        seen.add(old)
        counted = _count_old_runs(old)
        if counted:
            found.append(counted)
    return found


def resolve(env=None, cwd=None, home=None, check: bool = True) -> WorkFolder:
    """The work folder and the facts about it.

    env: a mapping with the env vars (default `os.environ`). cwd: the current folder (default `Path.cwd()`).
    home: the home folder (default `Path.home()`). A test gives fake ones.
    check: True looks at the disk (`exists`, `writable`, the old place of runs); False reads the text only. It never writes.
    """
    env = os.environ if env is None else env
    home = _home() if home is None else Path(home)
    read = []

    def here() -> Path | None:           # the current folder is read when a rule needs it, and at most once
        if not read:
            read.append(_current_folder(cwd))
        return read[0]

    raw = env.get(ENV)
    desktop = is_desktop_extension(env)
    fix = fix_text(env)
    notes: list = []

    texts = {ENV: raw, ENV_DEFAULT: env.get(ENV_DEFAULT)}
    label = {ENV: "The work folder setting", ENV_DEFAULT: "The default work folder (TOPOPT_DEFAULT_WORKSPACE)"}
    chosen, source, broken = None, CURRENT, {}
    for name in (ENV, ENV_DEFAULT):
        value = _value(texts[name], home)
        if value.kind == "ok":
            chosen, source = value, name
            break
        if value.kind == "broken":
            broken[name] = value.problem

    relative = chosen is not None and not os.path.isabs(chosen.text)
    base = here() if chosen is None or relative else None
    no_current = (chosen is None or relative) and base is None       # the rule needs the current folder, and it is gone
    if no_current:
        path = Path(os.curdir) if chosen is None else Path(chosen.text)
    elif chosen is None:
        path = base.resolve()
    else:
        path = (base / chosen.text if relative else Path(chosen.text)).resolve()
    expanded = [] if chosen is None else chosen.expanded

    fix_needed: list = []
    blocked = None
    if no_current:
        what = "No work folder is set" if chosen is None else f'The work folder setting is "{texts[source]}". It is not a full path'
        blocked = f"{what}, and the current folder of the server is gone, so the server has no work folder. {fix}"
        fix_needed.append(blocked)
    if broken:       # one text for all the values that cannot be used: each value by name, then the used folder, then the fix once
        problem = {
            "variable": "It has a variable that the server does not know" + (", and Claude Desktop did not replace it" if desktop else ""),
            "tilde": "It starts with ~, and the server cannot replace that with a home folder",
        }
        said = " ".join(f'{label[name]} is "{texts[name]}". {problem[kind]}.' for name, kind in broken.items())
        fix_needed.append(said + ("" if no_current else f" The server used {path}.") + f" {fix}")
    if chosen is not None:
        shown = f'{label[source]} is "{texts[source]}".'
        if any(x.startswith("${") for x in expanded):
            notes.append(
                f"{shown} " + ("Claude Desktop did not replace the variable, so the server did."
                               if desktop else "The variable was not replaced before the server started, so the server did.")
                + f" The work folder is {path}."
                + (" To show a plain path in the setting, choose the Work folder with the folder button." if desktop and source == ENV else ""))
        if relative and not no_current:
            text = f"{shown} It is not a full path, so the server started from its current folder. "
            if desktop:
                text += "The current folder of the server is the folder of the extension, and an update of the extension can remove it. "
            text += f"The work folder is {path}. "
            if desktop:   # the option: the full form of the text, from the home folder
                text += "To keep the runs, type a full path in the Work folder setting" \
                        + (f", for example {os.path.normpath(home / chosen.text)}." if home is not None else ".")
            else:
                text += "A full path does not depend on the current folder."
            notes.append(text)

    if _in_scratch_workspace(path):
        notes.append(f"The work folder {path} is in a scratch workspace that Claude Desktop made for this session. "
                     "The app removes this folder when the session ends, and the runs in it are lost. "
                     + ("Choose a folder that you keep. " if desktop else "Open the session in a project folder that you keep. ")
                     + change_text(env))
    exists = writable = None
    old = None
    if check:
        if no_current:
            exists = writable = False
        else:
            exists, writable, cause = where(path, home)
            if not writable and exists and where(path / "runs")[1]:
                # The runs go in runs/, and the process can write there: the runs work. Only the lock files of verify and ratchet do not.
                notes.append(f"The server cannot write in the work folder {path}, but it can write in {path / 'runs'}, so the runs go on. "
                             "The lock files of verify_run and ratchet_run are in the work folder, so they cannot be written there.")
            elif not writable:
                blocked = f"The server cannot use the work folder {path}: {cause}. {fix}"
                fix_needed.append(blocked)
        texts_of_old = [raw] if raw and "${" in raw else []
        if desktop and OLD_DEFAULT not in texts_of_old:
            texts_of_old.append(OLD_DEFAULT)
        found = _old_runs(texts_of_old, here, path)
        for item in found:
            notes.append(f"Runs from before are in {item['path']}. "
                         + ("An update of the extension can remove this folder. " if desktop else "")
                         + "Copy them to the work folder if you want to keep them.")
        if found:
            old = dict(found[0], **({"also": found[1:]} if len(found) > 1 else {}))
    return WorkFolder(path=path, source=source, raw=raw, expanded=list(expanded), exists=exists, writable=writable,
                      fix_needed=fix_needed, notes=notes, old_runs=old, blocked=blocked, fix=fix)
