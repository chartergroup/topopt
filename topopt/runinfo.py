"""Small run facts that need no other topopt module (CONTRACTS.md section 24.1).

This module uses the standard library only. Any module can import it:
the optimizer, the post step, the CLI, the MCP server and the tools.

- Peak memory: `peak_rss_bytes()` and `children_peak_rss_bytes()`. Both return bytes on
  macOS and on Linux. `ru_maxrss` is in bytes on macOS and in KiB on Linux. On Linux
  `peak_rss_bytes()` reads `VmHWM` of `/proc/self/status`, which a child does not inherit.
- `history.jsonl`: `read_history(path)` is the shared reader. It skips a partial last line
  and gives that as a fact. A line that does not parse and is not the last line is an error.
- `append_log_line(run_dir, line)`: add one whole line to `run.log`.
- `write_atomic(path, data, mode=None)`: write `<name>.tmp` in the same folder, then `os.replace`. With `mode`, the file has that mode from the start.
- `sha256_file(path)`: the SHA-256 of a file, read in blocks. The one file-hash helper for run inputs.
- `region_files(regions, base_dir, root=None, yaml_path=None)`: every region file of a problem, keyed
  `<role>:<relative path>`. With `root` (the work folder), a file outside it is keyed `<role>:<outside>/<12 hex>/<file name>`.
  With `yaml_path`, a file is keyed by the path that the YAML names (a link keeps its own name).
- `run_yaml_path(run, run_dir=None)`: the YAML of a run, from the `yaml_path` of `run.json`. The one rule for every reader of
  that key: a relative path starts at the run folder (not at the folder of the process). `resolved_run(run, run_dir)` is a
  copy of the `run.json` dict that holds this path as text, so a function that takes the dict only needs no folder.
  `missing_yaml_text(tried, run, option, given=False)` is the error text when that YAML is not there.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

from . import workfolder


# --- peak memory ---------------------------------------------------------------------------

def ru_maxrss_to_bytes(value, platform: str | None = None) -> int:
    """Convert a `ru_maxrss` value to bytes. macOS counts bytes. Other systems count KiB.

    `platform` is a `sys.platform` text. The default is the platform of this process.
    A test gives a fake platform to check both branches.
    """
    plat = sys.platform if platform is None else platform
    return int(value) if plat == "darwin" else int(value) * 1024


_PROC_STATUS = "/proc/self/status"


def _vmhwm_bytes(path=None) -> int | None:
    """`VmHWM` (the peak resident memory of this process, in KiB) of `/proc/self/status`, in bytes, or None."""
    try:
        for line in Path(_PROC_STATUS if path is None else path).read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def peak_rss_info() -> tuple[int, str]:
    """(peak resident memory of this process in bytes, where the number comes from).

    The source is `"VmHWM"` where `/proc/self/status` has it (Linux), else `"ru_maxrss"` (macOS).
    `ru_maxrss` of a process that a parent started with `subprocess` (vfork or posix_spawn, then exec) starts at the
    peak of the parent on Linux. `VmHWM` belongs to the memory of this process only, so it has no such floor.
    """
    hwm = _vmhwm_bytes()
    if hwm is not None:
        return hwm, "VmHWM"
    import resource
    return ru_maxrss_to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "ru_maxrss"


def peak_rss_bytes() -> int:
    """Peak resident memory of this process, in bytes. See `peak_rss_info` for the source of the number."""
    return peak_rss_info()[0]


def children_peak_rss_bytes() -> int:
    """Peak resident memory of the largest child process that this process has waited for, in bytes.

    It is 0 when no child has ended. It is the largest single child, not the sum of the children.
    On Linux a child that `subprocess` starts (vfork or posix_spawn, then exec) starts with the peak of its parent;
    a child of `os.fork` plus exec starts with the current resident memory of the parent. So the value can be at
    least the peak that this process had when its largest child started, and a small child of a big process looks
    big. Measured: the `gcc` and `ld` children of `pypardiso` gave 1.70 GB in a hook run whose own peak was 1.71 GB.
    Use it for a child that is much bigger than its parent.
    """
    import resource
    return ru_maxrss_to_bytes(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


# --- history.jsonl --------------------------------------------------------------------------

class HistoryError(ValueError):
    """A line of `history.jsonl` does not parse and it is not the last line."""


class HistoryRead(NamedTuple):
    rows: list[dict]     # the rows that parse, in file order
    facts: list[str]     # what the reader skipped, one short sentence each (empty if nothing)


def _row_problem(line: str):
    """Return (row, None) if the line is a JSON object, else (None, the reason)."""
    try:
        row = json.loads(line)
    except ValueError as exc:
        return None, f"does not parse ({exc})"
    if not isinstance(row, dict):
        return None, "is not a JSON object"
    return row, None


def read_history(path, strict: bool = True) -> HistoryRead:
    """Read `history.jsonl`. Return `HistoryRead(rows, facts)`.

    The writer adds each record as one whole line (`RunWriter.add_history`). A reader can still see
    a partial last line: a read in the middle of a write, or a process that was killed.
    - The last line does not parse: it is skipped and one fact says so.
    - Another line does not parse: `strict=True` (default) raises `HistoryError` with the line number.
      With `strict=False` the reader skips the line and gives one fact with the line numbers.
    A blank line is ignored. A missing file raises `FileNotFoundError`.
    """
    raw = Path(path).read_bytes().decode("utf-8", errors="replace")
    numbered = [(n, line) for n, line in enumerate(raw.splitlines(), 1) if line.strip()]
    rows: list[dict] = []
    facts: list[str] = []
    skipped: list[int] = []
    last_n = numbered[-1][0] if numbered else 0
    for n, line in numbered:
        row, problem = _row_problem(line)
        if row is not None:
            rows.append(row)
        elif n == last_n:
            facts.append(f"history.jsonl: skipped a partial last line (line {n}, {len(line.encode())} bytes)")
        elif strict:
            raise HistoryError(f"history.jsonl line {n} {problem}, and it is not the last line")
        else:
            skipped.append(n)
    if skipped:
        shown = ", ".join(str(n) for n in skipped[:5]) + (", ..." if len(skipped) > 5 else "")
        facts.insert(0, f"history.jsonl: skipped {len(skipped)} line(s) that do not parse (line {shown})")
    return HistoryRead(rows, facts)


# --- run.log --------------------------------------------------------------------------------

def append_log_line(run_dir, line: str) -> None:
    """Add one whole line to `<run_dir>/run.log`: one write, then a flush."""
    with open(Path(run_dir) / "run.log", "ab") as f:
        f.write((line.rstrip("\n") + "\n").encode())
        f.flush()


# --- atomic file write ----------------------------------------------------------------------

def write_atomic(path, data: bytes, *, mode: int | None = None) -> None:
    """Write `data` to `<path>.tmp` in the same folder, then move it to `path` with `os.replace`.

    A reader sees the old file or the new file, never a part file. The same rule as
    `RunWriter` uses for `status.json` and the snapshots.
    `mode` (for example `0o600`): the file has exactly this mode from the moment it exists. The temporary file is made with it,
    so no reader sees a wider mode, and the umask cannot narrow it. Without `mode` the file gets the default mode (0o666 before the umask).
    A `<name>.tmp` that is there already (a stale file, or a link) is removed first, and the new one is made with `O_EXCL` (and `O_NOFOLLOW` where
    the system has it), so the data never goes to the target of a link.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)       # a stale file, or a link (the link goes, not its target): the data never goes through a link
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(tmp, flags, 0o666 if mode is None else mode), "wb") as f:
        if mode is not None:
            os.fchmod(f.fileno(), mode)    # the umask cannot narrow the mode of the file
        f.write(data)
    os.replace(tmp, path)


# --- file hash -----------------------------------------------------------------------------

def sha256_handle(f, block: int = 1 << 20) -> str:
    """The SHA-256 of the rest of an open binary file, as a hex string, read in blocks of `block` bytes.

    A caller that has read a file through a handle hashes the same bytes through the same handle: a writer that replaces the file by
    name (`os.replace`) does not change an open handle.
    """
    h = hashlib.sha256()
    while chunk := f.read(block):
        h.update(chunk)
    return h.hexdigest()


def sha256_file(path, block: int = 1 << 20) -> str:
    """The SHA-256 of the file at `path` as a hex string. The file is read in blocks of `block` bytes.

    Use this helper for every input hash (checkpoint, case manifest, run record), so that the same file gives the same
    value in each record.
    """
    with open(path, "rb") as f:
        return sha256_handle(f, block)


OUTSIDE = "<outside>"   # the part of a key for a region file outside the root folder (no folder name of that place is written)


def region_names(yaml_file) -> dict | None:
    """The file names under `regions` as the YAML text gives them: `{role: [name, ...]}`. None if the text has none.

    It reads the text only (the problem does not need to load). It needs PyYAML; the import is inside the function, so
    this module stays standard-library only at import time.
    """
    import yaml
    try:
        data = yaml.safe_load(Path(yaml_file).read_bytes())
    except (OSError, yaml.YAMLError):
        return None
    regions = data.get("regions") if isinstance(data, dict) else None
    if not isinstance(regions, dict):
        return None
    out: dict = {}
    for role, value in regions.items():
        items = list(value.values()) if isinstance(value, dict) else value if isinstance(value, list) else [value]
        names = [x for x in items if isinstance(x, str) and x.strip()]
        if names:
            out[str(role)] = names
    return out


def location_of(path) -> Path:
    """Where a path is: the folders are resolved, the last part is not (a link keeps its own name)."""
    p = os.path.abspath(path)
    return Path(os.path.realpath(os.path.dirname(p))) / os.path.basename(p)


def outside_key(path) -> str:
    """`<outside>/<12 hex>`: the start of the SHA-256 of the resolved path text. It holds no folder name.

    This is a digest of text, not of a file; `sha256_file` stays the one file-hash helper.
    """
    return f"{OUTSIDE}/{hashlib.sha256(os.path.realpath(path).encode()).hexdigest()[:12]}"


def key_root(yaml_path) -> Path:
    """The folder that decides when a region file is outside: the work folder (the one rule of `topopt.workfolder`, the same
    folder as the MCP server takes) if the YAML lies in it, else the folder of the YAML."""
    here = Path(os.path.realpath(Path(yaml_path).parent))
    try:
        work = Path(os.path.realpath(workfolder.resolve(check=False).path))
    except OSError:                       # no work folder: the current folder is gone and nothing sets another
        return here
    try:
        here.relative_to(work)
    except ValueError:
        return here
    return work


def region_files(regions, base_dir, root=None, yaml_path=None) -> dict:
    """Every region file of a problem as `{"<role>:<path>": Path}`.

    `role` is the field name of `regions` (`design`, `fixed`, `keep_out`, `load`, and any field that a later version
    adds, for example `solid`). `path` is relative to `base_dir` (the YAML folder), with `/` as separator, so two files
    with the same name in different folders get different keys. A field can hold a path, a list of paths, or a list of
    objects with a `.path` attribute (load regions). A field that is None is left out. The checkpoint and the case
    manifest use this function, so the same file has the same key in each record.

    `root` (default None: no rule). A file outside `root` has the key `<role>:<outside>/<12 hex>/<file name>`: the 12 hex
    characters come from the resolved path (`outside_key`), so the key holds no folder name of the outside place (a user
    name, for example). The case uses the work folder as `root`; the checkpoint uses `key_root`.

    `yaml_path` (default None): the file names are taken from the text of the YAML, and the key holds the path as the YAML
    names it. A symbolic link keeps its own name; the hash of the file (`sha256_file`) is of the content. Without it the
    paths of `regions` are used (the loader resolves a link). A role whose number of names differs from the number of
    paths keeps the paths of `regions`. `regions` can be None when `yaml_path` is given (the problem does not load): the
    files are then those that the text names. The path in the value is the path to read.
    """
    base = os.path.abspath(base_dir)
    names = region_names(yaml_path) if yaml_path is not None else None
    root_real = Path(os.path.realpath(root)) if root is not None else None

    def named(name) -> Path:
        return Path(os.path.normpath(os.path.join(base, name)))

    entries: list = []
    if regions is None:
        for role, role_names in (names or {}).items():
            entries += [(role, named(n)) for n in role_names]
    else:
        if dataclasses.is_dataclass(regions):
            roles = [f.name for f in dataclasses.fields(regions)]
        else:
            roles = list(vars(regions))
        for role in roles:
            value = getattr(regions, role)
            items = value if isinstance(value, (list, tuple)) else [value]
            paths = []
            for item in items:
                path = item if isinstance(item, (str, os.PathLike)) else getattr(item, "path", None)
                if path is not None:
                    paths.append(Path(path))
            role_names = (names or {}).get(role)
            if role_names is not None and len(role_names) == len(paths):
                paths = [named(n) for n in role_names]
            entries += [(role, p) for p in paths]
    out: dict = {}
    for role, path in entries:
        if root_real is not None:
            try:
                location_of(path).relative_to(root_real)
            except ValueError:
                out[f"{role}:{outside_key(path)}/{path.name}"] = path
                continue
        try:
            rel = Path(os.path.relpath(path, base)).as_posix()
        except ValueError:  # another drive (Windows): no relative path exists
            rel = Path(path).as_posix()
        out[f"{role}:{rel}"] = path
    return out


def run_yaml_path(run, run_dir=None) -> Path | None:
    """The YAML file of a run: the `yaml_path` of the `run.json` dict, or None if it has none.

    An absolute path (or one with `~`) keeps its meaning. A relative path starts at the run folder `run_dir`, not at the
    folder of the process, so a recording that holds `../../examples/hook/hook.yaml` finds its YAML from any current
    folder. With no `run_dir` a relative path is returned as it is. The path is not normalized and its file is not
    tested: the caller tests `is_file()`. Region files are not here: they start at the folder of the YAML.
    """
    text = (run or {}).get("yaml_path")
    if not text:
        return None
    p = Path(str(text)).expanduser()
    if not p.is_absolute() and run_dir is not None:
        p = Path(run_dir) / p
    return p


def resolved_run(run: dict, run_dir) -> dict:
    """A copy of the `run.json` dict whose `yaml_path` is the path of `run_yaml_path` as text.

    For a function that takes the dict only and reads the YAML folder from it (`post.region_paths`,
    `interfaces.specs_from_run`): it gets the folder that the rule gives. A dict with no `yaml_path` is returned as it is.
    """
    p = run_yaml_path(run, run_dir)
    return run if p is None else {**run, "yaml_path": str(p)}


def missing_yaml_text(tried, run, option: str, given: bool = False) -> str:
    """The error text for a YAML file of a run that is not there: the facts, then the option.

    `tried` is the path that was tested (None: `run.json` has no `yaml_path`); the text shows it as an absolute path, with the
    `..` of a relative `yaml_path` resolved. `option` is the name of the option that gives another file (`--yaml`, or `yaml_path`
    for a tool). `given` is True when the user gave that file with the option. `refine`, `buckling` and `verify` use this one text.
    """
    if tried is not None:
        tried = os.path.realpath(tried)
    if given:
        return f"the YAML file given with {option} is missing: {tried}"
    if tried is None:
        return f"run.json has no yaml_path, so the YAML file of the run is not known. Give it with {option}"
    text = f"the YAML file of the run is missing: {tried}"
    stored = str(run.get("yaml_path", ""))
    if not Path(stored).expanduser().is_absolute():
        text += f" (yaml_path in run.json is {stored!r}: a relative path starts at the run folder)"
    return text + f". Give its new place with {option}"
