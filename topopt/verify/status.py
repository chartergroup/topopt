"""The status file of a verify run and the clean-up of its folder. CONTRACTS.md section 16, "Files".

A real run writes `<run_dir>/verify/status.json` while it works, so that another process (the MCP server)
can watch it. The file is written atomically (a temp file, then a rename). States:
`starting`, `meshing`, `solving`, `writing`, `done`, `error`.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

STATUS_FILE = "status.json"
STATES = ("starting", "meshing", "solving", "writing", "done", "error")
BUSY_STATES = ("starting", "meshing", "solving", "writing")
# The names that a run writes into `verify/`. A new real run removes these and nothing else.
OUTPUT_FILES = ("verify.json", "fields.npz", "mesh.npz", STATUS_FILE)


def clear_outputs(out_root: Path) -> list[str]:
    """Remove the files and the `h<size>` folders of an earlier run from `out_root`. Return the names removed.

    Other files in the folder stay. A new run does not mix its files with the files of another run.
    """
    removed = []
    if not out_root.is_dir():
        return removed
    for name in OUTPUT_FILES:
        for f in (out_root / name, out_root / (name + ".tmp")):
            if f.exists():
                f.unlink()
                removed.append(f.name)
    for d in sorted(out_root.iterdir()):
        if d.is_dir() and d.name.startswith("h") and d.name[1:].replace(".", "", 1).isdigit():
            shutil.rmtree(d)
            removed.append(d.name + "/")
    return removed


class StatusWriter:
    """Writes `status.json` in `out_root`. `update(state, **fields)` merges the fields and writes the file."""

    def __init__(self, out_root: Path, **fixed):
        self.path = Path(out_root) / STATUS_FILE
        self.data = dict(format=1, pid=os.getpid(), started_at=time.time(), **fixed)

    def update(self, state: str, **fields) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        self.data.update(fields, state=state, updated_at=time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        os.replace(tmp, self.path)


def read_status(out_root) -> dict | None:
    """The content of `status.json` in `out_root`, or None if it is missing or not readable."""
    try:
        data = json.loads((Path(out_root) / STATUS_FILE).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
