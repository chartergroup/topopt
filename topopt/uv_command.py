"""The `uv` of a repair command (REL-9).

The launcher `scripts/topopt-env.sh` uses the `uv` of the user when there is one on PATH. When there is none, it downloads a pinned
uv into the environment folder and exports `TOPOPT_UV=<full path>` to its children. A repair command that topopt prints for the user
or the agent (`uv pip install ...`) must then name that file, because a bare `uv` is not on PATH.

`uv_command()` is the text to put in front of `pip install ...`: the quoted full path in `TOPOPT_UV` when that names an executable file,
else `uv`. A stale or wrong `TOPOPT_UV` gives `uv` (the command stays runnable on a machine that has uv on PATH). Standard library only.
"""
from __future__ import annotations

import os
import shlex


def uv_command() -> str:
    """`'<TOPOPT_UV>'` (shell-quoted) when the variable names an executable file with a full path, else `uv`."""
    path = os.environ.get("TOPOPT_UV", "")
    if path and os.path.isabs(path) and os.path.isfile(path) and os.access(path, os.X_OK):
        return shlex.quote(path)
    return "uv"
