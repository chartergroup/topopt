"""The names of the options in messages: the command line (`--max-gb`) or the MCP tool (`max_gb`).

An error text that names an option must name the one that the reader can use. The CLI is the default.
The MCP server sets the style "mcp" for its calls (`use_style`), and for the child process that it starts
(the env var `TOPOPT_VERIFY_STYLE`).
"""
from __future__ import annotations

import contextlib
import contextvars
import os

OPTIONS = {
    "mesh_size": ("--mesh-size", "mesh_size_mm"),
    "levels": ("--levels", "levels_mm"),
    "region_band": ("--region-band-mm", "region_band_mm"),
    "allowable": ("--allowable-mpa", "allowable_mpa"),
    "percentile": ("--percentile", "percentile"),
    "max_gb": ("--max-gb", "max_gb"),
    "force": ("--force", "force"),
    "solver": ("--solver", "solver"),
    "yaml": ("--yaml", "yaml_path"),
    "near": ("--near-mm", "near_region_mm"),
    "sensitivity": ("--no-sensitivity", "load_sensitivity"),
    "part_only": ("--part-only", "part_only"),
}

_style: contextvars.ContextVar = contextvars.ContextVar("topopt_verify_style", default=None)


def style() -> str:
    """"cli" or "mcp"."""
    s = _style.get() or os.environ.get("TOPOPT_VERIFY_STYLE") or "cli"
    return "mcp" if s == "mcp" else "cli"


@contextlib.contextmanager
def use_style(name: str):
    """Use the option names of `name` ("cli" or "mcp") inside the block."""
    token = _style.set(name)
    try:
        yield
    finally:
        _style.reset(token)


def opt(key: str) -> str:
    """The name of an option in the current style: `opt("max_gb")` is `--max-gb` or `max_gb`."""
    cli, mcp = OPTIONS[key]
    return mcp if style() == "mcp" else cli
