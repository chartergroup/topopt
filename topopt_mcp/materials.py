"""The material reference of topopt (CONTRACTS.md section 25.7): a read-only document that the MCP server offers as a resource.

The source is one file: `topopt_mcp/materials/reference.md`, shipped as package data. It holds data sheet ranges of printed PETG
and PLA, with the address of each sheet and what the numbers leave out. The engine never reads it, no
tool returns a number from it, and there is no default allowable (decision D7). The sourced filament catalog is another file
(`topopt/data/materials.yaml`, read by `topopt.materials_catalog`; decision D7, revised on 2026-10-24).

This module needs only the standard library.
"""
from __future__ import annotations

import functools
from importlib import resources

URI = "topopt://materials/reference"
NAME = "material-reference"
TITLE = "Material reference: printed PETG and PLA, data sheet ranges"
DESCRIPTION = ("Read-only. Tensile strength ranges of printed PETG and PLA from data sheets, with the test conditions, "
               "the address of each sheet and what the numbers leave out. Topopt has no default allowable: you choose it.")
FILE = ("materials", "reference.md")   # the place of the file in the package `topopt_mcp`


class MaterialFileError(Exception):
    """The material reference file is missing, unreadable or empty."""


def read_reference() -> str:
    """The text of the file. Raise `MaterialFileError` if it cannot be read or has no text."""
    place = resources.files("topopt_mcp").joinpath(*FILE)
    try:
        text = place.read_text(encoding="utf-8")
    except OSError as exc:
        raise MaterialFileError(f"material reference: cannot read topopt_mcp/{'/'.join(FILE)} ({exc})") from exc
    if not text.strip():
        raise MaterialFileError(f"material reference: topopt_mcp/{'/'.join(FILE)} is empty")
    return text


@functools.lru_cache(maxsize=1)
def load_reference() -> str:
    """The text of the installed package, read once."""
    return read_reference()
