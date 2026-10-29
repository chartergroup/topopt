"""K1-FU-C (#145): the list of result keys of `verify_run` is in the card `verify_run`, and the tool text points to the card.

The tool text of `verify_run` was full (4094 of 4096 characters). The paragraph "Result keys (verify_status, when done): ..." and the line
"YAML blocks add keys: ..." moved to the card `verify_run` (`topopt/levers_cards.py`), word for word. The tool text keeps `ok`, `exit_code` with
its values and the key of the p99.9 value, and one pointer. These tests check that no key name was lost and that the tool text is shorter.
"""
import asyncio
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("mcp")

from topopt import levers  # noqa: E402
from topopt_mcp import server  # noqa: E402

# The old text, copied from main 74a1e46 (`topopt_mcp/server.py`, the docstring of `verify_run`; 1,080 characters from the first word).
OLD_PARAGRAPH = (
    "Result keys (verify_status, when done): ok, exit_code (0 done, 3 limit failed), run, yaml_vs_run (YAML against run.json), material, stl, band_mm, "
    "backend, extras, memory_limit, estimates_before_meshing, mesh (nodes, tets, dofs, quality, P2 volume over STL volume), regions (per region: band, "
    "counts, area, centroid, counts_by_band; load regions also normal_check), supports (joint and spring forces), memory, timing_s, cases (per load "
    "case: von_mises with p50, p95, p99, p99.9 by volume, peak_qp_MPa, peak_vertex_MPa, peak_qp_over_p99_9, the same without bad tets, near_regions "
    "(p99.9 without the points near the regions; the share of the top 0.1 % volume near each), top_elements; linear_solve, equilibrium, forces with "
    "the resultant error and load_centre (optimizer centre, P2 traction centroid, distance, moment ratio)), load_band_sensitivity, levels, limit, "
    "gate, files.")
OLD_LINE = (
    "YAML blocks add keys: buckling.check: buckling (mode_facts, rigid_body), buckling_check; limits rows: motion; bore_checks: bore_checks, "
    "bore_facts, goodman, pressfit if press_fit. No pressfit with a joint.")
OLD_LENGTH = 4094              # the length of the description of `verify_run` on main 74a1e46
OLD_TEXT_LENGTH = 1080         # the paragraph and the line, from "Result keys" to the end of the description

# Every key name of the old text, typed by hand. The first test below checks that the list is complete (every snake_case word of the old text is in it).
KEYS = (
    "ok", "exit_code", "run", "yaml_vs_run", "material", "stl", "band_mm", "backend", "extras", "memory_limit", "estimates_before_meshing",
    "mesh", "nodes", "tets", "dofs", "quality", "regions", "band", "counts", "area", "centroid", "counts_by_band", "normal_check", "supports",
    "memory", "timing_s", "cases", "von_mises", "p50", "p95", "p99", "p99.9", "peak_qp_MPa", "peak_vertex_MPa", "peak_qp_over_p99_9", "near_regions",
    "top_elements", "linear_solve", "equilibrium", "forces", "load_centre", "load_band_sensitivity", "levels", "limit", "gate", "files",
    "buckling.check", "buckling", "mode_facts", "rigid_body", "buckling_check", "motion", "bore_checks", "bore_facts", "goodman", "pressfit", "press_fit",
)


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 60))


def description():
    return next(t.description for t in run(server.mcp.list_tools()) if t.name == "verify_run")


def pointer(text):
    """The address of the card that the tool text gives for the result keys: `topopt://levers/verify_run` (not the cards `verify_run:gate` and `:part_only`)."""
    found = re.findall(r"topopt://levers/verify_run(?![:\w])", text)
    assert len(found) == 1, f"the tool text has {len(found)} pointers to the card verify_run"
    return found[0]


def test_the_list_of_key_names_is_the_list_of_the_old_text():
    old = OLD_PARAGRAPH + "\n" + OLD_LINE
    assert len(old) == OLD_TEXT_LENGTH
    assert all(k in old for k in KEYS), [k for k in KEYS if k not in old]
    snake = set(re.findall(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b", old)) - {"verify_status"}       # the tool that gives the result, no key of it
    assert snake <= set(KEYS), sorted(snake - set(KEYS))


def test_the_card_holds_the_old_text_word_for_word_and_every_key_name():
    card = levers.reference("verify_run")
    assert OLD_PARAGRAPH in card and OLD_LINE in card
    assert [k for k in KEYS if k not in card] == []


def test_the_address_in_the_tool_text_gives_the_card_with_the_keys():
    address = pointer(description())
    contents = list(run(server.mcp.read_resource(address)))
    assert len(contents) == 1 and contents[0].mime_type == "text/markdown"
    text = contents[0].content
    assert "#### verify_run\n" in text and OLD_PARAGRAPH in text and OLD_LINE in text


def test_the_tool_text_is_shorter_keeps_the_first_keys_and_has_no_key_list():
    text = description()
    assert len(text) < OLD_LENGTH           # the old text (1,080 characters) went out; one pointer sentence came in
    pointer(text)
    assert "Result keys (verify_status, when done): ok, exit_code (0 done, 3 limit failed)" in text
    assert "cases.<case>.von_mises.p99_9_by_volume_MPa" in text
    assert OLD_PARAGRAPH not in text and OLD_LINE not in text
    # a key name that only the old list had is gone from the tool text
    for key in ("peak_vertex_MPa", "timing_s", "estimates_before_meshing", "linear_solve", "top_elements", "bore_facts", "rigid_body", "mode_facts"):
        assert key not in text, key
