"""Tests for the surface of the MCP tools: the title and the four hints of each tool (P9-11), and the reply key guides.

The title and the hints live in one table, `TOOL_SURFACE` in topopt_mcp/server.py. `_TopoptServer.tool` gives them to the SDK.
CONTRACTS.md section 10 has the sort rules.
"""
import asyncio
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt_mcp import replies, server  # noqa: E402

HINTS = ("read_only_hint", "destructive_hint", "idempotent_hint", "open_world_hint")


def listed_tools():
    return {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}


# --- the title and the four hints ----------------------------------------------------------------------------------

def test_every_tool_has_a_title_and_the_four_hints():
    tools = listed_tools()
    assert len(tools) >= 27
    for name, tool in tools.items():
        assert isinstance(tool.title, str) and tool.title.strip(), f"{name}: no title"
        assert tool.annotations is not None, f"{name}: no annotations"
        for hint in HINTS:
            assert isinstance(getattr(tool.annotations, hint), bool), f"{name}: {hint} is not set (the spec default would apply)"
        if tool.annotations.read_only_hint:
            assert tool.annotations.destructive_hint is False, f"{name}: a read-only tool is not destructive"


def test_every_row_of_the_table_has_a_tool_and_every_tool_has_a_row():
    tools = listed_tools()
    assert set(server.TOOL_SURFACE) == set(tools), (sorted(set(server.TOOL_SURFACE) - set(tools)), sorted(set(tools) - set(server.TOOL_SURFACE)))


def test_the_tool_list_carries_the_values_of_the_table():
    for name, tool in listed_tools().items():
        row = server.TOOL_SURFACE[name]
        assert tool.title == row.title, name
        assert (tool.annotations.read_only_hint, tool.annotations.destructive_hint, tool.annotations.idempotent_hint,
                tool.annotations.open_world_hint) == (row.read_only, row.destructive, row.idempotent, row.open_world), name


def test_a_tool_that_writes_again_derived_output_or_removes_a_dead_record_is_not_destructive():
    """run_result writes the part card result/report.md again (derived output); environment, verify_status and ratchet_status remove the
    record of a process that has ended. None of them is read-only, none is destructive, and each is idempotent (CONTRACTS.md section 10)."""
    tools = listed_tools()
    for name in ("run_result", "environment", "verify_status", "ratchet_status"):
        a = tools[name].annotations
        assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint) == (False, False, True), name


def test_titles_differ_and_carry_no_item_code():
    titles = [row.title for row in server.TOOL_SURFACE.values()]
    assert len(set(titles)) == len(titles), "two tools have one title"
    for title in titles:
        assert not re.search(r"\b[A-Z]\d*-\d+[a-z]?\b", title), f"{title!r} looks like an item code"


NETWORK_CLIENTS = {"urllib.request", "urllib3", "http.client", "requests", "httpx", "aiohttp", "ftplib", "smtplib", "websockets", "websocket"}


def test_no_module_of_the_server_or_the_engine_imports_a_network_client():
    """The table says open_world false for every tool: no tool reaches the network or a system outside this machine. This guards the claim.
    The dashboard serves on 127.0.0.1 only (http.server, a server, not a client). A tool that adds a download changes this test and its row."""
    import ast
    found = {}
    for folder in ("topopt", "topopt_mcp"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names] if isinstance(node, ast.ImportFrom) else []
                for name in names:
                    if name in NETWORK_CLIENTS or name.split(".")[0] in {"requests", "httpx", "aiohttp", "urllib3"}:
                        found.setdefault(str(path.relative_to(ROOT)), set()).add(name)
    assert not found, found
    assert not [name for name, row in server.TOOL_SURFACE.items() if row.open_world]


def test_a_tool_with_no_row_stops_the_registration():
    fresh = server._TopoptServer("test")
    with pytest.raises(ValueError, match="no row in TOOL_SURFACE"):
        @fresh.tool()
        def tool_that_has_no_row() -> dict:
            """A tool that is not in the table."""
            return {}


def test_the_decorator_needs_its_brackets():
    fresh = server._TopoptServer("test")
    with pytest.raises(TypeError, match="must be called"):
        @fresh.tool
        def list_runs() -> dict:
            return {}


def test_the_table_gives_the_hints_to_a_tool_of_another_server_instance():
    fresh = server._TopoptServer("test")

    @fresh.tool()
    def list_runs() -> dict:
        """A tool with the name of a row."""
        return {}

    (tool,) = asyncio.run(fresh.list_tools())
    row = server.TOOL_SURFACE["list_runs"]
    assert tool.title == row.title and tool.annotations.read_only_hint is True and tool.annotations.destructive_hint is False


def test_the_build_script_finds_every_tool_that_the_table_names():
    """packaging/mcpb/build.py reads the `@mcp.tool(...)` decorators with `ast`: the table does not hide a tool from it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mcpb_build_script", ROOT / "packaging" / "mcpb" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found = [tool["name"] for tool in module.server_tools(ROOT / "topopt_mcp" / "server.py")]
    assert sorted(found) == sorted(server.TOOL_SURFACE)


# --- the reply key guides: resource template topopt://replies/{tool} ---------------------------------------------------

MOVED = ("apply_marks", "build_regions", "check_problem", "marks_to_hints", "render_part", "run_result", "environment")
GUIDES = tuple(sorted(MOVED + ("run_topopt",)))   # run_topopt (#318): the detail of the fine pass; its text names fine_from, fine_pass and the address

# The key names that the text of each tool explained before the lines moved to the guide (P9-11). A key cannot drop out of the guide
# by accident: add a key to the reply of one of these tools, add it to the guide and to this list.
OLD_KEYS = {
    "check_problem": ["grid", "voxel_counts", "load_regions", "solid_regions", "active_voxels", "forced_solid_share", "free_dofs",
                      "connectivity", "solver", "backends", "time_estimate", "gpu_mg", "memory_estimate", "lost_force", "max_lost_percent",
                      "warnings", "assumptions", "derived", "filter", "min_feature_note", "robust", "time_model", "estimates",
                      "estimate_bounds", "region_planes", "changes_part", "recommendation", "hint", "snapshot_disk_mb", "solid_pull",
                      "interface_taken", "trim_mm", "buckling_off", "memory_estimate_bytes",
                      "Key mirror:", "orphans", "Key perimeter:", "perimeter_max_mm2", "p_min_mm2", "bound_over_p_min", "fixed_part_mm2",
                      "Key repeated_load_cases:", "relations", "weights", "compliance_limits", "stress_limits_mpa", "objective_weight",
                      "remote_force", "buckling_row", "sign_rows", "limit_rows"],
    "build_regions": ["ok", "name", "run_dir_rel", "yaml_rel", "files_rel", "voxel_size", "voxel_auto", "retries", "builder_ok", "problems",
                      "report", "check", "check_error", "builder_warnings", "other_files_rel", "overwrote_existing_folder", "warnings",
                      "assumptions", "derived", 'check["lost_force"]'],
    "apply_marks": ["marks", "partly_applied", "rolled_back", "guard", "edit_reach_mm", "clipped_by_frozen_mm3", "thickness_near_mm",
                    "dropped_bodies", "result/stress.npz", "totals", "frozen_exact", "restored_mm3", "triangles_replaced", "surface_quality",
                    "routes", "files", "problems", "out_dir_rel", "edited_rel", "report_rel"],
    "run_topopt": ["fine_from", "fine_pass", "fine_voxel", "fine_passes", "fine_tol", "fine_run_dir", "fine_run_dir_rel", "fine_run", "tolerance_from",
                   "voxel_from", "max_from", "schedule_offset", "settings.fine_pass.objective_change", "<out_dir>-fine", "guards", "refused",
                   "stop_run", "continue_from", "brief_stated", "brief_facts"],
    "environment": ["problem", "ram", "cores", "recommendations", "license_note", "speedup_factor", "gpu_solver", "used", "device", "case",
                    "self_test", "why", "to_use_the_gpu", "topopt_version", "code_folder", "checkout", "last_fetch_utc", "update_command",
                    "python", "platform", "packages", "tet", "backends", "resolved_by", "solver_setting", "experimental_solvers", "opencl",
                    "auto_case", "auto_solver", "auto_choice", "threads", "run_interpreter", "note", "work_folder", "server", "other_servers"],
    "run_result": ["metrics.verdict", "grey_design", "lost_force", "levers", "kind", "layer", "when", "final_stl", "nominal_stl", "nominal_stl_rel",
                   "interface_warnings", "mesh_fallbacks", "part_only_stl", "part_only_stl_rel", "report_md", "report_md_rel", "print_check", "unknown_with_limit",
                   "metrics.interfaces.integrity", "interface_tests", "metrics.buckling", "metrics.part_only", "contact", "thinnest_wall_mm",
                   "area_below_voxel_mm2", "log_tail", "possible_causes", "independent_check",
                   "verify_partial", "topopt://levers"],
    "marks_to_hints": ["summary", "marks", "skipped_by_option", "no_effect", "facts_only", "option_used", "region_file", "proposed_yaml",
                       "labels_changed", "volume_target", "check", "grid", "problems", "out_dir_rel", "yaml_rel", "report_rel"],
    "render_part": ["out_dir_rel", "montage_rel", "width", "height", "bytes", "views", "section", "role", "bbox", "triangles", "watertight",
                    "winding_consistent", "back faces", "missing files", "notes", "font", "load_arrows"],
}


def read_text_resource(uri):
    contents = list(asyncio.run(server.mcp.read_resource(uri)))
    assert len(contents) == 1
    assert contents[0].mime_type == "text/markdown"
    return contents[0].content


def test_the_guide_has_the_moved_tools_the_fine_pass_guide_and_the_old_key_list_has_the_same():
    assert tuple(sorted(replies.REPLY_KEYS)) == GUIDES
    assert sorted(OLD_KEYS) == list(GUIDES)


@pytest.mark.parametrize("tool", MOVED)
def test_the_tool_text_names_its_guide_and_does_not_repeat_the_lines(tool):
    text = listed_tools()[tool].description
    assert f"Reply keys: resource topopt://replies/{tool} (the meaning of each key)." in text, tool
    for line in replies.REPLY_KEYS[tool]:
        assert line not in text, f"{tool}: a moved line is still in the tool text"
    assert replies.uri(tool) == f"topopt://replies/{tool}"


def test_the_run_topopt_text_names_its_guide_and_the_brief_keys_are_only_in_the_guide():
    """#174: the sentence about brief_stated and brief_facts moved from the tool text to the guide of run_topopt."""
    text = listed_tools()["run_topopt"].description
    assert "Reply keys: resource topopt://replies/run_topopt" in text
    assert "brief_stated" not in text and "brief_facts" not in text
    guide = replies.guide("run_topopt")
    assert "The reply has `brief_stated` (true when a current brief is recorded) and `brief_facts`." in guide
    assert "A changed method setting does not stop: `brief_facts` names it." in guide


@pytest.mark.parametrize("tool", GUIDES)
def test_reading_the_uri_gives_the_moved_text(tool):
    text = read_text_resource(f"topopt://replies/{tool}")
    assert text == replies.guide(tool)
    assert text.startswith(f"Reply keys of {tool}.")
    for line in replies.REPLY_KEYS[tool]:
        assert line in text


@pytest.mark.parametrize("tool", GUIDES)
def test_the_guide_has_every_key_that_the_old_text_explained(tool):
    text = read_text_resource(f"topopt://replies/{tool}")
    missing = [key for key in OLD_KEYS[tool] if key not in text]
    assert not missing, f"{tool}: the guide lost {missing}"


def test_an_unknown_name_gives_a_short_text_that_names_the_guides():
    text = read_text_resource("topopt://replies/no_such_tool")
    assert "no_such_tool" in text
    for tool in GUIDES:
        assert tool in text
    for tool in GUIDES:
        for line in replies.REPLY_KEYS[tool]:
            assert line not in text
    assert text == replies.guide("no_such_tool")


def test_the_text_of_run_topopt_names_the_fine_pass_guide_and_keeps_the_detail_out():
    text = listed_tools()["run_topopt"].description
    assert replies.uri("run_topopt") in text and "fine_from" in text and "fine_pass" in text
    for line in replies.REPLY_KEYS["run_topopt"]:
        assert line not in text
    assert len(text) <= 4096 - 20, len(text)   # some margin under the cut of the client


def test_the_template_is_listed_next_to_the_lever_cards():
    templates = {t.uri_template: t for t in asyncio.run(server.mcp.list_resource_templates())}
    assert replies.TEMPLATE in templates and "topopt://levers/{query}" in templates
    assert templates[replies.TEMPLATE].mime_type == "text/markdown"
    assert templates[replies.TEMPLATE].name == replies.NAME and templates[replies.TEMPLATE].description == replies.DESCRIPTION


def test_the_guide_follows_the_freedom_rule():
    """The lines were in the tool texts, where the tests check that there is no habitual must or never. They keep that check in the guide."""
    for tool in GUIDES:
        assert not re.search(r"\b(must|never|always|forbidden)\b", replies.guide(tool), re.I), tool


@pytest.mark.parametrize("tool", MOVED)
def test_the_first_line_of_the_tool_text_is_still_a_sentence(tool):
    """packaging/mcpb/build.py puts the first docstring line in the manifest of the extension."""
    first = (getattr(server, tool).__doc__ or "").strip().splitlines()[0]
    assert first[0].isupper() and first.endswith("."), first
