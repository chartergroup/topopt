"""Tests for the role `finish` and its call (CONTRACTS.md section 25.8).

The role text names tools, keys, views and sections of CONTRACTS.md. These tests check that each name exists in the code
and in the contract, that the role holds no number of its own, and that the skill `start-here` and the role `lead-design`
call the pass after each successful run (both copies of each).
"""
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt import render_input  # noqa: E402
from topopt_mcp import server, team  # noqa: E402

ROLE = (ROOT / "topopt_mcp" / "roles" / "finish.md").read_text(encoding="utf-8")
CONTRACTS = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
START_HERE = ("skills/start-here/SKILL.md", ".agents/skills/start-here/SKILL.md")
LEAD = ("topopt_mcp/roles/lead-design.md", "agents/lead-design-engineer.md")


def one_line(path):
    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


def test_finish_is_the_last_role_and_has_an_agent_copy():
    assert team.ROLE_IDS[-1] == "finish"
    role = team.load_roles()["finish"]
    assert role.agent == "finish-engineer" and (ROOT / "agents" / "finish-engineer.md").is_file()
    assert "does not exist yet" in role.text


def test_the_tools_and_parameters_that_the_role_names_exist():
    tools = {t.name: t for t in server.mcp._tool_manager.list_tools()}
    named = set(re.findall(r"`([a-z_]+)`", ROLE))
    for name in ("render_part", "print_check", "cleanup_list", "apply_marks", "ratchet_run", "ratchet_status", "run_topopt",
                 "compare_runs", "verify_run"):
        assert name in named and name in tools, name
    run_topopt = set(inspect.signature(tools["run_topopt"].fn).parameters)
    assert {"resume", "allow_changed_inputs", "fine_from", "fine_voxel", "out_dir"} <= run_topopt
    assert {"estimate_only"} <= set(inspect.signature(tools["ratchet_run"].fn).parameters)
    assert {"views", "regions"} <= set(inspect.signature(tools["render_part"].fn).parameters)


def test_the_sections_that_the_role_names_are_headings_of_the_contract():
    sections = set(re.findall(r"section (\d+(?:\.\d+)?)", ROLE))
    assert {"9.1", "15", "19", "24.2", "24.6", "29", "29.3", "32", "36", "41"} <= sections
    assert "41.8" not in sections
    for number in sections:
        assert re.search(rf"^#{{2,3}} {re.escape(number)}\.? ", CONTRACTS, re.M), number


def test_the_view_fl2_of_the_role_is_a_valid_view_and_the_mesh_key_is_a_setting():
    view = "fl2:-1.333,-1,0.373"
    assert f"`{view}`" in ROLE
    parsed = render_input.parse_views([view])
    assert parsed and parsed[0][0] == "fl2"
    assert "`settings.mesh_method`" in ROLE and "`sdf`" in ROLE and "`blur`" in ROLE


def test_the_role_gives_no_number_of_its_own():
    """No threshold for a step (the step fact of #361 is NO-GO): numbers are only list numbers, sections and the view."""
    text = ROLE.partition("\n---\n")[2]
    # The section "Voxel steps are a known limit" (#452) gives the facts of the drone-arm example (a share, a tilt, a step length); they are
    # facts about one example, not a threshold. tests/test_voxel_steps_text.py pins that section.
    # Only the pinned paragraph is cut (FIRST to LAST); tests/test_voxel_steps_text.py asserts that the section holds nothing else.
    from test_voxel_steps_text import FIRST, LAST
    assert text.count(FIRST) == 1
    text = re.sub(re.escape(FIRST) + r".*?" + re.escape(LAST), "", text, flags=re.S)
    assert "8.9" not in text
    text = re.sub(r"section \d+(\.\d+)?", "", text)
    text = re.sub(r"`[^`]*`", "", text)   # names of keys, views and tools hold digits (`fl2`, `volume_final_mm3`)
    text = re.sub(r"^\d\. ", "", text, flags=re.M)
    assert not re.findall(r"\d", text), re.findall(r".{20}\d.{10}", text)


def test_start_here_and_lead_design_call_the_finishing_pass_after_each_successful_run():
    for path in START_HERE:
        text = one_line(path)
        assert "After each successful run, call the finishing pass before the final report to the user" in text, path
        assert "`finish`" in text and "`topopt:finish-engineer`" in text and "does the pass itself" in text, path
    assert (ROOT / START_HERE[0]).read_text() == (ROOT / START_HERE[1]).read_text()
    for path in LEAD:
        text = one_line(path)
        assert "4. Finish. After each successful run, before the final review, call the role `finish`" in text, path
        assert 'team_guide("finish")' in text and "does the pass itself" in text, path


def test_route_2_gives_the_price_source_and_the_run_facts_first():
    """The optimizer stiffness does not change with the mesh step: the role points to the tet value and to the lever card."""
    route = ROLE.partition("2. Mesh step")[2].partition("3. Edits")[0]
    for word in ("`metrics.mesh.method`", "`metrics.mesh.fallback`", "`verify_run`", "`topopt://levers`", "voxel model", "NEW file"):
        assert word in route, word
    assert "seconds" in route and "minutes" in route


def test_route_3_names_the_print_check_parameters_that_exist_and_the_stiffness_route():
    tools = {t.name: t for t in server.mcp._tool_manager.list_tools()}
    parameters = set(inspect.signature(tools["print_check"].fn).parameters)
    route = ROLE.partition("3. Edits")[2].partition("4. Fine pass")[0]
    for name in ("path", "yaml_path", "resolution_mm"):
        assert name in parameters and f"`{name}`" in route, name
    assert "`edited.stl`" in route and "`stiffness`" in ROLE and "no `run.json`" in ROLE
    assert "of the edited STL" not in ROLE


def test_the_end_report_takes_the_after_states_from_the_tool_reply():
    report = ROLE.partition("## At the end of your work")[2]
    assert "`unknown_with_limit` is only in `run_result`" in report and "as it is" in report


def test_the_role_follows_the_requirement_rule_and_the_fine_pass_goes_with_an_estimate():
    assert "You do not change the design" not in ROLE and "only as \"Who decides\" says" in ROLE
    assert "a sub-agent gives it to the caller" in ROLE and "keep the run files of the first run as they are" in ROLE
    for path in START_HERE:
        text = one_line(path)
        assert "is an offer to the user with the estimate" in text and "`topopt:fea-engineer` (loads, supports, solver, run), `topopt:print-engineer`" in text, path


def test_lead_design_keeps_its_buckling_pointer_and_closing_line():
    for path in LEAD:
        text = one_line(path)
        assert "Buckling: the FEA role says when to turn on the check" in text and "A printable part is not a certified part." in text, path
