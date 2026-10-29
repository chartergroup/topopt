"""Tests for the lever cards (CONTRACTS.md section 41, item P7 PR-1).

One table: topopt/levers_cards.py (data). One function makes the text: topopt/levers.py. The MCP resource `topopt://levers` and the command
`topopt levers` print that text. The key lists of the loader are named constants in topopt/config.py (`KEY_BLOCKS`), and the tests derive
the accepted key paths from them: a new key without a card fails here.
"""
import asyncio
import ast
import contextlib
import dataclasses
import hashlib
import inspect
import io
import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from topopt import config, levers  # noqa: E402
from topopt import levers_cards as lc  # noqa: E402
from topopt.cli import build_parser, main  # noqa: E402
from topopt_mcp import server  # noqa: E402

from test_optimize import box, make_case  # noqa: E402

URI = "topopt://levers"
HABIT = re.compile(r"\b(must|never|always)\b", re.I)  # the freedom rule: facts and options, no habitual caps


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 60))


def yaml_cards():
    return [c for c in lc.CARDS if c.group == "yaml"]


def tool_cards():
    return [c for c in lc.CARDS if c.group == "tool"]


# --- the keys: one card for each accepted key path -----------------------------------------------------------------------

def test_every_accepted_key_path_has_exactly_one_card():
    accepted = set(config.accepted_key_paths())
    assert len(accepted) > 200                       # 214 at the base; a block that lost its keys would show here
    held = [k for c in yaml_cards() for k in (c.key, *c.also)]
    assert len(held) == len(set(held)), sorted({k for k in held if held.count(k) > 1})
    assert set(held) >= accepted, f"keys without a card: {sorted(accepted - set(held))}"


def test_no_card_names_a_key_the_loader_does_not_accept():
    accepted = set(config.accepted_key_paths())
    stale = [k for c in yaml_cards() for k in (c.key, *c.also) if k not in accepted]
    assert not stale, f"cards of keys that the loader does not accept: {stale}"


def test_the_accepted_key_paths_come_from_the_constants_of_the_loader():
    paths = set(config.accepted_key_paths())
    for block, (required, optional) in config.KEY_BLOCKS.items():
        for key in (*required, *optional):
            if block == "joints[].section":
                assert f"{block}.{key}" in paths
            else:
                assert (f"{block}.{key}" if block else key) in paths, (block, key)
    assert "settings.penal" in paths and "load_cases[].forces[].force" in paths and "bore_checks.goodman.materials.<label>.su_mpa" in paths


def test_the_loader_calls_hold_no_inline_key_list():
    """Item 2 of the brief: every `_check_keys` / `_bc_keys` call reads a named constant, not a list typed at the call."""
    tree = ast.parse((ROOT / "topopt" / "config.py").read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in {"_check_keys", "_bc_keys"}:
            for arg in node.args[3:5]:
                if isinstance(arg, (ast.List, ast.Tuple, ast.Set)) and arg.elts:
                    bad.append(node.lineno)
    assert not bad, f"inline key lists in topopt/config.py at lines {bad}"


# --- the fields of a card -------------------------------------------------------------------------------------------------

def test_the_table_has_no_fault():
    assert levers.problems() == []


def test_card_fields_are_valid_and_complete():
    for c in lc.CARDS:
        assert c.kind in lc.KINDS and c.layer in lc.LAYERS and c.confidence in lc.CONFIDENCE and c.group in lc.GROUPS, c.key
        assert c.effect.strip() and not c.effect.startswith(" "), c.key
        assert (c.kind == "both") == bool(c.kind_note), f"{c.key}: a card of kind both has a kind_note, and no other card has one"
        if c.layer == 3:
            assert c.call.strip() and c.cost.strip(), f"{c.key}: a layer 3 card names its call and its cost"
        if c.confidence == "measured":
            assert c.evidence and all(e.runs.strip() and e.part_class in lc.PART_CLASSES for e in c.evidence), c.key
        else:
            assert not c.evidence, c.key
        for s in c.symptoms:
            assert s.words.strip() and s.detect.strip(), c.key


def test_a_card_that_is_measured_has_a_row_with_run_ids_and_a_part_class():
    measured = [c for c in lc.CARDS if c.confidence == "measured"]
    assert len(measured) >= 20
    for c in measured:
        assert any(e.runs and e.part_class for e in c.evidence), c.key


def test_layer_3_cards_are_the_keys_that_a_resume_of_a_finished_run_applies():
    """The layer-3 keys are the ones that the post step reads. A key that acts in the passes is layer 1."""
    layer3 = {c.key for c in yaml_cards() if c.layer == 3}
    for key in ("settings.mesh_method", "print.build_direction", "regions.not_printed", "material.density", "buckling.check", "name"):
        assert key in layer3, key
    for key in ("settings.penal", "settings.volume_fraction", "settings.move_limit", "regions.keep_out", "settings.iterations"):
        assert key not in layer3, key


def test_the_rule_for_agents_is_in_the_reference():
    text = levers.reference()
    for words in ("You change it without asking", "You may tighten one without asking", "You loosen one only with the user's consent",
                  "`record_brief`", "origin `user` is a requirement", "No tool refuses a change because of it"):
        assert words in text, words
    assert levers.RULE_TEXT in text


# --- the tools: one card for each engine tool -----------------------------------------------------------------------------

CLI_TO_TOOL = {"run": "run_topopt", "stop": "stop_run", "check": "check_problem", "render": "render_part", "verify": "verify_run",
               "printcheck": "print_check", "step": "export_step"}


def mcp_tools():
    return {t.name: t for t in run(server.mcp.list_tools())}


def cli_commands():
    parser = build_parser()
    return set(next(a for a in parser._actions if getattr(a, "choices", None) and a.dest == "cmd").choices)


def test_every_tool_card_names_a_real_tool_or_command():
    tools, commands = mcp_tools(), cli_commands()
    for c in tool_cards():
        base, _, arg = c.key.partition(":")
        assert base in tools or base in commands, c.key
        if arg:
            assert base in tools and arg in inspect.signature(getattr(server, base)).parameters, f"{c.key}: {arg} is not an argument of the tool"
    for name in lc.READ_ONLY_TOOLS:
        assert name in tools or name in commands, name


def test_every_mcp_tool_and_cli_command_has_a_card_or_is_listed_as_read_only():
    carded = {c.key.partition(":")[0] for c in tool_cards()}
    known = carded | set(lc.READ_ONLY_TOOLS)
    assert not (carded & set(lc.READ_ONLY_TOOLS)), carded & set(lc.READ_ONLY_TOOLS)
    missing_tools = sorted(set(mcp_tools()) - known)
    assert not missing_tools, f"MCP tools without a card or a read-only entry: {missing_tools}"
    missing_cli = sorted(c for c in cli_commands() if c not in known and CLI_TO_TOOL.get(c) not in known)
    assert not missing_cli, f"CLI commands without a card or a read-only entry: {missing_cli}"


# --- the reference text ---------------------------------------------------------------------------------------------------

def test_the_reference_holds_every_card_once():
    text = levers.reference()
    headings = re.findall(r"^#### (.+?)(?:  \(also: .*\))?$", text, re.M)
    assert len(headings) == len(lc.CARDS)
    assert sorted(headings) == sorted(c.key for c in lc.CARDS)
    for c in lc.CARDS:
        for key in c.also:
            assert key in text, key
    assert f"{len(lc.CARDS)} cards" in text


def test_the_reference_has_no_habitual_caps():
    text = levers.reference()
    assert not HABIT.search(text), text[max(0, HABIT.search(text).start() - 80): HABIT.search(text).end() + 40]


def test_the_symptom_index_lists_each_word_and_its_cards():
    index = levers.symptom_index()
    assert "film" in " ".join(index) and any("film" in w for w in index)
    for word, keys in index.items():
        assert keys and all(k in {c.key for c in lc.CARDS} for k in keys), word
    text = levers.reference()
    assert "## Symptom index" in text and any(f"- {w}: " in text for w in list(index)[:5])


def test_find_matches_a_key_a_key_family_a_symptom_word_and_a_fact_path():
    assert [c.key for c in levers.find("settings.penal")] == ["settings.penal"]
    assert [c.key for c in levers.find("settings.stop_rules.window")] == ["settings.stop_rules.window"]
    assert [c.key for c in levers.find("bore_checks.goodman.materials.<label>.se_mpa_source")] == ["bore_checks.goodman.materials.<label>.su_mpa"]
    assert {(c.group, c.key) for c in levers.find("buckling")} == {("yaml", "buckling"), ("tool", "buckling")}   # an exact key gives its cards only
    assert {c.key for c in levers.find("buckl")} >= {"buckling", "buckling.check", "buckling.modes"}      # a part of a word gives every match
    assert "cleanup_list" in {c.key for c in levers.find("film")}
    assert "settings.min_member_mm" in {c.key for c in levers.find("webs")}
    assert "settings.filter_solid_neighbours" in {c.key for c in levers.find("check_problem.solid_pull.share_above")}
    assert levers.find("no such lever word anywhere") == []
    assert "No card matches" in levers.reference("no such lever word anywhere")


# --- the MCP resource and the command -------------------------------------------------------------------------------------

def test_resource_is_listed_and_returns_the_reference():
    async def go():
        resources = {str(r.uri): r for r in await server.mcp.list_resources()}
        assert URI in resources
        return resources[URI], list(await server.mcp.read_resource(URI))

    res, contents = run(go())
    assert res.name == levers.NAME and res.title == levers.TITLE and res.description == levers.DESCRIPTION
    assert res.mime_type == "text/markdown" and levers.URI == URI == config.LEVERS_RESOURCE
    assert len(contents) == 1 and contents[0].mime_type == "text/markdown"
    assert contents[0].content == levers.reference()
    assert len(levers.DESCRIPTION) <= 400 and not HABIT.search(levers.DESCRIPTION)


def test_stdio_server_lists_and_serves_the_reference():
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(ROOT))

    async def go():
        async with Client(params) as client:
            resources = {str(r.uri): r for r in (await client.list_resources()).resources}
            read = await client.read_resource(URI)
            return resources, read.contents[0]

    resources, content = run(go())
    assert URI in resources and resources[URI].description == levers.DESCRIPTION
    assert content.text == levers.reference() and content.mime_type == "text/markdown"


def test_the_template_resource_gives_the_cards_of_one_query():
    """Round 2: `topopt://levers/{query}` is the text of `reference(query)`; `topopt://levers` stays the full text."""
    async def go():
        templates = {t.name: t for t in await server.mcp.list_resource_templates()}
        reads = {}
        for query in ("settings.penal", "too%20heavy", "load_cases%5B%5D.forces%5B%5D.force", "film", "no%20such%20lever%20word%20anywhere"):
            reads[query] = list(await server.mcp.read_resource(f"{URI}/{query}"))
        return templates, reads, list(await server.mcp.read_resource(URI))

    templates, reads, full = run(go())
    t = templates[levers.TEMPLATE_NAME]
    assert t.uri_template == levers.TEMPLATE == f"{URI}/{{query}}" and t.mime_type == "text/markdown"
    assert t.title == levers.TEMPLATE_TITLE and t.description == levers.TEMPLATE_DESCRIPTION and not HABIT.search(t.description)
    for query, text in {"settings.penal": "settings.penal", "too%20heavy": "too heavy", "load_cases%5B%5D.forces%5B%5D.force": "load_cases[].forces[].force",
                        "film": "film", "no%20such%20lever%20word%20anywhere": "no such lever word anywhere"}.items():
        assert len(reads[query]) == 1 and reads[query][0].mime_type == "text/markdown"
        assert reads[query][0].content == levers.reference(text), query
    assert "#### settings.penal" in reads["settings.penal"][0].content and "#### settings.beta_schedule" not in reads["settings.penal"][0].content
    assert "No card matches" in reads["no%20such%20lever%20word%20anywhere"][0].content
    assert full[0].content == levers.reference() and len(full[0].content) > 20 * len(reads["settings.penal"][0].content)
    assert levers.TEMPLATE in levers.reference()                     # the header names the template


def test_stdio_server_lists_and_serves_the_template():
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(ROOT))

    async def go():
        async with Client(params) as client:
            templates = (await client.list_resource_templates()).resource_templates
            read = await client.read_resource(f"{URI}/settings.penal")
            return templates, read.contents[0]

    templates, content = run(go())
    assert [t.uri_template for t in templates if t.name == levers.TEMPLATE_NAME] == [levers.TEMPLATE]
    assert content.text == levers.reference("settings.penal") and content.mime_type == "text/markdown"


def cli(capsys, *args):
    code = main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_the_command_prints_the_same_text_as_the_resource(capsys):
    code, out, err = cli(capsys, "levers")
    assert code == 0 and out == levers.reference() and not err


def test_the_command_finds_a_card_by_key_and_by_symptom_word(capsys):
    code, out, _ = cli(capsys, "levers", "settings.penal")
    assert code == 0 and out == levers.reference("settings.penal") and "#### settings.penal" in out and "#### settings.beta_schedule" not in out
    code, out, _ = cli(capsys, "levers", "icicles")
    assert code == 0 and "#### cleanup_list" in out and "#### settings.robust" in out
    code, out, _ = cli(capsys, "levers", "too", "heavy")                # two words are two queries, the cards of both
    assert code == 0 and "#### settings.volume_fraction" in out


def test_the_command_says_when_no_card_matches(capsys):
    code, out, err = cli(capsys, "levers", "no such lever word anywhere")
    assert code == 1 and not out and "no lever card matches" in err and "topopt levers" in err


# --- the unknown-key error names the known keys and the reference --------------------------------------------------------

def bore(extra=None):
    return {"name": "b1", "point": [10, 5, 0], "axis": [0, 0, 1], "radius_mm": 1.5, "length_mm": 8, "radius_mm_source": "s", "length_mm_source": "s", **(extra or {})}


def with_(path, value):
    """A mutator: set `d[path[0]][path[1]]... = value`; a path element that is an int is a list index."""
    def f(d):
        node = d
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = value
    return f


def put_bogus(parent, *path):
    """Add the key `bogus` to the mapping at `path` (created from `parent` when it is a dict to add)."""
    def f(d):
        node = d
        for p in path:
            node = node[p]
        node["bogus"] = 1
    return f


def chain(*fs):
    def f(d):
        for g in fs:
            g(d)
    return f


# The block path of `config.KEY_BLOCKS` -> a mutator that puts a valid entry of that block in the case and adds the key `bogus` to it.
BLOCK_CASES = {
    "": lambda d: d.update(bogus=1),
    "regions": lambda d: d["regions"].update(bogus=1),
    "regions.fixed[]": lambda d: d["regions"].update(fixed=[{"file": "plate.stl", "axes": ["x"], "bogus": 1}]),
    "load_cases[]": lambda d: d["load_cases"][0].update(bogus=1),
    "load_cases[].forces[]": lambda d: d["load_cases"][0]["forces"][0].update(bogus=1),
    "load_cases[].pressures[]": lambda d: d["load_cases"][0].update(pressures=[{"region": "tip", "value_mpa": 0.1, "bogus": 1}]),
    "material": lambda d: d.update(material={"bogus": 1}),
    "settings": lambda d: d["settings"].update(bogus=1),
    "settings.stop_rules": lambda d: d["settings"].update(stop_rules={"bogus": 1}),
    "settings.fine_pass": lambda d: d["settings"].update(fine_pass={"bogus": 1}),
    "mirror[]": lambda d: d.update(mirror=[{"normal": "y", "at_mm": 5, "bogus": 1}]),
    "limits": lambda d: d.update(limits={"bogus": 1}),
    "limits.displacement[]": lambda d: d.update(limits={"displacement": [{"region": "tip", "limit_mm": 1, "bogus": 1}]}),
    "limits.alignment[]": lambda d: d.update(limits={"alignment": [{"region_a": "tip", "region_b": "tip", "axis": [0, 0, 1], "shift_limit_mm": 1, "bogus": 1}]}),
    "buckling": lambda d: d.update(buckling={"bogus": 1}),
    "print": lambda d: d.update(print={"bogus": 1}),
    "interfaces[]": lambda d: d.update(interfaces=[{"name": "f", "kind": "face", "point": [1, 0, 0], "normal": [1, 0, 0], "bogus": 1}]),
    "springs[]": lambda d: d.update(springs=[{"region": "tip", "stiffness": [1, 1, 1], "bogus": 1}]),
    "joints[]": lambda d: d.update(joints=[{"region": "tip", "to": [0, 0, -30], "section": {"kind": "round", "d": 4}, "bogus": 1}]),
    "joints[].section": lambda d: d.update(joints=[{"region": "tip", "to": [0, 0, -30], "section": {"kind": "round", "d": 4, "bogus": 1}}]),
    "acceptance": lambda d: d.update(acceptance={"rules": [{"rule": "watertight"}], "bogus": 1}),
    "acceptance.rules[]": lambda d: d.update(acceptance={"rules": [{"rule": "stress", "bogus": 1}]}),
    "bore_checks": lambda d: d.update(bore_checks={"bores": [bore()], "goodman": {"materials": {"m": {"su_mpa": 1, "sy_mpa": 1, "se_mpa": 1, "source": "s"}}}, "bogus": 1}),
    "bore_checks.bores[]": lambda d: d.update(bore_checks={"bores": [bore({"bogus": 1})], "goodman": {"materials": {"m": {"su_mpa": 1, "sy_mpa": 1, "se_mpa": 1, "source": "s"}}}}),
    "bore_checks.goodman": lambda d: d.update(bore_checks={"bores": [bore()], "goodman": {"materials": {"m": {"su_mpa": 1, "sy_mpa": 1, "se_mpa": 1, "source": "s"}}, "bogus": 1}}),
    "bore_checks.goodman.materials.<label>": lambda d: d.update(bore_checks={"bores": [bore()], "goodman": {"materials": {"m": {"su_mpa": 1, "sy_mpa": 1, "se_mpa": 1, "source": "s", "bogus": 1}}}}),
    "bore_checks.press_fit": lambda d: d.update(bore_checks={"bores": [bore({"diametral_interference_mm": 0.1, "diametral_interference_mm_source": "s"})], "press_fit": {"cups": [{"name": "c", "k_mpa_per_mm": 1e6, "k_mpa_per_mm_source": "s"}], "bogus": 1}}),
    "bore_checks.press_fit.cups[]": lambda d: d.update(bore_checks={"bores": [bore({"diametral_interference_mm": 0.1, "diametral_interference_mm_source": "s"})], "press_fit": {"cups": [{"name": "c", "k_mpa_per_mm": 1e6, "k_mpa_per_mm_source": "s", "bogus": 1}]}}),
}


def test_the_cases_cover_every_block_of_the_loader():
    assert set(BLOCK_CASES) == set(config.KEY_BLOCKS)


BY_KIND = {"acceptance.rules[]", "interfaces[]", "joints[].section"}   # these blocks take other keys for each kind or rule: the error names the keys of that one


@pytest.mark.parametrize("block", sorted(BLOCK_CASES))
def test_an_unknown_key_names_the_known_keys_and_the_reference(tmp_path, block):
    path = make_case(tmp_path, h=2.0)
    d = yaml.safe_load(path.read_text())
    BLOCK_CASES[block](d)
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(config.UserError) as err:
        config.load_config(path)
    msg = str(err.value)
    assert "unknown key 'bogus'" in msg, msg
    required, optional = config.KEY_BLOCKS[block]
    m = re.search(r"Known keys: (.+?)\.(?: |$)", msg) or re.search(r"takes: (.+?)\)", msg)   # the form of the interface error: "(kind 'face' takes: ...)"
    assert m, msg
    named = [k.strip() for k in m.group(1).split(",")]
    assert named and set(named) <= {*required, *optional}, (named, msg)
    if block not in BY_KIND:
        assert set(named) == {*required, *optional}, (named, msg)
    assert URI in msg and "topopt levers" in msg, msg


def test_a_rule_with_no_key_of_its_own_names_rule_as_the_known_key(tmp_path):
    """The rule `watertight` takes no key besides `rule`: the error names `rule` (which the check reads first), not `none` (fix round 1, F8)."""
    path = make_case(tmp_path, h=2.0)
    d = yaml.safe_load(path.read_text())
    d["acceptance"] = {"rules": [{"rule": "watertight", "bogus": 1}]}
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(config.UserError, match=r"unknown key 'bogus' in 'acceptance.rules\[0\]'. Known keys: rule\. Options: use a known key, or remove 'bogus'\. "):
        config.load_config(path)
    d["acceptance"] = {"rules": [{"rule": "mass", "bogus": 1}]}                # a rule with keys: `rule` is in the list, the keys follow
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(config.UserError, match=r"Known keys: [^.]*\brule\b[^.]*\. Options: "):
        config.load_config(path)


def test_the_mirror_error_has_an_options_part_like_the_other_blocks(tmp_path):
    path = make_case(tmp_path, h=2.0)
    d = yaml.safe_load(path.read_text())
    d["mirror"] = [{"normal": "y", "at_mm": 0, "bogus": 1}]
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(config.UserError) as err:
        config.load_config(path)
    msg = str(err.value)
    assert "unknown key 'bogus' in 'mirror[0]'. Known keys: " in msg and "Options: use a known key, or remove 'bogus'. " in msg and URI in msg, msg


# --- the layer 3 route: a resume of a finished run runs the post step only ------------------------------------------------

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_of(folder):
    """sha256 of the history, the checkpoint and every snapshot: what a pass would change."""
    folder = Path(folder)
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()
            and (p.name in {"history.jsonl", "checkpoint.npz"} or "snapshots" in p.relative_to(folder).parts)}


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A small run with features of many blocks (mirror, a limit, a stress limit, a bore, a spring, the buckling check, a print limit)."""
    tmp = tmp_path_factory.mktemp("levers_run")
    path = make_case(tmp, h=2.0, iterations=12)
    d = yaml.safe_load(path.read_text())
    d["mirror"] = [{"normal": "y", "at_mm": 5}]
    d["limits"] = {"displacement": [{"region": "tip", "limit_mm": 5.0}]}
    d["buckling"] = {"check": True, "modes": 2, "min_factor": 1.0}      # a limit: the row writes `history.constraints[]` entries with `ks_offset`
    d["load_cases"][0]["stress_limit"] = 100.0
    d["interfaces"] = [{"name": "pin", "kind": "bore", "point": [10, 5, 0], "axis": [0, 0, 1], "radius_mm": 1.5, "length_mm": 10, "stock_mm": 0.2}]
    d["print"] = {"min_wall_mm": 1.0}
    d["springs"] = [{"region": "tip", "stiffness": [50.0, 50.0, 50.0]}]
    d["settings"]["perimeter_max_mm2"] = 100000.0
    path.write_text(yaml.safe_dump(d))
    out = tmp / "run"
    assert main(["run", str(path), "--out", str(out), "--quiet"]) == 0
    return path, out


def resumed(finished, tmp_path, edit):
    """Copy the finished run, edit the YAML, resume with the override. Return (the folder, the YAML text before, the folder before)."""
    path, base = finished
    work = tmp_path / "case"
    shutil.copytree(path.parent, work, ignore=shutil.ignore_patterns("run"))
    out = tmp_path / "run"
    shutil.copytree(base, out)
    for name in ("job.DONE", "job.guard", "job.json"):
        (out / name).unlink(missing_ok=True)
    yaml_path = work / path.name
    d = yaml.safe_load(yaml_path.read_text())
    edit(d)
    yaml_path.write_text(yaml.safe_dump(d))
    before = state_of(out)
    assert main(["run", str(yaml_path), "--out", str(out), "--quiet", "--resume", "--allow-changed-inputs"]) == 0
    return out, before


def test_the_layer_3_route_runs_the_post_step_only_for_the_mesh_method(finished, tmp_path):
    """The card of `settings.mesh_method` says layer 3: the history, the checkpoint and the snapshots stay, and the mesh files change."""
    _, base = finished
    out, before = resumed(finished, tmp_path, lambda d: d["settings"].update(mesh_method="blur"))
    assert state_of(out) == before and before                                   # no pass ran
    assert "The post step runs again" in (out / "run.log").read_text()
    assert "settings.mesh_method: <not set> -> blur" in (out / "run.log").read_text()
    assert sha(out / "result" / "final.stl") != sha(base / "result" / "final.stl")        # the mesh is made again with the other method
    assert json.loads((out / "result" / "metrics.json").read_text())["mesh"]["method"] == "blur"
    assert json.loads((out / "result" / "metrics.json").read_text())["mesh"]["requested"] == "blur"


def test_a_key_that_acts_in_the_passes_is_accepted_and_the_part_does_not_change(finished, tmp_path):
    """The hazard on the card `run_topopt:allow_changed_inputs`: the resume of a finished run accepts `settings.volume_fraction`, and nothing is designed again."""
    _, base = finished
    out, before = resumed(finished, tmp_path, lambda d: d["settings"].update(volume_fraction=0.25))
    assert state_of(out) == before
    assert sha(out / "result" / "final.stl") == sha(base / "result" / "final.stl")
    assert json.loads((out / "run.json").read_text())["volume_fraction"] == 0.25      # run.json records the new value for a part that was made at 0.3
    assert json.loads((base / "run.json").read_text())["volume_fraction"] == 0.3


def test_the_resume_appends_one_resume_point_to_the_verdict(finished, tmp_path):
    out, _ = resumed(finished, tmp_path, lambda d: d.update(name="renamed"))
    points = json.loads((out / "result" / "metrics.json").read_text())["verdict"]["resume_points"]
    assert len(points) == 1 and set(points[0]) >= {"iteration", "checkpoint", "resumed_at"}


# --- the fact paths of the symptoms ---------------------------------------------------------------------------------------

# A `metrics.*` path is checked against the metrics.json of the short run above (the fixture `finished`). Blocks that the short run cannot
# have (a joint, a robust run, a part with `regions.not_printed`) are checked against the code that writes them: each plain segment of the
# path stands as a quoted string in one of the writer files. The tests say below which paths took this second route.
WRITER_FILES = {
    "metrics.reactions": ["topopt/springs.py", "topopt/joints.py", "topopt/optimize.py"],
    "run.joints": ["topopt/joints.py"],
    "run.mirror": ["topopt/symmetry.py"],
    "check_problem.extended": ["topopt/springs.py", "topopt/joints.py", "topopt/loads.py"],
    "metrics.part_only": ["topopt/part_only.py"],
    "metrics.verdict.interface_tests[]": ["topopt/interfaces.py"],   # only a run with `interfaces[].full_surface` (#90)
    "metrics.interfaces.warnings": ["topopt/interfaces.py"],       # only a run with an interface warning (#90)
    "check_problem.derived.volume.target_mass_asked_g": ["topopt/cli.py"],   # only a YAML with `settings.target_mass_g` (I7); the key is in the dict that `derived_numbers` builds
    "run.derived.volume.target_mass_asked_g": ["topopt/cli.py"],            # the same dict goes into run.json (I7)
    "run.fine_from": ["topopt/fine_pass.py"],                                # only a fine pass (#318): `fine_pass.fine_record` builds the record
    "metrics.mass_asked_g": ["topopt/post.py"],                              # only a run whose run.json has that key (I7, review M2)
}


def segments(path):
    out = []
    for part in path.split("."):
        out.append((part[:-2], True) if part.endswith("[]") else (part, False))
    return out


def resolve(doc, segs):
    """True when the key path exists in `doc`. `[]` takes every entry of a list (an empty list proves nothing), `<name>` any key."""
    if not segs:
        return True
    (name, is_list), rest = segs[0], segs[1:]
    if name.startswith("<") and name.endswith(">"):
        return isinstance(doc, dict) and bool(doc) and any(resolve(v, rest) for v in doc.values())
    if not isinstance(doc, dict) or name not in doc:
        return False
    value = doc[name]
    if is_list:
        return isinstance(value, list) and bool(value) and any(resolve(v, rest) for v in value)
    return resolve(value, rest)


def written_by(path):
    """The second route: every plain segment of `path` (after the source) is a quoted string in a writer file of the longest matching prefix."""
    prefix = max((p for p in WRITER_FILES if path == p or path.startswith(p + ".")), key=len, default=None)
    if prefix is None:
        return False
    text = "\n".join((ROOT / f).read_text(encoding="utf-8") for f in WRITER_FILES[prefix])
    for name, _ in segments(path)[len(prefix.split(".")):]:
        if name.startswith("<"):
            continue
        if not re.search(rf"""["']{re.escape(name)}["']|\b{re.escape(name)}\s*=""", text):
            return False
    return True


@pytest.fixture(scope="module")
def documents(finished):
    path, out = finished
    history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines() if line.strip()]
    merged = {}
    for rec in history:                  # a history field can be in some records only (`stage_end`): merge them for the path check
        for k, v in rec.items():
            merged.setdefault(k, v)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert main(["check", str(path), "--json"]) == 0
    return {"metrics": json.loads((out / "result" / "metrics.json").read_text()), "run": json.loads((out / "run.json").read_text()),
            "history": merged, "check_problem": json.loads(stdout.getvalue())}


def all_facts():
    return [(c.key, f) for c in lc.CARDS for s in c.symptoms for f in s.facts]


def test_the_fact_paths_are_in_the_one_path_form():
    tools = {c.key.partition(":")[0] for c in tool_cards()} | set(lc.READ_ONLY_TOOLS)
    facts = all_facts()
    assert len(facts) > 100
    for key, f in facts:
        source, _, rest = f.partition(".")
        assert f in tools or (source in lc.FACT_SOURCES and rest), (key, f)
        assert not re.search(r"\s|`|\.\.|\.$|\[\d", f), (key, f)
    assert {f.partition(".")[0] for _, f in facts} >= {"metrics", "run", "history", "check_problem", "verify", "ratchet", "print_check"}


def test_a_symptom_without_a_fact_says_so_in_its_text():
    for c in lc.CARDS:
        for s in c.symptoms:
            if not s.facts:
                assert re.search(r"no fact yet|error text", s.detect), (c.key, s.words, s.detect)


def check_paths(documents, source):
    """The facts of one source: (paths found in the short run, paths proved by a writer file, paths with no proof)."""
    in_run, from_code, missing = [], [], []
    for key, f in all_facts():
        if f.partition(".")[0] != source:
            continue
        if resolve(documents[source], segments(f.partition(".")[2])):
            in_run.append(f)
        elif written_by(f):
            from_code.append(f)
        else:
            missing.append((key, f))
    return in_run, from_code, missing


@pytest.mark.parametrize("source", ["metrics", "run", "history", "check_problem"])
def test_every_fact_path_is_a_key_that_the_code_writes(documents, source):
    """A `metrics.*`, `run.*`, `history.*` or `check_problem.*` path is a key of the document of the short run (fixture `finished`), or
    the code that writes it is named in WRITER_FILES (a key that appears in some runs only). A path with no proof fails."""
    in_run, from_code, missing = check_paths(documents, source)
    assert not missing, f"paths that the short run does not have and no writer file proves: {missing}"
    assert in_run, f"the short run has no {source} fact at all"
    print(f"{source}: {len(in_run)} paths in the short run, {len(set(from_code))} proved by the writer files: {sorted(set(from_code))}")


def test_the_paths_of_the_tools_are_checked_by_form_only():
    """`verify.*`, `ratchet.*` and `print_check.*` paths need a tet mesh, a ratchet or a print check: the tests here check their form only (see CONTRACTS.md 41)."""
    sources = {f.partition(".")[0] for _, f in all_facts()}
    assert {"verify", "ratchet", "print_check"} <= sources


# --- the rule and the pointer in the texts that agents read ---------------------------------------------------------------

def test_the_start_here_skill_gives_the_rule_and_the_pointer_in_step_6():
    text = (ROOT / "skills" / "start-here" / "SKILL.md").read_text(encoding="utf-8")
    step = text.partition("## Step 6. Refine from feedback")[2].partition("\n## ")[0]
    assert URI in step and "topopt levers" in step
    for part in ("A method setting", "A user requirement", "tighten it without asking", "You loosen it only with the user's consent", "`record_brief`"):
        assert part in step, part
    assert (ROOT / ".agents" / "skills" / "start-here" / "SKILL.md").read_text(encoding="utf-8") == text   # the copy that other clients read
    assert not HABIT.search(step)


def test_the_server_texts_point_to_the_resource_and_run_topopt_gives_the_rule():
    """INSTRUCTIONS has no room for the pointer (the cut of 2048 characters, #380: the pointer moved to the description of `run_result`, then (#174) to its key guide, and the
    `levers` block of the reply has it too); the rule is in the description of `run_topopt`."""
    tools = mcp_tools()
    from topopt_mcp import replies
    assert URI in replies.guide("run_result") and URI in tools["run_topopt"].description and URI in tools["check_problem"].description
    assert "`levers`" in server.INSTRUCTIONS and URI not in server.INSTRUCTIONS
    run_text = tools["run_topopt"].description
    for part in ("a method setting", "you change without asking", "A user requirement", "you may tighten", "you loosen it only with the user's consent",
                 "a voxel guard", "the memory estimate against the RAM", "the machine can freeze", "`settings.mesh_method`", "record_brief"):
        assert part in run_text, part
    assert "No tool refuses a change" in levers.RULE_TEXT      # the description has no room for it (the limit of 4,096); the reference has it
    for text in (server.INSTRUCTIONS, run_text, tools["check_problem"].description):
        assert not HABIT.search(text) and len(text) <= 4096          # the limit of tests/test_plugin_manifest.py


def test_the_role_playbooks_that_list_keys_point_to_the_resource():
    for role in ("lead-design", "fea", "print"):
        assert URI in (ROOT / "topopt_mcp" / "roles" / f"{role}.md").read_text(encoding="utf-8"), role
    lead = (ROOT / "topopt_mcp" / "roles" / "lead-design.md").read_text(encoding="utf-8")
    assert "You loosen it only with the user's consent" in lead and "`record_brief`" in lead


# --- round 2: the fact for a resume of a finished run with a changed key that acts in the passes -----------------------------

from topopt.resume_fact import OPTIONS as RESUME_OPTIONS, pass_key_note  # noqa: E402


def test_the_note_names_the_keys_of_the_passes_and_no_other_key():
    changes = ["name: a -> b", "settings.penal: 3.0 -> 3.5", "settings.mesh_method: <not set> -> blur", "print.min_wall_mm: <not set> -> 1.0",
               "load_cases[0].forces[0].force: [0, 0, -2.0] -> [0, 0, -3.0]", "settings.beta_schedule[1][0]: 10 -> 12",
               "settings.volume_fraction: 0.2 -> 0.25 (the YAML is the same: a default of topopt changed)"]
    note = pass_key_note(changes, True)
    assert note["keys"] == ["settings.penal", "load_cases[].forces[].force", "settings.beta_schedule", "settings.volume_fraction"]
    for text in ("settings.penal (3.0 -> 3.5)", "load_cases[0].forces[0].force ([0, 0, -2.0] -> [0, 0, -3.0])", "settings.volume_fraction (0.2 -> 0.25)"):
        assert text in note["fact"], text
    for other in ("settings.mesh_method", "print.min_wall_mm", "name:"):
        assert other not in note["fact"]
    assert "no pass runs again" in note["fact"] and note["options"] == RESUME_OPTIONS and any("new out_dir" in o for o in note["options"])
    assert not HABIT.search(note["fact"] + " ".join(note["options"]))
    assert pass_key_note(changes, False) is None and pass_key_note([], True) is None                  # an unfinished run runs its passes
    assert pass_key_note(["name: a -> b", "settings.mesh_method: <not set> -> blur"], True) is None     # the post step reads these
    assert pass_key_note(["region file keep_out:blk.stl has other content (sha256 abc -> def)"], True)["keys"] == ["regions"]


def test_the_note_follows_the_layer_of_the_cards():
    """The table decides: a YAML card of layer 1 is named, a card of layer 3 or 4 is not."""
    checked = {1: 0, 3: 0, 4: 0}
    for c in yaml_cards():
        key = re.sub(r"\[\]", "[0]", c.key)
        if "<" in key:
            continue
        note = pass_key_note([f"{key}: 1 -> 2"], True)
        if c.layer == 1:
            assert note and note["keys"] == [c.key], c.key
        else:
            assert note is None, c.key
        checked[c.layer] += 1
    assert checked[1] > 50 and checked[3] > 10 and checked[4] > 10


def test_a_resume_of_a_finished_run_logs_the_fact_for_a_key_of_the_passes(finished, tmp_path):
    _, base = finished
    out, before = resumed(finished, tmp_path, lambda d: d["settings"].update(penal=3.5, mesh_method="blur"))
    log = [ln for ln in (out / "run.log").read_text().splitlines() if "no pass runs again" in ln]
    assert len(log) == 1 and log[0].startswith("resume: the run is finished") and "settings.penal (<not set> -> 3.5)" in log[0]
    assert "settings.mesh_method" not in log[0]
    assert state_of(out) == before                                                                 # no pass ran, as the fact says
    assert sha(out / "result" / "final.stl") != sha(base / "result" / "final.stl")                   # the post-step key took effect
    out2, _ = resumed(finished, tmp_path / "second", lambda d: d["settings"].update(mesh_method="blur"))
    assert "no pass runs again" not in (out2 / "run.log").read_text()


@pytest.fixture
def server_clean():
    server._current.clear()
    yield
    server._current.clear()


def test_run_topopt_gives_the_fact_in_its_reply(finished, tmp_path, monkeypatch, server_clean):
    monkeypatch.setattr(server, "_command", lambda *a: [sys.executable, "-c", "pass"])
    path, base = finished
    work, out = tmp_path / "case", tmp_path / "run"
    shutil.copytree(path.parent, work, ignore=shutil.ignore_patterns("run"))
    shutil.copytree(base, out)
    for name in ("job.DONE", "job.guard", "job.json"):
        (out / name).unlink(missing_ok=True)
    yaml_path = work / path.name
    d = yaml.safe_load(yaml_path.read_text())
    d["settings"]["volume_fraction"] = 0.25
    yaml_path.write_text(yaml.safe_dump(d))
    before = state_of(out)
    r = server.run_topopt(str(yaml_path), str(out), resume=True, allow_changed_inputs=True)
    server._current["proc"].wait()
    assert "error" not in r, r
    assert r["resume"]["pass_keys"] == ["settings.volume_fraction"] and r["resume"]["options"] == RESUME_OPTIONS
    assert len(r["resume"]["facts"]) == 1 and "no pass runs again" in r["resume"]["facts"][0] and state_of(out) == before   # the reply writes nothing
    # a post-step key, or no override: no `resume` key
    d["settings"]["volume_fraction"] = 0.3
    d["settings"]["mesh_method"] = "blur"
    yaml_path.write_text(yaml.safe_dump(d))
    server._current.clear()
    r = server.run_topopt(str(yaml_path), str(out), resume=True, allow_changed_inputs=True)
    server._current["proc"].wait()
    assert "error" not in r and "resume" not in r


# --- the hook evidence of the E1 series (round 2) -------------------------------------------------------------------------

HOOK_CLASSES = ("hook", "hook fine", "hook_stress", "hook_buckling-2mm")
OPTIMIZER_PRICE = re.compile(r"optimizer hang x([\d.]+), pull_off x([\d.]+), mass x([\d.]+)")
P2_KEYS = ("settings.penal", "settings.min_member_mm", "settings.filter_radius", "settings.mesh_method", "settings.voxel_size")


def card_of(key, group="yaml"):
    return next(c for c in lc.CARDS if c.key == key and c.group == group)


def hook_rows():
    return [(c, e) for c in lc.CARDS for e in c.evidence if e.part_class in HOOK_CLASSES]


def test_the_hook_rows_name_run_ids_of_the_known_series():
    rows = hook_rows()
    assert sum("E1/" in e.runs for _, e in rows) >= 55
    assert sum("E2/" in e.runs for _, e in rows) >= 50
    for c, e in rows:
        ids = [i.strip() for i in e.runs.split(",")]
        assert ids and all(re.fullmatch(r"(E1|E2|P7-1/l3|P7-1/r3)/[A-Za-z0-9_.-]+", i) for i in ids), (c.key, e.runs)
    assert set(HOOK_CLASSES) <= set(lc.PART_CLASSES)


def test_a_hook_row_is_marked_inside_the_noise_when_all_three_optimizer_values_are():
    """The bands are those of the E1 series for the 2 mm series: stiffness +-0.3 %, mass +-0.1 % (the fine grid has its own base)."""
    marked = 0
    for c, e in hook_rows():
        m = OPTIMIZER_PRICE.search(e.price)
        if e.part_class != "hook" or not m:
            continue
        hang, pull, mass = map(float, m.groups())
        inside = abs(hang - 1) <= 0.003 and abs(pull - 1) <= 0.003 and abs(mass - 1) <= 0.001
        assert inside == ("inside the noise" in e.price), (c.key, e.runs, e.price)
        marked += inside
    assert marked >= 5


# The bands of the E2 series: the range of the six base runs of the stress case, against their mean. The stress case does not repeat,
# so its band is a range and not a tolerance. The coarse buckling copy repeats byte for byte and has the bands of the hook family.
STRESS_BAND = {"hang": (0.989, 1.013), "pull_off": (0.986, 1.014), "mass": (0.983, 1.009)}


def inside_the_stress_band(hang, pull, mass):
    return (STRESS_BAND["hang"][0] <= hang <= STRESS_BAND["hang"][1] and STRESS_BAND["pull_off"][0] <= pull <= STRESS_BAND["pull_off"][1]
            and STRESS_BAND["mass"][0] <= mass <= STRESS_BAND["mass"][1])


def test_a_stress_row_is_marked_inside_the_noise_exactly_when_it_is_inside_the_range_of_the_six_base_runs():
    marked = unmarked = 0
    for c in lc.CARDS:
        for e in c.evidence:
            if e.part_class != "hook_stress":
                continue
            m = OPTIMIZER_PRICE.search(e.price)
            assert m, (c.key, e.runs, "a stress row gives the optimizer ratios in the form of the other hook rows")
            inside = inside_the_stress_band(*map(float, m.groups()))
            assert inside == ("inside the noise" in e.price), (c.key, e.runs, e.price)
            marked += inside
            unmarked += not inside
    assert marked >= 1 and unmarked >= 15


def test_a_buckling_row_is_marked_inside_the_noise_with_the_bands_of_the_hook_family():
    marked = 0
    for c in lc.CARDS:
        for e in c.evidence:
            m = OPTIMIZER_PRICE.search(e.price)
            if e.part_class != "hook_buckling-2mm" or not m:
                continue
            hang, pull, mass = map(float, m.groups())
            inside = abs(hang - 1) <= 0.003 and abs(pull - 1) <= 0.003 and abs(mass - 1) <= 0.001
            assert inside == ("inside the noise" in e.price), (c.key, e.runs, e.price)
            marked += inside
    assert marked >= 1


def test_the_part_classes_of_the_stress_case_and_the_buckling_copy_give_the_base_and_the_noise():
    stress, buckling = lc.PART_CLASSES["hook_stress"], lc.PART_CLASSES["hook_buckling-2mm"]
    for words in ("33.09 g", "874.3", "1079.7", "770.5", "813.3", "do NOT repeat", "x0.983 to x1.009", "x0.989 to x1.013", "x0.986 to x1.014", "inside the noise"):
        assert words in stress, words
    for words in ("16.55 s", "1401.87", "123.085", "byte for byte", "E2/BK0 and E2/BK1"):
        assert words in buckling, words
    assert not HABIT.search(stress + buckling)


STRESS_KEYS = ("load_cases[].stress_limit", "settings.stress_q", "settings.stress_power", "settings.stress_scale", "settings.stress_step_acceptance", "settings.move_limit",
               "settings.filter_solid_neighbours", "settings.volume_fraction", "settings.iterations")


@pytest.mark.parametrize("key", STRESS_KEYS)
def test_a_card_with_stress_rows_says_the_noise_of_the_stress_case_and_is_measured(key):
    c = card_of(key)
    rows = [e for e in c.evidence if e.part_class == "hook_stress"]
    assert c.confidence == "measured" and rows, key
    assert all(e.runs.startswith("E2/") for e in rows), key
    assert lc.STRESS_NOISE in c.price_note, f"{key}: the price note says that a stress run does not repeat"
    assert "does not repeat" in lc.STRESS_NOISE and "inside the noise" in lc.STRESS_NOISE
    assert "- Reading the price: " in levers.reference(key)


def test_stress_scale_is_a_card_of_kind_both_that_says_which_kind_was_chosen_and_why():
    c = card_of("settings.stress_scale")
    assert c.kind == "both" and c.kind_note
    for words in ("moves the limit", "1 or more tightens", "below 1 loosens", "consent", "origin"):
        assert words in c.kind_note, words
    assert any("stress_limit_MPa" in f for s in c.symptoms for f in s.facts)
    # the P2 stress stayed 2 to 3 times the limit while the row said met: the stress_limit card says that "met" is the relaxed measure
    limit = card_of("load_cases[].stress_limit")
    assert any("though the run says met" in s.words and "verify.cases.<case>.von_mises.p99_9_by_volume_MPa" in s.facts for s in limit.symptoms)
    assert "'Met' is about the relaxed measure" in limit.price_note and "2.45 to 2.67 times the limit" in " ".join(s.when for s in limit.symptoms)


def test_the_buckling_row_card_gives_the_time_the_p2_price_and_the_jitter():
    c = card_of("buckling.min_factor")
    assert c.confidence == "measured" and c.kind == "requirement"
    assert {e.part_class for e in c.evidence} == {"hook_buckling-2mm"}
    assert {i.strip() for e in c.evidence for i in e.runs.split(",")} == {"E2/BK0", "E2/BK1", "E2/BK2", "E2/BKfL", "E2/BKfH"}
    for words in ("10.4 to 13.8 times", "100: 11.4 times", "12 modes took 498 s", "1.5 times `min_factor`", "section 39"):
        assert words in c.cost, words
    for words in ("x0.917", "x1.033", "not isolated", "jittery", "100 against 123.1", "+8.2 %"):
        assert words in c.price_note, words
    assert card_of("buckling.ks_q").confidence == "measured" and card_of("buckling.ks_q").evidence
    assert any("12 modes" in e.effect or "490.4" in e.effect for e in card_of("buckling.modes").evidence)


E2_HOOK_KEYS = {
    "settings.eta_eroded": {"E2/RE60", "E2/RE75"},
    "settings.perimeter_max_mm2": {"E2/PM30", "E2/PM14", "E2/PM11"},
    "mirror": {"E2/MI2", "E2/MI2c", "E2/MI2s", "E2/MI2cs", "E2/MIR"},
    "mirror[].align": {"E2/MI2", "E2/MI2c", "E2/MIR"},
    "regions.solid": {"E2/SOL1", "E2/SOL1v"},
    "limits.displacement[].limit_mm": {"E2/LDa", "E2/LDb", "E2/LDc"},
}


@pytest.mark.parametrize("key", sorted(E2_HOOK_KEYS))
def test_a_key_of_the_e2_hook_family_is_measured_with_its_runs_and_a_price_note(key):
    c = card_of(key)
    runs = {i.strip() for e in c.evidence for i in e.runs.split(",")}
    assert c.confidence == "measured" and E2_HOOK_KEYS[key] <= runs, (key, runs)
    assert c.price_note and "- Reading the price: " in levers.reference(key)


def test_every_e2_row_with_an_optimizer_ratio_gives_the_p2_ratio_too_unless_the_run_has_no_p2():
    checked = 0
    for c in lc.CARDS:
        for e in c.evidence:
            if e.runs.startswith("E2/") and "optimizer hang" in e.price:
                checked += 1
                assert "P2 hang" in e.price or "verify" in e.effect, (c.key, e.runs)
    assert checked >= 45


def test_the_mirror_cards_say_the_price_is_the_load_asymmetry_and_the_align_card_gives_the_orphans():
    mirror, align = card_of("mirror"), card_of("mirror[].align")
    assert "asymmetry of the load" in mirror.price_note and "about 1 %" in mirror.price_note and "the grids differ" in mirror.price_note
    assert any("E2/MIR" == e.runs and e.part_class == "hook fine" for e in mirror.evidence)
    for words in ("570 orphan", "3.36 mm", "0.462 mm"):
        assert words in align.price_note or any(words in e.effect for e in align.evidence), words
    assert any("check_problem.mirror.planes[].orphans" in s.facts and "run.mirror.end.rho_deviation" in s.facts for s in align.symptoms)


def test_the_solid_and_displacement_cards_say_what_the_report_found():
    solid, limit = card_of("regions.solid"), card_of("limits.displacement[].limit_mm")
    assert "design->load" in solid.values and "22889.6 to 23299.2" in solid.values and "0.19325" in solid.values
    assert "0.0207" in limit.price_note and "no line" in limit.price_note and "NOT met" in limit.price_note


def test_the_displacement_list_card_points_to_the_limit_card_that_holds_the_rows_and_the_price():
    """Fix round 1: the rows and the price note are on one card (`limit_mm`); the list card keeps one pointer and no duplicate."""
    lst, limit = card_of("limits.displacement"), card_of("limits.displacement[].limit_mm")
    assert lst.confidence == "code" and not lst.evidence and not lst.price_note
    assert "`limits.displacement[].limit_mm`" in lst.values
    assert limit.confidence == "measured" and {i.strip() for e in limit.evidence for i in e.runs.split(",")} == {"E2/LDa", "E2/LDb", "E2/LDc"}


def test_the_limit_card_has_a_symptom_for_a_displacement_row_that_is_not_met():
    limit = card_of("limits.displacement[].limit_mm")
    sym = [s for s in limit.symptoms if "metrics.rows[].met" in s.facts]
    assert len(sym) == 1 and "displacement" in sym[0].detect and "E2/LDc" in sym[0].when
    assert {"metrics.rows[].value_end", "metrics.rows[].limit", "metrics.rows[].g_end"} <= set(sym[0].facts)


def test_the_wording_of_audit_p7_5_is_the_wording_of_the_sources():
    """The six facts of AUDIT-P7-5.md (M1 to M6) and the notes N1, N2, N3 and N6, each against the report line or the run file that the audit names."""
    q = card_of("settings.stress_q")
    assert "104 rejected tries against 34 to 43" in q.values and "3 times" not in q.values                      # M2: report l.128
    assert "all on the better side" in q.price_note and "no measure of the table is worse than the base range" in q.price_note   # M1: report l.127, l.185
    assert "except the lower mass" not in q.price_note
    sa = card_of("settings.stress_step_acceptance")
    assert "0.1 % above the range" in sa.evidence[0].price and "1096.467" in sa.evidence[0].price and "1095.152" in sa.evidence[0].price   # M3: report l.177
    assert "0.2 %" not in sa.evidence[0].price + sa.price_note
    assert "in the review of the stress rows" in sa.price_note and "CONTRACTS.md section 17" not in sa.price_note and "5 MPa" in sa.price_note   # N6
    pc = card_of("print_check", "tool")
    assert any("2.13 s of the first and 2.77 s of the second" in e.price for e in pc.evidence) and "2.13 to 2.77 s" in pc.cost   # M4: pc1.json, pc2.json
    assert "2.1 to 2.2 s" not in pc.cost + " ".join(e.price for e in pc.evidence)
    cmp_ = card_of("compare_runs", "tool")
    nolab = [e for e in cmp_.evidence if e.runs == "E2/T-compare" and "check_labels` false" in e.change][0]
    assert "11 summary lines" in nolab.effect and "labels_vs_disk" in nolab.effect and "without the label line" not in nolab.effect   # M5: cmp_re60_nolab.json
    assert "labels_vs_disk.checked" in cmp_.values
    ss = card_of("settings.stress_scale")
    assert "agree in sign and differ in size" in ss.price_note and "x0.464 (optimizer) and x0.567 (P2)" in ss.price_note and "x1.899 and x1.637" in ss.price_note   # M6: report l.132-134
    assert "sign and size" not in ss.price_note
    mf = card_of("buckling.min_factor")
    assert "only when the row binds" not in mf.price_note and "slack at the last pass (g -0.0713)" in mf.price_note   # N1
    assert "second candidate" not in card_of("ratchet_run", "tool").values                                       # N2: not run
    assert "no effect found at this size" in card_of("settings.filter_solid_neighbours").price_note             # N3: report l.324


def test_the_eta_card_gives_the_gap_the_warning_and_the_refusal():
    c = card_of("settings.eta_eroded")
    for words in ("2.0, 5.1, 9.8 and 27.0 %", "x0.979, x0.967, x0.958, x0.941", "22b", "grid scale", "0.99 voxels"):
        assert words in c.price_note + c.values, words


TOOL_ROWS = {
    "print_check": {"E2/T-print"},
    "cleanup_list": {"E2/T-cleanup", "E2/T-cleanup-nov"},
    "apply_marks": {"E2/T-marks", "E2/T-marks-smooth", "E2/T-marks-add"},
    "compare_runs": {"E2/T-compare"},
    "ratchet_run": {"E2/T-ratchet", "E2/T-ratchet2", "E2/T-ratchet-M", "E2/T-ratchet-S"},
    "verify_run": {"E2/T-verifyb"},
    "buckling": {"E2/T-bk-def", "E2/T-bk-m2", "E2/T-bk-m12", "E2/T-bk-gray", "E2/T-bk-ms", "E2/T-bk-gb", "E2/T-bk-H0"},
}


@pytest.mark.parametrize("key", sorted(TOOL_ROWS))
def test_a_local_tool_has_the_measured_rows_with_seconds_and_a_card_that_is_measured(key):
    c = card_of(key, "tool")
    runs = {i.strip() for e in c.evidence for i in e.runs.split(",")}
    assert c.confidence == "measured" and TOOL_ROWS[key] <= runs, (key, runs)
    rows = [e for e in c.evidence if e.runs.startswith("E2/")]
    assert rows and all(re.search(r"\d s\b", e.price) for e in rows), key
    assert not c.also, "a tool card has no also"


def test_the_costs_of_the_tools_hold_the_measured_range_and_the_host_load():
    assert "3.2 to 3.9 s" in card_of("print_check", "tool").cost and "host load" in card_of("print_check", "tool").cost
    assert "0.63 s" in card_of("cleanup_list", "tool").cost
    assert "2.4 to 3.1 s" in card_of("apply_marks", "tool").cost
    assert "0.65 to 0.67 s" in card_of("compare_runs", "tool").cost and "0.12 s" in card_of("compare_runs", "tool").cost
    ratchet = card_of("ratchet_run", "tool")
    assert "15.5 s" in ratchet.cost and "23.3 s" in ratchet.cost and "3.7 s and 4.3 s" in ratchet.cost
    assert "47.3 s and 1.8 GB" in card_of("buckling", "tool").cost
    assert "not measured" not in " ".join(card_of(k, "tool").cost for k in TOOL_ROWS)


def test_compare_runs_says_what_it_does_not_show():
    values = card_of("compare_runs", "tool").values
    for words in ("settings.move_limit", "`setup.yaml_diff` only", "settings.eta_eroded", "labels_diff.comparable", "'design->load'"):
        assert words in values, words


def test_the_verify_row_of_the_buckling_check_gives_the_tet_factors_and_the_ratio_to_the_voxel_values():
    rows = [e for e in card_of("verify_run", "tool").evidence if e.runs == "E2/T-verifyb"]
    assert rows and "1250.2" in rows[0].effect and "109.294" in rows[0].effect and "0.892 and 0.888" in rows[0].effect and "6.6 s" in rows[0].price


@pytest.mark.parametrize("key", P2_KEYS)
def test_a_key_that_the_optimizer_price_misleads_gives_the_p2_ratio_and_says_which_to_read(key):
    c = card_of(key)
    assert re.search(r"read P2", c.price_note, re.I) and "optimizer" in c.price_note, key
    both = [e for e in c.evidence if e.part_class in HOOK_CLASSES and "optimizer hang" in e.price and "P2 hang" in e.price]
    assert both, f"{key}: a row with the optimizer ratio and the P2 ratio side by side"
    text = levers.reference(key)
    assert "- Reading the price: " in text and c.price_note in text


def test_the_penal_rows_of_the_phone_stand_are_labelled_as_optimizer_values():
    c = card_of("settings.penal")
    stand = [e for e in c.evidence if e.part_class == "phone stand"]
    assert stand and all("optimizer" in e.price for e in stand)
    assert "phone stand are optimizer values" in c.price_note


def test_the_filter_symptoms_hold_no_fact_of_the_thinnest_member_estimate_and_the_detect_text_says_why():
    for key in ("settings.min_member_mm", "settings.filter_radius"):
        for s in card_of(key).symptoms:
            assert not any("thinnest_member" in f for f in s.facts), (key, s.words, s.facts)
    detect = " ".join(s.detect for s in card_of("settings.min_member_mm").symptoms)
    assert "thinnest_member" in detect and "did not follow the filter" in detect and "so it does not detect a filter effect" in detect


def test_the_keep_out_finding_is_on_the_keep_out_and_the_volume_fraction_cards():
    keep, vf = levers.reference("regions.keep_out"), levers.reference("settings.volume_fraction")
    for text in (keep, vf):
        assert "volume_fraction" in text and "E1/K1v" in text and "E1/K3v" in text, text[:200]
    assert "(target_volume_mm3 - forced_solid_new_mm3) / free_design_new_mm3" in keep
    assert "derived.volume.target_volume_mm3" in keep and "derived.volume.forced_solid_mm3" in keep and "free_design_mm3" in keep
    assert "x0.800 / x0.902" in keep
    symptoms = {s.words: s for s in card_of("regions.keep_out").symptoms}
    lost = next(s for w, s in symptoms.items() if "lost mass" in w)
    assert "metrics.mass_saved_pct" in lost.detect and "does not show the loss" in lost.detect
    assert "check_problem.derived.volume.target_mass_g" in lost.facts and "check_problem.derived.volume.free_design_mm3" in lost.facts


def test_the_target_mass_card_is_a_requirement_of_layer_1_and_gives_the_refusals_the_facts_and_no_tolerance():
    c = card_of("settings.target_mass_g")
    assert c.kind == "requirement" and not c.kind_note and c.layer == 1 and c.confidence == "code" and not c.evidence
    for part in ("exactly one of the two keys", "Both keys, or neither, is refused", "not above the mass of the forced-solid voxels", "above the mass of all active voxels solid",
                 "`settings.objective: mass`", "`check_problem.derived.volume.target_mass_asked_g`", "`derived.volume_fraction_origin`", "`metrics.mass_final_g`",
                 "the engine sets no tolerance", "a new density gives a new fraction", "`metrics.mass_asked_g`", "final mesh 27.31 g against 30 g asked"):
        assert part in (c.effect + " " + c.values), part
    assert not HABIT.search(c.effect + c.values)
    facts = {f for s in c.symptoms for f in s.facts}
    assert facts == {"check_problem.derived.volume.target_mass_asked_g", "check_problem.derived.volume.target_mass_g", "run.derived.volume.target_mass_asked_g",
                     "metrics.mass_asked_g", "metrics.mass_final_g"}
    assert any("grams" in s.words and "g," in s.words for s in c.symptoms)
    # the two cards that the key replaces or changes say so
    assert "exactly one of `volume_fraction` and `target_mass_g`" in card_of("settings").values and "`voxel_size`, `volume_fraction`, `iterations`" not in card_of("settings").values
    assert "`settings.target_mass_g`" in card_of("settings.volume_fraction").values
    # review n4: a mass in grams goes to the new key, and the kind note of the fraction says so
    note = card_of("settings.volume_fraction").kind_note
    assert "a volume or share target" in note and "a mass in grams: `settings.target_mass_g`" in note and "method when you chose it" in note
    # the key is a key of the passes: a resume of a finished run names it, and the table decides it
    note = pass_key_note(["settings.target_mass_g: 25.0 -> 20.0"], True)
    assert note and note["keys"] == ["settings.target_mass_g"] and "settings.target_mass_g (25.0 -> 20.0)" in note["fact"]
    # the symptom words find the card
    assert "settings.target_mass_g" in [x.key for x in levers.find("grams")]


def test_the_cards_of_the_load_path_of_the_mesh_and_of_the_threshold_stability_name_the_facts_of_every_run():
    """I6: `metrics.load_paths` (the final mesh joins a load region to a support) and `metrics.threshold_stability` (the thresholded part at 0.4, 0.5 and 0.6) are in every run."""
    stl = "metrics.load_paths.regions.<region>.final_stl"
    for key in ("regions.fixed", "settings.mesh_method"):
        s = next(s for s in card_of(key).symptoms if f"{stl}.connected_to_fixed" in s.facts)
        assert "floating or not attached in the STL" in s.words and set(s.facts) == {f"{stl}.connected_to_fixed", f"{stl}.touches_a_body", f"{stl}.bodies_with_the_region"}, key
        assert "false" in s.when and "connected_to_fixed" in s.when and "final.stl" in s.detect and "metrics.load_paths.final_stl.bodies" in s.detect
    for key in ("settings.beta_schedule", "settings.iterations"):
        s = next(s for s in card_of(key).symptoms if "metrics.threshold_stability.levels[].volume_mm3" in s.facts)
        assert set(s.facts) == {"metrics.threshold_stability.levels[].volume_mm3", "metrics.threshold_stability.levels[].bodies"}, key
        assert "0.4, 0.5 and 0.6" in s.detect and "sets no limit" in s.detect and not s.when and not HABIT.search(s.detect), key
    others = {c.key for c in yaml_cards() for s in c.symptoms if any(f.startswith(("metrics.load_paths", "metrics.threshold_stability")) for f in s.facts)}
    assert others == {"regions.fixed", "settings.mesh_method", "settings.beta_schedule", "settings.iterations"}
    assert {x.key for x in levers.find("a gap in the part")} == {"regions.fixed", "settings.mesh_method"}
    assert {x.key for x in levers.find("thin members that come and go at another threshold")} == {"settings.beta_schedule", "settings.iterations"}


def test_iterations_and_beta_schedule_hold_the_unscaled_and_the_scaled_rows():
    for key in ("settings.iterations", "settings.beta_schedule"):
        runs = {e.runs for e in card_of(key).evidence if e.part_class == "hook"}
        assert {"E1/IT20", "E1/IT20s", "E1/IT80", "E1/IT80s"} <= runs, key
    c = card_of("settings.iterations")
    unscaled = next(e for e in c.evidence if e.runs == "E1/IT20")
    scaled = next(e for e in c.evidence if e.runs == "E1/IT20s")
    assert "x0.791" in unscaled.price and "x0.989" in scaled.price and "gray 0.142" in unscaled.effect


def test_the_reference_holds_the_cost_of_a_try_by_layer_and_the_text_on_how_to_read_a_price():
    text = levers.reference()
    for n in (1, 2, 3, 4):
        assert re.search(rf"^- Layer {n}: .* On the hook: ", text, re.M), n
    assert "## How to read a price" in text and lc.PRICE_TEXT in text
    for words in ("`P2` is `topopt verify <run> --no-sensitivity`", "inside the noise", "gray"):
        assert words in lc.PRICE_TEXT, words
    assert set(lc.LAYER_COSTS) == set(levers.LAYER_TEXT)
    assert not HABIT.search(lc.PRICE_TEXT + " ".join(lc.LAYER_COSTS.values()))


def test_a_price_note_is_empty_or_a_text_with_a_fact_and_no_habitual_cap():
    noted = [c for c in lc.CARDS if c.price_note]
    assert len(noted) >= 10
    for c in noted:
        assert c.confidence == "measured" and any(e.part_class in HOOK_CLASSES for e in c.evidence), c.key
        assert not HABIT.search(c.price_note), c.key


# --- the flags of the engine in the symptoms, and the field `when` (round 2, addendum) ------------------------------------

FLAG_FACTS = {
    "metrics.filter.facts": ("settings.min_member_mm", "settings.filter_radius"),
    "metrics.stress.cases.<case>.relaxed.met": ("load_cases[].stress_limit",),
    "metrics.mesh.fallback": ("settings.mesh_method",),
    "metrics.mesh.extra_bodies_removed": ("settings.mesh_method",),
    "metrics.interfaces.error": ("interfaces",),
    "metrics.verdict.notes": ("settings.stop_rules",),
    "verify.bore_facts[].status": ("bore_checks.bores[].point",),
    "metrics.load_paths.regions.<region>.final_stl.connected_to_fixed": ("regions.fixed", "settings.mesh_method"),   # I6
    "run.derived.load_cases[].applies_load": ("load_cases[].forces[].force",),      # U21
    "run.derived.load_cases[].load_reaches_free_dofs": ("load_cases[].forces[].force",),      # F4-37
}


def test_the_flags_of_the_engine_are_facts_of_the_matching_cards_and_say_when_they_mean_the_symptom():
    for path, keys in FLAG_FACTS.items():
        for key in keys:
            hits = [s for s in card_of(key).symptoms if path in s.facts]
            assert len(hits) == 1, (key, path)
            assert hits[0].when.strip() and path.rpartition(".")[2].replace("[]", "") in hits[0].when, (key, path, hits[0].when)
    # `min_feature_note` stays next to `facts` (it is written in every run)
    for key in FLAG_FACTS["metrics.filter.facts"]:
        s = next(s for s in card_of(key).symptoms if "metrics.filter.facts" in s.facts)
        assert "metrics.filter.min_feature_note" in s.facts


def test_the_force_card_names_the_load_case_that_applies_no_load_and_its_fact():
    """U21: a user who says "zero force" or "the load case does nothing" finds the card; the fact is `applies_load`, in `check_problem` and in `run.json`."""
    c = card_of("load_cases[].forces[].force")
    [s] = [s for s in c.symptoms if "applies_load" in " ".join(s.facts)]
    assert "zero force" in s.words and "does nothing" in s.words
    assert s.facts == ("check_problem.derived.load_cases[].applies_load", "run.derived.load_cases[].applies_load",
                       "check_problem.derived.load_cases[].load_reaches_free_dofs", "run.derived.load_cases[].load_reaches_free_dofs")   # F4-37: the second reason
    assert "applies_load" in s.when and "exactly 0" in s.when
    assert "load_reaches_free_dofs" in s.when and "every force lands on fixed dofs" in s.when and "load_reaches_free_dofs" in c.values
    # review m1: the two causes have their own words (the cancel words do not say "fixed"), review m4: the run-time refusal is "next to the same rows"
    assert "nodal forces add up to 0.0 at every node" in s.when and "`load_reach`" in s.when and "nodal_zero" in s.when and "fixed_net" in s.when
    assert "opposite forces on the same nodes cancel" in c.values and "`load_reach` names it" in c.values
    assert "0.0 or subnormal in double precision (1e-300 N) stops the run at the first pass where that holds, next to the same rows" in c.values
    assert "fact: load case" in s.detect and "start of `run.log`" in s.detect
    assert "A moment alone is a load" in c.values and "applies no load" in c.values
    assert not HABIT.search(s.words + s.detect + s.when)
    assert [x.key for x in levers.find("zero force")] == ["load_cases[].forces[].force"]
    assert "load_cases[].forces[].force" in {x.key for x in levers.find("check_problem.derived.load_cases[].applies_load")}


def test_the_filter_radius_card_says_that_a_radius_alone_gives_no_filter_line():
    """E1/FR20 and E1/FR40 (a `filter_radius` alone) have an empty `metrics.filter.facts`; `members.member_facts` gives a line only when `min_member_mm` is set."""
    import types
    from topopt import members
    alone = types.SimpleNamespace(filter_radius=2.0, filter_radius_source="settings", min_member_mm=None, robust="off")
    derived = types.SimpleNamespace(filter_radius=1.877, filter_radius_source="min_member_mm", min_member_mm=1.0, robust="off")
    assert members.member_facts(alone, 2.0) == [] and len(members.member_facts(derived, 2.0)) >= 1
    c = card_of("settings.filter_radius")
    sym = c.symptoms[0]
    assert "metrics.filter.size.radius_voxels" in sym.facts and "only when `min_member_mm` sets the radius" in c.values
    assert "radius_voxels" in sym.when and "empty for a `filter_radius` alone" in sym.when


def test_a_when_text_renders_after_the_detect_text_and_needs_facts():
    text = levers.reference("settings.mesh_method")
    assert re.search(r"fallback[^\n]*\. Present when: `metrics.mesh.fallback` is not null", text), text
    assert ".." not in text
    s = lc.Symptom("a", "b", when="x")
    bad = dataclasses.replace(card_of("settings.penal"), symptoms=(s,))
    assert any("`when` text with no facts" in p for p in levers.problems([bad]))
    good = dataclasses.replace(card_of("settings.penal"), symptoms=(dataclasses.replace(s, facts=("metrics.x",)),))
    assert not any("`when`" in p for p in levers.problems([good]))
    assert sum(bool(s.when) for c in lc.CARDS for s in c.symptoms) >= 8       # the symptoms with a flag; the others are read from `detect`


# --- fix round 1 (review REVIEW-P7-1-3.md) --------------------------------------------------------------------------------

def test_a_text_key_that_the_passes_do_not_read_is_a_layer_3_card_and_the_resume_fact_does_not_name_it():
    """F2: `allowable_source` is read by the post step and the ratchet only (`stress_report.py`, `ratchet.py`); the passes do not use it (`optimize.py`)."""
    card = card_of("load_cases[].allowable_source")
    assert card.layer == 3 and card.cost == lc.RESUME_COST and card.call == lc.RESUME_CALL
    assert pass_key_note(["load_cases[0].allowable_source: 'a' -> 'b'"], True) is None
    assert pass_key_note(["load_cases[0].allowable_source: <not set> -> 'a'", "settings.penal: 3.0 -> 3.5"], True)["keys"] == ["settings.penal"]
    layer3_list = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8").partition("A key has layer 3 only when")[2].partition("A key that the passes read")[0]
    assert "`load_cases[].allowable_source`" in layer3_list


def test_the_keys_that_the_resume_fact_cannot_see_say_so_on_their_cards():
    """F3 and F4: the soft keys are notes on a resume, so the fact never sees them; the buckling keys are named by the layer only; a labels-only change is not in the reply."""
    from topopt.checkpoint import SOFT_KEYS, diff_inputs
    soft = [c for c in yaml_cards() if any(c.key == k or c.key.startswith(k + ".") for k in SOFT_KEYS)]
    assert {c.key for c in soft} >= {"settings.fine_pass", "settings.fine_pass.objective_change", "settings.stop_rules", "settings.stop_rules.enabled", "settings.stop_rules.objective_change", "settings.stop_rules.window",
                                     "settings.stop_rules.grayness_max", "settings.stop_rules.time_budget_s", "settings.max_active_voxels", "settings.max_grid_voxels"}
    for c in soft:
        assert "the resume fact" in c.values and "does not name" in c.values, c.key
    old = {"yaml_parsed": {"settings": {"stop_rules": {"window": 5}, "max_active_voxels": 100, "max_grid_voxels": 1000}}}
    new = {"yaml_parsed": {"settings": {"stop_rules": {"window": 9, "enabled": True}, "max_active_voxels": 200, "max_grid_voxels": 2000}}}
    changes, notes = diff_inputs(old, new)
    assert changes == [] and len(notes) == 4, (changes, notes)         # notes, not changes: the resume fact never sees them
    assert pass_key_note(changes, True) is None
    for key in ("buckling.modes", "buckling.max_gb"):
        assert "reads the layer of the card only" in card_of(key).values and "does not name" in card_of(key).values, key
    stock = card_of("interfaces[].stock_mm").values
    assert "`resume` of the MCP tool `run_topopt` compares the YAML keys and the region files only" in stock and "the run log line names it" in stock
    contracts = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8").partition("### 41.8")[2].partition("### 41.9")[0]
    for part in ("- Limits.", "`buckling.modes`", "`checkpoint.SOFT_KEYS`", "compares the YAML keys and the region files only", "`interfaces[].stock_mm`"):
        assert part in contracts, part


def test_the_verify_run_card_names_every_refusal_of_the_check():
    """F5: the card lists what `_check_loads_and_supports` refuses; the words come from the code."""
    src = (ROOT / "topopt" / "verify" / "run.py").read_text(encoding="utf-8")
    values = card_of("verify_run", "tool").values
    for what in ("a moment", "a remote force", "a pressure", "a per-axis support"):
        assert what in src and what in values, what
    role = (ROOT / "topopt_mcp" / "roles" / "lead-design.md").read_text(encoding="utf-8")
    assert "a pressure and a per-axis support. It takes springs and joints" in role


def test_the_query_not_converged_gives_the_cards_that_relabel_and_the_card_that_changes_the_part():
    """F6: the cheapest card for `not converged` changes the verdict and leaves the part the same: the card says so, and the card that changes the part is found too."""
    keys = [c.key for c in levers.find("not converged")]
    assert "settings.stop_rules.objective_change" in keys and "settings.iterations" in keys, keys
    assert "changes the verdict and leaves the part the same" in card_of("settings.stop_rules.objective_change").values


def test_the_rule_words_are_the_same_in_the_run_topopt_description_and_the_readme():
    """F9: the words of `RULE_TEXT` (the role has no room: its body is at the limit of 6,000 characters)."""
    run_text = " ".join(mcp_tools()["run_topopt"].description.split())
    readme = " ".join((ROOT / "docs" / "reference" / "problem-and-run.md").read_text(encoding="utf-8").split())
    for text in (run_text, readme):
        for part in ("stress limits and safety factors", "without asking", "measured price"):
            assert part in text, part
        assert re.search(r"brings? the facts", text), text[:200]
    assert "beta schedule, move limit, solver and tolerances, voxel size, iterations, robust, mesh method, smoothing" in readme


def test_the_two_voxel_guards_are_method_cards_that_give_the_memory_facts_and_the_risk():
    """The agent decides a guard with the facts and reports what, why and the price. The card gives the facts and the risk."""
    guards = [c for c in lc.CARDS if c.key in ("settings.max_active_voxels", "settings.max_grid_voxels") and c.group == "yaml"]
    assert len(guards) == 2
    for c in guards:
        assert c.kind == "method" and not c.kind_note, c.key
        for part in ("You decide with the facts", "`check_problem.memory_estimate`", "`ram_bytes`", "`environment`", "a run above the RAM can freeze the machine",
                     "a 0.5 mm whole-domain grid did on a 16 GB Mac", "a larger `voxel_size`", "report what you took, why and the price"):
            assert part in c.values, (c.key, part)
        assert not HABIT.search(c.values) and not re.search(r"(?:^|[.;:]\s+)do not\b", c.values, re.I) and "ask the user" not in c.values.lower(), c.key
    assert "settings.allow_large_grid" in next(c for c in guards if c.key == "settings.max_grid_voxels").values       # the guard of build_regions is named
    assert next(c for c in lc.CARDS if c.key == "settings.voxel_size").kind == "method"                   # the origin in `record_brief` decides, not the card
    counts = {k: sum(c.kind == k for c in lc.CARDS) for k in lc.KINDS}
    assert counts == {"requirement": 82, "method": 72, "both": 11}, counts                                 # CONTRACTS.md 41.2 says the same numbers
    contracts = " ".join((ROOT / "CONTRACTS.md").read_text(encoding="utf-8").split())
    assert "72 `method`, 82 `requirement` and 11 `both` cards" in contracts
    assert f"it holds {len(lc.CARDS)} cards: 138 for YAML keys" in contracts                           # 41.1 says the count
    text = levers.reference()
    assert not re.search(r"allow_large_grid\s*:|max_(?:active|grid)_voxels\s*:", text), "a card text does not show an override as YAML"


def test_the_guard_rule_of_main_101_is_replaced_by_the_method_rule_and_the_facts_and_the_rest_of_101_stays():
    """A voxel guard is a method setting. The agent reads the memory estimate against the RAM, decides, and reports what, why and
    the price. Kept from #101: ask for a missing load, "Go on (a cheap part) or wait ...", a sub-agent returns its questions, the voxel size that the user gave."""
    tools = mcp_tools()

    def read(*parts):
        return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")

    old_rule = re.compile(r"each guard override|guard override[^.]{0,60}after the user agrees|(apply|put) (it|the override)[^.]{0,40}only after the user agrees|"
                          r"Do not put an override in the spec yourself|Ask first \(see above\)|Options for the user", re.I)
    guard_texts = {
        "INSTRUCTIONS": server.INSTRUCTIONS, "run_topopt": tools["run_topopt"].description, "build_regions": tools["build_regions"].description,
        "lead-design": read("topopt_mcp", "roles", "lead-design.md"), "fea": read("topopt_mcp", "roles", "fea.md"), "cad": read("topopt_mcp", "roles", "cad.md"),
        "agent lead-design": read("agents", "lead-design-engineer.md"), "agent fea": read("agents", "fea-engineer.md"), "agent cad": read("agents", "cad-engineer.md"),
        "start-here": read("skills", "start-here", "SKILL.md"), "part-to-topopt": read(".claude", "skills", "part-to-topopt", "SKILL.md"),
    }
    guard_texts = {k: " ".join(v.split()) for k, v in guard_texts.items()}                  # one line: a description breaks lines
    for name, text in guard_texts.items():
        assert not old_rule.search(text), (name, old_rule.search(text).group(0))
        assert "RAM" in text and re.search(r"memory[ _]estimate", text), name                  # the facts that the agent reads
    for name in ("INSTRUCTIONS", "run_topopt", "build_regions", "part-to-topopt", "start-here", "cad", "fea"):
        assert re.search(r"(freeze|crash)", guard_texts[name]), name                           # the risk
    assert "what, why and the price" in guard_texts["start-here"] and "what, why and the price" in guard_texts["part-to-topopt"]
    assert "voxel guards" in guard_texts["lead-design"] and "the memory estimate against the RAM" in guard_texts["run_topopt"]
    # kept from #101
    assert "ask first" in guard_texts["run_topopt"] and "ask before the run" in guard_texts["build_regions"] and 'also when the user says "run it"' in guard_texts["INSTRUCTIONS"]
    for name in ("lead-design", "agent lead-design"):
        assert "Go on (a cheap part) or wait for an answer (a costly part, a missing load)" in guard_texts[name]
        assert "The user owns the goal, the loads, the limits, the material and the release." in guard_texts[name] and "Return your questions to the caller." in guard_texts[name]
    assert "Do not start a run on a load that you chose." in guard_texts["part-to-topopt"] and "ask for it before the run" in guard_texts["start-here"]
    assert "its origin in `record_brief` is `user`" in guard_texts["part-to-topopt"]              # the voxel size that the user gave is a requirement
    assert "Install or build something only after the user agrees." in guard_texts["start-here"]
    for copy, source in ((".agents/skills/start-here/SKILL.md", "skills/start-here/SKILL.md"), (".agents/skills/part-to-topopt/SKILL.md", ".claude/skills/part-to-topopt/SKILL.md")):
        assert read(*copy.split("/")) == read(*source.split("/")), copy


def test_the_beta_card_names_the_fact_that_the_tools_give_before_the_run(tmp_path, capsys):
    """P7-4 is on main: `topopt check`, `check_problem.warnings` and `run.log` say that the passes end before the last beta stage. The card names them."""
    from beta_fact_helpers import START
    prefix = START.split(" is ")[0]                                                # "fact: settings.iterations"
    c = card_of("settings.beta_schedule")
    assert "prints no line" not in c.values and "no line about it" not in c.values
    for part in (prefix, "`topopt check`", "`check_problem`", "`run.log`", "two options", "the measured price", "the run goes on"):
        assert part in c.values, part
    [s] = [s for s in c.symptoms if "check_problem.warnings" in s.facts]
    assert prefix in s.detect and prefix in s.when and "topopt check" in s.detect and "run.log" in s.detect
    assert not [x for x in c.symptoms if x is not s and "check_problem" in " ".join(x.facts)]
    # the fact of the card is the fact of the tools: a case that ends before the last stage gives the line in both places
    make_case(tmp_path, h=2.0)
    p = tmp_path / "c.yaml"
    d = yaml.safe_load(p.read_text())
    d["settings"]["iterations"], d["settings"]["beta_schedule"] = 6, [[0, 1], [10, 2], [20, 4]]
    p.write_text(yaml.safe_dump(d))
    got = server.check_problem(str(p))
    line = [w for w in got["warnings"] if w.startswith(prefix)]
    assert got["ok"] and len(line) == 1 and "options:" in line[0] and "Measured on" in line[0]
    assert main(["check", str(p)]) == 0 and line[0] in capsys.readouterr().out


def test_the_evidence_run_ids_name_runs_and_the_texts_give_reasons_in_place_of_orders():
    """F10: a run id is not a file name; a card does not tell the agent "do not ..." (the reason is given)."""
    for c in lc.CARDS:
        for e in c.evidence:
            assert not e.runs.endswith(".json"), (c.key, e.runs)
    text = levers.reference()
    assert not re.search(r"(?:^|[.;:]\s+)do not\b", text, re.I | re.M), re.search(r"(?:^|[.;:]\s+)do not\b.{0,60}", text, re.I | re.M)
    assert "do not take it as the detecting fact" not in text and "Do not call it while" not in text
    assert "so it does not detect a filter effect" in text and "writes the same key `buckling` of `result/metrics.json`" in text


# --- round 3: the cards of the keys that main added (P4b-3 verify --part-only, P4b-5 continue, P4b-6 the film key) -------------

FILM = "settings.part_only_min_film_mm"


def test_the_film_key_is_a_layer_3_card_with_the_cost_of_a_resume_and_the_resume_fact_does_not_name_it():
    c = card_of(FILM)
    assert c.layer == 3 and c.kind == "both" and c.kind_note and c.confidence == "measured"
    assert c.cost == lc.RESUME_COST and c.call == lc.RESUME_CALL
    assert FILM in config.accepted_key_paths() and FILM.partition(".")[2] in config.SETTINGS_OPTIONAL
    assert pass_key_note([f"{FILM}: <not set> -> 0.05"], True) is None and pass_key_note([f"{FILM}: 0.05 -> 0.1"], True) is None
    assert FILM in card_of("run_topopt:allow_changed_inputs", "tool").values                    # the list of the keys that the post step reads
    assert FILM in card_of("regions.not_printed").values and "verify_run" in card_of("regions.not_printed").values
    for e in c.evidence:                                                                            # recorded in CONTRACTS.md, not run again by P7
        assert e.part_class == lc.NOT_PRINTED and e.runs.startswith("CONTRACTS 9.1: "), e
    assert "CONTRACTS-recorded" in lc.PART_CLASSES[lc.NOT_PRINTED] and "did not run them again" in lc.PART_CLASSES[lc.NOT_PRINTED]


def test_the_film_key_is_read_in_the_post_step_only():
    """The layer of the card is a claim about the code. In `topopt/optimize.py` (the passes) the key is named in `problem_facts` only, which
    writes one fact line at the start; the value reaches the post step through `plan["min_film_mm"]` of `run.json`, which `part_only.cut` reads."""
    tree = ast.parse((ROOT / "topopt" / "optimize.py").read_text(encoding="utf-8"))
    homes = set()

    def walk(node, stack):
        stack = stack + [node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else stack
        if (isinstance(node, ast.Attribute) and node.attr == "part_only_min_film_mm") or (isinstance(node, ast.Constant) and node.value == "part_only_min_film_mm"):
            homes.add(stack[0] if stack else "<module>")
        for child in ast.iter_child_nodes(node):
            walk(child, stack)

    walk(tree, [])
    assert homes == {"problem_facts"}, homes
    snapshot = (ROOT / "topopt" / "snapshot.py").read_text(encoding="utf-8")
    assert 'plan["min_film_mm"] = float(cfg.settings.part_only_min_film_mm)' in snapshot
    assert 'plan.get("min_film_mm")' in (ROOT / "topopt" / "part_only.py").read_text(encoding="utf-8")


def test_the_facts_of_the_film_symptoms_are_keys_that_part_only_writes():
    c = card_of(FILM)
    facts = [f for s in c.symptoms for f in s.facts]
    assert facts and all(f.startswith("metrics.part_only.") and written_by(f) for f in facts), facts


def test_the_continue_cards_are_layer_2_and_the_cost_of_layer_2_is_a_measured_row():
    params = inspect.signature(server.run_topopt).parameters
    for key in ("run_topopt:continue_from", "run_topopt:more_iterations"):
        c = card_of(key, "tool")
        assert c.layer == 2 and c.confidence == "measured" and c.kind == "method" and key.partition(":")[2] in params, key
        assert all(e.part_class == "hook" and re.fullmatch(r"(P7-1/r3/\w+|E1/\w+)(, (P7-1/r3/\w+|E1/\w+))*", e.runs) for e in c.evidence), key
    assert "nobody measured" not in lc.LAYER_COSTS[2] and re.search(r"\d+\.\d to \d+\.\d s wall", lc.LAYER_COSTS[2])
    assert "continue_from" in levers.LAYER_TEXT[2] and "more_iterations" in levers.LAYER_TEXT[2]
    assert "On the hook: a continuation" in levers.reference()
    assert re.search(r"^- Layer 2: .*continue_from.* On the hook: a continuation", levers.reference(), re.M)
    # the card holds the one thing that a resume of a finished run cannot do: a changed key of the passes takes effect
    values = card_of("run_topopt:continue_from", "tool").values
    assert "a resume of a finished run (layer 3) does not do" in values and "allow_changed_inputs=true" in values
    keys = [c.key for c in levers.find("not converged")]
    assert "run_topopt:continue_from" in keys and "settings.iterations" in keys, keys
    assert "run_topopt:continue_from" in card_of("settings.iterations").values and "run_topopt:continue_from" in card_of("run_topopt:resume", "tool").values


def test_the_continue_rows_state_ratios_that_follow_from_their_values():
    """The price of a hook row is a ratio against B0 (hang 1294.703, pull_off 1316.773 N/mm, mass 25.72 g): the ratios of the three runs of round 3 follow from the values of the base run."""
    b0 = (1294.703, 1316.773, 25.72)
    runs = {"P7-1/r3/C10": (1295.52, 1317.553, 25.71), "P7-1/r3/C20": (1295.2, 1318.2, 25.71), "P7-1/r3/CV25": (1584.259, 1568.804, 29.46)}
    rows = {e.runs.split(",")[0]: e for k in ("run_topopt:continue_from", "run_topopt:more_iterations") for e in card_of(k, "tool").evidence}
    for name, vals in runs.items():
        m = OPTIMIZER_PRICE.search(rows[name].price)
        stated = tuple(float(x) for x in m.groups())
        assert stated == pytest.approx(tuple(v / b for v, b in zip(vals, b0)), abs=0.0011), (name, stated)


def test_the_verify_part_only_card_names_the_flag_the_parameter_and_the_rules_of_contracts():
    c = card_of("verify_run:part_only", "tool")
    assert c.layer == 4 and c.kind == "method" and "part_only" in inspect.signature(server.verify_run).parameters
    assert build_parser().parse_args(["verify", "somefolder", "--part-only"]).part_only is True
    contracts = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    for words in ("boundary_condition: none", "yaml_vs_run", "selected_area_over_contact", "part_only_contact_s"):
        assert words in contracts and words in c.values + c.cost, words
    assert "part_only" in card_of("verify_run", "tool").values
    assert all(e.part_class == lc.NOT_PRINTED and e.runs.startswith("CONTRACTS 9.1: ") for e in c.evidence)


def _gate_doc():
    """A real `gate` key (the function that writes it), for the fact paths of the cards: one stress, one buckling and one row record."""
    from topopt.verify import gate as vg
    lim = dict(stress={"a": dict(limit=30.0, source="--allowable-mpa (all cases)", replaced_limit=20.0, replaced_source="load_cases[a].stress_limit", loosened=True)},
               buckling=dict(limit=2.0, source="buckling.min_factor"), applies_load={"a": True},
               rows=[dict(name="displacement:tip:a", check="displacement", case="a", limit=0.5, source="limits.displacement[tip].limit_mm")])
    level = dict(size_mm=2.0, cases={"a": {"von_mises": {"percentile_by_volume_MPa": 25.0, "peak_qp_MPa": 60.0}}},
                 buckling={"status": "done", "cases": [{"name": "a", "load_factors": [3.0]}]},
                 motion={"status": "done", "rows": [{"name": "displacement:tip:a", "unit": "mm", "tet_value": 0.6}]})
    return {"gate": vg.evaluate(level, lim, percentile=99.9)}


def test_the_verify_gate_card_names_the_flag_the_argument_the_keys_it_reads_and_the_gap():
    c = card_of("verify_run:gate", "tool")
    assert c.layer == 4 and c.kind == "method" and not c.kind_note and c.confidence == "code" and not c.evidence
    assert "gate" in inspect.signature(server.verify_run).parameters and build_parser().parse_args(["verify", "somefolder", "--gate"]).gate is True
    assert "verify_run(run_dir, gate=true, estimate_only=false)" in c.call and "topopt verify <run> --gate" in c.call
    for words in ("`load_cases[].stress_limit`", "`buckling.min_factor`", "`limits.displacement[].limit_mm`", "`limits.alignment[].shift_limit_mm`", "`.tilt_limit_rad`", "`met: null`",
                  "The test is strict", "Exit code 3", "not_gated", "does not change a limit", "TET value", "`verify_run:allowable_mpa`", "no compression"):
        assert words in c.values, words
    assert "2.45 to 2.67 times, buckling factor 0.61 to 0.79 times, a displacement row 1.23 to 1.63 times" in c.values
    contracts = " ".join((ROOT / "CONTRACTS.md").read_text(encoding="utf-8").split())
    assert "the p99.9 stress of `pull_off` was 2.45 to 2.67 times the limit" in contracts
    assert "0.61 to 0.63 of the voxel factor on the `blur` STL and 0.71 to 0.79 on the `sdf` STL" in contracts
    assert "1.23 times (`sdf` STL) and 1.63 times (`blur` STL) the voxel value" in contracts                    # the numbers of the card are in CONTRACTS section 16
    assert not HABIT.search(c.values + c.effect)
    (s,) = c.symptoms
    assert "pass or fail" in s.words and s.facts[:2] == ("verify.gate.passed", "verify.gate.failed") and "`verify.gate.passed` is false" in s.when
    assert "verify_run:gate" in [x.key for x in levers.find("does it meet the limits")] and "verify_run:gate" in [x.key for x in levers.find("pass or fail")]
    assert "verify_run:gate" in card_of("verify_run", "tool").values


def test_the_fact_paths_of_the_gate_cards_are_keys_that_a_real_gate_result_has():
    from topopt import levers_run as lr
    doc = _gate_doc()
    assert doc["gate"]["passed"] is False and doc["gate"]["failed"] == ["displacement:displacement:tip:a"] and doc["gate"]["not_gated"]
    for key in ("verify_run:gate", "verify_run:allowable_mpa"):
        for s in card_of(key, "tool").symptoms:
            for path in s.facts:
                source, segs = lr._split(path)
                assert source == "verify" and list(lr._match(doc, segs)), path


def test_the_run_levers_flag_a_failed_gate():
    from topopt import levers_run as lr
    flag = next(f for f in lr.FLAGS if f.path == "verify.gate.passed")
    assert flag.rule == "false" and flag.joined == ("verify.gate.failed",)
    assert lr.flagged(flag.rule, flag.arg, False) and not lr.flagged(flag.rule, flag.arg, True)


def test_the_allowable_card_is_a_requirement_and_says_that_a_looser_value_needs_the_consent_of_the_user():
    c = card_of("verify_run:allowable_mpa", "tool")
    assert c.layer == 4 and c.kind == "requirement" and "allowable_mpa" in inspect.signature(server.verify_run).parameters
    for words in ("Tighten it freely", "loosens that requirement", "the consent of the user", "`replaced_limit`", "`replaced_source`", "`loosened`", "refuses nothing"):
        assert words in c.values, words
    (s,) = c.symptoms
    assert "verify.gate.checks.stress.<case>.loosened" in s.facts and "is true" in s.when and not HABIT.search(c.values)
    sub = build_parser()._subparsers._group_actions[0].choices["verify"]
    allow = " ".join(next(a for a in sub._actions if "--allowable-mpa" in a.option_strings).help.split())
    assert "replaces the stress limit of every load case in the YAML" in allow and "consent of the user" in allow and "replaced limit" in allow
    assert "topopt levers verify_run:gate" in " ".join(next(a for a in sub._actions if "--gate" in a.option_strings).help.split())


def test_the_cards_of_the_keys_that_the_gate_reads_say_that_the_gate_reads_them_as_a_limit_of_the_tet_value():
    sentence = "`verify_run(gate=true)` reads this as the limit of the tet value (card `verify_run:gate`)"
    for key in ("load_cases[].stress_limit", "load_cases[].allowable_mpa", "load_cases[].safety_factor", "limits.displacement[].limit_mm", "limits.alignment[].shift_limit_mm"):
        assert sentence in card_of(key).values, key
    assert "limits.alignment[].tilt_limit_rad" in card_of("limits.alignment[].shift_limit_mm").also
    bk = card_of("buckling.min_factor").values
    assert "second reader" in bk and "turns the buckling check of `verify` on" in bk and "the row of the optimizer and the limit of the gate together" in bk


def test_the_ratchet_cards_say_the_move_to_the_worse_side_and_have_the_owner_sentence_and_a_symptom_for_the_buckling_floor():
    rules = card_of("acceptance.rules").values
    assert "no_input` in this version" not in rules and "read the tet rows of `limits`" in rules
    margin = card_of("acceptance.rules[].margin")
    assert "the move to the worse side of the parent value that is allowed" in margin.values and "a fall of the factor for the rule `buckling`" in margin.values
    assert {"acceptance.rules[].tolerance_pct", "acceptance.rules[].against_parent", "acceptance.rules[].cases"} <= set(margin.also)
    mf = card_of("acceptance.rules[].min_factor")
    assert "tighten `min_factor` freely, with a report; loosen it only with the consent of the user" in mf.values
    (s,) = mf.symptoms
    assert "ratchet.facts.cases.<case>.buckling_lowest_factor" in s.facts and "buckle" in s.words
    assert "acceptance.rules[].min_factor" in [x.key for x in levers.find("the edit shifted the seats")]


def test_every_argument_of_verify_run_is_named_in_a_card_of_the_tool():
    """Scheme row 3 of the review of #136: a new argument of a carded tool with no word in a card failed no test. `gate` was the case."""
    cards = [c for c in lc.CARDS if c.group == "tool" and c.key.split(":")[0] == "verify_run"]
    named = " ".join(c.key.split(":")[-1] + " " + c.values for c in cards)
    missing = [p for p in inspect.signature(server.verify_run).parameters if p not in named]
    assert not missing, f"an argument of verify_run that no card names: {missing}"


def test_the_texts_that_agents_read_name_the_gate_and_the_stale_sentences_are_gone():
    texts = {f: (ROOT / f).read_text(encoding="utf-8") for f in ("skills/start-here/SKILL.md", ".agents/skills/start-here/SKILL.md", "topopt_mcp/roles/fea.md",
                                                                   "agents/fea-engineer.md", "topopt_mcp/README.md")}
    for f, t in texts.items():
        assert "gate=true" in t, f
        assert "pass or fail only for" not in t and "needs an allowable stress from the user" not in t, f
    assert "part_only=False, gate=False)" in texts["topopt_mcp/README.md"]
    contracts = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    for stale in ("`NO_PRODUCER`", "are `no_input` until a producer exists", "The judge runs with `buckling=False` (36.10)"):
        assert stale not in contracts, stale


# --- round 3: the finished-run fact names `continue_from` next to a new run ------------------------------------------------------

def test_the_options_of_the_resume_fact_name_continue_from_next_to_a_new_run():
    assert len(RESUME_OPTIONS) == 4
    first = next(i for i, o in enumerate(RESUME_OPTIONS) if o.startswith("run_topopt without resume=true and with a new out_dir"))
    option = RESUME_OPTIONS[first + 1]
    for words in ("run_topopt with continue_from=", "more_iterations=N", "allow_changed_inputs=true", "a new out_dir", "which is only read",
                  "the passes go on from its last checkpoint with the new values", "a changed grid is refused", "run_topopt:continue_from"):
        assert words in option, words
    assert {"continue_from", "more_iterations", "allow_changed_inputs", "out_dir"} <= set(inspect.signature(server.run_topopt).parameters)
    assert any("continue_from" in o for o in pass_key_note(["settings.penal: 3.0 -> 3.5"], True)["options"])
    assert not HABIT.search(" ".join(RESUME_OPTIONS))


def test_the_continue_option_applies_a_changed_key_of_the_passes_that_a_resume_of_the_finished_run_cannot(finished, tmp_path, capsys):
    """The option is true to the code: the call that the option names is refused without the override, and with it the new passes use the new
    value (the volume of the design moves to it), while the parent is only read."""
    path, base = finished
    work = tmp_path / "case"
    shutil.copytree(path.parent, work, ignore=shutil.ignore_patterns("run"))
    yaml_path = work / path.name
    d = yaml.safe_load(yaml_path.read_text())
    old_vf = d["settings"]["volume_fraction"]
    d["settings"]["volume_fraction"] = round(old_vf * 1.3, 4)
    yaml_path.write_text(yaml.safe_dump(d))
    parent_before = {p.relative_to(base).as_posix(): sha(p) for p in sorted(base.rglob("*")) if p.is_file()}
    out = tmp_path / "next"
    args = ["run", str(yaml_path), "--out", str(out), "--continue-from", str(base), "--more-iterations", "3", "--quiet"]
    assert main(args) != 0
    seen = capsys.readouterr()
    assert "--allow-changed-inputs" in seen.err + seen.out and "settings.volume_fraction" in seen.err + seen.out and not out.exists()
    assert main([*args, "--allow-changed-inputs"]) == 0
    rows = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    last_old = [json.loads(line) for line in (base / "history.jsonl").read_text().splitlines()][-1]
    assert rows[-1]["iteration"] == last_old["iteration"] + 3
    assert rows[-1]["volume_fraction"] > last_old["volume_fraction"] + 0.02         # the passes made the part heavier: the new value acts
    assert sha(out / "result" / "final.stl") != sha(base / "result" / "final.stl")
    assert {p.relative_to(base).as_posix(): sha(p) for p in sorted(base.rglob("*")) if p.is_file()} == parent_before    # the parent is only read


# --- round 3: a changed file of a not-printed region (open point 3 of fix round 1) ----------------------------------------------

def test_a_region_file_with_other_content_is_a_key_of_the_passes_only_when_it_moves_a_voxel_label():
    file_line = "region file fixed:plate.stl has other content (sha256 aaa -> bbb)"
    labels = "the voxel labels are not the same (the regions fill other voxels)"
    rename = "region file fixed:a.stl is now fixed:b.stl: the content is the same (sha256 aaa), only the path in the YAML changed"
    assert pass_key_note([file_line], True, labels_compared=True) is None                       # the labels were compared and are the same
    assert pass_key_note([file_line, labels], True, labels_compared=True)["keys"] == ["regions"]
    assert pass_key_note([rename], True) is None and pass_key_note([rename], True, labels_compared=True) is None   # the same bytes: no pass reads a difference
    unknown = pass_key_note([file_line], True)                                                  # the MCP reply builds no problem: it says so
    assert unknown["keys"] == ["regions"] and "region file fixed:plate.stl (this reply does not compare the voxels that it fills)" in unknown["fact"]
    assert "does not compare" not in pass_key_note([file_line, labels], True)["fact"]


@pytest.fixture(scope="module")
def printed_bar(tmp_path_factory):
    """A finished run with `regions.not_printed: [plate.stl]` (the plate is also the fixed region), voxel 1 mm, 6 passes."""
    tmp = tmp_path_factory.mktemp("levers_not_printed")
    path = make_case(tmp, h=1.0, iterations=6)
    d = yaml.safe_load(path.read_text())
    d["regions"]["not_printed"] = ["plate.stl"]
    path.write_text(yaml.safe_dump(d))
    out = tmp / "run"
    assert main(["run", str(path), "--out", str(out), "--quiet"]) == 0
    assert (out / "result" / "part_only.stl").is_file()
    return path, out


def plate_edit(printed_bar, tmp_path, tag, hi):
    """Copy the case and the run, write the plate as the box (-2,0,0)..hi, and return (the YAML, the run folder)."""
    path, base = printed_bar
    work, out = tmp_path / f"case_{tag}", tmp_path / f"run_{tag}"
    shutil.copytree(path.parent, work, ignore=shutil.ignore_patterns("run"))
    shutil.copytree(base, out)
    for name in ("job.DONE", "job.guard", "job.json"):
        (out / name).unlink(missing_ok=True)
    box(work / "plate.stl", (-2, 0, 0), hi)
    return work / path.name, out


def test_a_changed_file_of_a_not_printed_region_is_named_when_it_moves_labels_and_not_when_it_does_not(printed_bar, tmp_path):
    """Open point 3 of fix round 1. The file of a not-printed region is the file of a fixed, load, solid or keep_out region, and the change line has
    that role (there is no `not_printed:` line). A plate that reaches 1 mm into the design moves labels: the fact names it. A plate that
    moves by 0.01 mm gives other bytes and the same voxels: no pass reads a difference, so the resume log has no such fact."""
    yaml_moved, out_moved = plate_edit(printed_bar, tmp_path, "moved", (1.0, 10, 10))
    assert main(["run", str(yaml_moved), "--out", str(out_moved), "--resume", "--allow-changed-inputs", "--quiet"]) == 0
    log = (out_moved / "run.log").read_text()
    assert "region file fixed:plate.stl has other content" in log and "the voxel labels are not the same" in log
    fact = [ln for ln in log.splitlines() if "no pass runs again" in ln]
    assert len(fact) == 1 and "region file fixed:plate.stl" in fact[0] and "does not compare" not in fact[0], fact
    assert "not_printed:" not in log.split("no pass runs again")[0]

    yaml_tiny, out_tiny = plate_edit(printed_bar, tmp_path, "tiny", (0.01, 10, 10))
    before = state_of(out_tiny)
    assert main(["run", str(yaml_tiny), "--out", str(out_tiny), "--resume", "--allow-changed-inputs", "--quiet"]) == 0
    log = (out_tiny / "run.log").read_text()
    assert "region file fixed:plate.stl has other content" in log and "the voxel labels are not the same" not in log
    assert "no pass runs again" not in log and state_of(out_tiny) == before
    # the reply of the MCP tool builds no problem, so it cannot tell, and says so
    yaml_tiny2, out_tiny2 = plate_edit(printed_bar, tmp_path, "tiny2", (0.01, 10, 10))
    reply = server._resume_pass_keys(yaml_tiny2, out_tiny2)
    assert reply["pass_keys"] == ["regions"] and "does not compare the voxels that it fills" in reply["facts"][0]


# --- round 5 (P7-6): the E3 series, bracket and ladle ----------------------------------------------------------------------------------------

E3_CLASSES = ("bracket", "bracket-fitted", "ladle-4mm", "ladle")
E3_KEYS = ("settings.filter_radius", "settings.min_member_mm", "settings.iterations", "settings.beta_schedule", "settings.stop_rules", "settings.penal",
           "settings.volume_fraction", "settings.mesh_method")
BRACKET_PRICE = re.compile(r"optimizer pull x([\d.]+), mass x([\d.]+); P2 x([\d.]+)")
LADLE_PRICE = re.compile(r"optimizer scoop x([\d.]+), scrape x([\d.]+), stir x([\d.]+), objective x([\d.]+), mass x([\d.]+); "
                         r"P2 scoop x([\d.]+), scrape x([\d.]+), stir x([\d.]+), objective x([\d.]+)")
# The bands of the E3 series as fractions: (optimizer, mass, P2). A row is "inside the noise" when every value of its price is inside the band of ITS part class.
E3_BAND = {"bracket": (0.00045, 0.00037, 0.0209), "bracket-fitted": (0.0008, 0.00059, 0.01476),
           "ladle-4mm": (0.00439, 0.00084, 0.0154), "ladle": (0.00439, 0.00084, 0.0154)}   # the class `ladle` has no noise runs: it takes the bands of `ladle-4mm`


def e3_rows():
    return [(c, e) for c in lc.CARDS for e in c.evidence if e.part_class in E3_CLASSES]


def e3_values(price):
    """(optimizer values, mass ratio, P2 values) of an E3 price, or None for a row with no ratios (the time rows of `check_problem`)."""
    b, la = BRACKET_PRICE.search(price), LADLE_PRICE.search(price)
    if b:
        return [float(b[1])], float(b[2]), [float(b[3])]
    if la:
        g = [float(x) for x in la.groups()]
        return g[:4], g[4], g[5:]
    return None


def test_the_e3_rows_name_run_ids_of_the_e3_series_in_the_four_part_classes():
    rows = e3_rows()
    assert set(E3_CLASSES) <= set(lc.PART_CLASSES) and len(rows) == 49
    for c, e in rows:
        ids = [i.strip() for i in e.runs.split(",")]
        assert ids and all(re.fullmatch(r"E3/(BR|LAD)-[A-Za-z0-9_.]+", i) for i in ids), (c.key, e.runs)
        assert (e.part_class in ("bracket", "bracket-fitted")) == all(i.startswith("E3/BR-") for i in ids), (c.key, e.runs)
    by_key = {c.key for c, _ in rows}
    assert by_key == set(E3_KEYS) | {"check_problem"}
    assert all(c.confidence == "measured" for c in lc.CARDS if c.key in by_key)
    for e in (e for c in lc.CARDS for e in c.evidence if "E3/" in e.runs):
        assert e.part_class in E3_CLASSES, (e.change, e.part_class)      # a row never mixes an E3 run with a class of another series
    # the ladle on its own grid has the filter rows only
    assert {c.key for c, e in rows if e.part_class == "ladle"} <= {"settings.filter_radius", "settings.min_member_mm", "check_problem"}
    assert {e.runs for c, e in rows if e.part_class == "ladle" and c.key != "check_problem"} == {"E3/LAD-XFR1", "E3/LAD-XFR2", "E3/LAD-XMM2"}


def test_every_e3_row_with_ratios_gives_the_optimizer_ratio_and_the_p2_ratio():
    seen = 0
    for c, e in e3_rows():
        if c.key == "check_problem":
            continue
        assert e3_values(e.price) is not None, (c.key, e.runs, e.price)
        seen += 1
    assert seen == 46


def test_an_e3_row_is_marked_inside_the_noise_exactly_when_all_its_values_are_inside_the_band_of_its_part_class():
    marked = 0
    for c, e in e3_rows():
        vals = e3_values(e.price)
        if vals is None:
            continue
        opt, mass, p2 = vals
        band_opt, band_mass, band_p2 = E3_BAND[e.part_class]
        inside = all(abs(v - 1) <= band_opt for v in opt) and abs(mass - 1) <= band_mass and all(abs(v - 1) <= band_p2 for v in p2)
        assert inside == ("inside the noise" in e.price), (c.key, e.runs, e.price)
        marked += inside
    assert marked == 2       # E3/LAD-IT160 (iterations) and E3/LAD-SR160 (stop_rules)


def test_a_row_on_the_grey_bracket_base_says_so_and_a_row_on_the_fitted_base_does_not():
    for c, e in e3_rows():
        if e.part_class == "bracket":      # all 12 rows, the time row of `check_problem` too (AUDIT-P7-6, NOTE 6)
            assert "grey" in e.effect and "BR-B0" in e.effect and "0.129" in e.effect, (c.key, e.runs)       # "grey base (BR-B0 ends at beta 2, gray 0.129)" or "the base BR-B0 is grey (...)"
        if e.part_class == "bracket-fitted":
            assert "grey base" not in e.effect, (c.key, e.runs)
    assert sum(e.part_class == "bracket" for _, e in e3_rows()) == 12       # 11 change rows and the time row
    [time_row] = [e for c, e in e3_rows() if c.key == "check_problem" and e.part_class == "bracket"]
    assert time_row.effect.startswith("grey base (BR-B0 ends at beta 2, gray 0.129); the check estimated 29.5 s")


def test_the_audit_fixes_of_p7_6_say_which_classes_have_a_band_which_parts_have_time_rows_and_the_ranges_of_the_run_files():
    """AUDIT-P7-6, 3 MISMATCH lines and the NOTE lines 1 and 2: the words, and the ranges that the run files give (25.3 to 36.0 s for the 15 runs of 40 passes on the grey
    base; 7.9 to 13.4 s for the P2 check of the 22 runs at 4.0 mm)."""
    price = " ".join(lc.PRICE_TEXT.split())
    assert "Each of these classes has its own base; the bands are in its description (the class `ladle` uses the bands of `ladle-4mm`)" in price
    assert "its own noise band" not in price
    contracts = " ".join((ROOT / "CONTRACTS.md").read_text(encoding="utf-8").split())
    assert ("Each class gives its base values and the noise band it uses (the class `ladle` has no noise set of its own and uses the bands of `ladle-4mm`)" in contracts
            and "its measured noise band" not in contracts)
    assert "uses the bands of `ladle-4mm`" in lc.PART_CLASSES["ladle"] and "no noise set of its own" in lc.PART_CLASSES["ladle"]
    detect = " ".join(s.detect for s in card_of("check_problem", "tool").symptoms)
    assert "runs of the bracket and the ladle (E3)" in detect and "three parts" not in detect
    bracket, ladle4 = " ".join(lc.PART_CLASSES["bracket"].split()), " ".join(lc.PART_CLASSES["ladle-4mm"].split())
    assert "25.3 to 36.0 s for 40 passes" in bracket and "25 to 37 s" not in bracket
    assert "the P2 check adds 7.9 to 13.4 s" in ladle4 and "adds 8 to 18 s" not in ladle4


def test_the_part_classes_of_the_bracket_and_the_ladle_give_the_base_values_and_the_noise_bands():
    words = {
        "bracket": ("642.1 N/mm", "632.8 N/mm", "5398.2 mm3", "6.69 g", "0.129", "BR-B0 and BR-B1", "byte for byte", "BR-N1, BR-N2, BR-N3", "0.045 %", "0.037 %",
                    "2.09 %", "grey design", "inside the noise"),
        "bracket-fitted": ("[[0,1],[15,2],[25,4],[32,8]]", "710.9 N/mm", "656.0 N/mm", "5444.0 mm3", "0.020", "0.00096", "BR-SN1, BR-SN2", "0.08 %", "0.059 %",
                           "1.476 %", "no repeat run"),
        "ladle-4mm": ("401 s", "55 x 24 x 22", "267.7 / 408.1 / 166.2", "284.1 / 449.1 / 173.4", "186540.9 mm3", "231.31 g", "8.3155", "LAD-D0 and LAD-D1",
                      "0.439 %", "0.20 %", "0.084 %", "1.54 %", "1.3 %", "a definition of the E3 report, not an engine value", "272 to 276 s"),
        "ladle": ("68 x 30 x 27", "400.7 s", "381.2 s", "236.81 g", "280.8 / 424.9 / 175.4", "304.1 / 479.8 / 186.4", "7.9586",
                  "no noise set of its own", "uses the bands of `ladle-4mm`", "6.5 mm against 8.0 mm"),
    }
    for name, needed in words.items():
        text = lc.PART_CLASSES[name]
        for w in needed:
            assert w in text, (name, w)
        assert not HABIT.search(text), name
    assert "a general law" in lc.PRICE_TEXT and "A row is one run of one part class" in lc.PRICE_TEXT


# the sentences across parts: key -> words that the price note holds (the numbers are the objective ratios of the E3 report, section 5)
CROSS_PARTS = {
    "settings.filter_radius": ("On three parts (E3;", "1 voxel (no filter): the optimizer ratio differs, up on the hook (x1.009) and the grey bracket (x1.049), down on the fitted bracket "
                               "(x0.956) and the ladle (x0.894; at 3.25 mm x0.909)", "inside the band on the grey bracket (x0.987)",
                               "2 voxels: the optimizer ratio is down on the hook (x0.953), the bracket (x0.942; fitted x0.977) and the ladle (x0.953): same sign on the three parts; "
                               "the P2 ratio differs", "3 voxels: the optimizer ratio is down on the hook (x0.856), the bracket (x0.835; fitted x0.926) and the ladle (x0.821): "
                               "same sign on the three parts"),
    "settings.min_member_mm": ("On three parts (E3;", "optimizer ratio down on all three (hook x0.961, bracket x0.951, fitted bracket x0.981, ladle x0.968): same sign on the three parts, "
                               "with the P2 ratio inside the band of its part on all four (x0.971, x1.001, x1.005, x0.993)",
                               "optimizer and P2 ratios down on all three parts: same sign on the three parts"),
    "settings.iterations": ("On three parts (E3;", "down on all three (hook x0.812, bracket x0.939, ladle x0.730): same sign on the three parts",
                            "the sign differs, and on the bracket it depends on the base", "up on the bracket (x1.114; fitted x1.006)", "the sign differs"),
    "settings.beta_schedule": ("On three parts (E3;", "up on the grey bracket (x1.078) and down on the fitted bracket (x0.973): the sign differs, and on the bracket it depends on the base"),
    "settings.stop_rules": ("on three parts (E3;", "up on the bracket (x1.114 on the grey base, x1.006 on the fitted base", "the sign differs"),
    "settings.penal": ("On three parts (E3;", "penal 2.0 gives an optimizer ratio up on all three (hook x1.041, bracket x1.049, fitted bracket x1.022, ladle x1.169): "
                       "same sign on the three parts; the P2 ratio differs", "Penal 4.0 gives an optimizer ratio down on all three (hook x0.968, bracket x0.970, fitted bracket x0.967, "
                       "ladle x0.900): same sign on the three parts"),
    "settings.volume_fraction": ("On three parts (E3;", "same sign on the three parts in both measures", "x1.253; P2 x1.164, x1.154, x1.124, x1.234"),
    "settings.mesh_method": ("On three parts (E3;", "the optimizer ratios are x1.000 on every part", "P2 is lower on all three, the same sign", "hook x0.669", "bracket x0.968 (fitted bracket x0.938)",
                             "ladle x0.009", "The mass ratio differs in sign"),
}


@pytest.mark.parametrize("key", E3_KEYS)
def test_a_key_with_rows_on_three_parts_gives_the_sign_on_each_part_and_says_if_it_is_the_same(key):
    c = card_of(key)
    note = c.price_note
    for w in CROSS_PARTS[key]:
        assert w in note, (key, w)
    assert "same sign on the three parts" in note or "the sign differs" in note or "differs" in note, key
    assert not HABIT.search(note) and "one part" not in note.lower().replace("one part class", ""), key
    assert "general law" not in note, key
    assert c.confidence == "measured" and any(e.part_class in E3_CLASSES for e in c.evidence) and any(e.part_class in HOOK_CLASSES for e in c.evidence), key


def test_the_wording_of_one_part_is_gone_from_the_price_text_and_the_cards_of_the_e3_keys():
    assert "one part (the hook)" not in lc.PRICE_TEXT.lower() and "the rows are facts about this part" not in lc.PRICE_TEXT
    for key in E3_KEYS:
        c = card_of(key)
        text = " ".join([c.price_note, c.values, *(e.change + " " + e.effect + " " + e.price for e in c.evidence)]).lower()
        assert "on one part" not in text and "of one part" not in text and "one part only" not in text, key


def test_the_member_cost_of_two_voxels_is_on_five_rows_with_mass_and_grayness_and_the_warning_rules():
    """F4-19: facts only (no decision, no default change, no new warning)."""
    note = card_of("settings.min_member_mm").price_note
    for w in ("F4-19, the 2-voxel member on five rows (optimizer objective / P2 objective / mass, grayness base -> run)",
              "hook x0.819 / x0.848 / x0.974, 0.025 -> 0.112 (E1/MM40)", "bracket x0.755 / x0.907 / x0.970, 0.129 -> 0.314 (E3/BR-MM2, grey base)",
              "fitted bracket x0.889 / x0.977 / x0.986, 0.020 -> 0.137 (E3/BR-SMM2)", "ladle at 4.0 mm x0.697 / x0.856 / x0.973, 0.014 -> 0.074 (E3/LAD-MM2)",
              "ladle at 3.25 mm x0.814 / x0.968 / x0.981, 0.012 -> 0.070 (E3/LAD-XMM2)", "The cost is 18.1, 24.5, 11.1, 30.3 and 18.6 % in the optimizer and 15.2, 9.3, 2.3, 14.4 and 3.2 % in P2",
              "The 1-voxel member costs x0.951 to x0.981 in the optimizer, with P2 inside the band",
              "'A member needs 2 voxels or more to carry load'", "appears for a `min_member_mm` of 1 voxel", "it does not appear for a `min_member_mm` of 2 voxels",
              "for an explicit `filter_radius` of 1, 2 or 3 voxels", "(E1/MM20, E3/BR-MM1, E3/BR-SMM1, E3/LAD-MM1)"):
        assert w in note, w
    # the cost percentages follow from the ratios of the five rows
    for opt, cost in ((0.819, 18.1), (0.755, 24.5), (0.889, 11.1), (0.697, 30.3), (0.814, 18.6)):
        assert f"{(1 - opt) * 100:.1f}" == f"{cost}"
    for p2, cost in ((0.848, 15.2), (0.907, 9.3), (0.977, 2.3), (0.856, 14.4), (0.968, 3.2)):
        assert f"{(1 - p2) * 100:.1f}" == f"{cost}"
    rows = {e.runs: e for e in card_of("settings.min_member_mm").evidence}
    assert "optimizer pull x0.755, mass x0.970; P2 x0.907" in rows["E3/BR-MM2"].price and "gray 0.314" in rows["E3/BR-MM2"].effect
    assert "objective x0.697, mass x0.973; P2 scoop x1.008, scrape x0.993, stir x0.617, objective x0.856" in rows["E3/LAD-MM2"].price
    assert "no warning" in rows["E3/LAD-MM2"].effect and "1 warning" in rows["E3/LAD-MM1"].effect


def test_the_filter_card_gives_the_facts_of_a_radius_of_one_voxel_with_no_warning():
    """F4-25: 1 entry per row, no warning, P2 below the optimizer on four of five rows, the islands removed."""
    c = card_of("settings.filter_radius")
    for w in ("F4-25, a radius of 1 voxel (five rows: E1/FR20, E3/BR-FR1, E3/BR-SFR1, E3/LAD-FR1, E3/LAD-XFR1)", "`metrics.filter.size.entries_per_row` is 1 (10064 entries on the bracket",
              "`metrics.filter.facts` is empty and `run.log` has no `warning:` line", "'filter size: about 1 entries per row' (BR-FR1)",
              "the mesh step removes loose islands (`metrics.mesh.islands_removed`: hook 5, bracket 3, fitted bracket 2, ladle 27, ladle at 3.25 mm 35)",
              "lower than the base by 0.44 to 0.54 mm (hook, bracket, fitted bracket, ladle) and by 1.37 mm (ladle at 3.25 mm)",
              "x0.897 (hook), x0.940 (grey bracket), x1.009 (fitted bracket), x0.854 (ladle) and x0.859 (ladle at 3.25 mm)",
              "P2 is 6 to 15 % below the optimizer on four of the five rows and within 0.9 % of it on the fitted bracket"):
        assert w in c.price_note, w
    one_voxel = {e.runs: e for e in c.evidence if e.runs in ("E3/BR-FR1", "E3/BR-SFR1", "E3/LAD-FR1", "E3/LAD-XFR1")}
    assert len(one_voxel) == 4
    for runs, e in one_voxel.items():
        assert "1 entry per row" in e.effect and "no warning" in e.effect and "islands removed" in e.effect, runs
    for runs, islands in (("E3/BR-FR1", "3 islands"), ("E3/BR-SFR1", "2 islands"), ("E3/LAD-FR1", "27 islands"), ("E3/LAD-XFR1", "35 islands")):
        assert islands in one_voxel[runs].effect, runs
    # P2 / optimizer on the fitted bracket is the "within 0.9 %" row: 0.964 against 0.956
    opt, p2 = (float(x) for x in re.search(r"pull x([\d.]+).*P2 x([\d.]+)", one_voxel["E3/BR-SFR1"].price).groups())
    assert abs(p2 / opt - 1) < 0.009


def test_the_pairs_of_doubled_passes_give_the_gain_with_and_without_the_last_beta_stage():
    """F4-22: x1.000 to x1.080 with the stage reached, x1.064 to x1.391 (optimizer) and x0.987 to x1.287 (P2) without it, the 0.0017 contrast, the stop_rules run."""
    it = card_of("settings.iterations").price_note
    for w in ("F4-22, what doubling the passes gave in pairs of runs on the same part", "when the smaller run had reached the last beta stage (8 pairs:",
              "the gain was x1.000 to x1.080 in the optimizer and x1.000 to x1.080 in P2", "When it had not (4 pairs:",
              "the gain was x1.064 to x1.391 in the optimizer and x0.987 to x1.287 in P2", "the grayness of those four smaller runs was 0.085 to 0.176",
              "the verdict showed `last_beta_stage` not met on them and met on the other eight",
              "At the same end-of-run objective change of 0.0017 the hook (E1/B0, 6 design passes at beta 8) gained x1.001 (P2 x1.016) and the ladle (E3/LAD-IT40s, "
              "8 design passes at beta 8) gained x1.080 (P2 x1.080), so the objective change alone does not predict the gain"):
        assert w in it, w
    for w in ("the bracket pair at 0.00025 (E3/BR-B0, last beta stage not reached) gained x1.114 (P2 x1.038)", "at 4.8e-05 or less (`converged`, 15 design passes at beta 8)",
              "x1.001 and x1.000 with the schedule unchanged and x1.034 and x1.004 with the schedule scaled by 2"):
        assert w in it, w
    oc = card_of("settings.stop_rules.objective_change").price_note
    rules = card_of("settings.stop_rules").price_note
    for w in ("pass 48 of 80 (E1/SR80)", "pass 78 (E3/BR-SR80 with 80 passes, E3/BR-SR160 with 160)", "pass 79 of 160 (E3/LAD-SR160)",
              "flip 0.0 % against E3/BR-IT80 and against E3/LAD-D0", "the rule ended a 160-pass run 82 and 81 passes early"):
        assert w in rules, w
    rows = {e.runs: e for e in card_of("settings.stop_rules").evidence if e.part_class in E3_CLASSES}
    assert "stopped_by_rule at pass 79 (79 passes)" in rows["E3/LAD-SR160"].effect and "flip 0.0 %" in rows["E3/LAD-SR160"].effect
    assert "stopped_by_rule at pass 78 (78 passes)" in rows["E3/BR-SR80, E3/BR-SR160"].effect
    assert 160 - 78 == 82 and 160 - 79 == 81
    # the task asks for the pairs "in short": the card names no decision and no new default
    assert not HABIT.search(it + oc + rules)


def test_the_blur_row_of_the_ladle_gives_the_p2_values_as_one_run_with_the_cause_not_isolated():
    c = card_of("settings.mesh_method")
    row = next(e for e in c.evidence if e.runs == "E3/LAD-MBb")
    assert row.part_class == "ladle-4mm" and "one run, cause not isolated" in row.effect
    assert "P2 scoop x0.005, scrape x0.007, stir x0.071, objective x0.009" in row.price and "optimizer scoop x1.000" in row.price
    for w in ("ladle x0.009 (scoop x0.005, scrape x0.007, stir x0.071)", "from one run, cause not isolated", "x0.005 is not a measured stiffness of a printable part"):
        assert w in c.price_note, w
    r = next(e for e in c.evidence if e.runs == "E3/BR-SMBb")             # the fitted bracket: the bracket row that the card prefers
    assert r.part_class == "bracket-fitted" and "P2 x0.938" in r.price and "optimizer pull x1.000" in r.price


def test_the_grayness_below_the_verdict_test_is_not_taken_for_the_last_beta_stage_on_the_three_symptoms_that_name_the_grayness():
    text = lc.GRAY_NOT_ENOUGH
    for w in ("A grayness below 0.1 alone does not show", "E3/LAD-IT40", "grayness 0.085", "x0.730", "x0.923", "`last_beta_stage` (not met)"):
        assert w in text, w
    holders = {(c.key, s.words) for c in lc.CARDS for s in c.symptoms if text in s.detect}
    assert holders == {("settings.iterations", "grey, blurred, not finished, not converged"), ("settings.beta_schedule", "grey, blurred, fuzzy"),
                       ("settings.stop_rules.grayness_max", "grey")}
    assert "0.085" in levers.reference() and not HABIT.search(text)


def test_the_check_estimate_and_the_wall_time_of_the_ladle_and_the_bracket_are_on_the_check_card_and_the_voxel_card():
    rows = {e.runs: e for e in card_of("check_problem", "tool").evidence if "E3/" in e.runs}
    assert set(rows) == {"E3/LAD-B0", "E3/LAD-D0", "E3/BR-B0, E3/BR-B1"}
    for runs, words in (("E3/LAD-B0", ("160.5 s", "400.7 s", "381.2 s", "2.5 times", "21.51 23.92 27.16", "27.71 29.62 29.03")),
                        ("E3/LAD-D0", ("55.1 s", "118.8 s", "2.2 times", "24.92 27.73 28.41", "21.33 25.92 27.69")),
                        ("E3/BR-B0, E3/BR-B1", ("29.5 s", "36.0 s", "28.6 s", "1.2 times", "21.90 24.52 27.58", "22.17 24.38 27.43"))):
        e = rows[runs]
        for w in words:
            assert w in e.effect + " " + e.price, (runs, w)
    assert rows["E3/LAD-B0"].part_class == "ladle" and rows["E3/LAD-D0"].part_class == "ladle-4mm" and rows["E3/BR-B0, E3/BR-B1"].part_class == "bracket"
    voxel = card_of("settings.voxel_size").cost
    assert "4.0 mm took 119 s and 3.25 mm took 401 s of wall time (E3/LAD-D0, E3/LAD-B0), against checks of 55 s and 160 s" in voxel
    check_symptom = next(s for s in card_of("check_problem", "tool").symptoms if "take long" in s.words)
    assert "1.2 times the estimate on the bracket and 2.2 and 2.5 times on the ladle" in check_symptom.detect


def card_text(c):
    parts = [c.effect, c.values or "", c.cost or "", c.kind_note or "", c.price_note or ""]
    for s in c.symptoms:
        parts += [s.detect, s.when or ""]
    for e in c.evidence:
        parts += [e.change, e.effect, e.price]
    return " ".join(parts)


def test_the_pairs_of_f4_22_are_on_one_card_and_the_two_other_cards_have_a_summary_and_a_pointer():
    """The pair facts are on `settings.iterations` (the key that an agent changes for more passes). `beta_schedule` and `stop_rules.objective_change` have a
    summary and a pointer; `beta_schedule` keeps the rows about the schedule itself (the scaled-schedule runs)."""
    markers = ("pairs of runs on the same part", "(8 pairs", "(4 pairs", "E3/LAD-IT20 -> E3/LAD-IT40", "E3/BR-IT160s", "design passes at beta 8",
               "At the same end-of-run objective change of 0.0017", "the bracket pair at 0.00025", "the grayness of those four smaller runs")
    for m in markers:
        assert {c.key for c in lc.CARDS if m in card_text(c)} == {"settings.iterations"}, m
    for key in ("settings.beta_schedule", "settings.stop_rules.objective_change"):
        note = card_of(key).price_note
        for w in ("F4-22 (E3)", "x1.000 to x1.080", "up to x1.391 in the optimizer and x1.287 in P2", "does not predict the gain", "The pairs, the numbers and the runs are on the card `settings.iterations`"):
            assert w in note, (key, w)
        assert not HABIT.search(note)
    assert "E3/" not in card_of("settings.stop_rules.objective_change").price_note
    beta = {(e.part_class, e.runs) for e in card_of("settings.beta_schedule").evidence if "E3/" in e.runs}
    assert beta == {("bracket", "E3/BR-BS40s"), ("bracket", "E3/BR-IT20s"), ("bracket-fitted", "E3/BR-IT20s"), ("bracket-fitted", "E3/BR-IT80"),
                    ("ladle-4mm", "E3/LAD-IT40s"), ("ladle-4mm", "E3/LAD-IT160s")}
    iterations = {(e.part_class, e.runs) for e in card_of("settings.iterations").evidence if "E3/" in e.runs}
    assert len(iterations) == 9 and {"E3/LAD-IT40", "E3/LAD-IT160", "E3/BR-IT20", "E3/BR-IT80"} <= {r for _, r in iterations}


# --- F1-9: the wall rule against the rod rule (Refs #150) ------------------------------------------------------------------------------------------------------

F19_CLASSES = ("hook 80 passes (F1-9)", "bracket (F1-9)", "ladle (F1-9)")
F19_RUN = re.compile(r"F1-9/(hookA|bracket|ladle)-(WALL|ROD|G375)")


def f19_rows():
    return [e for e in card_of("settings.min_member_mm").evidence if e.part_class in F19_CLASSES]


def test_the_min_member_card_says_what_a_rod_needs_and_that_the_rod_rule_loosens_the_request():
    values = " ".join(card_of("settings.min_member_mm").values.split())
    for w in ("a rod needs about 1.76 x the value", "walls of 4.0 mm and rods of about 7.03 mm", "`metrics.filter.min_feature_note` gives both sizes",
              "`filter_radius` 1.068 x the value is the rod rule", "it makes walls thinner than the value, so it loosens the request", "use it only with the consent of the user",
              "continuum values for the interior of the design space and not limits of a result", "round members from the value to 1.76 x the value are also in the wall-rule results"):
        assert w in values, w
    assert not HABIT.search(values)
    fr = card_of("settings.filter_radius")
    text = " ".join((fr.effect + " " + fr.values).split())
    for w in ("a rod thinner than about 0.94 x R vanishes and a wall thinner than about 0.53 x R", "R = 1.877 x v for a wall (the wall rule of `min_member_mm`)",
              "R = 1.068 x v for a rod (the rod rule)", "loosens a request for a member size: use it only with the consent of the user", "on the card `settings.min_member_mm`"):
        assert w in text, w
    assert not HABIT.search(text)
    # the numbers of the sentence are the constants of the engine
    from topopt import members
    assert members.RADIUS_PER_MEMBER == 1.877 and round(members.ROD_CRITICAL * members.RADIUS_PER_MEMBER, 2) == 1.76 and round(members.WALL_CRITICAL, 2) == 0.53
    assert round(1 / members.ROD_CRITICAL, 3) == 1.068


def test_the_f19_rows_are_on_the_min_member_card_with_their_own_part_classes_and_the_old_rows_stay():
    card = card_of("settings.min_member_mm")
    rows = f19_rows()
    assert [e.runs for e in rows] == ["F1-9/hookA-WALL", "F1-9/hookA-ROD", "F1-9/hookA-G375", "F1-9/bracket-WALL", "F1-9/bracket-ROD", "F1-9/ladle-WALL", "F1-9/ladle-ROD"]
    cls_of = {"hookA": "hook 80 passes (F1-9)", "bracket": "bracket (F1-9)", "ladle": "ladle (F1-9)"}
    for e in rows:
        m = F19_RUN.fullmatch(e.runs)
        assert m and e.part_class == cls_of[m.group(1)], e.runs
        assert "optimizer" in e.price and "P2" in e.price and "mass x" in e.price, e.runs
        assert re.search(r"outside the band of the class: optimizer (yes|no), P2 (yes|no), mass (yes|no)$", e.effect), e.runs
        assert "thickness of the design material at density 0.5 (mm)" in e.effect and "round rod-like voxels from" in e.effect, e.runs
        assert set(e.part_class for e in rows) <= set(lc.PART_CLASSES)
    # the rows of the other series keep their classes (the E-series tests count them)
    assert card.evidence[0].runs == "E1/MM10" and {"E1/MM40", "E3/BR-MM2", "E3/LAD-XMM2", "case 1, F1-9"} <= {e.runs for e in card.evidence}
    assert not any(e.part_class in F19_CLASSES for c in lc.CARDS if c.key != "settings.min_member_mm" for e in c.evidence)
    assert levers.problems() == []


def test_the_f19_classes_give_the_base_values_the_definitions_and_the_band_they_use():
    for name, words in {
        "hook 80 passes (F1-9)": ("hook.yaml", "8982 active voxels", "code of origin/main 74a1e46", "4.0 mm = 2 voxels", "hookA-BASE", "1297.4, 1344.1", "1.3624", "1041.0, 921.9", "1.8397", "25.71 g",
                                  "the bands of the class `hook` are used (optimizer stiffness +-0.3 %, mass +-0.1 %, P2 +-3.5 %)", "F1-9/hookB-BASE"),
        "bracket (F1-9)": ("bracket.yaml", "12352 active voxels", "2.0 mm = 2 voxels", "bracket-BASE", "repeated BR-B0", "642.1 N/mm", "6.69 g", "0.129", "646.4 N/mm", "632.8 N/mm",
                           "the bands of the class `bracket` are used (optimizer 0.045 %, mass 0.037 %, P2 2.09 %)"),
        "ladle (F1-9)": ("ladle.yaml", "36542 active voxels", "6.5 mm = 2 voxels", "ladle-BASE", "repeated LAD-B0", "280.8, 424.9, 175.4", "7.9586", "236.81 g",
                         "no noise set of its own and uses the bands of `ladle-4mm`"),
    }.items():
        text = " ".join(lc.PART_CLASSES[name].split())
        for w in words:
            assert w in text, (name, w)
        for w in ("a stiffness ratio and an objective ratio are base over run", "the mass ratio is run over base", "a definition of the F1-9 report", "bias about a quarter of a voxel",
                  "at least 3 voxel centres from an inactive voxel", "l3/l2 is 0.5 or more", "checked on synthetic rods and bars only"):
            assert w.lower() in text.lower(), (name, w)
        assert not HABIT.search(text), name


def _x(price, label):
    """The value x<number> that follows `label` in a price string."""
    return float(re.search(re.escape(label) + r" x([\d.]+)", price).group(1))


def test_the_wall_over_rod_price_of_the_note_follows_from_the_rows():
    """WALL over ROD = (base over WALL) / (base over ROD) for each ratio of the two rows; the note gives it with 3 digits."""
    note = " ".join(card_of("settings.min_member_mm").price_note.split())
    assert "F1-9, the wall rule against the rod rule at the same request of 2 voxels" in note
    by = {e.runs: e for e in f19_rows()}
    claims = {
        "hookA": ("hook.yaml at 2 mm (request 4.0 mm)", ("optimizer", "objective"), ("P2", "objective")),
        "bracket": ("bracket (request 2.0 mm)", ("optimizer", "pull"), ("P2", None)),
        "ladle": ("ladle at 3.25 mm (request 6.5 mm)", ("optimizer", "objective"), ("P2", "objective")),
    }
    for part, (lead, opt, p2) in claims.items():
        w, r = by[f"F1-9/{part}-WALL"].price, by[f"F1-9/{part}-ROD"].price

        def pick(price, where):
            head, label = where
            seg = price.split("; ")[0 if head == "optimizer" else 1]
            if label is None:       # the bracket P2 value has no label: "P2 x0.894"
                return float(re.fullmatch(r"P2 x([\d.]+)", seg).group(1))
            return _x(seg, label)

        want_opt = pick(w, opt) / pick(r, opt)
        want_p2 = pick(w, p2) / pick(r, p2)
        want_mass = _x(w, "mass") / _x(r, "mass")
        m = re.search(re.escape(lead) + r" x([\d.]+) / x([\d.]+), mass ([\d.]+)", note)
        assert m, part
        assert abs(float(m.group(1)) - want_opt) < 0.0025 and abs(float(m.group(2)) - want_p2) < 0.0025 and abs(float(m.group(3)) - want_mass) < 0.0025, (part, m.groups(), want_opt, want_p2, want_mass)
        assert f"(F1-9/{part}-WALL, F1-9/{part}-ROD)" in note
    # the wall rule is the stiffer-losing side on every part, in the optimizer and in P2, and the P2 loss is the smaller one
    for part, (lead, opt, p2) in claims.items():
        m = re.search(re.escape(lead) + r" x([\d.]+) / x([\d.]+)", note)
        assert float(m.group(1)) < float(m.group(2)) < 1.0, part
    for w in ("hook_short.yaml x0.867 / x0.877, mass 0.981", "hookB-WALL repeats E1/MM40 in the optimizer", "The wall rule costs stiffness on every part, outside the bands of the class in the optimizer, in P2 and in mass",
              "the P2 loss is smaller than the optimizer loss", "11 % (wall rule) against 33 % (rod rule) and 49 % (default) on the hook", "1 % (wall rule) against 8 % (rod rule) and 9 % (default) on the bracket",
              "5 % (wall rule) against 25 % (rod rule) and 36 % (default) on the ladle", "that is a loosening of the requirement, and the consent of the user applies",
              "(share of the design material, wall rule / rod rule / default: hook 14 / 11 / 10 %, bracket 6 / 3 / 3 %, ladle 12 / 11 / 10 %)",
              "1.76 x the request is a continuum reference size and not a measured limit of these designs", "holds 89 % of the hook material and 37 % of the bracket material", "(68 to 162 voxels)"):
        assert w in note, w
    assert not HABIT.search(note) and "one part" not in note.lower().replace("one part class", "")
    # the old sentences of the note stay
    assert note.startswith("Read P2 where the radius is 1 voxel or less") and "F4-19, the 2-voxel member on five rows" in note

