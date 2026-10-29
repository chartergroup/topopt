"""P9-4b: a change of a method setting does not stop the brief gate (CONTRACTS.md section 30, "Method keys").

The kind of each YAML key comes from the lever cards (`topopt.levers_cards`). These tests read the kinds from the table at run time:
they pin no count of cards and no fixed list of keys, except the keys that the examples below change (`settings.solver`,
`settings.voxel_size`, `settings.volume_fraction`, `load_cases[].name`, `load_cases[].forces[].force`), and the test of the table
checks that each of them still has the kind that the example needs.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml as pyyaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from topopt import case  # noqa: E402
from topopt.levers_cards import CARDS  # noqa: E402
from test_optimize import box, make_case  # noqa: E402

BRIEF = "A bracket for a 2 kg shelf. The load is 20 N down on the tip."
KINDS = {c.key: c.kind for c in CARDS if c.group == "yaml"}


@pytest.fixture
def work(tmp_path):
    d = tmp_path / "runs" / "part"
    d.mkdir(parents=True)
    make_case(d)
    return tmp_path


@pytest.fixture
def yml(work):
    return (work / "runs" / "part" / "c.yaml").resolve()


def edit(yml, change):
    """Load the YAML, let `change(data)` edit it, write it again."""
    data = pyyaml.safe_load(yml.read_text())
    change(data)
    yml.write_text(pyyaml.safe_dump(data))


def record(yml, work):
    return case.record_brief(yml, work, BRIEF, None)


def brief_inputs(yml):
    return json.loads((yml.parent / "c.case.json").read_text())["brief"]["inputs"]


def log_events(yml):
    return [json.loads(x) for x in (yml.parent / "c.case-log.jsonl").read_text().splitlines()]


def passes(yml, work):
    """The gate lets the run start with a current brief; return its `brief_facts`."""
    g = case.gate(yml, work)
    assert g.stop is None, g.stop
    assert g.reply["brief_stated"] is True
    return g.reply["brief_facts"]


def stops(yml, work):
    """The gate stops with `awaiting_user`; return its facts."""
    g = case.gate(yml, work)
    assert g.stop is not None and g.stop["state"] == "awaiting_user"
    return g.stop["facts"]


# --- the kinds come from the cards ----------------------------------------------------------------------

def test_the_examples_use_the_kinds_that_they_need():
    """If a card changes its kind, the examples of this file say so before any other test fails."""
    for key in ("settings.solver", "settings.voxel_size", "settings.iterations", "load_cases[].name"):
        assert KINDS[key] == "method", key
    for key in ("settings.volume_fraction", "load_cases[].forces[].force", "regions.design"):
        assert KINDS[key] in ("requirement", "both"), key


def test_each_card_key_has_its_own_kind_and_the_more_specific_card_wins():
    for card in (c for c in CARDS if c.group == "yaml"):
        for key in (card.key, *card.also):
            tokens = case._key_tokens(card.key if "<" in key else key)
            assert case.key_kind(tokens) == card.kind, key
    pairs = [(a, b) for a in KINDS for b in KINDS if b.startswith((a + ".", a + "[]")) and KINDS[a] != KINDS[b]]
    assert pairs, "the table has no card whose kind differs from the card of its parent: the test cannot check the rule"
    for parent, child in pairs:
        assert case.key_kind(case._key_tokens(parent)) == KINDS[parent]
        assert case.key_kind(case._key_tokens(child)) == KINDS[child]


def test_a_key_with_no_card_takes_the_kind_of_the_nearest_card_and_a_top_level_key_with_none_is_a_requirement():
    assert case.key_kind(case._key_tokens("load_cases[].forces[].vector")) == KINDS["load_cases[].forces"]
    assert case.key_kind(case._key_tokens("settings.no_such_key")) == KINDS["settings"]
    assert case.key_kind(("no_such_block", "no_such_key")) == "requirement"
    assert case.key_kind(()) == "requirement"


def test_a_name_token_of_a_card_matches_any_name():
    named = [c for c in CARDS if c.group == "yaml" and "<" in c.key]
    assert named, "the table has no card with a <name> token: the test cannot check the rule"
    card = named[0]
    tokens = tuple("Ti64" if t.startswith("<") else t for t in case._key_tokens(card.key))
    assert case.key_kind(tokens) == card.kind
    other = tuple("Steel" if t.startswith("<") else t for t in case._key_tokens(card.key))
    assert case.key_kind(other) == card.kind


# --- the gate ignores a change of method keys only ---------------------------------------------------------

def test_a_solver_change_passes_the_gate_and_the_fact_names_the_key_and_both_values(work, yml):
    edit(yml, lambda d: d["settings"].update(solver="amg"))
    record(yml, work)
    edit(yml, lambda d: d["settings"].update(solver="gpu-mg"))
    facts = passes(yml, work)
    assert any("settings.solver" in f and "`amg`" in f and "`gpu-mg`" in f and "a method key changed from" in f
               and "brief stays current" in f for f in facts), facts
    st = case.status(yml, work)
    assert st["brief_stated"] is True and st["brief_stale"] is False and st["stale_files"] == []
    assert st["gate"] == "run_topopt starts"
    assert any("settings.solver" in f and "`gpu-mg`" in f for f in st["facts"])
    assert "gate_returned" not in {e["event"] for e in log_events(yml)}


def test_a_new_method_key_is_a_change_from_not_set(work, yml):
    record(yml, work)
    edit(yml, lambda d: d["settings"].update(solver="gpu-mg"))
    facts = passes(yml, work)
    assert any("settings.solver" in f and "not set" in f and "`gpu-mg`" in f for f in facts), facts


def test_a_voxel_size_change_passes_the_gate_and_the_fact_names_both_values(work, yml):
    record(yml, work)
    edit(yml, lambda d: d["settings"].update(voxel_size=2.0))
    facts = passes(yml, work)
    assert any("settings.voxel_size" in f and "`1.0`" in f and "`2.0`" in f for f in facts), facts
    assert case.status(yml, work)["brief_stale"] is False


def test_a_removed_method_key_and_a_list_value_are_named(work, yml):
    record(yml, work)

    def change(d):
        d["settings"]["iterations"] = 50
        d["settings"]["beta_schedule"] = [[0, 1], [10, 3]]
        d["settings"]["penal"] = 4.0

    edit(yml, change)
    facts = passes(yml, work)
    assert any("settings.iterations" in f and "`30`" in f and "`50`" in f for f in facts), facts
    assert any("settings.beta_schedule" in f and "[[0, 1], [10, 2], [20, 4]]" in f and "[[0, 1], [10, 3]]" in f
               for f in facts), facts
    assert any("settings.penal" in f and "not set" in f and "`4.0`" in f for f in facts), facts
    edit(yml, lambda d: d["settings"].pop("beta_schedule"))
    assert any("settings.beta_schedule" in f and "`[[0, 1], [10, 2], [20, 4]]`" in f and "to not set" in f
               for f in passes(yml, work))


def test_a_method_block_that_goes_away_is_not_stale(work, yml):
    """`settings.stop_rules` holds method keys only. The block can come and go, because a method key has no weight in the brief."""
    edit(yml, lambda d: d["settings"].update(stop_rules={"enabled": True, "window": 5}))
    record(yml, work)
    edit(yml, lambda d: d["settings"].pop("stop_rules"))
    facts = passes(yml, work)
    assert any("settings.stop_rules.enabled" in f and "`true`" in f and "not set" in f for f in facts), facts
    assert any("settings.stop_rules.window" in f and "`5`" in f for f in facts), facts


def test_a_value_that_json_cannot_hold_is_recorded(work, yml):
    """A YAML date has no JSON form: the record keeps its text, and the next compare finds no change."""
    edit(yml, lambda d: d.update(name=pyyaml.safe_load("2026-10-23")))
    record(yml, work)
    assert brief_inputs(yml)["method_values"]["name"] == "2026-10-23"
    assert not any("a method key changed" in f for f in passes(yml, work))


# --- a requirement key, a `both` key, a list path ----------------------------------------------------------

def test_a_force_change_is_stale_and_the_fact_names_the_key(work, yml):
    record(yml, work)
    edit(yml, lambda d: d["load_cases"][0]["forces"][0].update(force=[0, 0, -3.0]))
    facts = stops(yml, work)
    assert any("load_cases[0].forces[0].force" in f and "requirement" in f for f in facts), facts
    assert not any("load_cases[1]" in f for f in facts)
    st = case.status(yml, work)
    assert st["brief_stale"] is True and st["stale_files"] == ["c.yaml"] and "awaiting_user" in st["gate"]


def test_a_volume_fraction_change_is_stale_as_before(work, yml):
    record(yml, work)
    edit(yml, lambda d: d["settings"].update(volume_fraction=0.4))      # `both` counts as a requirement in this round
    facts = stops(yml, work)
    assert any("settings.volume_fraction" in f and "requirement" in f for f in facts), facts
    assert case.status(yml, work)["stale_files"] == ["c.yaml"]


def test_a_list_path_is_matched_by_its_card_for_every_item(work, yml):
    record(yml, work)
    edit(yml, lambda d: d["load_cases"][1].update(name="sideways"))     # `load_cases[].name`: method
    facts = passes(yml, work)
    assert any("load_cases[1].name" in f and "`side`" in f and "`sideways`" in f for f in facts), facts
    edit(yml, lambda d: d["load_cases"][1]["forces"][0].update(force=[1.0, 4.0, 0]))    # `load_cases[].forces[].force`: requirement
    facts = stops(yml, work)
    assert any("load_cases[1].forces[0].force" in f for f in facts), facts
    assert not any("load_cases[0].forces" in f for f in facts)


def test_a_new_and_a_removed_requirement_key_are_named(work, yml):
    record(yml, work)

    def change(d):
        d["load_cases"].append({"name": "third", "forces": [{"region": "tip", "force": [0, 1, 0]}]})
        del d["load_cases"][0]["weight"]

    edit(yml, change)
    (named,) = [f for f in stops(yml, work) if "requirement keys changed" in f]
    assert "`load_cases[2].forces[0].force` (new)" in named and "`load_cases[0].weight` (removed)" in named


def test_a_method_change_and_a_requirement_change_together_are_stale_and_both_are_named(work, yml):
    record(yml, work)

    def change(d):
        d["settings"].update(solver="gpu-mg", voxel_size=2.0)
        d["load_cases"][0]["forces"][0].update(force=[0, 0, -5.0])

    edit(yml, change)
    facts = stops(yml, work)
    assert any("load_cases[0].forces[0].force" in f and "requirement" in f for f in facts), facts
    assert any("settings.solver" in f and "`gpu-mg`" in f for f in facts), facts
    assert any("settings.voxel_size" in f and "`2.0`" in f for f in facts), facts
    assert not any("brief stays current" in f for f in facts)          # the brief does not stay current


def test_a_method_change_and_a_changed_stl_are_stale_for_the_file_and_the_method_fact_stays(work, yml):
    record(yml, work)
    edit(yml, lambda d: d["settings"].update(solver="gpu-mg"))
    box(yml.parent / "tip.stl", (18, 4, 0), (20, 6, 4))
    st = case.status(yml, work)
    assert st["brief_stale"] is True and st["stale_files"] == ["tip.stl"]
    assert any("settings.solver" in f for f in st["facts"])


def test_a_region_file_name_is_a_requirement(work, yml):
    record(yml, work)
    box(yml.parent / "other.stl", (0, 0, 0), (20, 10, 10))
    edit(yml, lambda d: d["regions"].update(design="other.stl"))
    facts = stops(yml, work)
    assert any("regions.design" in f for f in facts), facts


def test_a_comment_and_a_method_change_together_pass(work, yml):
    record(yml, work)
    edit(yml, lambda d: d["settings"].update(solver="amg"))
    yml.write_text("# the faster backend\n" + yml.read_text())
    assert any("settings.solver" in f for f in passes(yml, work))


# --- the record -------------------------------------------------------------------------------------------------

def test_the_brief_holds_the_split_and_a_run_entry_does_not(work, yml):
    record(yml, work)
    inputs = brief_inputs(yml)
    assert {"yaml", "yaml_content", "files", "yaml_requirements", "requirement_keys", "method_values"} <= set(inputs)
    assert inputs["method_values"]["settings.voxel_size"] == 1.0
    assert "settings.volume_fraction" in inputs["requirement_keys"]
    assert "settings.voxel_size" not in inputs["requirement_keys"]
    assert all(len(v) == 12 for v in inputs["requirement_keys"].values())
    g = case.gate(yml, work)
    case.run_started(g, yml, work, work / "runs" / "live", 4242)
    (entry,) = json.loads((yml.parent / "c.case.json").read_text())["runs"]
    assert set(entry["inputs"]) == {"yaml", "yaml_content", "files"}        # the run is compared with the whole content
    assert entry["inputs"] == case.current_inputs(yml, work).record()


def test_the_record_holds_no_absolute_path(work, yml):
    record(yml, work)
    text = (yml.parent / "c.case.json").read_text() + (yml.parent / "c.case-log.jsonl").read_text()
    assert str(work) not in text
    for needle in ("/home/", "/Users/", "/tmp/"):
        assert needle not in text, needle


def test_the_same_content_gives_the_same_split_and_a_changed_requirement_gives_another_digest(work, yml):
    a = case.current_inputs(yml, work)
    assert case.current_inputs(yml, work).split == a.split
    edit(yml, lambda d: d["settings"].update(solver="gpu-mg"))
    b = case.current_inputs(yml, work)
    assert b.split.requirements == a.split.requirements and b.yaml_content != a.yaml_content
    edit(yml, lambda d: d["settings"].update(volume_fraction=0.5))
    assert case.current_inputs(yml, work).split.requirements != a.split.requirements


def test_an_integer_and_an_equal_float_in_a_requirement_are_not_a_change(work, yml):
    record(yml, work)
    edit(yml, lambda d: d["load_cases"][0].update(weight=3))            # was 3.0
    assert not any("requirement" in f for f in case.status(yml, work)["facts"])
    assert case.status(yml, work)["brief_stale"] is False


# --- an old record ---------------------------------------------------------------------------------------------

def old_record(yml):
    """Take the new fields out of the brief, as a case file from before P9-4b has it."""
    path = yml.parent / "c.case.json"
    doc = json.loads(path.read_text())
    for key in ("yaml_requirements", "requirement_keys", "method_values"):
        del doc["brief"]["inputs"][key]
    path.write_text(json.dumps(doc))


def test_an_old_record_is_stale_on_a_method_change_as_before_and_says_how_to_get_the_rule(work, yml):
    record(yml, work)
    old_record(yml)
    assert passes(yml, work) == []
    edit(yml, lambda d: d["settings"].update(solver="gpu-mg"))
    facts = stops(yml, work)
    assert any("YAML content changed" in f for f in facts)
    (hint,) = [f for f in facts if "record_brief again" in f]
    assert "method key" in hint and "settings.solver" in hint
    st = case.status(yml, work)
    assert st["brief_stale"] is True and st["stale_files"] == ["c.yaml"] and hint in st["facts"]
    record(yml, work)                                       # the new brief has the rule
    assert passes(yml, work) == []
    edit(yml, lambda d: d["settings"].update(solver="amg"))
    assert any("settings.solver" in f and "`gpu-mg`" in f and "`amg`" in f for f in passes(yml, work))


def test_a_damaged_record_is_stale_and_does_not_raise(work, yml):
    record(yml, work)
    path = yml.parent / "c.case.json"
    doc = json.loads(path.read_text())
    doc["brief"]["inputs"]["requirement_keys"] = ["not", "a", "dict"]
    path.write_text(json.dumps(doc))
    edit(yml, lambda d: d["settings"].update(volume_fraction=0.4))
    facts = stops(yml, work)
    assert any("YAML content changed" in f for f in facts) and not any("case check failed" in f for f in facts)


def test_an_old_record_with_a_comment_change_has_no_hint(work, yml):
    record(yml, work)
    old_record(yml)
    yml.write_text("# note\n" + yml.read_text())
    assert not any("record_brief again" in f for f in passes(yml, work))


# --- the runs keep the whole comparison ---------------------------------------------------------------------------

def test_a_run_entry_is_stale_after_a_change_of_a_method_key_too(work, yml):
    """The guard of the run comparison: it is the whole content, as before P9-4b (this test passes before and after)."""
    record(yml, work)
    case.run_started(case.gate(yml, work, proceed_without_brief=True), yml, work, work / "runs" / "live", 4242)
    assert case.status(yml, work)["runs"][0]["stale"] is False
    edit(yml, lambda d: d["settings"].update(solver="gpu-mg"))
    (run,) = case.status(yml, work)["runs"]
    assert run["stale"] is True and run["stale_files"] == ["c.yaml"]
    assert any("c.yaml" in f and "content changed" in f for f in run["facts"])


def test_a_run_entry_is_stale_after_a_solver_change_while_the_brief_is_not(work, yml):
    record(yml, work)
    case.run_started(case.gate(yml, work), yml, work, work / "runs" / "live", 4242)
    edit(yml, lambda d: d["settings"].update(solver="gpu-mg"))
    st = case.status(yml, work)
    assert st["brief_stale"] is False and st["brief_stated"] is True
    (run,) = st["runs"]
    assert run["stale"] is True and run["stale_files"] == ["c.yaml"]
    assert any("YAML content changed" in f for f in run["facts"])
    assert not any("brief stays current" in f for f in run["facts"])
    # the whole comparison is the default of `compare`
    entry = json.loads((yml.parent / "c.case.json").read_text())["runs"][0]
    now = case.current_inputs(yml, work)
    assert case.compare(entry["inputs"], now, "c.yaml").stale is True
    brief = json.loads((yml.parent / "c.case.json").read_text())["brief"]["inputs"]
    assert case.compare(brief, now, "c.yaml").stale is True                     # not method_free
    assert case.compare(brief, now, "c.yaml", method_free=True).stale is False


def test_a_run_started_after_the_method_change_is_current(work, yml):
    record(yml, work)
    edit(yml, lambda d: d["settings"].update(solver="gpu-mg"))
    g = case.gate(yml, work)
    out = case.run_started(g, yml, work, work / "runs" / "live", 4243)
    assert out["brief_stated"] is True and any("settings.solver" in f for f in out["brief_facts"])
    st = case.status(yml, work)
    assert st["runs"][0]["stale"] is False
    edit(yml, lambda d: d["settings"].update(solver="amg"))
    assert case.status(yml, work)["runs"][0]["stale"] is True


# --- the case log keeps the method facts of a run start (review m1) ---------------------------------------------------

def test_run_started_logs_the_method_facts_and_the_entry_does_not(work, yml):
    record(yml, work)
    case.run_started(case.gate(yml, work), yml, work, work / "runs" / "live", 1)           # no method change: an empty list
    edit(yml, lambda d: d["settings"].update(solver="gpu-mg"))
    g = case.gate(yml, work)
    assert len(g.method_facts) == 1 and "settings.solver" in g.method_facts[0]
    case.run_started(g, yml, work, work / "runs" / "live", 2)
    started = [e for e in log_events(yml) if e["event"] == "run_started"]
    assert started[0]["method_facts"] == []
    assert started[1]["method_facts"] == g.method_facts and "`gpu-mg`" in started[1]["method_facts"][0]
    entries = json.loads((yml.parent / "c.case.json").read_text())["runs"]
    assert all("method_facts" not in e for e in entries)                                     # the entry is not changed


def test_the_override_with_a_stale_brief_logs_the_method_facts_too(work, yml):
    record(yml, work)

    def change(d):
        d["settings"].update(solver="gpu-mg")
        d["load_cases"][0]["forces"][0].update(force=[0, 0, -7.0])

    edit(yml, change)
    g = case.gate(yml, work, proceed_without_brief=True)
    assert g.override is not None and len(g.method_facts) == 1
    case.run_started(g, yml, work, work / "runs" / "live", 3)
    last = log_events(yml)[-1]
    assert last["event"] == "run_started" and "settings.solver" in last["method_facts"][0]
    assert "brief stays current" not in last["method_facts"][0]                              # the brief is stale here


def test_a_stop_and_a_current_brief_without_a_gate_have_no_method_facts(work, yml):
    assert case.Gate().method_facts == []
    record(yml, work)
    assert case.gate(yml, work).method_facts == []


# --- one fact form for every method key, also for a key that is not a setting (review n1) --------------------------------

def test_one_fact_form_for_every_method_key_and_no_word_setting(work, yml):
    record(yml, work)

    def change(d):
        d["name"] = "renamed_part"                          # a method key, but not a setting of the method
        d["assumptions"] = ["short note for people"]        # same
        d["load_cases"][0]["name"] = "hang2"
        d["settings"].update(solver="gpu-mg", voxel_size=2.5)

    edit(yml, change)
    facts = [f for f in passes(yml, work) if f.startswith("`")]
    paths = sorted(f.split("`")[1] for f in facts)
    assert paths == ["assumptions", "load_cases[0].name", "name", "settings.solver", "settings.voxel_size"], facts
    for f in facts:
        path = f.split("`")[1]
        assert f.startswith(f"`{path}`: a method key changed from ") and " to " in f and f.endswith("; the brief stays current"), f
        assert "setting" not in f.replace(path, ""), f
