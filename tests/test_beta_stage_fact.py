"""P7-4 (F4-18): a fact before the run when the planned passes end before the last beta stage.

The rule is the one of the verdict (`verdict.reaches_last_beta_stage`): the last pass is at or after the start of the last stage of the
schedule. The run makes the passes 0 to `settings.iterations`. The fact is in `optimize.problem_facts` (so in `topopt check`, `topopt check
--json` key `facts` and the run log) and in `check_data` `warnings` (so in the MCP tool `check_problem`). It blocks nothing.
"""
import json
import random
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from beta_fact_helpers import START  # noqa: E402
from test_checkpoint import finish, killed, log, resume  # noqa: E402
from test_optimize import make_case  # noqa: E402
from topopt import verdict  # noqa: E402
from topopt.cli import check_data, main  # noqa: E402
from topopt.config import StopRules, load_config  # noqa: E402
from topopt.optimize import beta_at, problem_facts  # noqa: E402
from topopt.verdict import HOOK_PRICE, _beta_text, beta_stage_fact, reaches_last_beta_stage, scaled_beta_schedule  # noqa: E402

HOOK = [[0, 1], [16, 2], [26, 4], [34, 8]]            # examples/hook/hook_short.yaml (written for 40 passes)
DEFAULT = [[0, 1], [30, 2], [50, 4], [65, 8]]          # the schedule when the key is absent


def case(tmp_path, iterations, schedule="default", h=2.0):
    """The cantilever of `make_case` with `iterations` and a schedule: a list, or "default" (the key is removed)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "c.yaml"
    if not p.exists():      # the STL files are made once for each folder; a call in the same folder rewrites the settings only
        make_case(tmp_path, h=h)
    d = yaml.safe_load(p.read_text())
    d["settings"]["iterations"] = iterations
    if schedule == "default":
        d["settings"].pop("beta_schedule", None)
    else:
        d["settings"]["beta_schedule"] = schedule
    p.write_text(yaml.safe_dump(d))
    return p


def beta_facts(p):
    return [f for f in problem_facts(load_config(p)) if f.startswith(START)]


def schedule_of(fact):
    """The YAML `beta_schedule` of the second option of a fact."""
    return yaml.safe_load(re.search(r"beta_schedule: (\[\[.*?\]\]) \(each start x", fact).group(1))


# --- when the fact is due ---------------------------------------------------------------------------------------------


def test_a_user_schedule_that_the_run_does_not_reach_gives_one_fact_with_the_options_and_the_price(tmp_path):
    [f] = beta_facts(case(tmp_path, 20, HOOK))
    assert f.startswith("fact: settings.iterations is 20; the last stage of settings.beta_schedule [[0, 1], [16, 2], [26, 4], [34, 8]] (beta 8) starts at pass 34.")
    assert "The run ends at pass 20 at beta 2 (the stage from pass 16) and does not reach beta 8: " in f
    assert "the verdict will say that the beta schedule has not reached its last stage, and a design that has not seen the last beta can stay grey." in f
    assert " options: settings.iterations 34 or more (pass 34 only measures; each pass above it is one change of the design at beta 8); " in f
    assert ("or beta_schedule: [[0, 1], [8, 2], [12, 4], [16, 8]] (each start x 20/42, rounded to a pass, the last one down; "
            "42 = the last start 34 + the 8 passes of the stage before it).") in f
    # the price comes after the options: measured on the hook example, named as such, with the run ids
    assert f.index("options:") < f.index("Measured on examples/hook/hook_short.yaml (2 mm voxels, one part, one run each")
    assert "x0.791 / x0.834, grayness 0.142 (IT20)" in f and "x0.989 / x0.970, grayness 0.026 (IT20o)" in f
    assert "x0.801 / x0.869, grayness 0.138 (BSc)" in f
    assert "stays grey" not in f and "IT20s" not in f      # grey only "can" stay (measured for the runs that end at beta 2)
    assert "\n" not in f and len(f) < 1100      # one line of the run log, options before the price


def test_the_default_schedule_gives_the_fact_too(tmp_path):
    [f] = beta_facts(case(tmp_path, 40))                                  # no `beta_schedule` key
    assert "the last stage of settings.beta_schedule [[0, 1], [30, 2], [50, 4], [65, 8]] (beta 8) starts at pass 65." in f
    assert "The run ends at pass 40 at beta 2 (the stage from pass 30)" in f
    assert " options: settings.iterations 65 or more (pass 65 only measures" in f
    # 65 x 40/80 = 32.5: the last start is rounded down (not to 33), so the pass 32 changes the design at beta 8
    assert "beta_schedule: [[0, 1], [15, 2], [25, 4], [32, 8]] (each start x 40/80, rounded to a pass, the last one down; 80 = the last start 65 + the 15 passes" in f
    [f] = beta_facts(case(tmp_path / "short", 20))
    assert "The run ends at pass 20 at beta 1 (the stage from pass 0)" in f      # not even the second stage
    assert schedule_of(f) == [[0, 1], [8, 2], [13, 4], [16, 8]]


@pytest.mark.parametrize("schedule", [HOOK, DEFAULT])
def test_the_fact_is_due_exactly_when_the_verdict_would_not_call_the_last_stage_reached(tmp_path, schedule):
    """The verdict's own comparison, at every pass count around the start of the last stage (also at exactly the start pass)."""
    sched = tuple((s, float(b)) for s, b in schedule)
    start = sched[-1][0]
    p = case(tmp_path, 1, schedule)
    d = yaml.safe_load(p.read_text())
    for n in range(1, start + 4):
        row = {"iteration": n, "beta": beta_at(sched, n), "grayness": 0.0, "constraints": []}
        met = verdict.measures([row], StopRules(), sched)["last_beta_stage"]["met"]
        assert reaches_last_beta_stage(n, sched) is met
        d["settings"]["iterations"] = n
        p.write_text(yaml.safe_dump(d))
        assert bool(beta_facts(p)) is (not met), n
    assert beta_facts(case(tmp_path / "at", start, schedule)) == []       # exactly the start pass: reached, no fact
    assert len(beta_facts(case(tmp_path / "below", start - 1, schedule))) == 1


def test_a_one_stage_schedule_and_a_run_that_reaches_the_last_stage_give_no_fact(tmp_path):
    assert beta_facts(case(tmp_path / "one", 1, [[0, 4]])) == []
    assert beta_facts(case(tmp_path / "ok", 80)) == [] and beta_facts(case(tmp_path / "hook", 40, HOOK)) == []
    assert beta_stage_fact(5, ((0, 1.0),)) == []


def test_the_helper_of_the_other_tests_takes_the_line_out_once_and_only_when_it_is_due(tmp_path, monkeypatch):
    """n5: the old tests of other facts read `problem_facts` without this line; a missing or a second line must not pass unseen."""
    import beta_fact_helpers as h
    due, not_due = load_config(case(tmp_path / "due", 6, HOOK)), load_config(case(tmp_path / "ok", 34, HOOK))
    assert len(problem_facts(due)) == 1 and h.problem_facts_without_beta_fact(due) == []
    assert h.problem_facts_without_beta_fact(not_due) == problem_facts(not_due) == []
    line = problem_facts(due)[0]
    monkeypatch.setattr(h, "problem_facts", lambda cfg: ["fact: another", line, line])
    with pytest.raises(AssertionError, match="exactly once"):
        h.problem_facts_without_beta_fact(due)
    monkeypatch.setattr(h, "problem_facts", lambda cfg: ["fact: another"])           # due, and not in the list
    with pytest.raises(AssertionError, match="exactly once"):
        h.problem_facts_without_beta_fact(due)
    monkeypatch.setattr(h, "problem_facts", lambda cfg: [START + "34; a line that is not the fact"])      # not due, but a line starts like it
    with pytest.raises(AssertionError, match="starts like"):
        h.problem_facts_without_beta_fact(not_due)


def test_the_hook_and_ladle_examples_reach_their_last_stage_so_they_get_no_fact():
    for name in ("hook/hook.yaml", "hook/hook_short.yaml", "hook/hook_stress.yaml", "hook/hook_buckling.yaml", "hook/hook_mirror.yaml", "ladle/ladle.yaml"):
        st = load_config(REPO / "examples" / name).settings
        assert beta_stage_fact(st.iterations, st.beta_schedule) == [], name


# --- the options ----------------------------------------------------------------------------------------------------


def test_the_scaled_schedule_numbers():
    assert scaled_beta_schedule(20, [(0, 1.0), (16, 2.0), (26, 4.0), (34, 8.0)]) == (42, [[0, 1.0], [8, 2.0], [12, 4.0], [16, 8.0]])
    # the default schedule: the span is the 80 passes that it fits (65 + the 15 passes of the stage before the last), halved for 40 passes;
    # the other starts are rounded half up (7.5 -> 8, 12.5 -> 13), the last start down (32.5 -> 32)
    assert scaled_beta_schedule(40, [(0, 1.0), (30, 2.0), (50, 4.0), (65, 8.0)]) == (80, [[0, 1.0], [15, 2.0], [25, 4.0], [32, 8.0]])
    assert scaled_beta_schedule(20, [(0, 1.0), (30, 2.0), (50, 4.0), (65, 8.0)]) == (80, [[0, 1.0], [8, 2.0], [13, 4.0], [16, 8.0]])
    assert scaled_beta_schedule(5, [(0, 1.0), (10, 8.0)]) == (20, [[0, 1.0], [2, 8.0]])          # two stages: the last as long as the first; 2.5 -> 2


def test_the_last_start_of_the_offer_is_below_iterations_so_the_design_changes_at_the_last_beta():
    """The cases of the review (m1): a short stage before the last one made the last start equal `iterations` (the last beta only on the pass that measures)."""
    assert scaled_beta_schedule(16, [(0, 1.0), (60, 2.0), (62, 8.0)])[1] == [[0, 1.0], [15, 8.0]]                 # was [[0,1],[15,2],[16,8]]
    assert scaled_beta_schedule(12, [(0, 1.0), (40, 2.0), (44, 4.0), (46, 8.0)])[1] == [[0, 1.0], [10, 2.0], [11, 8.0]]   # was [..., [11, 4], [12, 8]]
    # an earlier start that rounds half up past the last start (13 x 2/17 = 1.53 -> 2, 15 x 2/17 = 1.76 -> 1) is held at it, and the later stage stays
    assert scaled_beta_schedule(2, [(0, 1.0), (13, 2.0), (15, 8.0)]) == (17, [[0, 1.0], [1, 8.0]])
    rng = random.Random(20261023)
    for _ in range(3000):
        stages = rng.randint(2, 6)
        starts = [0] + sorted(rng.sample(range(1, 200), stages - 1))
        sched = [(s, float(rng.choice((1, 2, 4, 8, 16)))) for s in starts]
        n = rng.randint(1, starts[-1] - 1) if starts[-1] > 1 else 1
        if reaches_last_beta_stage(n, sched):
            continue
        span, scaled = scaled_beta_schedule(n, sched)
        got = [s for s, _ in scaled]
        assert got[0] == 0 and got == sorted(set(got)), (sched, n, scaled)            # what the loader needs: it starts at 0, the starts increase
        assert scaled[-1][1] == sched[-1][1] and got[-1] <= n - 1, (sched, n, scaled)  # the last beta is kept, and it has a pass that changes the design


def test_stages_that_fall_on_one_pass_become_one_stage_and_the_text_says_so(tmp_path):
    assert scaled_beta_schedule(1, [(0, 1.0), (30, 2.0), (50, 4.0), (65, 8.0)])[1] == [[0, 8.0]]
    [f] = beta_facts(case(tmp_path, 1))
    assert schedule_of(f) == [[0, 8]]
    # n3: the first beta changes when a later stage falls on pass 0: the text names it
    assert "Stages that fall on one pass are one stage: the later one stays (the run then starts at beta 8)." in f
    [g] = beta_facts(case(tmp_path / "h", 40, [[0, 1], [2, 2], [100, 8]]))
    assert schedule_of(g) == [[0, 2], [20, 8]] and "the later one stays (the run then starts at beta 2)." in g
    [m] = beta_facts(case(tmp_path / "m", 12, [[0, 1], [40, 2], [44, 4], [46, 8]]))
    assert "the later one stays." in m and "then starts" not in m                  # the first beta is the same: nothing to name
    [n] = beta_facts(case(tmp_path / "n", 20, HOOK))
    assert "Stages that fall on one pass" not in n


def test_a_beta_is_printed_so_that_yaml_reads_the_same_number():
    """n2: `:g` has 6 digits, and PyYAML reads `1e-05` and `1e+06` as strings."""
    for beta in (8.0, 2.5, 1.5, 16.0, 1.23456789, 1e6, 1e-5, 123456.7, 1234567.0, 1e16, 0.1, 3.14159265358979):
        text = _beta_text(beta)
        assert yaml.safe_load(f"[{text}]") == [beta], (beta, text)
    assert [_beta_text(b) for b in (8.0, 2.5, 1.0)] == ["8", "2.5", "1"]            # the usual betas stay short
    [f] = beta_stage_fact(20, [(0, 1.0), (16, 2.5), (26, 4.0), (34, 1.23456789)])
    assert "[[0, 1], [16, 2.5], [26, 4], [34, 1.23456789]]" in f and schedule_of(f)[-1] == [16, 1.23456789]
    [g] = beta_stage_fact(5, [(0, 1.0), (10, 1e-5)])
    assert "[[0, 1], [10, 1.0e-05]]" in g and schedule_of(g) == [[0, 1], [2, 1e-05]]


@pytest.mark.parametrize("schedule", ["default", HOOK, [[0, 1], [3, 2], [4, 4], [5, 8]], [[0, 2], [7, 16]]])
def test_both_options_pass_the_loader_and_end_the_fact(tmp_path, schedule):
    """For every pass count below the start: the YAML of the scaled schedule loads, reaches the last stage, and the fact is gone."""
    start = load_config(case(tmp_path, 1, schedule)).settings.beta_schedule[-1][0]
    for n in range(1, start):
        original = load_config(case(tmp_path, n, schedule)).settings.beta_schedule
        [f] = beta_facts(case(tmp_path, n, schedule))
        scaled = schedule_of(f)
        again = case(tmp_path, n, scaled)                                 # the user pastes the YAML of the option
        st = load_config(again).settings                                 # the loader takes it: starts at 0, the starts increase
        assert reaches_last_beta_stage(n, st.beta_schedule) and beta_facts(again) == [], (n, scaled)
        assert st.beta_schedule[-1][1] == original[-1][1]                # the last beta is the same
        assert max(s for s, _ in st.beta_schedule) <= n - 1                # the pass that changes the design at the last beta (m1)
        more = int(re.search(r"options: settings.iterations (\d+) or more", f).group(1))
        assert more == start and beta_facts(case(tmp_path, more, schedule)) == []   # the first option


# --- where it shows ---------------------------------------------------------------------------------------------------


def test_topopt_check_prints_the_fact_once_and_json_has_it_in_facts_and_warnings(tmp_path, capsys):
    p = case(tmp_path, 6, [[0, 1], [10, 2], [20, 4]])
    assert main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.count(START + "6;") == 1 and "warning: fact: settings.iterations" not in out
    assert "the last stage of settings.beta_schedule [[0, 1], [10, 2], [20, 4]] (beta 4) starts at pass 20." in out
    assert main(["check", str(p), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [f for f in data["facts"] if f.startswith(START)] == [f for f in data["warnings"] if f.startswith(START)]
    assert len([f for f in data["facts"] if f.startswith(START)]) == 1 and data["iterations"] == 6
    assert main(["check", str(case(tmp_path / "ok", 20, [[0, 1], [10, 2], [20, 4]])), "--json"]) == 0      # reaches the start pass
    ok = json.loads(capsys.readouterr().out)
    assert not [f for f in ok["facts"] + ok["warnings"] if f.startswith(START)]


def test_check_data_and_the_mcp_tool_check_problem_have_the_fact_in_warnings(tmp_path, monkeypatch):
    from topopt_mcp import server
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    p = case(tmp_path, 12, HOOK)
    [w] = [w for w in check_data(p)["warnings"] if w.startswith(START)]
    r = server.check_problem(str(p))
    assert r["ok"] is True and [x for x in r["warnings"] if x.startswith(START)] == [w]
    assert "The run ends at pass 12 at beta 1 (the stage from pass 0)" in w
    json.dumps(r)
    ok = server.check_problem(str(case(tmp_path / "ok", 34, HOOK)))
    assert ok["ok"] is True and not [x for x in ok["warnings"] if x.startswith(START)]


def test_a_run_with_the_fact_logs_it_before_the_first_pass_and_still_finishes(tmp_path):
    p = case(tmp_path, 3, [[0, 1], [10, 2], [20, 4]])
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0                    # not a refusal
    log = (out / "run.log").read_text().splitlines()
    at = [i for i, t in enumerate(log) if t.startswith(START)]
    first_pass = next(i for i, t in enumerate(log) if t.startswith("it   0"))
    assert len(at) == 1 and at[0] < first_pass
    assert "The run ends at pass 3 at beta 1 (the stage from pass 0) and does not reach beta 4: " in log[at[0]]
    assert json.loads((out / "status.json").read_text())["state"] == "done"
    assert len((out / "history.jsonl").read_text().splitlines()) == 4
    v = json.loads((out / "verdict.json").read_text())      # the verdict at the end says the same, with the same rule
    assert v["measures"]["last_beta_stage"]["met"] is False and v["measures"]["last_beta_stage"]["from_iteration"] == 20
    assert "has not reached its last stage (beta 4 starts at pass 20, the run ends at beta 1)" in v["text"]


def test_a_run_that_reaches_the_last_stage_logs_no_fact(tmp_path):
    p = case(tmp_path, 3, [[0, 1], [2, 2]])
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    assert START not in (out / "run.log").read_text()
    v = json.loads((out / "verdict.json").read_text())
    assert v["measures"]["last_beta_stage"]["met"] is True


# --- a continuation (round 2): the pass count is K + more_iterations, and settings.iterations of the YAML is not used ---------------------------------

CONT = "fact: the pass count of this run is "      # the start of the line of the fact of a continuation

# The line of a normal run (hook_short, 20 passes) as the run log had it before round 2 (scratch/p7-4/hook20/run_r1/run.log): it must not change.
HOOK_20 = (
    "fact: settings.iterations is 20; the last stage of settings.beta_schedule [[0, 1], [16, 2], [26, 4], [34, 8]] (beta 8) starts at pass 34. "
    "The run ends at pass 20 at beta 2 (the stage from pass 16) and does not reach beta 8: the verdict will say that the beta schedule has not "
    "reached its last stage, and a design that has not seen the last beta can stay grey. options: settings.iterations 34 or more (pass 34 only "
    "measures; each pass above it is one change of the design at beta 8); or beta_schedule: [[0, 1], [8, 2], [12, 4], [16, 8]] (each start x "
    "20/42, rounded to a pass, the last one down; 42 = the last start 34 + the 8 passes of the stage before it). Measured on "
    "examples/hook/hook_short.yaml (2 mm voxels, one part, one run each; stiffness hang / pull_off against the 40-pass run with its own "
    "schedule, grayness 0.025): 20 passes, schedule as it was (ends at beta 2): x0.791 / x0.834, grayness 0.142 (IT20); 20 passes, schedule "
    "scaled: x0.989 / x0.970, grayness 0.026 (IT20o); 40 passes, default schedule (ends at beta 2): x0.801 / x0.869, grayness 0.138 (BSc).")


def record(k, n):
    """The `continued_from` record of run.json (the keys that the fact reads)."""
    return {"iteration": k, "more_iterations": n, "iterations": k + n}


def test_the_fact_of_a_continuation_names_the_pass_count_and_the_flag_and_offers_a_new_run():
    """The repro: hook_short with 20 passes, then 5 more. The YAML says 20, the run has 25. The numbers are those of the run."""
    [f] = beta_stage_fact(25, HOOK, record(20, 5))
    assert f.startswith("fact: the pass count of this run is 25 (20 + 5 more_iterations); settings.iterations of the YAML is not used. "
                        "The last stage of settings.beta_schedule [[0, 1], [16, 2], [26, 4], [34, 8]] (beta 8) starts at pass 34. ")
    assert "The run ends at pass 25 at beta 2 (the stage from pass 16) and does not reach beta 8: " in f
    assert "a design that has not seen the last beta can stay grey. " in f
    assert (" options: more_iterations 14 or more (the flag --more-iterations; pass 34 only measures; each pass above it is one change of the "
            "design at beta 8); or a new run, not a continuation, with the schedule scaled to its passes (`topopt check` of the YAML with "
            "that settings.iterations gives the beta_schedule). Measured on examples/hook/hook_short.yaml") in f
    assert f.endswith(HOOK_PRICE + ".") and f.index("options:") < f.index("Measured on")      # the price is kept, after the options
    assert "settings.iterations is" not in f and "settings.iterations 34" not in f              # the key is not the one that sets the count
    assert "beta_schedule: [[" not in f and "each start x" not in f                               # no YAML list: a new beta_schedule is a changed input
    assert "\n" not in f


@pytest.mark.parametrize("schedule", [HOOK, DEFAULT])
def test_the_first_option_of_a_continuation_is_the_smallest_more_iterations_that_reaches_the_last_stage(schedule):
    last = schedule[-1][0]
    for k in range(0, last + 3):
        for n in (1, 2, 5, 14, last // 2, last):
            if reaches_last_beta_stage(k + n, schedule):
                assert beta_stage_fact(k + n, schedule, record(k, n)) == []
                continue
            [f] = beta_stage_fact(k + n, schedule, record(k, n))
            want = int(re.search(r"options: more_iterations (\d+) or more", f).group(1))
            assert want == last - k and want > n                                        # N is last - K, and the run asks for more than it has
            assert beta_stage_fact(k + want, schedule, record(k, want)) == []           # that count reaches the last stage
            assert beta_stage_fact(k + want - 1, schedule, record(k, want - 1)) != []   # one less does not (pass `last` is the first that reaches it)


def test_a_normal_run_keeps_its_text_byte_for_byte_and_an_empty_record_is_a_normal_run():
    assert beta_stage_fact(20, HOOK) == [HOOK_20]
    assert beta_stage_fact(20, HOOK, None) == [HOOK_20] and beta_stage_fact(20, HOOK, {}) == [HOOK_20]
    assert CONT not in HOOK_20 and START in HOOK_20


def test_problem_facts_gives_the_text_of_a_continuation_only_with_its_record(tmp_path):
    cfg = load_config(case(tmp_path, 25, HOOK))
    assert [f for f in problem_facts(cfg) if f.startswith(START)] == [beta_stage_fact(25, HOOK)[0]]
    rec = record(20, 5)
    got = problem_facts(cfg, continued_from=rec)
    assert [f for f in got if f.startswith(CONT)] == beta_stage_fact(25, HOOK, rec) and not [f for f in got if f.startswith(START)]
    ok = problem_facts(load_config(case(tmp_path / "ok", 40, HOOK)), continued_from=record(20, 20))      # K + N reaches pass 34: no fact
    assert not [f for f in ok if f.startswith((CONT, START))]


@pytest.fixture(scope="module")
def parents(tmp_path_factory):
    """Two finished parents of 2 passes (1 thread): the beta schedule ends at pass 20, and at pass 3."""
    root = tmp_path_factory.mktemp("cont_fact")
    made = {}
    for name, sched in (("far", [[0, 1], [10, 2], [20, 4]]), ("near", [[0, 1], [3, 2]])):
        d = root / name
        yml = case(d, 2, sched)
        out = root / f"parent-{name}"
        finish(yml, out)
        made[name] = (yml, out)
    return made


def first_pass_line(text):
    return next(i for i, t in enumerate(text.splitlines()) if re.match(r"it +\d+", t))


def test_a_real_continuation_that_ends_before_the_last_stage_logs_the_fact_with_its_own_pass_count(parents, tmp_path):
    """The YAML says 2 passes, the parent has 2, N is 2: the run has 4 passes, the last stage starts at 20."""
    yml, parent = parents["far"]
    out = tmp_path / "new"
    finish(yml, out, "--continue-from", str(parent), "--more-iterations", "2")
    text = log(out)
    lines = text.splitlines()
    [at] = [i for i, t in enumerate(lines) if t.startswith(CONT)]
    assert at < first_pass_line(text) and not [t for t in lines if t.startswith(START)]        # once, before the first pass, no normal line
    rec = json.loads((out / "run.json").read_text())["continued_from"]
    assert rec["iteration"] == 2 and rec["more_iterations"] == 2 and rec["iterations"] == 4
    assert lines[at] == beta_stage_fact(4, [[0, 1], [10, 2], [20, 4]], rec)[0]              # the text of the function, from the record of the run
    assert lines[at].startswith("fact: the pass count of this run is 4 (2 + 2 more_iterations); settings.iterations of the YAML is not used. ")
    assert "The run ends at pass 4 at beta 1 (the stage from pass 0) and does not reach beta 4: " in lines[at]
    assert "options: more_iterations 18 or more (the flag --more-iterations; pass 20 only measures;" in lines[at]
    assert "settings.iterations is" not in lines[at] and "beta_schedule: [[" not in lines[at]
    assert [t for t in (parent / "run.log").read_text().splitlines() if t.startswith(START)]    # the parent had the normal line, and is not changed
    assert json.loads((out / "status.json").read_text())["state"] == "done"


def test_a_real_continuation_that_reaches_the_last_stage_logs_no_fact(parents, tmp_path):
    yml, parent = parents["near"]
    out = tmp_path / "new"
    finish(yml, out, "--continue-from", str(parent), "--more-iterations", "2")                  # 4 passes, the last stage starts at pass 3
    assert CONT not in log(out) and START not in log(out)


def test_the_resume_of_a_continuation_logs_the_same_fact_again(parents, tmp_path):
    """A continuation that is killed after pass 3 and resumed with the same YAML (2 passes) takes the count 4 from its checkpoint."""
    yml, parent = parents["far"]
    out = tmp_path / "new"
    killed(yml, out, "after_checkpoint:3", "--continue-from", str(parent), "--more-iterations", "2")
    resume(yml, out)
    mine = [t for t in log(out).splitlines() if t.startswith(CONT)]
    assert len(mine) == 2 and mine[0] == mine[1]                                                 # the killed part and the resumed part
    assert mine[0].startswith("fact: the pass count of this run is 4 (2 + 2 more_iterations);") and START not in log(out)
