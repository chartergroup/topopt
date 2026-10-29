"""#380: `run_result` says in one plain sentence that a run stopped at its pass limit (key `end_fact`, CONTRACTS.md section 10).

A client that read only the plugin texts must be able to say "the run stopped at its pass limit" and not "not converged"; the
instructions of the server have room for the rule in a short form only, and this sentence is the place where the client reads it in full.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("mcp")

from topopt_mcp import server  # noqa: E402


def fake_done_run(folder, verdict):
    """A run folder in the state done: status.json, result/metrics.json (with `verdict` when given) and result/final.stl."""
    (folder / "result").mkdir(parents=True)
    (folder / "status.json").write_text(json.dumps({"state": "done"}))
    metrics = {"mass_final_g": 1.0}
    if verdict is not None:
        metrics["verdict"] = verdict
    (folder / "result" / "metrics.json").write_text(json.dumps(metrics))
    (folder / "result" / "final.stl").write_bytes(b"x")
    return folder


def verdict(state, **more):
    return {"state": state, "final_iteration": 40, "iterations": 40, "converged": False, "feasible": True, **more}


def test_a_run_at_its_pass_limit_gets_the_plain_sentence(tmp_path):
    res = server.run_result(str(fake_done_run(tmp_path, verdict("iteration_limit"))), preview=False)
    assert res["done"] is True
    assert res["end_fact"] == ("The run stopped at its pass limit (pass 40 of 40). "
                               "The objective had settled: no. Every constraint row is met: yes.")
    assert "must" not in res["end_fact"].lower()                      # a fact, not an order


@pytest.mark.parametrize("enabled", [False, True])
def test_the_sentence_names_no_stop_rule_and_is_the_result_line_of_the_part_card(tmp_path, enabled):
    """Review F1 of #380: the stop rules are off by default, so the sentence must not suggest that a rule was there and failed. One writer
    (part_card.result_sentence) gives the part card and run_result the same words."""
    from topopt import part_card
    v = verdict("iteration_limit", stop_rules={"enabled": enabled}, stop=None)
    res = server.run_result(str(fake_done_run(tmp_path, v)), preview=False)
    assert "stop rule" not in res["end_fact"]
    assert res["end_fact"] == part_card.result_sentence(v)


def test_a_test_that_is_not_known_is_said_so(tmp_path):
    res = server.run_result(str(fake_done_run(tmp_path, verdict("iteration_limit", converged=None, feasible=None))), preview=False)
    assert res["end_fact"].endswith("The objective had settled: not known. Every constraint row is met: not known.")


@pytest.mark.parametrize("state", ["converged", "stopped_by_rule", "stopped_by_request", "unfinished"])
def test_a_run_in_another_state_has_no_such_key(tmp_path, state):
    res = server.run_result(str(fake_done_run(tmp_path, verdict(state))), preview=False)
    assert res["done"] is True and "end_fact" not in res


def test_a_run_with_no_verdict_has_no_such_key(tmp_path):
    res = server.run_result(str(fake_done_run(tmp_path, None)), preview=False)
    assert res["done"] is True and "end_fact" not in res


def test_the_key_is_in_the_contract():
    text = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    assert "`end_fact`" in text and "only when `metrics.verdict.state` is `iteration_limit`" in text
