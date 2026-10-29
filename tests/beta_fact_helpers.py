"""The fact of P7-4 (F4-18, `verdict.beta_stage_fact`) for the tests of other facts.

The short fixtures of many tests end before the last stage of the default beta schedule, so `optimize.problem_facts` has the line of this
fact in their lists. A test of another fact reads the list without it.
"""
from topopt.optimize import problem_facts
from topopt.verdict import beta_stage_fact

START = "fact: settings.iterations is "      # the start of the line of the beta-stage fact


def problem_facts_without_beta_fact(cfg) -> list[str]:
    """`problem_facts(cfg)` without the line of the beta-stage fact. That line is taken out once, and only when it is due: when it is due
    and `problem_facts` does not have it exactly once, or another line starts like it, the helper fails."""
    facts = list(problem_facts(cfg))
    due = beta_stage_fact(cfg.settings.iterations, cfg.settings.beta_schedule)
    if due:
        assert facts.count(due[0]) == 1, "the beta-stage fact is due, and problem_facts does not have its line exactly once"
        facts.remove(due[0])
    assert not [f for f in facts if f.startswith(START)], "a line that starts like the beta-stage fact is left"
    return facts
