"""Lever cards: the reference text and the lookup (CONTRACTS.md section 41).

The table is in `topopt/levers_cards.py` (data only). This module makes the text from it. The MCP resource `topopt://levers` and the
command `topopt levers` print the text of the same function, `reference`. The module needs only the standard library and `topopt.config`.
"""
from __future__ import annotations

from .config import LEVERS_RESOURCE
from .levers_cards import (CARDS, CONFIDENCE, FACT_SOURCES, GROUPS, KINDS, LAYER_COSTS, LAYERS, PART_CLASSES, PRICE_TEXT, READ_ONLY_TOOLS,
                          Card)

URI = LEVERS_RESOURCE
NAME = "levers"
TITLE = "Levers: YAML keys and engine tools"
DESCRIPTION = ("Read-only. One card for each YAML key and each engine tool: what it does, who decides (user requirement or method "
               "setting), the cost of one try (layer), the words a user says and the facts in the result that detect them, and the "
               "measured evidence. Generated from topopt/levers_cards.py. The tools give facts and options: you decide.")

# The template resource: the cards that match one key path, tool name or word (the same text as `topopt levers <query>`).
TEMPLATE = URI + "/{query}"
TEMPLATE_NAME = "levers_cards"
TEMPLATE_TITLE = "Levers: the cards for one key, tool or word"
TEMPLATE_DESCRIPTION = ("Read-only. The cards of topopt://levers that match one query: a key path (settings.penal), a tool name (verify_run) or a "
                        "word or phrase of the user (film, too heavy). The same text as `topopt levers <query>`. The full text is topopt://levers "
                        "(over 100 KB); this one is short.")

LAYER_TEXT = {
    1: "a global setting: a new run. The cost is the run time of the problem (`check_problem` gives `derived.time_estimate_s`). "
       "Call: edit the key in the YAML, `check_problem`, then `run_topopt` with a new `out_dir`.",
    2: "continue or branch a run. The cost is a part of a run. Call: `run_topopt` with `resume=true` on a stopped run; or with `continue_from` and "
       "`more_iterations` and a new `out_dir`, to add passes to a run (a finished run too) in a new folder; or with `fine_from` and a new `out_dir`, "
       "to run a finished run again on a finer voxel.",
    3: "the post step only. The cost is seconds, with no new pass. Call: edit the key, then `run_topopt` with the same `out_dir`, "
       "`resume=true` and `allow_changed_inputs=true`; or the tool that the card names.",
    4: "a local edit or judgement of the finished part. The cost is seconds to minutes (and a verify). Call: the tool that the card names.",
}

LAYER_SHORT = {1: "a new run", 2: "continue or branch a run", 3: "post step only", 4: "local edit or judgement"}

RULE_TEXT = (
    "- `method`: a setting of the method (penal, beta schedule, move limit, solver and tolerances, voxel size, iterations, robust, "
    "mesh method, smoothing). You change it without asking, and you report what, why and the price.\n"
    "- `requirement`: a user requirement (loads, supports, keep-outs, interfaces, material, stress limits and safety factors, a "
    "stated mass or volume target, print limits, symmetry). You may tighten one without asking, with a report and its measured price. "
    "You loosen one only with the user's consent, and you bring the facts.\n"
    "- `both`: a key that can be either follows its origin in the brief record (`record_brief`): origin `user` is a requirement; origin "
    "`inferred` or `default` is yours (a method setting).\n\n"
    "This is guidance. No tool refuses a change because of it. The tools give facts, estimates and options; you decide.")

CONFIDENCE_TEXT = {
    "measured": "a measured row with run ids (the part class is named)",
    "code": "from the code or CONTRACTS.md only; nobody measured the effect on a part",
    "untested": "nobody ran the lever",
}


def block_of(card: Card) -> str:
    """The block of a YAML key path (`settings` for `settings.penal`; `top level` for `name`), or `tools` for a tool."""
    if card.group == "tool":
        return "tools"
    head, dot, _ = card.key.rpartition(".")
    return head if dot else "top level"


def _kind_text(card: Card) -> str:
    return card.kind if not card.kind_note else f"{card.kind} ({card.kind_note})"


def _symptom_line(s) -> str:
    line = f"  - {s.words}: {s.detect}"
    return f"{line.rstrip('.')}. Present when: {s.when}." if s.when else line


def render_card(card: Card) -> str:
    """One card as Markdown."""
    lines = [f"#### {card.key}" + (f"  (also: {', '.join(card.also)})" if card.also else ""),
             f"- Kind: {_kind_text(card)}. Layer {card.layer} ({LAYER_SHORT[card.layer]}). Confidence: {card.confidence}.",
             f"- Effect: {card.effect}"]
    if card.values:
        lines.append(f"- Values: {card.values}")
    if card.cost:
        lines.append(f"- Cost of one try: {card.cost}")
    if card.call:
        lines.append(f"- Call: {card.call}")
    if card.symptoms:
        lines.append("- Symptoms (what the user says; the fact that detects it):")
        lines += [_symptom_line(s) for s in card.symptoms]
    if card.evidence:
        lines.append("- Evidence:")
        lines += [f"  - {e.change}: {e.effect}. Price: {e.price}. Part: {e.part_class}. Runs: {e.runs}." for e in card.evidence]
    if card.price_note:
        lines.append(f"- Reading the price: {card.price_note}")
    return "\n".join(lines)


def symptom_index(cards=CARDS) -> dict[str, list[str]]:
    """User word -> keys of the cards that list it, in the order of the table."""
    index: dict[str, list[str]] = {}
    for card in cards:
        for s in card.symptoms:
            for word in s.words.split(","):
                word = word.strip()
                if word and card.key not in index.setdefault(word, []):
                    index[word].append(card.key)
    return index


def _matches(card: Card, word: str) -> bool:
    """A word matches the key, the keys in `also`, the symptom words and the facts that detect the symptoms, free text and key paths (not the effect)."""
    w = word.strip().lower()
    hay = [card.key, *card.also] + [t for s in card.symptoms for t in (s.words, s.detect, *s.facts)]
    return any(w in h.lower() for h in hay)


def find(queries, cards=CARDS) -> list[Card]:
    """The cards that match any of the words. A word that is exactly a key (or an `also` key) gives that card alone.

    The cards come in the order of the table.
    """
    words = [q for q in ([queries] if isinstance(queries, str) else list(queries)) if q.strip()]
    hit = []
    for word in words:
        exact = [c for c in cards if word.strip().lower() in (c.key.lower(), *(a.lower() for a in c.also))]
        hit += exact or [c for c in cards if _matches(c, word)]
    return [c for c in cards if c in hit]


def _counts(cards) -> tuple[int, int, int]:
    yaml = [c for c in cards if c.group == "yaml"]
    return len(yaml), sum(1 + len(c.also) for c in yaml), len(cards) - len(yaml)


def _header(cards) -> str:
    n_yaml, n_paths, n_tools = _counts(cards)
    layers = "\n".join(f"- Layer {n}: {text}" + (f" On the hook: {LAYER_COSTS[n]}" if n in LAYER_COSTS else "") for n, text in LAYER_TEXT.items())
    conf = "\n".join(f"- `{k}`: {v}" for k, v in CONFIDENCE_TEXT.items())
    used = {e.part_class for c in cards for e in c.evidence}
    parts = "\n".join(f"- {k}: {v}" for k, v in PART_CLASSES.items() if k in used) or "- none yet"
    return (
        f"# {TITLE}\n\n"
        "A lever is a key of the problem YAML or an engine tool that changes or judges a result. Each lever has one card. A card says what "
        "the lever does, who decides, what it costs, which words of the user point to it, which fact in the result detects the symptom, "
        "and what was measured. The user says \"too heavy\" or \"extra film here\"; find the words below, pick the cheapest layer, "
        "compare the result (`compare_runs`) and report the price.\n\n"
        f"This text has {len(cards)} cards: {n_yaml} for YAML keys (they cover {n_paths} key paths) and {n_tools} for engine tools. "
        "Generated from `topopt/levers_cards.py`. The same text is the command `topopt levers`; `topopt levers <key or word>` prints "
        f"the matching cards, and the resource template `{TEMPLATE}` gives the same short text (a key path, a tool name or a word in place "
        "of `query`). A key path writes `[]` for an entry of a list (`load_cases[].weight`).\n\n"
        f"## Who decides: the kind of a lever\n\n{RULE_TEXT}\n\n"
        f"## Layers: the cost of one try\n\n{layers}\n\n"
        f"## Confidence of a card\n\n{conf}\n\n"
        f"## Part classes of the evidence\n\n{parts}\n\n"
        f"## How to read a price\n\n{PRICE_TEXT}\n")


def _tools_note() -> str:
    names = ", ".join(READ_ONLY_TOOLS)
    return ("Tools that only report have no card, because they change and judge nothing: " + names + ". "
            "`run_result` is the source of the facts that the cards name.")


def reference(query=None, cards=CARDS) -> str:
    """The reference text. With no query: all cards. With a key path or a word (or a list of them): the matching cards."""
    if query is not None and (query if not isinstance(query, str) else query.strip()):
        found = find(query, cards)
        words = [query] if isinstance(query, str) else list(query)
        head = f"# Lever cards for: {', '.join(words)}\n\n{len(found)} cards match.\n"
        if not found:
            return head + "\nNo card matches. Options: `topopt levers` prints all cards; a key path such as `settings.penal`, a tool name " \
                          "such as `verify_run`, or a user word such as `film` finds a card.\n"
        return head + "\n" + "\n\n".join(render_card(c) for c in found) + "\n"
    out = [_header(cards)]
    index = symptom_index(cards)
    if index:
        out.append("## Symptom index (user word: cards)\n\n" + "\n".join(f"- {w}: {', '.join(index[w])}" for w in sorted(index)) + "\n")
    for group, title in (("yaml", "YAML keys"), ("tool", "Engine tools")):
        out.append(f"## {title}\n")
        blocks: dict[str, list[Card]] = {}
        for card in cards:
            if card.group == group:
                blocks.setdefault(block_of(card), []).append(card)
        for block, members in blocks.items():
            if group == "yaml":
                out.append(f"### {block}\n")
            out.append("\n\n".join(render_card(c) for c in members) + "\n")
        if group == "tool":
            out.append(_tools_note() + "\n")
    return "\n".join(out)


def problems(cards=CARDS) -> list[str]:
    """Faults of the table itself (a bad field, a repeated key). The levers test asserts that this list is empty."""
    out, seen = [], {}
    tools = {c.key.split(":")[0] for c in cards if c.group == "tool"} | set(READ_ONLY_TOOLS)
    for c in cards:
        where = f"card '{c.key}'"
        if not c.key.strip():
            out.append("a card has an empty key")
        if c.group not in GROUPS:
            out.append(f"{where}: group {c.group!r} is not one of {', '.join(GROUPS)}")
        if c.kind not in KINDS:
            out.append(f"{where}: kind {c.kind!r} is not one of {', '.join(KINDS)}")
        if c.layer not in LAYERS:
            out.append(f"{where}: layer {c.layer!r} is not one of {', '.join(map(str, LAYERS))}")
        if c.confidence not in CONFIDENCE:
            out.append(f"{where}: confidence {c.confidence!r} is not one of {', '.join(CONFIDENCE)}")
        if not c.effect.strip():
            out.append(f"{where}: no effect")
        for e in c.evidence:
            if e.part_class not in PART_CLASSES:
                out.append(f"{where}: part class {e.part_class!r} is not in PART_CLASSES")
        if c.confidence == "measured" and not any(e.runs.strip() and e.part_class.strip() for e in c.evidence):
            out.append(f"{where}: confidence is measured, and no evidence row has run ids and a part class")
        if c.confidence != "measured" and c.evidence:
            out.append(f"{where}: has evidence rows, and the confidence is {c.confidence}")
        if c.group == "tool" and c.also:
            out.append(f"{where}: a tool card has no `also` keys")
        for s in c.symptoms:
            if not s.words.strip() or not s.detect.strip():
                out.append(f"{where}: a symptom has no words or no detecting fact (write 'no fact yet')")
            if s.when and (not s.when.strip() or not s.facts):
                out.append(f"{where}: the symptom {s.words!r} has a `when` text with no facts to read it from")
            for f in s.facts:
                if f != f.strip() or " " in f or "`" in f or not f:
                    out.append(f"{where}: fact path {f!r} has a space, a backtick or an empty part")
                elif f.split(".")[0] not in FACT_SOURCES and f not in tools:
                    out.append(f"{where}: fact path {f!r} starts with neither a source ({', '.join(FACT_SOURCES)}) nor a tool name")
                elif len(set(s.facts)) != len(s.facts):
                    out.append(f"{where}: the fact paths of the symptom {s.words!r} repeat a path")
        for k in (c.key, *c.also):
            if (c.group, k) in seen:
                out.append(f"key {k!r} is in the card '{seen[(c.group, k)]}' and in the card '{c.key}' ({c.group})")
            seen[(c.group, k)] = c.key
    return out
