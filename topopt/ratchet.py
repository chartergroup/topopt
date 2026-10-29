"""The ratchet: keep a candidate edit of a finished part only if the rules of the case pass (CONTRACTS.md section 36).

    python -m topopt.ratchet check   RUN_DIR [--yaml F]                  the rules, what each needs, what is missing; no solve
    python -m topopt.ratchet judge   STL --parent RUN_DIR [--yaml F]     one judge call, no promotion
    python -m topopt.ratchet run     RUN_DIR --source marks:FILE | stl:FOLDER [--root DIR] ...
    python -m topopt.ratchet status  ROOT [--json]
    python -m topopt.ratchet approve ROOT [--id c007] [--decline]        the opt-in gate: promote (or drop) a waiting candidate

One keep rule, one judge, one log, one runner:

- `keep(candidate_facts, parent_facts, rules)` is a pure function over plain dicts. It returns `ACCEPT`, `REJECT` or
  `BASELINE` and the names of the failed rules. It runs no solve and reads no file.
- The judge (`judge_facts`) builds the facts of one candidate with `topopt printcheck` and `topopt verify`.
- The log `ratchet.jsonl` is append-only: one JSON line for each judged candidate, written whole and flushed.
- The runner (`run_ratchet`) judges the start part (the baseline), then each candidate of a source against the current
  parent, and promotes an accepted candidate to a new run-shaped folder `acc_NN`.

No limit is in this file. Every number comes from the case YAML (the block `acceptance`, the load cases and the `print`
block), each with its source text. A rule that is on and has no input is not skipped: it fails with the reason.
The word for the outcome is `decision`: `verdict` is taken (the convergence verdict of the optimizer).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

from .config import ACCEPTANCE_RULES, SOURCE_NOT_RECORDED, AcceptanceRule, Config, UserError, load_config
from .ratchet_sources import Candidate, CandidateSource, parse_source  # noqa: F401  (Candidate is re-exported)
from .runinfo import HistoryError, peak_rss_bytes, read_history, region_files, run_yaml_path, sha256_file, write_atomic

FORMAT = 1
LOG_FILE = "ratchet.jsonl"
STATE_FILE = "ratchet.json"
APPROVE_JOB = "approve_job"          # the job folder of `approve` (the job files of the run stay in the root)
UNLOGGED = "unlogged"                # the folders that no log line names are moved here, never deleted
DECISIONS = ("ACCEPT", "REJECT", "BASELINE")
STATES = ("pass", "fail", "no_input", "off")
OWNER_APPROVAL = ("pending", "given", "declined", "not_required")
EXIT_OK, EXIT_BAD_INPUT, EXIT_COULD_NOT_RUN, EXIT_GATE = 0, 1, 2, 3

Facts = dict   # plain JSON-safe dict: flat keys, a `cases` dict, a `sources` dict and an `errors` dict. No object, no array.


class RatchetError(Exception):
    """A mistake of the caller (not of the user's input): for example two mesh sizes in one comparison."""


# ---------------------------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------------------------

# What each rule needs from the judge. "printcheck" and "verify" are the tools; "mirror" is `mirror_error_mm`.
RULE_NEEDS = {
    "watertight": ("printcheck",), "bodies": ("printcheck",), "loose_island": ("printcheck",), "mass": ("printcheck",),
    "print_gate": ("printcheck",), "stress": ("verify",), "stiffness": ("verify",), "symmetry": ("mirror",),
    "displacement": ("verify",), "alignment": ("verify",), "buckling": ("verify", "buckling"),
}
# The rows of the key `motion` (section 31) that each rule reads: the part of the row name before the first ":".
MOTION_KINDS = {"displacement": ("displacement", "displacement_max", "displacement_min"), "alignment": ("shift", "tilt")}


@dataclass(frozen=True)
class Rules:
    """The rules of one ratchet, with every limit that the YAML gives. `keep` reads this and the facts, nothing else."""
    rules: tuple[AcceptanceRule, ...]
    case_names: tuple[str, ...]
    stress_limits: dict            # case -> (limit in MPa or None, source text)
    max_bodies: int | None = None
    max_bodies_source: str = ""
    mesh_size_mm: float | None = None
    percentile: float | None = None
    owner_gate: bool = False
    has_mirror: bool = False

    def spec(self, name: str) -> AcceptanceRule | None:
        return next((r for r in self.rules if r.rule == name), None)

    def needs(self) -> set[str]:
        """The tools and helpers that the listed rules need."""
        out: set[str] = {"printcheck"}      # the facts of the part (volume, mass, closed, bodies) are always logged
        for r in self.rules:
            out.update(RULE_NEEDS[r.rule])
        return out

    def to_dict(self) -> dict:
        """The rules with every resolved limit: JSON-safe, stable. Its hash tells that the rules changed."""
        return {"rules": [{"rule": r.rule, "options": [[k, list(v) if isinstance(v, tuple) else v] for k, v in r.options],
                           "sources": [list(x) for x in r.sources]} for r in self.rules],
                "case_names": list(self.case_names),
                "stress_limits": {k: list(v) for k, v in self.stress_limits.items()},
                "max_bodies": [self.max_bodies, self.max_bodies_source], "mesh_size_mm": self.mesh_size_mm,
                "percentile": self.percentile, "owner_gate": self.owner_gate, "has_mirror": self.has_mirror}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def make_rules(rules: list, *, case_names=(), stress_limits=None, max_bodies=None, max_bodies_source="",
               mesh_size_mm=None, percentile=None, owner_gate=False, has_mirror=False) -> Rules:
    """Build `Rules` in code (the tests and a caller without a YAML). Each item of `rules` is an `AcceptanceRule` or a dict
    with the key `rule` and the keys of the YAML entry. A number with no `source` has the source "source not recorded"."""
    if not rules:
        raise UserError("no rules: an empty rule set would accept every candidate. Give at least one rule, for example "
                        "{'rule': 'watertight'} (known rules: " + ", ".join(ACCEPTANCE_RULES) + ")")
    out = []
    for item in rules:
        if isinstance(item, AcceptanceRule):
            out.append(item)
            continue
        item = dict(item)
        name = item.pop("rule")
        if name not in ACCEPTANCE_RULES:
            raise UserError(f"unknown rule {name!r} (known rules: {', '.join(ACCEPTANCE_RULES)})")
        wide = item.pop("source", None)
        sources = {k[:-7]: item.pop(k) for k in [k for k in item if k.endswith("_source")]}
        if wide is not None:
            for k in item:
                if k.startswith(("max_", "min_")) or k == "margin":
                    sources.setdefault(k, wide)
        opts = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in item.items())
        out.append(AcceptanceRule(name, opts, tuple(sources.items())))
    return Rules(tuple(out), tuple(case_names), dict(stress_limits or {}), max_bodies, max_bodies_source, mesh_size_mm,
                 percentile, owner_gate, has_mirror)


def rules_from_config(cfg: Config) -> Rules:
    """The rules of a case YAML. Raise `UserError` if the YAML has no `acceptance` block."""
    acc = cfg.acceptance
    if acc is None or not acc.rules:
        raise UserError(f"no rules: the YAML {cfg.yaml_path} has no 'acceptance' block, so the ratchet has nothing to judge "
                        "with. Add one: acceptance: {rules: [{rule: watertight}, ...]} (CONTRACTS.md section 36)")
    limits = {}
    for c in cfg.load_cases:
        lim = c.stress_limit_mpa
        if lim is None:
            limits[c.name] = (None, "no stress limit in the YAML")
        else:
            note = c.allowable_source if c.allowable_source else SOURCE_NOT_RECORDED
            limits[c.name] = (lim, f"load_cases[{c.name}].{c.stress_limit_from} (source: {note})")
    ps = cfg.print_settings
    src = ps.limit_source("max_bodies")
    max_bodies = ps.limit("max_bodies")
    return Rules(acc.rules, tuple(c.name for c in cfg.load_cases), limits, None if max_bodies is None else int(max_bodies),
                 f"print.max_bodies ({'the YAML' if src == 'yaml' else 'an assumed default'})", acc.mesh_size_mm,
                 acc.percentile, acc.owner_gate, bool(cfg.mirror))


# ---------------------------------------------------------------------------------------------------------------
# The keep rule (pure)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    """The result of one rule (for one load case, where the rule is per case)."""
    rule: str
    state: str                       # pass | fail | no_input | off
    reason: str
    case: str | None = None
    value: Any = None
    parent_value: Any = None
    limit: str | None = None         # the limit as text, for example "drop <= 5 %"
    limit_value: Any = None
    limit_source: str | None = None  # where the limit comes from (YAML place and source text)
    input_source: str | None = None  # where the value comes from (tool, file, key)
    form: str | None = None          # the parent form: absolute | not_worse | cap_or_parent | falls | delta
    delta: Any = None                # value minus parent value (in the unit of the value), when both are known
    delta_pct: Any = None            # the same change in percent of the parent value (None if the parent value is 0)

    def to_dict(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None or k in ("value", "parent_value")}


@dataclass(frozen=True)
class Outcome:
    decision: str                    # ACCEPT | REJECT | BASELINE
    failed: tuple[str, ...]          # the rules whose state is fail or no_input, in rule order
    checks: tuple[Check, ...]

    def to_dict(self) -> dict:
        return {"decision": self.decision, "failed": list(self.failed), "checks": [c.to_dict() for c in self.checks]}


def _get(facts: dict | None, *path) -> Any:
    cur = facts
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _src(facts: dict, key: str) -> str | None:
    got = _get(facts, "sources", key)
    return got if isinstance(got, str) else None


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _g(x) -> str:
    return f"{x:.6g}"


def _fin(x):
    """`x` if it is a finite real number (not a bool, not NaN, not inf), else None. The rules read every value through this."""
    return x if _num(x) else None


def _why_missing(facts: dict, step: str, what: str, raw=None) -> str:
    err = _get(facts, "errors", step)
    if err:
        return f"{what} is missing: {step} failed ({err.get('message', 'no message')})"
    if raw is not None:
        return f"{what} is not a finite number: {raw!r}"
    return f"{what} is not in the facts"


def _no_input(rule, facts, step, what, case=None, raw=None, **kw) -> Check:
    return Check(rule, "no_input", _why_missing(facts, step, what, raw), case=case, **kw)


def _off_baseline(rule, case=None, **kw) -> Check:
    return Check(rule, "off", "no parent (the baseline): a rule against the parent has nothing to compare", case=case, **kw)


def _mesh_gap(cand: dict, parent: dict) -> str | None:
    """A reason if the mesh size of one side is not a number: two solved values are comparable only on one known mesh."""
    gone = [who for who, f in (("candidate", cand), ("parent", parent)) if _fin(f.get("mesh_size_mm")) is None]
    if gone:
        return (f"the mesh size of the {' and the '.join(gone)} is not in the facts, so the two solved values cannot be shown to "
                "come from one mesh")
    return None


def _same_mesh(rule: str, cand: dict, parent: dict) -> None:
    a, b = cand.get("mesh_size_mm"), parent.get("mesh_size_mm")
    if a is not None and b is not None and a != b:
        raise RatchetError(f"rule {rule}: the candidate was solved at mesh size {a} mm and the parent at {b} mm. A "
                           "comparison needs one mesh size for both (the judge passes the size of the baseline)")


def _rule_watertight(spec, cand, parent, rules):
    raw = cand.get("watertight")
    v = raw if isinstance(raw, bool) else None            # only true or false is a fact; the text "false" or a number is not
    src = _src(cand, "watertight") or "printcheck:mesh_closed"
    base = dict(value=raw, limit="closed manifold, positive volume", limit_source="definition (printcheck mesh_closed)",
                input_source=src, form="absolute")
    if v is None:
        return [_no_input("watertight", cand, "printcheck", "the closed-mesh fact", **base) if raw is None else
                Check("watertight", "no_input", f"the closed-mesh fact is not true or false: {raw!r}", **base)]
    if v:
        return [Check("watertight", "pass", "the mesh is closed", **base)]
    return [Check("watertight", "fail", f"the mesh is not closed: {cand.get('open_edges')} open edge(s), "
                  f"{cand.get('nonmanifold_edges')} edge(s) with more than 2 faces", **base)]


def _rule_bodies(spec, cand, parent, rules):
    raw, lim = cand.get("bodies"), rules.max_bodies
    n = _fin(raw)
    base = dict(value=raw, limit=None if lim is None else f"<= {lim} bodies", limit_value=lim,
                limit_source=rules.max_bodies_source or None, input_source=_src(cand, "bodies") or "printcheck:bodies.count",
                form="absolute")
    if n is None:
        return [_no_input("bodies", cand, "printcheck", "the body count", raw=raw, **base)]
    if lim is None:
        return [Check("bodies", "no_input", "no max_bodies limit is known", **base)]
    if n <= lim:
        return [Check("bodies", "pass", f"{n} body(ies); the limit is {lim}", **base)]
    return [Check("bodies", "fail", f"{n} bodies; the limit is {lim}", **base)]


def _moved(v, pv):
    """(the change `v - pv`, the change in percent of `pv`). The percent is None if `pv` is 0."""
    d = v - pv
    return d, (100.0 * d / pv if pv else None)


def _pct_text(dp) -> str:
    return "n/a" if dp is None else f"{dp:+.3g} %"


def _rule_loose_island(spec, cand, parent, rules):
    cap, rel = spec.get("max_mm3"), bool(spec.get("relative_to_parent", False))
    raw, praw = cand.get("loose_mm3"), _get(parent, "loose_mm3")
    v, pv = _fin(raw), _fin(praw)
    bound, pbound = bool(cand.get("loose_lower_bound")), bool(_get(parent, "loose_lower_bound"))
    note = (" The report lists the largest bodies only, so this number is a lower bound; the exact sum of all bodies is not "
            "in the facts") if bound else ""
    base = dict(value=raw, parent_value=praw,
                limit=f"{'increase over the parent ' if rel else ''}<= {_g(cap)} mm3",
                limit_value=cap, limit_source=f"acceptance.rules[loose_island].max_mm3 (source: {spec.source_of('max_mm3')})",
                input_source=_src(cand, "loose_mm3") or "printcheck:bodies.largest[1:]", form="delta" if rel else "absolute")
    if v is None:
        return [_no_input("loose_island", cand, "printcheck", "the volume of the loose bodies", raw=raw, **base)]
    if not rel:
        if v > cap:                                    # a lower bound above the limit is a sure fail
            return [Check("loose_island", "fail", f"loose bodies {'at least ' if bound else ''}{_g(v)} mm3"
                          f"{' (a lower bound: the report lists the largest bodies only)' if bound else ''}; the limit is "
                          f"{_g(cap)} mm3", **base)]
        if bound:
            return [Check("loose_island", "no_input", f"loose bodies are at least {_g(v)} mm3, not above the limit "
                          f"{_g(cap)} mm3, but a lower bound is no pass.{note}", **base)]
        return [Check("loose_island", "pass", f"loose bodies {_g(v)} mm3; the limit is {_g(cap)} mm3", **base)]
    if parent is None:
        return [_off_baseline("loose_island", **base)]
    if pv is None:
        return [Check("loose_island", "no_input", "the parent has no loose-body volume", **base)]
    d, dp = _moved(v, pv)
    base.update(delta=d, delta_pct=dp)
    if pbound:
        return [Check("loose_island", "no_input", f"the loose volume of the parent ({_g(pv)} mm3) is a lower bound, so the "
                      "increase over it has no value", **base)]
    if d > cap:
        return [Check("loose_island", "fail", f"loose bodies {'at least ' if bound else ''}{_g(v)} mm3, parent {_g(pv)} mm3, "
                      f"increase {'at least ' if bound else ''}{_g(d)} mm3{' (a lower bound)' if bound else ''}; the limit is "
                      f"{_g(cap)} mm3", **base)]
    if bound:
        return [Check("loose_island", "no_input", f"loose bodies are at least {_g(v)} mm3 (parent {_g(pv)} mm3, increase at "
                      f"least {_g(d)} mm3, not above the limit {_g(cap)} mm3), but a lower bound is no pass.{note}", **base)]
    return [Check("loose_island", "pass", f"loose bodies {_g(v)} mm3, parent {_g(pv)} mm3, increase {_g(d)} mm3; the limit is "
                  f"{_g(cap)} mm3", **base)]


def _cases_of(spec, rules):
    return list(spec.get("cases") or rules.case_names)


def _rule_stress(spec, cand, parent, rules):
    form = spec.get("against_parent", "absolute")
    margin = spec.get("margin")
    tol = spec.get("tolerance_pct")                    # None: strict (no tolerance)
    tsrc = spec.source_of("tolerance_pct")
    out = []
    for case in _cases_of(spec, rules):
        raw, praw = _get(cand, "cases", case, "p_by_volume_MPa"), _get(parent, "cases", case, "p_by_volume_MPa")
        v, pv = _fin(raw), _fin(praw)
        limit, lsrc = rules.stress_limits.get(case, (None, "no stress limit in the YAML"))
        eff = None if limit is None else (limit * margin if margin is not None else limit)
        if limit is not None and margin is not None:
            lsrc += f"; margin {_g(margin)} (source: {spec.source_of('margin')})"
        pct = _get(cand, "percentile")
        base = dict(case=case, value=raw, parent_value=praw, form=form, limit_value=eff, limit_source=lsrc,
                    input_source=_src(cand, f"cases.{case}.p_by_volume_MPa") or
                    f"verify.json:cases.{case}.von_mises.percentile_by_volume_MPa")
        if v is not None and pv is not None:
            base["delta"], base["delta_pct"] = _moved(v, pv)
        if v is None:
            out.append(_no_input("stress", cand, "verify", f"the stress of case {case}", raw=raw, **base))
            continue
        if form != "not_worse" and eff is None:
            out.append(Check("stress", "no_input", f"case {case} has no stress limit in the YAML. Options: add stress_limit, "
                             "or allowable_mpa with allowable_source, to the load case; or use against_parent: not_worse",
                             **base))
            continue
        where = f"p{_g(pct)} by volume" if pct is not None else "the stress by volume"
        if form == "absolute" or (form == "cap_or_parent" and parent is None):
            lim = f"<= {_g(eff)} MPa"
            ok = v <= eff
            out.append(Check("stress", "pass" if ok else "fail",
                             f"{where} {_g(v)} MPa; the limit is {_g(eff)} MPa", **dict(base, limit=lim)))
            continue
        if parent is None:                     # not_worse at the baseline
            out.append(_off_baseline("stress", **base))
            continue
        _same_mesh("stress", cand, parent)
        if pv is None:
            out.append(Check("stress", "no_input", f"the parent has no stress for case {case}", **base))
            continue
        gap = _mesh_gap(cand, parent)
        if gap:
            out.append(Check("stress", "no_input", gap, **base))
            continue
        allowed = pv if tol is None else pv * (1.0 + tol / 100.0)
        moved = _pct_text(base.get("delta_pct"))
        tol_text = ("no tolerance_pct is set, so any rise fails (a new mesh of a changed surface moves the stress a little "
                    "even for a harmless edit; choose a tolerance_pct with a source if that is not wanted)"
                    if tol is None else f"the tolerance is {_g(tol)} % (source: {tsrc})")
        if form == "not_worse":
            ok = v <= allowed
            note = "" if v <= pv else f"; {tol_text}" if not ok or tol is None else f"; within the tolerance: {tol_text}"
            out.append(Check("stress", "pass" if ok else "fail",
                             f"{where} {_g(v)} MPa, parent {_g(pv)} MPa, moved {moved}{note}",
                             **dict(base, limit="<= parent" + ("" if tol is None else f" + {_g(tol)} %"),
                                    limit_value=allowed)))
        else:                                  # cap_or_parent
            cap = max(eff, allowed)
            tail = "" if tol is None else f" + {_g(tol)} % (source: {tsrc})"
            out.append(Check("stress", "pass" if v <= cap else "fail",
                             f"{where} {_g(v)} MPa, parent {_g(pv)} MPa, moved {moved}; the limit is the larger of "
                             f"{_g(eff)} MPa and the parent{tail}",
                             **dict(base, limit=f"<= max({_g(eff)} MPa, parent{'' if tol is None else f' + {_g(tol)} %'})",
                                    limit_value=cap)))
    return out


def _rule_stiffness(spec, cand, parent, rules):
    cap = spec.get("max_drop_pct")
    lsrc = f"acceptance.rules[stiffness].max_drop_pct (source: {spec.source_of('max_drop_pct')})"
    out = []
    for case in _cases_of(spec, rules):
        raw, praw = _get(cand, "cases", case, "stiffness_N_per_mm"), _get(parent, "cases", case, "stiffness_N_per_mm")
        v, pv = _fin(raw), _fin(praw)
        base = dict(case=case, value=raw, parent_value=praw, limit=f"drop <= {_g(cap)} %", limit_value=cap, limit_source=lsrc,
                    input_source=_src(cand, f"cases.{case}.stiffness_N_per_mm") or
                    f"verify.json:cases.{case}.stiffness_N_per_mm", form="delta")
        if v is None:
            out.append(_no_input("stiffness", cand, "verify", f"the stiffness of case {case}", raw=raw, **base))
        elif parent is None:
            out.append(_off_baseline("stiffness", **base))
        elif pv is None or pv <= 0:
            out.append(Check("stiffness", "no_input", f"the parent has no stiffness for case {case}", **base))
        elif v <= 0:
            out.append(Check("stiffness", "no_input", f"the stiffness of case {case} is not positive ({_g(v)} N/mm)", **base))
        elif _mesh_gap(cand, parent):
            out.append(Check("stiffness", "no_input", _mesh_gap(cand, parent) or "", **base))
        else:
            _same_mesh("stiffness", cand, parent)
            drop = 100.0 * (pv - v) / pv
            ok = drop <= cap
            base["delta"], base["delta_pct"] = _moved(v, pv)
            out.append(Check("stiffness", "pass" if ok else "fail",
                             f"stiffness {_g(v)} N/mm, parent {_g(pv)} N/mm, drop {_g(drop)} %; the limit is {_g(cap)} %",
                             **base))
    return out


def _rule_mass(spec, cand, parent, rules):
    raw, praw = cand.get("mass_g"), _get(parent, "mass_g")
    v, pv = _fin(raw), _fin(praw)
    fall = bool(spec.get("must_fall", False))
    cap = spec.get("max_rise_pct")
    base = dict(value=raw, parent_value=praw, input_source=_src(cand, "mass_g") or "printcheck volume x material.density",
                form="falls" if fall else "delta",
                limit="mass falls" if fall else f"rise <= {_g(cap)} %", limit_value=None if fall else cap,
                limit_source="acceptance.rules[mass].must_fall (a named option)" if fall else
                f"acceptance.rules[mass].max_rise_pct (source: {spec.source_of('max_rise_pct')})")
    if v is None:
        return [_no_input("mass", cand, "printcheck", "the mass", raw=raw, **base)]
    if v <= 0:
        return [Check("mass", "no_input", f"the mass is not positive ({_g(v)} g): the STL is empty or inside out", **base)]
    if parent is None:
        return [_off_baseline("mass", **base)]
    if pv is None or pv <= 0:
        return [Check("mass", "no_input", "the parent has no mass", **base)]
    base["delta"], base["delta_pct"] = _moved(v, pv)
    if fall:
        return [Check("mass", "pass" if v < pv else "fail", f"mass {_g(v)} g, parent {_g(pv)} g", **base)]
    rise = 100.0 * (v - pv) / pv
    return [Check("mass", "pass" if rise <= cap else "fail",
                  f"mass {_g(v)} g, parent {_g(pv)} g, rise {_g(rise)} %; the limit is {_g(cap)} %", **base)]


def _rule_print_gate(spec, cand, parent, rules):
    g, unknown = cand.get("print_gate"), cand.get("print_unknown")
    base = dict(value=g, limit="print block limits, strict gate", limit_source="the print block of the YAML (E1a)",
                input_source=_src(cand, "print_gate") or "printcheck:gate.state", form="absolute")
    if g is None:
        return [_no_input("print_gate", cand, "printcheck", "the print gate", **base)]
    if g == "fail":
        return [Check("print_gate", "fail", "the print check fails: " + ", ".join(cand.get("print_failed") or ["?"]), **base)]
    if g != "pass":
        return [Check("print_gate", "no_input", f"the print gate state is {g!r}, not 'pass' (the judge asks for the strict gate)",
                      **base)]
    if not isinstance(unknown, list):
        return [Check("print_gate", "no_input", "the facts do not say which print checks are unknown (print_unknown is missing), "
                      "so a limit that was not checked cannot be excluded", **base)]
    if unknown:
        why = "; ".join(f"{u.get('id')} (limit {u.get('limit')!r}, {u.get('limit_key') or 'no key'}) is unknown: {u.get('reason')}"
                        for u in unknown)
        return [Check("print_gate", "no_input", "a print check that has a limit is unknown, and an unknown check is not a pass. "
                      f"{why}. Options: change the limit, or the part, so that the check can decide; or leave print_gate "
                      "out of the rules", **base)]
    return [Check("print_gate", "pass", "no print check fails, and no check that has a limit is unknown", **base)]


def _bounded(rule, spec, cand, parent, *, case, what, unit, raw, praw, limit, limit_source, input_source, upper, row=None):
    """One check of a value against a bound, in the forms of the stress rule: `absolute`, `not_worse`, `cap_or_parent`.

    `upper`: the value must be at most the bound (displacement, shift, tilt); else at least (the buckling factor). The bound is
    `limit`, the value of the parent moved by `tolerance_pct` (no tolerance: strict), or the looser of the two (`cap_or_parent`).
    `row` is the name of a motion row, shown in the reason.
    """
    form = spec.get("against_parent", "absolute")
    tol = spec.get("tolerance_pct")
    tsrc = spec.source_of("tolerance_pct")
    v, pv = _fin(raw), _fin(praw)
    sign = 1.0 if upper else -1.0
    op = "<=" if upper else ">="
    u = f" {unit}" if unit else ""
    label = f"{what}{' (row ' + row + ')' if row else ''}"
    base = dict(case=case, value=raw, parent_value=praw, form=form, limit_value=limit, limit_source=limit_source,
                input_source=input_source)
    if v is None:
        return _no_input(rule, cand, "verify", f"the {label} of case {case}", raw=raw, **base)
    if pv is not None:
        base["delta"], base["delta_pct"] = _moved(v, pv)
    if form != "not_worse" and limit is None:
        return Check(rule, "no_input", f"the {label} of case {case} has no limit. Options: against_parent: not_worse, or give the limit",
                     **base)
    if form == "absolute" or (form == "cap_or_parent" and parent is None):
        ok = sign * v <= sign * limit
        return Check(rule, "pass" if ok else "fail", f"{label} {_g(v)}{u}; the limit is {op} {_g(limit)}{u}",
                     **dict(base, limit=f"{op} {_g(limit)}{u}"))
    if parent is None:                     # not_worse at the baseline
        return _off_baseline(rule, **base)
    _same_mesh(rule, cand, parent)
    if pv is None:
        return Check(rule, "no_input", f"the parent has no {label} for case {case}", **base)
    gap = _mesh_gap(cand, parent)
    if gap:
        return Check(rule, "no_input", gap, **base)
    allowed = pv + sign * (0.0 if tol is None else abs(pv) * tol / 100.0)
    moved = _pct_text(base.get("delta_pct"))
    tol_text = ("no tolerance_pct is set, so any move to the worse side fails (a new mesh of a changed surface moves the value a "
                "little even for a harmless edit; choose a tolerance_pct with a source if that is not wanted)"
                if tol is None else f"the tolerance is {_g(tol)} % (source: {tsrc})")
    tail = "" if tol is None else f" {'+' if upper else '-'} {_g(tol)} %"
    if form == "not_worse":
        ok = sign * v <= sign * allowed
        note = "" if sign * v <= sign * pv else f"; {tol_text}"
        return Check(rule, "pass" if ok else "fail", f"{label} {_g(v)}{u}, parent {_g(pv)}{u}, moved {moved}{note}",
                     **dict(base, limit=f"{op} parent{tail}", limit_value=allowed))
    cap = max(limit, allowed) if upper else min(limit, allowed)          # cap_or_parent: the looser bound
    return Check(rule, "pass" if sign * v <= sign * cap else "fail",
                 f"{label} {_g(v)}{u}, parent {_g(pv)}{u}, moved {moved}; the limit is the looser of {_g(limit)}{u} and the parent{tail}",
                 **dict(base, limit=f"{op} looser of {_g(limit)} and parent{tail}", limit_value=cap))


def _rule_buckling(spec, cand, parent, rules):
    """The lowest tet buckling load factor of each case (key `buckling` of verify.json, section 33) against `min_factor`."""
    floor = spec.get("min_factor")
    lsrc = (f"acceptance.rules[buckling].min_factor (source: {spec.source_of('min_factor')})" if floor is not None
            else "no min_factor (against_parent: not_worse)")
    out = []
    for case in _cases_of(spec, rules):
        raw, praw = _get(cand, "cases", case, "buckling_lowest_factor"), _get(parent, "cases", case, "buckling_lowest_factor")
        src = _src(cand, f"cases.{case}.buckling_lowest_factor") or f"verify.json:buckling.cases[{case}].load_factors[0]"
        base = dict(case=case, value=raw, form=spec.get("against_parent", "absolute"), limit_value=floor, limit_source=lsrc, input_source=src)
        if _fin(raw) is None and not _get(cand, "errors", "verify"):
            if cand.get("buckling_problem"):
                why = cand["buckling_problem"]
            elif "buckling_lowest_factor" in (_get(cand, "cases", case) or {}):
                why = f"the buckling check found no load factor for case {case}"
            else:
                why = (f"the verify result has no buckling load factor for case {case} (the check did not run for it, or the facts "
                       "come from a verify without the check)")
            out.append(Check("buckling", "no_input", why, **base))
            continue
        out.append(_bounded("buckling", spec, cand, parent, case=case, what="lowest buckling load factor", unit="", raw=raw, praw=praw,
                            limit=floor, limit_source=lsrc, input_source=src, upper=False))
    return out


def _rule_motion(spec, cand, parent, rules):
    """The rows of `limits.displacement` or `limits.alignment` on the tets (key `motion`, section 31), each against its own limit."""
    kinds = MOTION_KINDS[spec.rule]
    cases = set(_cases_of(spec, rules)) if spec.get("cases") else None
    rows = cand.get("motion_rows")
    mine = {n: r for n, r in (rows or {}).items() if r.get("kind") in kinds and (cases is None or r.get("case") in cases)}
    prow = (parent or {}).get("motion_rows") or {}
    if not mine:
        base = dict(limit_source=f"limits.{spec.rule} of the YAML", form=spec.get("against_parent", "absolute"))
        if cand.get("motion_problem"):
            return [Check(spec.rule, "no_input", cand["motion_problem"], **base)]
        err = _get(cand, "errors", "verify")
        if err:
            return [Check(spec.rule, "no_input", f"the rows are missing: verify failed ({err.get('message', 'no message')})", **base)]
        return [Check(spec.rule, "no_input", f"the verify result has no {spec.rule} row" + (" for the listed cases" if cases else "")
                      + f". Option: add limits.{spec.rule} rows to the YAML, or leave the rule out", **base)]
    out = []
    for name, r in mine.items():
        lim = _fin(r.get("limit"))
        out.append(_bounded(spec.rule, spec, cand, parent, case=r.get("case"), what=f"{spec.rule} value", row=name,
                            unit={"tilt": "rad"}.get(r.get("kind"), "mm"), raw=r.get("value"), praw=(prow.get(name) or {}).get("value"),
                            limit=lim, limit_source=f"limits.{spec.rule} row, as verify read it (verify.json:motion.rows[{name}].limit)",
                            input_source=_src(cand, "motion_rows") or "verify.json:motion.rows[].tet_value", upper=True))
    return out


def _rule_symmetry(spec, cand, parent, rules):
    cap = spec.get("max_mm")
    raw = cand.get("mirror_error_mm")
    v = _fin(raw)
    base = dict(value=raw, parent_value=_get(parent, "mirror_error_mm"), limit=f"<= {_g(cap)} mm", limit_value=cap,
                limit_source=f"acceptance.rules[symmetry].max_mm (source: {spec.source_of('max_mm')})",
                input_source=_src(cand, "mirror_error_mm") or "mirror_error_mm", form="absolute")
    if v is None or v < 0:
        return [_no_input("symmetry", cand, "mirror", "the mirror error", raw=raw, **base)]
    return [Check("symmetry", "pass" if v <= cap else "fail",
                  f"the largest distance of a mirror image of the surface to the surface is {_g(v)} mm; the limit is {_g(cap)} mm",
                  **base)]


_RULE_FUNCS = {"watertight": _rule_watertight, "bodies": _rule_bodies, "loose_island": _rule_loose_island,
               "stress": _rule_stress, "stiffness": _rule_stiffness, "mass": _rule_mass, "print_gate": _rule_print_gate,
               "displacement": _rule_motion, "alignment": _rule_motion, "buckling": _rule_buckling,
               "symmetry": _rule_symmetry}


def keep(candidate: Facts, parent: Facts | None, rules: Rules) -> Outcome:
    """Judge the facts of a candidate against the facts of its parent. Pure: no file, no solve.

    `rules` must list at least one rule (else `RatchetError`). Every number that a rule reads must be a finite number: a value that
    is missing, NaN, infinite or of another type is `no_input`, never a pass. `decision` is `BASELINE` when `parent` is None: the facts are recorded, the rules that need a parent are `off`, and
    `failed` lists the absolute rules that the start part fails (a fact; nothing stops). Otherwise it is `ACCEPT` when no
    check failed, else `REJECT`. A rule that is listed in `rules` and has no input (a fact is missing, a tool failed, there is
    no producer) has the state `no_input` and counts as failed, with the reason. A rule that is not listed is off: it makes
    no check. Raises `RatchetError` if a comparison joins two mesh sizes.
    """
    if not rules.rules:
        raise RatchetError("no rules: an empty rule set would accept every candidate. Give at least one rule "
                           "(acceptance.rules in the YAML, CONTRACTS.md section 36)")
    checks: list[Check] = []
    for spec in rules.rules:
        checks.extend(_RULE_FUNCS[spec.rule](spec, candidate, parent, rules))
    failed = tuple(dict.fromkeys(c.rule for c in checks if c.state in ("fail", "no_input")))
    if parent is None:
        decision = "BASELINE"
    else:
        decision = "REJECT" if failed else "ACCEPT"
    return Outcome(decision, failed, tuple(checks))


# ---------------------------------------------------------------------------------------------------------------
# Facts: from the reports of the tools
# ---------------------------------------------------------------------------------------------------------------

def jsonable(x: Any) -> Any:
    """A JSON-safe copy: numpy numbers and arrays become Python values, a path becomes text, a non-finite number becomes None."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        return jsonable(x.tolist())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    item = getattr(x, "item", None)
    if callable(item):
        return jsonable(item())
    return x


def printcheck_facts(report: dict, density_g_cm3: float | None) -> Facts:
    """The facts of a `topopt printcheck` report (run with the strict gate on): closed, bodies, loose bodies, mass."""
    by_id = {c["id"]: c for c in report["checks"]}
    closed, bodies = by_id["mesh_closed"], by_id["bodies"]["value"]
    loose = round(sum(b["volume_mm3"] for b in bodies["largest"][1:]), 3)
    volume = closed["value"]["volume_mm3"]
    src = "printcheck:checks[mesh_closed]"
    # A check that has a limit and whose state is `unknown` was not decided: the strict gate does not fail for it, but it is no pass.
    unknown = [{"id": c["id"], "limit": (c.get("limit") or {}).get("value"), "limit_key": (c.get("limit") or {}).get("key"),
                "reason": c.get("reason")}
               for c in report["checks"] if c.get("state") == "unknown" and (c.get("limit") or {}).get("value") is not None]
    facts: Facts = {
        "watertight": closed["state"] == "pass", "open_edges": closed["value"]["open_edges"],
        "nonmanifold_edges": closed["value"]["nonmanifold_edges"], "volume_mm3": volume,
        "bodies": bodies["count"], "loose_mm3": loose, "loose_lower_bound": bool(bodies["count"] > len(bodies["largest"])),
        "closed_voids": by_id["closed_voids"]["value"]["count"],
        "print_gate": report["gate"]["state"], "print_failed": list(report["gate"].get("failed") or []),
        "print_unknown": unknown,
        "sources": {"watertight": f"{src}.state", "volume_mm3": f"{src}.value.volume_mm3",
                    "bodies": "printcheck:checks[bodies].value.count",
                    "loose_mm3": "printcheck:checks[bodies].value.largest[1:].volume_mm3",
                    "print_gate": "printcheck:gate.state"}}
    if density_g_cm3 is not None:
        facts["mass_g"] = volume * density_g_cm3 * 1e-3      # mm3 x g/cm3 -> g
        facts["sources"]["mass_g"] = "printcheck volume_mm3 x material.density (g/cm3) x 1e-3"
    return facts


def verify_facts(report: dict) -> Facts:
    """The facts of a `topopt verify` report: stiffness, compliance, displacement and the stress by volume of each load case."""
    cases, sources = {}, {}
    pct = None
    for name, c in report["cases"].items():
        vm = c["von_mises"]
        pct = vm.get("percentile", pct)
        cases[name] = {"stiffness_N_per_mm": c.get("stiffness_N_per_mm"), "compliance_Nmm": c.get("compliance_Nmm"),
                       "max_displacement_mm": c.get("max_displacement_mm"),
                       "p_by_volume_MPa": vm.get("percentile_by_volume_MPa"), "peak_qp_MPa": vm.get("peak_qp_MPa")}
        base = f"verify.json:cases.{name}"
        sources.update({f"cases.{name}.stiffness_N_per_mm": f"{base}.stiffness_N_per_mm",
                        f"cases.{name}.compliance_Nmm": f"{base}.compliance_Nmm",
                        f"cases.{name}.max_displacement_mm": f"{base}.max_displacement_mm",
                        f"cases.{name}.p_by_volume_MPa": f"{base}.von_mises.percentile_by_volume_MPa",
                        f"cases.{name}.peak_qp_MPa": f"{base}.von_mises.peak_qp_MPa"})
    sources["mesh_size_mm"] = "verify.json:size_mm"
    out: Facts = {"mesh_size_mm": report["size_mm"], "percentile": pct, "cases": cases, "sources": sources}
    bk = report.get("buckling")
    if bk is not None:       # only when verify ran the check; a rule that reads it and finds no key says why (the check was off)
        for rec in bk.get("cases") or []:
            lf = rec.get("load_factors") or []      # the measured value; the projected value of the rigid-body rule is a fact of the gate entry, not read here
            if rec.get("name") in cases:
                cases[rec["name"]]["buckling_lowest_factor"] = lf[0] if lf else None
                sources[f"cases.{rec['name']}.buckling_lowest_factor"] = f"verify.json:buckling.cases[{rec['name']}].load_factors[0]"
        if bk.get("status") != "done":
            out["buckling_problem"] = _record_problem("the buckling check", bk)
    mo = report.get("motion")
    if mo is not None:
        out["motion_rows"] = {r["name"]: {"kind": r["kind"], "case": r["case"], "limit": r["limit"], "value": r["tet_value"]}
                              for r in mo.get("rows") or [] if r.get("name")}
        sources["motion_rows"] = "verify.json:motion.rows[].tet_value (and limit)"
        if mo.get("status") != "done":
            out["motion_problem"] = _record_problem("the motion rows", mo)
    weld = (report.get("mesh") or {}).get("surface_weld_retry")
    if weld:           # a note, not a rule: verify judged the input with its short edges welded (CONTRACTS 14)
        from .verify.report import weld_line
        out["notes"] = [weld_line(weld)]
        sources["notes"] = "verify.json:mesh.surface_weld_retry"
    clean = (report.get("mesh") or {}).get("surface_clean_retry")
    if clean:          # a note, not a rule: verify judged the prepared surface with its short edges cleaned (CONTRACTS 14)
        from .verify.report import clean_line
        out["notes"] = [clean_line(clean)]
        sources["notes"] = "verify.json:mesh.surface_clean_retry"
    shell = (report.get("mesh") or {}).get("surface_shell_retry")
    if shell:          # a note, not a rule: verify judged the surface without the shell that lies in the material of another (CONTRACTS 14)
        from .verify.report import shell_line
        out["notes"] = [*out.get("notes", []), shell_line(shell)]
        sources["notes"] = "; ".join(filter(None, [sources.get("notes"), "verify.json:mesh.surface_shell_retry"]))
    return out


def _record_problem(what: str, rec: dict) -> str:
    """Why a verify record (`buckling`, `motion`) has no value: its status, reason and message."""
    bits = [str(rec[k]) for k in ("reason", "message") if rec.get(k)]
    return f"{what} did not run in verify (status {rec.get('status')!r}" + (f", {'; '.join(bits)}" if bits else "") + ")"


def _plane_parts(plane) -> tuple[int, float]:
    """(axis index, position in mm) of a mirror plane: a dict with `axis` (x, y, z or 0, 1, 2) and `at_mm`, or an object with those fields."""
    axis = plane.get("axis_index", plane.get("axis")) if isinstance(plane, dict) else getattr(plane, "axis")
    at = plane["at_mm"] if isinstance(plane, dict) else getattr(plane, "at_mm")
    index = axis if isinstance(axis, int) else "xyz".index(str(axis).lower())
    return int(index), float(at)


def mirror_error_mm(stl, planes) -> float:
    """The largest distance from the mirror image of a vertex of the surface to the surface, over the planes (mm).

    A part that is symmetric about the plane gives a small number (the triangles of the two halves are not mirror images,
    so the number is not zero: it is of the order of the sagitta of the triangles). A part with an extra feature on one side
    gives about the size of the feature. `planes` is a list of dicts with `axis` (x, y, z or 0, 1, 2) and `at_mm`.
    """
    import numpy as np
    import trimesh
    if not planes:
        raise UserError("no mirror plane: the mirror error needs at least one plane (the 'mirror' block of the YAML, or "
                        "the planes in run.json)")
    mesh: Any = trimesh.load(str(stl), file_type="stl", force="mesh", process=False)
    pts = np.asarray(mesh.vertices, dtype=float)
    worst = 0.0
    for plane in planes:
        axis, at = _plane_parts(plane)
        image = pts.copy()
        image[:, axis] = 2.0 * at - image[:, axis]
        _, dist, _ = trimesh.proximity.closest_point(mesh, image)
        worst = max(worst, float(np.max(dist)))
    return worst


# ---------------------------------------------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class JudgeContext:
    """What the judge needs besides the folders. A test passes a small one to a fake judge."""
    yaml_path: Path | None = None
    needs: frozenset = frozenset({"printcheck"})
    density_g_cm3: float | None = None
    mesh_size_mm: float | None = None          # the size of the baseline; None for the baseline itself (verify picks it)
    percentile: float | None = None            # None: the default of verify
    mirror_planes: tuple = ()
    # The judge does not use these blocks: the load sensitivity, the buckling check (`buckling.check` of the YAML would
    # otherwise turn it on for each candidate) and the bore checks (the `bore_checks` block of the YAML would turn them on).
    # The buckling check runs only when a rule `buckling` is listed (`needs` has "buckling"): `judge_facts` then sets it on.
    verify_options: dict = field(default_factory=lambda: {"load_sensitivity": False, "buckling": False, "bore_checks": False})

    @classmethod
    def from_config(cls, cfg: Config, rules: Rules, run: dict | None = None) -> "JudgeContext":
        planes: tuple = ()
        if "mirror" in rules.needs():
            from .apply_marks import _planes_of_run
            found, _notes = _planes_of_run(run or {})
            planes = tuple(found) if found else tuple({"axis": p.axis, "at_mm": p.at_mm} for p in cfg.mirror)
        return cls(Path(cfg.yaml_path), frozenset(rules.needs()), cfg.material.density, rules.mesh_size_mm,
                   rules.percentile, planes)


def _step_error(exc: BaseException) -> dict[str, Any]:
    """The record of a tool that failed: an exit code like the tool's own (1 bad input, 2 could not run) and the message."""
    from .verify.errors import VerifyError
    if isinstance(exc, UserError):
        out = {"exit_code": 1, "message": str(exc)}
    elif isinstance(exc, VerifyError):
        out = {"exit_code": 2, "message": str(exc)}
    else:
        out = {"exit_code": 2, "message": f"internal error: {type(exc).__name__}: {exc}"}
    facts = getattr(exc, "facts", None)
    if isinstance(facts, dict):
        out["facts"] = jsonable(facts)
    return out


def _exact_loose(stl: Path, facts: Facts) -> None:
    """The report of printcheck lists the largest bodies only, so with more bodies `loose_mm3` is a lower bound. Add up all the
    bodies from the mesh health of printcheck. If that fails, the bound stays and `loose_lower_bound` stays true (the rule is
    then `no_input` unless the bound is already above the limit)."""
    from . import printcheck
    try:
        health = printcheck.mesh_health(*printcheck.load_stl_float32(stl))
        facts["loose_mm3"] = round(sum(b["volume_mm3"] for b in health["bodies"][1:]), 3)
        facts["loose_lower_bound"] = False
        facts["sources"]["loose_mm3"] = "printcheck mesh_health bodies[1:].volume_mm3 (all bodies)"
    except Exception as exc:       # keep the lower bound, say why
        facts.setdefault("notes", []).append(f"the volume of all loose bodies could not be added up: {type(exc).__name__}: {exc}")


def judge_facts(shadow_dir, parent_dir, ctx: JudgeContext) -> Facts:
    """The facts of one candidate. `shadow_dir` is a run-shaped folder (run.json and result/final.stl).

    It runs `printcheck` on the STL and, when a rule needs it, `verify` on the folder at the mesh size of the baseline
    (`ctx.mesh_size_mm`), and `mirror_error_mm` when the symmetry rule is on. A tool that fails does not raise: the fact
    `errors[step]` holds the exit code and the message, and the rules that need the step get `no_input`.
    The facts also hold `tools`, `timing_s` and `peak_rss_bytes` (the runner moves them into the `judge` block of the record).
    """
    from . import printcheck
    shadow = Path(shadow_dir)
    stl = shadow / "result" / "final.stl"
    facts: Facts = {"errors": {}, "sources": {}}
    tools: dict = {}
    timing: dict = {}

    t0 = time.perf_counter()
    try:
        facts.update(printcheck_facts(printcheck.check(stl, yaml_path=ctx.yaml_path, strict=True), ctx.density_g_cm3))
        tools["printcheck"] = "exit 0"
        if facts.get("loose_lower_bound"):
            _exact_loose(stl, facts)
    except KeyboardInterrupt:
        raise
    except Exception as exc:       # a tool failure is a fact, not a crash
        facts["errors"]["printcheck"] = err = _step_error(exc)
        tools["printcheck"] = f"exit {err['exit_code']}"
    timing["printcheck"] = round(time.perf_counter() - t0, 2)

    if "verify" in ctx.needs:
        from .verify.run import verify_run
        t0 = time.perf_counter()
        kw = dict(ctx.verify_options)
        if "buckling" in ctx.needs:      # a rule `buckling` is on: verify runs the check for the judge (not `buckling.check` of the YAML)
            kw["buckling"] = True
        if ctx.mesh_size_mm is not None:
            kw["mesh_size"] = ctx.mesh_size_mm
        if ctx.percentile is not None:
            kw["percentile"] = ctx.percentile
        try:
            rep = verify_run(shadow, yaml_path=str(ctx.yaml_path), **kw)
            vf = verify_facts(rep)
            facts["sources"].update(vf.pop("sources"))
            if "notes" in vf:
                facts.setdefault("notes", []).extend(vf.pop("notes"))
            facts.update(vf)
            tools["verify"] = f"exit {rep.get('exit_code', 0)}"
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            facts["errors"]["verify"] = err = _step_error(exc)
            tools["verify"] = f"exit {err['exit_code']}"
        timing["verify"] = round(time.perf_counter() - t0, 2)
    if "mirror" in ctx.needs:
        t0 = time.perf_counter()
        try:
            facts["mirror_error_mm"] = mirror_error_mm(stl, ctx.mirror_planes)
            facts["sources"]["mirror_error_mm"] = "mirror_error_mm (reflected vertices to the surface)"
            tools["mirror"] = "ok"
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            facts["errors"]["mirror"] = err = _step_error(exc)
            tools["mirror"] = f"exit {err['exit_code']}"
        timing["mirror"] = round(time.perf_counter() - t0, 2)
    facts["tools"], facts["timing_s"] = tools, timing
    facts["peak_rss_bytes"] = peak_rss_bytes()
    return jsonable(facts)


JudgeFn = Callable[[Path, "Path | None", JudgeContext], Facts]    # judge(shadow_dir, parent_dir, ctx) -> Facts


# ---------------------------------------------------------------------------------------------------------------
# The log, the state and the folders
# ---------------------------------------------------------------------------------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_row(root, row: dict) -> None:
    """Add one record to `ratchet.jsonl`: one whole line, written with `os.write`, then `fsync`.

    If the file ends in a partial line (a process was killed in a write), a newline is written first, so the new record
    starts on its own line and the reader skips only the partial one.
    """
    line = (json.dumps(jsonable(row), separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(Path(root) / LOG_FILE, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            line = b"\n" + line
        view = memoryview(line)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def read_log(root) -> tuple[list[dict], list[str]]:
    """(the records, facts about lines that were skipped). A missing log is no records. A partial last line is skipped."""
    path = Path(root) / LOG_FILE
    if not path.is_file():
        return [], []
    got = read_history(path, strict=False)
    return got.rows, [f.replace("history.jsonl", LOG_FILE) for f in got.facts]


_ACC = re.compile(r"^acc_(\d+)$")


@dataclass
class State:
    """What the log says now. The log is the truth; `ratchet.json` is a copy for a reader that does not parse the log."""
    next_seq: int = 0
    next_acc: int = 0
    parent: str | None = None            # the folder of the current parent, for example "acc_02"
    parent_row: dict | None = None       # the record that made the parent: its facts are the facts of the parent
    baseline: dict | None = None         # the last baseline record that had no tool error
    mesh_size_mm: float | None = None
    pending: list = field(default_factory=list)   # accepted records that wait for the owner gate
    counts: dict = field(default_factory=lambda: {"candidates": 0, "ACCEPT": 0, "REJECT": 0})


def facts_of(row: dict | None) -> Facts | None:
    """The facts of a record, with its errors, as `keep` reads them."""
    if row is None:
        return None
    return dict(row.get("facts") or {}, errors=row.get("errors") or {})


def derive(rows: list[dict]) -> State:
    st = State()
    by_id: dict = {}
    approved: dict = {}
    for r in rows:
        st.next_seq = max(st.next_seq, int(r.get("seq", -1)) + 1)
        kind = r.get("type")
        if kind in ("baseline", "candidate"):
            by_id[r.get("id")] = r
        if kind == "baseline" and not r.get("errors"):
            st.baseline = r
            st.mesh_size_mm = (r.get("facts") or {}).get("mesh_size_mm")
            if r.get("rebase_of"):
                st.parent_row = r        # the parent was judged again: later candidates are compared with this record
        if kind == "candidate":
            st.counts["candidates"] += 1
            st.counts[r.get("decision")] = st.counts.get(r.get("decision"), 0) + 1
        if kind == "approval":
            approved[r.get("id")] = r
        name = r.get("promoted")
        m = _ACC.match(name) if isinstance(name, str) else None
        if m:
            st.next_acc = max(st.next_acc, int(m.group(1)) + 1)
            st.parent = name
            st.parent_row = by_id.get(r.get("id")) if kind == "approval" else r
    st.pending = [r for r in rows if r.get("type") == "candidate" and r.get("owner_approval") == "pending"
                  and r.get("id") not in approved]
    return st


def _move_dir(src: Path, dst: Path) -> None:
    """The move that makes a candidate the new parent. A test replaces this function to simulate a kill."""
    os.replace(src, dst)


def _copy_atomic(src: Path, dst: Path, link: bool = False) -> bool:
    """Copy `src` to `dst` in one step. With `link` a hard link is tried first (the same bytes are stored once; no file of the
    ratchet is changed in place, so the two names are safe); a file system without hard links gets a copy. Return True if linked."""
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    linked = False
    if link:
        try:
            os.link(src, tmp)
            linked = True
        except OSError:
            pass
    if not linked:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return linked


def make_run_folder(dest, start_run, stl, yaml_path, *, info: dict, labels: bool = False, link: bool = False) -> Path:
    """A run-shaped folder: `run.json` (a copy of the start run, with the YAML path made absolute and a `ratchet` key) and
    `result/final.stl`. With `labels` also `labels.bin`. It has no `metrics.json`, so `verify` finds no optimizer
    stiffness to compare with. `verify` and `apply_marks` read this folder as a run.
    `link`: store the STL once, as a hard link to `stl` (the STL of a candidate folder that the root owns)."""
    dest, start_run = Path(dest), Path(start_run)
    run = json.loads((start_run / "run.json").read_text())
    if yaml_path is None:   # a relative stored path starts at the start run, not at `dest` (R2): resolve it there
        yaml_path = run_yaml_path(run, start_run)
    if yaml_path is not None:
        run["yaml_path"] = str(Path(yaml_path).resolve())
    run["ratchet"] = dict(info)
    (dest / "result").mkdir(parents=True, exist_ok=True)
    write_atomic(dest / "run.json", json.dumps(run, indent=1).encode())
    _copy_atomic(Path(stl), dest / "result" / "final.stl", link=link)
    if labels and (start_run / "labels.bin").is_file():
        _copy_atomic(start_run / "labels.bin", dest / "labels.bin")
    return dest


def _row_of_promotion(rows: list[dict], row: dict) -> dict:
    """The candidate record behind a promotion (an approval record points to it by id)."""
    if row.get("type") != "approval":
        return row
    return next(r for r in rows if r.get("type") == "candidate" and r.get("id") == row.get("id"))


def _acc_is_candidate(acc: Path, src: dict) -> str | None:
    """None if the folder `acc` holds the candidate of the record `src`; else the reason it does not."""
    try:
        info = json.loads((acc / "run.json").read_text()).get("ratchet", {})
    except (OSError, ValueError):
        return "its run.json cannot be read"
    if info.get("id") != src.get("id"):
        return f"its run.json names the candidate {info.get('id')!r}, and the record is {src.get('id')!r}"
    want = (src.get("inputs") or {}).get("candidate_sha256")
    final = acc / "result" / "final.stl"
    if want and (not final.is_file() or sha256_file(final) != want):
        return "its result/final.stl is not the candidate of the record (another sha256)"
    return None


def promote(root, start_run, shadow_name: str, acc_name: str, *, with_labels: bool, expect: dict | None = None) -> None:
    """Make the shadow folder the folder `acc_name`: add `labels.bin` (a copy of the start run's), then move the folder.

    Safe to repeat after a kill: if `acc_name` is there and the shadow is gone, the move is done, and only the labels are
    checked. A folder `acc_name` that is there next to the shadow is not the folder of this record, and a folder that holds
    another candidate than `expect` (the record) is not either: `UserError`, nothing is changed. The log line with `promoted`
    is written first.
    """
    root = Path(root)
    acc, shadow = root / acc_name, root / shadow_name
    if acc.is_dir():
        if shadow.is_dir():
            raise UserError(f"{acc_name} is in {root}, and so is {shadow_name}, the folder that the log promotes to it. "
                            f"{acc_name} is not the folder of that record (a folder of an earlier ratchet?). Nothing was "
                            f"changed. Options: move {acc_name} out of the root, then run again (the promotion goes on); "
                            "or name a new root")
        why = _acc_is_candidate(acc, expect) if expect is not None else None
        if why:
            raise UserError(f"{acc_name} in {root} is not the candidate {expect.get('id') if expect else '?'} that the log "
                            f"promoted to it: {why}. Nothing was changed. Options: move {acc_name} out of the root; or name "
                            "a new root")
    target = acc if acc.is_dir() else shadow
    if with_labels and (Path(start_run) / "labels.bin").is_file() and not (target / "labels.bin").is_file():
        _copy_atomic(Path(start_run) / "labels.bin", target / "labels.bin")
    if not acc.is_dir():
        _move_dir(shadow, acc)


def recover(root, start_run, rows: list[dict], *, with_labels: bool) -> list[str]:
    """Finish the promotions that the log holds and the folders do not: a process was killed after the log line.

    Returns what it did, one sentence each. Raises `UserError` if a promoted folder and its shadow are both gone, or if a
    folder `acc_NN` is not the candidate that the log promoted to it.
    """
    done = []
    for r in rows:
        name = r.get("promoted")
        if not name or r.get("type") == "baseline":
            continue
        src = _row_of_promotion(rows, r)
        if (Path(root) / name).is_dir():
            promote(root, start_run, src.get("shadow_dir") or "", name, with_labels=with_labels, expect=src)
            continue
        shadow = src.get("shadow_dir")
        if not shadow or not (Path(root) / shadow).is_dir():
            raise UserError(f"the log says {src.get('id')} was promoted to {name}, but neither {name} nor "
                            f"{shadow or 'its shadow folder'} is in {root}. The folders were removed after the log line")
        promote(root, start_run, shadow, name, with_labels=with_labels, expect=src)
        done.append(f"finished the promotion of {src.get('id')} to {name} (a process stopped after the log line)")
    return done


def move_unlogged(root, names: list[str]) -> list[str]:
    """Move the named folders of the root into `unlogged/<UTC>/`. Nothing is deleted. Return one sentence for each folder."""
    root = Path(root)
    names = [n for n in names if (root / n).exists()]
    if not names:
        return []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest, n = root / UNLOGGED / stamp, 1
    while dest.exists():
        n += 1
        dest = root / UNLOGGED / f"{stamp}-{n}"
    dest.mkdir(parents=True)
    for name in names:
        os.replace(root / name, dest / name)
    return [f"{name} to {UNLOGGED}/{dest.name}/" for name in names]


def drop_unlogged(root, rows: list[dict]) -> list[str]:
    """Move the folders of a candidate that no log line names (a process was killed during the judge) to `unlogged/<UTC>/`.

    Nothing is deleted. The candidate is made and judged again, in a new folder. A `.tmp` file of an interrupted write is
    removed (it holds no result: the only delete in the ratchet). Return one sentence for each folder that was moved.
    """
    logged = {int(r["seq"]) for r in rows if isinstance(r.get("seq"), int)}
    names: list[str] = []
    for p in sorted(Path(root).iterdir()) if Path(root).is_dir() else []:
        m = re.match(r"^(cand|shadow)_(\d+)$", p.name)
        if m and p.is_dir() and int(m.group(2)) not in logged:
            names.append(p.name)
        elif p.name.endswith(".tmp") and p.is_file():
            p.unlink()
    return move_unlogged(root, names)


def unlogged_acc(root, rows: list[dict]) -> list[str]:
    """The `acc_NN` folders of the root that no log line names. They are never used and never changed (a new parent takes the
    next free number). `acc_00` counts when no baseline is logged yet only if the root has a log (it is made again)."""
    named = {r.get("promoted") for r in rows if r.get("promoted")}
    out = []
    for p in sorted(Path(root).iterdir()) if Path(root).is_dir() else []:
        m = _ACC.match(p.name)
        if m and p.is_dir() and p.name not in named and int(m.group(1)) != 0:
            out.append(p.name)
    return out


def free_acc_number(root, first: int) -> int:
    """The first number from `first` on for which no folder `acc_NN` is in the root."""
    n = first
    while (Path(root) / f"acc_{n:02d}").exists():
        n += 1
    return n


def write_state(root, st: State, pins: dict, status: str) -> None:
    pend = [{"id": r.get("id"), "seq": r.get("seq"), "shadow_dir": r.get("shadow_dir")} for r in st.pending]
    data = {"format": FORMAT, "state": status, "parent": st.parent, "next_seq": st.next_seq, "next_acc": st.next_acc,
            "mesh_size_mm": st.mesh_size_mm, "pending": pend, "counts": st.counts, **pins, "updated_at": now_utc()}
    write_atomic(Path(root) / STATE_FILE, json.dumps(jsonable(data), indent=1).encode())


def read_state_file(root) -> dict | None:
    try:
        data = json.loads((Path(root) / STATE_FILE).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


_OWN = ("ratchet.jsonl", "ratchet.json", "ratchet.json.tmp", "marks_in.json", UNLOGGED, APPROVE_JOB)


def _own_entry(name: str) -> bool:
    return (name in _OWN or name.startswith("job.") or re.match(r"^(cand|shadow|acc|earlier)_\d+(\.tmp)?$", name) is not None)


def archive_root(root) -> str:
    """Move the files of an earlier ratchet into `earlier_NN` in the same root (nothing is deleted). Return that folder name.

    The files go into `earlier_NN.tmp` first, and one rename makes the folder `earlier_NN` visible, so a process that is killed in
    the middle leaves a `.tmp` folder that the next call finishes (it does not leave an earlier ratchet split in two).
    """
    root = Path(root)
    tmp = next((p for p in sorted(root.iterdir()) if re.match(r"^earlier_\d+\.tmp$", p.name)), None)
    if tmp is None:
        numbers = [int(m.group(1)) for p in root.iterdir() if (m := re.match(r"^earlier_(\d+)(\.tmp)?$", p.name))]
        tmp = root / f"earlier_{max(numbers, default=0) + 1:02d}.tmp"
        tmp.mkdir()
    for p in sorted(root.iterdir()):
        if p == tmp or p.name.startswith("job.") or re.match(r"^earlier_\d+(\.tmp)?$", p.name):
            continue
        os.replace(p, tmp / p.name)
    final = tmp.with_name(tmp.name[:-4])
    os.replace(tmp, final)
    return final.name


# ---------------------------------------------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------------------------------------------

def _call_judge(judge: JudgeFn, shadow: Path, parent: Path | None, ctx: JudgeContext) -> Facts:
    """Call the judge. A failure of the judge is a fact (`errors.judge`) and a REJECT, not a crash."""
    facts: Any
    try:
        facts = judge(shadow, parent, ctx)
    except KeyboardInterrupt:
        raise
    except RatchetError:
        raise
    except Exception as exc:
        facts = {"errors": {"judge": _step_error(exc)}}
    clean: Facts = jsonable(facts)
    clean["errors"] = clean.get("errors") or {}
    return clean


def _split_facts(facts: Facts) -> tuple[dict, dict]:
    """(the facts that the record keeps in `facts`, the `judge` block: tools, timing and memory)."""
    judge = {"tools": facts.get("tools"), "timing_s": dict(facts.get("timing_s") or {}),
             "peak_rss_bytes": facts.get("peak_rss_bytes")}
    kept = {k: v for k, v in facts.items() if k not in ("tools", "timing_s", "peak_rss_bytes", "errors")}
    return kept, judge


def _default_context(start: Path, rules: Rules, yaml_path, size) -> JudgeContext:
    """The judge context for a caller that gave none: read from the YAML of the run (density, mirror planes). If the YAML
    cannot be read, a small context: the rules that need the density or the planes then get `no_input` with the reason."""
    from .verify.run import read_run
    try:
        info = read_run(start, yaml_path)
        base = JudgeContext.from_config(cast(Config, info.cfg), rules, info.run)
        return dataclasses.replace(base, mesh_size_mm=size)
    except UserError:
        return JudgeContext(yaml_path=Path(yaml_path) if yaml_path else None, needs=frozenset(rules.needs()),
                            mesh_size_mm=size, percentile=rules.percentile)


def _need_parent_folder(start: Path, source: CandidateSource | None) -> list[str]:
    """Why the start run cannot be the parent of a ratchet (v1). An empty list: it can."""
    out = []
    if not (start / "run.json").is_file():
        out.append(f"no run.json in {start}: this is not a run folder")
    if not (start / "result" / "final.stl").is_file():
        out.append(f"no result/final.stl in {start}: the run is not finished")
    if (start / "result" / "nominal.stl").is_file():
        out.append("the run has machining stock (result/nominal.stl). Version 1 of the ratchet does not support it: verify "
                   "reads nominal.stl, and the candidates edit final.stl, so the parent and the candidate would be "
                   "different parts. Option: run the ratchet on a run without stock; support is a follow-up")
    if source is not None and getattr(source, "needs_labels", False) and not (start / "labels.bin").is_file():
        out.append(f"no labels.bin in {start}: the {source.kind} source reads the frozen regions from it")
    return out


def _check_root(root: Path, start: Path) -> None:
    for p in (root, *root.parents):
        if (p / "run.json").is_file() or (p / "status.json").is_file():
            raise UserError(f"the root {root} is inside the run folder {p} (it has run.json or status.json). The candidates "
                            "are written under the root, and apply_marks does not write inside a run folder. Name another root")
    if root == start or start in root.parents:
        raise UserError(f"the root {root} is the start run or inside it. Name a folder beside the run folder")
    if root.exists() and not root.is_dir():
        raise UserError(f"the root {root} is a file")


def _flat(value, prefix="") -> dict:
    """The leaves of a YAML document as `{"load_cases[0].forces[0].force[2]": -5.0}`. A list of scalars is kept per item."""
    out: dict = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(_flat(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = value
    return out


def inputs_digest(yaml_path, start: Path | None = None) -> dict:
    """What the judge reads besides the candidate: the YAML (material, loads, settings, print block), every region file that the
    YAML names, and `run.json` of the start run. The result is JSON-safe: the hashes, and the leaves of the YAML (`values`) so
    that a difference can be named. A YAML that cannot be read is a fact (`error`), not an exception."""
    out: dict = {"yaml_path": None, "yaml_sha256": None, "files": {}, "values": {}, "run_json_sha256": None, "error": None}
    if yaml_path and Path(yaml_path).is_file():
        y = Path(yaml_path).resolve()
        out.update(yaml_path=str(y), yaml_sha256=sha256_file(y))
        try:
            import yaml as _yaml
            out["values"] = jsonable(_flat(_yaml.safe_load(y.read_bytes())))
            cfg = load_config(y)
            files = dict(region_files(cfg.regions, y.parent, yaml_path=y))
            for it in cfg.interfaces:
                if getattr(it, "path", None) is not None:
                    files[f"interface:{it.name}"] = Path(it.path)
            out["files"] = {k: (sha256_file(v) if Path(v).is_file() else "missing") for k, v in sorted(files.items())}
        except UserError as exc:
            out["error"] = str(exc)
        except Exception as exc:          # a YAML that does not parse
            out["error"] = f"{type(exc).__name__}: {exc}"
    if start is not None and (Path(start) / "run.json").is_file():
        out["run_json_sha256"] = sha256_file(Path(start) / "run.json")
    return out


def inputs_sha256(digest: dict | None) -> str | None:
    """One hash of the hashes of a digest (the log keeps it in each record)."""
    if not digest:
        return None
    core = {k: digest.get(k) for k in ("yaml_sha256", "files", "run_json_sha256")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def input_differences(old: dict | None, new: dict | None) -> list[str]:
    """Facts about what changed between two digests: the YAML values, the region files, `run.json`. Empty: the same inputs.
    A digest that is None (nothing pinned) makes no difference."""
    if not old or not new:
        return []
    out: list[str] = []
    if new.get("error"):
        out.append(f"the YAML cannot be read now: {new['error']}")
    if old.get("yaml_sha256") != new.get("yaml_sha256"):
        a, b = old.get("yaml_sha256"), new.get("yaml_sha256")
        out.append(f"the YAML {Path(str(new.get('yaml_path') or old.get('yaml_path'))).name} changed "
                   f"(sha256 {str(a)[:10]} -> {str(b)[:10]})")
        ov, nv = old.get("values") or {}, new.get("values") or {}
        for k in sorted(set(ov) | set(nv)):
            if ov.get(k, "<absent>") != nv.get(k, "<absent>"):
                out.append(f"  {k}: {ov.get(k, '<absent>')!r} -> {nv.get(k, '<absent>')!r}")
    of, nf = old.get("files") or {}, new.get("files") or {}
    for k in sorted(set(of) | set(nf)):
        if of.get(k) != nf.get(k):
            out.append(f"region file {k} " + ("is new" if k not in of else "is gone" if k not in nf else
                                              f"changed (sha256 {str(of[k])[:10]} -> {str(nf[k])[:10]})"))
    if old.get("run_json_sha256") != new.get("run_json_sha256"):
        out.append("run.json of the start run changed")
    return out


def _pins(start: Path, rules: Rules, yaml_path, source: CandidateSource | None, gate: bool) -> dict:
    digest = inputs_digest(yaml_path, start)
    return {"start_run": str(start), "start_final_sha256": sha256_file(start / "result" / "final.stl"),
            "rules_sha256": rules.digest(), "yaml_sha256": digest["yaml_sha256"], "inputs": digest,
            "owner_gate": bool(gate), "source": source.describe() if source is not None else None,
            "with_labels": bool(getattr(source, "needs_labels", False))}


def _inputs(cand_sha, parent_dir: Path, pins: dict, source: CandidateSource | None) -> dict:
    out = {"candidate_sha256": cand_sha, "parent_sha256": sha256_file(parent_dir / "result" / "final.stl"),
           "yaml_sha256": pins.get("yaml_sha256"), "inputs_sha256": inputs_sha256(pins.get("inputs")),
           "rules_sha256": pins["rules_sha256"]}
    desc = source.describe() if source is not None else {}
    if desc.get("sha256"):
        out["marks_sha256"] = desc["sha256"]
    return out


def _logged_inputs(rows: list[dict]) -> dict:
    """The hashes that the last record of the log holds (the log is the truth when the state file is missing)."""
    for r in reversed(rows):
        if r.get("type") in ("baseline", "candidate") and r.get("inputs"):
            return r["inputs"]
    return {}


def _shaped(names: list[str]) -> list[str]:
    """The folders of an earlier ratchet: acc_NN (NN above 0), cand_NNN, shadow_NNN, and an archive that was not finished. A root
    whose log has no finished baseline cannot explain them. `acc_00` alone is a baseline that was killed before its log line
    (it is made again)."""
    out = []
    for n in names:
        m = re.match(r"^(acc|cand|shadow)_(\d+)$", n)
        if (m and not (m.group(1) == "acc" and int(m.group(2)) == 0)) or re.match(r"^earlier_\d+\.tmp$", n):
            out.append(n)
    return out


def _check_pins(root: Path, pins: dict, mesh_size_mm, overwrite: bool, rebaseline: bool, rows: list[dict], say) -> list | None:
    """A root that holds a ratchet goes on with it only if the start part, the rules, the mesh size and the inputs of the judge
    (the YAML, the region files, `run.json`) are the same. Return None when the run goes on as it is, and the list of the
    differences when the run must start with a re-baseline.

    A difference stops the run with the facts and the options: a new root; `rebaseline` (judge the current parent again under the
    inputs of now and go on with the log); `overwrite` (move the files of the root into `earlier_NN` and start again).
    """
    foreign = [p.name for p in root.iterdir() if not _own_entry(p.name)]
    old = read_state_file(root) or {}
    last = _logged_inputs(rows)
    has_log = bool(rows) or (root / STATE_FILE).is_file()
    fatal, diffs = [], []
    if old.get("start_final_sha256") not in (None, pins["start_final_sha256"]):
        fatal.append("the start part (result/final.stl) is not the part of the ratchet that is in this root")
    if old.get("rules_sha256") not in (None, pins["rules_sha256"]):
        diffs.append("the rules (the acceptance block and the limits they read) differ from the rules of the ratchet in this root")
    if mesh_size_mm is not None and old.get("mesh_size_mm") not in (None, mesh_size_mm):
        diffs.append(f"the mesh size {mesh_size_mm:g} mm differs from the size of the baseline in this root "
                     f"({old.get('mesh_size_mm'):g} mm)")
    new_inputs = pins.get("inputs")
    if old.get("inputs"):
        found = input_differences(old["inputs"], new_inputs)
        if found:
            diffs.append("the inputs of the judge changed since the ratchet in this root was started: " + "; ".join(found))
    elif last.get("inputs_sha256") and new_inputs and last["inputs_sha256"] != inputs_sha256(new_inputs):
        diffs.append("the inputs of the judge (the YAML, the region files or run.json) differ from the last record of the log "
                     "(the state file is missing, so the difference cannot be named)")
    elif last.get("yaml_sha256") and pins.get("yaml_sha256") and last["yaml_sha256"] != pins["yaml_sha256"]:
        diffs.append(f"the YAML differs from the YAML of the last record (sha256 {last['yaml_sha256'][:10]} -> "
                     f"{pins['yaml_sha256'][:10]})")
    unexplained = _shaped([p.name for p in root.iterdir()])
    if unexplained and not any(r.get("type") == "baseline" and not r.get("errors") for r in rows):
        fatal.append("the root holds the folders of an earlier ratchet, and its log has no finished baseline "
                     f"({', '.join(unexplained[:5])}). The log was removed, or an archive was not finished")
    if foreign:
        fatal.append(f"the root holds other files ({', '.join(foreign[:5])})")
    if not (fatal or diffs):
        return None
    what = "; ".join(fatal + diffs)
    if overwrite:
        say(f"ratchet: overwrite: moved the earlier files of the root to {archive_root(root)}")
        return None
    if rebaseline and diffs and not fatal and any(r.get("type") == "baseline" and not r.get("errors") for r in rows):
        return diffs
    options = ("name a new root; or overwrite=true (--overwrite) moves what is in the root into earlier_NN and starts again "
               "from the start part (nothing is deleted)")
    if diffs and not fatal and has_log:
        options = ("name a new root; or rebaseline=true (--rebaseline) judges the current parent again under the inputs and "
                   "rules of now and goes on with this log (the earlier records stay as they are); or overwrite=true "
                   "(--overwrite) moves what is in the root into earlier_NN and starts again from the start part "
                   "(nothing is deleted)")
    raise UserError(f"the root {root} cannot be used as it is: {what}. Options: {options}")


def _guard_inputs(pins: dict, start: Path, yaml_path, what: str) -> None:
    """Stop if the inputs of the judge are not the inputs of the pins: a facts-only stop, nothing is logged for the candidate in hand.

    The judge reads the YAML and the region files from disk at every call. A change between two calls would compare a candidate
    that was solved under the new inputs with a parent that was solved under the old ones.
    """
    found = input_differences(pins.get("inputs"), inputs_digest(yaml_path, start))
    if found:
        raise RatchetError(f"the inputs of the judge changed {what}: " + "; ".join(found) + ". Nothing was logged for the "
                           "candidate in hand. Options: put the inputs back and run again with the same root; or run again "
                           "with rebaseline=true (--rebaseline) to judge the current parent again under the new inputs")


def run_ratchet(parent_run, source: CandidateSource, rules: Rules, *, root, judge: JudgeFn = judge_facts,
                ctx: JudgeContext | None = None, yaml_path=None, max_candidates: int | None = None,
                owner_gate: bool | None = None, mesh_size_mm: float | None = None, overwrite: bool = False,
                rebaseline: bool = False, override_lock: bool = False, say=print) -> dict:
    """Judge the start part, then the candidates of `source`, one by one against the current parent. Return a summary.

    `parent_run` is a finished run folder. Nothing in it is written. `root` is the folder of the ratchet: the log, the state,
    the candidates (`cand_NNN`), the judged folders (`shadow_NNN`) and the parents (`acc_NN`, `acc_00` is the start part).
    A second call with the same root goes on where the first one stopped (a killed process, an owner gate, `max_candidates`),
    if the start part, the rules, the mesh size and the inputs of the judge (YAML, region files, run.json) are the same.
    `rebaseline`: when only the rules, the mesh size or the inputs differ, judge the current parent again under the new ones
    (a new baseline record) and go on. `overwrite` moves the files of the root to `earlier_NN` and starts again.
    `owner_gate`: stop before an accepted candidate is promoted (exit code 3); None: as the rules say.
    Raises `UserError` for bad input and `JobHeld` if another ratchet runs in the root (`override_lock` runs anyway).
    The summary has `exit_code`: 0 finished, 2 the baseline could not be judged or the inputs changed during the run,
    3 stopped at the owner gate.
    """
    from .runjob import run_job
    start = Path(parent_run).expanduser().resolve()
    problems = _need_parent_folder(start, source)
    if problems:
        raise UserError("; ".join(problems))
    root = Path(os.path.abspath(Path(root).expanduser()))
    _check_root(root, start)
    gate = rules.owner_gate if owner_gate is None else bool(owner_gate)
    size = mesh_size_mm if mesh_size_mm is not None else rules.mesh_size_mm
    ctx = ctx or _default_context(start, rules, yaml_path, size)
    yaml_path = Path(yaml_path) if yaml_path else ctx.yaml_path
    pins = _pins(start, rules, yaml_path, source, gate)
    summary: dict = {"root": str(root), "accepted": [], "rejected": [], "stopped": None, "pending": None, "notes": []}

    def body(job) -> int:
        rows, _ = read_log(root)
        rebase = _check_pins(root, pins, mesh_size_mm, overwrite, rebaseline, rows, say)
        code = _loop(root, start, source, rules, ctx, judge, pins, yaml_path, max_candidates, gate, size, say, summary,
                     rebase=rebase)
        summary["exit_code"] = code
        return code

    try:
        code = run_job(root, body, success_codes=(EXIT_OK, EXIT_GATE), override=override_lock)
    except BaseException as exc:
        _record_error(root, exc)
        raise
    summary["exit_code"] = code
    rows, _ = read_log(root)
    st = derive(rows)
    summary.update(parent=st.parent, records=len(rows), mesh_size_mm=st.mesh_size_mm, counts=st.counts,
                   log=str(root / LOG_FILE))
    return summary


def _loop(root, start, source, rules, ctx, judge, pins, yaml_path, max_candidates, gate, size, say, summary,
          rebase: list | None = None) -> int:
    root.mkdir(parents=True, exist_ok=True)
    rows, skipped = read_log(root)
    summary["notes"] += skipped
    moved = drop_unlogged(root, rows)
    if rows and not any(r.get("type") == "baseline" and not r.get("errors") for r in rows):
        moved += move_unlogged(root, ["acc_00"])         # an unfinished baseline made it; it is made again
    summary["notes"] += [f"moved the unfinished folder {m} (no log line names it; nothing is deleted)" for m in moved]
    summary["notes"] += recover(root, start, rows, with_labels=bool(source and source.needs_labels))
    stray = unlogged_acc(root, rows)
    if stray:
        summary["unlogged_acc"] = stray
        summary["notes"].append(f"{', '.join(stray)} are in the root and no log line names them (a folder of an earlier ratchet?). "
                                "They are not used and not changed; a new parent takes the next free number")
    st = derive(rows)
    if st.baseline is None:
        code = _baseline(root, start, rules, ctx, judge, pins, yaml_path, source, size, say, summary)
        rows, _ = read_log(root)
        st = derive(rows)
        write_state(root, st, pins, "baseline_failed" if code else "running")
        if code:
            summary["stopped"] = "baseline_failed"
            return code
    elif rebase is not None:
        if st.pending:
            raise UserError(f"{', '.join(r['id'] for r in st.pending)} wait for the owner, and they were judged under the old inputs "
                            "or rules. Decide them first (approve --decline), then run again with rebaseline")
        code = _rebaseline(root, start, st, rules, ctx, judge, pins, yaml_path, size, say, summary, rebase)
        rows, _ = read_log(root)
        st = derive(rows)
        if code:
            summary["stopped"] = "rebaseline_failed"
            return code
        write_state(root, st, pins, "running")
    if st.pending:
        return _stop_for_gate(root, st, pins, summary, say)
    judged = 0
    while max_candidates is None or judged < max_candidates:
        seq = st.next_seq
        cid, cand_dir, shadow = f"c{seq:03d}", root / f"cand_{seq:03d}", root / f"shadow_{seq:03d}"
        parent_dir = root / st.parent
        _guard_inputs(pins, start, yaml_path, "since the ratchet started")
        cand = source.propose(parent_dir, cand_dir, rows)
        if cand is None:
            summary["stopped"] = "no more candidates"
            here = sum(1 for r in rows if r.get("type") == "candidate" and (r.get("source") or {}).get("kind") == source.kind)
            count = source.describe().get("count")
            if judged:        # this call judged candidates: say how many, and that the source has no more
                lead = (f"this call judged {judged} candidate(s), and the source ({count} candidate(s)) has no more for this root: "
                        f"{here} candidate(s) of kind {source.kind} are in the log")
            else:
                lead = (f"the source ({count} candidate(s)) has nothing new for this root: "
                        f"{here} candidate(s) of kind {source.kind} are in the log")
            summary["notes"].append(lead + " with the same name and the same content. "
                                    "A file with the same name and other content is judged as a new candidate")
            break
        cand.id = cid
        judged += 1
        t_all = time.perf_counter()
        inputs = _inputs(sha256_file(cand.stl) if cand.stl and Path(cand.stl).is_file() else None, parent_dir, pins, source)
        parent_facts = facts_of(st.parent_row)
        if cand.stl is None:
            outcome, facts, judge_block = Outcome("REJECT", ("no_candidate",), ()), {"errors": {}}, None
            if cand.error:
                facts["errors"] = {"source": cand.error}
            kept = {}
        else:
            make_run_folder(shadow, start, cand.stl, yaml_path,
                            info={"format": FORMAT, "id": cid, "parent": st.parent, "kind": cand.kind},
                            link=cand.dir is not None and Path(cand.stl).parent == cand_dir)
            facts = _call_judge(judge, shadow, parent_dir, dataclasses.replace(ctx, mesh_size_mm=st.mesh_size_mm))
            _guard_inputs(pins, start, yaml_path, "while the candidate was judged")
            outcome = keep(facts, parent_facts, rules)
            kept, judge_block = _split_facts(facts)
        if judge_block is not None:
            judge_block["timing_s"] = {"source": cand.seconds, **judge_block["timing_s"],
                                       "total": round(time.perf_counter() - t_all + (cand.seconds or 0.0), 2)}
        acc_n = free_acc_number(root, st.next_acc)
        if acc_n != st.next_acc and outcome.decision == "ACCEPT":
            summary["notes"].append(f"acc_{st.next_acc:02d} is in the root and no log line names it, so {cid} takes acc_{acc_n:02d}")
        promoted_name = f"acc_{acc_n:02d}" if outcome.decision == "ACCEPT" and not gate else None
        approval = "not_required" if outcome.decision != "ACCEPT" or not gate else "pending"
        row = {"format": FORMAT, "type": "candidate", "seq": seq, "at": now_utc(), "id": cid,
               "decision": outcome.decision, "failed": list(outcome.failed), "parent": st.parent,
               "candidate_dir": cand_dir.name if cand_dir.is_dir() else None,
               "shadow_dir": shadow.name if shadow.is_dir() else None, "promoted": promoted_name,
               "owner_approval": approval,
               "source": {"kind": cand.kind, "ref": cand.ref, "note": cand.note, "facts": jsonable(cand.facts)},
               "inputs": inputs, "facts": kept, "checks": [c.to_dict() for c in outcome.checks], "judge": judge_block,
               "errors": facts["errors"] or None}
        append_row(root, row)                                  # 1. the log line (with `promoted`)
        if promoted_name:
            promote(root, start, row["shadow_dir"], promoted_name, with_labels=bool(source.needs_labels), expect=row)   # 2. the move
        rows.append(row)
        st = derive(rows)
        write_state(root, st, pins, "running")                 # 3. the state file
        say(f"ratchet: {cid} {outcome.decision}" + (f" ({', '.join(outcome.failed)})" if outcome.failed else "")
            + (f" -> {promoted_name}" if promoted_name else "") + f"  {time.perf_counter() - t_all:.1f} s")
        (summary["accepted"] if outcome.decision == "ACCEPT" else summary["rejected"]).append(cid)
        if approval == "pending":
            return _stop_for_gate(root, st, pins, summary, say)
    else:
        summary["stopped"] = "max_candidates"
    write_state(root, st, pins, "finished")
    return EXIT_OK


def _record_error(root, exc: BaseException) -> None:
    """Keep the text of the error that ended a ratchet call in the job record (`error` in job.json), so a later `status` reader
    (the script or `ratchet_status`) sees why the call stopped, not only the exit code.

    `run_job` has written `job.FAILED` and `job.json` (state `failed`, the exit code) when it re-raises. This adds the key
    `error`. It does nothing for `JobHeld` (the record belongs to the process that holds the lock), for a record that is not
    this process's own failed record, and for an exit that has no message (a simulated kill or `SystemExit`).
    """
    from .runjob import INFO_FILE, JobHeld
    if isinstance(exc, JobHeld):
        return
    if isinstance(exc, KeyboardInterrupt):
        text = "stopped by Ctrl+C"
    elif isinstance(exc, (UserError, RatchetError)):
        text = str(exc)
    elif isinstance(exc, Exception):
        text = f"internal error: {type(exc).__name__}: {exc}"
    else:
        return
    path = Path(root) / INFO_FILE
    try:
        info = json.loads(path.read_text())
        if info.get("state") != "failed" or info.get("pid") != os.getpid():
            return
        write_atomic(path, json.dumps(dict(info, error=text), indent=1).encode())
    except (OSError, ValueError):
        return


def _stop_for_gate(root, st: State, pins, summary, say) -> int:
    ids = [r["id"] for r in st.pending]
    summary["stopped"] = "owner_gate"
    summary["pending"] = ids
    write_state(root, st, pins, "waiting_for_owner")
    say(f"ratchet: {', '.join(ids)} passed the rules and waits for the owner. Nothing is promoted. "
        f"Options: python -m topopt.ratchet approve {root} --id {ids[0]}  (or --decline)")
    return EXIT_GATE


def _baseline(root, start, rules, ctx, judge, pins, yaml_path, source, size, say, summary) -> int:
    rows, _ = read_log(root)
    seq = derive(rows).next_seq
    dest = root / "acc_00"
    summary["notes"] += [f"moved the unfinished folder {m} (no log line names it; nothing is deleted)"
                         for m in move_unlogged(root, ["acc_00"])]      # a copy of the start part from an earlier try
    t_all = time.perf_counter()
    make_run_folder(dest, start, start / "result" / "final.stl", yaml_path,
                    info={"format": FORMAT, "id": f"c{seq:03d}", "parent": None, "kind": "baseline"}, labels=True)
    facts = _call_judge(judge, dest, None, dataclasses.replace(ctx, mesh_size_mm=size))
    _guard_inputs(pins, start, yaml_path, "while the start part was judged")
    outcome = keep(facts, None, rules)
    kept, judge_block = _split_facts(facts)
    judge_block["timing_s"]["total"] = round(time.perf_counter() - t_all, 2)
    failed_tool = bool(facts["errors"])
    row = {"format": FORMAT, "type": "baseline", "seq": seq, "at": now_utc(), "id": f"c{seq:03d}",
           "decision": "BASELINE", "failed": list(outcome.failed), "parent": None, "candidate_dir": None,
           "shadow_dir": None, "promoted": None if failed_tool else "acc_00", "owner_approval": "not_required",
           "source": {"kind": "baseline", "ref": "start part", "note": "", "facts": {}},
           "inputs": {"candidate_sha256": pins["start_final_sha256"], "parent_sha256": None,
                      "yaml_sha256": pins.get("yaml_sha256"), "inputs_sha256": inputs_sha256(pins.get("inputs")),
                      "rules_sha256": pins["rules_sha256"]},
           "facts": kept, "checks": [c.to_dict() for c in outcome.checks], "judge": judge_block,
           "errors": facts["errors"] or None}
    append_row(root, row)
    if failed_tool:
        say("ratchet: the baseline could not be judged: " + "; ".join(f"{k}: {v.get('message')}" for k, v in facts["errors"].items()))
        return EXIT_COULD_NOT_RUN
    say(f"ratchet: baseline judged" + (f" (fails as a fact: {', '.join(outcome.failed)})" if outcome.failed else "")
        + f"  {time.perf_counter() - t_all:.1f} s")
    return EXIT_OK


def _rebaseline(root, start, st: State, rules, ctx, judge, pins, yaml_path, size, say, summary, differences: list) -> int:
    """Judge the current parent again under the inputs, rules and mesh size of now, and log it as a new baseline record.

    The earlier records stay as they are. The later candidates are compared with this record (`derive` takes its facts as the
    facts of the parent). The state file keeps the old pins until the record has no tool error, so a failed re-baseline is
    asked for again.
    """
    rows, _ = read_log(root)
    seq = derive(rows).next_seq
    cid, shadow, parent_dir = f"c{seq:03d}", root / f"shadow_{seq:03d}", root / str(st.parent)
    t_all = time.perf_counter()
    make_run_folder(shadow, start, parent_dir / "result" / "final.stl", yaml_path,
                    info={"format": FORMAT, "id": cid, "parent": st.parent, "kind": "rebaseline"})
    use = size if size is not None else st.mesh_size_mm
    facts = _call_judge(judge, shadow, None, dataclasses.replace(ctx, mesh_size_mm=use))
    _guard_inputs(pins, start, yaml_path, "while the parent was judged again")
    outcome = keep(facts, None, rules)
    kept, judge_block = _split_facts(facts)
    judge_block["timing_s"]["total"] = round(time.perf_counter() - t_all, 2)
    failed_tool = bool(facts["errors"])
    row = {"format": FORMAT, "type": "baseline", "seq": seq, "at": now_utc(), "id": cid, "decision": "BASELINE",
           "failed": list(outcome.failed), "parent": None, "rebase_of": st.parent, "candidate_dir": None,
           "shadow_dir": shadow.name, "promoted": None, "owner_approval": "not_required",
           "source": {"kind": "rebaseline", "ref": f"parent {st.parent}", "note": "the parent was judged again under new inputs",
                      "facts": {"differences": list(differences)}},
           "inputs": {"candidate_sha256": sha256_file(parent_dir / "result" / "final.stl"), "parent_sha256": None,
                      "yaml_sha256": pins.get("yaml_sha256"), "inputs_sha256": inputs_sha256(pins.get("inputs")),
                      "rules_sha256": pins["rules_sha256"]},
           "facts": kept, "checks": [c.to_dict() for c in outcome.checks], "judge": judge_block,
           "errors": facts["errors"] or None}
    append_row(root, row)
    if failed_tool:
        old = {k: v for k, v in (read_state_file(root) or {}).items() if k in _PIN_KEYS}
        write_state(root, derive(read_log(root)[0]), old, "baseline_failed")
        say("ratchet: the parent could not be judged again: " + "; ".join(f"{k}: {v.get('message')}" for k, v in facts["errors"].items()))
        return EXIT_COULD_NOT_RUN
    say(f"ratchet: re-baseline: {st.parent} judged again under the new inputs"
        + (f" (fails as a fact: {', '.join(outcome.failed)})" if outcome.failed else "") + f"  {time.perf_counter() - t_all:.1f} s")
    summary["notes"].append(f"re-baseline of {st.parent} ({cid}): " + "; ".join(differences))
    return EXIT_OK


# ---------------------------------------------------------------------------------------------------------------
# approve, status
# ---------------------------------------------------------------------------------------------------------------

_PIN_KEYS = ("start_run", "start_final_sha256", "rules_sha256", "yaml_sha256", "inputs", "owner_gate", "source", "with_labels")


def approve(root, cid: str | None = None, *, decline: bool = False, say=print) -> dict:
    """The owner gate: promote the accepted candidate that waits (or drop it, with `decline`). Return the facts.

    It finishes a promotion that an earlier call left unfinished (a kill after the approval line), then appends an `approval`
    record (`owner_approval: given` or `declined`) and moves the folder like the runner. An approval needs the same inputs as the
    judge had: if the YAML or a region file changed since the ratchet started, it stops with the facts. A second call finds
    nothing waiting. To go on with the next candidates, run the ratchet again on the same root.
    The job record of the run (`job.json`) is not touched: `approve` has its own job folder, and it stops while a runner lives.
    """
    from .runjob import JobHeld, live_holder, run_job
    root = Path(os.path.abspath(Path(root).expanduser()))
    if not root.is_dir():
        raise UserError(f"the ratchet root {root} does not exist")
    holder = live_holder(root)
    if holder:
        raise JobHeld(holder["message"], holder)
    result: dict = {}

    def body(job) -> int:
        rows, _ = read_log(root)
        state = read_state_file(root) or {}
        start = Path(state.get("start_run", ""))
        pins = {k: v for k, v in state.items() if k in _PIN_KEYS}
        done = recover(root, start, rows, with_labels=bool(state.get("with_labels")))
        st = derive(rows)
        if not st.pending:
            if done:
                last = rows[-1]
                result.update(id=last.get("id"), owner_approval=last.get("owner_approval"), promoted=last.get("promoted"),
                              parent=st.parent, recovered=done)
                write_state(root, st, pins, "stopped_after_approval")
                say(f"ratchet: {done[0]}")
                return EXIT_OK
            raise UserError(f"no candidate waits for the owner in {root}")
        if cid is None and len(st.pending) > 1:
            raise UserError(f"{len(st.pending)} candidates wait ({', '.join(r['id'] for r in st.pending)}): name one with --id")
        row = next((r for r in st.pending if cid in (None, r["id"])), None)
        if row is None:
            raise UserError(f"{cid} does not wait for the owner. Waiting: {', '.join(r['id'] for r in st.pending)}")
        if row.get("parent") != st.parent:
            raise UserError(f"{row['id']} was judged against {row.get('parent')}, and the parent is {st.parent} now: judge it again")
        if not decline and pins.get("inputs"):
            found = input_differences(pins["inputs"], inputs_digest(pins["inputs"].get("yaml_path"), start))
            if found:
                raise UserError(f"{row['id']} was judged under other inputs than the inputs of now: " + "; ".join(found) +
                                ". Options: put the inputs back; or decline it (--decline) and run again with rebaseline")
        name = None if decline else f"acc_{free_acc_number(root, st.next_acc):02d}"
        append_row(root, {"format": FORMAT, "type": "approval", "seq": st.next_seq, "at": now_utc(), "id": row["id"],
                          "decision": row["decision"], "parent": row["parent"], "promoted": name,
                          "shadow_dir": row.get("shadow_dir"), "owner_approval": "declined" if decline else "given"})
        if name:
            promote(root, start, row["shadow_dir"], name, with_labels=bool(state.get("with_labels")), expect=row)
        rows, _ = read_log(root)
        st = derive(rows)
        write_state(root, st, pins, "waiting_for_owner" if st.pending else "stopped_after_approval")
        result.update(id=row["id"], owner_approval="declined" if decline else "given", promoted=name, parent=st.parent)
        say(f"ratchet: {row['id']} {'declined' if decline else 'approved and promoted to ' + str(name)}")
        return EXIT_OK

    run_job(root / APPROVE_JOB, body)
    return result


def _brief(row: dict) -> dict:
    why = [c.get("reason") for c in row.get("checks") or [] if c.get("state") in ("fail", "no_input")]
    out = {k: row.get(k) for k in ("id", "type", "decision", "failed", "parent", "promoted", "owner_approval")}
    out["ref"] = (row.get("source") or {}).get("ref")
    if why:
        out["why"] = why
    if row.get("errors"):
        out["errors"] = {k: v.get("message") for k, v in row["errors"].items()}
    return out


_FINAL_STATES = ("finished", "waiting_for_owner", "baseline_failed", "stopped_after_approval")


def _promotions_open(root: Path, rows: list[dict]) -> list[str]:
    """The folders that the log promotes and the root does not hold yet (a process was killed after the log line)."""
    return [r["promoted"] for r in rows if r.get("promoted") and r.get("type") != "baseline"
            and not (root / r["promoted"]).is_dir()]


def status(root, last: int = 5, starting_pid: int | None = None) -> dict:
    """The state of a ratchet root as facts: counts, the parent, the baseline, the last records, a waiting gate, the job.

    The state is derived from the state file and the job files of the root, for a runner of any kind (the CLI, the MCP server):
    a live job lock means the process runs (a final state in the file waits for the process to end); a file that says `running`
    and no live lock means the process stopped with no final state (`failed`). `starting_pid`: a process that the caller started
    and that has not written its lock yet.
    """
    from .runjob import job_facts
    root = Path(os.path.abspath(Path(root).expanduser()))
    if not root.is_dir():
        raise UserError(f"the ratchet root {root} does not exist")
    rows, skipped = read_log(root)
    st = derive(rows)
    file_state = read_state_file(root) or {}
    job = job_facts(root)
    base = st.baseline or {}
    file_name = file_state.get("state", "none" if not rows else "unknown")
    live = job is not None and job.get("state") == "running"
    pid = job.get("pid") if live else starting_pid
    alive = live or starting_pid is not None
    state, extra = file_name, {}
    summary = (f"{st.counts['candidates']} candidate(s): {st.counts.get('ACCEPT', 0)} accepted, "
               f"{st.counts.get('REJECT', 0)} rejected; parent {st.parent}"
               + (f"; {len(st.pending)} wait for the owner" if st.pending else ""))
    if alive:
        if file_name in ("none", "unknown"):
            state = "starting"
        elif file_name in _FINAL_STATES:
            state, extra["file_state"] = "running", file_name
            summary = (f"the job process is still alive (pid {pid}), so the state is not final yet; the state file says "
                       f"'{file_name}'. " + summary)
    elif job and job.get("state") == "failed" and job.get("exit_code") not in (0, 3, None) and file_name != "baseline_failed":
        # The last process ended with an exit code (a refused start, an error): the file holds the state of an earlier call, and
        # the job record holds the exit code and the text of the error (`error`, written by `_record_error`).
        state, code, err = "failed", job.get("exit_code"), job.get("error")
        extra.update(file_state=file_name, last_error={"exit_code": code, "message": err})
        summary = (f"the last ratchet process ended with exit code {code}: "
                   + (err or "no error text was recorded (the job record has none)")
                   + f". The state file says '{file_name}', the state of an earlier call. Change what the error names, then run it "
                   "again with the same root. " + summary)
    elif (file_name in ("running", "starting")
          or (file_name == "unknown" and job and job.get("state") in ("failed", "stale_lock", "unfinished"))):
        # No exit code: the process was killed (for example for lack of memory), or it was stopped from outside.
        state = "failed"
        summary = ("the last ratchet process stopped with no final state and no exit code (it can have been killed, for "
                   f"example for lack of memory); {(job or {}).get('summary', 'no job record')}. The state file says "
                   f"'{file_name}'. A killed process goes on when it is run again with the same root. " + summary)
        extra["file_state"] = file_name
    open_promotions = _promotions_open(root, rows)
    if open_promotions:
        extra["promotions_open"] = open_promotions
        summary += (f"; the log promotes {', '.join(open_promotions)} and the folder is not in the root yet (a process stopped "
                    "after the log line; the next run or approve finishes it)")
    return {"root": str(root), "state": state, "parent": st.parent,
            "records": len(rows), "counts": st.counts, "mesh_size_mm": st.mesh_size_mm,
            "baseline": {"failed": base.get("failed"), "facts": base.get("facts")} if base else None,
            "pending": [r["id"] for r in st.pending], "last": [_brief(r) for r in rows[-last:]],
            "owner_gate": file_state.get("owner_gate"), "source": file_state.get("source"),
            "job": job, "process_alive": alive, "pid": pid, "log_facts": skipped, "log": str(root / LOG_FILE),
            "summary": summary, **extra}


# ---------------------------------------------------------------------------------------------------------------
# check (no solve), judge (one call)
# ---------------------------------------------------------------------------------------------------------------

def _verify_readiness(start: Path, yaml_path, rules: Rules) -> dict:
    """Can verify run on this run and this machine? Estimates only: nothing is meshed."""
    from .verify.run import preflight
    kw: dict = {}
    if rules.percentile is not None:
        kw["percentile"] = rules.percentile
    if "buckling" in rules.needs():
        kw["buckling"] = True
    try:
        plan = preflight(start, estimate_only=True, yaml_path=yaml_path, mesh_size=rules.mesh_size_mm, **kw)
    except UserError as exc:
        return {"ready": False, "reasons": [str(exc)]}
    why = []
    if plan.chosen is None:
        why.append(str(plan.base["backend"].get("problem") or "no linear solver can run"))
    for name in ("gmsh", "scikit-fem"):
        if plan.base["extras"].get(name, {}).get("state") != "ok":
            why.append(f"{name} is not ready: {plan.base['extras'][name].get('detail') or plan.base['extras'][name].get('state')}")
    est = plan.estimates[0]
    if est["above_limit"]:
        why.append(f"the memory estimate {est['estimate_gb']:.1f} GB is above the limit")
    elif est.get("above_limit_with_buckling"):
        why.append("the memory estimate with the buckling check is above the limit, so verify skips the check and the rule buckling "
                   "has no input")
    return {"ready": not why, "reasons": why, "solver": plan.chosen, "size_mm": est["size_mm"],
            "estimated_seconds": est["estimated_seconds"], "seconds_by_backend": est["seconds_by_backend"],
            "estimate_gb": est["estimate_gb"], "yaml_matches_run": plan.base["yaml_vs_run"]["matches"],
            "yaml_differences": plan.base["yaml_vs_run"]["differences"]}


def _printcheck_readiness(start: Path, yaml_path, rules: Rules, density) -> dict:
    """Run `printcheck` once on the start part (about a second; no solve): can it run, and what does the print gate say?"""
    from . import printcheck
    t0 = time.perf_counter()
    try:
        facts = printcheck_facts(printcheck.check(start / "result" / "final.stl", yaml_path=yaml_path, strict=True), density)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return {"ready": False, "reasons": [f"printcheck cannot run on the start part: {_step_error(exc)['message']}"]}
    return {"ready": True, "reasons": [], "seconds": round(time.perf_counter() - t0, 2), "print_gate": facts["print_gate"],
            "print_failed": facts["print_failed"], "print_unknown": facts["print_unknown"], "bodies": facts["bodies"],
            "loose_lower_bound": facts["loose_lower_bound"]}


def _tolerance_note(spec: AcceptanceRule) -> str | None:
    """A fact for a stress rule that compares with the parent and has no tolerance (no number: the facts are in the plan doc)."""
    if spec.rule != "stress" or spec.get("against_parent", "absolute") == "absolute" or spec.get("tolerance_pct") is not None:
        return None
    return ("no tolerance_pct is set, so any rise of the stress over the parent fails. A new mesh of a changed surface moves "
            "the stress by a little even for a harmless edit, and more when the mesh is coarse. Every record shows delta_pct. Option: tolerance_pct with a "
            "tolerance_pct_source in the rule")


def check_report(run_dir, yaml_path=None, source: str | None = None, root=None) -> dict:
    """Which rules are on, what each needs, and what is missing. No solve, no mesh. Raise `UserError` for a bad input.

    With `source` (`marks:FILE` or `stl:FOLDER`) it also counts the candidates and estimates the verify time.
    With `root` it counts what the log of that root still has to judge.
    """
    from .verify.run import read_run
    start = Path(run_dir).expanduser().resolve()
    src = None
    if source:
        info0 = read_run(start, yaml_path)
        src = parse_source(source, yaml_path=info0.yaml_path)
    problems = _need_parent_folder(start, src)
    info = read_run(start, yaml_path) if (start / "run.json").is_file() and (start / "result" / "final.stl").is_file() else None
    if info is None:
        raise UserError("; ".join(problems))
    rules = rules_from_config(cast(Config, info.cfg))
    ctx = JudgeContext.from_config(cast(Config, info.cfg), rules, info.run)
    needs = rules.needs()
    vstate = _verify_readiness(start, info.yaml_path, rules) if "verify" in needs else None
    pstate = _printcheck_readiness(start, info.yaml_path, rules, cast(Config, info.cfg).material.density)
    rows = []
    for spec in rules.rules:
        missing = []
        notes = [n for n in (_tolerance_note(spec),) if n]
        if "printcheck" in RULE_NEEDS[spec.rule] and not pstate["ready"]:
            missing += pstate["reasons"]
        if spec.rule == "print_gate" and pstate["ready"] and pstate["print_unknown"]:
            missing += [f"the start part: print check {u['id']} is unknown although it has a limit ({u['limit_key']} "
                        f"{u['limit']}): {u['reason']}. Every candidate fails this rule until the check can decide"
                        for u in pstate["print_unknown"]]
        if spec.rule in MOTION_KINDS and not getattr(info.cfg.limits, spec.rule):
            missing.append(f"the YAML has no limits.{spec.rule} rows, so verify has no {spec.rule} value (CONTRACTS.md section 31)")
        if spec.rule == "buckling" and spec.get("against_parent", "absolute") != "not_worse" and spec.get("min_factor") is None:
            missing.append("no min_factor")
        if spec.rule == "stress":
            form = spec.get("against_parent", "absolute")
            for case in _cases_of(spec, rules):
                if form != "not_worse" and rules.stress_limits.get(case, (None,))[0] is None:
                    missing.append(f"case {case} has no stress limit (load_cases[].stress_limit, or allowable_mpa with "
                                   "allowable_source); or use against_parent: not_worse")
        if "verify" in RULE_NEEDS[spec.rule] and vstate is not None and not vstate["ready"]:
            missing += [f"verify cannot run: {r}" for r in vstate["reasons"]]
        if spec.rule == "symmetry" and not ctx.mirror_planes:
            missing.append("no mirror plane was found in run.json or in the YAML")
        if spec.rule == "bodies" and rules.max_bodies is None:
            missing.append("no max_bodies limit")
        rows.append({"rule": spec.rule, "needs": list(RULE_NEEDS[spec.rule]), "ready": not missing, "missing": missing,
                     "notes": notes, "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.options},
                     "sources": dict(spec.sources)})
    out = {"run_dir": str(start), "yaml": str(info.yaml_path), "problems": problems, "rules": rows,
           "off_rules": [r for r in ACCEPTANCE_RULES if rules.spec(r) is None],
           "tools": {"printcheck": pstate, "verify": vstate}, "mesh_size_mm": rules.mesh_size_mm,
           "percentile": rules.percentile, "owner_gate": rules.owner_gate,
           "ready": not problems and all(r["ready"] for r in rows)}
    if src is not None:
        d = src.describe()
        judged, baseline_done = 0, False
        if root is not None and (Path(root) / LOG_FILE).is_file():
            old_rows, _ = read_log(root)
            judged = sum(1 for r in old_rows if r.get("type") == "candidate")
            baseline_done = any(r.get("type") == "baseline" and not r.get("errors") for r in old_rows)
        left = max(0, d["count"] - judged)
        calls = left + (0 if baseline_done else 1)
        each = (vstate or {}).get("estimated_seconds")
        out["estimate"] = {"source": d, "candidates": left, "already_judged_in_root": judged, "judge_calls": calls,
                           "verify_calls": calls if vstate is not None else 0, "verify_seconds_each": each,
                           "verify_seconds_total": None if each is None or vstate is None else each * calls,
                           "note": "the time of verify only (its own estimate at the mesh size of the start part). The time of "
                                   "printcheck and of apply_marks is not in it. The start part is judged first (one call), "
                                   "unless the root already holds its baseline. With a root, the candidates that its log "
                                   "holds are not counted (an upper bound: a source that changed may judge more)."}
    return out


def format_check(rep: dict) -> str:
    lines = [f"ratchet check: {rep['run_dir']}", f"  YAML: {rep['yaml']}"]
    for p in rep["problems"]:
        lines.append(f"  PROBLEM: {p}")
    for r in rep["rules"]:
        lines.append(f"  rule {r['rule']}: {'ready' if r['ready'] else 'MISSING INPUT'} (needs {', '.join(r['needs'])})")
        for m in r["missing"]:
            lines.append(f"      - {m}")
        for n in r.get("notes", []):
            lines.append(f"      note: {n}")
    lines.append(f"  rules that are off: {', '.join(rep['off_rules']) or 'none'}")
    pc = rep["tools"]["printcheck"]
    lines.append(f"  printcheck: {'ready' if pc['ready'] else 'NOT READY'}"
                 + (f" (ran on the start part in {pc['seconds']} s; print gate {pc['print_gate']})" if pc["ready"] else ""))
    for r in pc["reasons"]:
        lines.append(f"      - {r}")
    v = rep["tools"]["verify"]
    if v is not None:
        lines.append(f"  verify: {'ready' if v['ready'] else 'NOT READY'}"
                     + (f" ({v.get('solver')}, mesh {v.get('size_mm', 0):.4g} mm, about {v['estimated_seconds']:.0f} s for each call)"
                        if v.get("estimated_seconds") else ""))
        for r in v["reasons"]:
            lines.append(f"      - {r}")
    else:
        lines.append("  verify: no rule needs it, so it does not run")
    e = rep.get("estimate")
    if e:
        lines.append(f"  candidates: {e['candidates']} ({e['source'].get('kind')}); judge calls: {e['judge_calls']}"
                     + (f"; verify time about {e['verify_seconds_total']:.0f} s" if e["verify_seconds_total"] else ""))
    lines.append(f"  owner gate: {'on (the runner stops before it promotes)' if rep['owner_gate'] else 'off'}")
    lines.append("  ready: " + ("yes" if rep["ready"] else "no (see above; a rule with a missing input fails every candidate)"))
    return "\n".join(lines)


def judge_pair(stl, parent_run, *, out_dir, yaml_path=None, mesh_size_mm=None, judge: JudgeFn = judge_facts,
               ctx: JudgeContext | None = None, say=print) -> dict:
    """One judge call without a runner: judge the parent (the baseline), then the STL against it. No promotion.

    The folders `parent` and `candidate` are made in `out_dir` (a new folder). Return the two outcomes.
    """
    from .verify.run import read_run
    start = Path(parent_run).expanduser().resolve()
    problems = _need_parent_folder(start, None)
    if problems:
        raise UserError("; ".join(problems))
    info = read_run(start, yaml_path)
    rules = rules_from_config(cast(Config, info.cfg))
    ctx = ctx or JudgeContext.from_config(cast(Config, info.cfg), rules, info.run)
    out_dir = Path(os.path.abspath(Path(out_dir).expanduser()))
    _check_root(out_dir, start)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise UserError(f"{out_dir} holds files. Name a new folder with --out")
    out_dir.mkdir(parents=True, exist_ok=True)
    size = mesh_size_mm if mesh_size_mm is not None else rules.mesh_size_mm
    base = make_run_folder(out_dir / "parent", start, start / "result" / "final.stl", info.yaml_path,
                           info={"format": FORMAT, "id": "parent", "kind": "judge"})
    pfacts = _call_judge(judge, base, None, dataclasses.replace(ctx, mesh_size_mm=size))
    cand = make_run_folder(out_dir / "candidate", start, Path(stl), info.yaml_path,
                           info={"format": FORMAT, "id": "candidate", "kind": "judge"})
    cfacts = _call_judge(judge, cand, base, dataclasses.replace(ctx, mesh_size_mm=pfacts.get("mesh_size_mm", size)))
    return {"rules": rules, "parent_facts": pfacts, "candidate_facts": cfacts,
            "baseline": keep(pfacts, None, rules), "outcome": keep(cfacts, pfacts, rules), "out_dir": str(out_dir)}


def format_outcome(label: str, out: Outcome) -> str:
    lines = [f"{label}: {out.decision}" + (f" (failed: {', '.join(out.failed)})" if out.failed else "")]
    for c in out.checks:
        case = f"[{c.case}] " if c.case else ""
        lines.append(f"  {c.state:8s} {c.rule} {case}{c.reason}")
        if c.limit_source and c.state in ("fail", "no_input"):
            lines.append(f"           limit from: {c.limit_source}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------------------------------------------

def _print_json(obj) -> None:
    print(json.dumps(jsonable(obj), indent=1))


def build_rules_for_run(run_dir, yaml_path=None):
    """(info, rules) of a run folder: the YAML is read and the `acceptance` block is turned into rules."""
    from .verify.run import read_run
    info = read_run(run_dir, yaml_path)
    return info, rules_from_config(cast(Config, info.cfg))


def cmd_run(a) -> int:
    start = Path(a.run_dir).expanduser().resolve()
    info, rules = build_rules_for_run(start, a.yaml)
    source = parse_source(a.source, yaml_path=info.yaml_path)
    root = Path(a.root).expanduser() if a.root else start.parent / f"{start.name}-ratchet"
    ctx = JudgeContext.from_config(cast(Config, info.cfg), rules, info.run)
    res = run_ratchet(start, source, rules, root=root, ctx=ctx, yaml_path=info.yaml_path, max_candidates=a.max_candidates,
                      owner_gate=True if a.stop_before_promote else None, mesh_size_mm=a.mesh_size, overwrite=a.overwrite,
                      rebaseline=a.rebaseline, override_lock=a.override_lock)
    for note in res.get("notes", []):
        print(f"ratchet: {note}")
    print(f"ratchet: {res['counts']['candidates']} candidate(s) in the log, parent {res['parent']}, root {res['root']}")
    return res["exit_code"]


def cmd_judge(a) -> int:
    start = Path(a.parent).expanduser().resolve()
    out_dir = Path(a.out).expanduser() if a.out else start.parent / f"{start.name}-judge"
    res = judge_pair(a.stl, start, out_dir=out_dir, yaml_path=a.yaml, mesh_size_mm=a.mesh_size)
    print(format_outcome("parent (baseline)", res["baseline"]))
    print(format_outcome("candidate", res["outcome"]))
    print(f"folders: {res['out_dir']}")
    return EXIT_OK


def cmd_check(a) -> int:
    rep = check_report(a.run_dir, a.yaml, a.source, root=a.root)
    _print_json(rep) if a.json else print(format_check(rep))
    return EXIT_OK


def cmd_status(a) -> int:
    rep = status(a.root)
    if a.json:
        _print_json(rep)
    else:
        print(f"ratchet {rep['root']}: {rep['state']}; {rep['summary']}")
        for r in rep["last"]:
            print(f"  {r['id']} {r['type']} {r['decision']}" + (f" failed: {', '.join(r['failed'])}" if r.get("failed") else "")
                  + (f" -> {r['promoted']}" if r.get("promoted") else "") + (f" [{r['owner_approval']}]" if r.get("owner_approval") == "pending" else ""))
            for w in r.get("why", []):
                print(f"      {w}")
        if rep["job"]:
            print(rep["job"]["summary"])
    return EXIT_OK


def cmd_approve(a) -> int:
    approve(a.root, a.id, decline=a.decline)
    return EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m topopt.ratchet", description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="the rules, what each needs, what is missing (no solve)")
    p.add_argument("run_dir")
    p.add_argument("--yaml", default=None)
    p.add_argument("--source", default=None, help="marks:FILE or stl:FOLDER: also count the candidates and estimate the time")
    p.add_argument("--root", default=None, help="a ratchet root: count only what its log has not judged yet")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("judge", help="one judge call: the STL against the start run, no promotion")
    p.add_argument("stl")
    p.add_argument("--parent", required=True, help="the run folder that the STL is a candidate for")
    p.add_argument("--yaml", default=None)
    p.add_argument("--out", default=None, help="a new folder for the judged folders (default: <run folder>-judge)")
    p.add_argument("--mesh-size", type=float, default=None, help="the mesh size in mm for both (default: as the YAML says, else the size verify picks)")
    p.set_defaults(fn=cmd_judge)
    p = sub.add_parser("run", help="judge the start part, then the candidates of a source; promote what passes")
    p.add_argument("run_dir")
    p.add_argument("--source", required=True, help="marks:FILE (a marks file of the Refine page) or stl:FOLDER")
    p.add_argument("--root", default=None, help="the folder of the ratchet (default: <run folder>-ratchet; an earlier ratchet there goes on)")
    p.add_argument("--yaml", default=None)
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--mesh-size", type=float, default=None)
    p.add_argument("--stop-before-promote", action="store_true", help="the owner gate: stop (exit code 3) before an accepted candidate is promoted")
    p.add_argument("--overwrite", action="store_true", help="move an earlier ratchet of other inputs in the root to earlier_NN and start again")
    p.add_argument("--rebaseline", action="store_true", help="the rules, the mesh size or the inputs (YAML, region files) changed: judge the current parent again under the new ones, keep the log, and go on")
    p.add_argument("--override-lock", action="store_true", help="run although another process holds the job lock of the root (both can damage each other's files)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("status", help="the state of a ratchet root")
    p.add_argument("root")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("approve", help="the owner gate: promote the accepted candidate that waits")
    p.add_argument("root")
    p.add_argument("--id", default=None)
    p.add_argument("--decline", action="store_true", help="drop the waiting candidate instead of promoting it")
    p.set_defaults(fn=cmd_approve)
    a = ap.parse_args(argv)
    try:
        return int(a.fn(a))
    except KeyboardInterrupt:
        print("ratchet: stopped (Ctrl+C). The log is kept; run again with the same root to go on.", file=sys.stderr)
        return 130
    except (UserError, HistoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except RatchetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_COULD_NOT_RUN


if __name__ == "__main__":
    raise SystemExit(main())
