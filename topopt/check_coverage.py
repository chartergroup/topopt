"""Coverage of a report: which sub-checks ran, which failed and which were not run (CONTRACTS.md section 44).

A report of several sub-checks (`topopt verify`, the print report, `topopt printcheck`) writes the failure or the skip of a
sub-check as a fact in that sub-check's own record, and the other sub-checks go on. This module reads those records and builds ONE
summary for the whole report, so that a reader does not have to open every record:

    partial   true when a sub-check that the inputs ask for failed or was not run, else false
    coverage  ran      the names of the sub-checks that ran
              failed   [{name, reason}]  a sub-check that the inputs ask for and that stopped with an error
              not_run  [{name, reason, asked}]  a sub-check that gave no result. `asked` is false when the inputs do not ask for it
                       (for example no bore in the YAML): it is listed, and it does NOT make the report partial

Every sub-check is in exactly one of the three lists. The mappings below (`from_verify`, `from_print_section`, `from_printcheck`)
read the status that each record already has. They do not run a check again and they add no number or limit. A report that is
`partial` is still a report: its other numbers stand. Facts only: nothing here changes an exit code or a gate.
"""
from __future__ import annotations

from dataclasses import dataclass

RAN, FAILED, NOT_RUN = "ran", "failed", "not_run"

# The text for the `definitions` of a report that has the keys (the print report quotes it).
DEFINITION = ("which sub-checks of this report ran, failed or were not run. partial is true when a sub-check that the inputs ask for failed or "
              "did not run: the other numbers still stand. coverage.not_run[].asked is false for a sub-check that the inputs do not ask for "
              "(it does not make the report partial)")


@dataclass(frozen=True)
class Entry:
    """The outcome of one sub-check. `asked` is false for a sub-check that the inputs do not ask for."""
    name: str
    outcome: str                   # RAN, FAILED or NOT_RUN
    asked: bool = True
    reason: str | None = None      # the text for FAILED and NOT_RUN


def build(entries) -> tuple[bool, dict]:
    """`(partial, coverage)` for a list of `Entry`. The order of the lists is the order of the entries."""
    entries = list(entries)
    ran = [e.name for e in entries if e.outcome == RAN]
    failed = [dict(name=e.name, reason=e.reason or "no reason recorded") for e in entries if e.outcome == FAILED]
    not_run = [dict(name=e.name, reason=e.reason or "no reason recorded", asked=bool(e.asked)) for e in entries if e.outcome == NOT_RUN]
    partial = bool(failed) or any(x["asked"] for x in not_run)
    return partial, dict(ran=ran, failed=failed, not_run=not_run)


def attach(report: dict, entries, after: str | None = None) -> dict:
    """A copy of `report` with the keys `partial` and `coverage` placed right after the key `after` (at the front when it is absent)."""
    partial, cov = build(entries)
    out: dict = {}
    if after is None or after not in report:
        out.update(partial=partial, coverage=cov)
    for k, v in report.items():
        out[k] = v
        if k == after:
            out.update(partial=partial, coverage=cov)
    return out


def _text(value) -> str:
    return " ".join(str(value).split())


def short(reason: str) -> str:
    """The first sentence of a reason, for the one line text. The full text is in the record."""
    text = _text(reason)
    cut = text.find(". ")
    return text[:cut] if cut >= 0 else text.rstrip(".")


def line(partial: bool, cov: dict) -> str:
    """The one line of the text output: `partial: ...`. A failed or not-run sub-check is named with its reason (first sentence).

    A sub-check that the inputs do not ask for is only counted ("2 not asked"): it is not a defect, and its name and reason are in the record.
    """
    ran, failed = cov.get("ran") or [], cov.get("failed") or []
    asked_not_run = [x for x in cov.get("not_run") or [] if x.get("asked", True)]
    not_asked = len(cov.get("not_run") or []) - len(asked_not_run)
    n = len(ran) + len(failed) + len(asked_not_run)
    if not partial:
        text = f"partial: no; {len(ran)} of {n} checks ran ({', '.join(ran)})"
    else:
        parts = []
        if failed:
            parts.append(f"{len(failed)} of {n} checks failed (" + "; ".join(f"{x['name']}: {short(x['reason'])}" for x in failed) + ")")
        if asked_not_run:
            parts.append(f"{len(asked_not_run)} of {n} checks not run (" + "; ".join(f"{x['name']}: {short(x['reason'])}" for x in asked_not_run) + ")")
        text = "partial: " + "; ".join(parts) + "; ran: " + (", ".join(ran) if ran else "none")
    return text + (f"; {not_asked} not asked" if not_asked else "")


def line_of(report: dict | None) -> str | None:
    """The line of a report that has the keys `partial` and `coverage`, else None (a report of an older version)."""
    if not isinstance(report, dict) or not isinstance(report.get("coverage"), dict) or not isinstance(report.get("partial"), bool):
        return None
    return line(report["partial"], report["coverage"])


# --------------------------------------------------------------------------- #
# topopt verify
# --------------------------------------------------------------------------- #

def _why(rec: dict) -> str:
    """The text of a record that did not run: its `message` (an error), else its first fact, else its `reason` code."""
    if rec.get("message"):
        return _text(rec["message"])
    facts = rec.get("facts") or []
    if facts:
        return _text(facts[0])
    return _text(rec.get("reason") or "no reason recorded")


def _by_status(rec) -> tuple[str, str | None]:
    """RAN for `done`, NOT_RUN for `skipped`, FAILED for `error`: the `status` that the records of verify share."""
    if not isinstance(rec, dict):
        return NOT_RUN, "the result has no record of this check"
    status = rec.get("status")
    if status == "done":
        return RAN, None
    if status == "skipped":
        return NOT_RUN, _why(rec)
    if status == "error":
        return FAILED, _why(rec)
    return NOT_RUN, f"the record has the status {status!r}"


def _buckling(rec) -> tuple[str, str | None]:
    """The record `buckling`: its own status, then the status of each load case (an error, or an eigen solve that did not finish)."""
    outcome, why = _by_status(rec)
    if outcome != RAN:
        return outcome, why
    failed, unfinished = [], []
    for c in rec.get("cases") or []:
        name = c.get("name")
        if c.get("status") == "error":
            failed.append(f"case {name}: {_why(c)}")
        elif c.get("status") in ("not_converged", "time_budget"):
            unfinished.append(f"case {name}: {short(_why(c))}")
        rb = c.get("rigid_body")
        if isinstance(rb, dict) and rb.get("status") == "error":
            failed.append(f"case {name}: the rigid-body test stopped: {_text(rb.get('message') or 'no message')}")
    if failed:
        return FAILED, "; ".join(failed + unfinished)
    if unfinished:
        return NOT_RUN, "; ".join(unfinished)
    return RAN, None


def _bores(entries) -> tuple[str, str | None]:
    """The list `bore_facts`: an entry with `error` is a failure, an entry with `empty` (no wall facet or no sample point) is a bore that was not checked."""
    if not isinstance(entries, list) or not entries:
        return NOT_RUN, "the result has no record of this check"
    failed, empty = [], []
    for e in entries:
        if e.get("status") == "error":
            failed.append(f"bore '{e.get('name')}' stopped at the {e.get('stage')}: {_text(e.get('message') or 'no message')}")
        elif e.get("status") == "empty":
            empty.append(f"bore '{e.get('name')}': " + ("no wall facet" if e.get("facets") == 0 else "no sample point") + " (see bore_facts)")
    if failed:
        return FAILED, "; ".join(failed + empty)
    if empty:
        return NOT_RUN, "; ".join(empty)
    return RAN, None


def _presence(rec) -> tuple[str, str | None]:
    return (RAN, None) if rec else (NOT_RUN, "the result has no record of this check")


def _merge(name: str, levels: list, results: list) -> Entry:
    """One entry from the results of every mesh size: the worst one decides, and with several sizes the reason names the size."""
    bad = [(lv.get("size_mm"), outcome, why) for lv, (outcome, why) in zip(levels, results) if outcome != RAN]
    if not bad:
        return Entry(name, RAN)
    outcome = FAILED if any(o == FAILED for _, o, _ in bad) else NOT_RUN
    if len(levels) == 1:
        return Entry(name, outcome, True, bad[0][2])
    return Entry(name, outcome, True, "; ".join(f"mesh size {s:g} mm: {w}" for s, _, w in bad))


def from_verify(report: dict, levels: list, plan) -> list[Entry]:
    """The entries of a `topopt verify` report. `levels` are the level dicts of `run_level`; `plan` is the `Plan` of `preflight`.

    The sub-checks, in the order of the report: `stress` (always), `supports` (joints and springs), `buckling`, `motion` (the A5 rows),
    `bore_facts`, `goodman`, `pressfit` and `gate`. The name of a sub-check is the key of its record. A failure of `stress` or of
    `supports` stops verify (no `verify.json`), so a report always has them as `ran` when the YAML asks for them. A sub-check
    that the inputs do not ask for is NOT_RUN with `asked` false.
    """
    from .verify import motion as vmotion

    cfg = plan.info.cfg
    bc = cfg.bore_checks if plan.bore_checks else None
    bores_off = ("the YAML has no bore_checks block" if cfg.bore_checks is None else "the call turned the bore checks off")

    def per_level(name, classify, asked, not_asked):
        if not asked:
            return Entry(name, NOT_RUN, False, not_asked)
        return _merge(name, levels, [classify(lv.get(name)) for lv in levels])

    gate_entry = (Entry("gate", RAN) if "gate" in report else Entry("gate", NOT_RUN, True, "the result has no record of this check")) \
        if plan.gate_limits is not None else Entry("gate", NOT_RUN, False, "the call did not ask for the gate")
    return [
        Entry("stress", RAN),
        per_level("supports", _presence, bool(cfg.joints or cfg.springs), "the YAML has no joint and no spring"),
        per_level("buckling", _buckling, bool(plan.buckling), "buckling.check is off in the YAML and the call did not turn the check on"),
        per_level("motion", _by_status, vmotion.wanted(cfg), "the YAML has no limits.displacement and no limits.alignment row"),
        per_level("bore_facts", _bores, bc is not None, bores_off),
        per_level("goodman", _by_status, bc is not None and bc.goodman,
                  bores_off if bc is None else "the bore_checks block has no goodman block"),
        per_level("pressfit", _by_status, bc is not None and bc.press_fit,
                  bores_off if bc is None else "the bore_checks block has no press_fit block"),
        gate_entry,
    ]


# --------------------------------------------------------------------------- #
# The print report and the print check
# --------------------------------------------------------------------------- #

def from_print_section(section: dict) -> list[Entry]:
    """The entries of the `print` section of `metrics.json` (CONTRACTS.md section 18): `bed_fit`, `overhang`, `plate_contact`,
    `support_volume` and `thinnest_member`.

    The first three come from one measure of the configured build direction and are always there. `support_volume` is not asked for a
    printer type that needs no support. `thinnest_member` has `error` (the estimate stopped) or no sample (no chord could be measured).
    """
    entries = [Entry("bed_fit", RAN), Entry("overhang", RAN), Entry("plate_contact", RAN)]
    if section.get("support_needed") is False:
        entries.append(Entry("support_volume", NOT_RUN, False, _text(section.get("support_note") or "the printer type needs no support")))
    else:
        entries.append(Entry("support_volume", RAN))
    thin = section.get("thinnest_member") or {}
    if thin.get("error"):
        entries.append(Entry("thinnest_member", FAILED, True, _text(thin["error"])))
    elif not thin.get("samples"):
        entries.append(Entry("thinnest_member", NOT_RUN, True, "no chord could be measured on this mesh"))
    else:
        entries.append(Entry("thinnest_member", RAN))
    return entries


def from_failed_print_section(error: str) -> list[Entry]:
    """The entries of a `print` section that could not be made (`{"ok": false, "error": ...}`): the whole report failed."""
    return [Entry("print_report", FAILED, True, _text(error))]


def from_printcheck(checks: list) -> list[Entry]:
    """The entries of a `topopt printcheck` report, one for each check (CONTRACTS.md section 32).

    A `pass` or a `fail` is a verdict: the check ran. A check that stopped has the key `error`: it failed. A check with the state `unknown`
    ran when the print block gives it no limit (its value is a fact and the inputs ask for no verdict); it is NOT_RUN and asked when the
    block gives a limit and the check could not decide, the case that `ratchet.printcheck_facts` lists as `print_unknown`. A support
    estimate for a printer type that needs no support, and a thickness estimate with no chord, are NOT_RUN.
    """
    out = []
    for c in checks:
        cid, state = c["id"], c.get("state")
        value, limit = c.get("value") or {}, c.get("limit") or {}
        if c.get("error"):
            out.append(Entry(cid, FAILED, True, _text(c["error"])))
        elif cid == "support_volume" and value.get("support_needed") is False:
            out.append(Entry(cid, NOT_RUN, False, _text(c.get("reason") or "the printer type needs no support")))
        elif cid == "wall_thickness" and not value.get("samples"):
            out.append(Entry(cid, NOT_RUN, True, "no chord could be measured on this mesh"))
        elif state == "unknown" and limit.get("value") is not None:
            out.append(Entry(cid, NOT_RUN, True, _text(c.get("reason") or "the check could not decide")))
        else:
            out.append(Entry(cid, RAN))
    return out
