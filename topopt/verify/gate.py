"""The opt-in hard gate of `topopt verify --gate` (K1-B, CONTRACTS.md section 16, "The gate").

Without the switch, `verify.json` and the exit codes are as before. With the switch, the result has the key `gate`: for each check and each
case the value, the limit, where the limit comes from, and `met`. The limits come from keys that exist; the gate adds no YAML key.

- `stress`: the stress by volume at the `--percentile`, against `load_cases[].stress_limit_mpa` (section 17), or against `--allowable-mpa`
  for every case. A case with no limit is listed in `not_gated`.
- `buckling`: the lowest tet load factor of a case, against `buckling.min_factor`. The gate turns the buckling check on when the YAML has
  `min_factor`. Without `min_factor`, the check is `not_gated`. A load case with no compression (status `no_positive_factor`, a pure tension
  state) is `not_gated` too: the check ran and found nothing that can buckle. It is not a failure and not a pass.
- `displacement`, `alignment`: the tet value of each row of `limits.displacement` and `limits.alignment` (key `motion`, section 31).

A check that the gate asked for and that could not run (a skipped or error record, a row that is missing, a value that is not a finite
number) has `met: null`. It fails the gate (decision D2 of the plan). A gate that gates no check is refused before the run (`gate_plan`).
The test is strict: a value on the limit passes, a value above it (or a factor below it) fails. No tolerance.

When `--allowable-mpa` replaces the stress limit of a case in the YAML, the record keeps the replaced limit and its source, and says whether the
new limit is looser (`loosened`). A `--percentile` that is not the default is a fact of the record in the same way. These are facts, not refusals.

`evaluate` is a pure function over plain dicts: it reads no file and runs no solve.
"""
from __future__ import annotations

import math

from ..config import UserError
from ..rows_motion import row_specs
from .names import opt

KINDS = ("stress", "buckling", "displacement", "alignment")
UNITS = {"stress": "MPa", "buckling": "load factor"}
NO_COMPRESSION = "no compression in this load case, nothing can buckle"      # the reason of a case with status `no_positive_factor`
NOTE = ("the gate is strict (a value on the limit passes, no tolerance); a check that could not run has met null and fails the gate; "
        "the stress is the linear static p-value by volume of one mesh, so fatigue and the peaks at sharp features are not checked")


def gate_limits(cfg, allowable_mpa: float | None = None) -> dict:
    """The limits that the gate reads, from the config of the run. Plain dict (JSON-safe), so a test can write one by hand.

    - `stress`: `{case: {"limit": MPa or None, "source": text}}`. `--allowable-mpa` gives every case the same limit. When it replaces a limit
      of the YAML, the entry also has `replaced_limit`, `replaced_source` and `loosened` (the new limit is above the replaced one).
    - `buckling`: `{"limit": min_factor, "source": text}` or None.
    - `applies_load`: `{case: bool}`.
    - `rows`: one entry for each row of `limits.displacement` and `limits.alignment`: name, check, case, limit, source.
    """
    stress = {}
    for c in cfg.load_cases:
        if allowable_mpa is not None:
            stress[c.name] = dict(limit=float(allowable_mpa), source=f"{opt('allowable')} (all cases)")
            if c.stress_limit_mpa is not None:      # a limit of the YAML is replaced: a fact, so the reader sees a pass that comes from a looser number
                stress[c.name].update(replaced_limit=float(c.stress_limit_mpa), replaced_source=f"load_cases[{c.name}].{c.stress_limit_from}",
                                      loosened=bool(float(allowable_mpa) > float(c.stress_limit_mpa)))
        elif c.stress_limit_mpa is not None:
            stress[c.name] = dict(limit=float(c.stress_limit_mpa), source=f"load_cases[{c.name}].{c.stress_limit_from}")
        else:
            stress[c.name] = dict(limit=None, source=None)
    mf = cfg.buckling.min_factor
    rows = []
    for r in row_specs(cfg):
        check = "displacement" if r.part == "displacement" else "alignment"
        key = {"displacement": "limit_mm", "shift": "shift_limit_mm", "tilt": "tilt_limit_rad"}[r.part]
        rows.append(dict(name=r.name, check=check, case=r.case, limit=float(r.limit),
                         source=f"limits.{check}[{r.source.name}].{key}"))
    return dict(stress=stress, buckling=None if mf is None else dict(limit=float(mf), source="buckling.min_factor"),
                applies_load={c.name: bool(c.applies_load) for c in cfg.load_cases}, rows=rows)


def gate_plan(limits: dict, *, buckling_flag: bool | None) -> bool:
    """Check before the run that the gate has something to gate. Return True when the buckling check must run for the gate.

    Only a check that can run counts: a stress limit, a buckling limit with at least one load case that applies a load, a limits row.
    Raise `UserError` (exit code 1) when no such check exists, and when `--no-buckling` meets a runnable buckling check (the user asks for
    a check and for no check).
    """
    want_bk = limits["buckling"] is not None
    runs_bk = want_bk and any(limits["applies_load"].get(case, True) for case in limits["stress"])      # a case with no load has no buckling check
    if runs_bk and buckling_flag is False:
        raise UserError("the gate asks for the buckling check, because the YAML has buckling.min_factor, and --no-buckling turns the check "
                        "off. Options: remove --no-buckling; or remove buckling.min_factor from the YAML to leave buckling out of the gate")
    gated = (any(s["limit"] is not None for s in limits["stress"].values()) or runs_bk or bool(limits["rows"]))
    if not gated:
        bk_text = ("the YAML has buckling.min_factor, but no load case applies a load, so there is no buckling check" if want_bk
                   else "the YAML has no buckling.min_factor")
        raise UserError(f"the gate has no limit to check: no load case has stress_limit (or allowable_mpa), {bk_text} "
                        f"and no limits.displacement or limits.alignment rows. Options: add a limit to the YAML (CONTRACTS.md sections 17, 31, 33); "
                        f"or give {opt('allowable')}; or run without the gate")
    return runs_bk


def _finite(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _limit_over_value(v, lim):
    """The margin of a check that needs `value <= limit`: limit / value, or None for a value of 0 (nothing to divide)."""
    return lim / v if v > 0 else None


def _value_over_limit(v, lim):
    """The margin of a check that needs `value >= limit` (a buckling factor): value / limit, or None for a limit of 0."""
    return v / lim if lim > 0 else None


def _record(value, limit, source, unit, ok, margin) -> dict:
    """One check. `ok(value, limit)` is the test; `margin(value, limit)` is the number that is 1 or more when the check is met (None when it
    is not defined). A value that is not a finite number gives `met: null` and no margin."""
    v = _finite(value)
    rec = dict(value=v, limit=limit, source=source, unit=unit, met=None if v is None else bool(ok(v, limit)))
    rec["margin"] = None if v is None else margin(v, limit)
    if v is None:
        rec["reason"] = "the check has no value"
    return rec


def _record_reason(rec) -> str:
    """The reason of a record that did not run, from its own keys (`status`, `reason`, `message`)."""
    if not isinstance(rec, dict):
        return "the result has no such record"
    parts = [f"status {rec.get('status')!r}"] if rec.get("status") else []
    parts += [str(rec[k]) for k in ("reason", "message") if rec.get(k)]
    return "the check did not run: " + ", ".join(parts) if parts else "the check did not run"


VALUE_SOURCE = "buckling.cases[{case}].load_factors[0]"                 # where the gate reads its value: the lowest load factor of the case
PROJECTED_SOURCE = "buckling.cases[{case}].rigid_body.load_factors[0]"      # the projected value: a fact next to it, never the gate value
PROJECTED_REASON = ("The lowest mode of this case is a motion of the part on its supports (the rigid-body rule ran, CONTRACTS.md section 33). "
                    "Without the six rigid-body motions of the part, the part itself buckles at {projected:.6g}. The gate reads {low:.6g}, the "
                    "lowest load factor of the case. {projected:.6g} is higher, and a higher number in place of the measured one would loosen "
                    "buckling.min_factor, a requirement. This is a fact for the user to weigh; the gate does not read it.")
PROJECTED_FAILED = ("The rigid-body rule ran for this case, but the solve without the rigid-body motions of the part gave the status {status!r}"
                    "{why}, so there is no projected value. The gate reads {low:.6g}, the lowest load factor of the case.")


def projected_fact(case_rec: dict | None, case: str, low) -> dict:
    """The projected factor of a case as a fact next to the gate value: `projected_value`, `projected_source`, `projected_reason`.

    The rigid-body rule (`buckling_p2.rule_trigger`) solves a case again without the rigid-body motions of the part when its lowest mode is a
    swing on the supports. The result is in `rigid_body`. The gate and the ratchet do not read it: it is higher than the measured value, and a
    higher number in place of the measured one would loosen `buckling.min_factor`, a requirement. So it is a fact. The result is `{}` when the
    rule did not run for the case. When the rule ran and the solve gave no factor, `projected_value` is None and the reason says why.
    """
    rb = (case_rec or {}).get("rigid_body")
    if not (isinstance(rb, dict) and rb.get("applied")):
        return {}
    source = PROJECTED_SOURCE.format(case=case)
    factors = rb.get("load_factors") or []
    if rb.get("status") == "ok" and factors:
        return dict(projected_value=factors[0], projected_source=source, projected_reason=PROJECTED_REASON.format(projected=factors[0], low=low))
    why = f" ({rb['message']})" if rb.get("message") else ""
    return dict(projected_value=None, projected_source=source, projected_reason=PROJECTED_FAILED.format(status=rb.get("status"), why=why, low=low))


def _gate_case(level: dict, case: str):
    """`(value, reason, case record, no_compression)` of a case in the record `buckling`.

    `value` is `load_factors[0]` of the case, the lowest load factor, as the ratchet reads it (decision of K1-B/K1-C: the measured value, not the
    projected one, `projected_fact`). `no_compression` is True for a case of status `no_positive_factor` (a pure tension state,
    `topopt.buckling.NO_POSITIVE`): the check ran and found nothing that can buckle. That is a result, not a check that could not run.
    """
    from ..buckling import NO_POSITIVE      # inside the function: the module pulls in scipy, and `evaluate` is otherwise a pure function of dicts
    bk = level.get("buckling")
    if not isinstance(bk, dict) or bk.get("status") != "done":
        return None, _record_reason(bk) if bk is not None else "the result has no buckling record (the check was not on)", None, False
    for c in bk.get("cases") or []:
        if c.get("name") == case:
            lf = c.get("load_factors") or []
            if lf:
                return lf[0], None, c, False
            if c.get("status") == NO_POSITIVE:
                return None, NO_COMPRESSION, c, True
            return None, f"the case has no load factor (status {c.get('status')!r})" + (f": {c['message']}" if c.get("message") else ""), c, False
    return None, "the buckling record has no entry for this case", None, False


def _lowest_factor(level: dict, case: str):
    """`(value, reason)`: the factor that the gate reads for a case in the record `buckling`, or None and why it is missing."""
    val, why, _rec, _none = _gate_case(level, case)
    return val, why


def _motion_rows(level: dict):
    """`({row name: row}, reason)`."""
    mo = level.get("motion")
    if not isinstance(mo, dict) or mo.get("status") != "done":
        return {}, _record_reason(mo) if mo is not None else "the result has no motion record"
    return {r.get("name"): r for r in mo.get("rows") or []}, None


def _values(level: dict, limits: dict) -> dict:
    """`{(check, name): value or None}` of one level, for the comparison with the level before."""
    out = {}
    for case, s in limits["stress"].items():
        if s["limit"] is not None:
            out[("stress", case)] = _finite((((level.get("cases") or {}).get(case) or {}).get("von_mises") or {}).get("percentile_by_volume_MPa"))
    if limits["buckling"] is not None:
        for case in limits["stress"]:
            out[("buckling", case)] = _finite(_lowest_factor(level, case)[0])
    rows, _ = _motion_rows(level)
    for r in limits["rows"]:
        out[(r["check"], r["name"])] = _finite((rows.get(r["name"]) or {}).get("tet_value"))
    return out


def evaluate(level: dict, limits: dict, *, percentile: float, previous: dict | None = None) -> dict:
    """The key `gate` for the reference level `level`. `previous` is the level before it (`--levels`), or None.

    `level` is a level dict of `run_level` (keys `cases`, `buckling`, `motion`, `size_mm`). `limits` is the dict of `gate_limits`.
    With `previous`, each record has `previous_value` and `change_from_previous` ((finer - coarser) / coarser; null if the
    coarser value is 0 or a value is missing).

    Each record has `value`, `limit`, `source`, `unit`, `met` and `margin` (1 or more when the check is met: limit / value for a stress and
    a row, value / limit for a buckling factor; null when it is not defined). A stress record also has `percentile`, `peak_qp_MPa` (the
    highest value at one point, for information), and, when `--allowable-mpa` replaced a limit of the YAML, `replaced_limit`,
    `replaced_source` and `loosened`. When the percentile is not the default, it has `replaced_percentile` (the default) and
    `percentile_loosened` (a lower percentile gives a lower value). A buckling record has `value_source` (the key that the value comes from:
    `load_factors[0]` of the case, the measured lowest load factor), `run_lowest_load_factor` (`{case, value}`, a copy of the headline
    `buckling.lowest_load_factor`, the lowest over all cases) and, when the rigid-body rule ran for the case, `projected_value`,
    `projected_source` and `projected_reason` (`projected_fact`: a fact, never the gate value).
    """
    from .run import DEFAULT_PERCENTILE      # inside the function: run.py imports this module
    checks: dict = {k: {} for k in KINDS}
    not_gated = []
    for case, s in limits["stress"].items():
        if s["limit"] is None:
            not_gated.append(dict(check="stress", case=case, reason="the load case has no stress_limit or allowable_mpa in the YAML, "
                                                                  f"and no {opt('allowable')} was given"))
            continue
        vm = ((level.get("cases") or {}).get(case) or {}).get("von_mises")
        raw = (vm or {}).get("percentile_by_volume_MPa")
        rec = _record(raw, s["limit"], s["source"], UNITS["stress"], lambda v, lim: v <= lim, _limit_over_value)
        rec["percentile"] = float(percentile)
        rec["peak_qp_MPa"] = _finite((vm or {}).get("peak_qp_MPa"))
        for key in ("replaced_limit", "replaced_source", "loosened"):
            if key in s:
                rec[key] = s[key]
        if float(percentile) != DEFAULT_PERCENTILE:
            rec["replaced_percentile"] = DEFAULT_PERCENTILE
            rec["percentile_loosened"] = bool(float(percentile) < DEFAULT_PERCENTILE)
        if rec["met"] is None:
            rec["reason"] = "the result has no stress for this case"
        checks["stress"][case] = rec
    bk = limits["buckling"]
    if bk is None:
        not_gated.append(dict(check="buckling", reason="the YAML has no buckling.min_factor"))
    else:
        for case in limits["stress"]:
            if not limits["applies_load"].get(case, True):
                not_gated.append(dict(check="buckling", case=case, reason="the load case applies no load"))
                continue
            val, why, crec, no_compression = _gate_case(level, case)
            if no_compression:      # the check ran: there is nothing that can buckle, so the limit has nothing to compare with
                not_gated.append(dict(check="buckling", case=case, reason=why))
                continue
            rec = _record(val, bk["limit"], bk["source"], UNITS["buckling"], lambda v, lim: v >= lim, _value_over_limit)
            if rec["value"] is not None:
                rec["value_source"] = VALUE_SOURCE.format(case=case)
                run_low = (level.get("buckling") or {}).get("lowest_load_factor")      # the headline of the key `buckling`: the lowest over all cases
                if isinstance(run_low, dict) and _finite(run_low.get("value")) is not None:
                    rec["run_lowest_load_factor"] = dict(case=run_low.get("case"), value=run_low["value"])
                rec.update(projected_fact(crec, case, rec["value"]))
            if rec["met"] is None:
                rec["reason"] = why or rec["reason"]
            checks["buckling"][case] = rec
    by_check: dict = {"displacement": [], "alignment": []}
    for r in limits["rows"]:
        by_check[r["check"]].append(r)
    rows, why_rows = _motion_rows(level) if limits["rows"] else ({}, None)
    for kind in ("displacement", "alignment"):
        if not by_check[kind]:
            not_gated.append(dict(check=kind, reason=f"the YAML has no limits.{kind} rows"))
        for r in by_check[kind]:
            row = rows.get(r["name"])
            rec = _record(None if row is None else row.get("tet_value"), r["limit"], r["source"], (row or {}).get("unit"),
                          lambda v, lim: v <= lim, _limit_over_value)
            rec["case"] = r["case"]
            if rec["met"] is None:
                rec["reason"] = why_rows or ("the motion record has no value for this row" if row is not None
                                             else "the motion record has no such row")
            checks[kind][r["name"]] = rec
    out = dict(requested=True, mesh_size_mm=level.get("size_mm"), percentile=float(percentile), note=NOTE)
    if previous is not None:
        before, now = _values(previous, limits), _values(level, limits)
        out["previous_level_mm"] = previous.get("size_mm")
        for kind in KINDS:
            for name, rec in checks[kind].items():
                a, b = before.get((kind, name)), now.get((kind, name))
                rec["previous_value"] = a
                rec["change_from_previous"] = (b - a) / a if a not in (None, 0) and b is not None else None
    checks = {k: v for k, v in checks.items() if v}
    failed = [f"{kind}:{name}" for kind, recs in checks.items() for name, rec in recs.items() if rec["met"] is not True]
    out.update(checks=checks, not_gated=not_gated, failed=failed, passed=bool(checks) and not failed)
    if not checks:      # every check the gate asked for was not applicable after the solve: no pass, and no failed check to name
        out["reason"] = "no check was gated: every check is in not_gated, so there is nothing to compare with a limit"
    return out


def replaced_sentences(rec: dict) -> list[str]:
    """What `--allowable-mpa` and `--percentile` replaced in the record of a stress check: a fact, in sentences without the final stop."""
    parts = []
    if rec.get("replaced_limit") is not None:
        side = "looser" if rec["loosened"] else ("the same" if rec["limit"] == rec["replaced_limit"] else "tighter")
        unit = f" {rec['unit']}" if rec.get("unit") else ""
        parts.append(f"The limit replaces {rec['replaced_limit']:.4g}{unit} of {rec['replaced_source']}; the new limit is {side}")
    if rec.get("replaced_percentile") is not None:
        side = "looser (a lower percentile gives a lower value)" if rec["percentile_loosened"] else "tighter (a higher percentile gives a higher value)"
        parts.append(f"The percentile {rec['percentile']:g} replaces the default {rec['replaced_percentile']:g}; the test is {side}")
    return parts


def value_sentences(rec: dict) -> list[str]:
    """Where the value of a buckling record comes from, the headline of the run and the projected fact: sentences without the final stop.

    One gate line holds the three numbers that `verify.json` can show for one case: the gate value, the lowest value of the run, and the value
    without the rigid-body motions of the part.
    """
    parts = []
    if rec.get("value_source"):
        text = f"Value from {rec['value_source']}"
        low = rec.get("run_lowest_load_factor")
        if low:
            text += f"; the lowest of all cases is {low['value']:.4g} (case {low['case']}, buckling.lowest_load_factor)"
        parts.append(text)
    if rec.get("projected_value") is not None:
        parts.append(f"Without the rigid-body motions of the part the value is {rec['projected_value']:.4g} ({rec['projected_source']}); "
                     f"a fact, the gate does not read it")
    elif "projected_value" in rec:
        parts.append(f"The solve without the rigid-body motions of the part gave no value ({rec['projected_source']})")
    return parts


def gate_verdict(g: dict) -> str:
    """The verdict line: `gate: PASSED`, or `gate: FAILED (<the checks that failed>)`. The text `plain` of `verify_status` has it too."""
    why = g["failed"] and ", ".join(g["failed"]) or g.get("reason")
    return f"gate: {'PASSED' if g['passed'] else 'FAILED'}" + (f" ({why})" if why else "")


def gate_lines(g: dict) -> list[str]:
    """The lines that `topopt verify` prints for the key `gate`: one line for each check, then the verdict."""
    lines = []
    for kind, recs in g["checks"].items():
        for name, rec in recs.items():
            val = "no value" if rec["value"] is None else f"{rec['value']:.4g}"
            word = {True: "met", False: "NOT MET", None: "NOT MET (could not run)"}[rec["met"]]
            op = ">=" if kind == "buckling" else "<="
            lines.append(f"gate: {kind} {name}: {val} (needs {op} {rec['limit']:.4g}{' ' + rec['unit'] if rec.get('unit') else ''}; "
                         f"{rec['source']}): {word}" + (f", margin {rec['margin']:.3g}" if rec.get("margin") is not None else "")
                         + (f". {rec['reason']}" if rec.get("reason") else "") + "".join(f". {t}" for t in replaced_sentences(rec) + value_sentences(rec))
                         + (f". Change from the level {g['previous_level_mm']:g} mm: {100 * rec['change_from_previous']:+.1f} %"
                            if rec.get("change_from_previous") is not None else ""))
    for n in g["not_gated"]:
        lines.append(f"gate: {n['check']}{' ' + str(n['case']) if n.get('case') else ''}: not gated. {n['reason']}")
    lines.append(gate_verdict(g))
    return lines
