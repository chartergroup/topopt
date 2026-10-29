"""The Goodman fatigue check at the bores of a part (`topopt verify`, key `goodman`). CONTRACTS.md section 40.

The rule is that of the pipeline check (`s9_print/final/goodman.py`):

    n = 1 / (sa / Se + sm / Su)             the Goodman line (the factor on the alternating stress)
    yield ratio = (|sm| + sa) / Sy

`sa` is the von Mises statistic of a load case at the bore sample set (`bore.py`). Each case is taken fully reversed about the
mean stress `sm`. `sm` is the hoop stress of the press fit at the same bore and statistic (B4b), or the `mean_stress_mpa` of the
YAML, or 0 when the YAML gives neither and has no press fit. The statistics are p99.9 and max. The worst case of a row is the case
with the smallest `n`. Su, Sy and Se are the numbers of the YAML: topopt applies no factor (decision D7).

This module has no import of an extra and no part, material or case name. The function `goodman_row` is the one formula: the product
(`build_record`) and the level-2b converter of the acceptance case call it.
"""
from __future__ import annotations

from .names import opt

KIND = "goodman"
STATISTICS = ("p99_9", "max")          # the statistics of the bore sample set (plan 4.3; `bore.BORE_PERCENTILE` is the p99.9)

RULE = ("n = 1 / (sa/Se + sm/Su) (Goodman line); yield ratio = (|sm| + sa) / Sy; sa = von Mises amplitude of the load case at the bore "
        "(the case is taken fully reversed about sm); sm = mean stress: the hoop stress of the press fit at the same bore and statistic, "
        "or the mean stress of the YAML, or 0 when the YAML gives neither and has no press fit; statistics: p99.9 and max of the bore "
        "sample set; Su, Sy and Se are the values of the YAML (no factor is applied); the worst case has the smallest n.")

FACT_MIX = ("sa is a von Mises value and sm is a hoop (normal) stress. This mix is the method of the pipeline check; it is not a signed "
            "equivalent stress. The result is a number for comparison with the pipeline, not a verdict.")
FACT_CASES = ("Each load case is solved alone (a case is one load, with no cup spring). The case is taken as fully reversed about sm: "
              "the alternating stress is the whole von Mises value of the case.")
FACT_STRENGTHS = ("Su, Sy and Se are the numbers of the YAML with their source texts. Se is used as given (fully corrected); "
                  "topopt applies no factor and takes no strength from a catalog.")
FACT_SCATTER = ("n uses the p99.9 and the max of the von Mises stress at the sample set of the bore. The P2 stress is not smoothed: it scatters "
                "from point to point, and the size of the scatter depends on the mesh realization, not only on the mesh size. These two statistics "
                "take the top of the scatter, so they can be above the stress that the load gives, and n is low by the same fraction (at sm = 0, "
                "n = Se/sa). The median of the sample set is in service_stress and in the text lines of the bore: the gap between the median and "
                "the p99.9 shows the size of the scatter and of the stress gradient on this mesh. The measured scatter of a thin ring is in "
                "CONTRACTS.md section 40.7.")
FACT_NEGATIVE_SM = ("sm is below 0 in {n} row(s). The Goodman line is applied as it is, and the yield ratio uses |sm|. A compressive mean "
                    "stress is not always a benefit. The usual rule for a compressive mean stress is a flat line (n = Se/sa), and the Goodman line "
                    "extended to sm below 0 gives a larger n. Each such row has the field n_goodman_at_sm0, the n for sm = 0 (the flat line), with "
                    "its worst case in worst_case_at_sm0; the text line of the row shows it.")
OPTION_NEGATIVE_SM = "Give mean_stress_mpa = 0 for the bore (and no press_fit), to see the rows with no mean stress."


def goodman_point(sa: float, sm: float, su: float, sy: float, se: float):
    """`(n, yield ratio)` of one load case. `n` is None when `sa/Se + sm/Su` is 0 or below (no stress, or a negative `sm` that cancels `sa`).

    The order of the operations is that of `goodman.py:17`, so a row of the pipeline is reproduced bit for bit (`sm` is 0 or above there).
    """
    den = sa / se + sm / su
    n = 1.0 / den if den > 0 else None
    return n, (abs(sm) + sa) / sy


def goodman_row(sa_by_case: dict, sm: float, su: float, sy: float, se: float) -> dict:
    """The row of one (material, variant, bore, statistic): the worst case and the values of every case.

    `sa_by_case` is `{case name: sa}` in the order of the cases. The worst case is the one with the smallest `n` among the cases whose `n` is
    not None; the first wins a tie (`n < worst`, `goodman.py:18`). If every case has `n = None`: `n_goodman` is None, the worst case is the one
    with the largest yield ratio (the first wins a tie), and `all_n_none` is True.

    A row with `sm` below 0 also has `n_goodman_at_sm0` and `worst_case_at_sm0`: the same rule for `sm = 0` (`Se/sa`, the flat line of the usual rule for
    a compressive mean stress). A row with `sm` of 0 or above has neither key.
    """
    cases, worst, best_y = [], None, None
    for name, sa in sa_by_case.items():
        n, y = goodman_point(sa, sm, su, sy, se)
        cases.append(dict(case=name, sa_MPa=sa, n_goodman=n, peak_over_yield=y))
        if n is not None and (worst is None or n < worst["n_goodman"]):
            worst = cases[-1]
        if best_y is None or y > best_y["peak_over_yield"]:
            best_y = cases[-1]
    pick = worst if worst is not None else best_y
    row = dict(worst_case=pick["case"], sa_MPa=pick["sa_MPa"], sm_MPa=sm, n_goodman=pick["n_goodman"] if worst is not None else None,
               peak_over_yield=pick["peak_over_yield"], all_n_none=worst is None, cases=cases)
    if sm < 0:     # a compressive mean stress: n for sm = 0 (Se/sa), the flat-line rule, so that the user needs no second run
        at0 = goodman_row(sa_by_case, 0.0, su, sy, se)
        row.update(n_goodman_at_sm0=at0["n_goodman"], worst_case_at_sm0=at0["worst_case"])
    return row


def _skip_record(reason: str, facts: list, options: list, missing: list | tuple = (), **extra) -> dict:
    return dict(kind=KIND, status="skipped", reason=reason, rule=RULE, statistics=list(STATISTICS), rows=[], missing_rows=list(missing),
                facts=facts, options=options, **extra)


def error_record(message: str) -> dict:
    """The record of a check that stopped with an exception. `topopt verify` goes on."""
    return dict(kind=KIND, status="error", reason="exception", message=message, rule=RULE, statistics=list(STATISTICS), rows=[],
                missing_rows=[], facts=[f"The check stopped: {message}"],
                options=["Read the message, then run the check again; or remove the `bore_checks` block from the YAML (or use `--no-bore-checks`)."])


def _mean_stress(bc) -> dict:
    """Where `sm` comes from (plan 5.1): the press fit, the YAML mean stress of a bore, or 0."""
    if bc.press_fit:
        return dict({"from": "press_fit"}, note="sm is the hoop stress of the press fit at the same bore and statistic (one value for each cup variant).")
    by_bore = {b.name: ("yaml_mean" if b.mean_stress_mpa is not None else "zero_mean") for b in bc.bores}
    src = "yaml_mean" if "yaml_mean" in by_bore.values() else "zero_mean"
    note = ("sm is the mean_stress_mpa of the bore in the YAML (a bore with none has sm = 0)." if src == "yaml_mean"
            else "The YAML gives no mean stress and no press fit: sm = 0 (a fully reversed load about zero).")
    return dict({"from": src}, by_bore=by_bore, note=note)


def build_record(bc, cases: list, service: dict, *, hoop: dict | None = None, press_reason: str | None = None,
                 press_options: list | tuple = (), dropped: list | tuple = ()) -> dict:
    """The record `goodman` of one level.

    `bc` is `Config.bore_checks`. `cases` are the load case names in the YAML order. `service` is `{case: {bore: stats}}`, with the keys of
    `bore.SampleSet.stats` (`vm_p99_9_MPa`, `vm_max_MPa`, ...); a bore that is not in it has no sample set. `dropped` lists
    `{"bore", "reason", "facts", "options"}` for each bore that was dropped (no facet, no sample point). With a `press_fit` block, `hoop` is
    `{variant: {bore: {statistic: sm}}}` from the press-fit stage; without it (`press_reason`, `press_options`: the stage did not run or gave
    nothing) a variant has no rows and `missing_rows` says so (plan 5.1; there is no `sm = 0` row for a part with a press fit).
    """
    if not bc.goodman:
        raise ValueError("build_record needs a bore_checks block with a goodman block")
    bores = [b for b in bc.bores if all(b.name in service.get(c, {}) for c in cases)]
    missing = [dict(variant=None, bore=d["bore"], reason=d.get("reason", "no_sample_set"), options=list(d.get("options", []))) for d in dropped]
    if not bores:
        return _skip_record("no_bore", ["No bore has a sample set, so there is no stress to check. The facts of `bore_facts` say why."],
                            ["Check the geometry of the bores against the part, or the knobs of `bore_checks` (see `bore_facts`).",
                             "Remove the `bore_checks` block from the YAML (or use `--no-bore-checks`)."],
                            missing, mean_stress=_mean_stress(bc))
    mats = {m.label: dict(su_mpa=m.su_mpa, sy_mpa=m.sy_mpa, se_mpa=m.se_mpa, sources=dict(m.sources)) for m in bc.materials}
    variants = ([c.name for c in bc.cups] if bc.press_fit else [None])
    rows, facts, options = [], [], []
    for mat in bc.materials:
        for var in variants:
            for b in bores:
                if var is None:
                    sm_by_stat = {s: (b.mean_stress_mpa if b.mean_stress_mpa is not None else 0.0) for s in STATISTICS}
                    label = "yaml_mean" if b.mean_stress_mpa is not None else "zero_mean"
                else:
                    sm_by_stat = (hoop or {}).get(var, {}).get(b.name)
                    label = var
                    if sm_by_stat is None:
                        if mat is bc.materials[0]:
                            missing.append(dict(variant=var, bore=b.name, reason=press_reason or "no_press_fit_result", options=list(press_options)))
                        continue
                for stat in STATISTICS:
                    sa_by_case = {c: service[c][b.name][f"vm_{stat}_MPa"] for c in cases}
                    r = goodman_row(sa_by_case, sm_by_stat[stat], mat.su_mpa, mat.sy_mpa, mat.se_mpa)
                    rows.append(dict(id=f"{mat.label}/{label}/{b.name}/{stat}", material=mat.label, variant=label, bore=b.name, statistic=stat, **r))
    facts += [f"Rule: {RULE}", FACT_MIX, FACT_CASES, FACT_STRENGTHS, FACT_SCATTER]
    ms = _mean_stress(bc)
    facts.append(ms["note"])
    if bc.press_fit and hoop is None:
        facts.append(f"The press-fit result is not available ({press_reason or 'no result'}), so no Goodman row is given for the cup variants: "
                     "a row with sm = 0 would describe a part with no press fit.")
        options.extend(press_options)
    elif missing and any(m["variant"] is not None for m in missing):
        facts.append("Some bores or variants have no press-fit result and no Goodman row; `missing_rows` names them with the reason.")
    nn = [r["id"] for r in rows if r["all_n_none"]]
    if nn:
        facts.append(f"In {len(nn)} row(s) every load case has sa/Se + sm/Su at or below 0, so n is null; the worst case is the one with the "
                     f"largest yield ratio: {', '.join(nn[:6])}{' ...' if len(nn) > 6 else ''}.")
    neg = [r for r in rows if r["sm_MPa"] < 0]
    if neg:
        facts.append(FACT_NEGATIVE_SM.format(n=len(neg)))
        options.append(OPTION_NEGATIVE_SM)
    ns = [(r["n_goodman"], r["id"]) for r in rows if r["n_goodman"] is not None]
    lowest = min(ns, key=lambda t: t[0]) if ns else None
    below = [i for n, i in ns if n < 1.0]
    if below:
        facts.append(f"n is below 1 in {len(below)} of {len(rows)} row(s): {', '.join(below[:8])}{' ...' if len(below) > 8 else ''}. "
                     "1 is the Goodman line itself, not a limit of topopt; there is no pass or fail and no exit code.")
    options.extend([f"`{opt('levels')}` (several mesh sizes) to see how n and the bore statistics change with the mesh. A smaller mesh size is not "
                    "a sure cure: the scatter of the stress that is not smoothed changes with the mesh realization.",
                    "The facet knobs of `bore_checks` (radius_tolerance_mm, end_margin_mm, normal_limit, sample_band_mm) change the bore set; "
                    "`bore_facts` gives the numbers that show why a value moves.",
                    "Remove the `bore_checks` block from the YAML (or use `--no-bore-checks`) to leave the check off."])
    service_out = {c: {b.name: dict(service[c][b.name]) for b in bores} for c in cases}
    return dict(kind=KIND, status="done", reason=None, rule=RULE, statistics=list(STATISTICS), mean_stress=ms, materials=mats,
                service_stress=service_out, rows=rows, missing_rows=missing,
                lowest_n=None if lowest is None else dict(id=lowest[1], value=lowest[0]), facts=facts, options=options)
