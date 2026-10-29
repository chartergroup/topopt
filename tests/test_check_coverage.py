"""`topopt.check_coverage` (TC-2, CONTRACTS.md section 44): the record of the sub-checks that ran, failed or were not run.

These tests use made-up records (the status keys that the real records have), so they run without gmsh. The real runs of `topopt verify`
are in `test_verify_coverage.py`, the print report and the print check in `test_print_report.py` and `test_printcheck.py`.
"""
from types import SimpleNamespace


from topopt import check_coverage as cc
from topopt.check_coverage import FAILED, NOT_RUN, RAN, Entry



# --------------------------------------------------------------------------- #
# build, attach, line
# --------------------------------------------------------------------------- #

def test_every_check_is_in_exactly_one_list_and_a_clean_report_is_not_partial():
    partial, cov = cc.build([Entry("a", RAN), Entry("b", RAN), Entry("c", NOT_RUN, False, "the YAML has no c")])
    assert partial is False
    assert cov == {"ran": ["a", "b"], "failed": [], "not_run": [{"name": "c", "reason": "the YAML has no c", "asked": False}]}


def test_a_failed_check_makes_the_report_partial_with_its_reason():
    partial, cov = cc.build([Entry("a", RAN), Entry("b", FAILED, True, "ValueError: boom")])
    assert partial is True and cov["failed"] == [{"name": "b", "reason": "ValueError: boom"}] and cov["ran"] == ["a"]


def test_a_check_that_was_asked_and_not_run_makes_the_report_partial_and_one_that_was_not_asked_does_not():
    assert cc.build([Entry("a", RAN), Entry("b", NOT_RUN, True, "memory")])[0] is True
    assert cc.build([Entry("a", RAN), Entry("b", NOT_RUN, False, "not in the YAML")])[0] is False


def test_a_missing_reason_is_said_not_left_out():
    _, cov = cc.build([Entry("a", FAILED), Entry("b", NOT_RUN)])
    assert cov["failed"][0]["reason"] == "no reason recorded" and cov["not_run"][0]["reason"] == "no reason recorded"


def test_attach_puts_the_keys_after_the_named_key_and_keeps_the_others_in_order():
    out = cc.attach({"format": 1, "kind": "x", "ok": True, "z": 3}, [Entry("a", RAN)], after="kind")
    assert list(out) == ["format", "kind", "partial", "coverage", "ok", "z"] and out["partial"] is False
    assert list(cc.attach({"ok": True}, [Entry("a", RAN)], after="nope"))[:2] == ["partial", "coverage"]       # absent anchor: at the front
    src = {"ok": True}
    cc.attach(src, [Entry("a", RAN)], after="ok")
    assert src == {"ok": True}                                                                                # a copy


def test_the_line_of_a_partial_report_names_the_failed_and_the_unrun_checks_with_the_first_sentence_of_the_reason():
    partial, cov = cc.build([Entry("stress", RAN), Entry("bore_facts", FAILED, True, "bore 'a' stopped at the build: ValueError: x. More text."),
                             Entry("pressfit", NOT_RUN, True, "The press-fit stage makes its own factors. A second sentence."),
                             Entry("gate", NOT_RUN, False, "the call did not ask for the gate"),
                             Entry("motion", NOT_RUN, False, "the YAML has no rows")])
    assert cc.line(partial, cov) == ("partial: 1 of 3 checks failed (bore_facts: bore 'a' stopped at the build: ValueError: x); "
                                     "1 of 3 checks not run (pressfit: The press-fit stage makes its own factors); ran: stress; 2 not asked")


def test_the_line_of_a_clean_report_counts_what_ran_and_names_no_check_that_was_not_asked():
    partial, cov = cc.build([Entry("stress", RAN), Entry("goodman", RAN), Entry("buckling", NOT_RUN, False, "off")])
    text = cc.line(partial, cov)
    assert text == "partial: no; 2 of 2 checks ran (stress, goodman); 1 not asked" and "buckling" not in text


def test_line_of_reads_a_report_and_gives_none_for_a_report_of_an_older_version():
    assert cc.line_of({"partial": False, "coverage": {"ran": ["a"], "failed": [], "not_run": []}}) == "partial: no; 1 of 1 checks ran (a)"
    assert cc.line_of({"cases": {}}) is None and cc.line_of(None) is None and cc.line_of({"partial": True}) is None


# --------------------------------------------------------------------------- #
# topopt verify: the records that already exist
# --------------------------------------------------------------------------- #

def plan(*, joints=(), springs=(), bore=None, buckling=False, gate=False, rows=False):
    """The parts of a `Plan` that `from_verify` reads. `bore` is `(goodman, press_fit)` for a bore_checks block, or None."""
    bc = None if bore is None else SimpleNamespace(goodman=bore[0], press_fit=bore[1])
    row = SimpleNamespace(sense="abs", name="tip", limit_mm=0.02, cases=("pull",))      # a displacement row, as `rows_motion.row_specs` reads it
    limits = SimpleNamespace(displacement=[row] if rows else [], alignment=[])
    cfg = SimpleNamespace(joints=tuple(joints), springs=tuple(springs), bore_checks=bc, limits=limits)
    return SimpleNamespace(info=SimpleNamespace(cfg=cfg), bore_checks=bc is not None, buckling=buckling,
                           gate_limits={} if gate else None)


def by_name(entries):
    return {e.name: e for e in entries}


def test_a_plain_run_has_stress_as_the_only_check_that_was_asked():
    es = by_name(cc.from_verify({}, [dict(size_mm=3.0)], plan()))
    assert es["stress"].outcome == RAN
    assert [n for n, e in es.items() if e.asked] == ["stress"] and all(e.outcome == NOT_RUN for n, e in es.items() if n != "stress")
    assert es["bore_facts"].reason == "the YAML has no bore_checks block" and es["gate"].reason == "the call did not ask for the gate"
    assert cc.build(es.values())[0] is False


def test_buckling_done_is_ran_and_skipped_for_memory_is_asked_and_not_run():
    done = dict(status="done", cases=[dict(name="hang", status="ok", load_factors=[3.0]), dict(name="pull", status="no_positive_factor", load_factors=[])])
    assert by_name(cc.from_verify({}, [dict(size_mm=3.0, buckling=done)], plan(buckling=True)))["buckling"].outcome == RAN
    skipped = dict(status="skipped", reason="memory", cases=[], facts=["The memory estimate with the check is 9 GB, above the limit 8 GB. The check is not started."])
    e = by_name(cc.from_verify({}, [dict(size_mm=3.0, buckling=skipped)], plan(buckling=True)))["buckling"]
    assert (e.outcome, e.asked) == (NOT_RUN, True) and e.reason.startswith("The memory estimate with the check is 9 GB")


def test_buckling_error_record_is_failed_with_its_message():
    rec = dict(status="error", reason="exception", message="RuntimeError: phase 33 error -4", facts=["The check stopped: RuntimeError: phase 33 error -4"])
    e = by_name(cc.from_verify({}, [dict(size_mm=3.0, buckling=rec)], plan(buckling=True)))["buckling"]
    assert (e.outcome, e.reason) == (FAILED, "RuntimeError: phase 33 error -4")


def test_buckling_that_is_done_but_has_a_case_that_failed_or_did_not_finish_is_not_ran():
    case_error = dict(status="done", cases=[dict(name="a", status="ok", load_factors=[2.0]), dict(name="b", status="error", message="ValueError: x", load_factors=[])])
    e = by_name(cc.from_verify({}, [dict(size_mm=3.0, buckling=case_error)], plan(buckling=True)))["buckling"]
    assert (e.outcome, e.reason) == (FAILED, "case b: ValueError: x")
    slow = dict(status="done", cases=[dict(name="a", status="time_budget", load_factors=[], facts=["The eigen solve stopped at the time budget of 5 s after 12 solves."])])
    e = by_name(cc.from_verify({}, [dict(size_mm=3.0, buckling=slow)], plan(buckling=True)))["buckling"]
    assert (e.outcome, e.reason) == (NOT_RUN, "case a: The eigen solve stopped at the time budget of 5 s after 12 solves")
    rigid = dict(status="done", cases=[dict(name="a", status="ok", load_factors=[2.0], rigid_body=dict(status="error", message="LinAlgError: y"))])
    e = by_name(cc.from_verify({}, [dict(size_mm=3.0, buckling=rigid)], plan(buckling=True)))["buckling"]
    assert e.outcome == FAILED and "the rigid-body test stopped: LinAlgError: y" in e.reason


def test_the_a5_rows_follow_the_status_of_their_record_and_are_not_asked_without_rows():
    skipped = dict(status="skipped", reason="rows_cannot_work", facts=["region 'tip' has no node on the tet model"], rows=[])
    e = by_name(cc.from_verify({}, [dict(size_mm=3.0, motion=skipped)], plan(rows=True)))["motion"]
    assert (e.outcome, e.asked, e.reason) == (NOT_RUN, True, "region 'tip' has no node on the tet model")
    assert by_name(cc.from_verify({}, [dict(size_mm=3.0)], plan()))["motion"].asked is False
    err = dict(status="error", message="ValueError: bad measure STL", rows=[])
    assert by_name(cc.from_verify({}, [dict(size_mm=3.0, motion=err)], plan(rows=True)))["motion"].outcome == FAILED


def test_a_bore_that_stopped_is_a_failure_and_a_bore_with_no_facet_is_not_run():
    ok = dict(name="a", status="ok", facets=40)
    err = dict(name="b", status="error", stage="build", message="ValueError: boom")
    empty = dict(name="c", status="empty", facets=0)
    pl = plan(bore=(False, False))
    assert by_name(cc.from_verify({}, [dict(size_mm=3.0, bore_facts=[ok])], pl))["bore_facts"].outcome == RAN
    e = by_name(cc.from_verify({}, [dict(size_mm=3.0, bore_facts=[ok, err, empty])], pl))["bore_facts"]
    assert e.outcome == FAILED and e.reason == "bore 'b' stopped at the build: ValueError: boom; bore 'c': no wall facet (see bore_facts)"
    e = by_name(cc.from_verify({}, [dict(size_mm=3.0, bore_facts=[ok, empty])], pl))["bore_facts"]
    assert (e.outcome, e.asked, e.reason) == (NOT_RUN, True, "bore 'c': no wall facet (see bore_facts)")
    assert "no sample point" in by_name(cc.from_verify({}, [dict(size_mm=3.0, bore_facts=[dict(name="d", status="empty", facets=7)])], pl))["bore_facts"].reason


def test_goodman_and_press_fit_are_asked_only_with_their_block_and_say_why_when_the_bore_checks_are_off():
    es = by_name(cc.from_verify({}, [dict(size_mm=3.0)], plan()))
    assert es["goodman"].reason == "the YAML has no bore_checks block" and es["pressfit"].asked is False
    es = by_name(cc.from_verify({}, [dict(size_mm=3.0, bore_facts=[dict(name="a", status="ok", facets=1)])], plan(bore=(True, False))))
    assert es["goodman"].asked is True and es["pressfit"].asked is False and es["pressfit"].reason == "the bore_checks block has no press_fit block"
    assert es["bore_facts"].outcome == RAN and es["goodman"].outcome == NOT_RUN and es["goodman"].reason == "the result has no record of this check"


def test_press_fit_skipped_for_a_joint_and_goodman_with_no_bore_are_asked_and_not_run():
    pf = dict(status="skipped", reason="supports", facts=["The press-fit stage makes its own factors of the stiffness matrix K. More."], variants=[])
    gd = dict(status="skipped", reason="no_bore", facts=["No bore has a sample set, so there is no stress to check."], rows=[])
    lv = dict(size_mm=3.0, pressfit=pf, goodman=gd, bore_facts=[dict(name="a", status="ok", facets=1)], supports=dict(kind="x"))
    es = by_name(cc.from_verify({}, [lv], plan(bore=(True, True), joints=[object()])))
    assert (es["pressfit"].outcome, es["pressfit"].asked) == (NOT_RUN, True) and es["pressfit"].reason.startswith("The press-fit stage makes its own factors")
    assert (es["goodman"].outcome, es["goodman"].reason) == (NOT_RUN, "No bore has a sample set, so there is no stress to check.")
    assert es["supports"].outcome == RAN
    assert cc.build(es.values())[0] is True


def test_the_gate_is_ran_when_asked_and_the_result_has_it():
    assert by_name(cc.from_verify({"gate": {}}, [dict(size_mm=3.0)], plan(gate=True)))["gate"].outcome == RAN
    e = by_name(cc.from_verify({}, [dict(size_mm=3.0)], plan(gate=True)))["gate"]
    assert (e.outcome, e.asked) == (NOT_RUN, True)
    assert by_name(cc.from_verify({}, [dict(size_mm=3.0)], plan()))["gate"].asked is False


def test_with_several_mesh_sizes_the_worst_level_decides_and_the_reason_names_the_size():
    done = dict(status="done", cases=[dict(name="a", status="ok", load_factors=[2.0])])
    skipped = dict(status="skipped", reason="memory", cases=[], facts=["Above the limit."])
    levels = [dict(size_mm=4.0, buckling=done), dict(size_mm=2.0, buckling=skipped)]
    e = by_name(cc.from_verify({}, levels, plan(buckling=True)))["buckling"]
    assert (e.outcome, e.reason) == (NOT_RUN, "mesh size 2 mm: Above the limit.")
    err = dict(status="error", message="RuntimeError: z", cases=[])
    e = by_name(cc.from_verify({}, [dict(size_mm=4.0, buckling=err), dict(size_mm=2.0, buckling=skipped)], plan(buckling=True)))["buckling"]
    assert e.outcome == FAILED and e.reason == "mesh size 4 mm: RuntimeError: z; mesh size 2 mm: Above the limit."
    assert by_name(cc.from_verify({}, [dict(size_mm=4.0, buckling=done), dict(size_mm=2.0, buckling=done)], plan(buckling=True)))["buckling"].outcome == RAN


# --------------------------------------------------------------------------- #
# The print report and the print check
# --------------------------------------------------------------------------- #

def check(cid, state="pass", value=None, limit=None, reason="r", **extra):
    return dict(id=cid, state=state, value=value if value is not None else {}, limit={"value": limit, "key": cid, "source": "yaml"}, reason=reason, **extra)


def test_printcheck_verdicts_are_ran_and_an_unknown_without_a_limit_is_ran_with_its_value():
    es = by_name(cc.from_printcheck([check("mesh_closed", "pass"), check("bodies", "fail", limit=1), check("overhang", "unknown", limit=None)]))
    assert all(e.outcome == RAN for e in es.values())
    assert cc.build(es.values())[0] is False


def test_printcheck_unknown_with_a_limit_is_asked_and_not_run_with_the_reason_of_the_check():
    es = by_name(cc.from_printcheck([check("wall_thickness", "unknown", value={"samples": 100}, limit=1.2, reason="the limit 1.2 mm is under 3 x the resolution")]))
    e = es["wall_thickness"]
    assert (e.outcome, e.asked, e.reason) == (NOT_RUN, True, "the limit 1.2 mm is under 3 x the resolution")
    assert cc.build(es.values())[0] is True


def test_printcheck_support_volume_for_a_printer_that_needs_no_support_is_not_asked():
    c = check("support_volume", "pass", value={"volume_mm3": None, "support_needed": False}, reason="printer_type sls needs no support: the powder bed holds the part")
    e = by_name(cc.from_printcheck([c]))["support_volume"]
    assert (e.outcome, e.asked) == (NOT_RUN, False) and "needs no support" in e.reason
    assert cc.build([e])[0] is False


def test_printcheck_wall_thickness_with_no_chord_is_asked_and_not_run_and_a_check_with_an_error_is_failed():
    e = by_name(cc.from_printcheck([check("wall_thickness", "unknown", value={"samples": 0})]))["wall_thickness"]
    assert (e.outcome, e.asked, e.reason) == (NOT_RUN, True, "no chord could be measured on this mesh")
    e = by_name(cc.from_printcheck([check("overhang", "unknown", error="RuntimeError: rtree is gone")]))["overhang"]
    assert (e.outcome, e.reason) == (FAILED, "RuntimeError: rtree is gone")


def test_print_section_lists_its_five_checks_and_the_power_bed_printer_does_not_ask_for_support():
    section = dict(ok=True, support_needed=True, thinnest_member=dict(samples=500, thinnest_member_mm=1.1))
    es = by_name(cc.from_print_section(section))
    assert list(es) == ["bed_fit", "overhang", "plate_contact", "support_volume", "thinnest_member"] and all(e.outcome == RAN for e in es.values())
    section = dict(ok=True, support_needed=False, support_note="no support needed: the powder bed supports the part", thinnest_member=dict(samples=500))
    e = by_name(cc.from_print_section(section))["support_volume"]
    assert (e.outcome, e.asked, e.reason) == (NOT_RUN, False, "no support needed: the powder bed supports the part")
    assert cc.build(cc.from_print_section(section))[0] is False


def test_print_section_thinnest_member_with_an_error_or_no_chord_makes_it_partial():
    es = by_name(cc.from_print_section(dict(support_needed=True, thinnest_member=dict(samples=0, error="MemoryError: x"))))
    assert (es["thinnest_member"].outcome, es["thinnest_member"].reason) == (FAILED, "MemoryError: x")
    es = by_name(cc.from_print_section(dict(support_needed=True, thinnest_member=dict(samples=0, thinnest_member_mm=None))))
    assert (es["thinnest_member"].outcome, es["thinnest_member"].asked) == (NOT_RUN, True)


def test_a_print_section_that_could_not_be_made_is_one_failed_check():
    partial, cov = cc.build(cc.from_failed_print_section("ValueError:  bad   mesh"))
    assert partial is True and cov == {"ran": [], "failed": [{"name": "print_report", "reason": "ValueError: bad mesh"}], "not_run": []}
