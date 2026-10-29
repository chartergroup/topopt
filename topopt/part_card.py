"""The part card: one page, `result/report.md`, for a finished run (CONTRACTS.md section 9.2).

The card is a view of files that the run folder already has: `result/metrics.json`, `run.json`, `verdict.json`,
`history.jsonl`, the case file of the problem YAML (`<stem>.case.json`), `verify/verify.json` and the PNG files of
`renders/`. It computes no new result and it sets no threshold. The only arithmetic is a unit change (a fraction to a
percent). A fact that a file does not hold is written as "not known" or "not run", with the step that gives it.

Rules:
- The card reads files only. It reads every key with `.get()`, so a key that a later build adds shows only when it is there,
  and a file from an older build gives "not known" and no error.
- The same run folder gives the same bytes. The card has no time of writing. The date in it is `started_at` of `run.json`.
- No absolute path is written. A link is relative to the folder of the card (`result/`).
- A section that fails to build gives one line that names the error. The other sections stay.
- `write` writes the file only when its bytes change, with `runinfo.write_atomic`. It raises on a write error.
  `write_logged` never raises: it logs one line and returns None. The post step uses it, so a card never fails a run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import quote

from .volume_basis import mass_asked_line

CARD_FILE = "report.md"            # result/report.md
NOT_KNOWN = "not known"
VERIFY_JSON = "verify/verify.json"
RESULT_MESHES = (                  # file in result/, what it is (the first text is used when no nominal.stl exists)
    ("final.stl", "the printable mesh in mm"),
    ("nominal.stl", "the finished shape without machining stock"),
    ("part_only.stl", "the part without the bodies that are not printed"),
    ("final.step", "final.stl as a faceted STEP solid"),
    ("part_only.step", "part_only.stl as a faceted STEP solid"),
)
OTHER_FILES = (                    # path from the run folder, what it is
    ("result/metrics.json", "all numbers of the run"),
    ("verify/verify.json", "the result of the independent check"),
    ("run.log", "the log of the run"),
)


class NoResult(ValueError):
    """The run folder has no `result/metrics.json`: there is nothing to write a card about."""


# --- small helpers ------------------------------------------------------------------------------

def card_path(run_dir) -> Path:
    return Path(run_dir) / "result" / CARD_FILE


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _n(v, spec: str = ".10g", unit: str = "") -> str:
    """A number as text (no trailing zeros, no `-0`) and its unit, or NOT_KNOWN (with no unit) for anything else."""
    if not _is_num(v):
        return NOT_KNOWN
    return format(0.0 if v == 0 else v, spec) + (" " + unit if unit else "")


def _vec(v, unit: str = "") -> str:
    if not isinstance(v, (list, tuple)) or not all(_is_num(x) for x in v):
        return NOT_KNOWN
    return "[" + ", ".join(_n(x) for x in v) + "]" + (" " + unit if unit else "")


def _text(v) -> str:
    """One line of text: the white space is one space, and a missing value is NOT_KNOWN."""
    s = " ".join(str(v).split()) if v is not None else ""
    return s or NOT_KNOWN


def _cell(v) -> str:
    return _text(v).replace("|", "\\|")


def _yes(v) -> str:
    return "yes" if v is True else "no" if v is False else NOT_KNOWN


def _dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _list(v) -> list:
    return v if isinstance(v, list) else []


class _Files:
    """The JSON files of one run folder, each read once. A file that is not there or does not parse gives None and is recorded."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.read: list[str] = []
        self.missing: list[str] = []
        self.unreadable: list[str] = []
        self.metrics = self.json("result/metrics.json")
        self.run = self.json("run.json")
        self.verdict = _dict(self.metrics).get("verdict")      # metrics.json holds the verdict; verdict.json is the fallback
        if not isinstance(self.verdict, dict):
            self.verdict = self.json("verdict.json")
        self.verify = self.json(VERIFY_JSON)
        self.history = self.rows("history.jsonl")

    def json(self, rel: str):
        path = self.dir / rel
        if not path.is_file():
            self.missing.append(rel)
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.unreadable.append(rel)
            return None
        if not isinstance(data, dict):
            self.unreadable.append(rel)
            return None
        self.read.append(rel)
        return data

    def rows(self, rel: str) -> list[dict]:
        from .runinfo import read_history
        path = self.dir / rel
        if not path.is_file():
            self.missing.append(rel)
            return []
        try:
            rows = read_history(path, strict=False).rows
        except (OSError, ValueError):
            self.unreadable.append(rel)
            return []
        self.read.append(rel)
        return rows


# --- sections: each returns the lines after its heading ------------------------------------------

def _part(f: _Files) -> list[str]:
    run, m, v = _dict(f.run), _dict(f.metrics), _dict(f.verdict)
    from .runinfo import run_yaml_path
    yaml = run_yaml_path(run, f.dir)
    out = [f"- Name: {_text(run.get('name')) if run else NOT_KNOWN + ' (run.json is missing)'}",
           f"- Problem file: {yaml.name if yaml else NOT_KNOWN}",
           f"- Started: {_text(run.get('started_at'))}",
           f"- Run time: {_n(m.get('run_time_s'), unit='s')}",
           f"- Model: voxel size {_n(run.get('h'), unit='mm')}, solver {_text(m.get('solver') or run.get('solver'))}, "
           f"mesh method {_text(_dict(m.get('mesh')).get('method') or run.get('mesh_method'))}"]
    out.append(f"- Result: {result_sentence(v, v.get('iterations', m.get('iterations')))}")
    if v.get("text"):
        out.append(f"- Verdict text of the run: {_text(v.get('text'))}")
    return out


def result_sentence(v: dict, planned=None) -> str:
    """How the run ended, in plain words, from the verdict `v` (`verdict.json`) and the planned passes. The part card says it, and
    run_result says it as `end_fact` for a run that stopped at its pass limit (#380): one writer, so the two texts stay the same."""
    v = _dict(v)
    state, last = v.get("state"), v.get("final_iteration")
    where = f"pass {_n(last)} of {_n(v.get('iterations') if planned is None else planned)}"
    settled, feasible = v.get("converged"), v.get("feasible")
    tests = f" The objective had settled: {_yes(settled)}. Every constraint row is met: {_yes(feasible)}."
    if state == "converged":
        return f"The run converged at the last pass ({where}): the objective settled and every constraint row is met."
    if state == "iteration_limit":
        return f"The run stopped at its pass limit ({where})." + tests
    if state == "stopped_by_rule":
        return f"A stop rule ended the run before its pass limit ({where})." + tests
    if state == "stopped_by_request":
        return f"A request to stop ended the run ({where})." + tests
    if state == "unfinished":
        return f"The run did not reach its last pass ({where}) and no stop rule fired."
    return f"The verdict is {NOT_KNOWN}."


def _fine(f: _Files) -> list[str] | None:
    """The fine pass (#318, CONTRACTS.md section 24.6): a run that is a fine pass says where it started from and how it ended, and a run whose
    voxel is too coarse for the thinnest feature of the part carries the recommendation (the words of `fine_pass.fact_text`, written by the run).
    None for any other run: the section is not in the card."""
    run, v = _dict(f.run), _dict(f.verdict)
    rec, block = _dict(run.get("fine_from")), _dict(run.get("part_voxel"))
    out = []
    if rec:
        parent = Path(str(rec.get("run_dir") or "")).name or NOT_KNOWN         # the name of the folder only: the card has no absolute path
        out.append(f"- This run is a fine pass. It started from the design of the run folder `{_text(parent)}` (pass {_n(rec.get('iteration'))}, "
                   f"voxel {_n(rec.get('parent_voxel_mm'), unit='mm')}) on a voxel of {_n(rec.get('voxel_mm'), unit='mm')}"
                   + (" (the voxel that the part needs)" if rec.get("voxel_from") == "part_voxel" else " (given with --fine-voxel)") + ".")
        vol, mapped = _dict(rec.get("volume")), _dict(rec.get("map"))
        out.append(f"- The design of the parent was mapped by place onto {_n(mapped.get('points'))} design voxels (source {_text(rec.get('source'))}); "
                   f"the mean density was fitted to the volume target with a shift of {_n(vol.get('shift'), '.4g')} of the design variable "
                   f"(mean density {_n(vol.get('mean_before'), '.5g')} before, {_n(vol.get('mean_after'), '.5g')} after).")
        state, stop = v.get("state"), _dict(v.get("stop"))
        if state == "stopped_by_rule" and stop.get("rule") == "settled":
            ended = f"the loss settled at pass {_n(v.get('final_iteration'))}"
        elif state == "iteration_limit" and not rec.get("stop_rule"):
            ended = f"it ended at pass {_n(v.get('final_iteration'))}"
        elif state == "iteration_limit":
            ended = f"the maximum of passes ended it at pass {_n(v.get('final_iteration'))} before the loss settled"
        else:
            ended = f"state {_text(state)}"
        n_passes = run.get("iterations", rec.get("max_passes"))   # a continuation of a fine pass has more passes than the record's maximum
        if rec.get("stop_rule"):
            out.append(f"- The fine pass ends when the loss has settled (objective change at or below {_n(rec.get('tolerance'), '.4g')} over "
                       f"{_n(rec.get('window'))} passes), or after {_n(n_passes)} passes at most. Here: {ended}.")
        else:
            out.append(f"- All stop rules were off (--no-stop-rules), so the fine pass ran to its maximum of {_n(n_passes)} passes. Here: {ended}.")
    if block.get("recommend") and block.get("fact"):
        out.append("- " + _text(block["fact"]))
    return out or None


def _mass(f: _Files) -> list[str]:
    run, m = _dict(f.run), _dict(f.metrics)
    derived = _dict(run.get("derived"))
    block = _dict(derived.get("volume")).get("active_block_mm3")
    before_vol = f" and {_n(block, unit='mm3')}" if _is_num(block) else ""
    out = [f"- The design space and its other active regions (fixed, load, solid), all solid (before): {_n(m.get('mass_design_space_g'), unit='g')}{before_vol}",
           f"- Part, as in `final.stl` (after): {_n(m.get('mass_final_g'), unit='g')} and {_n(m.get('volume_final_mm3'), unit='mm3')}",
           f"- Mass saved against the design space: {_n(m.get('mass_saved_pct'), unit='%')}"]
    asked = mass_asked_line(m)   # a run with settings.target_mass_g (I7): the same fact line as the run log and run_status
    if asked:
        out.append(f"- {asked}")
    density = derived.get("density_g_cm3")
    if _is_num(density):
        out.append(f"- Density used for the mass: {_n(density, unit='g/cm3')} (from the YAML)")
    mesh = _dict(m.get("mesh"))
    out.append(f"- Mesh: {_n(m.get('triangles'))} triangles, watertight: {_yes(m.get('watertight'))}, "
               f"method {_text(mesh.get('method'))}")
    po = _dict(m.get("part_only"))
    if po.get("applied") is True:
        removed = ", ".join(_text(x) for x in _list(po.get("removed"))) or NOT_KNOWN
        out.append(f"- Part without the bodies that are not printed (`part_only.stl`): {_n(po.get('mass_g'), unit='g')} and "
                   f"{_n(po.get('volume_after_mm3'), unit='mm3')}; left out: {removed}")
    if m.get("interfaces"):
        out.append("- Machining stock: `final.stl` is the as-printed mesh (with stock); `nominal.stl` is the finished shape.")
    return out


def _force_text(entry: dict) -> str:
    kind = entry.get("kind", "force")
    where = _text(entry.get("region"))
    if kind == "pressure":
        return f"{where}: pressure {_n(entry.get('pressure_mpa'), unit='MPa')} on {_n(entry.get('area_mm2'), unit='mm2')}"
    text = f"{where}: {_vec(entry.get('force'), 'N')}"
    if kind != "force":
        text += f" ({_text(kind)}"
        if entry.get("moment") is not None:
            text += f", moment {_vec(entry.get('moment'), 'N mm')}"
        text += ")"
    return text


def _stiffness(f: _Files) -> list[str]:
    run, m = _dict(f.run), _dict(f.metrics)
    cases = [c for c in _list(run.get("load_cases")) if isinstance(c, dict)]
    stiff = {s.get("name"): s for s in _list(m.get("stiffness_cases")) if isinstance(s, dict)}
    if not cases:   # no run.json: the names come from the metrics
        cases = [{"name": n} for n in stiff]
    sizes = {c.get("name"): c.get("total_magnitude_n")
             for c in _list(_dict(run.get("derived")).get("load_cases")) if isinstance(c, dict)}
    robust = _dict(m.get("robust"))
    if robust.get("on") and robust.get("complete") and isinstance(robust.get("compliance_cases_nominal"), list):
        comp = robust["compliance_cases_nominal"]
    else:
        last = _dict(f.history[-1]) if f.history else {}
        comp = last.get("compliance_cases") if isinstance(last.get("compliance_cases"), list) else []
    out = ["Stiffness is the load per mm of motion at the load (N/mm): a larger value is stiffer. "
           "Compliance is the work of the load (N mm): a smaller value is stiffer.", "",
           "| Load case | Force | Stiffness (N/mm) | Compliance at the end (N mm) |", "|---|---|---|---|"]
    for i, case in enumerate(cases):
        name = case.get("name")
        forces = [x for x in _list(case.get("forces")) if isinstance(x, dict)]
        force = "; ".join(_force_text(x) for x in forces) if forces else NOT_KNOWN
        if _is_num(sizes.get(name)):
            force += f" (size {_n(sizes[name], unit='N')})"
        s = _dict(stiff.get(name))
        if _is_num(s.get("N_per_mm")):
            value = _n(s["N_per_mm"])
        else:
            value = NOT_KNOWN + (f" ({_text(s.get('reason'))})" if s.get("reason") else "")
        c = _n(comp[i], ".4g") if i < len(comp) else NOT_KNOWN
        out.append(f"| {_cell(name)} | {_cell(force)} | {_cell(value)} | {c} |")
    out += ["", f"- Weighted compliance of all load cases (the objective): {_n(m.get('compliance_initial'), '.4g')} at the start, "
                f"{_n(m.get('compliance_final'), '.4g')} at the end (N mm)"]
    if robust.get("on"):
        out.append("- Robust mode is on: stiffness and compliance are for the nominal design, the part in `final.stl`.")
    return out


def _brief(f: _Files) -> list[str]:
    run = _dict(f.run)
    from . import case
    from .runinfo import run_yaml_path
    yaml = run_yaml_path(run, f.dir)
    yaml_assumptions = [a for a in _list(run.get("assumptions")) if isinstance(a, str)]
    if yaml is None:
        return [f"- Brief: {NOT_KNOWN} (run.json is missing or has no yaml_path)."]
    manifest = case.manifest_path(yaml)
    read = case.read_case(manifest, yaml.name)
    (f.read if read.kind == "ok" else f.missing).append(manifest.name)
    brief = _dict(_dict(read.doc).get("brief"))
    if not brief:
        if read.kind == "ok":
            why = "The case file has no brief."
        elif read.facts:
            why = _text("; ".join(read.facts)) + "."
        else:
            why = "There is no case file beside the problem YAML."
        out = [f"- No brief. {why} `record_brief` records the requirement and its assumptions with their origin."]
        if yaml_assumptions:
            out.append("- The YAML lists these assumptions (the origin is not recorded):")
            out += [f"  - {_text(a)}" for a in yaml_assumptions]
        return out
    out = [f"- Requirement: {_text(brief.get('text'))}",
           f"- Brief {_n(brief.get('id'))} recorded: {_text(brief.get('recorded_at'))}"]
    mine = None   # the run entry of this folder: it tells which brief the run started with
    for entry in (e for e in _list(_dict(read.doc).get("runs")) if isinstance(e, dict)):   # `runs` of the case file (CONTRACTS.md section 30)
        try:
            if entry.get("run_dir") is not None and (yaml.parent / entry["run_dir"]).resolve() == f.dir.resolve():
                mine = entry
        except (OSError, TypeError, ValueError):
            continue
    if mine is not None and "brief" in mine and mine.get("brief") != brief.get("id"):
        used = mine.get("brief")
        out.append("- This run started " + ("without a brief." if used is None else f"with brief {_n(used)}, not with this one.")
                   + " The brief above is the one now in the case file.")
    for key, title in (("assumptions", "Assumptions"), ("criteria", "Criteria")):
        items = [x for x in _list(brief.get(key)) if isinstance(x, dict)]
        out.append(f"- {title}:" if items else f"- {title}: none recorded")
        out += [f"  - {_text(x.get('text'))} (origin: {_text(x.get('origin'))})" for x in items]
    return out


def _print(f: _Files) -> list[str]:
    p = _dict(_dict(f.metrics).get("print"))
    how = "`print_check` (MCP) or `topopt printcheck <run folder>` judge a part against print limits."
    if not p:
        return [f"- Not run: `metrics.json` has no print block (a run from an older build). {how}"]
    if p.get("ok") is False:
        return [f"- The print report failed: {_text(p.get('error'))}. {how}"]
    s, c, t = _dict(p.get("settings")), _dict(p.get("configured")), _dict(p.get("thinnest_member"))
    bed = [x for x in _list(s.get("bed_mm")) if _is_num(x)]
    out = [f"- Settings: printer {_text(s.get('printer_type'))}, build direction {_text(s.get('build_direction_label'))}, "
           f"bed {(' x '.join(_n(x) for x in bed) + ' mm') if bed else NOT_KNOWN}, self-supporting angle "
           f"{_n(s.get('self_supporting_angle_deg'), unit='degrees')}"]
    if _list(s.get("assumed")):
        out.append(f"- Assumed, not in the YAML: {', '.join(_text(x) for x in s['assumed'])}")
    out.append(f"- Bed fit: {_yes(c.get('bed_fit'))}; part size in the build frame {_vec(c.get('extents_mm'), 'mm')}; "
               f"spare length {_n(c.get('bed_margin_mm'), unit='mm')}")
    if p.get("support_needed") is False:
        out.append(f"- Support: not needed for this printer type ({_text(p.get('support_note'))})")
    else:
        pct = c.get("overhang_fraction")
        share = f" ({_n(pct * 100, '.1f', '%')} of the surface)" if _is_num(pct) else ""
        out.append(f"- Overhang area: {_n(c.get('overhang_area_mm2'), unit='mm2')}{share}; plate contact "
                   f"{_n(c.get('plate_contact_area_mm2'), unit='mm2')}; support volume estimate {_n(c.get('support_volume_mm3'), unit='mm3')} "
                   "(an upper bound)")
    below, below_pct = _dict(t.get("area_below_mm2")), _dict(t.get("area_below_pct"))
    out.append(f"- Thinnest member (an estimate): {_n(t.get('thinnest_member_mm'), unit='mm')}; surface thinner than 2.0 mm: "
               f"{_n(below.get('2.0'), unit='mm2')} ({_n(below_pct.get('2.0'), unit='%')})")
    return out


def _verify(f: _Files) -> list[str]:
    v = _dict(f.verify)
    how = "`verify_run` (MCP) or `topopt verify <run folder>` runs an independent stress check on a tet mesh."
    if not v:
        if VERIFY_JSON in f.unreadable:
            return [f"- `{VERIFY_JSON}` is there and cannot be read."]
        return [f"- Not run: the run folder has no `{VERIFY_JSON}`. {how}"]
    if v.get("ok") is not True or v.get("estimate_only") is True:
        return [f"- `{VERIFY_JSON}` holds no result (ok: {_yes(v.get('ok'))}, estimate only: {_yes(v.get('estimate_only'))}). {how}"]
    run, mesh = _dict(v.get("run")), _dict(v.get("mesh"))
    name = run.get("verified_file") if isinstance(run.get("verified_file"), str) else "final.stl"
    out = [f"- Checked file: `{_text(name)}`; mesh size {_n(v.get('reference_level_mm'), '.4g', 'mm')}, {_n(mesh.get('tets'))} tets, "
           f"solver {_text(_dict(v.get('backend')).get('solver'))}"]
    out.append(f"- {_same_mesh(f.dir / 'result' / Path(name).name, run.get('final_stl_sha256'))}")
    limit, cases = _dict(v.get("limit")), _dict(v.get("cases"))
    lcases = _dict(limit.get("cases"))
    heads = ["Load case", "Stress p99.9 by volume (MPa)", "Peak (MPa)", "Largest motion (mm)", "Stiffness of the tet model (N/mm)",
             "Stiffness of the optimizer (N/mm)"] + (["Margin (allowable / p99.9)"] if lcases else [])
    out += ["", "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for cname in cases:
        c, vm = _dict(cases[cname]), _dict(_dict(cases[cname]).get("von_mises"))
        row = [_cell(cname), _n(vm.get("p99_9_by_volume_MPa"), ".4g"), _n(vm.get("peak_qp_MPa"), ".4g"),
               _n(c.get("max_displacement_mm"), ".4g"), _n(c.get("stiffness_N_per_mm"), ".5g"),
               _n(c.get("optimizer_stiffness_N_per_mm"), ".5g")]
        if lcases:
            row.append(_n(_dict(lcases.get(cname)).get("margin"), ".4g"))
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    if limit:
        out.append(f"- Your allowable stress: {_n(limit.get('allowable_MPa'), unit='MPa')} at the percentile {_n(limit.get('percentile'))}; "
                   f"exceeded: {_yes(limit.get('exceeded'))}")
    else:
        out.append("- No allowable stress was given, so there is no pass or fail. The peak can move with the mesh at sharp features.")
    extra = [k for k in ("buckling", "motion", "goodman", "pressfit") if v.get(k)]
    if extra:
        out.append(f"- Also in `{VERIFY_JSON}`: {', '.join(extra)}")
    if _dict(v.get("yaml_vs_run")).get("matches") is False:
        out.append("- The YAML differed from `run.json` when verify ran (see `yaml_vs_run` in the file).")
    return out


def _same_mesh(path: Path, recorded) -> str:
    """One fact: is the file that verify meshed the file that is in `result/` now? (the SHA-256 of the file)"""
    from .runinfo import sha256_file
    if not isinstance(recorded, str):
        return "The check does not record the SHA-256 of its mesh file: whether it is the mesh now in `result/` is " + NOT_KNOWN + "."
    try:
        now = sha256_file(path)
    except OSError:
        return f"`{path.name}` is not in `result/` now: whether the check is of the mesh now there is {NOT_KNOWN}."
    if now == recorded:
        return f"The check is of the mesh now in `result/{path.name}` (same SHA-256)."
    return f"The check is NOT of the mesh now in `result/{path.name}` (another SHA-256): run verify again."


def mesh_fallback_lines(metrics) -> list[str]:
    """The fallbacks that the mesh step took in the volume calibration (`mesh.fallbacks`, #445), in the order of the list. The same step with the
    same error in several rounds of a pass is one line that names the rounds, so a step that fails in each round does not fill the card. Empty when none."""
    groups: dict = {}
    for e in _list(_dict(_dict(metrics).get("mesh")).get("fallbacks")):
        e = _dict(e)
        place = ("without_move" if e.get("build") == "without_move" else "setup" if e.get("round") is None else "round", _text(e.get("pass")))
        groups.setdefault((place, _text(e.get("step")), _text(e.get("error_type")), _text(e.get("error"))), []).append(e)
    out = []
    for ((kind, pass_no), step, error_type, error), es in groups.items():
        if kind == "without_move":
            place = "the build without the move"
        elif kind == "setup":
            place = "the setup"
        else:
            rounds = [_text(e.get("round")) for e in es]
            kept = [_text(e.get("round")) for e in es if e.get("kept")]
            place = (f"round {rounds[0]}" if len(rounds) == 1 else f"rounds {', '.join(rounds)}") + f" of pass {pass_no}"
            if kept:
                place += " (the kept round)" if len(rounds) == 1 else f" ({kept[0]} is the kept round)"
        out.append(f"the mesh step took a fallback in {place}, step {step}: {error_type}: {error}. The part can differ from a part with no fallback. "
                   "Option: check the part with `topopt verify` (MCP: verify_run), or report the error in an issue.")
    return out


def _other_checks(f: _Files) -> list[str]:
    m = _dict(f.metrics)
    out = []
    st = _dict(m.get("stress"))
    if st.get("error"):
        out.append(f"- Stress row: the report failed: {_text(st.get('error'))}")
    elif st:
        out.append("- Stress row of the optimizer (voxel estimates, not a tet check):")
        out += [f"  - {_text(name)}: {_text(_dict(c).get('summary'))}" for name, c in _dict(st.get("cases")).items()]
    bk = _dict(m.get("buckling"))
    if bk.get("status") == "done":
        low = _dict(bk.get("lowest_load_factor"))
        per = ", ".join(f"{_text(c.get('name'))} {_n(c['load_factors'][0], '.6g')}" for c in _list(bk.get("cases"))
                        if isinstance(c, dict) and _list(c.get("load_factors")) and _is_num(c["load_factors"][0]))
        out.append(f"- Buckling (linear, voxel estimate, not a verdict): lowest load factor {_n(low.get('value'), '.6g')} "
                   f"(case {_text(low.get('case'))})" + (f"; lowest of each case: {per}" if per else ""))
    elif bk:
        why = _text(bk.get("message") or "; ".join(str(x) for x in _list(bk.get("facts"))) or bk.get("reason"))
        out.append(f"- Buckling check: {_text(bk.get('status'))}: {why}")
    for line in _list(_dict(m.get("interfaces")).get("warnings")):
        out.append(f"- Interface warning: {_text(line)}")
    out += [f"- Mesh fallback: {line}" for line in mesh_fallback_lines(m)]
    return out or ["- None. The run has no stress row and no buckling result."]


def _lost_force(f: _Files) -> list[str]:
    lost = _dict(_dict(f.run).get("lost_force"))
    if not lost:
        return [f"- {NOT_KNOWN}: `run.json` has no `lost_force`."]
    rows = [c for c in _list(lost.get("cases")) if isinstance(c, dict) and _is_num(c.get("lost_magnitude_n")) and c["lost_magnitude_n"] > 0]
    if not rows:
        return ["- None. No load force sits on fixed nodes."]
    return [f"- Case {_text(c.get('name'))}: {_n(c['lost_magnitude_n'], '.4g', 'N')} of {_n(c.get('total_magnitude_n'), '.4g', 'N')} "
            f"is lost on nodes that a load region and a fixed region share ({_n(c.get('lost_percent'), '.3g', '%')})" for c in rows]


def _images(f: _Files) -> list[str]:
    found = []
    for folder in ("renders", "result"):
        found += sorted(p for p in (f.dir / folder).glob("*.png") if p.is_file()) if (f.dir / folder).is_dir() else []
    if not found:
        return ["- No image yet. `render_part` (MCP) or `topopt render <run folder>` writes PNG views to `renders/`. "
                "Write the card again to list them."]
    out = []
    try:   # modification times only: no threshold. They are not exact after a copy without `cp -p`, so the words are "may be"
        made = (f.dir / "result" / "final.stl").stat().st_mtime_ns
    except OSError:
        made = None
    for p in found:
        rel = ("../renders/" if p.parent.name == "renders" else "") + p.name
        try:
            older = made is not None and p.stat().st_mtime_ns < made
        except OSError:
            older = False
        line = f"- ![{p.name}]({quote(rel)})" if p.name.endswith("_montage.png") else f"- [{p.name}]({quote(rel)})"
        out.append(line + (" (may be from an earlier result: the file is older than `result/final.stl`)" if older else ""))
    out.append("- The images are the PNG files in `renders/` now. After a new run in this folder, render again.")
    return out


def _files(f: _Files) -> list[str]:
    result = f.dir / "result"
    out = []
    listed = [("result/" + name, what) for name, what in RESULT_MESHES] + list(OTHER_FILES)
    for rel, what in listed:
        if (f.dir / rel).is_file():
            if rel == "result/final.stl" and (result / "nominal.stl").is_file():
                what = "the as-printed mesh in mm (with machining stock)"
            link = rel[len("result/"):] if rel.startswith("result/") else "../" + rel   # the link starts at the folder of the card
            out.append(f"- [{rel}]({quote(link)}): {what}")
    return out or ["- No result file."]


def _not_checked(f: _Files) -> list[str]:
    m, v = _dict(f.metrics), _dict(f.verify)
    ran_verify = bool(v) and v.get("ok") is True and v.get("estimate_only") is not True
    stress_row = bool(_dict(m.get("stress"))) and not _dict(m.get("stress")).get("error")
    out = []
    if not stress_row and not ran_verify:
        out.append("- Stress: not checked. The run had no stress row and verify did not run.")
    elif not ran_verify:
        out.append("- Stress on a tet mesh: not checked. Only the stress row (voxel estimates) ran.")
    bk = _dict(m.get("buckling"))
    if bk and bk.get("status") != "done":
        out.append(f"- Buckling: no load factors, the check is {_text(bk.get('status'))} (see the section Other checks that ran).")
    elif not bk:
        out.append("- Buckling: not run. Set `buckling: {check: true}` in the YAML, or run `topopt buckling <run folder>`.")
    if not (ran_verify and v.get("goodman")):
        out.append("- Fatigue: not checked.")
    out.append("- The printed part: not checked. No print, no measurement and no load test is in these files.")
    out.append("- Print limits: not applied. The print facts above are numbers, not a pass or a fail.")
    if not ran_verify:
        out.append("- Independent check of the stiffness and the stress on a tet mesh: not run (see the section Verify).")
    return out


def _sources(f: _Files) -> list[str]:
    out = [f"- Read: {', '.join(f.read) if f.read else 'nothing'}"]
    if f.missing:
        out.append(f"- Not found: {', '.join(f.missing)}")
    if f.unreadable:
        out.append(f"- Not readable: {', '.join(f.unreadable)}")
    return out


SECTIONS = (
    ("The part", _part),
    ("Fine pass", _fine),
    ("Mass and volume", _mass),
    ("Stiffness", _stiffness),
    ("The brief", _brief),
    ("Print facts", _print),
    ("Verify (independent check)", _verify),
    ("Other checks that ran", _other_checks),
    ("Lost force", _lost_force),
    ("Images", _images),
    ("Files", _files),
    ("Not checked", _not_checked),
    ("Sources", _sources),
)


# --- the card -----------------------------------------------------------------------------------

def build(run_dir) -> str:
    """The text of the card for a run folder. Raises `NoResult` when `result/metrics.json` is not there."""
    run_dir = Path(run_dir)
    if not (run_dir / "result" / "metrics.json").is_file():
        raise NoResult(f"{run_dir.name} has no result/metrics.json: the run has no result")
    f = _Files(run_dir)
    run = _dict(f.run)
    lines = [f"# Part card: {_text(run.get('name') or run_dir.name)}", "",
             "This card collects the facts that the run folder had when it was written. Every number comes from a file of the "
             "run folder. The card computes no new result and sets no threshold. A fact that no file holds is marked "
             f"\"{NOT_KNOWN}\" or \"not run\". After a new fact file appears (verify, buckling, renders, STEP), write the card "
             "again: `topopt report <run folder>`, or the MCP tool `run_result`."]
    for title, section in SECTIONS:
        try:
            body = section(f)
        except Exception as exc:    # one section must not take the card down
            body = [f"- This section could not be written: {type(exc).__name__}: {_text(exc)}"]
        if body is None:            # a section that has nothing to say for this run (the fine pass) is not in the card
            continue
        lines += ["", f"## {title}", ""] + body
    return "\n".join(lines) + "\n"


def write(run_dir) -> Path:
    """Write `result/report.md` and return its path. The file is not touched when its bytes would not change.

    Raises `NoResult` (a run folder with no result) or `OSError` (the file cannot be written).
    """
    from .runinfo import write_atomic
    path = card_path(run_dir)
    data = build(run_dir).encode("utf-8")
    try:
        if path.is_file() and path.read_bytes() == data:
            return path
    except OSError:
        pass
    write_atomic(path, data)
    return path


def write_logged(run_dir, log=None) -> Path | None:
    """`write`, but it never raises. A failure is one line to `log` (a function that takes a text) and the result is None."""
    try:
        return write(run_dir)
    except Exception as exc:     # the run is done: a card problem must not break it
        if log is not None:
            log("warning: part card not written: " + " ".join(f"{type(exc).__name__}: {exc}".split()))
        return None
