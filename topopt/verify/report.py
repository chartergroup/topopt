"""Text for the user: the lines that `topopt verify` prints. CONTRACTS.md section 16."""
from __future__ import annotations

from .. import check_coverage
from . import memory
from . import part_only as vpart_only
from .gate import gate_lines, gate_verdict, replaced_sentences
from .names import opt


def yaml_note_lines(compared: dict) -> list[str]:
    """The `note:` lines of a `compare_yaml_run` result (also for `topopt refine` and `topopt buckling`). One text for all."""
    return [f"note: the YAML differs from run.json: {d}" for d in compared.get("differences", [])]


def solver_lines(b: dict) -> list[str]:
    """The factor that runs and how it was chosen (the `backend` block). A fact for the user, never a silent switch.

    With `auto`, one line for each factor that was skipped, with the reason (the note of that factor).
    """
    if not b["solver"]:
        return [f"solver:         none can run: {b['problem']}"]
    lines = [f"solver:         {b['solver']}  (requested {b['requested']})"]
    for name, note in (b.get("skipped") or {}).items():
        lines.append(f"                auto skipped {name}: {note}")
    return lines


def pressfit_estimate_text(pf: dict) -> str:
    """The line about the press-fit stage in the estimate: the variants, the one-factor rule and the time (None: no measurement for the factor).

    With a joint in the YAML (K1-A) the stage is skipped: the line says so, with the reason (`skipped_with_joints`).
    """
    if pf.get("skipped_with_joints"):
        return f"with the press-fit check: skipped, the YAML has a joint. {pf['skipped_with_joints']}"
    return (f"with the press-fit check: {pf['variants']} cup variant(s), one factor for each, one at a time (the memory estimate above holds)"
            + (f"; time about {pf['estimated_seconds']:.0f} s" if pf.get("estimated_seconds") is not None else "; no time estimate for this solver"))


def _r(values, digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


def bond_line(case: str, name: str, r: dict) -> str:
    """The line about one joint in one load case: the force and the moment of the bond on the part (6 numbers) and the point of the moment."""
    return (f"  case {case}: joint {name}: bond force {_r(r['bond_force_N'], 4)} N, "
            f"moment {_r(r['bond_moment_Nmm'], 3)} N mm about {_r(r['about_mm'], 2)} mm")


def supports_lines(sup: dict) -> list[str]:
    """The lines about the joints and the springs (the key `supports` of `verify.json`): for each load case, the 6 numbers of each bond
    (force, moment, point of the moment) and the force and the moment of each spring on the part, all from `supports.cases`, and the
    share of the strain energy that the bond holds (a fact; the engine sets no limit on it)."""
    lines = [f"supports: {len(sup['joints'])} joint(s) and {len(sup['springs'])} spring(s); the force (N) and the moment (N mm) of each on the part:"]
    for case, c in sup["cases"].items():
        for name, r in c["joints"].items():
            share = r.get("bond_energy_share")
            lines.append(bond_line(case, name, r) + (f"; the bond holds {100 * share:.2g} % of the strain energy" if share is not None else ""))
        for name, r in c["springs"].items():
            lines.append(f"  case {case}: spring {name}: force {_r(r['force_N'], 4)} N, moment {_r(r['moment_Nmm'], 3)} N mm about {_r(r['about_mm'], 2)} mm")
    return lines


def part_only_lines(po: dict) -> list[str]:
    """The lines about `--part-only` (the key `part_only` of `verify.json`): the file, one line for each removed body, the singular
    note and the notes. Before meshing the block has no facts of the bodies, so only the head and the notes are there."""
    names = vpart_only.titles(po["removed"])
    lines = [f"mesh file: {po['file']} (the part without the bodies of regions.not_printed: {', '.join(names) or 'none'}; final.stl holds them)"]
    lines += [vpart_only.say_line(r, name) for r, name in zip(po["removed"], names) if "boundary_condition" in r]
    if po.get("singular_note"):
        lines.append(f"note: {po['singular_note']}")
    lines += [f"note: {t}" for t in po.get("notes", [])]
    lines += [f"note: {d}" for d in po.get("yaml_vs_run", {}).get("differences", [])]
    return lines


def estimate_lines(res: dict) -> list[str]:
    """The lines of `--estimate-only` (and the head of a run): memory estimate for each mesh size."""
    lines = [f"run:            {res['run']['name']}  (voxel {res['run']['voxel_size_mm']:g} mm, band "
             f"{res['band_mm']:g} mm{' (default: 0.5 voxel)' if res['band_is_default'] else ''})",
             f"final STL:      volume {res['stl']['volume_mm3']:.1f} mm3, area {res['stl']['area_mm2']:.1f} mm2, "
             f"default mesh size {res['stl']['default_mesh_size_mm']:.3g} mm"]
    if res.get("part_only"):
        lines.extend(part_only_lines(res["part_only"]))
    b = res["backend"]
    lines.extend(solver_lines(b))
    avail = b.get("available")
    superlu_ok = True if avail is None else bool(avail.get("superlu", {}).get("available"))   # SuperLU instead: only if it can run
    ml = res["memory_limit"]
    lines.append("memory limit:   " + (f"{memory.gb_text(ml['limit_bytes'])} ({ml['limit_source']})"
                                       if ml["limit_bytes"] is not None else "not known here")
                 + (f"; available RAM {memory.gb_text(ml['available_bytes'])}" if ml["available_bytes"] is not None else ""))
    lines.extend(bore_checks_lines(res.get("bore_checks")))
    for e in res["estimates_before_meshing"]:
        rng = e.get("range_bytes")
        basis = (f"central value, no margin; range {memory.gb_text(rng['low'])} to {memory.gb_text(rng['high'])} from the "
                 f"measured spread" if rng else f"rule {memory.gb_text(e['rule_bytes'])} x 1.25")
        lines.append(f"mesh {e['size_mm']:g} mm:    about {e['estimated_dofs']:,} dofs, memory estimate "
                     f"{memory.gb_text(e['estimate_bytes'])} ({basis}; {e['estimate_model']})"
                     + ("  ABOVE THE LIMIT" if e["above_limit"] else ""))
        for text in e.get("notes", []):
            lines.append(f"                note: {text}")
        if "buckling_bytes" in e:
            lines.append(f"                with the buckling check: {memory.gb_text(e['estimate_with_buckling_bytes'])} "
                         f"(+{memory.gb_text(e['buckling_bytes'])} on top of the factor)"
                         + ("; above the limit: the check can be skipped after meshing (the decision uses the real counts of the "
                            f"mesh; {opt('force')} starts the check) and then the stress check goes on"
                            if e["above_limit_with_buckling"] and not e["above_limit"] and not res["memory_limit"].get("force") else ""))
        if "pressfit" in e:
            lines.append("                " + pressfit_estimate_text(e["pressfit"]))
        sl = (e.get("ranges_by_backend") or {}).get("superlu")
        if b["solver"] != "superlu" and sl and e["bytes_by_backend"].get("superlu") is not None and superlu_ok:
            lines.append(f"                memory estimate with SuperLU instead: {memory.gb_text(e['bytes_by_backend']['superlu'])} "
                         f"(range {memory.gb_text(sl['low'])} to {memory.gb_text(sl['high'])}; compactness C "
                         f"{e['compactness']:.3f}; the SuperLU fill grows fast for a compact part)")
        by = {k: v for k, v in e["seconds_by_backend"].items() if v is not None}
        if by:
            lines.append(f"                time estimate: " + ", ".join(
                f"{k} about {v:.0f} s" + (" (this run)" if k == b["solver"] else "") for k, v in by.items())
                + f" ({memory.TIME_HOST}; no measurement for the others)")
        else:
            lines.append("                time estimate: none for this solver (no measurement)")
    lines.extend(yaml_note_lines(res.get("yaml_vs_run", {})))
    lines.extend(f"note: {t}" for t in res.get("notes", []))
    return lines


def weld_line(w: dict) -> str:
    """One line for the step `weld_short_edges` of the surface preparation (`mesh.surface_weld_retry` of the result). It says that
    the verified surface is the input with its short edges welded, and what the weld did."""
    changed = [f"body {i + 1} from genus {g0} to {g1} (Euler characteristic {e0} to {e1})"
               for i, (g0, g1, e0, e1) in enumerate(zip(w["genus_before"], w["genus_after"], w["euler_before"], w["euler_after"]))
               if g0 != g1]
    return (f"surface: the prepared surface was refused for an edge under {w['limit_mm']:g} mm, so the input was welded and "
            f"prepared again: {w['edges_welded']:,} edges welded, {w['vertices_merged']:,} vertices merged and "
            f"{w['vertices_dropped']:,} dropped, largest move {w['max_vertex_move_mm']:.4f} mm (the weld radius is "
            f"{w['limit_mm']:g} mm), volume change {w['volume_change_mm3']:+.4f} mm3; "
            + (f"genus change: {', '.join(changed)}" if changed else "genus unchanged"))


def clean_line(c: dict) -> str:
    """One line for the step `clean_short_edges` of the surface preparation (`mesh.surface_clean_retry` of the result). It says that
    the verified surface is the prepared surface that was refused for a short edge, cleaned, and what the cleaning did. The weld part
    is left out when no edge was welded. The distance is exact for all the vertices that the cleaning removed (a vertex whose faces
    all go is not bound by the weld radius)."""
    def count(n, word, plural=None):
        return f"{n:,} {word if n == 1 else plural or word + 's'}"
    n = c["short_edges_before"]
    parts = [f"{c['short_edges_before'] - c['short_edges_after_collapse']:,} collapsed"]
    if c["edges_welded"]:
        parts.append(f"{c['edges_welded']:,} welded ({count(c['vertices_merged'], 'vertex', 'vertices')} merged, "
                     f"largest move {c['max_vertex_move_mm']:.4f} mm)")
    parts.append(f"{count(c['faces_before'] - c['faces_after'], 'face')} removed")
    repairs = c.get("fold_repairs") or []
    repaired = sum(r["fold_edges_made"] - r["fold_edges_left"] for r in repairs)
    if repaired:   # F4-27: the fold repair of the first try ran on the surface of the cleaning (review MN3: "made" was wrong)
        turned = sum(r.get("faces_turned", 0) for r in repairs)
        parts.append(f"{count(repaired, 'fold edge')} repaired"
                     + (f" (the repair turned {count(turned, 'face')} over)" if turned else ""))
    removed = c.get("removed_vertices")
    if removed is not None:
        parts.append(f"{count(removed, 'vertex', 'vertices')} removed (the farthest is {c['removed_vertex_distance_max_mm']:.4f} mm from "
                     "the cleaned surface)" if removed else "no vertex removed")
    return (f"surface: the prepared surface was refused for an edge under {c['limit_mm']:g} mm and the weld of the input did not help, so "
            f"its {count(n, 'edge')} under the limit {'was' if n == 1 else 'were'} cleaned: {', '.join(parts)}, "
            f"volume change {c['volume_change_mm3']:+.4f} mm3")


def shell_line(r: dict) -> str:
    """One line for the step `drop_nested_shells` (`mesh.surface_shell_retry` of the result, F4-45). It says that the verified mesh has no
    shell that lies in the material of another shell, which shells these are, and why gmsh could not mesh the surface with them."""
    parts = []
    for s in r["left_out"]:
        lo, hi = s["bbox_min_mm"], s["bbox_max_mm"]
        box = ", ".join(f"{a} {lo[i]:.4g} to {hi[i]:.4g}" for i, a in enumerate("xyz"))
        parts.append(f"shell {s['shell']} ({s['volume_mm3']:.4g} mm3, {s['faces_count']:,} faces, {box} mm)")
    n = len(r["left_out"])
    return (f"surface: gmsh could not mesh the prepared surface ({'; '.join(r['whole_surface_errors'])}). It has {r['shells_in_surface']} "
            f"closed shells, and {'this shell lies' if n == 1 else 'these shells lie'} in the material of another shell: {'; '.join(parts)}. "
            f"{'It adds' if n == 1 else 'They add'} no boundary to the solid, but the volume of the surface counted that material twice "
            f"({r['volume_before_mm3']:.6g} mm3). {'It was' if n == 1 else 'They were'} left out and the surface was meshed again "
            f"({r['volume_after_mm3']:.6g} mm3). If the shell is a thin void in the part, repair the STL and verify again")


def summary_lines(res: dict) -> list[str]:
    """The result of a run: one block of numbers per load case. Units: mm, MPa."""
    m = res["mesh"]
    t = res["timing_s"]
    mem = res["memory"]
    lines = [f"verify: {res['run']['name']}, mesh {res['size_mm']:g} mm, {m['tets']:,} tets, {m['dofs']:,} dofs, "
             f"solver {res['backend']['solver']}, {t['verify_total_s']:.1f} s"
             + (f" (estimate {mem['estimate_seconds_after_meshing']:.0f} s)" if mem.get("estimate_seconds_after_meshing") is not None else "")
             + f", peak memory {memory.gb_text(mem['peak_rss_bytes'])} (estimate {memory.gb_text(mem['estimate_after_meshing_bytes'])}"
             + (f", {memory.gb_text(mem['estimate_with_buckling_bytes'])} with the buckling check" if "estimate_with_buckling_bytes" in mem else "")
             + ")",
             f"mesh quality: min {m['quality_min']:.3f}, {m['below_0p05']} tets below 0.05; "
             f"P2 volume {m['p2_volume_over_stl']:.3f} of the STL volume; band {res['band_mm']:g} mm"
             + (f"; {m['bodies']['count']} {'body' if m['bodies']['count'] == 1 else 'bodies'}" if "bodies" in m else "")]
    cov = check_coverage.line_of(res)      # TC-2: near the top, so that a partial result never reads as clean
    if cov:
        lines.insert(1, cov)
    if m.get("surface_weld_retry"):
        lines.append(weld_line(m["surface_weld_retry"]))
    if m.get("surface_clean_retry"):
        lines.append(clean_line(m["surface_clean_retry"]))
    if m.get("surface_shell_retry"):
        lines.append(shell_line(m["surface_shell_retry"]))
    if res.get("part_only"):
        lines.extend(part_only_lines(res["part_only"]))
    elif res["run"].get("verified_file", "final.stl") != "final.stl":
        lines.append(f"mesh file: {res['run']['verified_file']} (the finished part; final.stl is the as-printed mesh with machining stock)")
    for n, c in res["cases"].items():
        v = c["von_mises"]
        lines.append(f"case {n}: von Mises by volume p50 {v['p50_by_volume_MPa']:.3g}, p95 {v['p95_by_volume_MPa']:.3g}, "
                     f"p99 {v['p99_by_volume_MPa']:.3g}, p99.9 {v['p99_9_by_volume_MPa']:.3g} MPa; "
                     f"peak at quadrature points {v['peak_qp_MPa']:.3g}, at vertices {v['peak_vertex_MPa']:.3g} MPa")
        rr = c["linear_solve"]["relative_residual"]
        lines.append(f"  max displacement {c['max_displacement_mm']:.3g} mm; solve residual "
                     + (f"{rr:.1e}" if rr is not None else "not defined (the load is zero)")
                     + f"; equilibrium residual {c['equilibrium']['force_relative_residual']:.1e}")
        nr = v.get("near_regions")
        if nr and nr["p99_9_by_volume_MPa_excluding_near_regions"] is not None:
            sh = nr["top_volume_share_near"]
            lines.append(f"  without points within {nr['distance_mm']:g} mm of a region: p99.9 "
                         f"{nr['p99_9_by_volume_MPa_excluding_near_regions']:.3g} MPa; the top 0.1 % volume is near a region: "
                         + ", ".join(f"{k} {100 * x:.0f} %" for k, x in sh.items() if k not in ("any", "none") and x is not None)
                         + f" (any {100 * sh['any']:.0f} %)")
        for fo in c["forces"]:
            lc = fo.get("load_centre", {})
            if lc.get("distance_mm") is not None:
                lines.append(f"  load {fo['region']}: P2 traction centre is {lc['distance_mm']:.2f} mm from the optimizer centre; "
                             f"moment about the support {lc['moment_ratio']:.3f} of the optimizer moment"
                             if lc.get("moment_ratio") is not None else
                             f"  load {fo['region']}: P2 traction centre is {lc['distance_mm']:.2f} mm from the optimizer centre")
    if res.get("supports"):
        lines.extend(supports_lines(res["supports"]))
    sens = res.get("load_band_sensitivity")
    if sens:
        for row in sens["bands"]:
            if "cases" not in row:
                lines.append(f"load band {row['band_mm']:g} mm: empty load region {', '.join(row['empty_load_regions'])}")
                continue
            # `verify/run.py` sets the change to None when the base p99.9 is 0 and the stiffness to None when the work of the load is not
            # above 0: both are so for a case that applies no load (U21). The summary says "n/a" for them.
            lines.append(f"load band {row['band_mm']:g} mm: " + "; ".join(
                f"{n} p99.9 {c['p99_9_by_volume_MPa']:.3g} MPa "
                + (f"({100 * c['p99_9_change_vs_base']:+.0f} %)" if c["p99_9_change_vs_base"] is not None else "(n/a)")
                + ", stiffness " + (f"{c['stiffness_N_per_mm']:.0f} N/mm" if c["stiffness_N_per_mm"] is not None else "n/a")
                for n, c in row["cases"].items()))
    for r in res["regions"]["load"]:
        if r.get("role") in ("joint", "spring"):          # a support patch bonds the faces that look away from the design (`supports_p2.bonded_facets`): no note about the sides
            continue
        nc = r.get("normal_check")
        if nc and nc["coherence"] is not None and nc["coherence"] < 0.9:
            lines.append(f"note: load region {r['region']}: the facets face different ways (normal coherence "
                         f"{nc['coherence']:.2f}); the band may reach another side of the part")
    lines.extend(yaml_note_lines(res.get("yaml_vs_run", {})))
    pv = res.get("previous_verify")
    if pv and not pv["same_final_stl"]:
        lines.append("note: the earlier verify.json was for another mesh file (run.verified_file) (it was removed)")
    lines.extend(f"note: {t}" for t in res.get("notes", []))
    if res.get("buckling"):
        lines.extend(buckling_lines(res["buckling"]))
    if res.get("motion"):
        lines.extend(motion_lines(res["motion"]))
    if res.get("bore_checks"):
        lines.extend(bore_checks_lines(res["bore_checks"]))
    if res.get("bore_facts"):
        lines.extend(bore_lines(res["bore_facts"], res.get("goodman")))
    if res.get("goodman"):
        lines.extend(goodman_lines(res["goodman"]))
    if res.get("pressfit"):
        lines.extend(pressfit_lines(res["pressfit"]))
    if "levels" in res:
        ch = res["levels"]["relative_change_first_to_last"]
        lines.append(f"levels {', '.join(f'{s:g}' for s in res['levels']['sizes_mm'])} mm, change from {ch['from_mm']:g} to "
                     f"{ch['to_mm']:g} mm:")
        def pct(x):     # None: the coarser value is 0, as for a case that applies no load (U21)
            return "n/a" if x is None else f"{100 * x:+.1f} %"
        for n, c in ch["cases"].items():
            lines.append(f"  {n}: p99.9 {pct(c['p99_9_by_volume_MPa'])}, peak qp {pct(c['peak_qp_MPa'])}, "
                         f"peak vertex {pct(c['peak_vertex_MPa'])}"
                         + (f", lowest buckling load factor {100 * ch['buckling_lowest_load_factor'][n]:+.1f} %"
                            if (ch.get("buckling_lowest_load_factor") or {}).get(n) is not None else ""))
    if "limit" in res:
        lim = res["limit"]
        for n, c in lim["cases"].items():
            lines.append(f"limit: case {n}: p{lim['percentile']:g} {c['value_MPa']:.3g} MPa against the allowable "
                         f"{lim['allowable_MPa']:g} MPa: margin {c['margin']:.2f}"
                         + ("  EXCEEDED" if c["exceeded"] else ""))
    elif res.get("no_allowable"):
        lines.append("limit: " + res["no_allowable"]["note"])
    if res.get("gate"):
        lines.extend(gate_lines(res["gate"]))
    if res.get("verify_json"):
        lines.append(f"wrote {res['verify_json']}")
    return lines


def _plain_name(name) -> str:
    """A case name on one line: control characters and runs of white space become one space (a name can hold a new line)."""
    return " ".join(str(name).split())


def plain_lines(res: dict) -> list[str]:
    """The answer to "will it break?" in plain words (P9-17): one line for each load case, then the lines that all cases share.

    It reads the keys that `verify.json` already has (`cases`, `limit`, `no_allowable`, `buckling`, `goodman`) and adds no number.
    A line says what the number is (a volume percentile of the von Mises stress), who gave the allowable, and the margin
    (allowable divided by the value). It states no promise: the check compares two numbers. With no allowable there is no
    comparison, and the shared line says where typical data sheet values are and how to give one. The option name follows the
    style of the caller (`--allowable-mpa` or `allowable_mpa`). An empty list: the result has no load case.
    """
    lim = res.get("limit")
    cases = res.get("cases") or {}
    names = list(lim["cases"]) if lim else list(cases)
    allowable_opt = opt("allowable")
    lines = []
    for n in names:
        who = _plain_name(n)
        if lim:
            p, c = lim["percentile"], lim["cases"][n]
            v, a = c["value_MPa"], lim["allowable_MPa"]
        else:
            vm = cases[n]["von_mises"]
            p, v = vm.get("percentile", 99.9), vm.get("percentile_by_volume_MPa", vm["p99_9_by_volume_MPa"])
        what = (f"case {who}: {p:g} % of the volume has von Mises stress at or below {v:.3g} MPa "
                f"(p{p:g} by volume" + (", the value that topopt compares)." if lim else ")."))
        g = ((res.get("gate") or {}).get("checks") or {}).get("stress", {}).get(n)
        replaced = "".join(f" {t}." for t in replaced_sentences(g)) if g else ""      # a limit or percentile that the caller gave in place of the YAML
        if not lim and g:        # the gate gives the limit of the case (`--gate`)
            word = {True: "at or below", False: "above", None: "not compared with"}[g["met"]]
            text = what + f" The gate limit is {g['limit']:g} MPa ({g['source']}): the value is {word} the limit"
            text += (f", margin {g['margin']:.2f} (limit divided by value)." if g.get("margin") is not None else ".")
            if g.get("peak_qp_MPa") is not None and g["peak_qp_MPa"] > g["limit"] and g["met"]:
                text += (f" The highest value at one point, {g['peak_qp_MPa']:.3g} MPa, is above the limit; the check compares "
                         f"the p{p:g} value only.")
            lines.append(text + replaced)
            continue
        if not lim:
            lines.append(what + " No allowable was given, so there is no yes or no for this case.")
            continue
        if v <= 0:
            lines.append(f"case {who}: the stress is 0 MPa: no load reaches the free part in this case, so there is nothing to "
                         f"compare with the allowable of {a:g} MPa.")
            continue
        side = "above" if c["exceeded"] else ("equal to" if v == a else "below")
        text = (f"{what} The allowable is {a:g} MPa, given by the user ({allowable_opt}); topopt has no default. "
                f"The value is {side} the allowable, margin {c['margin']:.2f} (allowable divided by value).")
        pk = c.get("peak_qp_margin")
        if pk is not None and pk < 1 and not c["exceeded"]:
            text += (f" The highest value at one point, {c['peak_qp_MPa']:.3g} MPa, is above the allowable; the check compares "
                     f"the p{p:g} value only.")
        lines.append(text + replaced)
    if not names:
        return lines
    if res.get("partial") and check_coverage.line_of(res):      # TC-2: the first line of the plain text, so that an answer to "will it break?" does not leave it out
        lines.insert(0, check_coverage.line_of(res))
    if res.get("gate"):      # the verdict of `--gate`, for the reader of `verify_status.plain` (the lines of `gate_lines` are in the CLI text only)
        lines.append(gate_verdict(res["gate"]))
    if not lim and not (res.get("gate") or {}).get("checks", {}).get("stress"):
        uri = (res.get("no_allowable") or {}).get("resource") or "topopt://materials/reference"
        lines.append(f"No allowable stress was given. Typical data sheet values for printed PETG and PLA are in {uri}; they are "
                     f"not the values of your material. Give an allowable ({allowable_opt}) to get a comparison.")
    done = lambda key: (res.get(key) or {}).get("status") == "done"      # noqa: E731 - a check that ran covers its subject
    left = [t for t, key in (("fatigue", "goodman"), ("buckling", "buckling")) if not done(key)]
    left += ["printing defects", "print direction"]
    lines.append("This check is the linear static stress of one mesh, one load case at a time, with one isotropic material. "
                 f"It does not cover {', '.join(left[:-1])} or {left[-1]}.")
    return lines


def projected_text(rec: dict) -> str:
    """` (without the rigid-body motions of the part: X, a fact)` for a case on which the rigid-body rule ran and gave a factor, else the empty text."""
    rb = rec.get("rigid_body")
    if isinstance(rb, dict) and rb.get("status") == "ok" and rb.get("load_factors"):
        return f" (without the rigid-body motions of the part: {rb['load_factors'][0]:.6g}, a fact)"
    return ""


def buckling_lines(bk: dict) -> list[str]:
    """The lines about the linear buckling check of the tet model (the key `buckling` of `verify.json`)."""
    if bk["status"] != "done":
        head = f"buckling: {bk['status']} ({bk.get('reason', '')})"
        return [head] + [f"  fact: {t}" for t in bk["facts"]] + [f"  option: {t}" for t in bk["options"]]
    lines = [f"buckling (P2 tets, solver {bk['solver']}, {bk['modes']} modes, {bk['seconds']:.1f} s):"]
    for c in bk["cases"]:
        lf = c.get("load_factors") or []
        if not lf:
            lines.append(f"  case {c['name']}: {c['status']}")
            continue
        v = c.get("voxel")
        text = f"  case {c['name']}: lowest load factor {lf[0]:.6g}" + projected_text(c)
        if v and v.get("lowest") is not None and c.get("tet_over_voxel") is not None:
            text += f" (voxel {v['lowest']:.6g}, tet over voxel {c['tet_over_voxel']:.3f}"
            if c.get("stiffness_over_optimizer") is not None:
                text += f"; static stiffness tet over optimizer {c['stiffness_over_optimizer']:.3f}"
            text += ")"
        text += f"; {c['solves']} solves, {c['seconds']:.1f} s"
        if c.get("near_repeated"):
            text += f"; near-repeated factors (modes {c['near_repeated_modes']})"
        lines.append(text)
    if any(c.get("tet_over_voxel") is not None for c in bk["cases"]):
        lines.append("  note: tet over voxel is the difference of two models (P2 tets of the STL, hex8 voxels), not an error of one "
                     "of them")
    lines += [f"  fact: {t}" for t in bk["facts"]] + [f"  option: {t}" for t in bk["options"]]
    return lines


def motion_lines(mo: dict) -> list[str]:
    """The lines about the A5 rows on the tet model (the key `motion` of `verify.json`)."""
    if mo["status"] != "done":
        return [f"motion rows: {mo['status']} ({mo.get('reason', '')})"] + [f"  fact: {t}" for t in mo["facts"]] \
            + [f"  option: {t}" for t in mo["options"]]
    lines = ["motion rows (A5, P2 tets):"]
    for r in mo["rows"]:
        v = r["voxel_value"]
        lines.append(f"  {r['name']}: tet {r['tet_value']:.5g} {r['unit']}, "
                     + (f"voxel {v:.5g}, difference {r['difference']:+.3g}, " if v is not None else "no voxel value, ")
                     + f"limit {r['limit']:g}, tet over limit {r['tet_over_limit']:.3g}")
    lines += [f"  fact: {t}" for t in mo["facts"]] + [f"  option: {t}" for t in mo["options"]]
    return lines


def bore_checks_lines(bc: dict | None) -> list[str]:
    """The lines about what the bore checks use (the base key `bore_checks` of `verify.json`): the checks, the bores, the knobs and their origin."""
    if not bc:
        return []
    lines = [f"bore checks:    {', '.join(bc['checks'])} ({bc['source']}); {len(bc['bores'])} bore(s): "
             + ", ".join(b["name"] for b in bc["bores"]) + "; statistics p99.9 and max"]
    for k, d in bc["knobs"].items():
        text = f"source: {d['source']}" if d.get("source") else d.get("default_source", "")
        lines.append(f"                {k} {d['value']:g} ({d['origin']})" + (f", {text}" if text else ""))
    if bc.get("percentile_note"):
        lines.append(f"                note: {bc['percentile_note']}")
    return lines


def bore_lines(entries: list, goodman: dict | None = None) -> list[str]:
    """The lines about the bores (the key `bore_facts` of `verify.json`): the wall facets, the sample set, the von Mises statistics of each case
    (p99.9, max and the median of the sample set of the bore, from `goodman.service_stress`; the median shows the scatter of the stress that is not
    smoothed), the facts and the options."""
    service = goodman["service_stress"] if goodman and goodman.get("status") == "done" else {}
    lines = []
    for e in entries:
        head = f"bore {e['name']} ({e['geometry_from']}): "
        if e.get("status") == "error":
            head += f"stopped at the {e['stage']}: {e['message']}"
        elif e["facets"] == 0:
            head += "no wall facet"
        else:
            ss = e.get("sample_set")
            head += (f"{e['facets']} wall facets, area {e['area_mm2']:.1f} mm2 ({e['area_over_analytic']:.3f} of the analytic area)"
                     + (f", {ss['tets']} tets, {ss['points']} sample points (band {ss['band_mm']:g} mm)" if ss else ", no sample point"))
        lines.append(head)
        for case, by_bore in service.items():
            st = by_bore.get(e["name"])
            if st:
                lines.append(f"  case {case}: von Mises at the sample set: p99.9 {st['vm_p99_9_MPa']:.4g}, max {st['vm_max_MPa']:.4g}, "
                             f"median {st['vm_median_MPa']:.4g} MPa")
        lines += [f"  fact: {t}" for t in e["facts"]] + [f"  option: {t}" for t in e["options"]]
    return lines


def goodman_lines(g: dict) -> list[str]:
    """The lines about the Goodman check (the key `goodman` of `verify.json`): one line for each row, then the facts and the options."""
    if g["status"] != "done":
        return [f"goodman: {g['status']} ({g.get('reason', '')})"] + [f"  fact: {t}" for t in g["facts"]] + [f"  option: {t}" for t in g["options"]]
    lines = [f"goodman (sm from {g['mean_stress']['from']}; statistics {', '.join(g['statistics'])}):"]
    for r in g["rows"]:
        n = r["n_goodman"]
        lines.append(f"  {r['id']}: worst case {r['worst_case']}, sa {r['sa_MPa']:.4g} MPa, sm {r['sm_MPa']:.4g} MPa, "
                     f"n {'none' if n is None else format(n, '.4g')}"
                     + (f" (n at sm = 0: {'none' if r['n_goodman_at_sm0'] is None else format(r['n_goodman_at_sm0'], '.4g')})" if "n_goodman_at_sm0" in r else "")
                     + f", yield ratio {r['peak_over_yield']:.4g}")
    for m in g["missing_rows"]:
        lines.append(f"  no row for {m['variant'] or 'the bore'} / {m['bore']}: {m['reason']}")
    lines += [f"  fact: {t}" for t in g["facts"]] + [f"  option: {t}" for t in g["options"]]
    return lines


def pressfit_lines(p: dict) -> list[str]:
    """The lines about the press-fit check (the key `pressfit` of `verify.json`): one line for each variant and bore, then the facts and the options."""
    if p["status"] != "done":
        return [f"pressfit: {p['status']} ({p.get('reason', '')})"] + [f"  fact: {t}" for t in p["facts"]] + [f"  option: {t}" for t in p["options"]]
    lines = [f"pressfit ({len(p['variants'])} cup variant(s), solver {p['model']['solver']}, {p['seconds']:.1f} s; statistics {', '.join(p['statistics'])}):"]
    for v in p["variants"]:
        for b in v["bores"]:
            c = b["contact_pressure_MPa"]
            net = sum(x * x for x in b["net_spring_force_N"]) ** 0.5
            lines.append(f"  {v['name']} / {b['name']}: k {b['k_mpa_per_mm']:.5g} MPa/mm, hoop p99.9 {b['hoop_p99_9_MPa']:.4g} MPa, max {b['hoop_max_MPa']:.4g} MPa, median {b['hoop_median_MPa']:.4g} MPa; "
                         f"contact pressure area mean {c['area_mean']:.4g} MPa (vertex mean {c['mean']:.4g}, max {c['max']:.4g}, min {c['min']:.4g}; "
                         f"{c['negative_nodes']} of {c['nodes']} nodes below 0); net spring force {net:.4g} N")
    for v, by_bore in (p.get("combined") or {}).items():
        for b, by_case in by_bore.items():
            for case, c in by_case.items():
                lines.append(f"  combined {v} / {b} / {case}: von Mises p99.9 {c['vm_p99_9_MPa']:.4g}, max {c['vm_max_MPa']:.4g} MPa; "
                             f"hoop p99.9 {c['hoop_p99_9_MPa']:.4g}, max {c['hoop_max_MPa']:.4g} MPa")
    lines += [f"  fact: {t}" for t in p["facts"]] + [f"  option: {t}" for t in p["options"]]
    return lines
