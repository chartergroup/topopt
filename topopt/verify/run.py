"""`topopt verify`: an independent P2 stress check of the final STL of a run. CONTRACTS.md section 16.

`verify_run(run_dir, ...)` reads the run (`run.json`, its YAML, `result/nominal.stl` when the run has one, else `result/final.stl`), meshes that STL with
tetrahedra (gmsh, `topopt.tet`), solves each load case with quadratic tets (P2) and writes `verify/verify.json`,
`verify/fields.npz` and `verify/mesh.npz`. With `part_only` it meshes `result/part_only.stl` instead
(P4b-3, `topopt.verify.part_only`).

Two checks are added when the run asks for them (B3): the linear buckling load
factors of the tet model (`buckling_p2.py`, the key `buckling`) and the A5 rows of the YAML on the tet displacement
(`motion.py`, the key `motion`). With both off, `verify.json` has no new key.

The `bore_checks` block of the YAML adds the bore checks (B4): the bore facts (`bore.py`, the key
`bore_facts`), the Goodman fatigue check (`fatigue.py`, the key `goodman`) and the press-fit check (`pressfit.py`, the key `pressfit`).
With the block absent (or `bore_checks=False`), `verify.json` has no new key.

A part held by joints (beam supports) and springs runs with the physics of the voxel model (K1-A, `supports_p2.py`, the key `supports` of
a level). Without a joint or a spring, `verify.json` has no new key.

The option `gate` (K1-B, `gate.py`) adds the key `gate`: a pass or a fail for the stress, buckling and A5 limits that the YAML already has.
Without it, `verify.json` and the exit codes are as before.

A sub-check that fails or does not run is a fact in its own record, and verify goes on. The keys `partial` and `coverage` of `verify.json` sum
those records up in one place (TC-2, `topopt.check_coverage`): which sub-checks ran, which failed and which were not run, each with its reason.

The imports of the extras (gmsh, scikit-fem, cholespy) are inside the functions that need them. The CLI
(`topopt verify`) and the MCP tool (`verify_run`) call this function.
"""
from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import check_coverage
from ..config import NotPrinted, UserError, load_config
from ..runinfo import missing_yaml_text, run_yaml_path
from . import bodies as mesh_bodies
from . import memory
from . import part_only as vpart_only
from . import gate as vgate
from .errors import EmptyRegion, MemoryRefused, SingularSystem, SolverUnavailable, VerifyError, VerifyUnavailable
from .names import opt
from .report import bond_line, pressfit_estimate_text, solver_lines

FORMAT = 1
REFINE_STEPS = 3                 # refinement steps of the solve (the pipeline verifier uses 3)
DEFAULT_BAND_FACTOR = 0.5        # the band is this times the voxel size of the run
PRESSFIT_JOINT_FACT = ("The press-fit stage makes its own factors of the stiffness matrix K and does not take the border of the joints of the YAML "
                       "(a spring is in K, so a part held by springs only keeps the stage).")
PRESSFIT_JOINT_OPTION = "Verify a run without joints, or leave the press-fit check off."
PRESSFIT_SPRING_FACT = ("The springs of the YAML are in K, as in the static solve. solve.clamp_reaction_N is the reaction of the clamps (fixed regions) only: "
                        "with a spring support it does not balance net_spring_force_N, because the springs carry the rest.")
BOND_STEP_FACTOR = DEFAULT_BAND_FACTOR   # K1-A: the facets of a joint or spring look at a point this times the voxel size away (the neighbour voxel centre)
DEFAULT_PERCENTILE = 99.9
DEFAULT_NEAR_MM = 2.0            # the stress numbers "near the regions" use this distance (mm) from a fixed or load region
# The optimizer setting of gmsh for the verify mesh.
DEFAULT_OPTIMIZE = "Gmsh+Netgen"
EXIT_OK, EXIT_BAD_INPUT, EXIT_COULD_NOT_RUN, EXIT_LIMIT = 0, 1, 2, 3


def _noop(_msg: str) -> None:
    pass


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"cannot write {type(o).__name__} to JSON")


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

@dataclass
class RunInfo:
    run_dir: Path
    run: dict
    cfg: object                  # topopt.config.Config
    yaml_path: Path
    final_stl: Path
    voxel_mm: float
    final_sha256: str = ""
    verified_file: str = "final.stl"   # the file in result/ that is meshed: nominal.stl (the finished part) when the run has one
    part_only: dict | None = None      # the plan block of `--part-only` (`part_only.plan_block`); None when the option is off
    part_only_exists: bool = False     # result/part_only.stl is in the folder (a fact line when the option is off)
    contact_cache: list = field(default_factory=list)   # contact area of each removed body, in the order of cfg.not_printed (the same for every mesh level)
    contact_s: float | None = None     # the seconds that the one measurement of contact_cache took (every level reports it in timing_s)


def read_run(run_dir, yaml_path=None, part_only: bool = False) -> RunInfo:
    """Read what the check needs from a run folder. Raise `UserError` for a missing or unreadable input.

    The mesh is `result/nominal.stl` when the run has one, else `result/final.stl`. With `part_only` it is `result/part_only.stl` (a run with
    `regions.not_printed`): a run without that file is a `UserError` with the facts and the options.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    run_json = run_dir / "run.json"
    if not run_dir.is_dir():
        raise UserError(f"run folder not found: {run_dir}")
    if not run_json.is_file():
        raise UserError(f"no run.json in {run_dir}")
    stl = run_dir / "result" / "final.stl"
    if not stl.is_file():
        raise UserError(f"no final mesh {stl}. The run must finish first")
    nominal = run_dir / "result" / "nominal.stl"
    if nominal.is_file():   # a run with machining stock (section 23): final.stl is the as-printed mesh, nominal.stl the part in service
        stl = nominal
    try:
        run = json.loads(run_json.read_text())
    except ValueError as exc:
        raise UserError(f"cannot read {run_json}: {exc}") from exc
    part_only_stl = run_dir / "result" / vpart_only.FILE
    has_part_only = part_only_stl.is_file()
    if part_only:
        if not has_part_only:
            raise UserError(vpart_only.missing_text(run_dir, run))
        stl = part_only_stl
    given = Path(yaml_path).expanduser() if yaml_path else None
    ypath = given if given is not None else run_yaml_path(run, run_dir)    # a relative path starts at the run folder
    if ypath is None or not ypath.is_file():
        raise UserError(missing_yaml_text(ypath, run, opt("yaml"), given is not None))
    cfg = load_config(ypath)
    h = run.get("h", cfg.settings.voxel_size)
    sha = hashlib.sha256(stl.read_bytes()).hexdigest()
    plan = vpart_only.plan_block(cfg, run, sha, nominal.is_file()) if part_only else None
    return RunInfo(run_dir, run, cfg, ypath.resolve(), stl, float(h), sha, verified_file=stl.name, part_only=plan,
                   part_only_exists=has_part_only)


def _same_vec(a, b) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= 1e-9 * max(1.0, abs(float(y))) for x, y in zip(a, b))


def _same_num(a, b) -> bool:
    return a is not None and b is not None and _same_vec([a], [b])


def _same_section(a, b: dict) -> bool:
    return isinstance(a, dict) and a.get("kind") == b["kind"] and set(a) == set(b) and all(
        _same_num(a[k], b[k]) for k in b if k != "kind")


def _compare_supports(run: dict, cfg) -> list[str]:
    """The differences of the `joints` and `springs` of the YAML against `run.json` (K1-A, REVIEW-135 M2), as texts.

    `run.json` holds `joints.joints[]` (region, `to_mm`, `section`, `beam_E_mpa`, `beam_nu`, `bond_stiffness_per_area_n_per_mm3`) and
    `springs.regions[]` (region, `stiffness_n_per_mm`, `stiffness_per_area_n_per_mm3`) with the numbers the run used. A run without the key had
    no entry. The material (E, nu) is not in `run.json`; the beam E and nu and the bond stiffness of a joint are, and they follow it.
    """
    from ..config import joint_axis_problem
    from ..joints import BOND_C, beam_frame
    diffs = []
    rj = {r.get("region"): r for r in ((run.get("joints") or {}).get("joints") or []) if isinstance(r, dict)}
    yj = {j.region: j for j in cfg.joints}
    if set(rj) != set(yj):
        diffs.append(f"joints: run.json has {sorted(rj)}, the YAML has {sorted(yj)}")
    h = run.get("h", cfg.settings.voxel_size)
    for name in sorted(set(rj) & set(yj)):
        r, y = rj[name], yj[name]
        if not _same_vec(r.get("to_mm", []), y.to):
            diffs.append(f"joint {name}: far point {r.get('to_mm')} in run.json, {list(y.to)} in the YAML")
        if not _same_section(r.get("section"), y.section.as_dict()):
            diffs.append(f"joint {name}: section {r.get('section')} in run.json, {y.section.as_dict()} in the YAML")
        for key, label, want in (("beam_E_mpa", "beam E", cfg.material.E if y.E is None else y.E),
                                 ("beam_nu", "beam nu", cfg.material.nu if y.nu is None else y.nu),
                                 ("bond_stiffness_per_area_n_per_mm3", "bond stiffness per area",
                                  BOND_C * cfg.material.E / h if y.bond_stiffness_per_area is None else y.bond_stiffness_per_area)):
            if key in r and not _same_num(r[key], want):
                diffs.append(f"joint {name}: {label} {r[key]:g} in run.json, {want:g} in the YAML")
        if y.origin is not None and not _same_vec(r.get("reference_mm", []), y.origin):
            diffs.append(f"joint {name}: origin {r.get('reference_mm')} in run.json, {list(y.origin)} in the YAML")
        if isinstance(r.get("frame"), dict) and r.get("reference_mm") is not None and "y" in r["frame"] \
                and joint_axis_problem(r["reference_mm"], y.to, y.orient) is None:
            ey = beam_frame(np.asarray(y.to, float) - np.asarray(r["reference_mm"], float), y.orient)[:, 1]
            if not _same_vec(r["frame"]["y"], ey):              # the section y axis: `orient` of the entry, or the default rule
                diffs.append(f"joint {name}: section y axis {r['frame']['y']} in run.json, {[float(v) for v in ey]} in the YAML "
                             f"({'orient' if y.orient is not None else 'default rule'})")
    rs = {r.get("region"): r for r in ((run.get("springs") or {}).get("regions") or []) if isinstance(r, dict)}
    ys = {sp.region: sp for sp in cfg.springs}
    if set(rs) != set(ys):
        diffs.append(f"springs: run.json has {sorted(rs)}, the YAML has {sorted(ys)}")
    for name in sorted(set(rs) & set(ys)):
        r, y = rs[name], ys[name]
        key, label, want = ("stiffness_n_per_mm", "stiffness", y.stiffness) if y.stiffness is not None \
            else ("stiffness_per_area_n_per_mm3", "stiffness per area", y.per_area)
        if key in r and not _same_vec(r[key], want):
            diffs.append(f"spring {name}: {label} {r[key]} in run.json, {list(want)} in the YAML")
    return diffs


def compare_yaml_run(info: RunInfo) -> dict:
    """Compare the YAML with `run.json`: case names, weights, forces, the region files, and the joints and springs. Return facts.

    The check reads the YAML that the user gave (or the one in `run.json`) and the record of the run. A difference is a
    fact in the report, not an error: the YAML can have been edited after the run, and then the check does not
    describe the printed part. `matches` is True if nothing differs. A key that `run.json` does not hold is skipped.
    """
    run, cfg = info.run, info.cfg
    diffs = []
    if run.get("name") not in (None, cfg.name):
        diffs.append(f"name: run.json has {run['name']!r}, the YAML has {cfg.name!r}")
    rcases = run.get("load_cases")
    if isinstance(rcases, list):
        yn, rn = [c.name for c in cfg.load_cases], [c.get("name") for c in rcases]
        if yn != rn:
            diffs.append(f"load case names: run.json has {rn}, the YAML has {yn}")
        else:
            for yc, rc in zip(cfg.load_cases, rcases):
                if "weight" in rc and abs(float(rc["weight"]) - yc.weight) > 1e-9 * max(1.0, yc.weight):
                    diffs.append(f"case {yc.name}: weight {rc['weight']} in run.json, {yc.weight} in the YAML")
                rf = [(f.get("region"), f.get("force")) for f in rc.get("forces", [])]
                yf = [(f.region, list(f.force)) for f in yc.forces]
                if [r for r, _ in rf] != [r for r, _ in yf]:
                    diffs.append(f"case {yc.name}: force regions {[r for r, _ in rf]} in run.json, {[r for r, _ in yf]} in the YAML")
                else:
                    for (r, a), (_, b) in zip(rf, yf):
                        if a is not None and not _same_vec(a, b):
                            diffs.append(f"case {yc.name}, region {r}: force {a} in run.json, {b} in the YAML")
    rreg = run.get("regions")
    if isinstance(rreg, dict):
        def names(v):
            return sorted(Path(str(x)).name for x in (v or []))
        pairs = [("design", [Path(str(rreg["design"])).name] if rreg.get("design") else None, [cfg.regions.design.name]),
                 ("fixed", names(rreg.get("fixed")) if "fixed" in rreg else None, names(cfg.regions.fixed)),
                 ("keep_out", names(rreg.get("keep_out")) if "keep_out" in rreg else None, names(cfg.regions.keep_out)),
                 ("solid", names(rreg.get("solid")) if "solid" in rreg else None, names(cfg.regions.solid))]   # A3a, section 22a
        for key, a, b in pairs:
            if a is not None and sorted(a) != sorted(b):
                diffs.append(f"{key} region files: run.json has {a}, the YAML has {b}")
        if isinstance(rreg.get("load"), dict):
            a = {k: Path(str(v)).name for k, v in rreg["load"].items()}
            b = {lr.name: lr.path.name for lr in cfg.regions.load}
            if a != b:
                diffs.append(f"load regions: run.json has {a}, the YAML has {b}")
    diffs += _compare_supports(run, cfg)
    note = ("the YAML that is read now is compared with the record of the run (run.json). "
            "A difference means the check may not describe the part of the run.")
    if cfg.joints or cfg.springs or run.get("joints") or run.get("springs"):
        note += (" Joints and springs are compared by region, far point, section, beam E and nu, bond stiffness and spring stiffness. "
                 "The material (E, nu) is not in run.json: it is not compared, and a change of it shows only in the beam E and nu and the "
                 "bond stiffness of a joint.")
    return dict(matches=not diffs, differences=diffs, yaml=str(info.yaml_path), note=note)


def compare_yaml_problem(run: dict, problem) -> dict:
    """`compare_yaml_run` for a `Problem` that was built from a YAML (`topopt refine` and `topopt buckling`).

    The same rule as `topopt verify`: case names, weights, forces and region files, as facts and not as a stop. `run.json`
    does not hold the material (`E`, `nu`) or `penal`, so they are not compared (the text says so). The result has no path
    (`yaml` is left out), so it can go into `metrics.json` and into the page of the refine step.
    """
    from types import SimpleNamespace
    cfg = problem.config
    res = compare_yaml_run(SimpleNamespace(run=run, cfg=cfg, yaml_path=cfg.yaml_path))
    res.pop("yaml", None)
    res["note"] += " The material (E, nu) and penal are not in run.json: they are not compared."
    return res


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #

def _check_options(mesh_size, levels, region_band_mm, allowable_mpa, percentile, max_gb, near_region_mm=None):
    def positive(key, v):
        if v is not None and not (isinstance(v, (int, float)) and np.isfinite(v) and v > 0):
            raise UserError(f"{opt(key)} must be a number above 0, got {v!r}")
    positive("mesh_size", mesh_size)
    positive("region_band", region_band_mm)
    positive("allowable", allowable_mpa)
    positive("max_gb", max_gb)
    positive("near", near_region_mm)
    if levels is not None:
        if not levels:
            raise UserError(f"{opt('levels')} needs at least one size")
        for v in levels:
            positive("levels", v)
    if mesh_size is not None and levels:
        raise UserError(f"give {opt('mesh_size')} or {opt('levels')}, not both")
    if not (isinstance(percentile, (int, float)) and 0 < percentile <= 100):
        raise UserError(f"{opt('percentile')} must be above 0 and at most 100, got {percentile!r}")


def parse_levels(text: str) -> list[float]:
    """Read `--levels 1.5,1.0` as a list of floats. Raise `UserError` for a bad text."""
    try:
        out = [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]
    except ValueError as exc:
        raise UserError(f"{opt('levels')} must be numbers in mm, separated by commas, got {text!r}") from exc
    if not out:
        raise UserError(f"{opt('levels')} needs at least one size")
    return out


# --------------------------------------------------------------------------- #
# Extras
# --------------------------------------------------------------------------- #

DEB_LIBS = "libglu1-mesa libxft2 libxinerama1 libopengl0"


def extras_facts() -> dict:
    """The state of the optional packages: gmsh (probe in a child Python), scikit-fem and cholespy.

    cholespy has the state `ok`, `missing` (not installed) or `load_failed` (installed, but the import fails: `detail`
    is the real error).
    """
    from ..tet.gmsh_probe import probe_subprocess
    from .factor import cholespy_problem
    probe = probe_subprocess(cache=True)
    out = {"gmsh": probe.as_dict()}
    try:
        import skfem
        out["scikit-fem"] = {"state": "ok", "version": getattr(skfem, "__version__", None)}
    except Exception as exc:
        out["scikit-fem"] = {"state": "missing", "detail": f"{type(exc).__name__}: {exc}"}
    problem = cholespy_problem()
    if problem is None:
        import cholespy
        out["cholespy"] = {"state": "ok", "version": getattr(cholespy, "__version__", None)}
    else:
        out["cholespy"] = {"state": problem[0], "detail": problem[1]}
    return out


def require_extras(facts: dict) -> None:
    """Raise `VerifyUnavailable` with the install step if gmsh or scikit-fem is missing.

    Both texts give the command in the same form: it installs into the Python that runs topopt now (`--python`).
    `topopt verify` needs gmsh, scikit-fem and cholespy, and the `verify` extra holds all three (the `tet` extra holds gmsh only).
    """
    from ..tet.gmsh_probe import TetProbe
    from .factor import verify_install_command
    g = facts["gmsh"]
    if g["state"] != "ok":
        probe = TetProbe(**g)
        text = f"topopt verify needs gmsh (the 'tet' extra): {probe.line()}"
        if g["state"] == "missing":
            text += (f" topopt verify also needs scikit-fem and cholespy; the 'verify' extra installs all three: "
                     f"{verify_install_command()}.")
        if sys.platform == "linux":
            text += f" On a headless Debian or Ubuntu host the gmsh wheel needs: apt install {DEB_LIBS}."
        raise VerifyUnavailable(text)
    if facts["scikit-fem"]["state"] != "ok":
        raise VerifyUnavailable(
            "topopt verify needs scikit-fem (BSD-3-Clause) from the 'verify' extra. "
            f"Install the extra with: {verify_install_command()}. " + facts["scikit-fem"].get("detail", ""))


# --------------------------------------------------------------------------- #
# Estimates
# --------------------------------------------------------------------------- #

def stl_geometry(path) -> dict:
    """Volume (mm^3), area (mm^2) and default mesh size (mm) of the STL, read as a closed surface."""
    from ..tet.mesher import _default_size, surface_volume
    from ..tet.surface import load_stl_surface
    from .regions import facet_areas
    V, F = load_stl_surface(str(path))
    return dict(volume_mm3=float(surface_volume(V, F)), area_mm2=float(facet_areas(V, F).sum()),
                default_size_mm=float(_default_size(V, F)), faces=int(len(F)))


def level_estimate(geom: dict, size: float, lim: dict, n_cases: int = 2, backend: str | None = None,
                   sensitivity: bool = True, buckling_modes: int | None = None) -> dict:
    """The memory and time estimate before meshing, for one mesh size (facts, no decision).

    `estimate_bytes` is for `backend` (None: the rule of the pipeline); `bytes_by_backend` has it for every factor.
    SuperLU has its own model, which depends on the compactness of the part (`compactness`, from the STL). Its estimate is
    the central value, with no margin, and `range_bytes` (`ranges_by_backend["superlu"]`) is the range from the measured
    spread; `above_limit` compares the central value with the limit, and `upper_end_above_limit` says that only the upper
    end is above it (the run starts; `notes` says so, with the options). `notes` also holds the 32-bit index fact.
    `estimated_seconds` is for `backend` (None if there is no measurement for it); `seconds_by_backend` has
    the number for each factor that has a time model. The time is the fit of `topopt.verify.memory` (one Linux host).
    With `buckling_modes` (the linear buckling check is on), the estimate also has `buckling_bytes` (the term of the check, on
    top of the factor), `estimate_with_buckling_bytes` and `above_limit_with_buckling`. The guard does not count the term.
    """
    dofs = memory.estimate_dofs(geom["volume_mm3"], geom["area_mm2"], size)
    c = memory.compactness(geom["volume_mm3"], geom["area_mm2"])
    est = memory.estimate_bytes(dofs, backend, c)
    limit = lim["limit_bytes"]
    by = {name: memory.estimate_seconds(dofs, name, n_cases, sensitivity, c) for name in ("cholesky", "pardiso", "cholespy", "superlu")}
    by_bytes = {name: memory.estimate_bytes(dofs, name, c) for name in ("cholesky", "pardiso", "cholespy", "superlu")}
    ranges = {"superlu": _range_dict(memory.memory_range_bytes(dofs, "superlu", c))}
    rng = ranges.get(backend)
    out = dict(size_mm=float(size), estimated_dofs=int(dofs), rule_bytes=int(memory.rule_bytes(dofs, backend, c)),
               estimate_bytes=est, estimate_gb=est / memory.GB, estimate_model=memory.model_text(backend, c),
               compactness=float(c), bytes_by_backend=by_bytes, ranges_by_backend=ranges, range_bytes=rng,
               above_limit=bool(limit is not None and est > limit),
               upper_end_above_limit=bool(limit is not None and rng is not None and est <= limit < rng["high"]),
               notes=memory.notes(dofs, backend, c, limit),
               estimated_seconds=by.get(backend) if backend else None, seconds_by_backend=by,
               seconds_note=f"fit on {memory.TIME_HOST}, {n_cases} load cases; None: no measurement for that factor")
    if buckling_modes is not None:
        term = memory.buckling_bytes(dofs, buckling_modes, cases=n_cases)
        out.update(buckling_bytes=int(term), estimate_with_buckling_bytes=int(est + term),
                   above_limit_with_buckling=bool(limit is not None and est + term > limit))
    return out


def _range_dict(r) -> dict | None:
    return None if r is None else dict(low=int(r[0]), central=int(r[1]), high=int(r[2]))


def refusal_text(est_bytes: int, lim: dict, size: float, when: str, dofs: int, backend: str | None = None,
                 c: float | None = None) -> str:
    limit_bytes, source = lim["limit_bytes"], lim["limit_source"]
    other = (f" or a factor that needs less memory (pypardiso on Linux x86_64, extra 'pardiso'; cholespy, extra 'verify'; "
             f"then {opt('solver')} NAME)," if backend == "superlu" else "")
    rng = memory.memory_range_bytes(dofs, backend, c)
    basis = (f"x 1.25" if rng is None else
             f"central value, no margin; range {memory.gb_text(rng[0])} to {memory.gb_text(rng[2])} from the measured spread")
    extra = (" The SuperLU model is uncertain for a shape that was not measured. " + memory.int32_fact(dofs, c)).rstrip() \
        if backend == "superlu" else ""
    return (f"the memory estimate for mesh size {size:g} mm is {memory.gb_text(est_bytes)} ({when}, {dofs:,} dofs, "
            f"{'solver ' + backend + ': ' if backend else ''}{memory.model_text(backend, c)}, {basis}), above the limit "
            f"{memory.gb_text(limit_bytes)} ({source}).{extra} "
            f"Options: a larger {opt('mesh_size')},{other} a larger {opt('max_gb')}, or {opt('force')} to run anyway "
            f"(the machine can swap or stop the process).")


def _check_estimate(est_bytes, lim, size, when, dofs, force, backend=None, c=None):
    limit = lim["limit_bytes"]
    if limit is not None and est_bytes > limit and not force:
        raise MemoryRefused(refusal_text(est_bytes, lim, size, when, dofs, backend, c),
                            dict(size_mm=float(size), estimate_bytes=int(est_bytes), limit_bytes=int(limit),
                                 limit_source=lim["limit_source"], when=when, dofs=int(dofs), solver=backend))


# --------------------------------------------------------------------------- #
# One mesh level
# --------------------------------------------------------------------------- #

def _compact_mesh_facts(facts: dict) -> dict:
    keep = ("size", "size_given", "optimize", "optimize_used", "tets", "nodes", "quality_min", "quality_mean",
            "below_0p1", "below_0p05", "volume_fraction_below_0p05", "volume_mm3", "volume_stl_mm3", "volume_ratio",
            "volume_ratio_stl", "deviation_from_input_mm", "stl_sha256", "sha256", "threads", "attempts",
            "crashes", "isolated", "gmsh_version", "seconds", "seconds_surface", "seconds_gmsh", "surface_faces",
            "surface_weld_retry", "surface_clean_retry", "surface_shell_retry")
    return {k: facts[k] for k in keep if k in facts}


def _crossing_bodies(cfg, crossings: dict | None, limit: int = 10) -> str:
    """Text for a meshing failure that has places where the surface crosses itself (`MeshingFailed.crossings`, P4, F2-6).

    It says which bodies of the YAML (the region STLs) are nearest to each place.
    It is a fact to help the reader, never an error. On any problem it gives the empty text.
    A place is listed with the regions at or nearest to it.
    The text gives the distance to the region surface in mm, or "inside" when the place is inside the region.
    `limit` is the length of the list in the message.
    """
    try:
        from . import regions as vregions
        from ..tet.intersect import describe_places
        facts = crossings or {}
        # The places that caused the stop: the input places if the retries stopped because the STL crosses itself (the places of
        # the prepared surface can be elsewhere), else the prepared surface, else the input (P4 review, L5).
        source = ("input" if facts.get("stopped") == "input_crosses_itself" or not facts.get("prepared", {}).get("count")
                  else "prepared")
        pts = np.asarray(facts.get(source, {}).get("points", []), float).reshape(-1, 3)[:limit]
        if not len(pts):
            return ""
        paths = {}
        for role, items in (("design", [cfg.regions.design]), ("fixed", cfg.regions.fixed), ("keep_out", cfg.regions.keep_out),
                            ("solid", cfg.regions.solid)):
            for pth in items:
                paths[f"{role} {Path(pth).name}"] = pth
        for lr in cfg.regions.load:
            paths[f"load {lr.name} ({Path(lr.path).name})"] = lr.path
        import trimesh
        sd = {name: trimesh.proximity.signed_distance(vregions.load_region_mesh(pth), pts) for name, pth in paths.items()}
        rows = []
        for i, pt in enumerate(pts):
            near = sorted(((abs(float(v[i])), name, float(v[i]) > 0) for name, v in sd.items()))[:2]
            rows.append(f"{describe_places(pt[None, :])}: " + ", ".join(
                f"{name} {'inside' if inside else f'{dist:.2f} mm away'}" for dist, name, inside in near))
        what = ("where the prepared surface crosses itself" if source == "prepared" else "where the STL crosses itself")
        return f". The bodies of the YAML that are nearest to the places {what} (the two nearest for each place): " + "; ".join(rows)
    except Exception:  # noqa: BLE001  (a help text must not hide the real error)
        return ""


def _singular_with_mesh(exc: SingularSystem, body, fixed_nodes, solver=None, detail=None, dofs=None, extra: str = "",
                        extra_facts: dict | None = None) -> SingularSystem:
    """The error for a singular matrix with the facts about the mesh (bodies, fixed nodes) in the text and in `facts`.

    `extra` and `extra_facts` are added to the text and the facts (`--part-only`: the file and the removed bodies).
    """
    f = dict(exc.facts) if exc is not None else dict(reason="singular_matrix", solver=solver, detail=detail, dofs=dofs)
    held = [k + 1 for k, b in enumerate(body.facts) if b.get("support_facets", 0) > 0]            # K1-A: the bodies that a joint or spring holds
    supports = any("support_facets" in b for b in body.facts)
    context = (f"The mesh has {body.count} {'body' if body.count == 1 else 'bodies'} (each has a fixed node"
               + (f" or a facet of a joint or spring; {len(held)} {'is' if len(held) == 1 else 'are'} held by a joint or spring facet" if supports else "")
               + f") and {len(fixed_nodes):,} fixed nodes. " + extra)
    from .factor import singular_message
    return SingularSystem(singular_message(f.get("solver"), f.get("detail", ""), f.get("dofs"), context, supports),
                          dict(f, bodies=body.count, fixed_nodes=int(len(fixed_nodes)), body_facts=body.facts, **(extra_facts or {})))


def _optimizer_stiffness(info: RunInfo) -> dict:
    """`{case: N/mm}` from `result/metrics.json` of the run (|F|^2 / compliance of the voxel model), or {}."""
    try:
        data = json.loads((info.run_dir / "result" / "metrics.json").read_text())
        return {c["name"]: float(c["N_per_mm"]) for c in data.get("stiffness_cases", []) if c.get("N_per_mm")}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _optimizer_center(info: RunInfo, case_name: str, region: str, k: int):
    """The `center` of a force in `run.json` (the centre of the load voxels that the optimizer used), or None."""
    try:
        for rc in info.run.get("load_cases", []):
            if rc.get("name") == case_name:
                forces = rc["forces"]
                f = next((x for x in forces if x.get("region") == region), forces[k])
                c = f.get("center")
                return [float(x) for x in c] if c is not None else None
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return None


def _region_masks(fixed_sel, load_sel, sup_sel, measure_sel=None) -> dict:
    """`{region name: bool mask over the P2 nodes}` of the fixed, load, joint, spring and measure regions: the names of the mode facts (K1-C).

    A joint or spring region that also carries a force is the same selection in `load_sel` and `sup_sel`: it is listed once (an `is` test).
    `measure_sel` is `{name: selection}` of the measure regions that the rows of `limits` use (`motion.motion_record` fills it), so that the
    largest motion of a mode can be in a seat. Two different regions with one name get `name#role` for the later one.
    """
    out: dict = {}
    seen: list = []
    for s in list(fixed_sel) + list(load_sel.values()) + list(sup_sel.values()) + list((measure_sel or {}).values()):
        if any(s is t for t in seen):
            continue
        seen.append(s)
        key = s.name if s.name not in out else f"{s.name}#{s.role}"
        if key not in out:
            out[key] = s.node_mask
    return out


def _bond_selection(model, cfg, sup_sel: dict, region_meshes: dict, voxel_mm: float) -> dict:
    """K1-A: the facets that carry the bond or the spring of each support region, by the rule of the voxel model (`supports_p2.bonded_facets`).

    The point `BOND_STEP_FACTOR` times the voxel size outside a facet is the centre of the neighbour voxel of a voxel face. The files of
    the design, the fixed and the load regions (the labels DESIGN, FIXED and LOAD) and of the keep-out regions are read once.
    """
    from . import regions, supports_p2
    meshes = {str(lr.path): region_meshes[lr.name] for lr in cfg.regions.load if lr.name in region_meshes}

    def mesh_of(path):
        if str(path) not in meshes:
            meshes[str(path)] = regions.load_region_mesh(path)
        return meshes[str(path)]
    neighbours = [mesh_of(cfg.regions.design)] + [mesh_of(p) for p in cfg.regions.fixed] + [mesh_of(lr.path) for lr in cfg.regions.load]
    keep_out = [mesh_of(p) for p in cfg.regions.keep_out]
    step = BOND_STEP_FACTOR * voxel_mm
    return {name: supports_p2.bonded_facets(model, sel, neighbours, keep_out, step) for name, sel in sup_sel.items()}


def _fixed_reference(fixed_sel, support_sel=()) -> list[float]:
    """The point for the moment of a load: the area-weighted centroid of the fixed regions.

    A part that is held by joints and springs only (`fixed: []`) uses the regions of those: the same rule.
    """
    fixed_sel = list(fixed_sel) or list(support_sel)
    if not fixed_sel:
        return [0.0, 0.0, 0.0]
    c = np.array([s.facts["centroid_mm"] for s in fixed_sel], float)
    w = np.array([s.facts.get("area_mm2", 0.0) for s in fixed_sel], float)
    return list((c * w[:, None]).sum(axis=0) / w.sum()) if w.sum() > 0 else list(c.mean(axis=0))


def _build_loads(model, cfg, load_sel, fixed, info=None, reference=None):
    """The load vector of each case: `(F_all (N, ncase), cases)`. `cases` is the start of the case facts.

    With `info` and `reference`, each force also gets the `load_centre` facts (the centre of the optimizer and of the
    P2 traction, and the moment about `reference`).
    """
    from . import bc
    cases, B = {}, []
    for case in cfg.load_cases:
        f = np.zeros(model.N)
        applied = []
        for k, fo in enumerate(case.forces):
            sel = load_sel[fo.region]
            fv, finfo = bc.traction_load(model, sel.facets, fo.force)
            f += fv
            entry = dict(region=fo.region, force_N=list(fo.force), **finfo,
                         resultant_relative_error=bc.resultant_error(finfo["resultant_N"], fo.force))
            if info is not None:
                entry["load_centre"] = bc.load_centre(sel.facts["centroid_mm"], _optimizer_center(info, case.name, fo.region, k),
                                                      fo.force, reference)
            applied.append(entry)
        total = np.linalg.norm(f.reshape(-1, 3).sum(axis=0))
        lost = f.reshape(-1, 3)[np.unique(fixed // 3)].sum(axis=0) if len(fixed) else np.zeros(3)
        cases[case.name] = dict(weight_of_the_optimizer=case.weight, weight_used=False, forces=applied,
                                force_on_fixed_nodes_N=[float(x) for x in lost],
                                force_on_fixed_nodes_relative=float(np.linalg.norm(lost) / total) if total > 0 else 0.0)
        B.append(f)
    return np.column_stack(B), cases


def _load_band_sensitivity(model, cfg, E, nu, rf, free, fixed, load_names, load_paths, region_meshes, sd_caches, bfacets, belems,
                           band, base_cases, percentile, say, support_regions=()) -> dict:
    """p99.9, peak and stiffness at a load band of 0 and of 2 times the band. The factor of the base run is reused.

    The band of the fixed regions does not change, so the matrix and its factor do not change: only the load vectors do.
    A band where a load region is empty is a fact (`empty`), not an error.
    """
    from . import regions, stats
    rows = []
    for label, b in (("0", 0.0), ("2x", 2.0 * band)):
        sels, empty = {}, []
        for name in load_names:
            sel = regions.select_region(model, bfacets, load_paths[name], name, "load", b, belems=belems,
                                        region_mesh=region_meshes[name], sd_cache=sd_caches[name], allow_empty=True)
            if sel.empty:
                empty.append(name)
            else:
                sels[name] = sel
        if empty:
            rows.append(dict(label=label, band_mm=float(b), empty_load_regions=empty))
            continue
        F_all, cases = _build_loads(model, cfg, sels, fixed)
        X = rf.solve(F_all[free])
        out = {}
        for j, case in enumerate(cfg.load_cases):
            u = np.zeros(model.N)
            u[free] = X[:, j]
            f = F_all[:, j]
            rep, _s, _e = stats.stress_report(model, u, E, nu, None, percentile)
            total = np.sum([fo.force for fo in case.forces], axis=0)
            stiff = float(total @ total / (f @ u)) if f @ u > 0 else None
            base = base_cases[case.name]
            bp, bk = base["von_mises"]["p99_9_by_volume_MPa"], base["stiffness_N_per_mm"]
            out[case.name] = dict(
                p99_9_by_volume_MPa=rep["p99_9_by_volume_MPa"], peak_qp_MPa=rep["peak_qp_MPa"],
                peak_vertex_MPa=rep["peak_vertex_MPa"], stiffness_N_per_mm=stiff,
                p99_9_change_vs_base=(rep["p99_9_by_volume_MPa"] - bp) / bp if bp else None,
                stiffness_change_vs_base=(stiff - bk) / bk if stiff and bk else None)
        rows.append(dict(label=label, band_mm=float(b),
                         load_regions={n: dict(facets=int(len(s.facets)), area_mm2=s.facts["area_mm2"],
                                               centroid_mm=s.facts["centroid_mm"]) for n, s in sels.items()},
                         cases=out))
        say(f"  load band {b:g} mm: " + ", ".join(f"{n} p99.9 {c['p99_9_by_volume_MPa']:.3g} MPa" for n, c in out.items()))
    return dict(bands=rows, base_band_mm=float(band),
                note="the same load, with the load facets chosen at another band: the factor of the base run is reused. "
                     "A large change means that the stress near the load depends on which facets take the traction"
                     + (f". A joint or spring region ({', '.join(support_regions)}) keeps the facets of the base band: its matrix is in the factor"
                        if support_regions else ""))


def _meshing_failed_text(size: float, exc: Exception, from_levels: bool = False, places: str = "") -> str:
    """The text of `VerifyError` for a meshing failure at `size` mm: what failed, the size, and the next step.

    A missing gmsh (`TetUnavailable`) and a refusal of the input STL itself do not depend on the mesh size: no hint for them.
    A refusal of the prepared surface (`PreparedSurfaceRejected`) and a gmsh failure (`MeshingFailed`) can change with the
    size, because the surface is prepared for that size. The text names the option and the size that failed. No retry is done.
    `places` is the text of `_crossing_bodies` (the places where the surface crosses itself, or ""): it comes before the next step.
    The option is the one that the call used: `levels` when the call gave levels (`--mesh-size` and `--levels` cannot be given
    together), else `mesh_size` (also for the default size). The ratchet calls with one size and has no `--levels`.
    """
    from ..tet import MeshingFailed
    from ..tet.surface import PreparedSurfaceRejected
    text = f"meshing the final STL at {size:g} mm failed: {exc}{places}"
    if isinstance(exc, (PreparedSurfaceRejected, MeshingFailed)):
        text += ". The surface is prepared for the mesh size, so another size can work: "
        if from_levels:
            text += f"change that size in {opt('levels')}"
        else:
            text += f"set {opt('mesh_size')} to a value other than {size:g} mm (larger or smaller)"
    return text


def run_level(info: RunInfo, size: float, *, solver: str, band_mm: float, percentile: float, lim: dict,
              force: bool, threads: int, optimize, pre_estimate: dict, out_dir: Path | None, say=_noop,
              near_mm: float = DEFAULT_NEAR_MM, sensitivity: bool = True, progress=None, buckling: bool = False,
              bore_checks: bool = False, from_levels: bool = False) -> dict:
    """Mesh, solve and report one mesh size. Return the report dict of this level (and write the files).

    `progress(state, text)` is called at each stage (`meshing`, `solving`, `writing`).
    With `buckling`, the level also has the key `buckling` (a record of `buckling_p2`: done, skipped or error). With A5 rows in the
    YAML, it has the key `motion` (`motion.py`). Both use the plain factor of the static solve before it is freed.
    With `bore_checks` (the YAML has the block), the level has `bore_facts` and, for each check of the block, `goodman` and `pressfit`.
    `from_levels` is True when the call gave levels: a meshing failure then names `--levels`, not `--mesh-size` (`_meshing_failed_text`).
    """
    from scipy import sparse as sp  # noqa: F401  (scipy is a main dependency)

    from ..tet import MeshingFailed, TetUnavailable, mesh_stl
    from ..tet.surface import SurfaceRejected
    from . import bc, regions, stats
    from .factor import FACTOR_TEST_RESIDUAL, RefinedFactor, make_factor
    from .p2 import STRESS_ORDER, P2Model, quadrature

    def stage(state, text):
        say(text)
        if progress:
            progress(state, text)

    cfg = info.cfg
    E, nu = cfg.material.E, cfg.material.nu
    rows_wanted = bool(cfg.limits.displacement or cfg.limits.alignment)      # A5 rows in the YAML: verify computes them too
    T: dict = {}
    t_all = time.perf_counter()

    # ---- mesh ----
    t0 = time.perf_counter()
    stage("meshing", f"level {size:g} mm: meshing the final STL (estimate before meshing: "
          f"{pre_estimate['estimated_dofs']:,} dofs, {memory.gb_text(pre_estimate['estimate_bytes'])})")
    try:
        mesh = mesh_stl(info.final_stl, size=size, threads=threads, optimize=optimize, log=None)
    except (TetUnavailable, MeshingFailed, SurfaceRejected) as exc:
        raise VerifyError(_meshing_failed_text(size, exc, from_levels, _crossing_bodies(cfg, getattr(exc, "crossings", None)))) from exc
    T["mesh_s"] = time.perf_counter() - t0
    say(f"  mesh: {len(mesh.tets):,} tets, {len(mesh.nodes):,} nodes, quality min {mesh.facts['quality_min']:.3f}, "
        f"{mesh.facts['below_0p05']} tets below 0.05 ({T['mesh_s']:.1f} s)")

    # ---- P2 model and the memory estimate with the real dofs ----
    t0 = time.perf_counter()
    model = P2Model.from_tetmesh(mesh)
    T["model_s"] = time.perf_counter() - t0
    compact = pre_estimate.get("compactness")
    after_bytes = memory.estimate_bytes(model.N, solver, compact)
    after_range = memory.memory_range_bytes(model.N, solver, compact)
    after_seconds = memory.estimate_seconds(model.N, solver, len(cfg.load_cases), sensitivity, compact)
    stage("solving", f"  P2 model: {model.N:,} dofs; solver {solver}; memory estimate {memory.gb_text(after_bytes)}"
          + (f" (range {memory.gb_text(after_range[0])} to {memory.gb_text(after_range[2])})" if after_range else "")
          + f" (before meshing: {memory.gb_text(pre_estimate['estimate_bytes'])})"
          + (f"; time estimate about {after_seconds:.0f} s" if after_seconds is not None else ""))
    _check_estimate(after_bytes, lim, size, "after meshing", model.N, force, solver, compact)
    for text in memory.notes(model.N, solver, compact, lim["limit_bytes"]):
        if text not in pre_estimate.get("notes", []):         # a note that the user saw before meshing is not said twice
            say("  note: " + text)
    quality = mesh.quality()
    body = mesh_bodies.find_bodies(model)                 # the body count is a fact before any solve
    say("  " + mesh_bodies.one_line(body))

    # ---- regions ----
    t0 = time.perf_counter()
    bfacets, belems = regions.boundary_facets(model.t.T, return_elems=True)
    X = model.node_xyz.T
    reach = (2.0 if sensitivity else 1.0) * band_mm             # the signed distance is exact inside this reach
    load_paths = {lr.name: lr.path for lr in cfg.regions.load}
    used = []
    for case in cfg.load_cases:
        for fo in case.forces:
            if fo.region not in used:
                used.append(fo.region)
    # `--part-only` (P4b-3): the mesh is result/part_only.stl. The region rule is the same; a removed body is not in the mesh, so the rule
    # finds the part next to its surface. Only a removed fixed region may select nothing (a fact, not an error).
    po = info.part_only
    removed = set(cfg.not_printed) if po is not None else set()
    po_sel: dict = {}                                       # removed body (NotPrinted) -> its Selection, for the facts
    po_extra, po_facts = ((vpart_only.error_context(po), dict(part_only=dict(file=vpart_only.FILE, removed=[r["label"] for r in po["removed"]])))
                          if po is not None else ("", None))
    fixed_sel, fixed_names = [], []
    for p in cfg.regions.fixed:
        gone = NotPrinted("fixed", Path(p).name, p) in removed
        rm = regions.load_region_mesh(p)
        sel = regions.select_region(model, bfacets, p, Path(p).stem, "fixed", band_mm, region_mesh=rm,
                                    sd_cache=regions.signed_distance(rm, X, reach), allow_empty=gone)
        if gone:
            po_sel[NotPrinted("fixed", Path(p).name, p)] = sel
        if not sel.empty:                                   # only a removed body can be empty here; it clamps nothing
            fixed_sel.append(sel)
    region_meshes, sd_caches, load_sel = {}, {}, {}
    for name in used:
        region_meshes[name] = regions.load_region_mesh(load_paths[name])
        sd_caches[name] = regions.signed_distance(region_meshes[name], X, reach)
        try:
            load_sel[name] = regions.select_region(model, bfacets, load_paths[name], name, "load", band_mm, belems=belems,
                                                   region_mesh=region_meshes[name], sd_cache=sd_caches[name])
        except EmptyRegion as exc:                         # no body of the part touches the load region
            mesh_bodies.add_regions(body, fixed_sel, {})
            raise mesh_bodies.empty_load_error(exc, body, po_extra, po_facts) from exc
        if NotPrinted("load", name, load_paths[name]) in removed:
            po_sel[NotPrinted("load", name, load_paths[name])] = load_sel[name]
    for n in (cfg.not_printed if po is not None else ()):     # the removed bodies that give no boundary condition: facts only
        if n not in po_sel:
            rm = regions.load_region_mesh(n.path)
            po_sel[n] = regions.select_region(model, bfacets, n.path, n.name, n.role, band_mm, belems=belems, region_mesh=rm,
                                              sd_cache=regions.signed_distance(rm, X, reach), allow_empty=True)
    if po is not None:                                       # what the removed fixed bodies clamp: part of the text of a body with no support
        fixed_text, fixed_nodes_by_body = vpart_only.selection_text([(n.label, n.path, po_sel[n].facts) for n in cfg.not_printed if n.role == "fixed"])
        po_extra += fixed_text
        po_facts["part_only"]["fixed_selected_nodes"] = fixed_nodes_by_body
    sup_sel: dict = {}                                       # K1-A: the regions of the joints and springs, with or without a force
    for name in dict.fromkeys([j.region for j in cfg.joints] + [sp_.region for sp_ in cfg.springs]):
        if name in load_sel:
            sup_sel[name] = load_sel[name]
            continue
        rm = region_meshes[name] = regions.load_region_mesh(load_paths[name])
        try:
            sup_sel[name] = regions.select_region(model, bfacets, load_paths[name], name, "load", band_mm, belems=belems,
                                                  region_mesh=rm, sd_cache=regions.signed_distance(rm, X, reach))
        except EmptyRegion as exc:                         # no body of the part touches the joint or spring region
            mesh_bodies.add_regions(body, fixed_sel, {})
            raise mesh_bodies.empty_load_error(exc, body, po_extra, po_facts) from exc
        sup_sel[name].role = sup_sel[name].facts["role"] = "joint" if any(j.region == name for j in cfg.joints) else "spring"
        if NotPrinted("load", name, load_paths[name]) in removed:
            po_sel[NotPrinted("load", name, load_paths[name])] = sup_sel[name]
    sup_only = [s for n, s in sup_sel.items() if n not in load_sel]       # a joint or spring region with no force: a region of the report like a load region
    sup_bond = _bond_selection(model, cfg, sup_sel, region_meshes, info.voxel_mm) if sup_sel else {}
    if sup_bond:
        from . import supports_p2
        supports_p2.check_bonded(cfg, sup_bond)
    mesh_bodies.add_regions(body, fixed_sel, load_sel, held=list(sup_bond.values()))
    mesh_bodies.check_supported(body, fixed_sel, band_mm, reach, po_extra, po_facts, list(sup_sel.values()))      # a body with no fixed node and no joint or spring: singular, user error
    fixed_nodes = np.flatnonzero(np.any([s.node_mask for s in fixed_sel], axis=0)) if fixed_sel else np.zeros(0, np.int64)
    fixed = bc.fixed_dofs(fixed_nodes)
    free = np.setdiff1d(np.arange(model.N), fixed)
    # the quadrature points near each region (for the stress numbers without the region patches)
    qp = model.qp_points(quadrature(STRESS_ORDER)[0])
    masks = {}
    for s in fixed_sel + list(load_sel.values()) + sup_only:
        key = s.name if s.name not in masks else f"{s.name}#{s.role}"
        masks[key] = regions.near_region_mask(model, s, qp, near_mm)
    del qp
    T["regions_s"] = time.perf_counter() - t0
    say(f"  regions: {len(fixed_nodes):,} fixed nodes; " + ", ".join(
        f"{s.name}: {s.facts['facets']} facets {s.facts['area_mm2']:.1f} mm2" for s in (*load_sel.values(), *sup_only)))
    po_level = None
    if po is not None:
        if not info.contact_cache:
            t0 = time.perf_counter()
            info.contact_cache.extend(vpart_only.contact_areas(info.final_stl, [n.path for n in cfg.not_printed]))
            info.contact_s = time.perf_counter() - t0
        if info.contact_s is not None:
            T["part_only_contact_s"] = info.contact_s
        po_level = vpart_only.level_block([vpart_only.body_facts(n, po_sel[n], n.role != "load" or n.name in used, info.contact_cache[i])
                                           for i, n in enumerate(cfg.not_printed)])
        for r, name in zip(po_level["removed"], vpart_only.titles(po_level["removed"])):
            say("  " + vpart_only.say_line(r, name))
        if "singular_note" in po_level:
            say("  note: " + po_level["singular_note"])

    # ---- the bores (B4): the wall facets and the sample sets are found once for the mesh ----
    bores, service, service_points = [], {}, {}
    if bore_checks:
        from . import bore as vbore
        t0 = time.perf_counter()
        bores = vbore.build(model, cfg.bore_checks)
        T["bore_s"] = time.perf_counter() - t0
        say("  bores: " + "; ".join(_bore_say(b) for b in bores))

    # ---- load vectors ----
    reference = _fixed_reference(fixed_sel, sup_sel.values())
    F_all, cases = _build_loads(model, cfg, load_sel, fixed, info, reference)

    # ---- matrix and factor ----
    t0 = time.perf_counter()
    K = model.stiffness(E, nu)
    sup = ssys = None
    if sup_sel:                                            # K1-A: K + Kb of the springs and bonds, and the bordered matrix of the joints
        from . import supports_p2
        sup = supports_p2.build_supports(model, cfg, sup_bond, info.voxel_mm, supports_p2.voxel_areas(info.run), BOND_STEP_FACTOR * info.voxel_mm,
                                         supports_p2.voxel_frames(info.run))
        K = (K + sup.Ks).tocsr()
        for line in sup.facts["near_singular_lines"]:
            say("  note: " + line)
        vox = lambda f: "" if f["voxel_area_mm2"] is None else f" (voxel run: {f['voxel_area_mm2']:.1f} mm2)"      # noqa: E731
        say("  supports: " + "; ".join([f"joint {j.name}: {j.facts['facets']} facets {j.facts['area_mm2']:.1f} mm2{vox(j.facts)}, beam length "
                                         f"{j.facts['length_mm']:.4g} mm" for j in sup.joints]
                                        + [f"spring {s_.name}: {s_.facts['facets']} facets {s_.facts['area_mm2']:.1f} mm2{vox(s_.facts)}"
                                           for s_ in sup.springs]))
    T["assemble_s"] = time.perf_counter() - t0
    mem_after_assembly = memory.peak_rss_bytes()
    t0 = time.perf_counter()
    Kff = K[free][:, free]
    border_bytes = 0
    if sup is not None:
        ssys = sup.system(Kff, free)                       # [[K + Kb, C], [C^T, D_bond + B]] as one CSR matrix, no Schur step
        if ssys.n_border:                                  # the bordered copy of Kff is one more matrix in memory (REVIEW-135 L3)
            border_bytes = supports_p2.bordered_copy_bytes(Kff.nnz)
            after_bytes += border_bytes
            say(f"  note: the bordered copy of the stiffness matrix for the joints adds {memory.gb_text(border_bytes)} to the memory "
                f"estimate: {memory.gb_text(after_bytes)}")
            _check_estimate(after_bytes, lim, size, "after meshing, with the bordered copy of K", model.N, force, solver, compact)
    A_solve = Kff if ssys is None else ssys.A
    try:
        fac = make_factor(A_solve, solver)
    except SingularSystem as exc:                          # add what is known about the mesh to the text
        raise _singular_with_mesh(exc, body, fixed_nodes, extra=po_extra, extra_facts=po_facts) from exc
    rf = RefinedFactor(A_solve, fac, steps=REFINE_STEPS)
    if ssys is not None:
        rf = ssys.padded(rf)                               # grid vectors in and out; the 6 m border rows are inside
    T["factor_s"] = time.perf_counter() - t0
    mem_after_factor = memory.peak_rss_bytes()
    say(f"  factor ({solver}): {T['factor_s']:.1f} s, stiffness {T['assemble_s']:.1f} s")

    # ---- the buckling check: the memory decision comes now, before the stress loop, because the check keeps the stress of every case ----
    bk_record, bk_run, bk_mem, sigs = None, False, None, {}
    if buckling:
        from . import buckling_p2
        term = buckling_p2.term_bytes(cfg, model, K, Kff, free)
        bk_record, bk_mem = buckling_p2.memory_decision(cfg, solver, term, after_bytes, lim, force, size)
        bk_run = bk_record is None
        say(f"  buckling check: memory estimate with the check {memory.gb_text(after_bytes + term)} (+{memory.gb_text(term)})"
            + ("" if bk_run else "; above the limit, the check is skipped and the stress check goes on"))

    # ---- solve and stress ----
    optimizer_stiffness = _optimizer_stiffness(info)
    t0 = time.perf_counter()
    try:
        Xf = rf.solve(F_all[free])
    except SingularSystem as exc:
        raise _singular_with_mesh(exc, body, fixed_nodes, extra=po_extra, extra_facts=po_facts) from exc
    T["solve_s"] = time.perf_counter() - t0
    if not np.all(np.isfinite(Xf)):
        raise _singular_with_mesh(None, body, fixed_nodes, solver, "the solution has NaN or infinite values", len(free),
                                  po_extra, po_facts)
    resid = rf.residual_norms(F_all[free], Xf)
    berr = rf.backward_error(F_all[free], Xf)
    # The residual of a real load case stops a singular matrix that the factor did not stop (E3b, review M1). One fixed test vector of
    # the factor can miss a matrix with one free rotation (a part held along one line); a real load has a residual of 0.2 to 0.7 there.
    # |b| = 0 (a load case with no force on the free dofs): the relative residual is not defined, so the case is not checked.
    zero_load = np.linalg.norm(F_all[free], axis=0) == 0.0
    for j, case in enumerate(cfg.load_cases):
        if not zero_load[j] and not resid[j] < FACTOR_TEST_RESIDUAL:             # also true for NaN
            err = _singular_with_mesh(
                None, body, fixed_nodes, solver,
                f"the refined solve of load case '{case.name}' has a relative residual of {resid[j]:.1e} after {REFINE_STEPS} "
                f"refinement steps (threshold {FACTOR_TEST_RESIDUAL:.0e}; a very ill-conditioned valid matrix can also come near it, "
                f"and another factor, named with {opt('solver')}, may factor that)", len(free), po_extra, po_facts)
            err.facts.update(load_case=case.name, relative_residual=float(resid[j]), threshold=FACTOR_TEST_RESIDUAL)
            raise err
    t0 = time.perf_counter()
    fields = {}
    sup_cases = {}
    for j, case in enumerate(cfg.load_cases):
        u = np.zeros(model.N)
        u[free] = Xf[:, j]
        f = F_all[:, j]
        rep, _sig, elem_max = stats.stress_report(model, u, E, nu, quality, percentile, near_masks=masks, near_mm=near_mm)
        U = u.reshape(-1, 3)
        c = cases[case.name]
        c["linear_solve"] = dict(relative_residual=None if zero_load[j] else float(resid[j]),
                                 backward_error=None if zero_load[j] else float(berr[j]), refinement_steps=REFINE_STEPS)
        if zero_load[j]:
            c["linear_solve"]["note"] = "the load vector of this case is zero: the relative residual is not defined"
        if sup is not None:
            terms = sup.terms(u)
            c["equilibrium"] = bc.equilibrium(model, K, u, f, fixed, support=terms)
            sup_cases[case.name] = sup.case_record(model, u, f, terms)
            for jn, jr in sup_cases[case.name]["joints"].items():
                say(bond_line(case.name, jn, jr))
        else:
            c["equilibrium"] = bc.equilibrium(model, K, u, f, fixed)
        c["max_displacement_mm"] = float(np.linalg.norm(U, axis=1).max())
        c["compliance_Nmm"] = float(f @ u)                       # f.u, as the optimizer (`optimize.py`)
        total = np.sum([fo.force for fo in case.forces], axis=0)
        c["stiffness_N_per_mm"] = float(total @ total / (f @ u)) if f @ u > 0 else None     # |F|^2 / c, as metrics.json
        opt_k = optimizer_stiffness.get(case.name)
        if opt_k:
            c["optimizer_stiffness_N_per_mm"] = opt_k
            c["stiffness_over_optimizer"] = c["stiffness_N_per_mm"] / opt_k if c["stiffness_N_per_mm"] else None
        c["von_mises"] = rep
        fields[f"vm_elem_max_{case.name}"] = elem_max
        if bk_run:
            sigs[case.name] = _sig               # the stress at the 14 points: the buckling check builds Kg from it
        if bores and (cfg.bore_checks.goodman or cfg.bore_checks.press_fit):
            if cfg.bore_checks.goodman:
                service[case.name] = {}
            if cfg.bore_checks.press_fit:
                service_points[case.name] = {}
            for b in bores:
                if not b.ok:
                    continue
                try:
                    if cfg.bore_checks.goodman:
                        service[case.name][b.spec.name] = b.sample.stats(_sig)        # sa: the case at the sample set of each bore
                    if cfg.bore_checks.press_fit:
                        service_points[case.name][b.spec.name] = b.sample.points(_sig)     # the case plus the press fit: `combined` (plan 6.8)
                except Exception as exc:     # one bore that fails is dropped with a fact (it has no value for the earlier cases either); the others go on
                    b.fail(exc, "stats")
                    for done in (*service.values(), *service_points.values()):
                        done.pop(b.spec.name, None)
                    say(f"  bore {b.spec.name} stopped: {b.entry['message']}")
        del _sig
    T["stress_s"] = time.perf_counter() - t0

    motion, measure_sel = None, {}
    if rows_wanted:
        t0 = time.perf_counter()
        from . import motion as vmotion
        try:
            motion = vmotion.motion_record(cfg, info, model, free, load_sel, bfacets, band_mm, Xf, measure_sel)
        except Exception as exc:     # a failure of the rows is a fact in the record, not a stop of verify (as for the buckling check)
            motion = vmotion.error_record(" ".join(f"{type(exc).__name__}: {exc}".split()))
        if motion is not None:
            T["motion_s"] = time.perf_counter() - t0
            say("  motion rows: " + (f"{len(motion['rows'])} rows" if motion["status"] == "done" else
                                     f"{motion['status']} ({motion['reason']})" + (f": {motion['message']}" if "message" in motion else "")))

    sens = None
    if sensitivity:
        t0 = time.perf_counter()
        sens = _load_band_sensitivity(model, cfg, E, nu, rf, free, fixed, used, load_paths, region_meshes, sd_caches,
                                      bfacets, belems, band_mm, cases, percentile, say, support_regions=list(sup_sel))
        T["sensitivity_s"] = time.perf_counter() - t0
    if bk_run:
        t0 = time.perf_counter()
        stage("solving", f"  linear buckling check of the tet model ({solver}, {cfg.buckling.modes} modes)")
        try:
            bk_record = buckling_p2.run_check(model, cfg, info, fac=fac if ssys is None else ssys.grid_solver(fac),
                                              Kff=Kff if ssys is None else ssys.stiffness(), free=free, sigs=sigs, cases=cases,
                                              solver=solver, E=E, nu=nu, mem=bk_mem, say=say,
                                              supports=None if sup is None else sup.buckling_note(), held=sup is not None,
                                              region_masks=_region_masks(fixed_sel, load_sel, sup_sel, measure_sel))
        except Exception as exc:     # a failure of the check is a fact in the record, not a stop of verify
            bk_record = buckling_p2.error_record(cfg, solver, " ".join(f"{type(exc).__name__}: {exc}".split()))
            say("  buckling check stopped: " + bk_record["message"])
        T["buckling_s"] = time.perf_counter() - t0
        sigs.clear()
    fac.free()
    rf = Kff = ssys = None     # the static factor and its matrix are not used again: the press-fit stage makes its own factors, one at a time
    pf_record = None
    if bores and cfg.bore_checks.press_fit and sup is not None and sup.joints:
        from . import pressfit as vpressfit
        pf_record = vpressfit.skip_record(cfg.bore_checks, "supports", [PRESSFIT_JOINT_FACT], [PRESSFIT_JOINT_OPTION])
        say("  press fit: skipped (the YAML has a joint)")
    elif bores and cfg.bore_checks.press_fit:
        pf_record = _pressfit_stage(cfg, bores, model, K, free, fixed, solver=solver, E=E, nu=nu, lim=lim, force=force, after_bytes=after_bytes,
                                    size=size, T=T, say=say, service_points=service_points)
        service_points.clear()
        if sup is not None and "facts" in pf_record:         # springs are in K (REVIEW-135 Q4); the record says what its reaction does not hold
            pf_record["facts"].append(PRESSFIT_SPRING_FACT)
    bore_facts, goodman, pressfit = (_bore_records(cfg, bores, service, T, say, pf_record) if bore_checks else (None, None, None))
    T["total_s"] = time.perf_counter() - t_all

    peak = memory.peak_rss_bytes()
    child_peak = memory.children_peak_rss_bytes()
    level = dict(
        size_mm=float(size),
        mesh=dict(**_compact_mesh_facts(mesh.facts), bodies=mesh_bodies.report(body), p2_nodes=int(model.nn),
                  edges=int(model.ne), dofs=int(model.N), free_dofs=int(len(free)), fixed_dofs=int(len(fixed)),
                  p2_volume_mm3=float(model.volume()), p2_volume_over_stl=float(model.volume() / mesh.facts["volume_stl_mm3"])),
        regions=dict(fixed=[s.facts for s in fixed_sel], load=[s.facts for s in (*load_sel.values(), *sup_only)],
                     fixed_nodes_total=int(len(fixed_nodes)), keep_out_used=False, near_region_mm=float(near_mm),
                     moment_reference_mm=reference),
        memory=dict(estimate_before_meshing_bytes=int(pre_estimate["estimate_bytes"]),
                    estimate_after_meshing_bytes=int(after_bytes), rule_bytes=int(memory.rule_bytes(model.N, solver, compact)),
                    estimate_range_after_meshing_bytes=_range_dict(after_range),
                    estimate_model=memory.model_text(solver, compact), estimate_seconds_after_meshing=after_seconds,
                    peak_rss_bytes=int(peak), peak_rss_after_assembly_bytes=int(mem_after_assembly),
                    peak_rss_after_factor_bytes=int(mem_after_factor), children_peak_rss_bytes=int(child_peak),
                    measured_over_estimate=float(peak / after_bytes), measured_over_rule=float(peak / memory.rule_bytes(model.N, solver, compact)),
                    note="peak_rss_bytes is the peak of this process so far (a real run from the command line is one process; "
                         "with several levels it does not fall between levels). children_peak_rss_bytes: on Linux a child "
                         "starts with the size of its parent, so it is at least the true peak of the child and can be "
                         "the size of this process at the start of the child. It is not a measurement of gmsh."),
        timing_s={k: float(v) for k, v in T.items()},
        cases=cases,
    )
    if sens is not None:
        level["load_band_sensitivity"] = sens
    if po_level is not None:
        level["part_only"] = po_level
    if bk_run:
        # the peak of the process now holds the check: compare it with the estimate that has the term of the check
        with_check = bk_mem["estimate_with_check_bytes"]
        level["memory"].update(estimate_with_buckling_bytes=int(with_check), measured_over_estimate_with_buckling=float(peak / with_check))
    if border_bytes:
        level["memory"]["estimate_bordered_copy_bytes"] = int(border_bytes)       # included in estimate_after_meshing_bytes
    if sup is not None:
        level["supports"] = dict(sup.facts, cases=sup_cases,
                                 note="per case and joint: bond_force_N and bond_moment_Nmm are the force and the moment of the bond on the part "
                                      "(N and N mm, about about_mm), from the nodal forces of the bond; beam.part is -B q and beam.ground is the "
                                      "reaction at the far point of the beam. Per spring: the force of the spring on the part, and its moment about the origin")
    if bk_record is not None:
        level["buckling"] = bk_record
    if motion is not None:
        level["motion"] = motion
    if bore_facts is not None:
        level["bore_facts"] = bore_facts
    if goodman is not None:
        level["goodman"] = goodman
    if pressfit is not None:
        level["pressfit"] = pressfit
    if out_dir is not None:
        stage("writing", f"  writing the files of level {size:g} mm")
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh.save(out_dir / "mesh.npz")
        np.savez_compressed(out_dir / "fields.npz", p=model.p, t=model.t.astype(np.int32), **fields,
                            node_xyz=model.node_xyz.astype(np.float32))
        level["files"] = dict(mesh="mesh.npz", fields="fields.npz")
    del K, Kff, model, fac, rf, mesh
    gc.collect()
    return level


def _bore_say(b) -> str:
    """The text of one bore in the line `bores:` of the progress output."""
    if b.entry["status"] == "error":
        return f"{b.spec.name}: stopped at the {b.entry['stage']} (see bore_facts)"
    return f"{b.spec.name}: {b.entry['facets']} wall facets" + (f", {b.sample.n_qp} sample points" if b.ok else ", dropped (see bore_facts)")


def _pressfit_stage(cfg, bores, model, K, free, fixed, *, solver, E, nu, lim, force, after_bytes, size, T, say, service_points=None):
    """The record `pressfit` of one level (plan 7.2): the stage runs inside `try`; an exception is a record with the status `error`, not a stop of verify."""
    from . import pressfit as vpressfit
    t0 = time.perf_counter()
    try:
        rec = vpressfit.run_stage(cfg.bore_checks, bores, model, K, free=free, fixed=fixed, solver=solver, E=E, nu=nu, refine_steps=REFINE_STEPS,
                                  lim=lim, force=force, after_bytes=after_bytes, size=size, service_points=service_points, say=say)
    except Exception as exc:     # a failure of the check is a fact in the record, not a stop of verify (as for the buckling check)
        rec = vpressfit.error_record(cfg.bore_checks, " ".join(f"{type(exc).__name__}: {exc}".split()))
        say("  press fit stopped: " + rec["message"])
    T["pressfit_s"] = time.perf_counter() - t0
    return rec


def _bore_records(cfg, bores, service, T, say, pressfit=None):
    """The records of the bore checks of one level: `(bore_facts, goodman or None, pressfit or None)`. A failure of a check is a fact, not a stop.

    `pressfit` is the record of the press-fit stage (None when the YAML has no `press_fit` block; the stage always gives a record). Its hoop stress is `sm` of the Goodman rows; a record
    with no result (skipped, error) gives the Goodman check no row for a cup variant, with its reason and options.
    """
    from . import fatigue
    from . import pressfit as vpressfit
    bc = cfg.bore_checks
    goodman = None
    if bc.goodman:
        t0 = time.perf_counter()
        dropped = [dict(bore=b.spec.name, reason=b.drop_reason, options=b.entry["options"]) for b in bores if not b.ok]
        extra = {}
        if bc.press_fit:
            hoop = vpressfit.hoop_values(pressfit)
            extra = dict(hoop=hoop, press_reason=None if hoop is not None else pressfit["reason"],
                         press_options=[] if hoop is not None else list(pressfit["options"]))
        try:
            goodman = fatigue.build_record(bc, [c.name for c in cfg.load_cases], service, dropped=dropped, **extra)
        except Exception as exc:     # a failure of the check is a fact in the record, not a stop of verify
            goodman = fatigue.error_record(" ".join(f"{type(exc).__name__}: {exc}".split()))
            say("  goodman check stopped: " + goodman["message"])
        T["goodman_s"] = time.perf_counter() - t0
        goodman["seconds"] = round(T["goodman_s"], 3)
        if goodman["status"] == "done":
            low = goodman["lowest_n"]
            say(f"  goodman: {len(goodman['rows'])} rows" + (f", lowest n {low['value']:.3g} ({low['id']})" if low else "")
                + (f", {len(goodman['missing_rows'])} variant or bore with no row" if goodman["missing_rows"] else ""))
        else:
            say(f"  goodman: {goodman['status']} ({goodman['reason']})")
    if bc.press_fit:
        say(f"  press fit: {pressfit['status']}" + (f" ({pressfit['reason']})" if pressfit.get("reason") else "")
            + (f", {len(pressfit['variants'])} variant(s), {pressfit['seconds']:.1f} s" if pressfit["status"] == "done" else ""))
    return [b.entry for b in bores], goodman, (pressfit if bc.press_fit else None)


# --------------------------------------------------------------------------- #
# Levels, limit, report
# --------------------------------------------------------------------------- #

def _case_numbers(level: dict, key: str, case: str) -> float:
    return level["cases"][case]["von_mises"][key]


def levels_block(levels: list[dict]) -> dict:
    """Relative change of p99.9 and of the peaks between mesh sizes. `levels` are sorted from coarse to fine.

    A change is None when the coarser value is 0 (`_change`): a case that applies no load (U21) has no stress at any level.
    """
    keys = ("p99_9_by_volume_MPa", "peak_qp_MPa", "peak_vertex_MPa")
    rows = []
    for lv in levels:
        rows.append(dict(size_mm=lv["size_mm"], dofs=lv["mesh"]["dofs"], tets=lv["mesh"]["tets"],
                         cases={c: {k: _case_numbers(lv, k, c) for k in keys} for c in lv["cases"]}))
    changes = []
    for a, b in zip(levels[:-1], levels[1:]):
        changes.append(dict(from_mm=a["size_mm"], to_mm=b["size_mm"], cases={
            c: {k: _change(_case_numbers(a, k, c), _case_numbers(b, k, c)) for k in keys}
            for c in a["cases"]}))
    first_last = dict(from_mm=levels[0]["size_mm"], to_mm=levels[-1]["size_mm"], cases={
        c: {k: _change(_case_numbers(levels[0], k, c), _case_numbers(levels[-1], k, c)) for k in keys}
        for c in levels[0]["cases"]})
    out = dict(sizes_mm=[lv["size_mm"] for lv in levels], rows=rows, relative_change=changes,
               relative_change_first_to_last=first_last,
               note="relative change = (finer - coarser) / coarser. p99.9 is the main number; the peaks can move "
                    "with the mesh at sharp features, so they are information")
    if any("buckling" in lv or "motion" in lv or "goodman" in lv or "pressfit" in lv for lv in levels):
        _add_levels_of_the_checks(out, levels)
    return out


def _lowest_load_factor(level: dict, case: str):
    b = level.get("buckling")
    if b and b.get("status") == "done":
        for c in b["cases"]:
            if c["name"] == case and c.get("load_factors"):
                return c["load_factors"][0]
    return None


def _motion_values(level: dict) -> dict:
    m = level.get("motion")
    return {r["name"]: r["tet_value"] for r in m["rows"]} if m and m.get("status") == "done" else {}


def _goodman_values(level: dict) -> dict:
    g = level.get("goodman")
    return {r["id"]: r["n_goodman"] for r in g["rows"]} if g and g.get("status") == "done" else {}


def _pressfit_values(level: dict) -> dict:
    """`{"<variant>/<bore>": hoop p99.9 (MPa)}` of the press-fit stage of a level; empty when the stage has no result."""
    p = level.get("pressfit")
    return {f"{v['name']}/{b['name']}": b["hoop_p99_9_MPa"] for v in p["variants"] for b in v["bores"]} if p and p.get("status") == "done" else {}


def _change(a, b):
    return (b - a) / a if a not in (None, 0) and b is not None else None


def _add_levels_of_the_checks(out: dict, levels: list) -> None:
    """The lowest buckling load factor and the A5 row values of each level, and their relative change (only with the checks on)."""
    cases = list(levels[0]["cases"])
    names = list(_motion_values(levels[-1]))
    gids = list(_goodman_values(levels[-1]))
    pids = list(_pressfit_values(levels[-1]))
    for lv, row in zip(levels, out["rows"]):
        if any("buckling" in x for x in levels):
            row["buckling_lowest_load_factor"] = {c: _lowest_load_factor(lv, c) for c in cases}
        if names:
            row["motion_tet_value"] = {n: _motion_values(lv).get(n) for n in names}
        if gids:
            row["goodman_n"] = {i: _goodman_values(lv).get(i) for i in gids}
        if pids:
            row["pressfit_hoop_p99_9_MPa"] = {i: _pressfit_values(lv).get(i) for i in pids}

    def block(a, b):
        res = {}
        if any("buckling" in x for x in levels):
            res["buckling_lowest_load_factor"] = {c: _change(_lowest_load_factor(a, c), _lowest_load_factor(b, c)) for c in cases}
        if names:
            va, vb = _motion_values(a), _motion_values(b)
            res["motion_tet_value"] = {n: _change(va.get(n), vb.get(n)) for n in names}
        if gids:
            ga, gb = _goodman_values(a), _goodman_values(b)
            res["goodman_n"] = {i: _change(ga.get(i), gb.get(i)) for i in gids}
        if pids:
            pa, pb = _pressfit_values(a), _pressfit_values(b)
            res["pressfit_hoop_p99_9_MPa"] = {i: _change(pa.get(i), pb.get(i)) for i in pids}
        return res
    for ch, a, b in zip(out["relative_change"], levels[:-1], levels[1:]):
        ch.update(block(a, b))
    out["relative_change_first_to_last"].update(block(levels[0], levels[-1]))


def limit_block(level: dict, allowable: float, percentile: float) -> dict:
    """The comparison with a user-given allowable stress, at the chosen percentile (by volume)."""
    cases = {}
    for name, c in level["cases"].items():
        v = c["von_mises"]["percentile_by_volume_MPa"]
        cases[name] = dict(value_MPa=float(v), margin=float(allowable / v) if v > 0 else float("inf"),
                           exceeded=bool(v > allowable), peak_qp_MPa=c["von_mises"]["peak_qp_MPa"],
                           peak_qp_margin=float(allowable / c["von_mises"]["peak_qp_MPa"]))
    exceeded = [n for n, c in cases.items() if c["exceeded"]]
    return dict(allowable_MPa=float(allowable), percentile=float(percentile), mesh_size_mm=level["size_mm"],
                cases=cases, exceeded=bool(exceeded), exceeded_cases=exceeded,
                note="the allowable is given by the user; topopt has no default limit. The exit code is 3 only "
                     "when the percentile value of a case is above it. The peak is shown for information.")


MATERIAL_REFERENCE_URI = "topopt://materials/reference"   # the MCP resource of topopt_mcp (CONTRACTS.md section 25.7)
MATERIAL_REFERENCE_FILE = "topopt_mcp/materials/reference.md"


def no_allowable_fact() -> dict:
    """The fact for a run with no allowable stress: the result has no `limit` block, and where the data sheet ranges are.

    Topopt has no default allowable (decision D7): the user or the agent chooses it. The reference is a document. The engine
    does not read it, and it gives the choice to the reader. The option name follows the style of the caller (`--allowable-mpa`
    for the command line, `allowable_mpa` for the MCP tool).
    """
    return dict(
        resource=MATERIAL_REFERENCE_URI, package_file=MATERIAL_REFERENCE_FILE,
        note=(f"no allowable stress was given ({opt('allowable')}): the result has no limit block and no pass or fail. "
              "Topopt has no default allowable. The allowable is your choice: it depends on the material, the print direction, the load time and the safety factor. "
              f"Data sheet ranges, with their sources and their limits, are in the MCP resource {MATERIAL_REFERENCE_URI} "
              f"(the file {MATERIAL_REFERENCE_FILE} in the package)."))


@dataclass
class Plan:
    """What is known before any meshing: the run, the solver, the extras, the sizes and their estimates."""
    info: RunInfo
    band: float
    chosen: str | None
    sizes: list
    estimates: list
    lim: dict
    base: dict                 # the first keys of `verify.json`
    buckling: bool = False     # the linear buckling check is on (the YAML `buckling.check`, or the flag)
    bore_checks: bool = False  # the bore checks are on (the YAML block `bore_checks`, or the flag)
    notes: list = field(default_factory=list)   # fact lines for the caller that are NOT in verify.json (the reply and the log have them)
    gate_limits: dict | None = None             # the limits of the gate (`gate.gate_limits`); None without `gate`


def _check_loads_and_supports(cfg) -> None:
    """Refuse a YAML with a moment, a remote force, a pressure or a per-axis support (CONTRACTS.md section 26.2).

    The P2 model clamps all 3 dofs of every fixed region and loads a plain traction `F / A` on each load region. A
    moment, a remote force or a pressure would be dropped or loaded as another load, and a roller would be a full
    clamp, with no message. A spring and a joint are modelled (K1-A, `supports_p2.py`), so they are not refused.
    """
    if not cfg.uses_extended_loads():
        return
    has = []
    if any(f.moment is not None for c in cfg.load_cases for f in c.forces):
        has.append("a moment")
    if any(f.at is not None for c in cfg.load_cases for f in c.forces):
        has.append("a remote force (at)")
    if any(c.pressures for c in cfg.load_cases):
        has.append("a pressure")
    if cfg.has_partial_supports():
        has.append("a per-axis support (fixed axes)")
    if not has:                                            # a spring or a joint only
        return
    raise UserError(
        f"topopt verify does not model {', '.join(has)} yet: its P2 model clamps all 3 dofs of a fixed region and "
        "loads a plain traction on a load region, so the result would not describe this part. "
        "Options: verify a run of the same part with plain forces and full clamps, "
        "or use the voxel results (metrics.json, run.json)")


def preflight(run_dir, *, mesh_size: float | None = None, levels: list[float] | None = None,
              region_band_mm: float | None = None, allowable_mpa: float | None = None,
              percentile: float = DEFAULT_PERCENTILE, solver: str | None = None, max_gb: float | None = None,
              force: bool = False, estimate_only: bool = False, yaml_path=None,
              near_region_mm: float = DEFAULT_NEAR_MM, load_sensitivity: bool = True, say=_noop,
              buckling: bool | None = None, bore_checks: bool | None = None, part_only: bool = False, gate: bool = False) -> Plan:
    """Everything that `verify_run` checks and computes before the first mesh. It writes nothing and meshes nothing.

    `buckling`: None follows `buckling.check` of the YAML of the run; True or False overrides it (`--buckling`, `--no-buckling`).
    `bore_checks`: None follows the block `bore_checks` of the YAML (on when it is there); False is off; True is on and needs the block
    (`--bore-checks`, `--no-bore-checks`).
    `gate`: the opt-in hard gate (`--gate`, K1-B). The plan reads the limits of the YAML (`gate.gate_limits`) and turns the buckling
    check on when the YAML has `buckling.min_factor`. A gate with no limit to check is a `UserError`, and so is `buckling=False` with
    a `buckling.min_factor`.
    `part_only`: mesh `result/part_only.stl` (a run with `regions.not_printed`) instead of the usual file (`--part-only`). The key `part_only`
    of the base has the file, its sha256 and the removed bodies. Without the option, a run that has the file gets one line in `Plan.notes`
    (and in `say`) that names the option; `verify.json` does not change.

    Raise `UserError` (bad input; `SolverUnavailable` when no direct factor can run or the named one cannot: exit
    code 1), `VerifyUnavailable` (a missing gmsh or scikit-fem; not for `estimate_only`) and
    `MemoryRefused` (the estimate is above the limit; not for `estimate_only`, and not with `force`).
    With `estimate_only`, a solver that cannot run is a fact in `backend.problem`, not an error.
    The MCP server calls it before it starts a detached run, so that a wrong input is answered at once.
    """
    _check_options(mesh_size, levels, region_band_mm, allowable_mpa, percentile, max_gb, near_region_mm)
    from .factor import AUTO_ORDER, check_name, factor_availability, resolve_factor

    try:
        check_name(solver)
    except ValueError as exc:
        raise UserError(str(exc)) from exc
    info = read_run(run_dir, yaml_path, part_only=part_only)
    _check_loads_and_supports(info.cfg)
    if bore_checks and info.cfg.bore_checks is None:
        raise UserError(f"--bore-checks needs a bore_checks block in the YAML ({info.yaml_path}): the block gives the bores, the strengths "
                        "and the sources. Options: add the block (CONTRACTS.md section 40), or leave the flag out")
    bc_on = info.cfg.bore_checks is not None if bore_checks is None else bool(bore_checks)
    band = float(region_band_mm) if region_band_mm is not None else DEFAULT_BAND_FACTOR * info.voxel_mm
    avail = factor_availability()
    solver_problem = None
    try:
        chosen = resolve_factor(solver, avail)
    except ValueError as exc:
        raise UserError(str(exc)) from exc
    except SolverUnavailable as exc:
        if not estimate_only:
            raise
        chosen, solver_problem = None, str(exc)
    skipped = {}
    if chosen is not None and (solver is None or solver == "auto"):
        for cand in AUTO_ORDER:                     # the factors that `auto` tried before it took `chosen`
            if cand == chosen:
                break
            skipped[cand] = avail.get(cand, {}).get("note", "not available")
    extras = extras_facts()
    if not estimate_only:
        require_extras(extras)

    geom = stl_geometry(info.final_stl)
    if levels:
        sizes = sorted({float(s) for s in levels}, reverse=True)
    else:
        sizes = [float(mesh_size) if mesh_size is not None else geom["default_size_mm"]]
    lim = memory.limits(max_gb)
    n_cases = len(info.cfg.load_cases)
    gate_limits = vgate.gate_limits(info.cfg, allowable_mpa) if gate else None
    gate_bk = vgate.gate_plan(gate_limits, buckling_flag=buckling) if gate else False
    bk_on = bool(info.cfg.buckling.check if buckling is None else buckling) or gate_bk
    estimates = [level_estimate(geom, s, lim, n_cases, chosen, load_sensitivity,
                                info.cfg.buckling.modes if bk_on else None) for s in sizes]
    if bc_on and info.cfg.bore_checks.press_fit:      # a key of each entry only with a press_fit block (plan 6.6)
        from . import pressfit as vpressfit
        for e in estimates:
            e["pressfit"] = vpressfit.estimate_before_meshing(info.cfg.bore_checks, e["estimate_bytes"], e["estimated_dofs"], chosen, e["compactness"])
            if info.cfg.joints:                            # K1-A: the stage is skipped for a joint; the estimate says so before the mesh is made
                e["pressfit"]["skipped_with_joints"] = PRESSFIT_JOINT_FACT
    base = dict(
        format=FORMAT, kind="topopt-verify",
        run=dict(run_dir=str(info.run_dir), name=info.cfg.name, yaml_path=str(info.yaml_path),
                 final_stl=str(info.final_stl), voxel_size_mm=info.voxel_mm, final_stl_sha256=info.final_sha256,
                 verified_file=info.verified_file),
        yaml_vs_run=compare_yaml_run(info),
        **({"part_only": info.part_only} if info.part_only is not None else {}),
        material=dict(E_MPa=info.cfg.material.E, nu=info.cfg.material.nu),
        stl=dict(volume_mm3=geom["volume_mm3"], area_mm2=geom["area_mm2"], default_mesh_size_mm=geom["default_size_mm"]),
        band_mm=band, band_is_default=region_band_mm is None, near_region_mm=float(near_region_mm),
        load_sensitivity=bool(load_sensitivity),
        backend=dict(solver=chosen, requested=solver or "auto", order=list(AUTO_ORDER), skipped=skipped, available=avail,
                     problem=solver_problem, refinement_steps=REFINE_STEPS),
        extras=extras, platform=f"{sys.platform} {platform.machine()}",
        memory_limit=dict(available_bytes=lim["available_bytes"], max_gb_bytes=lim["max_gb_bytes"],
                          limit_bytes=lim["limit_bytes"], limit_source=lim["limit_source"], force=bool(force)),
        estimates_before_meshing=estimates,
    )
    if allowable_mpa is None and not gate:      # a key only without an allowable (and without the gate, which has its own limits)
        base["no_allowable"] = no_allowable_fact()
    if bk_on:      # a key only when the check is on: with it off, verify.json has no new key
        b = info.cfg.buckling
        base["buckling_check"] = dict(on=True, source="the flag --buckling" if buckling else
                                      "the gate (buckling.min_factor in the YAML of the run)" if gate_bk and not info.cfg.buckling.check
                                      else "buckling.check in the YAML of the run",
                                      modes=b.modes, max_seconds=b.max_seconds)
    if bc_on:      # a key only when the checks are on: with them off, verify.json has no new key
        base["bore_checks"] = _bore_checks_block(info.cfg.bore_checks, bore_checks, percentile)
    for d in base["yaml_vs_run"]["differences"]:
        say(f"note: the YAML differs from run.json: {d}")
    notes = []              # a fact line that is not in verify.json (a run with part_only.stl, verified without the option)
    if info.part_only is not None:
        for text in info.part_only["yaml_vs_run"]["differences"] + info.part_only.get("notes", []):
            say("note: " + text)
    elif info.part_only_exists:
        notes.append(vpart_only.available_note(Path(info.final_stl).name, estimate_only))
        say("note: " + notes[0])
    if not estimate_only:
        for line in solver_lines(base["backend"]):          # the factor that runs is a printed fact, never a silent switch
            say(line)
        # one check before any work: a level that is already above the limit is refused with all the numbers
        for e in estimates:
            say(f"estimate for mesh size {e['size_mm']:g} mm: about {e['estimated_dofs']:,} dofs, "
                f"{memory.gb_text(e['estimate_bytes'])}"
                + (f", about {e['estimated_seconds']:.0f} s" if e["estimated_seconds"] is not None else "")
                + (f" (range {memory.gb_text(e['range_bytes']['low'])} to {memory.gb_text(e['range_bytes']['high'])})"
                   if e["range_bytes"] else "")
                + (" (above the limit)" if e["above_limit"] else ""))
            if "buckling_bytes" in e:
                say(f"  with the buckling check: {memory.gb_text(e['estimate_with_buckling_bytes'])} "
                    f"(+{memory.gb_text(e['buckling_bytes'])} on top of the factor)"
                    + ("; above the limit: the check can be skipped after meshing (the decision uses the real counts of the "
                       f"mesh; {opt('force')} starts the check) and then the stress check goes on"
                       if e["above_limit_with_buckling"] and not e["above_limit"] and not force else ""))
            if "pressfit" in e:
                say("  " + pressfit_estimate_text(e["pressfit"]))
            for text in e["notes"]:
                say("  note: " + text)
        for e in estimates:
            _check_estimate(e["estimate_bytes"], lim, e["size_mm"], "before meshing", e["estimated_dofs"], force, chosen,
                            e["compactness"])
    return Plan(info, band, chosen, sizes, estimates, lim, base, bk_on, bc_on, notes, gate_limits)


def _bore_checks_block(bc, flag, percentile) -> dict:
    """The base key `bore_checks` (plan 7.3): what is on, why, the knobs with their origin and the resolved bores."""
    from ..config import BORE_KNOB_DEFAULT_SOURCE, BORE_KNOB_KEYS
    from .fatigue import STATISTICS
    knobs = {}
    for k in BORE_KNOB_KEYS:
        d = dict(value=getattr(bc.knobs, k), origin=bc.knobs.origin(k))
        src = bc.knobs.source_of(k)
        if src:
            d["source"] = src
        elif d["origin"] == "default":
            d["default_source"] = BORE_KNOB_DEFAULT_SOURCE[k]
        knobs[k] = d
    bores = [dict(name=b.name, point=list(b.point), axis=list(b.axis), radius_mm=b.radius_mm, length_mm=b.length_mm,
                  geometry_from=b.geometry_from, sources=dict(b.sources),
                  **({"diametral_interference_mm": b.diametral_interference_mm} if b.diametral_interference_mm is not None else {}),
                  **({"mean_stress_mpa": b.mean_stress_mpa} if b.mean_stress_mpa is not None else {}))
             for b in bc.bores]
    for b, spec in zip(bores, bc.bores):
        if spec.geometry_from != "yaml":
            b["length_note"] = "length_mm is the cut length of the C2 entry; here it is the contact length"
    out = dict(on=True, source="the flag --bore-checks" if flag else "bore_checks in the YAML of the run", checks=list(bc.checks()),
               statistics=list(STATISTICS), knobs=knobs, bores=bores)
    if float(percentile) != DEFAULT_PERCENTILE:
        out["percentile_note"] = (f"--percentile {percentile:g} applies to the stress numbers of the whole part; the bore statistics stay "
                                  "p99.9 and max")
    return out


def _files_block(sizes, ref_size) -> dict:
    """The names of the files, relative to the `verify` folder. With several sizes, one folder for each size."""
    def sub(size):
        return "" if len(sizes) == 1 else f"h{size:g}/"
    files = dict(verify="verify.json", mesh=f"{sub(ref_size)}mesh.npz", fields=f"{sub(ref_size)}fields.npz")
    if len(sizes) > 1:
        files["levels"] = {f"{s:g}": dict(mesh=f"{sub(s)}mesh.npz", fields=f"{sub(s)}fields.npz") for s in sizes}
        files["note"] = "mesh and fields are the files of the reference level (the finest size)"
    return files


def verify_run(run_dir, *, mesh_size: float | None = None, levels: list[float] | None = None,
               region_band_mm: float | None = None, allowable_mpa: float | None = None,
               percentile: float = DEFAULT_PERCENTILE, solver: str | None = None, max_gb: float | None = None,
               force: bool = False, estimate_only: bool = False, yaml_path=None, threads: int = 1,
               optimize: str | None = DEFAULT_OPTIMIZE, write: bool = True,
               near_region_mm: float = DEFAULT_NEAR_MM, load_sensitivity: bool = True, log=None,
               buckling: bool | None = None, bore_checks: bool | None = None, part_only: bool = False, gate: bool = False) -> dict:
    """Check the final STL of a run with quadratic tets. Return a dict; write `<run_dir>/verify/`.

    Arguments (the options of `topopt verify`):

    - `mesh_size`: one mesh size in mm. Default: the size of the tet mesher (`default_size`, 2.5 times the mean
      edge of the STL, about 1.3 to 2.2 voxels for an `sdf` STL; 1.94 mm on a new `sdf` run of the hook example at
      1.2 mm). `levels`: a list of sizes (mm), solved one after the other (the finest is the reference level). Finer
      sizes cost much more memory and time; every size comes with its estimate (memory and time).
    - `region_band_mm`: the band of the region rule. Default: 0.5 times the voxel size of the run.
    - `allowable_mpa`, `percentile`: with an allowable, the result has a `limit` block and the exit code is 3 if
      the percentile value of a case is above it. There is no default allowable.
    - `solver`: `None` or "auto" (cholesky on macOS arm64, then pardiso, then cholespy, then scipy SuperLU), or one of
      `topopt.verify.factor.FACTOR_NAMES`. An unknown name raises `UserError`, and so does a factor that cannot run here
      (`SolverUnavailable`, exit code 1). The result names the factor that ran (`backend`), the factors that `auto` skipped
      and the estimates of memory and time.
    - `max_gb`: memory limit in GB. The check refuses to start above the available RAM or above this value.
      `force`: run anyway.
    - `estimate_only`: no meshing; return the estimates and the facts about the machine.
    - `yaml_path`: the problem YAML, if the path in `run.json` does not exist (a run folder that was moved).
    - `threads`, `optimize`: the gmsh thread count (1 gives a repeatable mesh) and optimizer step.
    - `near_region_mm`: the distance (mm) from a fixed or load region for the stress numbers "without the region patches".
    - `load_sensitivity`: also solve the load at a band of 0 and of 2 times the band (the factor is reused).
    - `write`: write the `verify/` folder (default). A real run removes the files of an earlier run first.
    - `buckling`: the linear buckling check of the tet model (key `buckling` of the result). None follows `buckling.check` of
      the YAML of the run (default off); True or False overrides it. It uses the factor of the static solve (no second factor);
      `buckling.modes` and `buckling.max_seconds` of the YAML are its settings.
    - The A5 rows of the YAML (`limits.displacement`, `limits.alignment`) are computed from the tet displacement
      (key `motion`) when the YAML has them.
    - `bore_checks`: the bore checks of the block `bore_checks` of the YAML (keys `bore_checks`, `bore_facts`, `goodman`, `pressfit`).
      None follows the YAML (on when the block is there); False is off; True is on and needs the block.
    - `part_only`: mesh `result/part_only.stl` (the part without the bodies of `regions.not_printed`) instead of `nominal.stl` or
      `final.stl`. A run without that file is a `UserError`. The result has the key `part_only`: the file and its sha256, the removed
      bodies with the boundary condition of each (a fixed body clamps its contact face, a load body puts its traction there, a solid or
      keep-out body gives none) and the singular note. Without the option the file is not read; a run that has it gets a line in `notes`
      of the returned dict (not in `verify.json`).

    - `gate`: the opt-in hard gate (`topopt verify --gate`, `topopt.verify.gate`). The result has the key `gate` with a `met` for each check
      and each case, and `passed`. The limits are the keys of the YAML (`stress_limit` or `allowable_mpa` of each case at `percentile`,
      `buckling.min_factor`, the rows of `limits`); `allowable_mpa` of this function replaces the stress limit of every case. It turns the
      buckling check on when the YAML has `buckling.min_factor`. A check that could not run has `met: null` and fails the gate. With
      `levels`, the gate reads the finest level and gives the change from the level before it. The exit code is 3 if the gate fails.

    The result has `partial` and `coverage` (TC-2): `partial` is true when a sub-check that the inputs ask for failed or did not run;
    `coverage` lists the sub-checks that `ran`, those that `failed` (with the reason) and those that were `not_run` (with the reason, and
    `asked: false` for one that the inputs do not ask for, which does not make the result partial). An `estimate_only` result has neither.

    The result has `exit_code` (0, or 3 if the allowable is exceeded or the gate fails) and `ok`. Raise `UserError` for bad input or
    no usable solver (exit code 1) and `VerifyError` (exit code 2) if the check cannot run.
    """
    from .status import StatusWriter, clear_outputs

    say = log or _noop
    plan = preflight(run_dir, mesh_size=mesh_size, levels=levels, region_band_mm=region_band_mm,
                     allowable_mpa=allowable_mpa, percentile=percentile, solver=solver, max_gb=max_gb, force=force,
                     estimate_only=estimate_only, yaml_path=yaml_path, near_region_mm=near_region_mm,
                     load_sensitivity=load_sensitivity, say=say if not estimate_only else _noop, buckling=buckling,
                     bore_checks=bore_checks, part_only=part_only, gate=gate)
    info, base, sizes, estimates, lim = plan.info, plan.base, plan.sizes, plan.estimates, plan.lim
    if estimate_only:
        base.update(ok=True, estimate_only=True, exit_code=EXIT_OK,
                    would_refuse=[e["size_mm"] for e in estimates if e["above_limit"] and not force])
        if plan.notes:
            base["notes"] = list(plan.notes)
        return base

    out_root = info.run_dir / "verify"
    previous = None
    sw = None
    if write:
        previous = _previous_verify(out_root, info.final_sha256)
        removed = clear_outputs(out_root)
        secs = [e["estimated_seconds"] for e in estimates]
        sw = StatusWriter(out_root, run_dir=str(info.run_dir), sizes_mm=sizes, solver=plan.chosen,
                          estimated_seconds_total=None if None in secs else float(sum(secs)),
                          estimates=[dict(size_mm=e["size_mm"], estimated_dofs=e["estimated_dofs"],
                                          estimated_seconds=e["estimated_seconds"], estimate_bytes=e["estimate_bytes"])
                                     for e in estimates])
        sw.update("starting", message=f"{len(sizes)} mesh size(s): {', '.join(f'{s:g}' for s in sizes)} mm")
    t_start = time.perf_counter()
    done = []
    try:
        for i, (size, est) in enumerate(zip(sizes, estimates)):
            sub = out_root if len(sizes) == 1 else out_root / f"h{size:g}"

            def progress(state, text, i=i, size=size):
                if sw is not None:
                    sw.update(state, level=i + 1, levels=len(sizes), size_mm=size, message=text)
            done.append(run_level(info, size, solver=plan.chosen, band_mm=plan.band, percentile=percentile, lim=lim,
                                  force=force, threads=threads, optimize=optimize, pre_estimate=est,
                                  out_dir=sub if write else None, say=say, near_mm=near_region_mm,
                                  sensitivity=load_sensitivity, progress=progress, buckling=plan.buckling,
                                  bore_checks=plan.bore_checks, from_levels=bool(levels)))
    except Exception as exc:
        if sw is not None:
            sw.update("error", error=str(exc), error_type=type(exc).__name__,
                      exit_code=EXIT_BAD_INPUT if isinstance(exc, UserError) else EXIT_COULD_NOT_RUN)
        raise
    ref = done[-1]                                      # the finest level is the reference
    report = dict(base)
    report.update(ok=True, estimate_only=False, reference_level_mm=ref["size_mm"])
    report.update({k: v for k, v in ref.items() if k not in ("size_mm", "files")})
    report["size_mm"] = ref["size_mm"]
    if "part_only" in ref:                              # the plan of the option (known before meshing) with the facts of the reference level
        report["part_only"] = vpart_only.merge_block(base["part_only"], ref["part_only"])
    report["peak_rss_bytes"] = ref["memory"]["peak_rss_bytes"]
    if previous is not None:
        report["previous_verify"] = dict(previous, removed=removed)
    if len(done) > 1:
        report["levels"] = levels_block(done)
    if allowable_mpa is not None:
        report["limit"] = limit_block(ref, allowable_mpa, percentile)
    report["timing_s"]["verify_total_s"] = time.perf_counter() - t_start
    if plan.gate_limits is not None:
        report["gate"] = vgate.evaluate(ref, plan.gate_limits, percentile=percentile, previous=done[-2] if len(done) > 1 else None)
    failed_limit = bool(report.get("limit", {}).get("exceeded")) or (plan.gate_limits is not None and not report["gate"]["passed"])
    report["exit_code"] = EXIT_LIMIT if failed_limit else EXIT_OK
    # TC-2: one summary of the sub-checks that ran, failed or were not run, read from the records above (`topopt.check_coverage`). It changes no number
    # and no exit code: a partial result is still a result.
    report = check_coverage.attach(report, check_coverage.from_verify(report, done, plan), after="kind")
    if write:
        report["files"] = _files_block(sizes, ref["size_mm"])
        out_root.mkdir(parents=True, exist_ok=True)
        path = out_root / "verify.json"
        path.write_text(json.dumps(report, indent=1, default=_json_default))
        report["verify_json"] = str(path)
        report["verify_dir"] = str(out_root)
        report["files_abs"] = {k: str(out_root / v) for k, v in report["files"].items() if isinstance(v, str)}
        sw.update("done", exit_code=report["exit_code"], message="done", verify_json=str(path),
                  seconds=report["timing_s"]["verify_total_s"])
    if plan.notes:                                      # after the write: the file keeps the keys of today (CONTRACTS.md section 16)
        report["notes"] = list(plan.notes)
    return report


def _previous_verify(out_root: Path, sha: str) -> dict | None:
    """Facts about an earlier `verify.json` in the folder (before it is removed): was it for the same final STL?"""
    try:
        old = json.loads((out_root / "verify.json").read_text())
        old_sha = old.get("run", {}).get("final_stl_sha256")
    except (OSError, ValueError, AttributeError):
        return None
    return dict(final_stl_sha256=old_sha, same_final_stl=bool(old_sha == sha),
                note="an earlier verify.json was in the folder; it was removed before this run")
