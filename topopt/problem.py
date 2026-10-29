"""The `Problem` object: output of the input layer (CONTRACTS.md section 3)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from . import symmetry
from .config import Config, UserError
from .loads import build_forces, check_rigid_modes, fixed_dofs_axes, fixed_dofs_from, lost_force, no_load_refusal, reach_kinds, reaches_free_dofs
from .voxelize import FIXED, voxelize


@dataclass
class Problem:
    config: Config
    origin: np.ndarray            # (3,) float64, mm
    h: float                      # mm
    shape: tuple[int, int, int]   # (nx, ny, nz)
    labels: np.ndarray            # (nx, ny, nz) uint8
    load_region_ids: np.ndarray   # (nx, ny, nz) int16
    fixed_dofs: np.ndarray        # int64, sorted, unique
    forces: np.ndarray            # (n_cases, 3*(nx+1)*(ny+1)*(nz+1)) float64
    case_names: list[str]
    case_weights: np.ndarray      # (n_cases,) float64
    region_facts: dict = field(default_factory=dict)   # counts from voxelize (solid regions, section 22a; `planes`: flat faces off the voxel grid)
    mirror: dict | None = None    # facts of the `mirror` key (section 26.1). None without it
    # CONTRACTS.md section 26.2. None when the YAML has no moment, remote force, pressure or per-axis support.
    # Else a dict: "entries" (for each case a list of `loads.LoadEntry`) and "supports" (facts of the supports).
    extended: dict | None = None
    # CONTRACTS.md section 35: the springs (`springs.SpringSet`), None when the YAML has no `springs`.
    springs: object | None = None
    # CONTRACTS.md section 38: the joints (`joints.JointSet`), None when the YAML has no `joints`.
    joints: object | None = None
    # F4-37: for each case, False if its load vector is exactly 0.0 on every free dof (`loads.reaches_free_dofs`). False for a case that
    # applies no load too. () for a Problem that a script builds by hand: the readers then take the YAML numbers only.
    load_reaches_free_dofs: tuple = ()
    # F4-37 (review m1): for each case, why its load reaches a free dof or not (`loads.reach_kinds`: free, nodal_zero, fixed, fixed_net). The words of the
    # refusal and the fact read it. () for a Problem that a script builds by hand.
    load_reach: tuple = ()


def resolve_volume(config: Config, labels: np.ndarray, h: float) -> Config:
    """The config with `settings.volume_fraction` set from `settings.target_mass_g` (I7, CONTRACTS.md section 3). Else `config` itself.

    The rule is the one of `volume_basis.volume_target` read backward: fraction = (target volume - forced-solid volume) / free design volume,
    with the counts of this grid (`labels`: the free design voxels are DESIGN, the forced-solid voxels are FIXED and LOAD, a `regions.solid` voxel
    and a union interface (boss, tube) are LOAD). `settings.target_mass_g` stays in the settings, so the run records where the fraction came from.
    A target at or below the mass of the forced-solid voxels, and a target above the mass of all active voxels solid, raise `UserError` with options.
    The numbers in the texts are the ones that `topopt check` reports (`derived.mass_active_block_g`, `derived.volume.forced_solid_mm3`).
    """
    st = config.settings
    if st.target_mass_g is None:
        return config
    from .volume_basis import fraction_for_mass, mass_of
    from .voxelize import DESIGN, FIXED, LOAD

    density = config.material.density
    n_free = int((labels == DESIGN).sum())
    n_forced = int(((labels == FIXED) | (labels == LOAD)).sum())
    n_active = n_free + n_forced
    vox = float(h) ** 3
    forced_g, block_g = mass_of(n_forced * vox, density), mass_of(n_active * vox, density)
    target = st.target_mass_g
    both = (f"The grid has {n_free} free design voxels and {n_forced} forced-solid voxels (fixed, load and solid regions, and union interfaces), voxel {h:g} mm, "
            f"density {density:g} g/cm3")
    if n_free == 0:   # `voxelize` refuses a design space with no free design voxel first, so a normal run never gets here; the guard keeps this function true alone
        raise UserError(f"settings.target_mass_g {target:g} g cannot set a volume fraction: the design space has no free design voxel "
                        f"({both}). Options: a larger design space, or smaller fixed, load and solid regions or union interfaces")
    fraction = fraction_for_mass(target, h, density, n_active, n_free)
    if fraction <= 0.0:
        raise UserError(f"settings.target_mass_g {target:g} g is not above the mass of the forced-solid voxels, {forced_g:.3f} g "
                        f"({n_forced * vox:.1f} mm3), which stay solid whatever the optimizer does ({both}). Options: a target above "
                        f"{forced_g:.3f} g; or smaller fixed, load and solid regions or union interfaces (they lower the forced-solid mass)")
    if fraction > 1.0:
        raise UserError(f"settings.target_mass_g {target:g} g is above the mass of all active voxels solid, {block_g:.3f} g "
                        f"(derived.mass_active_block_g of topopt check; {n_active * vox:.1f} mm3). The part cannot weigh more than the "
                        f"design space holds ({both}). Options: a target at or below {block_g:.3f} g; or a larger design space (regions.design)")
    return dataclasses.replace(config, settings=dataclasses.replace(st, volume_fraction=fraction))


def with_volume(config: Config, problem: "Problem") -> Config:
    """`config` with the settings of `problem.config`, which hold the volume fraction that `settings.target_mass_g` gave (I7).

    For code that keeps its own `Config` next to the `Problem` (`check_data`, the part builder). A config without the key is returned as it is.
    """
    if config.settings.target_mass_g is None:
        return config
    return dataclasses.replace(config, settings=problem.config.settings)


def build_problem(config: Config) -> Problem:
    """Load STLs, voxelize, and build loads and fixed dofs.

    With `settings.target_mass_g` the volume fraction is derived here, from the labels (`resolve_volume`): `Problem.config.settings.volume_fraction`
    is a number, and every stage that reads the problem sees it. A config that was not built (the result of `load_config`) has None there.
    """
    config, mirror = symmetry.resolve(config)   # mirror key: the parity of each plane. Else (config, None)
    out: dict = {}
    origin, h, shape, labels, load_ids = voxelize(config, out)
    if config.regions.fixed_axes:     # a `{file, axes}` entry (section 26.2)
        fixed_dofs = fixed_dofs_axes(config, out["fixed_members"])
        if config.has_partial_supports() and not config.has_springs() and not config.has_joints():   # the supports block the six rigid-body modes
            out["supports"] = check_rigid_modes(fixed_dofs, shape, h)
    else:
        fixed_dofs = fixed_dofs_from(labels == FIXED)
    springs = None
    if config.has_springs():          # springs are supports: the rigid-body test counts their dofs (section 35)
        from .springs import build_springs
        springs = build_springs(config, labels, load_ids, h)
        out["springs"] = springs.facts
        if not config.has_joints():
            out["supports"] = check_rigid_modes(fixed_dofs, shape, h, springs.dofs)
    joints = None
    if config.has_joints():           # a joint blocks all six rigid-body modes by its beam: the test is skipped (section 38)
        from .joints import build_joints
        joints = build_joints(config, labels, load_ids, origin, h)
        out["joints"] = joints.facts
        out["supports"] = {"fixed_dofs": int(len(fixed_dofs)), "rigid_body_test": (
            "skipped: a joint blocks all six rigid-body modes, because its beam holds the rigid body that carries the part")}
    forces = build_forces(config, load_ids, labels, origin, h, out)
    reaches = reaches_free_dofs(forces, fixed_dofs)   # F4-37: a case whose load vector is 0.0 on the free dofs feels no load
    reach = reach_kinds(forces, fixed_dofs, lambda: lost_force(config, load_ids, fixed_dofs, forces, out.get("entries")))   # and why (the words)
    refusal = no_load_refusal(config, reach)          # the U21 rule for it: refused next to a row that cannot work with it, else a fact
    if refusal:
        raise UserError(refusal)
    extended = {k: out[k] for k in ("entries", "supports", "springs", "joints") if k in out} or None
    region_facts = {k: out[k] for k in ("solid", "planes", "taken") if k in out}   # the A3a keys (section 22a), the planes off the voxel grid (P1), what the cuts take (section 23)
    if mirror:   # facts of the planes: label mismatches, orphans, load asymmetry
        mirror = symmetry.problem_facts(mirror, origin, h, labels, forces, [c.name for c in config.load_cases])
    config = resolve_volume(config, labels, h)   # I7: settings.target_mass_g -> volume_fraction; after the other input errors
    return Problem(
        config=config,
        origin=origin,
        h=h,
        shape=shape,
        labels=labels,
        load_region_ids=load_ids,
        fixed_dofs=fixed_dofs,
        forces=forces,
        case_names=[c.name for c in config.load_cases],
        case_weights=np.array([c.weight for c in config.load_cases], dtype=np.float64),
        region_facts=region_facts,
        mirror=mirror,
        extended=extended,
        springs=springs,
        joints=joints,
        load_reaches_free_dofs=tuple(reaches),
        load_reach=tuple(reach),
    )
