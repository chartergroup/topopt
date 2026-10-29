"""The volume basis of a run, in both directions (CONTRACTS.md section 3, "The volume basis").

`settings.volume_fraction` is the mean density of the FREE design voxels (the optimizer reads it so: `volume_constraint`). The voxels that are
solid on top of it are the forced-solid voxels (fixed, load and solid regions: `n_active - n_free_vox`). So the target volume of a run is

    target = (forced + volume_fraction * free) * h^3        and        target_mass = target / 1000 * density      (mm3, g/cm3 -> g)

`volume_target` is this rule from a fraction to a volume and a mass (the numbers of `topopt check`, the run report and the part builder).
`fraction_for_mass` is the same rule from a mass to a fraction (`settings.target_mass_g`, I7). Both use `_basis`, so that the round trip
`mass -> fraction -> mass` is exact up to the float rounding. The module needs no numpy and imports nothing from the package.
"""

from __future__ import annotations


def _basis(h: float, n_active: int, n_free_vox: int) -> tuple[float, int, float]:
    """`(vox_mm3, forced_voxels, forced_mm3)`: the voxel volume, the forced-solid voxels and their volume."""
    vox_mm3 = h ** 3
    frozen = n_active - n_free_vox
    return vox_mm3, frozen, frozen * vox_mm3


def volume_target(volume_fraction: float, h: float, density: float, n_active: int, n_free_vox: int) -> dict:
    """The volume basis of a run: the voxel counts, the volumes in mm3 and the target mass in g.

    One rule for `topopt check`, the run report and the part builder (`regions_builder.check_yaml`).
    `volume_fraction` is the mean density of the FREE design voxels (as the optimizer reads it, `volume_constraint`);
    the forced-solid voxels (fixed, load, solid) are solid on top: target = (forced + vf * free) * h^3.
    """
    vox_mm3, frozen, _forced_mm3 = _basis(h, n_active, n_free_vox)
    target = (frozen + volume_fraction * n_free_vox) * vox_mm3
    return {
        "free_design_voxels": int(n_free_vox),
        "free_design_mm3": round(n_free_vox * vox_mm3, 2),
        "forced_solid_mm3": round(frozen * vox_mm3, 2),
        "target_volume_mm3": round(target, 2),
        "target_mass_g": round(mass_of(target, density), 3),
    }


def fraction_for_mass(target_mass_g: float, h: float, density: float, n_active: int, n_free_vox: int) -> float:
    """The `volume_fraction` that gives `target_mass_g` (grams) on this grid: the inverse of `volume_target`.

    fraction = (target volume - forced-solid volume) / free design volume. The result is not clamped: a value at or below 0 means that the forced-solid
    voxels alone weigh `target_mass_g` or more, and a value above 1 means that all active voxels solid weigh less. The caller refuses both and
    names the options (`topopt.problem.resolve_volume`). Needs `n_free_vox` above 0.
    """
    vox_mm3, _frozen, forced_mm3 = _basis(h, n_active, n_free_vox)
    target_mm3 = target_mass_g * 1000.0 / density
    return (target_mm3 - forced_mm3) / (n_free_vox * vox_mm3)


def mass_of(volume_mm3: float, density: float) -> float:
    """The mass in g of `volume_mm3` at `density` g/cm3, unrounded."""
    return volume_mm3 / 1000.0 * density


def mass_asked_line(metrics: dict | None) -> str | None:
    """The line of fact for a run that asked for a mass (`settings.target_mass_g`, I7): the final mesh against the asked mass.

    `metrics` is `result/metrics.json`. The line is made from `mass_asked_g` and `mass_final_g`; a run without `mass_asked_g` (a YAML with
    `volume_fraction`, or a run from before I7) has no line (None). With `metrics.part_only` the line adds the mass of `part_only.stl`:
    the asked mass holds the forced-solid voxels of the `regions.not_printed` bodies, and that file is without them. It is a fact only:
    no percent, no tolerance and no verdict word. The run log, `run_status` and the tests all take the text from this function.
    """
    if not isinstance(metrics, dict):
        return None
    asked, final = metrics.get("mass_asked_g"), metrics.get("mass_final_g")
    if isinstance(asked, bool) or isinstance(final, bool) or not isinstance(asked, (int, float)) or not isinstance(final, (int, float)):
        return None
    line = f"fact: final mesh {final:.2f} g against {asked:g} g asked (settings.target_mass_g)"
    part_only = metrics.get("part_only")
    if isinstance(part_only, dict) and isinstance(part_only.get("mass_g"), (int, float)) and part_only.get("applied"):
        line += f"; part_only.stl {part_only['mass_g']:.2f} g (without regions.not_printed)"
    return line
