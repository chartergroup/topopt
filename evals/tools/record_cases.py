"""Record real topopt MCP answers for the case-level mocks and the user files of the cases.

No optimizer job: build_regions and check_problem only (a few seconds each).

Usage (in the repo root):
    sh scripts/topopt-env.sh python evals/tools/record_cases.py \
        "$PWD/scripts/topopt-env.sh" <empty work folder> evals

It writes:
  voxel-guard/mocks/topopt/fixtures/build_regions.json   the hook at voxel 0.3 mm (a guard stops it)
  lost-force/files/                                      a YAML and STLs where the load region overlaps the wall plate
  lost-force/mocks/topopt/fixtures/check_problem.json    the check of that YAML (lost_force)
  user-files/files/                                      the YAML and STLs of the short hook
"""
import asyncio
import copy
import shutil
import sys
from pathlib import Path

from mcp_record import recorder
from record_fixtures import HOOK_SPEC

SCREWS = {
    "screw_hole_top": [
        {"cylinder": {"p0": [-1, 0, 0], "p1": [5, 0, 0], "radius": 2.25}},
        {"cylinder": {"p0": [4, 0, 0], "p1": [140, 0, 0], "radius": 4.5}}],
    "screw_hole_bottom": [
        {"cylinder": {"p0": [-1, 0, -35], "p1": [5, 0, -35], "radius": 2.25}},
        {"cylinder": {"p0": [4, 0, -35], "p1": [140, 0, -35], "radius": 4.5}}],
}

POT_BRACKET_SPEC = {
    "name": "pot_bracket",
    "regions": {
        "design": [{"prism": {"plane": "xz", "extent": [-12, 12],
                              "points": [[3, -45], [120, 6], [120, 10], [3, 10]]}}],
        "fixed": {"wall_plate": [{"box": {"min": [0, -15, -45], "max": [4, 15, 10]}}]},
        "keep_out": SCREWS,
        "load": {"pot_seat": [{"box": {"min": [80, -12, 6], "max": [120, 12, 10]}}]},
    },
    "load_cases": [
        {"name": "pot", "weight": 1.0, "forces": [{"region": "pot_seat", "force": [0, 0, -49.05]}]},
        {"name": "pot_bump", "weight": 1.0,
         "forces": [{"region": "pot_seat", "force": [0, 15, -73.6]}]},
    ],
    "material": {"E": 3500.0, "nu": 0.35, "density": 1.24},
    "settings": {"voxel_size": "auto", "volume_fraction": 0.25, "iterations": 80},
    "assumptions": [
        "pot with soil and water: 5 kg x 9.81 = 49.05 N down",
        "abuse case: 1.5 x the weight plus a 15 N side push",
        "two screws into the wall, plate 4 mm",
        "PETG, E 3500 MPa",
    ],
}

# A user error: the load region starts at the wall, so it overlaps the fixed plate.
LOST_SPEC = copy.deepcopy(POT_BRACKET_SPEC)
LOST_SPEC["name"] = "shelf_bracket"
LOST_SPEC["regions"]["load"] = {"shelf_seat": [{"box": {"min": [0, -12, 6], "max": [10, 12, 10]}}]}
LOST_SPEC["load_cases"] = [
    {"name": "shelf", "weight": 1.0, "forces": [{"region": "shelf_seat", "force": [0, 0, -39.2]}]},
]
LOST_SPEC["assumptions"] = ["shelf with books: 4 kg x 9.81 = 39.2 N down", "PETG"]

FINE_HOOK_SPEC = copy.deepcopy(HOOK_SPEC)
FINE_HOOK_SPEC["settings"] = {"voxel_size": 0.3, "volume_fraction": 0.2, "iterations": 80}


def copy_problem(work, yaml_rel, dest):
    """Copy the YAML and the STL files of a built problem (not the case files)."""
    src = (Path(work) / yaml_rel).parent
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in src.iterdir():
        if f.suffix in (".yaml", ".stl"):
            shutil.copy2(f, dest / f.name)
    print(f"copied {src.name} -> {dest}", file=sys.stderr)


async def main(launcher, work, evals):
    evals = Path(evals)
    async with recorder(launcher, work) as rec:
        rec.save(evals / "voxel-guard/mocks/topopt/fixtures", "build_regions",
                 await rec.call("build_regions", spec=FINE_HOOK_SPEC))

        lost = await rec.call("build_regions", spec=LOST_SPEC)
        if not lost.get("yaml_rel"):
            raise SystemExit(f"lost-force spec wrote no YAML: {lost}")
        copy_problem(work, lost["yaml_rel"], evals / "lost-force/files")
        # Check the copy in <work>/files/, so the paths in the answer look like the user's folder.
        copy_problem(work, lost["yaml_rel"], Path(work) / "files")
        rec.save(evals / "lost-force/mocks/topopt/fixtures", "check_problem",
                 await rec.call("check_problem", yaml_path="files/shelf_bracket.yaml"))

        hook = await rec.call("build_regions", spec=HOOK_SPEC)
        copy_problem(work, hook["yaml_rel"], evals / "user-files/files")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:4]))
