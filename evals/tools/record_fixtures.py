"""Record real topopt MCP answers for the suite mocks (the T-shirt hook flow).

The script starts the real server, calls the tools of one short hook job and
writes each answer to <out>/<tool>.json. It runs ONE optimizer job (about 20 s).
Do not start it while another topopt job runs on this machine.

Usage (in the repo root):
    sh scripts/topopt-env.sh python evals/tools/record_fixtures.py \
        "$PWD/scripts/topopt-env.sh" <empty work folder> evals/mocks/topopt/fixtures

The work folder gets runs/tshirt_hook/. The fixtures hold placeholder paths
(see mcp_record.py), not the paths of this machine.

The two render_part fixtures are recorded with inline=false: the answer is the JSON facts
only (a mock answers text only). `render_part_yaml.json` is the render of the problem YAML
before the run (regions and force arrows). `render_part_run.json` is the render of the run
folder after the run (the part and the regions).
"""
import asyncio
import os
import signal
import sys
import time
from pathlib import Path

from mcp_record import recorder

# The spec of the skill example (examples/hook_spec.yaml), with the short
# settings of examples/hook/hook_short.yaml (voxel 2 mm, 40 passes).
HOOK_SPEC = {
    "name": "tshirt_hook",
    "regions": {
        "design": [{"prism": {"plane": "xz", "extent": [-14, 14],
                              "points": [[2, -30], [72, 8], [72, 11], [56, 15], [2, 36]]}}],
        "fixed": {"wall_plate": [{"box": {"min": [0, -17.5, -30], "max": [3.5, 17.5, 36]}}]},
        "keep_out": {
            "screw_hole_top": [
                {"cylinder": {"p0": [-1, 0, 30], "p1": [4, 0, 30], "radius": 2.25}},
                {"cylinder": {"p0": [3.5, 0, 30], "p1": [80, 0, 30], "radius": 4.5}}],
            "screw_hole_bottom": [
                {"cylinder": {"p0": [-1, 0, -20], "p1": [4, 0, -20], "radius": 2.25}},
                {"cylinder": {"p0": [3.5, 0, -20], "p1": [80, 0, -20], "radius": 4.5}}],
        },
        "load": {"hook_tip": [{"box": {"min": [56, -9, 11], "max": [76, 9, 15]}},
                              {"box": {"min": [73, -9, 11], "max": [76, 9, 24]}}]},
    },
    "load_cases": [
        {"name": "hang", "weight": 234.0, "forces": [{"region": "hook_tip", "force": [0, 0, -1.96]}]},
        {"name": "pull_off", "weight": 1.0,
         "forces": [{"region": "hook_tip", "force": [20.89, 5.21, -20.89]}]},
    ],
    "material": {"E": 3500.0, "nu": 0.35, "density": 1.24},
    "settings": {"voxel_size": 2.0, "volume_fraction": 0.2, "iterations": 40,
                 "beta_schedule": [[0, 1], [16, 2], [26, 4], [34, 8]]},
    "assumptions": [
        "shirt on a hanger: about 200 g, 1.96 N down",
        "pull-off at an angle: about 30 N",
        "two M4 screws into the wall, plate 3.5 mm",
        "PETG, E 3500 MPa",
    ],
}

BRIEF = ("A wall hook for one T-shirt on a hanger. Two screws hold the plate on the wall. "
         "Load 1: the shirt hangs (about 200 g). Load 2: a fast pull-off at an angle (about 30 N). "
         "PETG. Hook depth about 75 mm.")


async def main(launcher, work, out):
    async with recorder(launcher, work) as rec:
        env = await rec.call("environment")
        # The git state of the recording checkout (branch, dirty, behind) is noise for the evals:
        # a model reads it as a fact about the plugin under test.
        env.pop("checkout", None)
        rec.save(out, "environment", env)
        built = await rec.call("build_regions", spec=HOOK_SPEC)
        rec.save(out, "build_regions", built)
        if not built.get("ok"):
            raise SystemExit(f"build_regions not ok: {built}")
        yaml_rel = built["yaml_rel"]
        rec.save(out, "record_brief", await rec.call(
            "record_brief", yaml_path=yaml_rel, text=BRIEF,
            assumptions=[{"text": a, "origin": "inferred"} for a in HOOK_SPEC["assumptions"]]))
        rec.save(out, "check_problem", await rec.call("check_problem", yaml_path=yaml_rel))
        # The look before the run: the regions and the force arrows of the YAML. inline=false: the facts only.
        rec.save(out, "render_part_yaml", await rec.call("render_part", yaml_path=yaml_rel, inline=False))
        run_dir = str(Path(yaml_rel).parent / "run")
        started = await rec.call("run_topopt", yaml_path=yaml_rel, out_dir=run_dir)
        rec.save(out, "run_topopt", started)
        t0 = time.time()
        while True:
            await asyncio.sleep(3)
            st = await rec.call("run_status", run_dir=run_dir)
            if st.get("state") in ("done", "error"):
                rec.save(out, "run_status", st)
                break
            if time.time() - t0 > 900:
                raise SystemExit("run did not end in 15 min")
        rec.save(out, "run_result", await rec.call("run_result", run_dir=run_dir))
        # The look after the run: the part and the regions of the run folder.
        rec.save(out, "render_part_run", await rec.call("render_part", run_dir=run_dir, inline=False))
        rec.save(out, "print_check", await rec.call("print_check", path=run_dir))
        # After the run, the process only serves the dashboard: stop it.
        if started.get("pid"):
            try:
                os.kill(int(started["pid"]), signal.SIGTERM)
            except OSError:
                pass


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:4]))
