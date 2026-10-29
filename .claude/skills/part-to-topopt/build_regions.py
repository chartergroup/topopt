#!/usr/bin/env python3
"""Build region STLs and a topopt problem YAML from a short spec.

Thin wrapper. The code is in the package: topopt/regions_builder.py
(the MCP tool `build_regions` calls the same code).

Run it with a Python that has topopt installed. In a source checkout:
    .venv/bin/python .claude/skills/part-to-topopt/build_regions.py SPEC.yaml --out DIR
    .venv/bin/python .claude/skills/part-to-topopt/build_regions.py SPEC.yaml --out DIR --check
In the Claude Code plugin, SKILL.md gives the command that uses the plugin environment
(scripts/topopt-env.sh python <this file> ...).
"""
import sys

from topopt.regions_builder import (CAP_ACTIVE, SpecError, build_spec, main,  # noqa: F401
                                    make_primitive, target_active, time_factor, to_trimesh)

if __name__ == "__main__":
    sys.exit(main())
