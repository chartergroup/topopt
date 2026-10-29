"""A stand-in for a job process: take the job lock of a folder and end the job (a helper of tests/test_job_facts.py).

    python job_probe.py <folder>

The MCP server starts this in place of `topopt run` (`TOPOPT_MCP_CMD`) or of the ratchet (`TOPOPT_MCP_RATCHET_CMD`). The job record
(`job.json`) then shows who started the process.
"""
import sys

from topopt.runjob import run_job

raise SystemExit(run_job(sys.argv[1], lambda job: 0))
