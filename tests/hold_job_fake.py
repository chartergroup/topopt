"""Test stand-in for a run process (P9-14): take the job lock of the run folder, wait until the file `release` is in that folder, end with code 0.

The process lives on after the server that started it ends (the server starts a run in its own session). The test that uses it writes `release`.
"""
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from topopt import runjob  # noqa: E402

out = Path(sys.argv[1])


def wait_for_release(job):
    while not (out / "release").exists():
        time.sleep(0.05)
    return 0


sys.exit(runjob.run_job(out, wait_for_release))
