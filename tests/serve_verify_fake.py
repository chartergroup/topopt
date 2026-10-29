"""A stand-in for `topopt verify` in the MCP tests (env var TOPOPT_MCP_VERIFY_CMD). It writes the status file and ends.

Usage: serve_verify_fake.py <run_dir> <mode> [--step SECONDS]
Modes: ok (starting, meshing, solving, writing, done and a small verify.json), error (a final state `error`),
die (it stops in `meshing` with no final status, as a process that the system killed).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from topopt.verify.status import StatusWriter  # noqa: E402

run_dir, mode = Path(sys.argv[1]), sys.argv[2]
step = float(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 0.2
out = run_dir / "verify"
sw = StatusWriter(out, run_dir=str(run_dir), sizes_mm=[1.5], solver="fake", estimated_seconds_total=1.0)
sw.update("starting", message="fake job")
time.sleep(step)
sw.update("meshing", level=1, levels=1, size_mm=1.5, message="meshing")
time.sleep(step)
if mode == "die":
    os._exit(9)
if mode == "error":
    sw.update("error", error="fake failure", exit_code=2)
    sys.exit(2)
sw.update("solving", level=1, levels=1, size_mm=1.5, message="solving")
time.sleep(step)
sw.update("writing", level=1, levels=1, size_mm=1.5, message="writing")
stl = run_dir / "result" / "final.stl"
if (run_dir / "result" / "nominal.stl").is_file():      # as the real command: the finished part when the run has stock
    stl = run_dir / "result" / "nominal.stl"
sha = hashlib.sha256(stl.read_bytes()).hexdigest() if stl.is_file() else None
(out / "verify.json").write_text(json.dumps({
    "format": 1, "ok": True, "exit_code": 0, "run": {"final_stl_sha256": sha, "verified_file": stl.name}, "size_mm": 1.5,
    "files": {"verify": "verify.json", "mesh": "mesh.npz", "fields": "fields.npz"}, "cases": {}}))
sw.update("done", exit_code=0, message="done")
