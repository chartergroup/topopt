"""Tests of the real child process of the OpenCL probe on pocl (#353, CONTRACTS.md section 20). The tests with a faked child are in `test_gpu_mg_probe.py`.

The parent process must make no OpenCL call and must not import pyopencl for the auto pick and the OpenCL facts. A subprocess checks it with the
real child. The pocl CPU device does not count as a GPU for `auto`, so the second test lets it count in the child (it replaces `counts_as_gpu` in the
child and calls the module `main`, as `python -m` does) and runs the self-test for real, twice: the second child reads the cache file.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyopencl")

from topopt.config import UserError  # noqa: E402
from topopt.gpu_mg import device  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.opencl_context


@pytest.fixture(scope="module", autouse=True)
def dev():
    """The device that `auto` takes. The tests are skipped when there is none (a UserError), as the device fixtures of the other gpu-mg tests:
    with pyopencl installed but no OpenCL platform (CI installs no OpenCL driver) the child lists no device."""
    try:
        return device.select("auto", "auto")[0]
    except UserError as e:
        pytest.skip(f"no usable OpenCL device: {e}")


def run_python(code: str, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "TOPOPT_AUTO_GPU"}
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TOPOPT_STATE_DIR"] = str(tmp_path / "state")
    return subprocess.run([sys.executable, "-P", "-c", code, *args], capture_output=True, text=True, env=env, cwd=tmp_path, timeout=300)


def test_the_parent_does_not_import_pyopencl_for_the_auto_pick_and_the_facts(tmp_path):
    code = ("import json, sys\n"
            "from topopt import solver_choice\n"
            "from topopt.gpu_mg import probe\n"
            "c = solver_choice.run_solver('auto')\n"
            "f = probe.facts('auto', 'auto')\n"
            "from topopt.environment import environment_data\n"
            "e = environment_data()\n"
            "from topopt_mcp import server\n"
            "m = server.environment()\n"
            "print('RESULT ' + json.dumps({'pyopencl_in_parent': 'pyopencl' in sys.modules, 'solver': c.solver, 'cpu': c.cpu, 'reason': c.reason,\n"
            "                                 'pyopencl': f['pyopencl'], 'env_devices': len(e['opencl']['devices']), 'mcp_devices': len(m['opencl']['devices']), 'gpu_mg': e['solver_setting']['gpu-mg']['resolved'], 'devices': len(f['devices']), 'problem': f['problem']}))\n")
    r = run_python(code, tmp_path)
    assert r.returncode == 0, r.stderr
    out = json.loads(next(line for line in r.stdout.splitlines() if line.startswith("RESULT "))[len("RESULT "):])
    assert out["pyopencl_in_parent"] is False
    assert out["env_devices"] >= 1 and out["mcp_devices"] >= 1 and out["gpu_mg"] == "gpu-mg"       # environment and the MCP tool: no pyopencl in the parent
    assert out["devices"] >= 1 and out["pyopencl"] and out["problem"] is None      # the child listed the devices of pocl
    assert out["solver"] == out["cpu"] and "not a GPU" in out["reason"]            # a CPU device does not count for auto


def test_the_second_child_reads_the_cache_file_and_runs_no_self_test(tmp_path):
    code = ("import sys, json\n"
            "from topopt.gpu_mg import device, probe\n"
            "device.counts_as_gpu = lambda d: True\n"
            "probe.main(['--device', 'auto', '--case', 'auto'])\n")
    answers = []
    for _ in range(2):
        r = run_python(code, tmp_path)
        assert r.returncode == 0, r.stderr
        line = [x for x in r.stdout.splitlines() if x.startswith("TOPOPT-PROBE ")][-1]
        answers.append(json.loads(line[len("TOPOPT-PROBE "):]))
    first, second = (a["self_test"] for a in answers)
    assert first["passed"] is True and first["cached"] is False and first["tested_at"]
    assert second["passed"] is True and second["cached"] is True and second["tested_at"] == first["tested_at"]
    assert (tmp_path / "state" / "gpu-selftest.json").is_file()
    assert answers[1]["device"]["driver"] and answers[1]["device"]["name"] == answers[0]["device"]["name"]
