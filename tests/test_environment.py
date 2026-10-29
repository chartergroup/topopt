"""B-080: the `environment` tool and `topopt.environment.environment_data`.

The platform and the pypardiso import are faked (monkeypatch), so every rule runs on every platform.
"""
import asyncio
import json
import os
import platform
import subprocess
import sys
import types
from pathlib import Path

import pytest

import topopt
from topopt import cli, environment, solver_choice
from topopt.environment import environment_data
from topopt.gpu_mg import device
from helpers import SELF_TEST_FAILED, SELF_TEST_PASSED, fake_opencl, fake_shim, make_device
from test_optimize import make_case

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from topopt_mcp import server  # noqa: E402


@pytest.fixture
def fake_platform(monkeypatch, tmp_path):
    """`fake_platform("linux", "x86_64", pypardiso=None)`; `dylib` tells if the Mac shim exists and loads.

    `gpu` fakes the OpenCL state (no driver is loaded). The default is "pyopencl loads and no device is visible".
    `gpu="missing"`, `gpu="load_failed"`, or a list of `DeviceInfo` (see `make_device`) make the other states.
    `self_test` is the result of the self-test of `auto` on a fake device: the default is a failed test (the CPU rule is the pick).
    """
    def set_platform(plat, machine, pypardiso=None, dylib=False, shim=None, gpu=None, self_test=SELF_TEST_FAILED):
        monkeypatch.setattr(sys, "platform", plat)
        monkeypatch.setattr(platform, "machine", lambda: machine)
        if isinstance(gpu, str):
            fake_opencl(monkeypatch, gpu)
        else:
            fake_opencl(monkeypatch, "devices" if gpu else "no_device", gpu, self_test)

        def imp():
            if pypardiso is None:   # missing: the same error as the real import
                raise ModuleNotFoundError("No module named 'pypardiso'", name="pypardiso")
            return pypardiso
        monkeypatch.setattr(solver_choice, "import_pypardiso", imp)
        fake_shim(monkeypatch, tmp_path, shim or ("ok" if dylib else "missing"))
    return set_platform


def test_version_is_set():
    assert topopt.__version__ and topopt.__version__ != "unknown"


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"topopt {topopt.__version__}"


def test_code_folder_has_the_package_and_the_tools():
    """The agents of the plugin name tools/check_part.py and examples/: this is how they find them."""
    d = environment_data()
    folder = Path(d["code_folder"])
    assert (folder / "topopt" / "__init__.py").is_file()
    assert (folder / "tools" / "check_part.py").is_file() and (folder / "examples" / "hook" / "hook.yaml").is_file()


def test_real_machine_facts():
    d = environment_data()
    assert d["topopt_version"] == topopt.__version__
    assert d["python"]["executable"] == sys.executable
    assert d["python"]["version"] == platform.python_version()
    assert set(d["backends"]) == {"cholesky", "pardiso", "amg"}
    assert d["cores"]["usable"] >= 1
    assert d["ram"]["total_bytes"] is None or d["ram"]["total_bytes"] > 0
    assert d["problem"] is None
    json.dumps(d)   # the answer is plain data


def test_linux_without_pypardiso_offers_it_as_data(fake_platform):
    fake_platform("linux", "x86_64")
    d = environment_data()
    assert d["auto_solver"] == "amg"
    assert d["backends"]["pardiso"]["available"] is False
    assert d["backends"]["amg"]["resolved_by"] == ["auto", "amg"]
    assert d["solver_setting"]["cholesky"]["resolved"] is None and d["solver_setting"]["cholesky"]["error"]
    (rec,) = [r for r in d["recommendations"] if r["action"] == "install_pypardiso"]
    assert rec["speedup_factor"] == solver_choice.AMG_TIME_FACTOR
    assert "[pardiso]" in rec["install_command"] and "Intel Simplified Software License" in rec["license_note"]
    assert "reason" in rec


def test_linux_with_pypardiso_has_no_offer(fake_platform):
    fake_platform("linux", "x86_64", pypardiso=types.ModuleType("pypardiso"))
    d = environment_data()
    assert d["auto_solver"] == "pardiso" and d["backends"]["pardiso"]["available"] is True
    assert d["backends"]["pardiso"]["resolved_by"] == ["auto", "pardiso"]
    assert not [r for r in d["recommendations"] if r["action"] == "install_pypardiso"]


def test_mac_with_shim(fake_platform):
    fake_platform("darwin", "arm64", dylib=True)
    d = environment_data()
    assert d["auto_solver"] == "cholesky" and d["platform"]["apple_silicon"] is True
    assert d["backends"]["pardiso"]["available"] is False and "Linux x86_64" in d["backends"]["pardiso"]["note"]
    assert d["solver_setting"]["pardiso"]["resolved"] is None
    assert d["recommendations"] == []
    assert d["threads"]["openblas_threads_in_run"] is None   # no cap on macOS


def test_mac_without_shim_offers_the_build(fake_platform):
    fake_platform("darwin", "arm64", dylib=False)
    d = environment_data()
    assert d["auto_solver"] == "amg"          # without the shim, auto runs amg (it does not stop)
    (rec,) = d["recommendations"]
    assert rec["action"] == "build_accelerate_shim" and rec["command"].endswith("build.sh'")
    assert rec["command"].startswith("sh '") and "native dir" in rec["command"]   # quoted
    assert "speedup_factor" not in rec   # no measured Accelerate-vs-AMG figure exists
    assert "is not built" in rec["reason"] and "runs amg" in rec["reason"]
    assert f"{solver_choice.AMG_TIME_FACTOR:g}x" in rec["reason"] and "an estimate" in rec["reason"]
    assert "stops" not in rec["reason"]


def test_mac_with_a_shim_that_does_not_load_offers_the_build_again(fake_platform):
    fake_platform("darwin", "arm64", shim="broken")
    d = environment_data()
    assert d["auto_solver"] == "amg" and d["backends"]["cholesky"]["available"] is False
    assert "does not load" in d["backends"]["cholesky"]["note"]
    (rec,) = d["recommendations"]
    assert "is built but it does not load" in rec["reason"]


GIT_READ_ONLY = {"rev-parse", "symbolic-ref", "status", "rev-list", "log", "describe", "config"}


def test_tool_does_not_install_or_build(fake_platform, monkeypatch):
    fake_platform("linux", "x86_64")
    real_run = subprocess.run
    git_calls = []

    def boom(*a, **k):
        raise AssertionError("the environment tool must not run a command")

    def guarded(cmd, *a, **k):
        # The checkout facts read git. Only the read-only subcommands of git are allowed, and never a fetch.
        if cmd and cmd[0] == "git":
            assert cmd[1] == "-C"            # git -C <folder> <subcommand> ...
            sub = cmd[3]
            git_calls.append(sub)
            assert sub in GIT_READ_ONLY, cmd
            return real_run(cmd, *a, **k)
        return boom()
    monkeypatch.setattr(subprocess, "run", guarded)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(environment.os, "system", boom, raising=False)
    d = environment_data()
    assert d["checkout"]["git"] is ((Path(d["code_folder"]) / ".git").exists())
    assert git_calls or not d["checkout"]["git"]
    assert "fetch" not in git_calls and "pull" not in git_calls


def test_openblas_cap_state(fake_platform, monkeypatch):
    fake_platform("linux", "x86_64")
    for v in cli.THREAD_VARS:   # the cap stops at any of the three, so a run with OMP_NUM_THREADS set (parallel CI) needs all three out
        monkeypatch.delenv(v, raising=False)
    assert 1 <= environment_data()["threads"]["openblas_threads_in_run"] <= 4
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    t = environment_data()["threads"]
    assert t["openblas_threads_in_run"] == "7" and t["env"]["OPENBLAS_NUM_THREADS"] == "7"
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "many")
    assert environment_data()["threads"]["openblas_threads_in_run"] == "many"   # reported as it is


def test_thread_count_equals_what_the_cap_sets(fake_platform, monkeypatch):
    fake_platform("linux", "x86_64")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    for v in cli.THREAD_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items() if k != "numpy"})
    reported = environment_data()["threads"]["openblas_threads_in_run"]
    cli.cap_blas_threads()
    try:
        assert str(reported) == os.environ["OPENBLAS_NUM_THREADS"] == "2"
    finally:
        os.environ.pop("OPENBLAS_NUM_THREADS", None)


def write(root, name, text):
    f = root / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)


def test_cgroup_v2_limits(tmp_path):
    write(tmp_path, "memory.max", "4294967296\n")
    write(tmp_path, "memory.current", "1073741824\n")
    write(tmp_path, "cpu.max", "200000 100000\n")
    lim = cli.cgroup_limits(tmp_path)    # no memory.stat here: the used memory is memory.current (tests/test_cgroup_working_set.py has the rest)
    assert {k: lim[k] for k in ("memory_bytes", "memory_used_bytes", "cpu_cores", "source")} == {
        "memory_bytes": 4 * 2**30, "memory_used_bytes": 2**30, "cpu_cores": 2.0, "source": "v2"}


def test_cgroup_v2_no_limit(tmp_path):
    write(tmp_path, "memory.max", "max\n")
    write(tmp_path, "cpu.max", "max 100000\n")
    lim = cli.cgroup_limits(tmp_path)
    assert lim["memory_bytes"] is None and lim["cpu_cores"] is None and lim["source"] == "v2"


def test_cgroup_v1_limits(tmp_path):
    write(tmp_path, "memory/memory.limit_in_bytes", "2147483648\n")
    write(tmp_path, "memory/memory.usage_in_bytes", "1000\n")
    write(tmp_path, "cpu/cpu.cfs_quota_us", "150000\n")
    write(tmp_path, "cpu/cpu.cfs_period_us", "100000\n")
    lim = cli.cgroup_limits(tmp_path)
    assert lim["source"] == "v1" and lim["memory_bytes"] == 2 * 2**30 and lim["cpu_cores"] == 1.5
    write(tmp_path, "memory/memory.limit_in_bytes", "9223372036854771712\n")   # v1 "no limit"
    write(tmp_path, "cpu/cpu.cfs_quota_us", "-1\n")
    lim = cli.cgroup_limits(tmp_path)
    assert lim["memory_bytes"] is None and lim["cpu_cores"] is None


def test_no_cgroup_files(tmp_path):
    assert cli.cgroup_limits(tmp_path)["source"] is None


def test_ram_and_cores_use_the_cgroup_limit(tmp_path, monkeypatch):
    write(tmp_path, "memory.max", str(2 * 2**30))
    write(tmp_path, "memory.current", str(2**30))
    write(tmp_path, "cpu.max", "100000 100000")
    for v in cli.THREAD_VARS:   # a thread variable of the caller (parallel CI) changes openblas_threads_in_run
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(cli, "CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(cli, "host_ram_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(environment, "host_ram_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(environment, "_available_ram_bytes", lambda: 50 * 2**30)
    d = environment_data()
    ram = d["ram"]
    assert ram["host_total_bytes"] == 64 * 2**30 and ram["total_bytes"] == 2 * 2**30
    assert ram["available_bytes"] == 2**30 and ram["cgroup_limit_bytes"] == 2 * 2**30
    assert d["cores"]["cgroup_cpu_limit"] == 1.0 and d["cores"]["usable"] == 1
    assert cli.physical_ram_bytes() == 2 * 2**30   # the memory guard sees the limit too
    assert d["threads"]["openblas_threads_in_run"] in (1, os.environ.get("OPENBLAS_NUM_THREADS"))


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               1000.
Pages active:                             5000.
Pages inactive:                           2000.
Pages speculative:                         500.
"""


def test_parse_vm_stat():
    assert environment.parse_vm_stat(VM_STAT) == 3500 * 16384


def test_darwin_free_ram_uses_vm_stat(monkeypatch):
    class NoProc:
        def __init__(self, *a):
            pass

        def read_text(self):
            raise OSError("no /proc")
    monkeypatch.setattr(environment, "Path", NoProc)
    monkeypatch.setattr(sys, "platform", "darwin")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(stdout=VM_STAT)
    monkeypatch.setattr(environment.subprocess, "run", fake_run)
    assert environment._available_ram_bytes() == 3500 * 16384
    assert calls == [["vm_stat"]]


def test_stray_dylib_on_linux_does_not_count(tmp_path, monkeypatch):
    path = fake_shim(monkeypatch, tmp_path, "ok")      # a library that would load
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert cli.backend_availability()["cholesky"]["available"] is False
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert cli.backend_availability()["cholesky"]["available"] is True
    path.unlink()                                      # the file is gone: not available
    assert cli.backend_availability()["cholesky"]["available"] is False


def test_pyamg_offer_names_the_python_and_pin(fake_platform, monkeypatch):
    import importlib.util
    fake_platform("linux", "x86_64")
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "pyamg" else real(n, *a))
    (rec,) = [r for r in environment_data()["recommendations"] if r["action"] == "install_pyamg"]
    assert "--python" in rec["command"] and sys.executable in rec["command"] and "pyamg==" in rec["command"]


def test_problem_facts_voxelize_once(tmp_path, fake_platform, monkeypatch):
    import yaml
    fake_platform("linux", "x86_64")
    p = make_case(tmp_path, h=2.0)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "cholesky"
    p.write_text(yaml.safe_dump(d))
    calls = []
    real = cli.check_data
    monkeypatch.setattr(cli, "check_data", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert environment_data(p)["problem"]["ok"]
    assert len(calls) == 1


def test_tool_asks_the_run_python(tmp_path, monkeypatch):
    wrapper = tmp_path / "otherpy"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TOPOPT_PYTHON", str(wrapper))
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    r = server.environment()
    ri = r["run_interpreter"]
    assert ri["executable"] == str(wrapper) and ri["probe_error"] is None
    assert ri["report_is_for"] == r["python"]["executable"]


def test_tool_reports_a_run_python_that_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TOPOPT_PYTHON", str(tmp_path / "no_such_python"))
    r = server.environment()
    assert r["ok"] and r["run_interpreter"]["probe_error"]
    assert r["run_interpreter"]["report_is_for"] == sys.executable


def test_time_estimate_per_backend(tmp_path, fake_platform):
    fake_platform("linux", "x86_64")
    d = environment_data(make_case(tmp_path, h=2.0))
    p = d["problem"]
    assert p["ok"] and p["free_dofs"] > 0
    est = p["time_estimate_s"]
    assert set(est) == {"cholesky", "pardiso", "amg"}
    # The factor applies to the time per iteration only: setup and post-process time stay.
    fixed = cli.SETUP_S + cli.POST_S
    assert est["amg"] == pytest.approx(fixed + (est["pardiso"] - fixed) * solver_choice.AMG_TIME_FACTOR, rel=0.01)
    assert p["solver"]["resolved"] == "amg"
    rec = [r for r in d["recommendations"] if r["action"] == "install_pypardiso"][0]
    # The case is tiny: setup and post-process time (the same for both) are most of the total.
    assert rec["estimate_s"]["without_pardiso"] >= rec["estimate_s"]["with_pardiso"]


def test_problem_with_solver_that_cannot_run_still_gives_estimates(tmp_path, fake_platform):
    import yaml
    fake_platform("linux", "x86_64")
    p = make_case(tmp_path, h=2.0)
    d = yaml.safe_load(p.read_text())
    d["settings"]["solver"] = "cholesky"
    p.write_text(yaml.safe_dump(d))
    prob = environment_data(p)["problem"]
    assert prob["ok"] and "Accelerate" in prob["solver_setting_problem"]
    assert prob["time_estimate_s"]["amg"] > 0


def test_server_tool_wrong_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    r = server.environment("nope.yaml")
    assert r["ok"] is True and r["problem"]["ok"] is False and "not found" in r["problem"]["error"]
    assert r["topopt_version"] == topopt.__version__


def test_server_tool_with_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    y = make_case(tmp_path, h=2.0)
    r = server.environment(y.name)
    assert r["ok"] and r["problem"]["ok"] and r["problem"]["time_estimate_s"]["amg"] > 0


def test_stdio_call_tool():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], cwd=str(REPO))
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                res = await session.call_tool("environment", {})
                return json.loads(res.content[0].text)

    d = asyncio.run(asyncio.wait_for(go(), 60))
    assert d["ok"] and d["topopt_version"] == topopt.__version__ and "backends" in d


# --- the gpu_solver fact (P9-4d: gpu-mg is the main solver; the offers of the opt-in time are gone) --------------------

OLD_GPU_ACTIONS = {"use_gpu_mg", "install_gpu_extra"}


def no_gpu_offer(d):
    """The recommendations hold no GPU offer: the facts are in `gpu_solver`, and `auto` uses the GPU without an opt-in."""
    assert not [r for r in d["recommendations"] if r["action"] in OLD_GPU_ACTIONS]
    assert "experimental" not in json.dumps(d["recommendations"]) and "opt_in" not in json.dumps(d["recommendations"])


def test_no_device_gives_no_offer_and_the_fact_names_what_is_missing(fake_platform):
    for plat, machine in (("darwin", "arm64"), ("linux", "x86_64")):
        fake_platform(plat, machine, dylib=True)
        d = environment_data()
        no_gpu_offer(d)
        g = d["gpu_solver"]
        assert g["used"] is False and g["device"] is None and g["to_use_the_gpu"].startswith("OpenCL shows no device")
        assert g["why"] and g["test_command"].endswith("-m topopt gpu-check") and sys.executable in g["test_command"]
        assert d["experimental_solvers"] == {}
        json.dumps(d)


def test_a_gpu_that_passes_the_self_test_is_used_and_the_fact_says_so(fake_platform):
    fake_platform("darwin", "arm64", dylib=True, gpu=[make_device(0, "Apple M4", "GPU", platform_name="Apple")], self_test=SELF_TEST_PASSED)
    d = environment_data()
    no_gpu_offer(d)
    g = d["gpu_solver"]
    assert g["used"] is True and g["to_use_the_gpu"] is None and d["auto_solver"] == "gpu-mg"
    assert g["device"]["name"] == "Apple M4" and g["case"] == "C" and g["self_test"]["passed"] is True
    assert "Apple M4" in g["why"] and "gpu-mg" in g["why"]
    assert d["solver_setting"]["auto"]["choice"]["solver"] == "gpu-mg"
    json.dumps(d)


def test_a_gpu_that_fails_the_self_test_is_not_used_and_the_fact_gives_the_numbers(fake_platform):
    fake_platform("linux", "x86_64", pypardiso=types.ModuleType("pypardiso"), gpu=[make_device(0, "Test GPU", "GPU")],
                  self_test={**SELF_TEST_FAILED, "rel_diff": 4.2e-4})
    d = environment_data()
    g = d["gpu_solver"]
    assert g["used"] is False and d["auto_solver"] == "pardiso" and g["to_use_the_gpu"] is None      # a GPU is visible: no step to name
    assert g["self_test"] is None and g["device"] is None                           # the pick is the CPU solver: no device is in use
    assert "Test GPU" in g["why"] and "self-test" in g["why"] and "4.20e-04" in g["why"] and "above the limit 1e-06" in g["why"]


def test_the_gpu_mg_figures_are_the_measured_table(fake_platform):
    fake_platform("darwin", "arm64", dylib=True, gpu=[make_device()])
    g = environment_data()["gpu_solver"]
    assert g["measured"] == environment.GPU_MG_MEASURED
    m = g["measured"]
    assert m["machine"] == "MacBook Air M4, 16 GB" and "median seconds per optimizer iteration" in m["what"]
    assert [r["voxel_mm"] for r in m["rows"]] == [1.2, 1.0, 0.8]
    assert m["finest"]["default_memory_estimate_gib"] > m["finest"]["ram_gib"]       # the default solver does not fit; gpu-mg ran it
    assert not hasattr(environment, "gpu_mg_measured_text")


def test_no_grid_size_threshold_in_the_gpu_fact(fake_platform):
    """The tools give facts: the fact is the same for any problem file."""
    fake_platform("darwin", "arm64", dylib=True, gpu=[make_device()])
    a = environment_data()
    assert a["recommendations"] == [] and "threshold" not in json.dumps(a["gpu_solver"]).lower()


def test_only_a_cpu_device_is_not_used_and_the_fact_names_it(fake_platform):
    fake_platform("linux", "x86_64", gpu=[make_device(0, "pthread-cpu", "CPU", fp64=True, platform_name="Portable Computing Language")],
                  self_test=SELF_TEST_PASSED)
    d = environment_data()
    no_gpu_offer(d)
    g = d["gpu_solver"]
    assert g["used"] is False and d["auto_solver"] != "gpu-mg"                       # a CPU device never counts for auto
    assert g["to_use_the_gpu"].startswith("OpenCL shows no GPU, only: pthread-cpu (CPU)") and "settings.solver: gpu-mg" in g["to_use_the_gpu"]
    fake_platform("darwin", "arm64", dylib=True, gpu=[make_device(0, "cpu", "CPU")], self_test=SELF_TEST_PASSED)
    assert environment_data()["gpu_solver"]["used"] is False


def test_the_fact_names_the_device_that_auto_takes_not_the_first_non_cpu_device(fake_platform):
    """Review m4: `auto` takes the first GPU before an accelerator, in any list order."""
    fpga = make_device(0, "fpga emu", "ACCELERATOR", platform_name="Intel(R) FPGA Emulation Platform")
    arc = make_device(1, "Intel(R) Arc(TM) Graphics", "GPU", platform_name="Intel(R) OpenCL Graphics", vendor="Intel(R) Corporation")
    fake_platform("linux", "x86_64", pypardiso=types.ModuleType("pypardiso"), gpu=[fpga, arc], self_test=SELF_TEST_PASSED)
    d = environment_data()
    g = d["gpu_solver"]
    assert d["opencl"]["auto_device"] == 1 and g["used"] is True and g["device"]["name"] == "Intel(R) Arc(TM) Graphics"
    assert "fpga emu" not in g["why"]


def test_a_cpu_that_auto_takes_is_not_used_even_when_another_device_is_listed(fake_platform):
    """Review m4: devices [CPU, OTHER]: `auto` takes the CPU, and gpu-check says no non-CPU device is visible. The fact agrees."""
    fake_platform("linux", "x86_64", gpu=[make_device(0, "pthread-cpu", "CPU", True, "Portable Computing Language"),
                                          make_device(1, "odd device", "OTHER")], self_test=SELF_TEST_PASSED)
    d = environment_data()
    assert d["opencl"]["auto_device"] == 0 and d["gpu_solver"]["used"] is False
    from topopt import gpu_check
    plan = gpu_check.plan_facts(d, Path(d["code_folder"]), None)
    assert plan["device"] is None and "no OpenCL device that is not a CPU" in plan["blockers"][0]["summary"]


def test_an_accelerator_is_not_a_gpu_for_auto_and_gpu_check_takes_it(fake_platform):
    """`auto` uses a GPU only (`counts_as_gpu`); `gpu-check` takes the best non-CPU device, here the accelerator."""
    fake_platform("linux", "x86_64", gpu=[make_device(0, "pthread-cpu", "CPU", True, "Portable Computing Language"),
                                          make_device(1, "fpga emu", "ACCELERATOR", platform_name="Intel(R) FPGA Emulation Platform")],
                  self_test=SELF_TEST_PASSED)
    d = environment_data()
    assert d["gpu_solver"]["used"] is False and d["gpu_solver"]["to_use_the_gpu"].startswith("OpenCL shows no GPU, only: ")
    from topopt import gpu_check
    assert gpu_check.plan_facts(d, REPO, None)["device"]["name"] == "fpga emu"


def test_pyopencl_missing_gives_the_install_command_in_the_fact(fake_platform):
    fake_platform("darwin", "arm64", dylib=True, gpu="missing")
    d = environment_data()
    no_gpu_offer(d)
    g = d["gpu_solver"]
    cmd = d["opencl"]["install_command"]
    assert g["used"] is False and g["to_use_the_gpu"] == f"pyopencl is not installed. Install it with: {cmd}"
    assert "--python" in cmd and sys.executable in cmd and "[gpu]" in cmd.replace("'", "")
    assert d["solver_setting"]["gpu-mg"]["error"]                    # a run with `settings.solver: gpu-mg` cannot start here
    json.dumps(d)


def test_a_pyopencl_that_does_not_load_is_a_fault_to_fix_first(fake_platform):
    fake_platform("darwin", "arm64", dylib=True, gpu="load_failed")
    d = environment_data()
    no_gpu_offer(d)
    g = d["gpu_solver"]
    assert "load_failed" in d["opencl"]["problem"] and g["used"] is False
    assert g["to_use_the_gpu"].startswith("pyopencl is installed and does not load (") and "Fix that first" in g["to_use_the_gpu"]


def test_checkout_facts_are_in_the_environment_data():
    from topopt import doctor
    d = environment_data()
    assert d["checkout"]["code_folder"] == d["code_folder"]
    assert set(d["checkout"]) == set(doctor.checkout_facts())      # one function, no copy
    json.dumps(d["checkout"])


def test_server_tool_has_the_checkout_and_the_gpu_solver_fact(tmp_path, monkeypatch, fake_platform):
    fake_platform("darwin", "arm64", dylib=True, gpu=[make_device(0, "Apple M4", "GPU", platform_name="Apple")], self_test=SELF_TEST_PASSED)
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path))
    r = server.environment()
    assert r["ok"] and "checkout" in r and r["checkout"]["code_folder"] == r["code_folder"]
    assert r["recommendations"] == [] and r["gpu_solver"]["used"] is True
    from topopt_mcp import replies
    assert "checkout" in replies.guide("environment") and "gpu_solver" in server.environment.__doc__      # #174: the key list moved to the guide
    assert "use_gpu_mg" not in server.environment.__doc__ and "install_gpu_extra" not in server.environment.__doc__




def test_the_opencl_facts_and_the_offers_have_no_host_freeze_keys(fake_platform):
    """The fact about host freezes on an Intel GPU on Linux is not in the product: no key, not even a null."""
    intel = make_device(0, "Intel(R) Arc(TM) Graphics", "GPU", platform_name="Intel(R) OpenCL Graphics", vendor="Intel(R) Corporation")
    fake_platform("linux", "x86_64", pypardiso=types.ModuleType("pypardiso"), gpu=[intel])
    d = environment_data()
    assert "freeze_fact" not in d["opencl"] and "freeze_fact" not in json.dumps(d) and "freeze_options" not in json.dumps(d)
    assert not hasattr(device, "freeze_fact") and not hasattr(device, "freeze_options")
    assert "freez" not in json.dumps(d["recommendations"]).lower()
