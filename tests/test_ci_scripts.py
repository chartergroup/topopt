"""Tests of the CI scripts in `scripts/ci/` (CONTRACTS.md section 27).

These cover the logic of the steps: the choice of legs, the expected-skip check, the golden compare, the
licence step, the wheel check and the check of the workflow file. They start no optimizer run and no network access.
"""
import ast
import copy
import fnmatch
import importlib.util
import json
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / "scripts" / "ci"
sys.path.insert(0, str(CI))

import _common  # noqa: E402
import check_locks  # noqa: E402
import check_skips  # noqa: E402
import check_solver  # noqa: E402
import check_workflow  # noqa: E402
import install_env  # noqa: E402
import make_tiers  # noqa: E402
import license_check  # noqa: E402
import plan_legs  # noqa: E402
import run_pytest  # noqa: E402
import short_run  # noqa: E402
import wheel_check  # noqa: E402

# The tests of this file check the CI scripts (legs, tiers, the golden short run, the wheel check). They are in the selection `pytest -m equivalence`
# because the CI machinery that holds the golden file belongs with the comparisons; they do not compare with a record of the old pipeline.
pytestmark = pytest.mark.equivalence

SPEC = _common.load_spec()
LIVE_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"        # where GitHub reads it (a person moves the file there after the merge)
CI_WORKFLOW = ROOT / "ci" / "github-workflows" / "tests.yml"        # where the branch holds it (inactive)
WORKFLOWS = [LIVE_WORKFLOW, CI_WORKFLOW]
GOLDEN = ROOT / short_run.DEFAULT_GOLDEN


def _labels(include):
    """The labels of the matrix entries, each one once, in order. A leg that runs on two Pythons has two entries (PY-1)."""
    return list(dict.fromkeys(e["label"] for e in include))


def _workflow_path():
    found = [p for p in WORKFLOWS if p.is_file()]
    assert found, "no workflow file at .github/workflows/tests.yml or ci/github-workflows/tests.yml"
    return found[0]


# --- legs.toml -----------------------------------------------------------------------------------

def test_legs_file_has_the_legs_and_the_tiers():
    # V6 has the first two legs; E4 adds the Linux leg with the Intel pins and the manual Intel Mac leg (tests/test_ci_intel_legs.py).
    assert set(SPEC["legs"]) == {"linux-x86_64", "macos-arm64", "linux-x86_64-intel-pins", "macos-x86_64"}
    assert SPEC["legs"]["linux-x86_64"]["runner"] == "ubuntu-24.04"
    assert SPEC["legs"]["macos-arm64"]["runner"] == "macos-15"
    assert set(SPEC["tiers"]) >= {"bit", "cross"}
    for name, tier in SPEC["tiers"].items():
        assert set(tier) == set(short_run.METRICS), name
        for tol in tier.values():
            assert set(tol) == {"rtol", "atol"} and tol["rtol"] >= 0 and tol["atol"] >= 0


def test_no_leg_installs_a_gpu_extra_and_each_leg_names_a_known_tier_and_solver():
    for label, leg in SPEC["legs"].items():
        assert not [e for e in leg["extras"] if e.startswith("gpu")], label
        assert leg["tier"] in SPEC["tiers"], label
        assert leg["expect_solver"] in {"cholesky", "pardiso", "amg"}, label
        assert leg["short_run_limit_s"] > 0


def test_each_leg_sets_the_workers_and_the_thread_cap_of_the_pytest_step():
    """T1: the parallel run is data in legs.toml. A number of workers or `auto`, and a thread cap of 1 or more."""
    for label, leg in SPEC["legs"].items():
        assert run_pytest.workers_arg(leg["pytest_workers"]) != "0", label      # CI runs in parallel
        assert isinstance(leg["pytest_threads"], int) and leg["pytest_threads"] >= 1, label


def test_expected_solver_matches_the_platform_rule():
    # CONTRACTS section 4: macOS gives cholesky, Linux x86_64 with the pardiso extra gives pardiso.
    assert SPEC["legs"]["linux-x86_64"]["expect_solver"] == "pardiso" and "pardiso" in SPEC["legs"]["linux-x86_64"]["extras"]
    assert SPEC["legs"]["macos-arm64"]["expect_solver"] == "cholesky" and SPEC["legs"]["macos-arm64"]["build_shim"]
    assert "pardiso" not in SPEC["legs"]["macos-arm64"]["extras"]


# The repo paths of the test files, as `check_skips.py` sees them: an entry matches them with `fnmatch.fnmatchcase`.
TEST_FILES = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").rglob("*.py"))


def test_every_allowed_skip_entry_is_complete_and_names_a_test_file():
    for label, leg in SPEC["legs"].items():
        for a in leg["allowed_skip"]:
            assert set(a) == {"file", "reason", "why"}, (label, a)
            assert a["file"].startswith("tests/") and a["file"].endswith(".py")
            if "(open)" not in a["why"]:           # a file that an open PR or branch adds may not be on this branch yet
                # a path, or a glob that matches at least one file, with the same match as check_skips.py (`*` also crosses a `/`)
                assert [f for f in TEST_FILES if fnmatch.fnmatchcase(f, a["file"])], (label, a["file"])


def test_the_gmsh_tests_have_no_allowed_skip():
    """A leg must not allow a skip of the tet tests: the gmsh library has to load on both legs. A glob entry must not match them either."""
    tet = ("tests/test_tet.py", "tests/test_tet_worker_core.py")
    for label, leg in SPEC["legs"].items():
        assert not [a for a in leg["allowed_skip"] if any(fnmatch.fnmatchcase(t, a["file"]) for t in tet)], label


# The shared skip markers: (the module in tests/, the name of the marker) -> (the start of the reason text of the marker, the legs where the
# marker skips). A test file that uses a marker skips on those legs, so each of them needs an entry for the file. The CI legs start by hand
# only, so without this check a new user of a marker is found only by a run on a hosted runner (seen on 2026-10-26: test_joints, test_perimeter
# and test_verify_supports had no Accelerate entry).
SHARED_MARKERS = {
    ("helpers", "needs_accelerate"): ("needs Apple Accelerate", [k for k, v in SPEC["legs"].items() if v["expect_solver"] != "cholesky"]),
    ("helpers", "needs_pardiso"): ("needs pypardiso", [k for k, v in SPEC["legs"].items() if v["expect_solver"] != "pardiso"]),
    ("helpers", "needs_direct_solver"): ("needs a direct solver (Apple Cholesky on Apple silicon, or PARDISO",
                                         [k for k, v in SPEC["legs"].items() if v["expect_solver"] not in ("cholesky", "pardiso")]),
    # DIRECT = resolve_solver("auto") != "amg": the marker skips where `auto` gives amg
    ("test_checkpoint", "needs_direct"): ("needs a direct solver (PARDISO or Cholesky)",
                                          [k for k, v in SPEC["legs"].items() if v["expect_solver"] == "amg"]),
}


def _uses_marker(text: str, stem: str, module: str, marker: str) -> bool:
    """True if the test file (its text and its module name `stem`) uses the marker `marker` of `tests/<module>.py`. The file imports it
    (`from <module> import <marker> [as x]`, or `*`) and reads the name in any form (a decorator, `pytestmark`, `marks=`, a list), or it
    reads `<m>.<marker>` after `import <module> [as m]`. The module itself counts when it reads its own marker. The text is parsed with `ast`,
    so a comment, a text, or a local function with the same name and no import does not count."""
    tree = ast.parse(text)
    names = {marker} if stem == module else set()
    modules = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == module and not n.level:
            names |= {marker if a.name == "*" else a.asname or a.name for a in n.names if a.name in (marker, "*")}
        elif isinstance(n, ast.Import):
            modules |= {a.asname or a.name for a in n.names if a.name == module}
    return any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names
               or isinstance(n, ast.Attribute) and n.attr == marker and isinstance(n.value, ast.Name) and n.value.id in modules
               for n in ast.walk(tree))


def _users_of(module: str, marker: str) -> list[str]:
    """The test files of tests/ that use the marker `marker` of `tests/<module>.py` (see `_uses_marker`)."""
    return [f"tests/{p.name}" for p in sorted((ROOT / "tests").glob("test_*.py")) if _uses_marker(p.read_text(), p.stem, module, marker)]


@pytest.mark.parametrize("module, marker", sorted(SHARED_MARKERS))
def test_each_user_of_a_shared_skip_marker_has_an_entry_on_the_legs_where_the_marker_skips(module, marker):
    reason, legs = SHARED_MARKERS[(module, marker)]
    text = (ROOT / "tests" / f"{module}.py").read_text()
    assert f"{marker} = pytest.mark.skipif(" in text and reason in text        # the reason text is the start of the marker's reason
    users = _users_of(module, marker)
    assert users and legs, (marker, users, legs)
    for label in legs:
        allowed = SPEC["legs"][label]["allowed_skip"]
        missing = [f for f in users if not any(fnmatch.fnmatchcase(f, a["file"]) and a["reason"] in reason for a in allowed)]
        assert not missing, f"{label}: add an allowed_skip entry with the reason {reason!r} for {missing}"


@pytest.mark.parametrize("text, user", [
    ("from helpers import needs_pardiso\n@needs_pardiso\ndef test_a(): pass\n", True),
    ('from helpers import needs_pardiso\nP = [pytest.param("p", marks=needs_pardiso)]\n', True),
    ('from helpers import needs_pardiso\nP = [pytest.param("p", marks=[slow,\n                              needs_pardiso])]\n', True),
    ('from helpers import needs_pardiso\nP = [pytest.param("p", marks = (needs_pardiso,))]\n', True),
    ("from helpers import needs_pardiso as pardiso_only\n@pardiso_only\ndef test_a(): pass\n", True),
    ("import helpers\n@helpers.needs_pardiso\ndef test_a(): pass\n", True),
    ("import helpers as h\npytestmark = [pytest.mark.slow, h.needs_pardiso]\n", True),
    ("from helpers import needs_pardiso\npytestmark = needs_pardiso\n", True),
    ("from helpers import needs_pardiso\nMARKS = {'p': needs_pardiso}\n", True),
    ("from helpers import *\n@needs_pardiso\ndef test_a(): pass\n", True),
    ("def needs_pardiso():\n    return pytest.mark.skipif(True, reason='x')\n@needs_pardiso()\ndef test_a(): pass\n", False),
    ("# @needs_pardiso is not used here\ndef test_a(): pass\n", False),
    ('"""Uses @needs_pardiso in the text only."""\nimport helpers\n', False),
    ("from other import needs_pardiso\n@needs_pardiso\ndef test_a(): pass\n", False),
])
def test_the_scan_finds_every_form_of_use_of_a_shared_marker_and_nothing_else(text, user):
    assert _uses_marker(text, "test_x", "helpers", "needs_pardiso") is user


def test_a_module_that_defines_a_marker_and_uses_it_is_a_user():
    text = "needs_direct = pytest.mark.skipif(True, reason='x')\n@needs_direct\ndef test_a(): pass\n"
    assert _uses_marker(text, "test_checkpoint", "test_checkpoint", "needs_direct")
    assert not _uses_marker("needs_direct = pytest.mark.skipif(True, reason='x')\n", "test_checkpoint", "test_checkpoint", "needs_direct")


# The test files that the pyopencl glob entries of legs.toml match but that must run without pyopencl: their devices and the pyopencl
# import are faked, or they need no device. A skip of such a file would hide tests that must run, so the glob must not allow it.
GPU_MG_WITHOUT_OPENCL = {"tests/test_gpu_mg_device.py", "tests/test_gpu_mg_memory.py", "tests/test_gpu_mg_probe.py", "tests/test_gpu_mg_time.py", "tests/test_gpu_mg_x_merge.py"}
PYOPENCL_REASONS = ("could not import 'pyopencl': No module named 'pyopencl'", "no usable OpenCL device: clGetPlatformIDs failed")


def _needs_pyopencl(texts: dict[str, str]) -> set[str]:
    """The test files (keys of `texts`: tests/test_x.py) that skip without pyopencl: they call `importorskip("pyopencl")`, or they import a
    test module that does (as `from test_gpu_mg import ...`). Parsed with `ast`, so a comment or a text does not count."""
    calls, imports = set(), {}
    for f, text in texts.items():
        imports[f] = set()
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) == "importorskip":
                name = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "modname"), None)
                if isinstance(name, ast.Constant) and name.value == "pyopencl":       # pytest.importorskip(...) or importorskip(...)
                    calls.add(f)
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                imports[f].add(f"tests/{n.module}.py")
            elif isinstance(n, ast.Import):
                imports[f] |= {f"tests/{a.name}.py" for a in n.names}
    needs = set(calls)
    while True:
        more = {f for f, imported in imports.items() if f not in needs and imported & needs}
        if not more:
            return needs
        needs |= more


def _test_file_texts() -> dict[str, str]:
    return {f"tests/{p.name}": p.read_text() for p in sorted((ROOT / "tests").glob("test_*.py"))}


def _leg_environment(leg):
    """The marker environment of the platform of a leg, from its runner name (legs.toml names the platform by the runner)."""
    runner = leg["runner"]
    if runner.startswith("ubuntu"):
        return {"platform_system": "Linux", "platform_machine": "x86_64", "sys_platform": "linux"}
    assert runner.startswith("macos"), runner
    machine = "x86_64" if runner.endswith("-intel") or runner == "macos-13" else "arm64"
    return {"platform_system": "Darwin", "platform_machine": machine, "sys_platform": "darwin"}


def _installs_pyopencl(leg):
    """True when the install of the leg puts pyopencl in the venv. pyopencl is a main dependency with a marker (pyproject.toml). The mode `project`
    installs the main dependencies that the marker lets in. The mode `lock-first` installs the pins of the lock file and then the project with
    `--no-deps`: pyopencl is there only when the lock names it (the Intel lock does not)."""
    from packaging.requirements import Requirement
    deps = [Requirement(d) for d in tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]]
    req = next(r for r in deps if r.name == "pyopencl")
    if not req.marker.evaluate(_leg_environment(leg)):
        return False
    if leg.get("install_mode", "project") == "lock-first":
        return re.search(r"^pyopencl==", (ROOT / _common.lock_of(leg)).read_text(), re.M) is not None
    return True


def test_the_legs_that_install_pyopencl_are_the_linux_x86_64_leg_and_the_apple_silicon_leg():
    installs = {label: _installs_pyopencl(leg) for label, leg in SPEC["legs"].items()}
    assert installs == {"linux-x86_64": True, "macos-arm64": True, "linux-x86_64-intel-pins": False, "macos-x86_64": False}


def test_every_test_file_that_needs_pyopencl_is_allowed_to_skip_where_the_leg_cannot_run_it():
    """A test file that skips without pyopencl (or without an OpenCL device) needs an entry. Such a file calls `importorskip("pyopencl")` or imports
    from `test_gpu_mg` (which calls it at import). The entry `could not import 'pyopencl'` is required on a leg where the install gives no pyopencl
    (`_installs_pyopencl`) and must not be there on a leg where it does: such a skip would be a fault, and the entry would hide it. The entry
    `no usable OpenCL device` is required on every leg (no leg has an OpenCL driver that CI installs)."""
    gpu = sorted(_needs_pyopencl(_test_file_texts()))
    assert {"tests/test_gpu_mg.py", "tests/test_continue_gpu_mg.py"} <= set(gpu), gpu
    for label, leg in SPEC["legs"].items():
        for reason in PYOPENCL_REASONS:
            skips = [{"file": f, "name": "t", "reason": reason} for f in gpu]
            unexpected, _ = check_skips.match_skips(skips, leg["allowed_skip"])
            if reason.startswith("could not import") and _installs_pyopencl(leg):
                assert len(unexpected) == len(gpu), (label, "the leg installs pyopencl: no entry may allow the import skip")
            else:
                assert not unexpected, (label, reason, [s["file"] for s in unexpected])
        has_entry = any("could not import 'pyopencl'" in a["reason"] for a in leg["allowed_skip"])
        assert has_entry is (not _installs_pyopencl(leg)), label


def test_the_pyopencl_entries_match_only_the_files_that_need_pyopencl_and_the_listed_files_that_run_without_it():
    """A glob entry for a pyopencl reason must not allow the skip of a file that must run without pyopencl. So each file that such an entry
    matches either needs pyopencl or is in GPU_MG_WITHOUT_OPENCL, and no file of that list needs pyopencl. A new gpu-mg test file fails
    here until it is put in one of the two groups."""
    gpu = _needs_pyopencl(_test_file_texts())
    assert not GPU_MG_WITHOUT_OPENCL & gpu, sorted(GPU_MG_WITHOUT_OPENCL & gpu)
    assert all(f in TEST_FILES for f in GPU_MG_WITHOUT_OPENCL)
    for label, leg in SPEC["legs"].items():
        for a in leg["allowed_skip"]:
            if any(a["reason"] in r for r in PYOPENCL_REASONS):
                matched = {f for f in TEST_FILES if fnmatch.fnmatchcase(f, a["file"])}
                assert matched - gpu <= GPU_MG_WITHOUT_OPENCL, (label, a["file"], sorted(matched - gpu - GPU_MG_WITHOUT_OPENCL))


def test_the_pyopencl_scan_follows_the_imports_and_ignores_text():
    texts = {"tests/test_a.py": 'cl = pytest.importorskip("pyopencl")\n',
             "tests/test_b.py": "from test_a import helper\n",
             "tests/test_c.py": "import test_b\n",
             "tests/test_d.py": '"""Needs no pytest.importorskip("pyopencl")."""\n# pytest.importorskip("pyopencl")\n',
             "tests/test_e.py": 'pytest.importorskip("numba")\nfrom test_d import x\n',
             "tests/test_f.py": 'from pytest import importorskip\ncl = importorskip("pyopencl")\n',
             "tests/test_g.py": 'cl = pytest.importorskip(modname="pyopencl", minversion="2024")\n'}
    assert _needs_pyopencl(texts) == {"tests/test_a.py", "tests/test_b.py", "tests/test_c.py", "tests/test_f.py", "tests/test_g.py"}


def test_unknown_leg_and_tier_are_reported():
    with pytest.raises(_common.CiError, match="unknown leg"):
        _common.get_leg("linux-arm64")
    with pytest.raises(_common.CiError, match="unknown tolerance tier"):
        _common.get_tier("fuzzy")


# --- plan_legs -----------------------------------------------------------------------------------

def test_a_pull_request_runs_the_linux_leg_only_while_macos_is_off_for_pull_requests():
    include, skipped = plan_legs.plan("pull_request", SPEC)
    assert _labels(include) == ["linux-x86_64"] and include[0]["os"] == "ubuntu-24.04"
    assert [label for label, _ in skipped] == ["macos-arm64", "linux-x86_64-intel-pins"]


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_push_and_dispatch_run_every_leg_of_tests_yml(event):
    include, skipped = plan_legs.plan(event, SPEC)
    assert {e["label"] for e in include} == {"linux-x86_64", "macos-arm64", "linux-x86_64-intel-pins"} and not skipped


def test_one_line_in_legs_toml_turns_the_macos_leg_on_for_pull_requests():
    spec = copy.deepcopy(SPEC)
    spec["legs"]["macos-arm64"]["on_pull_request"] = True
    include, skipped = plan_legs.plan("pull_request", spec)
    assert {e["label"] for e in include} == {"linux-x86_64", "macos-arm64"}
    assert [label for label, _ in skipped] == ["linux-x86_64-intel-pins"]


def test_the_choice_linux_runs_exactly_the_leg_linux_x86_64():
    include, skipped = plan_legs.plan("workflow_dispatch", SPEC, legs="linux")
    assert _labels(include) == ["linux-x86_64"] and include[0]["os"] == "ubuntu-24.04"
    assert [label for label, _ in skipped] == ["macos-arm64", "linux-x86_64-intel-pins"]


def test_the_choice_all_runs_every_leg_of_tests_yml():
    include, skipped = plan_legs.plan("workflow_dispatch", SPEC, legs="all")
    assert _labels(include) == ["linux-x86_64", "macos-arm64", "linux-x86_64-intel-pins"] and not skipped


def test_the_choice_does_not_depend_on_the_event():
    for event in ("pull_request", "push", "workflow_dispatch"):
        assert _labels(plan_legs.plan(event, SPEC, legs="linux")[0]) == ["linux-x86_64"], event
        assert len(plan_legs.plan(event, SPEC, legs="all")[0]) == 6, event          # one Python: Linux 2 shards, macOS arm64 2, the Intel pins 2


def test_plan_legs_main_with_legs_linux_writes_the_one_leg_matrix(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    assert plan_legs.main(["--event", "workflow_dispatch", "--legs", "linux"]) == 0
    matrix = json.loads(out.read_text().splitlines()[0][len("matrix="):])["include"]
    assert len(matrix) == 2 and matrix[0] == {"label": "linux-x86_64", "os": "ubuntu-24.04", "python": "3.13", "shard": 1, "shards": 2, "suffix": "-s1of2", "part": ", shard 1/2"}
    assert matrix[-1]["python"] == "3.13" and matrix[-1]["shard"] == 2
    assert "legs=linux" in (tmp_path / "sum.md").read_text() and "legs linux" in capsys.readouterr().out
    out.write_text("")
    assert plan_legs.main(["--event", "workflow_dispatch", "--legs", "all"]) == 0
    assert _labels(json.loads(out.read_text().splitlines()[0][len("matrix="):])["include"]) == ["linux-x86_64", "macos-arm64", "linux-x86_64-intel-pins"]


def test_plan_legs_rejects_an_unknown_choice():
    with pytest.raises(SystemExit):                                   # argparse: invalid choice
        plan_legs.main(["--event", "workflow_dispatch", "--legs", "mac"])
    with pytest.raises(_common.CiError, match="legs must be one of linux, all"):
        plan_legs.plan("workflow_dispatch", SPEC, legs="mac")


def test_matrix_entries_have_the_keys_that_the_workflow_check_expects():
    include, _ = plan_legs.plan("push", SPEC)
    assert all(tuple(e) == plan_legs.MATRIX_KEYS for e in include)


def test_plan_writes_the_matrix_to_the_output_file(tmp_path, monkeypatch):
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    assert plan_legs.main(["--event", "push"]) == 0
    line = out.read_text().splitlines()[0]
    assert line.startswith("matrix=")
    assert json.loads(line[len("matrix="):])["include"][2] == {"label": "macos-arm64", "os": "macos-15", "python": "3.13", "shard": 1, "shards": 2, "suffix": "-s1of2", "part": ", shard 1/2"}
    assert "macos-arm64" in (tmp_path / "sum.md").read_text()


# --- the supported Python versions (PY-1) --------------------------------------------------------

def _pythons_allowed_by_pyproject():
    """The minor versions 3.8 to 3.30 that `requires-python` of pyproject.toml allows (the .0 release of each one)."""
    from packaging.specifiers import SpecifierSet
    spec = SpecifierSet(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"])
    return [f"3.{minor}" for minor in range(8, 31) if spec.contains(f"3.{minor}.0")]


def test_requires_python_allows_exactly_the_supported_pythons_of_the_ci_scripts():
    """One source of truth: SUPPORTED_PYTHONS in scripts/ci/_common.py. A package that claims a Python that the CI checks do not know, or CI that tests a
    Python that the package refuses, fails here."""
    assert _pythons_allowed_by_pyproject() == list(_common.SUPPORTED_PYTHONS) == ["3.13"]


def test_pip_refuses_python_3_12_and_takes_3_13():
    """#469: 3.13 only. pip and uv read `requires-python` of the package metadata and refuse an install on a Python that it does not allow."""
    from packaging.specifiers import SpecifierSet
    spec = SpecifierSet(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"])
    assert not spec.contains("3.12.13") and not spec.contains("3.14.0")
    assert spec.contains("3.13.0") and spec.contains("3.13.14")


def test_every_leg_lists_its_pythons_and_each_one_is_supported():
    for label, leg in SPEC["legs"].items():
        assert set(_common.pythons_of(leg, label)) <= set(_common.SUPPORTED_PYTHONS), label
    assert {label: _common.pythons_of(leg, label) for label, leg in SPEC["legs"].items()} == dict.fromkeys(SPEC["legs"], ("3.13",))   # one Python, the newer one (2026-10-27)


def test_the_lock_check_and_the_wheel_check_take_their_versions_from_the_one_tuple():
    assert check_locks.SUPPORTED_PYTHONS is _common.SUPPORTED_PYTHONS and wheel_check.SUPPORTED_PYTHONS is _common.SUPPORTED_PYTHONS
    assert not hasattr(check_locks, "PYTHON_VERSION")                    # the old single version is gone
    for module in (check_locks, wheel_check):
        text = (CI / f"{module.__name__}.py").read_text()
        assert not re.search(r"""["']3\.1[0-9]["']""", text), module.__name__        # no version literal in the code of the two checks


def test_pythons_of_names_a_missing_key_a_bad_value_a_version_that_is_not_supported_and_a_repeated_version():
    assert _common.pythons_of({"pythons": ["3.13"]}) == ("3.13",)
    for bad in ({}, {"pythons": []}, {"pythons": "3.12"}, {"pythons": [3.12]}):
        with pytest.raises(_common.CiError, match="my-leg: `pythons` in legs.toml must be a list"):
            _common.pythons_of(bad, "my-leg")
    with pytest.raises(_common.CiError, match=r"names 3\.12, which is not in SUPPORTED_PYTHONS \(3\.13\)"):
        _common.pythons_of({"pythons": ["3.13", "3.12"]}, "my-leg")
    with pytest.raises(_common.CiError, match=r"names 3\.11, which is not"):            # a repeated unknown version is named once
        _common.pythons_of({"pythons": ["3.11", "3.11"]}, "my-leg")
    with pytest.raises(_common.CiError, match=r"my-leg: `pythons` names 3\.13 more than once"):
        _common.pythons_of({"pythons": ["3.13", "3.13"]}, "my-leg")              # two equal jobs, two artifacts with one name


def test_the_choice_linux_runs_the_linux_leg_on_python_3_13_only():
    include, _ = plan_legs.plan("workflow_dispatch", SPEC, legs="linux")
    assert [(e["label"], e["python"], e["shard"]) for e in include] == [("linux-x86_64", "3.13", i) for i in (1, 2)]


def test_the_choice_all_adds_macos_arm64_and_the_intel_pins_on_python_3_13():
    include, _ = plan_legs.plan("workflow_dispatch", SPEC, legs="all")
    assert list(dict.fromkeys((e["label"], e["python"]) for e in include)) == [
        ("linux-x86_64", "3.13"), ("macos-arm64", "3.13"), ("linux-x86_64-intel-pins", "3.13")]
    first, after = plan_legs.split_stages(include, SPEC)
    assert {e["python"] for e in first} == {"3.13"} and len(first) == 2 and len(after) == 4      # stage 2 waits for both Linux jobs


def test_one_line_in_legs_toml_adds_a_second_python_to_the_macos_leg(monkeypatch):
    # The package supports 3.13 only (#469); the test adds a second version to SUPPORTED_PYTHONS, as a change that supports a new Python would.
    monkeypatch.setattr(_common, "SUPPORTED_PYTHONS", ("3.13", "3.14"))
    spec = copy.deepcopy(SPEC)
    spec["legs"]["macos-arm64"]["pythons"] = ["3.13", "3.14"]
    include, _ = plan_legs.plan("workflow_dispatch", spec, legs="all")
    assert [(e["label"], e["python"]) for e in include if e["label"] == "macos-arm64"] == [("macos-arm64", "3.13")] * 2 + [("macos-arm64", "3.14")] * 2      # 2 shards
    assert len(include) == 8


def test_plan_refuses_a_leg_without_pythons_or_with_a_python_that_is_not_supported():
    spec = copy.deepcopy(SPEC)
    del spec["legs"]["linux-x86_64"]["pythons"]
    with pytest.raises(_common.CiError, match="linux-x86_64: `pythons` in legs.toml must be a list"):
        plan_legs.plan("workflow_dispatch", spec, legs="all")
    spec = copy.deepcopy(SPEC)
    spec["legs"]["linux-x86_64"]["pythons"] = ["3.12", "3.14"]
    with pytest.raises(_common.CiError, match="not in SUPPORTED_PYTHONS"):
        plan_legs.plan("workflow_dispatch", spec, legs="all")


def test_the_summary_of_the_plan_names_the_python_of_each_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    assert plan_legs.main(["--event", "workflow_dispatch", "--legs", "linux"]) == 0
    text = (tmp_path / "sum.md").read_text()
    assert "| linux-x86_64 | 3.13 shard 1/2 | ubuntu-24.04 | runs |" in text and "| linux-x86_64 | 3.13 shard 2/2 | ubuntu-24.04 | runs |" in text


def test_the_docs_name_the_supported_python_and_say_that_the_launcher_makes_the_python_of_the_legs():
    install = (ROOT / "docs" / "reference" / "install.md").read_text()
    readme = (ROOT / "README.md").read_text()
    assert 'Python 3.13 (`requires-python = ">=3.13,<3.14"`' in install and "3.12 and 3.13" not in install       # 3.13 only (#469)
    assert "python-3.13-blue" in readme                                      # the badge names 3.13 only (tests/test_readme_structure.py)
    assert "Python 3.12 only" not in install + readme and "Python 3.12 only" not in (ROOT / "docs" / "reference" / "tests-and-limits.md").read_text()
    sec = _section_27()
    assert "SUPPORTED_PYTHONS" in sec and "`pythons`" in sec
    launcher_py = re.search(r"^LAUNCHER_PYTHON=(\d+\.\d+)$", (ROOT / "scripts" / "topopt-env.sh").read_text(), re.M).group(1)
    assert all(leg["pythons"] == [launcher_py] for leg in SPEC["legs"].values())      # the launcher makes the one Python that every leg tests (27.2)
    assert re.search(r"The launcher .{0,80}Python " + re.escape(launcher_py), readme + install, re.S)


# --- install_env ---------------------------------------------------------------------------------

def test_extras_that_pyproject_does_not_declare_are_dropped_and_named():
    use, dropped = install_env.split_extras(["dev", "mcp", "pardiso", "tet", "verify"], ["dev", "mcp", "pardiso", "tet"])
    assert use == ["dev", "mcp", "pardiso", "tet"] and dropped == ["verify"]
    assert install_env.declared_extras((ROOT / "pyproject.toml").read_text()) >= ["dev", "mcp", "pardiso", "tet"]


def test_linux_install_commands_give_apt_packages_and_the_locked_editable_install():
    leg = _common.get_leg("linux-x86_64", SPEC)
    cmds = install_env.install_commands(leg, ["dev", "mcp"], "/v/bin/python", system=True, platform="linux")
    assert cmds[0][:3] == ["sudo", "apt-get", "update"]
    assert "libglu1-mesa" in cmds[1] and "libxft2" in cmds[1] and "libxinerama1" in cmds[1]
    assert cmds[-1] == ["uv", "pip", "install", "--python", "/v/bin/python", "-c", "requirements.lock", "-e", ".[dev,mcp]"]
    assert len(cmds) == 3                                              # no shim build on Linux
    no_system = install_env.install_commands(leg, ["dev"], "/v/bin/python", system=False, platform="linux")
    assert len(no_system) == 1


def test_macos_install_builds_the_shim_and_installs_no_system_package():
    leg = _common.get_leg("macos-arm64", SPEC)
    cmds = install_env.install_commands(leg, ["dev", "mcp", "tet"], "/v/bin/python", system=True, platform="darwin")
    assert [c[0] for c in cmds] == ["uv", "sh"] and cmds[1] == ["sh", "topopt/native/build.sh"]


# --- check_solver --------------------------------------------------------------------------------

def test_solver_problem_names_what_was_reported():
    ok = {"solver": {"setting": "auto", "resolved": "pardiso", "cpu_solver": "pardiso", "platform": "linux x86_64", "error": None}}
    assert check_solver.solver_problem(ok, "pardiso") is None
    amg = {"solver": {"setting": "auto", "resolved": "amg", "cpu_solver": "amg", "platform": "linux x86_64", "error": None}}
    msg = check_solver.solver_problem(amg, "pardiso")
    assert "'amg'" in msg and "expects 'pardiso'" in msg and "CPU solver" in msg
    none = {"solver": {"setting": "auto", "resolved": None, "cpu_solver": None, "platform": "macOS arm64", "error": "no shim"}}
    assert "no shim" in check_solver.solver_problem(none, "cholesky")
    assert check_solver.solver_problem({}, "pardiso") is not None


def test_the_step_compares_the_cpu_rule_and_not_the_pick_of_auto():
    """A macOS runner can show an OpenCL GPU: `auto` then runs gpu-mg first. The CPU rule is the same, so the step passes; the pick is information."""
    gpu_pick = {"solver": {"setting": "auto", "resolved": "gpu-mg", "cpu_solver": "cholesky", "platform": "macOS arm64", "error": None,
                           "choice": {"solver": "gpu-mg", "line": "solver: auto -> gpu-mg on Apple M1 (GPU, platform Apple, case C); self-test passed"}}}
    assert check_solver.solver_problem(gpu_pick, "cholesky") is None
    assert "expects 'amg'" in check_solver.solver_problem(gpu_pick, "amg")
    assert check_solver.auto_pick_text(gpu_pick) == "gpu-mg: auto -> gpu-mg on Apple M1 (GPU, platform Apple, case C); self-test passed"
    cpu_pick = {"solver": {"setting": "auto", "resolved": "cholesky", "cpu_solver": "cholesky",
                           "choice": {"line": "solver: auto -> cholesky (no OpenCL GPU: pyopencl is not usable here)"}}}
    assert check_solver.auto_pick_text(cpu_pick) == "cholesky: auto -> cholesky (no OpenCL GPU: pyopencl is not usable here)"
    assert check_solver.auto_pick_text({"solver": {"resolved": "pardiso", "choice": {"line": None}}}) == "pardiso"
    assert check_solver.auto_pick_text({"solver": {"resolved": None}}).startswith("none")
    # a JSON with only the old key `resolved` does not pass: the field of the CPU rule has to be there
    assert check_solver.solver_problem({"solver": {"resolved": "pardiso"}}, "pardiso") is not None


def test_check_json_has_the_cpu_rule_as_its_own_field_and_it_is_never_gpu_mg_for_auto(tmp_path, monkeypatch):
    """`topopt check --json`: `solver.cpu_solver` is the CPU rule for the setting. With a GPU that passes the self-test (faked here), `solver.resolved`
    is gpu-mg and `solver.cpu_solver` stays the CPU solver. An explicit solver gives itself."""
    from topopt import cli, solver_choice
    from topopt.solver_choice import SolverChoice
    yaml_path = ROOT / short_run.DEFAULT_YAML
    cpu = solver_choice.resolve_solver("auto")
    d = cli.check_data(yaml_path, solver_setting="auto")
    assert d["solver"]["cpu_solver"] == cpu and d["solver"]["resolved"] in (cpu, "gpu-mg")
    monkeypatch.setattr(cli, "run_solver", lambda setting, cfg=None: SolverChoice(setting, "gpu-mg", cpu, fallback=cpu, case="C"))
    d = cli.check_data(yaml_path, solver_setting="auto")
    assert (d["solver"]["resolved"], d["solver"]["cpu_solver"]) == ("gpu-mg", cpu)
    monkeypatch.undo()
    assert cli.check_data(yaml_path, solver_setting="amg")["solver"]["cpu_solver"] == "amg"


# --- check_skips ---------------------------------------------------------------------------------

def _junit(tmp_path, cases):
    """A junit XML file. `cases` are (classname, name, skip message or None)."""
    body = []
    for cls, name, msg in cases:
        inner = f'<skipped type="pytest.skip" message="{msg}">x</skipped>' if msg is not None else ""
        body.append(f'<testcase classname="{cls}" name="{name}" time="0.1">{inner}</testcase>')
    f = tmp_path / "junit.xml"
    f.write_text(f'<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="{len(cases)}">{"".join(body)}</testsuite></testsuites>')
    return f


def test_the_test_file_comes_from_the_class_name_even_with_folders_above_the_repo(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_fe.py").write_text("")
    f = check_skips.classname_to_file
    assert f("tests.test_fe", "test_x", tmp_path) == "tests/test_fe.py"
    assert f("tests.test_fe.TestBeam", "test_x", tmp_path) == "tests/test_fe.py"
    assert f("data.workspace.coder.repo.tests.test_fe", "test_x", tmp_path) == "tests/test_fe.py"
    assert f("", "tests/test_gpu_mg.py", tmp_path) == "tests/test_gpu_mg.py"      # a name that is a path
    (tmp_path / "tests" / "test_gpu_mg.py").write_text("")
    assert f("", "tests.test_gpu_mg", tmp_path) == "tests/test_gpu_mg.py"         # a module-level skip: pytest writes the dotted module name


def test_skip_reason_of_a_collection_skip_is_the_text_of_the_skip():
    r = check_skips.skip_reason
    assert r("needs pypardiso", "/x/tests/test_fe.py:8: needs pypardiso") == "needs pypardiso"
    text = "('/x/tests/test_gpu_mg.py', 18, \"Skipped: could not import 'pyopencl': No module named 'pyopencl'\")"
    assert r("collection skipped", text) == "could not import 'pyopencl': No module named 'pyopencl'"
    assert r("collection skipped", "not a tuple") == "collection skipped"


def test_skips_of_a_real_pytest_run_are_read_with_their_file_and_reason(tmp_path):
    """The junit XML of a real pytest run: a skip at import time (`importorskip`) and a plain skip."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mod.py").write_text(
        "import pytest\npytest.importorskip('no_such_module_for_the_ci_test')\n\ndef test_x():\n    pass\n")
    (tmp_path / "tests" / "test_plain.py").write_text(
        "import pytest\n\n@pytest.mark.skipif(True, reason='a plain reason')\ndef test_y():\n    pass\n\ndef test_z():\n    pass\n")
    junit = tmp_path / "junit.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
                    f"--junitxml={junit}", str(tmp_path / "tests")], cwd=tmp_path, check=False, capture_output=True)
    skips = check_skips.skipped_tests(junit, tmp_path)
    got = sorted((s["file"], s["name"], s["reason"]) for s in skips)
    assert got == [("tests/test_mod.py", "tests.test_mod", "could not import 'no_such_module_for_the_ci_test': "
                                                           "No module named 'no_such_module_for_the_ci_test'"),
                   ("tests/test_plain.py", "test_y", "a plain reason")]
    allowed = [{"file": "tests/test_mod.py", "reason": "could not import 'no_such_module_for_the_ci_test'"}]
    unexpected, used = check_skips.match_skips(skips, allowed)
    assert [s["file"] for s in unexpected] == ["tests/test_plain.py"] and used == [1]


def test_skip_check_passes_when_every_skip_is_allowed(tmp_path):
    cases = [("tests.test_fe", "test_a[cholesky]", "needs Apple Accelerate (solver 'cholesky', macOS only); this platform is linux"),
             ("tests.test_fe", "test_b", "needs Apple Accelerate (solver 'cholesky', macOS only); this platform is linux"),
             ("tests.test_cli", "test_ok", None)]
    j = _junit(tmp_path, cases)
    skips = check_skips.skipped_tests(j, ROOT)
    assert [s["file"] for s in skips] == ["tests/test_fe.py"] * 2
    unexpected, used = check_skips.match_skips(skips, _common.get_leg("linux-x86_64", SPEC)["allowed_skip"])
    assert unexpected == [] and sum(used) == 2


def test_skip_check_fails_for_the_gmsh_tests_on_a_leg(tmp_path, monkeypatch, capsys):
    cases = [("tests.test_tet", "test_mesh_box", "needs gmsh (the 'tet' extra): load_failed: libGLU.so.1: cannot open shared object file. Install libGLU")]
    j = _junit(tmp_path, cases)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert _common_main(check_skips, ["--leg", "linux-x86_64", "--junit", str(j)]) == 1
    err = capsys.readouterr().err
    assert "tests/test_tet.py::test_mesh_box" in err and "libGLU.so.1" in err


def test_the_real_gpu_skip_of_conftest_is_allowed_on_macos_arm64_only():
    """With pyopencl and a visible non-CPU device, tests/conftest.py skips the context tests on macOS (the Apple platform of the runner)."""
    conftest = (ROOT / "tests" / "conftest.py").read_text()
    assert 'REAL_GPU_VARIABLE = "TOPOPT_TEST_REAL_GPU"' in conftest
    assert 'f"{problem}; the tests that make an OpenCL context use a real GPU only "' in conftest      # the two parts of the message
    assert 'f"when {REAL_GPU_VARIABLE}=1 (then they run on the GPU, case C)"' in conftest
    reason = ("a non-CPU OpenCL device is visible: Apple M1; the tests that make an OpenCL context use a real GPU only "
              "when TOPOPT_TEST_REAL_GPU=1 (then they run on the GPU, case C)")
    skip = [{"file": "tests/test_gpu_mg_cap.py", "name": "t", "reason": reason}]
    assert check_skips.match_skips(copy.deepcopy(skip), SPEC["legs"]["macos-arm64"]["allowed_skip"])[0] == []
    assert check_skips.match_skips(copy.deepcopy(skip), SPEC["legs"]["linux-x86_64"]["allowed_skip"])[0]     # Linux stops the session instead
    # every test file that puts the marker on a test (pytest.mark.opencl_context) has the skip on macos-arm64
    users = [f for f, text in _test_file_texts().items()
             if any(isinstance(n, ast.Attribute) and n.attr == "opencl_context" for n in ast.walk(ast.parse(text)))]
    assert "tests/test_gpu_mg.py" in users and "tests/test_opencl_guard.py" not in users, users     # the guard test fakes the marker
    for f in users:
        assert check_skips.match_skips([dict(skip[0], file=f)], SPEC["legs"]["macos-arm64"]["allowed_skip"])[0] == [], f


def test_a_glob_entry_allows_every_file_of_its_pattern_and_no_other_file():
    allowed = [{"file": "tests/test_*gpu_mg*.py", "reason": "could not import 'pyopencl'", "why": "a test"},
               {"file": "tests/test_fe.py", "reason": "needs Apple Accelerate", "why": "a test"}]
    def skip(file, reason="could not import 'pyopencl': No module named 'pyopencl'"):
        return {"file": file, "name": "t", "reason": reason}
    ok = [skip("tests/test_gpu_mg.py"), skip("tests/test_gpu_mg_axes.py"), skip("tests/test_bench_gpu_mg.py"),
          skip("tests/test_continue_gpu_mg.py"), skip("tests/test_fe.py", "needs Apple Accelerate (solver 'cholesky')")]
    bad = [skip("tests/test_fem.py"), skip("tests/test_gpu_check.py"), skip("tests/test_gpu_mg_axes.py", "no usable OpenCL device"),
           skip("tests/test_fe.py.bak", "needs Apple Accelerate"), skip("tests/Test_gpu_mg.py")]
    unexpected, used = check_skips.match_skips(ok + bad, allowed)
    assert [s["file"] for s in unexpected] == [s["file"] for s in bad] and used == [4, 1]


def test_an_allowed_file_with_another_reason_is_still_a_failure(tmp_path):
    j = _junit(tmp_path, [("tests.test_fe", "test_a", "some other reason")])
    unexpected, _ = check_skips.match_skips(check_skips.skipped_tests(j, ROOT), _common.get_leg("linux-x86_64", SPEC)["allowed_skip"])
    assert len(unexpected) == 1


def test_a_pardiso_skip_is_allowed_on_macos_and_not_on_linux(tmp_path):
    j = _junit(tmp_path, [("tests.test_fe", "test_pardiso_equals_amg", "needs pypardiso (Linux x86_64 only: install the 'pardiso' extra)")])
    skips = check_skips.skipped_tests(j, ROOT)
    mac, _ = check_skips.match_skips(copy.deepcopy(skips), _common.get_leg("macos-arm64", SPEC)["allowed_skip"])
    lin, _ = check_skips.match_skips(copy.deepcopy(skips), _common.get_leg("linux-x86_64", SPEC)["allowed_skip"])
    assert mac == [] and len(lin) == 1


def test_an_xfail_is_not_a_skip_and_an_idle_allowed_skip_is_printed(tmp_path, monkeypatch, capsys):
    """The junit XML of a real pytest run with one xfail test, one plain skip and one pass."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "import pytest\n\n@pytest.mark.xfail(reason='known bug')\ndef test_bug():\n    assert False\n\n"
        "@pytest.mark.skipif(True, reason='the reason of the skip')\ndef test_skip():\n    pass\n\ndef test_ok():\n    pass\n")
    junit = tmp_path / "junit.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
                    f"--junitxml={junit}", str(tmp_path / "tests")], cwd=tmp_path, check=False, capture_output=True)
    skips = check_skips.skipped_tests(junit, tmp_path)
    assert [(s["file"], s["name"]) for s in skips] == [("tests/test_x.py", "test_skip")]       # the xfail is left out
    assert check_skips.count_xfail(junit) == 1
    spec = {"legs": {"leg": {"allowed_skip": [
        {"file": "tests/test_x.py", "reason": "the reason of the skip", "why": "a test"},
        {"file": "tests/test_y.py", "reason": "never happens", "why": "idle"}]}}}
    monkeypatch.setattr(check_skips, "get_leg", lambda label: spec["legs"][label])
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert check_skips.main(["--leg", "leg", "--junit", str(junit)]) == 0
    out = capsys.readouterr().out
    assert "tests/test_y.py (never happens)" in out and "1 expected failure(s) (xfail)" in out and "OK: 1 skipped" in out


def test_a_missing_junit_file_is_a_failure(tmp_path, capsys):
    assert _common_main(check_skips, ["--leg", "linux-x86_64", "--junit", str(tmp_path / "none.xml")]) == 1
    assert "no junit XML file" in capsys.readouterr().err


def _common_main(module, argv):
    """Run `module.main(argv)` the way the script does: a CiError gives exit code 1."""
    try:
        return module.main(argv)
    except _common.CiError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


# --- the short run and the golden compare --------------------------------------------------------

def _golden():
    return json.loads(GOLDEN.read_text())


def test_the_golden_file_is_text_with_the_four_metrics_and_names_its_source():
    g = _golden()
    assert g["format"] == 1 and g["case"] == short_run.DEFAULT_YAML
    assert set(g["metrics"]) == set(short_run.METRICS)
    assert all(isinstance(v, float) and v > 0 for v in g["metrics"].values())
    for k in ("platform", "cpu", "solver", "threads", "python", "packages", "topopt_base", "date"):
        assert k in g["made_by"], k
    assert g["made_by"]["threads"] == 1


def test_the_golden_inputs_are_the_inputs_of_the_short_run_yaml():
    """If someone edits the YAML or an STL, the golden file needs a new run (make-golden)."""
    assert short_run.input_hashes(ROOT / short_run.DEFAULT_YAML) == _golden()["inputs_sha256"]


def test_input_hashes_find_an_stl_of_any_region_kind(tmp_path, monkeypatch):
    """A region kind that a later version adds (here `solid`) is hashed too, and a missing file is an error."""
    (tmp_path / "a.stl").write_bytes(b"A")
    (tmp_path / "b.STL").write_bytes(b"B")
    y = tmp_path / "p.yaml"
    y.write_text("regions:\n  design: a.stl\n  solid:\n    - name: s\n      file: b.STL\n  load: {tip: a.stl}\nnote: not-an-stl.txt\n")
    monkeypatch.setattr(short_run, "ROOT", tmp_path)
    got = short_run.input_hashes(y)
    assert list(got) == ["p.yaml", "a.stl", "b.STL"]                      # file order, each file once
    y.write_text("regions:\n  design: missing.stl\n")
    with pytest.raises(_common.CiError, match="missing.stl"):
        short_run.input_hashes(y)


def test_the_short_run_yaml_uses_the_general_path_with_few_iterations():
    cfg = yaml.safe_load((ROOT / short_run.DEFAULT_YAML).read_text())
    assert set(cfg) <= {"name", "regions", "load_cases", "material", "settings"}       # no key that only this part has
    assert cfg["settings"]["iterations"] <= 60 and cfg["settings"]["voxel_size"] >= 1.5


def test_the_golden_values_pass_every_tier_against_themselves():
    g = _golden()["metrics"]
    for tier in SPEC["tiers"].values():
        assert all(r["ok"] for r in short_run.compare(g, dict(g), tier))


def test_compare_uses_rtol_and_the_rounding_floor_of_each_metric():
    g = {"compliance_final": 1.0, "volume_fraction_final": 0.2, "volume_final_mm3": 20000.0, "mass_final_g": 25.0}
    tier = SPEC["tiers"]["cross"]
    near = dict(g, compliance_final=1.0 + 5e-9, volume_final_mm3=20000.1, mass_final_g=25.01)
    assert all(r["ok"] for r in short_run.compare(g, near, tier))
    far = dict(g, compliance_final=1.0 + 5e-8)
    rows = short_run.compare(g, far, tier)
    assert [r["metric"] for r in rows if not r["ok"]] == ["compliance_final"]
    assert rows[0]["rel_dev"] == pytest.approx(5e-8)
    assert [r["metric"] for r in short_run.compare(g, dict(g, volume_final_mm3=20000.3), tier) if not r["ok"]] == ["volume_final_mm3"]
    # The bit tier accepts equal numbers only.
    assert not all(r["ok"] for r in short_run.compare(g, near, SPEC["tiers"]["bit"]))


def test_compare_fails_for_a_missing_or_not_finite_number():
    g = {"compliance_final": 1.0, "volume_fraction_final": 0.2, "volume_final_mm3": 20000.0, "mass_final_g": 25.0}
    tier = SPEC["tiers"]["cross"]
    assert [r["ok"] for r in short_run.compare(g, {k: v for k, v in g.items() if k != "mass_final_g"}, tier)].count(False) == 1
    assert not short_run.compare(g, dict(g, compliance_final=float("nan")), tier)[0]["ok"]


def _fake_run(run_dir, **over):
    g = _golden()["metrics"]
    m = {"compliance_final": g["compliance_final"], "volume_final_mm3": g["volume_final_mm3"], "mass_final_g": g["mass_final_g"],
         "iterations": 40, "triangles": 1, "solver": "pardiso", "watertight": True, **over}
    (run_dir / "result").mkdir(parents=True)
    (run_dir / "snapshots").mkdir()
    (run_dir / "snapshots" / "it_0000.bin").write_bytes(b"x")
    (run_dir / "result" / "metrics.json").write_text(json.dumps(m))
    (run_dir / "history.jsonl").write_text(json.dumps({"iteration": 0, "volume_fraction": 0.5}) + "\n" +
                                           json.dumps({"iteration": 40, "volume_fraction": g["volume_fraction_final"] * (1 + over.pop("vf_dev", 0.0))}) + "\n")


def test_read_metrics_takes_the_last_history_row(tmp_path):
    _fake_run(tmp_path / "run")
    metrics, info = short_run.read_metrics(tmp_path / "run")
    assert metrics == _golden()["metrics"] and info["solver"] == "pardiso"


def _run_step(tmp_path, monkeypatch, fake_run_kwargs, wall=12.0, leg="linux-x86_64"):
    def fake_topopt(yaml_path, run_dir, solver):
        assert solver == SPEC["legs"][leg]["expect_solver"]          # the short run names the CPU solver of the leg
        _fake_run(run_dir, **fake_run_kwargs)
        return wall, "1.00 1.00 1.00", "1.00 1.00 1.00"
    monkeypatch.setattr(short_run, "run_topopt", fake_topopt)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    art = tmp_path / "artifacts"
    art.mkdir()
    return _common_main(short_run, ["run", "--leg", leg, "--artifacts", str(art)]), art


@pytest.mark.parametrize("leg", sorted(SPEC["legs"]))
def test_the_short_run_names_the_cpu_solver_of_the_leg_in_the_command(leg, tmp_path):
    """`auto` can pick the GPU solver gpu-mg on a machine with an OpenCL GPU. The golden file is a CPU result, so the command has `--solver <CPU solver>`."""
    solver = SPEC["legs"][leg]["expect_solver"]
    cmd = short_run.run_command(tmp_path / "y.yaml", tmp_path / "out", solver)
    assert cmd[cmd.index("--solver") + 1] == solver and solver in ("cholesky", "pardiso", "amg") and "auto" not in cmd
    assert cmd[1:4] == ["-m", "topopt", "run"] and cmd[-1] == "--quiet"


def test_make_golden_pins_a_cpu_solver_and_never_auto(tmp_path, monkeypatch):
    seen = []

    def fake_topopt(yaml_path, run_dir, solver):
        seen.append(solver)
        raise RuntimeError("stop before the real run")
    monkeypatch.setattr(short_run, "run_topopt", fake_topopt)
    for argv in (["make-golden"], ["make-golden", "--solver", "cholesky"]):
        with pytest.raises(RuntimeError, match="stop before"):
            short_run.main([*argv, "--artifacts", str(tmp_path)])
    assert seen == ["pardiso", "cholesky"]


def test_the_short_run_step_passes_inside_the_tier_and_drops_the_snapshots(tmp_path, monkeypatch):
    code, art = _run_step(tmp_path, monkeypatch, {"compliance_final": _golden()["metrics"]["compliance_final"] * (1 + 3e-9)})
    assert code == 0
    assert (art / "short-run" / "result" / "metrics.json").is_file() and not (art / "short-run" / "snapshots").exists()
    dev = json.loads((art / "short-run-deviation.json").read_text())
    assert dev["tier"] == "cross" and dev["metrics"][0]["rel_dev"] == pytest.approx(3e-9, rel=1e-3)


def test_the_short_run_step_fails_outside_the_tier_and_names_the_metric(tmp_path, monkeypatch, capsys):
    code, art = _run_step(tmp_path, monkeypatch, {"compliance_final": _golden()["metrics"]["compliance_final"] * 1.001})
    assert code == 1 and "outside tier cross: compliance_final" in capsys.readouterr().err
    assert (art / "short-run-deviation.json").is_file()          # the deviation is written also on failure


def test_the_short_run_step_fails_above_the_time_limit(tmp_path, monkeypatch, capsys):
    code, _ = _run_step(tmp_path, monkeypatch, {}, wall=SPEC["legs"]["linux-x86_64"]["short_run_limit_s"] + 1)
    assert code == 1 and "limit of the leg" in capsys.readouterr().err


def test_the_short_run_step_fails_when_the_inputs_differ_from_the_golden_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(short_run, "input_hashes", lambda p: {"examples/hook/design.stl": "0" * 64})
    code, _ = _run_step(tmp_path, monkeypatch, {})
    assert code == 1 and "make-golden" in capsys.readouterr().err


# --- licence step ----------------------------------------------------------------------------------

class FakeMeta(dict):
    def get_all(self, key, default=None):
        return self.get(key, default)


def meta(version="1.0", expr="", classifiers=(), text=""):
    return FakeMeta({"Version": version, "License-Expression": expr, "Classifier": list(classifiers), "License": text})


@pytest.mark.parametrize("expr, expected", [
    ("MIT", "permissive"), ("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", "permissive"),
    ("Apache-2.0 OR LGPL-2.1-or-later", "permissive"), ("BSD-2-Clause AND Apache-2.0 WITH LLVM-exception", "permissive"),
    ("GPL-3.0-or-later", "copyleft"), ("LGPL-2.1-or-later", "copyleft"), ("MIT AND GPL-2.0-only", "copyleft"),
    ("GPL-3.0-or-later WITH GCC-exception-3.1", "copyleft"), ("(MIT OR GPL-3.0-only) AND BSD-3-Clause", "permissive"),
    ("LicenseRef-Intel", "unknown"), ("", "unknown"),
])
def test_spdx_expressions(expr, expected):
    assert license_check.classify_expression(expr) == expected


def test_classifiers_and_text_are_the_fallbacks():
    assert license_check.classify_package(meta(classifiers=["License :: OSI Approved :: MIT License"]))[0] == "permissive"
    assert license_check.classify_package(meta(classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"]))[0] == "copyleft"
    assert license_check.classify_package(meta(classifiers=["License :: Other/Proprietary License"]))[0] == "proprietary"
    assert license_check.classify_package(meta(classifiers=["License :: OSI Approved :: BSD License", "Programming Language :: Python"]))[0] == "permissive"
    # The worst classifier counts.
    assert license_check.classify_package(meta(classifiers=["License :: OSI Approved :: MIT License", "License :: OSI Approved :: GNU General Public License (GPL)"]))[0] == "copyleft"
    assert license_check.classify_package(meta(text="Permission is hereby granted, free of charge"))[0] == "permissive"
    assert license_check.classify_package(meta(text="GNU General Public License version 2"))[0] == "copyleft"
    assert license_check.classify_package(meta())[0] == "unknown"
    assert license_check.classify_package(meta(expr="MIT", classifiers=["License :: Other/Proprietary License"]))[0] == "permissive"


PYPROJECT = textwrap.dedent("""\
    [project]
    name = "x"
    dependencies = ["alpha==1.0", "gamma==3.0"]
    [project.optional-dependencies]
    dev = ["pytest"]
    tet = ["delta==4.0; platform_system == 'Linux'"]
    other = ["beta==2.0"]
    """)

NOTICES = textwrap.dedent("""\
    ## Components whose licences matter

    - **gamma 3.0 (core): GPL-3.0-or-later.** It is a core package.

    ## Core packages

    | Package | Pin | Licence (SPDX) | Checked from |
    |---|---|---|---|
    | alpha | 1.0 | MIT | meta |

    ## Extra: dev

    | Package | Pin | Licence (SPDX) | Checked from |
    |---|---|---|---|
    | pytest | none | MIT | meta |
    """)


def test_declared_packages_follow_the_extras_of_the_leg():
    d = license_check.declared_packages(PYPROJECT, ["dev", "tet"])
    assert d == {"alpha": "core", "gamma": "core", "pytest": "extra dev", "delta": "extra tet"}


def test_notices_names_come_from_table_rows_and_bold_text():
    names = license_check.notices_names(NOTICES)
    assert names == {"gamma", "alpha", "pytest"}


def test_naming_rule_runs_when_the_file_exists_or_license_files_lists_it(tmp_path):
    ran, why = license_check.notices_required(PYPROJECT, tmp_path)
    assert not ran and "does not list it" in why and "#" not in why
    assert license_check.notices_required(PYPROJECT.replace('name = "x"', 'name = "x"\nlicense-files = ["LICENSE", "THIRD_PARTY_NOTICES.md"]'), tmp_path)[0]
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_text("x")
    assert license_check.notices_required(PYPROJECT, tmp_path)[0]


def _rows(declared, packages, policy, named):
    return {r["package"]: r for r in license_check.check_licenses(declared, packages, policy, named)}


def test_a_copyleft_package_needs_a_policy_entry():
    packages = {"alpha": meta(expr="MIT"), "gamma": meta(expr="GPL-3.0-or-later")}
    declared = {"alpha": "core", "gamma": "core"}
    rows = _rows(declared, packages, {}, None)
    assert not rows["alpha"]["problems"] and "no entry" in rows["gamma"]["problems"][0]
    policy = {"allow": [{"package": "Gamma", "decision": "accepted on 2026-10-19"}]}
    rows = _rows(declared, packages, policy, None)
    assert not rows["gamma"]["problems"] and rows["gamma"]["exception"] == "accepted on 2026-10-19"


def test_the_installed_licence_wins_over_the_declaration_and_an_override_corrects_the_metadata():
    packages = {"cholespy": meta(expr="BSD-3-Clause")}
    assert not _rows({"cholespy": "extra verify"}, packages, {}, None)["cholespy"]["problems"]
    policy = {"override": [{"package": "cholespy", "category": "copyleft", "why": "bundled CHOLMOD"}]}
    assert "no entry" in _rows({"cholespy": "extra verify"}, packages, policy, None)["cholespy"]["problems"][0]


def test_a_declared_package_and_an_exception_must_be_named_in_the_notices():
    packages = {"alpha": meta(expr="MIT"), "beta": meta(expr="MIT"), "gamma": meta(expr="GPL-3.0-or-later"),
                "other": meta(expr="MIT"), "intel": meta(classifiers=["License :: Other/Proprietary License"])}
    declared = {"alpha": "core", "beta": "extra other", "gamma": "core"}
    policy = {"allow": [{"package": "gamma", "decision": "d"}, {"package": "intel", "decision": "d"}]}
    rows = _rows(declared, packages, policy, license_check.notices_names(NOTICES))
    assert not rows["alpha"]["problems"] and not rows["gamma"]["problems"]
    assert "does not name it" in rows["beta"]["problems"][0]          # declared, no row
    assert not rows["other"]["problems"]                               # permissive and only needed by another package
    assert "does not name it" in rows["intel"]["problems"][0]          # an exception with no mention


def test_the_naming_rule_does_not_run_without_a_notices_file():
    rows = _rows({"beta": "extra other"}, {"beta": meta(expr="MIT")}, {}, None)
    assert not rows["beta"]["problems"]


def test_the_policy_file_holds_a_decision_for_every_entry_and_names_the_known_exceptions():
    policy = tomllib.loads((CI / "license_policy.toml").read_text())
    names = {e["package"] for e in policy["allow"]}
    assert {"gmsh", "cholespy", "mkl", "intel-openmp"} <= names
    assert all(e["decision"].strip() and e["scope"].strip() for e in policy["allow"])
    assert all(e["why"].strip() and e["category"] in license_check.RANK for e in policy["override"])


def test_the_licence_step_reads_every_installed_distribution_but_the_project():
    pkgs = license_check.installed_packages()
    assert "pytest" in pkgs and "pluggy" in pkgs and "pyyaml" in pkgs        # a package, and a package that it needs
    assert "topopt" not in pkgs                                              # the project is not a dependency of itself
    assert "pytest" not in license_check.installed_packages(exclude=("pytest",))
    assert all(n == license_check.norm(n) for n in pkgs)                     # normalised names, once each


def test_a_package_that_an_extra_requirement_brings_is_checked_too():
    """`mcp` requires `pyjwt[crypto]`, which brings cryptography and cffi. The set is not built from the requirement markers."""
    pkgs = license_check.installed_packages()
    if "pyjwt" not in pkgs:
        pytest.skip("the mcp extra is not installed in this environment")
    assert {"cryptography", "cffi"} <= set(pkgs)


def test_a_package_of_this_environment_has_a_known_category():
    """Every declared core package that is installed classifies as something else than unknown (the real metadata)."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    declared = license_check.declared_packages(pyproject, [])
    pkgs = {n: md for n, md in license_check.installed_packages().items() if n in declared}
    assert pkgs
    unknown = [n for n, md in pkgs.items() if license_check.classify_package(md)[0] == "unknown"]
    assert not unknown, unknown


# --- wheel check ---------------------------------------------------------------------------------

def test_wheel_check_reads_the_data_file_list_of_the_packaging_test():
    files = wheel_check.data_files_of_test(ROOT / "tests" / "test_packaging.py")
    assert "topopt/dashboard/index.html" in files and "topopt/native/build.sh" in files


def test_wheel_problems_name_a_missing_data_file_and_a_missing_licence_file():
    names = ["topopt/a.py", "topopt-0.1.0.dist-info/licenses/LICENSE", "topopt-0.1.0.dist-info/METADATA"]
    assert wheel_check.wheel_problems(names, ["topopt/a.py"], ["LICENSE"]) == []
    assert wheel_check.wheel_problems(names, ["topopt/a.py", "topopt/b.html"], []) == ["the wheel has no topopt/b.html"]
    p = wheel_check.wheel_problems(names, [], ["LICENSE", "THIRD_PARTY_NOTICES.md"])
    assert len(p) == 1 and "THIRD_PARTY_NOTICES.md" in p[0]


def _fake_wheel_environment(tmp_path, monkeypatch):
    """A fake `sh` for wheel_check.main: no uv, no network. `uv build` writes a small wheel with the files that the check wants. Return the list of commands."""
    import zipfile
    dist, cmds = tmp_path / "dist", []
    names = [*wheel_check.data_files_of_test(ROOT / "tests" / "test_packaging.py"),
             "topopt-0.0.0.dist-info/licenses/LICENSE", "topopt-0.0.0.dist-info/licenses/THIRD_PARTY_NOTICES.md"]

    def fake_sh(cmd, *, cwd, env=None, log):
        cmds.append([str(c) for c in cmd])
        out = "1 passed\n"
        if cmd[:2] == ["uv", "build"]:
            dist.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dist / "topopt-0.0.0-py3-none-any.whl", "w") as z:
                for n in names:
                    z.writestr(n, "x")
        elif "-I" in cmd:                                               # the import location: a file inside the venv
            out = str(Path(cmd[0]).parent.parent / "lib" / "topopt" / "__init__.py") + "\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    monkeypatch.setattr(wheel_check, "sh", fake_sh)
    monkeypatch.setattr(wheel_check.shutil, "which", lambda name: "/fake/uv")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    return dist, cmds


def test_the_wheel_check_builds_once_and_installs_the_wheel_in_a_clean_venv_of_each_supported_python(tmp_path, monkeypatch, capsys):
    """PY-1: the build and the contents check run once; the clean venv, the packaging test and the installed commands run for each supported
    Python. The package supports 3.13 only (#469), so the test adds a second version, as a change that supports a new Python would."""
    dist, cmds = _fake_wheel_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(wheel_check, "SUPPORTED_PYTHONS", ("3.13", "3.14"))
    assert wheel_check.main(["--dist", str(dist), "--artifacts", str(tmp_path / "art")]) == 0
    assert len([c for c in cmds if c[:2] == ["uv", "build"]]) == 1
    venvs = [c for c in cmds if c[:2] == ["uv", "venv"]]
    assert [c[c.index("--python") + 1] for c in venvs] == ["3.13", "3.14"]
    assert len({c[-1] for c in venvs}) == 2 and not any(Path(c[-1]).parent.exists() for c in venvs)      # a folder of its own for each Python, removed at the end
    assert len([c for c in cmds if c[:2] == ["uv", "pip"]]) == 2
    summary = (tmp_path / "sum.md").read_text()
    for v in ("3.13", "3.14"):
        assert f"clean venv (Python {v})" in summary and f"packaging test (Python {v})" in summary
    assert "(Python 3.13, 3.14)" in capsys.readouterr().out


def test_the_wheel_check_with_python_runs_that_version_only_and_refuses_an_unsupported_one(tmp_path, monkeypatch):
    dist, cmds = _fake_wheel_environment(tmp_path, monkeypatch)
    assert wheel_check.main(["--dist", str(dist), "--artifacts", str(tmp_path / "art"), "--python", "3.13"]) == 0
    assert [c[c.index("--python") + 1] for c in cmds if c[:2] == ["uv", "venv"]] == ["3.13"]
    for old in ("3.11", "3.12"):
        with pytest.raises(SystemExit):                                       # argparse: a version that is not supported
            wheel_check.main(["--python", old])


def test_a_failing_python_stops_the_wheel_check_and_the_summary_keeps_the_steps_before_it(tmp_path, monkeypatch):
    dist, cmds = _fake_wheel_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(wheel_check, "SUPPORTED_PYTHONS", ("3.13", "3.14"))      # a second version, as in the test above
    ok_sh = wheel_check.sh

    def failing_sh(cmd, *, cwd, env=None, log):
        if cmd[:2] == ["uv", "venv"] and cmd[cmd.index("--python") + 1] == "3.14":
            return subprocess.CompletedProcess(cmd, 1, "", "no python 3.14")
        return ok_sh(cmd, cwd=cwd, env=env, log=log)
    monkeypatch.setattr(wheel_check, "sh", failing_sh)
    with pytest.raises(_common.CiError, match="uv venv failed"):
        wheel_check.main(["--dist", str(dist), "--artifacts", str(tmp_path / "art")])
    summary = (tmp_path / "sum.md").read_text()
    assert "packaging test (Python 3.13)" in summary and "(Python 3.14)" not in summary


def test_collection_errors_are_read_from_the_blocks_of_pytest():
    out = ("tests/test_a.py::test_x\n\n==== ERRORS ====\n"
           "____ ERROR collecting tests/test_mcp.py ____\nImportError while importing test module\nTraceback:\n"
           "topopt_mcp/server.py:32: in <module>\n    from mcp.server import X\nE   ModuleNotFoundError: No module named 'mcp'\n"
           "____ ERROR collecting tests/test_mcp_workspace.py ____\nTraceback:\n"
           "E   ModuleNotFoundError: No module named 'mcp'\n"
           "==== short test summary info ====\nERROR tests/test_mcp.py\n"
           "!!!! Interrupted: 2 errors during collection !!!!\n")
    assert wheel_check.collection_errors(out) == [("tests/test_mcp.py", "ModuleNotFoundError: No module named 'mcp'"),
                                                  ("tests/test_mcp_workspace.py", "ModuleNotFoundError: No module named 'mcp'")]
    assert wheel_check.collection_errors("12 tests collected in 0.5s\n") == []


# --- the workflow file -----------------------------------------------------------------------------

def _wf():
    return yaml.safe_load(_workflow_path().read_text())


def test_the_workflow_file_passes_the_checks():
    problems = check_workflow.check_workflow(_wf(), ROOT, set(SPEC["legs"]))
    assert problems == [], problems


def test_the_workflow_has_the_manual_trigger_the_legs_and_the_wheel_job():
    doc = _wf()
    on = check_workflow._on(doc)
    assert set(on) == {"workflow_dispatch", "workflow_call"}      # workflow_call: kept for a release that runs the tests again
    assert set(doc["jobs"]) == {"plan", "test", "test-after", "wheel", "report"}
    assert doc["jobs"]["test"]["runs-on"] == "${{ matrix.os }}"
    assert doc["jobs"]["wheel"]["runs-on"] == "ubuntu-24.04"


def test_a_push_or_a_pull_request_never_starts_the_workflow():
    """CI runs by hand only. Neither event may come back without a new decision."""
    on = check_workflow._on(_wf())
    assert "push" not in on and "pull_request" not in on and "pull_request_target" not in on and "schedule" not in on


def test_the_workflow_has_one_input_legs_with_the_choices_linux_and_all_and_the_default_linux():
    legs = check_workflow._on(_wf())["workflow_dispatch"]["inputs"]
    assert set(legs) == {"legs", "tier"}
    assert legs["legs"]["type"] == "choice" and legs["legs"]["options"] == ["linux", "all"] and legs["legs"]["default"] == "linux"
    assert tuple(legs["legs"]["options"]) == plan_legs.LEGS_CHOICES        # the script and the workflow offer the same choices


def test_the_plan_job_passes_the_input_legs_to_plan_legs():
    step = [st for st in _wf()["jobs"]["plan"]["steps"] if "plan_legs.py" in st.get("run", "")]
    assert len(step) == 1 and '--legs "${{ inputs.legs }}"' in step[0]["run"]


def test_the_concurrency_group_is_one_for_all_runs_and_never_cancels_a_run():
    c = _wf()["concurrency"]
    assert c["cancel-in-progress"] is False
    assert c["group"] == "ci-tests"      # one run at a time, for tests.yml and intel-mac.yml together


def test_the_header_the_readme_and_contracts_say_how_to_start_the_manual_run():
    start = "gh workflow run tests.yml --repo chartergroup/topopt -f legs=linux"
    head = "\n".join(_workflow_path().read_text().splitlines()[:24])
    assert start in head and "MANUAL ONLY" in head and "Run workflow" in head and "MEASURED" in head
    tests_page = (ROOT / "docs" / "reference" / "tests-and-limits.md").read_text()
    assert start in tests_page and "by hand only" in tests_page
    assert start in _section_27() and "`legs`" in _section_27() and "never starts it" in _section_27()


def test_the_uv_version_is_pinned_in_every_setup_uv_step():
    versions = [st["with"].get("version") for job in _wf()["jobs"].values() for st in job["steps"]
                if str(st.get("uses", "")).startswith("astral-sh/setup-uv@")]
    assert len(versions) == 3 and len(set(versions)) == 1      # test, test-after and wheel and re.fullmatch(r"\d+\.\d+\.\d+", versions[0])


def test_the_workflow_calls_scripts_and_holds_no_step_logic():
    """A `run` step is one command that starts a script (or `uv venv`). No pipes, no `&&`, no loops."""
    for jid, job in _wf()["jobs"].items():
        for st in job["steps"]:
            run = st.get("run")
            if run is None:
                continue
            assert len(run.strip().splitlines()) == 1, (jid, st["name"])
            assert "scripts/ci/" in run or run.startswith("uv venv"), (jid, st["name"])
            assert not re.search(r"\||&&|;|\bfor\b|\bif\b", run), (jid, st["name"], run)


def test_every_test_step_runs_after_a_failed_step_and_the_upload_runs_on_failure():
    steps = _wf()["jobs"]["test"]["steps"]
    by_name = {s["name"]: s for s in steps}
    for s in steps:
        if "scripts/ci/" in s.get("run", "") and "install_env" not in s["run"]:
            assert "!cancelled()" in s["if"], s["name"]
    up = [s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")][0]
    assert up["if"] == "${{ always() }}" and up["with"]["name"] == "ci-${{ matrix.label }}-py${{ matrix.python }}${{ matrix.suffix }}" and up["with"]["retention-days"] == 14
    assert "Pytest (the files of this shard)" in by_name


@pytest.mark.parametrize("edit, text", [
    (lambda d: d["jobs"]["test"]["steps"][0].update({"run": "echo"}), "exactly one of `uses` and `run`"),
    (lambda d: d["jobs"]["test"]["steps"][0].update({"runs": "x"}), "unknown key 'runs'"),
    (lambda d: d["jobs"]["test"]["steps"][0].update({"uses": "actions/checkout"}), "not owner/repo@ref"),
    (lambda d: d["jobs"]["test"]["steps"][3].update({"run": ".venv/bin/python scripts/ci/nope.py"}), "scripts/ci/nope.py does not exist"),
    (lambda d: d["jobs"]["test"]["steps"][3].update({"run": "echo ${{ matrix.nothing }}"}), "matrix.nothing"),
    (lambda d: d["jobs"]["test"]["steps"][4].update({"if": "${{ steps.ghost.outcome == 'success' }}"}), "steps.ghost"),
    (lambda d: d["jobs"]["test"].update({"needs": "ghost"}), "needs 'ghost'"),
    (lambda d: d["jobs"]["test"]["steps"][-1]["with"].pop("retention-days"), "no retention-days"),
    (lambda d: d["jobs"]["test"]["steps"][-1].update({"if": "success()"}), "does not run on failure"),
    (lambda d: d["jobs"]["test"]["steps"][-1]["with"].update({"name": "ci"}), "does not include matrix.label"),
    (lambda d: d["jobs"]["test"].pop("runs-on"), "no `runs-on`"),
    (lambda d: d.pop("jobs"), "no `jobs`"),
    (lambda d: d["jobs"]["test"]["steps"][1].pop("name"), "no `name`"),
])
def test_the_workflow_check_finds_these_mistakes(edit, text):
    doc = copy.deepcopy(_wf())
    edit(doc)
    problems = check_workflow.check_workflow(doc, ROOT, set(SPEC["legs"]))
    assert any(text in p for p in problems), (text, problems)


def test_the_workflow_check_reads_on_as_the_events_even_though_yaml_makes_it_true():
    assert True in yaml.safe_load("on: push") and check_workflow._on(yaml.safe_load("on:\n  push:\n")) == {"push": None}
    doc = _wf()
    del doc[True]
    assert any("`on` must be a mapping" in p for p in check_workflow.check_workflow(doc, ROOT))


def test_a_workflow_copy_at_ci_github_workflows_equals_the_live_file_when_both_exist():
    if LIVE_WORKFLOW.is_file() and CI_WORKFLOW.is_file():
        assert LIVE_WORKFLOW.read_text() == CI_WORKFLOW.read_text()


def test_check_workflow_without_a_path_reads_the_live_file_or_else_the_file_of_the_branch(tmp_path):
    with pytest.raises(_common.CiError, match="no workflow file"):
        check_workflow.default_workflow(tmp_path)
    (tmp_path / "ci" / "github-workflows").mkdir(parents=True)
    (tmp_path / "ci" / "github-workflows" / "tests.yml").write_text("name: x\n")
    assert check_workflow.default_workflow(tmp_path) == tmp_path / "ci" / "github-workflows" / "tests.yml"
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "tests.yml").write_text("name: x\n")
    assert check_workflow.default_workflow(tmp_path) == tmp_path / ".github" / "workflows" / "tests.yml"
    assert check_workflow.default_workflow().is_file()                                   # this repo


# --- CONTRACTS.md section 27 -----------------------------------------------------------------------

def _section_27():
    text = (ROOT / "CONTRACTS.md").read_text()
    assert "## 27. CI, golden values and tolerance tiers" in text
    return text[text.index("## 27. CI, golden values and tolerance tiers"):]


def test_contracts_tier_table_equals_the_tiers_of_legs_toml():
    rows = re.findall(r"^\| (\w+) \| (\w+) \| ([0-9.e+-]+) \| ([0-9.e+-]+) \|$", _section_27(), re.M)
    got = {(t, m): (float(r), float(a)) for t, m, r, a in rows}
    want = {(t, m): (v["rtol"], v["atol"]) for t, tier in SPEC["tiers"].items() for m, v in tier.items()}
    assert got == want


def test_contracts_leg_table_equals_legs_toml():
    sec = _section_27()
    for label, leg in SPEC["legs"].items():
        row = [line for line in sec.splitlines() if line.startswith(f"| `{label}` |")]
        assert len(row) == 1, label
        cells = [c.strip() for c in row[0].strip("|").split("|")]
        assert cells[1] == f"`{leg['runner']}`"
        assert all(f"`{e}`" in cells[2] for e in leg["extras"]), (label, cells[2])
        assert cells[3].startswith(f"`{leg['expect_solver']}`")
        assert cells[4].startswith("yes" if leg["on_pull_request"] else "no")
        assert [p.strip("` ") for p in cells[5].split(",")] == leg["pythons"], (label, cells[5])        # PY-1: the column Pythons


def test_contracts_names_the_files_of_the_scripts_and_the_golden_file():
    sec = _section_27()
    for path in (short_run.DEFAULT_GOLDEN, short_run.DEFAULT_YAML, "scripts/ci/legs.toml", "scripts/ci/license_policy.toml",
                 ".github/workflows/tests.yml", ".github/workflows/intel-mac.yml"):
        assert path in sec, path
    assert "ci/github-workflows/tests.yml" not in sec    # the files moved
    for script in ("install_env", "license_check", "check_solver", "short_run", "run_pytest", "check_skips", "wheel_check"):
        assert (CI / f"{script}.py").is_file() and f"{script}.py" in sec, script
    for apt in SPEC["legs"]["linux-x86_64"]["system_packages"]:
        assert apt in sec, apt


# --- run_pytest.py: the parallel pytest step (T1) -------------------------------------------------

def test_workers_arg_takes_a_number_or_auto_and_nothing_else():
    assert [run_pytest.workers_arg(x) for x in ("auto", "AUTO", "4", 4, "0", 0, " 2 ")] == ["auto", "auto", "4", "4", "0", "0", "2"]
    for bad in ("-1", "many", "", "2.5", None):
        with pytest.raises(_common.CiError, match="workers"):
            run_pytest.workers_arg(bad)


def test_thread_env_sets_only_the_variables_that_the_caller_did_not_set():
    env, now = run_pytest.thread_env(2, {"PATH": "/bin", "OMP_NUM_THREADS": "5"})
    assert env["OMP_NUM_THREADS"] == "5"                                          # the caller's value stays
    assert all(env[v] == "2" for v in run_pytest.THREAD_VARS if v != "OMP_NUM_THREADS")
    assert set(now) == set(run_pytest.THREAD_VARS) - {"OMP_NUM_THREADS"} and env["PATH"] == "/bin"
    assert run_pytest.thread_env(None, {"A": "1"}) == ({"A": "1"}, [])             # no setting: nothing is set
    with pytest.raises(_common.CiError, match="threads"):
        run_pytest.thread_env(0, {})


def test_pytest_command_has_the_workers_and_zero_workers_means_one_process():
    cmd = run_pytest.pytest_command("py", "/r", "/a/junit.xml", "4", ["-k", "x"])
    assert cmd[:3] == ["py", "-m", "pytest"] and cmd[cmd.index("-n") + 1] == "4" and cmd[-2:] == ["-k", "x"]
    assert "--junitxml=/a/junit.xml" in cmd and "--rootdir" in cmd
    assert "-n" not in run_pytest.pytest_command("py", "/r", "/a/junit.xml", "0")


def _tiny_suite(tmp_path):
    """One test file for the nested run. It shows how it was started: the worker, and the thread variables."""
    f = tmp_path / "suite" / "test_tiny.py"
    f.parent.mkdir()
    f.write_text("import json, os, pytest\nfrom pathlib import Path\n"
                 "def test_how_it_runs():\n"
                 "    keys = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'PYTEST_XDIST_WORKER')\n"
                 "    (Path(__file__).parent / 'seen.json').write_text(json.dumps({k: os.environ.get(k) for k in keys}))\n"
                 "@pytest.mark.skip(reason='a plain reason')\n"
                 "def test_skipped():\n    pass\n")
    return f


@pytest.mark.parametrize("workers, threads, caller_omp, workers_seen", [
    ("2", "1", None, True), ("0", "3", None, False), ("2", "1", "6", True), ("3", None, None, True)])
def test_run_pytest_starts_a_real_pytest_with_the_workers_and_the_caps(tmp_path, monkeypatch, workers, threads, caller_omp, workers_seen):
    if importlib.util.find_spec("xdist") is None:
        pytest.skip("pytest-xdist is not installed (extra dev)")
    for v in (*run_pytest.THREAD_VARS, "PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID"):
        monkeypatch.delenv(v, raising=False)
    if caller_omp:
        monkeypatch.setenv("OMP_NUM_THREADS", caller_omp)
    suite, out = _tiny_suite(tmp_path), tmp_path / "artifacts"
    argv = ["--leg", "linux-x86_64", "--artifacts", str(out), "--workers", workers]
    argv += ["--threads", threads] if threads else []
    code = run_pytest.main(argv + ["--", str(suite)])
    assert code == 0
    log = (out / "pytest.log").read_text()
    assert f"workers {workers}" in log and ("-n " + workers in log) == (workers != "0")
    counts = run_pytest.junit_counts(out / "junit.xml")
    assert (counts["passed"], counts["skipped"], counts["failures"]) == (1, 1, 0)
    seen = json.loads((suite.parent / "seen.json").read_text())
    threads = threads or str(SPEC["legs"]["linux-x86_64"]["pytest_threads"])   # no --threads: the value of the leg
    assert seen["OMP_NUM_THREADS"] == (caller_omp or threads) and seen["NUMBA_NUM_THREADS"] == threads, seen
    assert (seen["PYTEST_XDIST_WORKER"] is not None) == workers_seen


def test_run_pytest_runs_in_one_process_and_says_so_when_xdist_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_pytest, "xdist_available", lambda: False)
    suite = tmp_path / "test_one.py"
    suite.write_text("def test_one():\n    pass\n")
    code = run_pytest.main(["--leg", "linux-x86_64", "--artifacts", str(tmp_path / "a"), "--workers", "4", "--threads", "1",
                            "--", str(suite)])
    out = capsys.readouterr().out
    assert code == 0 and "pytest-xdist is not installed" in out and "workers 0" in out and " -n " not in out


def test_release_workflow_is_manual_and_runs_no_tests():
    doc = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text())
    assert set(check_workflow._on(doc)) == {"workflow_dispatch"}
    assert set(doc["jobs"]) == {"release", "attach"}     # attach: the built files of an existing tag (#206, RELEASING.md)
    assert all("needs" not in j and "tests.yml" not in str(j) for j in doc["jobs"].values())
    assert check_workflow.check_workflow(doc, ROOT, set(SPEC["legs"]), name="release.yml") == []


def test_tests_workflow_accepts_the_input_legs_when_called():
    call = check_workflow._on(_wf())["workflow_call"]["inputs"]["legs"]
    assert call["type"] == "string" and call["default"] == "linux"


# --- test tiers (CONTRACTS.md 27.3) ------------------------------------------------------------------

def test_thread_variables_include_the_accelerate_cap():
    assert "VECLIB_MAXIMUM_THREADS" in run_pytest.THREAD_VARS


def test_every_leg_names_a_valid_fast_selection_and_the_expression_uses_known_markers():
    known = {"slow", "platform_sensitive", "not", "and", "or"}
    for label, leg in SPEC["legs"].items():
        for key in ("pytest_fast", "pytest_full"):
            value = str(leg.get(key, "")).strip()
            assert value == run_pytest.NO_TESTS or set(re.findall(r"[a-z_]+", value)) <= known, (label, key, value)
        assert run_pytest.tier_selection(leg, "fast") is None or isinstance(run_pytest.tier_selection(leg, "fast"), str)


def test_tier_selection_fast_full_and_none():
    assert run_pytest.tier_selection({}, "fast") == "not slow" and run_pytest.tier_selection({}, "full") == ""
    assert run_pytest.tier_selection({"pytest_fast": "none"}, "fast") is None
    assert run_pytest.tier_selection({"pytest_full": "not slow"}, "full") == "not slow"
    with pytest.raises(_common.CiError, match="tier"):
        run_pytest.tier_selection({}, "medium")


def test_the_legs_choose_the_tiers_of_the_plan():
    legs = SPEC["legs"]
    assert run_pytest.tier_selection(legs["linux-x86_64"], "fast") == "not slow" and run_pytest.tier_selection(legs["linux-x86_64"], "full") == ""
    assert run_pytest.tier_selection(legs["macos-arm64"], "fast") == "platform_sensitive and not slow"
    assert run_pytest.tier_selection(legs["linux-x86_64-intel-pins"], "fast") is None
    assert run_pytest.tier_selection(legs["linux-x86_64-intel-pins"], "full") == "not slow"


def test_run_pytest_with_a_leg_that_runs_no_test_in_the_fast_tier_writes_an_empty_junit(tmp_path, monkeypatch):
    monkeypatch.setitem(SPEC["legs"]["linux-x86_64"], "pytest_fast", "none")
    monkeypatch.setattr(run_pytest, "get_leg", lambda label: SPEC["legs"][label])
    out = tmp_path / "artifacts"
    assert run_pytest.main(["--leg", "linux-x86_64", "--artifacts", str(out), "--tier", "fast"]) == 0
    assert run_pytest.junit_counts(out / "junit.xml")["tests"] == 0 and "no pytest" in (out / "pytest.log").read_text()


def test_run_pytest_fast_tier_passes_the_marker_expression(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(run_pytest, "get_leg", lambda label: {"pytest_workers": "0", "pytest_fast": "not slow"})
    monkeypatch.setattr(run_pytest.subprocess, "Popen", lambda cmd, **kw: seen.setdefault("cmd", cmd) and (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        run_pytest.main(["--leg", "x", "--artifacts", str(tmp_path), "--tier", "fast"])
    assert seen["cmd"][seen["cmd"].index("-m", 3) + 1] == "not slow"


def _junit_file(path, cases):
    body = "".join(f'<testcase classname="{c}" name="{n}" time="{t}">{extra}</testcase>' for c, n, t, extra in cases)
    path.write_text(f'<testsuites><testsuite tests="{len(cases)}">{body}</testsuite></testsuites>')
    return path


def test_make_tiers_reads_junit_and_lists_the_slow_tests_and_the_platform_files(tmp_path):
    mac = make_tiers.junit_results(_junit_file(tmp_path / "mac.xml", [
        ("tests.test_a", "test_fast", 0.2, ""), ("tests.test_a", "test_slow[p-1]", 6.0, ""), ("tests.test_b", "test_same", 1, ""),
        ("tests.test_c", "test_differs", 1, "<failure message='x'/>")]))
    lin = make_tiers.junit_results(_junit_file(tmp_path / "lin.xml", [
        ("tests.test_a", "test_fast", 0.1, ""), ("tests.test_a", "test_slow[p-1]", 7.0, ""), ("tests.test_b", "test_same", 9, ""),
        ("tests.test_c", "test_differs", 1, "")]))
    assert mac["tests/test_c.py::test_differs"] == ("fail", 1.0) and lin["tests/test_c.py::test_differs"][0] == "pass"
    assert make_tiers.slow_ids([mac, lin]) == ["tests/test_a.py::test_slow[p-1]", "tests/test_b.py::test_same"]      # the longest time counts
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    for name, text in (("test_a", "x = 1"), ("test_b", "import sys\nsys.platform"), ("test_c", "y = 2")):
        (root / "tests" / f"{name}.py").write_text(text)
    assert make_tiers.platform_files(root, mac, lin) == ["tests/test_b.py", "tests/test_c.py"]     # b names the platform, c has another outcome
    (root / "tests" / "test_b.py").write_text(make_tiers.TIER_SKIP_LINE + "\nimport sys\nsys.platform")
    assert make_tiers.platform_files(root, mac, lin) == ["tests/test_c.py"]      # a file with the skip line is in no list
    assert make_tiers.skipped(root, "tests/test_b.py") and not make_tiers.skipped(root, "tests/test_c.py") and not make_tiers.skipped(root, "tests/none.py")


def test_the_tier_lists_name_files_that_exist_and_the_slow_list_is_not_empty():
    assert make_tiers.check(ROOT) == []
    listed = {i.split("::")[0] for i in make_tiers.read_list(make_tiers.SLOW_FILE) + make_tiers.read_list(make_tiers.PLATFORM_FILE)}
    listed |= set(json.loads(make_tiers.DURATIONS_FILE.read_text(encoding="utf-8"))["files"])
    assert not [f for f in listed if make_tiers.skipped(ROOT, f)]           # a file with the skip line is in no list
    assert len(make_tiers.read_list(make_tiers.SLOW_FILE)) > 100 and len(make_tiers.read_list(make_tiers.PLATFORM_FILE)) > 20


def test_conftest_adds_the_markers_from_the_tier_lists():
    """Run pytest on one listed slow test and one listed platform file without running them: --collect-only shows the markers."""
    slow = make_tiers.read_list(make_tiers.SLOW_FILE)[0]
    res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", "slow", slow],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0 and slow.split("::")[1] in res.stdout, res.stdout[-800:]
    plat = make_tiers.read_list(make_tiers.PLATFORM_FILE)[0]
    res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", "platform_sensitive", plat],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0 and plat in res.stdout, res.stdout[-800:]


# --- the legs run in order: Linux first (stage 1), the rest only when it has passed ----------------------

def test_split_stages_puts_the_linux_leg_first_and_macos_and_the_pins_after():
    include, _ = plan_legs.plan("workflow_dispatch", SPEC, legs="all")
    first, after = plan_legs.split_stages(include, SPEC)
    assert _labels(first) == ["linux-x86_64"]
    assert _labels(after) == ["macos-arm64", "linux-x86_64-intel-pins"]
    assert plan_legs.split_stages(plan_legs.plan("workflow_dispatch", SPEC, legs="linux")[0], SPEC)[1] == []


def test_plan_legs_main_writes_first_after_and_has_after(tmp_path, monkeypatch):
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    assert plan_legs.main(["--event", "workflow_dispatch", "--legs", "all"]) == 0
    got = dict(ln.split("=", 1) for ln in out.read_text().splitlines())
    assert _labels(json.loads(got["first"])["include"]) == ["linux-x86_64"] and got["has_after"] == "true"
    assert _labels(json.loads(got["after"])["include"]) == ["macos-arm64", "linux-x86_64-intel-pins"]
    out.write_text("")
    assert plan_legs.main(["--event", "workflow_dispatch", "--legs", "linux"]) == 0
    assert dict(ln.split("=", 1) for ln in out.read_text().splitlines())["has_after"] == "false"


def test_the_job_test_after_has_the_steps_of_the_job_test_and_waits_for_it():
    jobs = _wf()["jobs"]
    assert set(jobs) == {"plan", "test", "test-after", "wheel", "report"}
    assert jobs["test-after"]["steps"] == jobs["test"]["steps"]
    assert jobs["test-after"]["needs"] == ["plan", "test"] and "has_after" in jobs["test-after"]["if"]
    assert jobs["test"]["strategy"]["matrix"].endswith("outputs.first) }}") and jobs["test-after"]["strategy"]["matrix"].endswith("outputs.after) }}")
    assert jobs["wheel"]["needs"] == "test"          # no wheel job when the Linux leg fails
    assert {k: v for k, v in jobs["test-after"].items() if k not in ("needs", "if", "strategy")} == {k: v for k, v in jobs["test"].items() if k not in ("needs", "strategy")}
