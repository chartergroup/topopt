"""The CI legs of E4: the Linux leg with the Intel pins and the manual Intel Mac workflow (CONTRACTS.md 27.10).

These cover the keys `lock`, `workflow` and `check_locks` of `scripts/ci/legs.toml`, the lock choice of `install_env.py`, the per-workflow matrix of
`plan_legs.py`, the per-file rule of `check_workflow.py`, and the two workflow files. They start no optimizer run and no network access
(`tests/test_locks.py` has the lock checks).
"""
import copy
import fnmatch
import json
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / "scripts" / "ci"
sys.path.insert(0, str(CI))

import _common  # noqa: E402
import check_locks  # noqa: E402
import check_workflow  # noqa: E402
import install_env  # noqa: E402
import plan_legs  # noqa: E402

SPEC = _common.load_spec()
LEGS = SPEC["legs"]
INTEL_LOCK = "requirements-macos-x86_64.lock"
WORKFLOW_FILES = {"tests": "tests.yml", "intel-mac": "intel-mac.yml"}
MOVE_BOTH = "mkdir -p .github/workflows && git mv ci/github-workflows/tests.yml ci/github-workflows/intel-mac.yml .github/workflows/"
MOVE_INTEL = "mkdir -p .github/workflows && git mv ci/github-workflows/intel-mac.yml .github/workflows/intel-mac.yml"
PREPARE = "git switch main && git pull --ff-only && git status --short"


def _path(name: str) -> Path:
    """The workflow file: the live path if a person has moved it there, else the path of the branch."""
    found = [p for p in (ROOT / ".github" / "workflows" / name, ROOT / "ci" / "github-workflows" / name) if p.is_file()]
    assert found, f"no workflow file {name}"
    return found[0]


def _load(name: str) -> dict:
    return yaml.safe_load(_path(name).read_text())


# --- legs.toml: the new keys -------------------------------------------------------------------

def test_the_keys_lock_workflow_and_check_locks_are_known_values():
    for label, leg in LEGS.items():
        assert (ROOT / _common.lock_of(leg)).is_file(), label
        assert _common.workflow_of(leg) in WORKFLOW_FILES, label
        assert isinstance(leg.get("check_locks", False), bool), label
    assert _common.lock_of(LEGS["linux-x86_64"]) == _common.DEFAULT_LOCK == "requirements.lock"
    assert {l: leg.get("install_mode", "project") for l, leg in LEGS.items()} == {
        "linux-x86_64": "project", "macos-arm64": "project", "linux-x86_64-intel-pins": "lock-first", "macos-x86_64": "lock-first"}
    assert _common.workflow_of(LEGS["linux-x86_64"]) == "tests"


def test_each_workflow_file_has_a_leg_and_each_leg_has_a_workflow_file():
    for workflow, name in WORKFLOW_FILES.items():
        assert [label for label, leg in LEGS.items() if _common.workflow_of(leg) == workflow], workflow
        _path(name)


def test_the_two_legs_with_the_intel_pins_use_the_intel_lock_and_the_others_use_the_main_lock():
    intel = {label for label, leg in LEGS.items() if _common.lock_of(leg) == INTEL_LOCK}
    assert intel == {"linux-x86_64-intel-pins", "macos-x86_64"}
    assert INTEL_LOCK == check_locks.INTEL_LOCK
    assert {_common.lock_of(LEGS[l]) for l in set(LEGS) - intel} == {"requirements.lock"}


def test_the_lock_check_runs_on_one_leg_only():
    assert [label for label, leg in LEGS.items() if leg.get("check_locks")] == ["linux-x86_64"]


def test_the_linux_leg_with_the_intel_pins_differs_from_the_linux_leg_only_in_the_lock_and_the_runs():
    """It is the Linux leg with other pins: the same extras, solver, packages, tier, limits and allowed skips. The install mode is part of the lock choice.
    One allowed skip differs: the Intel lock has no pyopencl and the lock-first install adds none, so this leg may skip a gpu-mg test file at import
    (`could not import 'pyopencl'`); the Linux leg installs pyopencl and may not."""
    a, b = copy.deepcopy(LEGS["linux-x86_64"]), copy.deepcopy(LEGS["linux-x86_64-intel-pins"])
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differing == {"lock", "install_mode", "on_pull_request", "check_locks", "pytest_fast", "pytest_full", "stage", "allowed_skip"}, differing
    assert b["on_pull_request"] is False and "check_locks" not in b
    assert [x for x in a["allowed_skip"] if x not in b["allowed_skip"]] == []
    assert [x["reason"] for x in b["allowed_skip"] if x not in a["allowed_skip"]] == ["could not import 'pyopencl'"]


def test_the_intel_mac_leg_expects_amg_builds_no_shim_and_has_no_pardiso():
    leg = LEGS["macos-x86_64"]
    assert (leg["runner"], leg["workflow"], leg["expect_solver"], leg["build_shim"]) == ("macos-15-intel", "intel-mac", "amg", False)
    assert leg["tier"] in SPEC["tiers"] and leg["short_run_limit_s"] > 0
    assert not leg["system_packages"]


def test_the_pardiso_extra_is_on_the_linux_legs_only():
    """CONTRIBUTING.md: MKL solvers are allowed only as an optional extra on Linux x86_64."""
    for label, leg in LEGS.items():
        assert ("pardiso" in leg["extras"]) == leg["runner"].startswith("ubuntu"), label


def test_the_extras_of_every_leg_are_declared_in_pyproject():
    declared = install_env.declared_extras((ROOT / "pyproject.toml").read_text())
    for label, leg in LEGS.items():
        assert not install_env.split_extras(leg["extras"], declared)[1], label


def test_the_intel_mac_leg_has_the_skips_of_a_mac_without_accelerate_and_pardiso():
    reasons = {(a["file"], a["reason"]) for a in LEGS["macos-x86_64"]["allowed_skip"]}
    assert ("tests/test_fe.py", "needs Apple Accelerate") in reasons and ("tests/test_fe.py", "needs pypardiso") in reasons
    assert ("tests/test_solver_choice.py", "needs pypardiso") in reasons
    for a in LEGS["macos-x86_64"]["allowed_skip"]:
        assert set(a) == {"file", "reason", "why"} and sorted(ROOT.glob(a["file"])), a      # a path, or a glob that matches a file


# The skips that a platform gives and that the legs must name (E4 review M1). Each pair is (test file, reason substring). The reason is a substring of
# the skip marker or of the `pytest.skip` call in the test file, or of the shared marker in tests/helpers.py. A skip that no entry names fails the leg
# after a paid run, so every pair below is checked here. The first group is the review list; the second group is the same kind of skip (a solver
# that the platform lacks) that a read of the skip markers found in the other files.
_NO_ACCELERATE = ("tests/test_springs.py", "needs Apple Accelerate")
_NO_PARDISO = [("tests/test_springs.py", "needs pypardiso"), ("tests/test_verify_p2.py", "needs pypardiso")]
_NO_PROC = ("tests/test_runinfo.py", "needs /proc (Linux)")
_NO_PROC_DL1A = [("tests/test_hostinfo.py", "needs /proc (Linux)"), ("tests/test_job_facts.py", "needs /proc (Linux)")]   # DL1a review M2
PREDICTED_SKIPS = {
    "linux-x86_64": [_NO_ACCELERATE],
    "linux-x86_64-intel-pins": [_NO_ACCELERATE],
    "macos-arm64": [*_NO_PARDISO, _NO_PROC, *_NO_PROC_DL1A],
    "macos-x86_64": [
        _NO_ACCELERATE, *_NO_PARDISO, _NO_PROC, *_NO_PROC_DL1A,
        ("tests/test_checkpoint.py", "needs a direct solver (PARDISO or Cholesky)"),
        ("tests/test_load_types.py", "this host has no direct solver"),
        ("tests/test_buckling.py", "needs a direct solver (cholesky on macOS arm64, or pardiso on Linux x86_64)"),
        ("tests/test_buckling.py", "the solver of this machine is not direct"),
        ("tests/test_solid_filter.py", "bit-equal runs need a direct solver"),
        ("tests/test_multirow_mma.py", "the values come from a direct solver"),
    ],
}


@pytest.mark.parametrize("label", sorted(PREDICTED_SKIPS))
def test_each_leg_names_the_skips_that_its_platform_gives(label):
    have = [(a["file"], a["reason"]) for a in LEGS[label]["allowed_skip"]]
    helpers = (ROOT / "tests" / "helpers.py").read_text()
    for file, reason in PREDICTED_SKIPS[label]:
        assert (file, reason) in have, f"{label}: no allowed_skip entry for {file} ({reason!r})"
        assert reason in (ROOT / file).read_text() or reason in helpers, f"{file}: no skip marker or skip call has the text {reason!r}"
    assert len(have) == len(set(have)), f"{label}: an allowed_skip entry is listed twice"


def test_the_reason_of_a_predicted_skip_matches_the_message_that_check_skips_reads():
    """`check_skips.match_skips` takes an entry when the file is equal and the reason is a substring of the skip message."""
    import check_skips
    skips = [{"file": "tests/test_springs.py", "name": "t", "reason": "needs Apple Accelerate (solver 'cholesky', Apple silicon only); this platform is macOS x86_64"},
             {"file": "tests/test_checkpoint.py", "name": "t", "reason": "needs a direct solver (PARDISO or Cholesky): AMG is not reproducible"}]
    unexpected, used = check_skips.match_skips(skips, LEGS["macos-x86_64"]["allowed_skip"])
    assert not unexpected and sum(used) == 2
    unexpected, _ = check_skips.match_skips(skips[:1], LEGS["macos-arm64"]["allowed_skip"])
    assert unexpected                                              # the same skip is not allowed on Apple silicon: Accelerate exists there


# --- install_env.py: the lock key ------------------------------------------------------------------

def test_the_install_command_uses_the_lock_of_the_leg():
    for label in ("linux-x86_64", "macos-arm64"):                       # mode project: the lock is a constraint
        leg = _common.get_leg(label, SPEC)
        cmds = install_env.install_commands(leg, ["dev", "mcp"], "/v/bin/python", system=False, platform="linux")
        assert cmds == [["uv", "pip", "install", "--python", "/v/bin/python", "-c", "requirements.lock", "-e", ".[dev,mcp]"]], label


NO_BUILD = ["--only-binary", "numba", "--only-binary", "llvmlite", "--only-binary", "cryptography"]


def test_the_legs_with_the_intel_pins_install_close_to_the_launcher_on_an_x86_64_mac():
    """`-r lock` with no source build for the three packages of the launcher, then the project with no deps, then the extras with the lock as constraint."""
    reqs = install_env.extra_requirements((ROOT / "pyproject.toml").read_text(), ["dev", "mcp", "pardiso"])
    for label in ("linux-x86_64-intel-pins", "macos-x86_64"):
        leg = _common.get_leg(label, SPEC)
        cmds = install_env.install_commands(leg, ["dev", "mcp", "pardiso"], "/v/bin/python", system=False, platform="linux", extra_reqs=reqs)
        uv = ["uv", "pip", "install", "--python", "/v/bin/python"]
        assert cmds == [[*uv, *NO_BUILD, "-r", INTEL_LOCK], [*uv, "-c", INTEL_LOCK, "--no-deps", "--no-build-isolation", "-e", "."], [*uv, "-c", INTEL_LOCK, *reqs]], label
    assert len(install_env.install_commands(_common.get_leg("macos-x86_64", SPEC), [], "/v/bin/python", system=False, platform="linux")) == 2
    for label in ("linux-x86_64", "macos-arm64"):                      # the other legs build nothing differently: no flag
        assert "--no-build-package" not in install_env.install_commands(_common.get_leg(label, SPEC), ["dev"], "/v/bin/python", system=False, platform="linux")[0]


def test_the_lock_first_install_forbids_the_source_build_of_the_packages_that_the_launcher_names():
    """The leg names the same three packages as `no_build_flags` of scripts/topopt-env.sh, and the launcher gives them to the `-r <lock>` install.
    The spelling of the launcher flag is not checked here: it is the launcher's test (PR #50 checks it against the real uv). Main before PR #50 prints
    `--no-build-package <name>`, which `uv pip install` rejects; PR #50 prints `--only-binary <name>`. This test holds in both states. The leg
    uses `--only-binary <name>`, the form that `uv pip install` accepts (uv 0.12.1), and never `--no-build-package`."""
    import re
    launcher = (ROOT / "scripts" / "topopt-env.sh").read_text()
    body = launcher[launcher.index("no_build_flags() {"):]
    body = body[:body.index("\n}\n")]
    assert re.findall(r"--(?:no-build-package|only-binary) (\w+)", body) == list(install_env.NO_BUILD_PACKAGES) == ["numba", "llvmlite", "cryptography"]
    assert re.search(r"\"\$UV\" pip install --python \"\$PY\" (?:\$REINSTALL )?\$\(no_build_flags\) -r \"\$LOCKFILE\"", launcher)      # the launcher gives them to the `-r <lock>` install
    cmd = install_env.install_commands(_common.get_leg("macos-x86_64", SPEC), [], "/v/bin/python", system=False, platform="darwin")[0]
    assert cmd[cmd.index("--only-binary"):cmd.index("-r")] == NO_BUILD and "--no-build-package" not in cmd


def test_the_real_uv_accepts_the_flags_of_the_lock_first_install(tmp_path):
    """`uv pip install --dry-run` with the three flags and an installed package: it changes nothing and needs no index. (uv rejects `--no-build-package` here.)"""
    import shutil
    import subprocess
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is not on PATH")
    req = tmp_path / "req.txt"
    req.write_text("pytest\n")
    r = subprocess.run([uv, "pip", "install", "--dry-run", "--python", sys.executable, *NO_BUILD, "-r", str(req)], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr


def test_the_intel_mac_legs_run_python_3_13_as_the_other_legs():
    """PY-1: every leg runs 3.13 only (2026-10-27). The launcher makes Python 3.13 on an Intel Mac too (#422), the Python of the legs."""
    assert LEGS["macos-x86_64"]["pythons"] == LEGS["linux-x86_64-intel-pins"]["pythons"] == ["3.13"]
    assert [(e["label"], e["python"]) for e in plan_legs.plan("workflow_dispatch", SPEC, workflow="intel-mac")[0]] == [("macos-x86_64", "3.13")]
    assert '"$UV" venv --python "$LAUNCHER_PYTHON" --managed-python --clear' in (ROOT / "scripts" / "topopt-env.sh").read_text()      # REL-9: the launcher calls its uv by full path


@pytest.mark.parametrize("name", ["tests.yml", "intel-mac.yml"])
def test_the_workflow_makes_the_venv_with_a_uv_managed_python_of_the_matrix_entry_and_names_no_version(name):
    """PY-1: a workflow has no Python version of its own. plan_legs.py gives it as `matrix.python`, from `pythons` of the leg in legs.toml."""
    for job, body in _load(name)["jobs"].items():
        runs = [s["run"] for s in body["steps"] if str(s.get("run", "")).startswith("uv venv")] if "steps" in body else []
        assert runs in ([], ["uv venv --python ${{ matrix.python }} --managed-python .venv"]), (job, runs)
    assert "--python 3." not in _path(name).read_text() and "python3.1" not in _path(name).read_text()


def test_the_requirements_of_the_extras_come_from_pyproject_in_order_and_once():
    text = '[project]\nname="x"\n[project.optional-dependencies]\na = ["p==1", "q==2; sys_platform == \'linux\'"]\nb = ["q==2; sys_platform == \'linux\'", "r"]\n'
    assert install_env.extra_requirements(text, ["a", "b", "missing"]) == ["p==1", "q==2; sys_platform == 'linux'", "r"]
    reqs = install_env.extra_requirements((ROOT / "pyproject.toml").read_text(), ["pardiso", "verify"])
    assert any(r.startswith("pypardiso==") for r in reqs) and any(r.startswith("cholespy==") for r in reqs)


def test_an_unknown_install_mode_is_an_error():
    with pytest.raises(_common.CiError, match="install_mode 'fast' is not one of project, lock-first"):
        install_env.install_commands({"install_mode": "fast"}, [], "/v/bin/python", system=False, platform="linux")


def test_the_intel_mac_leg_installs_no_apt_package_and_builds_no_shim():
    leg = _common.get_leg("macos-x86_64", SPEC)
    cmds = install_env.install_commands(leg, ["dev", "mcp", "tet", "verify"], "/v/bin/python", system=True, platform="darwin")
    assert {c[0] for c in cmds} == {"uv"} and cmds[0][-2:] == ["-r", INTEL_LOCK]


def test_the_linux_leg_with_the_intel_pins_installs_the_apt_packages_of_the_linux_leg():
    leg = _common.get_leg("linux-x86_64-intel-pins", SPEC)
    cmds = install_env.install_commands(leg, ["dev"], "/v/bin/python", system=True, platform="linux")
    assert cmds[0][:3] == ["sudo", "apt-get", "update"] and "libglu1-mesa" in cmds[1]


def test_install_env_dry_run_names_the_lock(capsys):
    assert install_env.main(["--leg", "linux-x86_64-intel-pins", "--dry-run", "--no-system"]) == 0
    out = capsys.readouterr().out
    assert f"lock {INTEL_LOCK} (install_mode lock-first)" in out and f"-r {INTEL_LOCK}" in out and f"-c {INTEL_LOCK} --no-deps --no-build-isolation -e ." in out
    assert "pypardiso==" in out and "cholespy==" in out


def test_install_env_stops_when_the_lock_file_of_the_leg_is_missing(monkeypatch):
    monkeypatch.setattr(install_env, "get_leg", lambda label: {**_common.get_leg("linux-x86_64", SPEC), "lock": "requirements-ghost.lock"})
    with pytest.raises(_common.CiError, match="names the lock 'requirements-ghost.lock'"):
        install_env.main(["--leg", "linux-x86_64", "--dry-run"])


# --- plan_legs.py: one matrix for each workflow ------------------------------------------------------

@pytest.mark.parametrize("event", ["pull_request", "push", "workflow_dispatch", "schedule"])
def test_tests_yml_never_plans_the_manual_intel_mac_leg(event):
    include, skipped = plan_legs.plan(event, SPEC)
    assert "macos-x86_64" not in {e["label"] for e in include} | {label for label, _ in skipped}


def test_the_intel_mac_workflow_plans_only_its_own_leg():
    include, skipped = plan_legs.plan("workflow_dispatch", SPEC, workflow="intel-mac")
    assert include == [{"label": "macos-x86_64", "os": "macos-15-intel", "python": "3.13", "shard": 1, "shards": 1, "suffix": "", "part": ""}] and not skipped
    assert tuple(include[0]) == plan_legs.MATRIX_KEYS


def test_plan_legs_main_takes_the_workflow_and_writes_its_matrix(tmp_path, monkeypatch):
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    assert plan_legs.main(["--workflow", "intel-mac", "--event", "workflow_dispatch"]) == 0
    assert json.loads(out.read_text().splitlines()[0][len("matrix="):]) == {"include": [{"label": "macos-x86_64", "os": "macos-15-intel", "python": "3.13", "shard": 1, "shards": 1, "suffix": "", "part": ""}]}
    assert "workflow `intel-mac`" in (tmp_path / "sum.md").read_text()
    with pytest.raises(_common.CiError, match="no leg of the workflow 'ghost'"):
        plan_legs.main(["--workflow", "ghost", "--event", "push"])


# --- check_locks.py: the leg key ---------------------------------------------------------------------

def test_check_locks_does_nothing_for_a_leg_without_the_key_and_needs_uv_for_a_leg_with_it(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(check_locks.shutil, "which", lambda name: None)
    assert check_locks.main(["--leg", "macos-arm64", "--artifacts", str(tmp_path)]) == 0
    assert "check_locks is not set" in capsys.readouterr().out
    with pytest.raises(_common.CiError, match="uv is not on PATH"):
        check_locks.main(["--leg", "linux-x86_64", "--artifacts", str(tmp_path)])
    with pytest.raises(_common.CiError, match="unknown leg"):
        check_locks.main(["--leg", "ghost"])


# --- check_workflow.py: the rule of each file --------------------------------------------------------

@pytest.mark.parametrize("name", ["tests.yml", "intel-mac.yml"])
def test_each_workflow_file_passes_the_check_with_its_own_rule(name, capsys):
    doc = _load(name)
    assert check_workflow.check_workflow(doc, ROOT, set(LEGS), name) == []
    assert check_workflow.main([str(_path(name))]) == 0
    assert f"OK: {_path(name)}" in capsys.readouterr().out


def test_the_rules_for_on_name_the_events_of_each_file():
    assert check_workflow.event_rule("tests.yml") == (("workflow_dispatch", "workflow_call"), True)   # manual only since 2026-10-24 (no extra budget); workflow_call: release.yml
    assert check_workflow.event_rule("intel-mac.yml") == (("workflow_dispatch",), True)
    assert check_workflow.event_rule("other.yml") == check_workflow.OTHER_RULE == (("workflow_dispatch",), True)     # any other file: by hand only
    assert set(check_workflow._on(_load("tests.yml"))) == {"workflow_dispatch", "workflow_call"}
    assert set(check_workflow._on(_load("intel-mac.yml"))) == {"workflow_dispatch"}     # the input `tier` is its only setting


def test_both_files_fail_with_a_second_event_or_without_workflow_dispatch():
    doc = copy.deepcopy(_load("intel-mac.yml"))
    doc[True]["push"] = {"branches": ["main"]}
    for name in ("intel-mac.yml", "tests.yml"):                       # the same strict rule for both files since 2026-10-24
        problems = check_workflow.check_workflow(doc, ROOT, set(LEGS), name)
        assert any("has the event push" in p and "allows only: workflow_dispatch" in p for p in problems), (name, problems)
    doc = copy.deepcopy(_load("tests.yml"))
    doc[True]["pull_request"] = {"paths-ignore": ["kb/**"]}
    problems = check_workflow.check_workflow(doc, ROOT, set(LEGS), "tests.yml")
    assert any("has the event pull_request" in p and "allows only: workflow_dispatch" in p for p in problems), problems
    doc = copy.deepcopy(_load("intel-mac.yml"))
    del doc[True]["workflow_dispatch"]
    doc[True]["schedule"] = [{"cron": "0 0 * * 0"}]
    problems = check_workflow.check_workflow(doc, ROOT, set(LEGS), "intel-mac.yml")
    assert any("`on` has no workflow_dispatch" in p for p in problems) and any("has the event schedule" in p for p in problems), problems
    doc = copy.deepcopy(_load("tests.yml"))
    del doc[True]["workflow_dispatch"]
    doc[True]["push"] = {"branches": ["main"]}
    problems = check_workflow.check_workflow(doc, ROOT, set(LEGS), "tests.yml")
    assert any("`on` has no workflow_dispatch" in p for p in problems) and any("has the event push" in p for p in problems), problems


def test_check_workflow_main_reads_the_rule_from_the_file_name(tmp_path, capsys):
    doc = copy.deepcopy(_load("intel-mac.yml"))
    doc[True]["pull_request"] = None
    bad = tmp_path / "intel-mac.yml"
    bad.write_text(yaml.safe_dump(doc))
    with pytest.raises(_common.CiError, match="problem"):
        check_workflow.main([str(bad)])
    assert "has the event pull_request" in capsys.readouterr().out
    other = tmp_path / "other-name.yml"
    other.write_text(yaml.safe_dump(doc))
    with pytest.raises(_common.CiError):                               # an unknown name gets the rule of tests.yml: manual only, so pull_request is refused
        check_workflow.main([str(other)])
    assert "has the event pull_request" in capsys.readouterr().out


# --- the two workflow files --------------------------------------------------------------------------

def test_the_test_job_of_the_manual_workflow_is_the_test_job_of_tests_yml():
    assert _load("intel-mac.yml")["jobs"]["test"] == _load("tests.yml")["jobs"]["test"]


def test_the_manual_workflow_has_no_event_but_workflow_dispatch_and_the_plan_asks_for_its_own_workflow():
    doc = _load("intel-mac.yml")
    assert set(doc["jobs"]) == {"plan", "test"} and doc["permissions"] == {"contents": "read"}
    plan_step = doc["jobs"]["plan"]["steps"][-1]
    assert "plan_legs.py --workflow intel-mac" in plan_step["run"]
    assert doc["concurrency"]["cancel-in-progress"] is False


def test_the_manual_workflow_says_it_is_inactive_names_the_runner_and_gives_the_move_commands():
    text = _path("intel-mac.yml").read_text()
    if "ci/github-workflows" in str(_path("intel-mac.yml")):
        head = "\n".join(text.splitlines()[:20])
        assert "INACTIVE" in text.splitlines()[1] and MOVE_INTEL in text and MOVE_BOTH in text and PREPARE in head
        assert "(A)" in head and "(B)" in head and "starts nothing on a push" in head and "tests.yml" in head
    assert LEGS["macos-x86_64"]["runner"] in text and "macos-26-intel" in text and "fall 2027" in text and "end date is" in text
    if "ci/github-workflows" in str(_path("tests.yml")):
        assert MOVE_BOTH in _path("tests.yml").read_text()
    else:                                                     # both headers say "by hand" and name the section of CONTRACTS.md
        assert "started by hand" in text.splitlines()[0] and "MANUAL ONLY" in _path("tests.yml").read_text().splitlines()[1]
        assert "CONTRACTS.md section 27" in text.splitlines()[0] and "CONTRACTS.md section 27" in _path("tests.yml").read_text().splitlines()[0]


def _pushed_legs():
    include, _ = plan_legs.plan("push", SPEC)
    return [e["label"] for e in include]


def test_no_kept_text_gives_the_price_of_a_runner_minute_or_the_cost_of_a_run():
    """The workflows, `legs.toml`, CONTRACTS.md and the reference pages give no price in dollars for a runner minute or for a run (a
    macOS minute counts more than a Linux minute: that is the fact that a contributor needs)."""
    files = {"contracts": (ROOT / "CONTRACTS.md").read_text(), "intel-mac.yml": _path("intel-mac.yml").read_text(), "tests.yml": _path("tests.yml").read_text(),
             "legs.toml": (CI / "legs.toml").read_text(), "readme": "\n".join(p.read_text() for p in [ROOT / "README.md", *sorted((ROOT / "docs" / "reference").glob("*.md"))])}
    for name, text in files.items():
        assert "$0.062" not in text and "$0.006" not in text and "billed minutes" not in text, name


def test_the_cache_glob_of_the_test_job_matches_every_lock_file_of_a_leg():
    steps = _load("tests.yml")["jobs"]["test"]["steps"]
    glob = [s["with"]["cache-dependency-glob"] for s in steps if str(s.get("uses", "")).startswith("astral-sh/setup-uv@")][0]
    for label, leg in LEGS.items():
        assert fnmatch.fnmatch(_common.lock_of(leg), glob), (label, glob)


def test_the_lock_check_is_a_step_of_the_test_job_and_runs_after_a_failed_step():
    st = [s for s in _load("tests.yml")["jobs"]["test"]["steps"] if "check_locks.py" in s.get("run", "")]
    assert len(st) == 1 and "!cancelled()" in st[0]["if"] and "--leg ${{ matrix.label }}" in st[0]["run"]
    assert "--python" not in st[0]["run"]      # every supported Python: no leg tests 3.12; 3.13 is the Python of the launcher


@pytest.mark.parametrize("name", ["tests.yml", "intel-mac.yml"])
def test_the_workflows_pin_the_uv_version_whose_error_text_the_lock_check_reads(name):
    """`check_locks.py` reads `<pin> has no usable wheels`, the wording of uv 0.12.1. Another uv version may word it otherwise (a false fail, never a false pass)."""
    steps = _load(name)["jobs"]["test"]["steps"]
    versions = [s["with"]["version"] for s in steps if str(s.get("uses", "")).startswith("astral-sh/setup-uv@")]
    assert versions == [check_locks.UV_TEXT_VERSION], (name, versions)


def test_the_runner_labels_of_the_legs_are_hosted_labels_that_exist():
    """Read on 2026-10-21 (actions/runner-images README): the Intel labels are macos-15-intel and macos-26-intel."""
    assert {leg["runner"] for leg in LEGS.values()} == {"ubuntu-24.04", "macos-15", "macos-15-intel"}


# --- the documents -----------------------------------------------------------------------------------

def test_contracts_27_10_names_the_files_and_the_legs_of_e4():
    text = (ROOT / "CONTRACTS.md").read_text()
    sec = text[text.index("### 27.10"):text.index("## 28. x86_64 Python on a Mac")]
    for word in ("check_locks.py", "tests/test_locks.py", "linux-x86_64-intel-pins", "macos-x86_64", "macos-15-intel", "intel-mac.yml",
                 "check_locks = true", "workflow_dispatch", "No solution found"):
        assert word in sec, word
    assert ".github/workflows/intel-mac.yml" in text and "a macOS minute counts more than a Linux minute" in text


def test_the_readme_section_platforms_matches_the_legs_and_the_locks():
    readme = (ROOT / "docs" / "reference" / "install.md").read_text()
    assert "\n## Platforms\n" in readme
    sec = readme[readme.index("\n## Platforms\n"):]
    sec = sec[:sec.index("\n## ", 5)]
    for word in (INTEL_LOCK, "intel-mac.yml", "macos-15-intel", "TOPOPT_ALLOW_SOURCE_BUILD", "amg", "cholesky"):
        assert word in sec, word
