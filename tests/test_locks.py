"""The lock files and the platform markers of `pyproject.toml` (E4).

Two checks:
- `marker_problems`: offline. For an Intel Mac, an Apple silicon Mac and Linux, every requirement of `pyproject.toml` that applies there must
  have the pin of the lock file of that platform. Mutants of the inputs show that the check sees a drift.
- `scripts/ci/check_locks.py`: the logic with a fake `uv`, and one real run with `uv` and the package index (it skips when `uv` or the index is
  not reachable). The real run is the check of the CI step: the Intel lock still resolves with wheels only for x86_64-apple-darwin.
"""
import shutil
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "ci"))

import _common  # noqa: E402
import check_locks  # noqa: E402

PYPROJECT = (ROOT / "pyproject.toml").read_text()
MAIN_LOCK = (ROOT / check_locks.MAIN_LOCK).read_text()
INTEL_LOCK = (ROOT / check_locks.INTEL_LOCK).read_text()

# The marker environments of the platforms that topopt names, with the lock file that each one uses.
ENVS = {
    "intel-mac": ({"platform_system": "Darwin", "platform_machine": "x86_64", "sys_platform": "darwin"}, "intel"),
    "apple-silicon": ({"platform_system": "Darwin", "platform_machine": "arm64", "sys_platform": "darwin"}, "main"),
    "linux-x86_64": ({"platform_system": "Linux", "platform_machine": "x86_64", "sys_platform": "linux"}, "main"),
    "linux-arm64": ({"platform_system": "Linux", "platform_machine": "aarch64", "sys_platform": "linux"}, "main"),
}
PY = {"python_version": "3.13", "python_full_version": "3.13.0"}
TWO_PYTHONS = ("3.13", "3.14")   # the tests of the loop over versions add a second one: the package supports 3.13 only (#469)
# The pins of the Intel lock that differ from the main lock and that `pyproject.toml` does not name: the pin of numba holds it
# (numba 0.62.1 needs llvmlite below 0.46).
TRANSITIVE_OK = {"llvmlite"}
# pyopencl (P9-4d): the requirement of pyproject.toml has a marker, an Apple silicon Mac or Linux x86_64. An Intel Mac has no requirement
# for it, so the Intel lock lacks the pin of pyopencl and the pins of the three packages that only it needs. The main lock is flat
# (a lock file has no markers), so it holds all four.
PYOPENCL_PINS = {"pyopencl", "pytools", "platformdirs", "siphash24"}
# A platform where the flat main lock holds a pin and the marker of pyproject.toml leaves the platform out. pyopencl has a wheel for Linux
# arm64 (uv resolves it), and the marker does not take that platform in yet.
LOCK_WIDER = {("pyopencl", "linux-arm64")}
ASKED_FOR_BY_NAME = {"gpu", "gpu-test"}


def marker_problems(pyproject_text: str, main_lock_text: str, intel_lock_text: str) -> list[str]:
    """Problems between the markers of pyproject.toml and the two lock files. An empty list means that they agree."""
    meta = tomllib.loads(pyproject_text)["project"]
    reqs = [(Requirement(s), "core") for s in meta["dependencies"]]
    for extra, items in meta.get("optional-dependencies", {}).items():
        reqs += [(Requirement(s), extra) for s in items]
    locks = {"main": check_locks.parse_pins(main_lock_text), "intel": check_locks.parse_pins(intel_lock_text)}
    problems: list[str] = []

    def pinned(r: Requirement):
        ops = list(r.specifier)
        return ops[0].version if len(ops) == 1 and ops[0].operator == "==" else None

    def applies(r: Requirement, env: dict) -> bool:
        return r.marker is None or r.marker.evaluate({**env, **PY})

    for env_name, (env, which) in ENVS.items():
        lock = locks[which]
        for r, extra in reqs:
            if not applies(r, env):
                continue
            name, want = check_locks.canonical(r.name), pinned(r)
            if want and name in lock and lock[name] != want:
                problems.append(f"{env_name}: pyproject.toml ({extra}) pins {name}=={want}, and the {which} lock has {name}=={lock[name]}")
    # A name that the pyproject pins and the lock holds must have a requirement that applies on every platform, except a pin that is
    # an override of the Intel Mac (a name that differs between the locks).
    differing = {n for n in locks["main"] if n in locks["intel"] and locks["main"][n] != locks["intel"][n]}
    names = {check_locks.canonical(r.name) for r, _ in reqs if pinned(r)}
    for name in sorted(differing):
        if name in names:
            if not any(applies(r, ENVS["intel-mac"][0]) and pinned(r) and check_locks.canonical(r.name) == name for r, _ in reqs):
                problems.append(f"intel-mac: the locks differ in {name}, and no requirement of pyproject.toml applies to it on an Intel Mac")
        elif name not in TRANSITIVE_OK:
            problems.append(f"the locks differ in {name}, and pyproject.toml has no requirement for it. Add an Intel marker line, "
                            f"or add {name} to TRANSITIVE_OK in this test with the reason")
    core_names = {check_locks.canonical(r.name) for r, extra in reqs if extra == "core" and pinned(r)}
    # The extras `gpu` and `gpu-test` are asked for by name and have no marker, so they say nothing about what a platform gets by default.
    default_reqs = [(r, extra) for r, extra in reqs if extra not in ASKED_FOR_BY_NAME]
    for name in sorted(names - differing):
        if name not in locks["main"] and name not in locks["intel"] and name not in core_names:
            continue
        for env_name, (env, which) in ENVS.items():
            has_requirement = any(applies(r, env) and pinned(r) and check_locks.canonical(r.name) == name for r, _ in default_reqs)
            if has_requirement and name not in locks[which]:
                problems.append(f"{env_name}: a requirement of pyproject.toml for {name} applies there, and the {which} lock has no pin for it")
            elif name in locks[which] and not has_requirement and (name, env_name) not in LOCK_WIDER:
                problems.append(f"{env_name}: {name} has a pin in pyproject.toml, and no requirement applies there")
    # The names of the two locks: the Intel lock has no name that the main lock lacks, and the main lock has the four pyopencl pins more.
    main_only = set(locks["main"]) - set(locks["intel"])
    if set(locks["intel"]) - set(locks["main"]):
        problems.append(f"the Intel lock pins {sorted(set(locks['intel']) - set(locks['main']))}, and the main lock does not")
    if main_only != PYOPENCL_PINS:
        problems.append(f"the main lock has {sorted(main_only)} and the Intel lock does not. It must be {sorted(PYOPENCL_PINS)} "
                        f"(pyopencl and what only it needs; an Intel Mac has no requirement for it)")
    # MKL: the pardiso extra gives no pin on a Mac (only Linux x86_64: CONTRIBUTING.md, MKL is an optional extra there).
    for r, extra in reqs:
        if extra == "pardiso":
            for env_name in ("intel-mac", "apple-silicon"):
                if applies(r, ENVS[env_name][0]):
                    problems.append(f"{env_name}: the pardiso extra gives a pin ({r.name}); MKL is allowed on Linux x86_64 only")
    return problems


# --- the offline marker check -------------------------------------------------------------------

def test_the_markers_of_pyproject_agree_with_both_lock_files():
    assert marker_problems(PYPROJECT, MAIN_LOCK, INTEL_LOCK) == []


def test_no_requirement_of_pyproject_has_a_marker_that_depends_on_the_python_version():
    """PY-1: one lock file serves every supported Python (the lock check resolves it for each one and gets the same pins; 3.13 only since #469).
    A marker on python_version would need a lock for each Python, and the check of the markers above (it fixes the Python at 3.13) would not see it."""
    meta = tomllib.loads(PYPROJECT)["project"]
    reqs = meta["dependencies"] + [r for items in meta["optional-dependencies"].values() for r in items]
    assert reqs and not [r for r in reqs if "python_" in r.partition(";")[2]], [r for r in reqs if "python_" in r.partition(";")[2]]


def test_the_locks_differ_in_the_four_pins_that_the_header_of_the_intel_lock_names_and_in_the_pyopencl_pins():
    main, intel = check_locks.parse_pins(MAIN_LOCK), check_locks.parse_pins(INTEL_LOCK)
    assert {n for n in intel if main[n] != intel[n]} == {"numpy", "numba", "llvmlite", "cryptography"}
    assert set(main) - set(intel) == PYOPENCL_PINS and set(intel) <= set(main)


def test_both_locks_pin_the_build_backend_at_a_version_that_build_system_allows():
    """#397: step 3 of the launcher links topopt with the setuptools of the environment (`--no-build-isolation`), not with one that uv gets from
    the index for an isolated build. So each lock (step 2 installs it) holds every package of `[build-system] requires`, at a version that the
    requirement allows, and the module of `build-backend` belongs to one of them."""
    build = tomllib.loads(PYPROJECT)["build-system"]
    reqs = [Requirement(r) for r in build["requires"]]
    assert check_locks.canonical(build["build-backend"].split(".")[0]) in {check_locks.canonical(r.name) for r in reqs}, build
    for which, text in (("main", MAIN_LOCK), ("intel", INTEL_LOCK)):
        pins = check_locks.parse_pins(text)
        for r in reqs:
            version = pins.get(check_locks.canonical(r.name))
            assert version is not None, f"the {which} lock has no pin for the build backend {r.name}: step 3 would fail offline"
            assert r.specifier.contains(version), f"the {which} lock pins {r.name}=={version}, and [build-system] requires {r}"


def test_the_check_reads_the_platform_of_each_marker():
    """The same markers give the Intel pins on an Intel Mac and the main pins elsewhere (so the check is not empty)."""
    meta = tomllib.loads(PYPROJECT)["project"]
    numpy = [Requirement(s) for s in meta["dependencies"] if Requirement(s).name == "numpy"]
    pins = {}
    for env_name, (env, _) in ENVS.items():
        hit = [str(next(iter(r.specifier)).version) for r in numpy if r.marker.evaluate({**env, **PY})]
        assert len(hit) == 1, env_name
        pins[env_name] = hit[0]
    assert pins == {"intel-mac": "2.3.5", "apple-silicon": "2.5.3", "linux-x86_64": "2.5.3", "linux-arm64": "2.5.3"}


def test_pyopencl_applies_on_an_apple_silicon_mac_and_on_linux_x86_64_only():
    """The marker of the brief: an Intel Mac and Linux arm64 have no requirement for pyopencl. The `gpu` extra has no marker."""
    meta = tomllib.loads(PYPROJECT)["project"]
    reqs = [Requirement(s) for s in meta["dependencies"] if Requirement(s).name == "pyopencl"]
    assert [str(next(iter(r.specifier)).version) for r in reqs] == ["2026.1.4"]
    hit = {env_name: reqs[0].marker.evaluate({**env, **PY}) for env_name, (env, _) in ENVS.items()}
    assert hit == {"intel-mac": False, "apple-silicon": True, "linux-x86_64": True, "linux-arm64": False}
    assert meta["optional-dependencies"]["gpu"] == ["pyopencl==2026.1.4"]


def _swap(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


@pytest.mark.parametrize("which, old, new, expect", [
    ("intel", "numba==0.62.1", "numba==0.62.0", "intel-mac: pyproject.toml (core) pins numba==0.62.1, and the intel lock has numba==0.62.0"),
    ("main", "numba==0.67.0", "numba==0.66.0", "apple-silicon: pyproject.toml (core) pins numba==0.67.0, and the main lock has numba==0.66.0"),
    ("main", "numpy==2.5.3", "numpy==2.5.4", "linux-x86_64: pyproject.toml (core) pins numpy==2.5.3, and the main lock has numpy==2.5.4"),
    ("pyproject", '"numpy==2.3.5;', '"numpy==2.3.4;', "intel-mac: pyproject.toml (core) pins numpy==2.3.4, and the intel lock has numpy==2.3.5"),
    ("pyproject", '"cryptography==48.0.1;', '"cryptography==48.0.0;', "intel-mac: pyproject.toml (mcp) pins cryptography==48.0.0"),
    ("pyproject", "scipy==1.18.1", "scipy==1.18.0", "pins scipy==1.18.0"),
    ("main", "pyopencl==2026.1.4", "pyopencl==2026.1.3",
     "linux-x86_64: pyproject.toml (core) pins pyopencl==2026.1.4, and the main lock has pyopencl==2026.1.3"),
    ("pyproject", '"pyopencl==2026.1.4; (platform_system', '"pyopencl==2026.1.3; (platform_system',
     "apple-silicon: pyproject.toml (core) pins pyopencl==2026.1.3, and the main lock has pyopencl==2026.1.4"),
])
def test_a_changed_pin_is_reported(which, old, new, expect):
    texts = {"pyproject": PYPROJECT, "main": MAIN_LOCK, "intel": INTEL_LOCK}
    texts[which] = _swap(texts[which], old, new)
    problems = marker_problems(texts["pyproject"], texts["main"], texts["intel"])
    assert any(expect in p for p in problems), problems


def test_a_removed_intel_line_is_reported():
    """Without the Intel line of numba, an Intel Mac installs numba 0.67.0 from pyproject.toml, and it has no wheel."""
    line = '  "numba==0.62.1; platform_system == \'Darwin\' and platform_machine == \'x86_64\'",\n'
    problems = marker_problems(_swap(PYPROJECT, line, ""), MAIN_LOCK, INTEL_LOCK)
    assert any("no requirement of pyproject.toml applies to it on an Intel Mac" in p and "numba" in p for p in problems), problems


def test_a_removed_intel_pin_of_cryptography_is_reported():
    """cryptography is a transitive package (under mcp). Without the Intel pin in the `mcp` extra, the Intel lock differs and nothing governs it."""
    line = ', "cryptography==48.0.1; platform_system == \'Darwin\' and platform_machine == \'x86_64\'"'
    problems = marker_problems(_swap(PYPROJECT, line, ""), MAIN_LOCK, INTEL_LOCK)
    assert any("differ in cryptography" in p for p in problems), problems


def test_a_missing_pyopencl_pin_is_reported():
    """The requirement applies on Linux x86_64 and on an Apple silicon Mac: the main lock must pin pyopencl and the three packages it needs."""
    for name in sorted(PYOPENCL_PINS):
        lines = MAIN_LOCK.splitlines(keepends=True)
        kept = "".join(l for l in lines if not l.startswith(f"{name}=="))
        assert kept != MAIN_LOCK, name
        problems = marker_problems(PYPROJECT, kept, INTEL_LOCK)
        assert any(f"the main lock has {sorted(PYOPENCL_PINS - {name})}" in p for p in problems), (name, problems)
        if name == "pyopencl":
            assert any("linux-x86_64: a requirement of pyproject.toml for pyopencl applies there, and the main lock has no pin" in p for p in problems)


def test_a_pyopencl_pin_in_the_intel_lock_is_reported():
    """An Intel Mac has no requirement for pyopencl: a pin in its lock has no reason (and the lock check would need a wheel there)."""
    problems = marker_problems(PYPROJECT, MAIN_LOCK, INTEL_LOCK + "pyopencl==2026.1.4\n")
    assert any("intel-mac: pyopencl has a pin in pyproject.toml, and no requirement applies there" in p for p in problems), problems


def test_a_marker_that_takes_in_an_intel_mac_is_reported():
    """Without the marker, `pip install .` on an Intel Mac asks for pyopencl, and the Intel lock has no pin for it."""
    old = '"pyopencl==2026.1.4; (platform_system == \'Darwin\' and platform_machine == \'arm64\') or (platform_system == \'Linux\' and platform_machine == \'x86_64\')"'
    problems = marker_problems(_swap(PYPROJECT, old, '"pyopencl==2026.1.4"'), MAIN_LOCK, INTEL_LOCK)
    assert any("intel-mac: a requirement of pyproject.toml for pyopencl applies there, and the intel lock has no pin" in p for p in problems), problems


def test_a_marker_that_hides_a_pin_on_a_platform_is_reported():
    problems = marker_problems(_swap(PYPROJECT, '"scipy==1.18.1"', '"scipy==1.18.1; platform_system != \'Darwin\'"'), MAIN_LOCK, INTEL_LOCK)
    assert any("apple-silicon: scipy has a pin in pyproject.toml, and no requirement applies there" in p for p in problems), problems


def test_a_pardiso_pin_on_a_mac_is_reported():
    old = "platform_system == 'Linux' and platform_machine == 'x86_64'\"]\n# Tet mesher"
    assert PYPROJECT.count(old) == 1
    problems = marker_problems(PYPROJECT.replace(old, "platform_system != 'Windows'\"]\n# Tet mesher"), MAIN_LOCK, INTEL_LOCK)
    assert any("the pardiso extra gives a pin" in p and "Linux x86_64 only" in p for p in problems), problems


# --- check_locks.py: the logic with a fake uv ----------------------------------------------------

NO_SOLUTION = ("  × No solution found when resolving dependencies:\n"
               "  ╰─▶ Because {pin} has no usable wheels and you require {pin}, we can conclude that your requirements are unsatisfiable.\n")


def fake_runner(calls, fail=None, network=()):
    """A runner like `run_compile`. `fail` maps a lock text pin to a failure; `network` lists platforms with a network error."""
    def runner(lock, platform, out):
        text = Path(lock).read_text()
        calls.append((Path(lock).name, platform))
        for pin in (fail or {}):
            if pin in text and platform == fail[pin]:
                return 1, NO_SOLUTION.format(pin=pin)
        if platform in network:
            return 2, "error: Failed to fetch: https://pypi.org/simple/numpy/ (error sending request)"
        if "numba==0.67.0" in text and platform == check_locks.INTEL_PLATFORM:
            return 1, NO_SOLUTION.format(pin="numba==0.67.0")
        if "cryptography==50.0.2" in text and platform == check_locks.INTEL_PLATFORM:
            return 1, NO_SOLUTION.format(pin="cryptography==50.0.2")
        out.write_text(text)
        return 0, f"Resolved {len(check_locks.parse_pins(text))} packages in 1ms\n"
    return runner


def test_parse_pins_reads_a_lock_and_the_output_of_uv():
    text = "# comment\nNumPy==2.5.3\nPyYAML==6.0.3 \\\n    --hash=sha256:abc\nscikit_image==0.26.0\n  indented==1\n"
    assert check_locks.parse_pins(text) == {"numpy": "2.5.3", "pyyaml": "6.0.3", "scikit-image": "0.26.0"}


@pytest.mark.parametrize("line", ["numba==0.62.1 ; sys_platform == 'linux'", "numba==0.62.1; sys_platform == 'linux'", "numba==0.62.1\t;python_version >= '3'"])
def test_parse_pins_rejects_a_line_with_an_environment_marker(line):
    with pytest.raises(_common.CiError, match=r"the lock, line 2 has an environment marker .*can hide a pin on a platform"):
        check_locks.parse_pins("a==1\n" + line + "\n")
    with pytest.raises(_common.CiError, match="requirements-macos-x86_64.lock, line 1 has an environment marker"):
        check_locks.parse_pins(line + "\n", where="requirements-macos-x86_64.lock")


def test_compare_pins_names_the_added_dropped_and_changed_pins():
    assert check_locks.compare_pins({"a": "1", "b": "2"}, {"a": "1", "b": "2"}) == ""
    assert check_locks.compare_pins({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "9", "d": "4"}) == (
        "added (resolved, not in the lock): d==4; dropped (in the lock, not resolved): c==3; changed (lock -> resolved): b 2 -> 9")


def test_with_pin_replaces_one_line_and_rejects_a_missing_name():
    assert check_locks.with_pin("a==1\nnumba==0.62.1\nnumba-x==1\n", "numba", "0.67.0") == "a==1\nnumba==0.67.0\nnumba-x==1\n"
    with pytest.raises(_common.CiError, match="no line for ghost"):
        check_locks.with_pin("a==1\n", "ghost", "1")


@pytest.mark.parametrize("rc, text, state", [
    (0, "Resolved 51 packages", "resolved"),
    (1, NO_SOLUTION.format(pin="numba==0.67.0"), "unresolvable"),
    (2, "error: Failed to fetch: https://pypi.org/simple/numpy/", "error"),
    (1, "", "error"),
])
def test_classify_separates_no_solution_from_other_errors(rc, text, state):
    assert check_locks.classify(rc, text) == state


def test_every_target_resolves_and_every_control_fails(tmp_path):
    calls = []
    rows = check_locks.check(fake_runner(calls), ROOT, tmp_path)
    assert [r["kind"] for r in rows] == ["target"] * 3 + ["control"] * 3
    assert all(r["ok"] for r in rows), rows
    assert calls[:3] == [(check_locks.INTEL_LOCK, "x86_64-apple-darwin"), (check_locks.MAIN_LOCK, "aarch64-apple-darwin"),
                         (check_locks.MAIN_LOCK, "x86_64-unknown-linux-gnu")]
    assert rows[0]["detail"] == "51 pins resolved, 51 in the lock"
    for slug, names in (("numba==0.67.0_and_llvmlite==0.49.0", ("numba==0.67.0", "llvmlite==0.49.0")), ("cryptography==50.0.2", ("cryptography==50.0.2",))):
        text = (tmp_path / f"Intel_lock_with_{slug}.in.txt").read_text()                 # the input of a control holds the changed pins
        assert all(text.count(name) == 1 for name in names)
        assert "numpy==2.3.5" in text and ("numba==0.62.1" in text) == ("numba==0.67.0" not in names)      # the other pins stay; a moved pin has only its new line


def test_a_target_that_stops_resolving_is_a_wrong_result(tmp_path):
    """The Intel lock gets a pin without a wheel (the drift that the check exists for)."""
    rows = check_locks.check(fake_runner([], fail={"numpy==2.3.5": "x86_64-apple-darwin"}), ROOT, tmp_path, controls=False)
    assert [(r["name"], r["state"], r["ok"]) for r in rows] == [
        ("Intel lock on x86_64-apple-darwin", "unresolvable", False),
        ("main lock on aarch64-apple-darwin", "resolved", True),
        ("main lock on x86_64-unknown-linux-gnu", "resolved", True)]
    assert "numpy==2.3.5 has no usable wheels" in rows[0]["detail"]


def _resolving(pins_of):
    """A fake uv that always resolves. It prints the pins that `pins_of(lock name, platform, lock text)` gives (what the real uv would resolve)."""
    def runner(lock, platform, out):
        out.write_text(pins_of(Path(lock).name, platform, Path(lock).read_text()))
        return 0, "Resolved\n"
    return runner


def _root(tmp_path, intel=None, main=None):
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    (root / check_locks.INTEL_LOCK).write_text(INTEL_LOCK if intel is None else intel)
    (root / check_locks.MAIN_LOCK).write_text(MAIN_LOCK if main is None else main)
    return root


def _intel_row(rows):
    return [r for r in rows if r["name"] == "Intel lock on x86_64-apple-darwin"][0]


def _full(name, platform, text):
    """The pins of the real locks: what uv resolves when nothing is missing."""
    return INTEL_LOCK if name == check_locks.INTEL_LOCK else MAIN_LOCK


def test_a_target_is_ok_only_when_the_resolved_pins_equal_the_pins_of_the_lock(tmp_path):
    rows = check_locks.check(_resolving(_full), _root(tmp_path / "ok"), tmp_path / "w1", controls=False)
    assert [(r["state"], r["ok"]) for r in rows] == [("resolved", True)] * 3
    # a name that the lock lacks, and that another pin needs: uv resolves it, so it is an added pin
    no_scipy = "\n".join(l for l in INTEL_LOCK.splitlines() if not l.startswith("scipy==")) + "\n"
    row = _intel_row(check_locks.check(_resolving(_full), _root(tmp_path / "a", intel=no_scipy), tmp_path / "w2", controls=False))
    assert (row["state"], row["ok"]) == ("differs", False)
    assert "50 in the lock" in row["detail"] and "added (resolved, not in the lock): scipy==1.18.1" in row["detail"]
    # a pin that uv does not resolve (a marker that hides it would do this; the marker itself is refused below)
    row = _intel_row(check_locks.check(_resolving(lambda n, p, text: "\n".join(l for l in _full(n, p, text).splitlines() if not l.startswith("numba==")) + "\n"),
                                       _root(tmp_path / "d"), tmp_path / "w3", controls=False))
    assert (row["state"], row["ok"]) == ("differs", False) and "dropped (in the lock, not resolved): numba==0.62.1" in row["detail"]
    # a pin that uv resolves to another version
    row = _intel_row(check_locks.check(_resolving(lambda n, p, text: _full(n, p, text).replace("numpy==2.3.5", "numpy==2.3.6")),
                                       _root(tmp_path / "c"), tmp_path / "w4", controls=False))
    assert (row["state"], row["ok"]) == ("differs", False) and "changed (lock -> resolved): numpy 2.3.5 -> 2.3.6" in row["detail"]


def test_a_lock_line_with_a_marker_is_a_wrong_result_and_uv_is_not_asked(tmp_path):
    """`numba==0.62.1 ; sys_platform == 'linux'` hides the pin on macOS: uv resolves 50 pins and the old check said ok."""
    marker = "\n".join(l + " ; sys_platform == 'linux'" if l.startswith("numba==") else l for l in INTEL_LOCK.splitlines()) + "\n"
    calls = []

    def runner(lock, platform, out):
        calls.append(Path(lock).name)
        return _resolving(_full)(lock, platform, out)
    rows = check_locks.check(runner, _root(tmp_path, intel=marker), tmp_path / "w", controls=False)
    row = _intel_row(rows)
    assert (row["state"], row["ok"]) == ("invalid lock", False)
    assert "line" in row["detail"] and "environment marker" in row["detail"] and "numba==0.62.1" in row["detail"]
    assert check_locks.INTEL_LOCK not in calls and [r["ok"] for r in rows] == [False, True, True]


def test_an_empty_lock_is_a_wrong_result_with_no_controls(tmp_path):
    """The old check said ok: uv resolves nothing from an empty file."""
    rows = check_locks.check(_resolving(lambda n, p, text: ""), _root(tmp_path, intel="# empty\n"), tmp_path / "w", controls=False)
    row = _intel_row(rows)
    assert (row["state"], row["ok"]) == ("invalid lock", False) and "has no `name==version` line" in row["detail"]


def test_an_empty_lock_with_the_controls_gives_rows_and_no_exception(tmp_path):
    """The control cannot make its input (the lock has no numba line). That is a wrong result in the table, not a stop before the table."""
    rows = check_locks.check(_resolving(lambda n, p, text: ""), _root(tmp_path, intel="# empty\n"), tmp_path / "w")
    controls = {r["name"]: r for r in rows if r["kind"] == "control"}
    assert len(controls) == 3 and controls["main lock on x86_64-apple-darwin"]["state"] != "invalid lock"
    for name in ("Intel lock with numba==0.67.0 and llvmlite==0.49.0", "Intel lock with cryptography==50.0.2"):
        assert (controls[name]["state"], controls[name]["ok"]) == ("invalid lock", False) and "has no line for" in controls[name]["detail"]
    assert not _intel_row(rows)["ok"]


def test_main_exits_with_1_for_each_of_the_three_bad_locks(tmp_path, monkeypatch, capsys):
    """The same three cases through `main`: the exit code is 1 and the table has the case with the word WRONG."""
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    monkeypatch.setattr(check_locks.shutil, "which", lambda name: "/fake/uv")
    no_scipy = "\n".join(l for l in INTEL_LOCK.splitlines() if not l.startswith("scipy==")) + "\n"
    marker = INTEL_LOCK.replace("numba==0.62.1", "numba==0.62.1 ; sys_platform == 'linux'")
    for n, bad in enumerate((no_scipy, marker, "# empty\n")):
        root = _root(tmp_path / f"r{n}", intel=bad)
        monkeypatch.setattr(check_locks, "ROOT", root)
        monkeypatch.setattr(check_locks, "run_compile", lambda uv, lock, plat, out, python: _resolving(_full)(lock, plat, out))
        with pytest.raises(_common.CiError, match="1 case.s. gave the wrong result: Intel lock on x86_64-apple-darwin"):
            check_locks.main(["--no-controls", "--python", "3.13", "--artifacts", str(tmp_path / f"art{n}")])
        assert "WRONG" in capsys.readouterr().out


def test_a_network_error_is_a_wrong_result_for_a_target_and_never_a_pass_for_a_control(tmp_path):
    rows = check_locks.check(fake_runner([], network=("x86_64-apple-darwin",)), ROOT, tmp_path)
    by = {r["name"]: r for r in rows}
    assert by["Intel lock on x86_64-apple-darwin"]["state"] == "error" and not by["Intel lock on x86_64-apple-darwin"]["ok"]
    for name in ("main lock on x86_64-apple-darwin", "Intel lock with numba==0.67.0 and llvmlite==0.49.0"):
        assert by[name]["state"] == "error" and not by[name]["ok"] and "network" in by[name]["detail"]


def test_a_control_that_resolves_is_a_wrong_result_with_the_meaning(tmp_path):
    """If numba 0.67.0 had a macOS x86_64 wheel, the control would resolve, and the check could not see this drift."""
    def runner(lock, platform, out):
        out.write_text(Path(lock).read_text())
        return 0, "Resolved 51 packages in 1ms\n"
    rows = check_locks.check(runner, ROOT, tmp_path)
    controls = [r for r in rows if r["kind"] == "control"]
    assert controls and not any(r["ok"] for r in controls)
    assert all("could not see this drift" in r["detail"] for r in controls)


# What uv 0.12.1 prints for a version conflict (numba 0.67.0 with llvmlite 0.45.1, on a platform where both have wheels; measured on x86_64-unknown-linux-gnu).
CONFLICT = ("  × No solution found when resolving dependencies:\n"
            "  ╰─▶ Because numba==0.67.0 depends on llvmlite>=0.49.0.dev0,<0.50 and you require llvmlite==0.45.1, we can conclude that your\n"
            "      requirements are unsatisfiable.\n")
CACHE = ("  × No solution found when resolving dependencies:\n"
         "  ╰─▶ Because annotated-types was not found in the cache and you require annotated-types==0.8.0, we can conclude that your\n"
         "      requirements are unsatisfiable.\n")


def _control_rows(tmp_path, text):
    """Every control fails with `text` ("No solution found"); the targets resolve."""
    def runner(lock, platform, out):
        if platform == check_locks.INTEL_PLATFORM and ("0.67.0" in Path(lock).read_text() or "50.0.2" in Path(lock).read_text()):
            return 1, text
        out.write_text(Path(lock).read_text())
        return 0, ""
    return [r for r in check_locks.check(runner, ROOT, tmp_path) if r["kind"] == "control"]


def test_a_control_must_fail_for_a_missing_wheel_and_not_for_another_reason(tmp_path):
    """A version conflict (the old numba control on Linux), or a package that is not in the cache, is `No solution found` too. It is not a pass."""
    for text in (CONFLICT, CACHE, NO_SOLUTION.format(pin="scipy==1.0")):                 # the last one: a wheel of a pin that the control did not move
        rows = [r for r in _control_rows(tmp_path / "w", text) if r["name"] != "main lock on x86_64-apple-darwin"]
        assert rows and all(r["state"] == "unresolvable" and not r["ok"] for r in rows), (text, rows)
        assert all("has no usable wheels" in r["detail"] and "uv 0.12.1" in r["detail"] for r in rows)
    main = [r for r in _control_rows(tmp_path / "m", CACHE) if r["name"] == "main lock on x86_64-apple-darwin"][0]
    assert not main["ok"]                                                                 # the main lock control needs the wheel text too
    assert [r["ok"] for r in _control_rows(tmp_path / "ok", NO_SOLUTION.format(pin="cryptography==50.0.2"))] == [True, False, True]


def test_a_control_needs_one_of_its_moved_pins_in_the_wheel_text_and_uv_may_wrap_the_line(tmp_path):
    wrapped = ("  × No solution found when resolving dependencies:\n  ╰─▶ Because llvmlite==0.49.0 has no usable\n"
               "      wheels and you require llvmlite==0.49.0, we can conclude that your requirements are unsatisfiable.\n")
    rows = [r for r in _control_rows(tmp_path, wrapped) if r["name"].startswith("Intel lock with numba")]
    assert [(r["state"], r["ok"]) for r in rows] == [("unresolvable", True)]             # llvmlite is one of the two moved pins
    pins = (("numba", "0.67.0"), ("llvmlite", "0.49.0"))
    assert check_locks.pins_without_wheels(wrapped, pins) == ["llvmlite==0.49.0"]
    assert check_locks.pins_without_wheels(CONFLICT, pins) == []
    assert check_locks.pins_without_wheels("Because numba==0.67.0 has no usable wheels and", ()) == ["numba==0.67.0"]
    assert check_locks.pins_without_wheels("Because numba==0.67.0 has no usable wheels and", (("numpy", "1"),)) == []


def test_main_gives_exit_code_1_for_a_wrong_result_and_writes_the_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    monkeypatch.setattr(check_locks.shutil, "which", lambda name: "/fake/uv")
    runner = fake_runner([], fail={"numpy==2.3.5": "x86_64-apple-darwin"})
    monkeypatch.setattr(check_locks, "run_compile", lambda uv, lock, plat, out, python: runner(lock, plat, out))
    with pytest.raises(_common.CiError, match="1 case.s. gave the wrong result: Intel lock on x86_64-apple-darwin"):
        check_locks.main(["--no-controls", "--python", "3.13", "--artifacts", str(tmp_path / "art")])
    assert "WRONG" in capsys.readouterr().out and "Lock files" in (tmp_path / "sum.md").read_text()
    runner = fake_runner([])
    monkeypatch.setattr(check_locks, "run_compile", lambda uv, lock, plat, out, python: runner(lock, plat, out))
    assert check_locks.main(["--python", "3.13", "--artifacts", str(tmp_path / "art2")]) == 0
    assert "3 target(s) resolve, 3 control(s) fail as they should (Python 3.13)" in capsys.readouterr().out
    assert (tmp_path / "art2" / "lock-checks").is_dir()


def _recording_run_compile(monkeypatch, calls, fail_python=None):
    """Replace `run_compile` by a fake uv that records the Python version of each call. With `fail_python`, the Intel lock fails for that version only."""
    def run_compile(uv, lock, plat, out, python):
        calls.append(python)
        fail = {"numpy==2.3.5": "x86_64-apple-darwin"} if python == fail_python else None
        return fake_runner([], fail=fail)(lock, plat, out)
    monkeypatch.setattr(check_locks, "run_compile", run_compile)
    monkeypatch.setattr(check_locks.shutil, "which", lambda name: "/fake/uv")


def test_main_checks_every_supported_python_and_keeps_the_logs_of_each_one_apart(tmp_path, monkeypatch, capsys):
    """PY-1: without --python the check runs for each supported Python: three targets and three controls for each. The package supports 3.13
    only (#469), so the test adds a second version, as a change that supports a new Python would."""
    calls = []
    _recording_run_compile(monkeypatch, calls)
    monkeypatch.setattr(check_locks, "SUPPORTED_PYTHONS", TWO_PYTHONS)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    assert check_locks.main(["--artifacts", str(tmp_path / "art")]) == 0
    assert len(calls) == 12 and calls[:6] == ["3.13"] * 6 and calls[6:] == ["3.14"] * 6
    assert "6 target(s) resolve, 6 control(s) fail as they should (Python 3.13, 3.14)" in capsys.readouterr().out
    for v in TWO_PYTHONS:
        assert len(list((tmp_path / "art" / "lock-checks" / f"python-{v}").glob("*.log"))) == 6, v
    summary = (tmp_path / "sum.md").read_text()
    assert "Python 3.13, 3.14" in summary and "| Python |" in summary


def test_main_checks_python_3_13_only_by_default(tmp_path, monkeypatch):
    """#469: the package supports 3.13 only, so the CI step (no --python) resolves the locks for 3.13 only."""
    calls = []
    _recording_run_compile(monkeypatch, calls)
    assert check_locks.main(["--artifacts", str(tmp_path / "art")]) == 0 and set(calls) == {"3.13"} and len(calls) == 6


def test_main_with_python_checks_that_version_only_and_a_wrong_result_names_it(tmp_path, monkeypatch, capsys):
    calls = []
    _recording_run_compile(monkeypatch, calls, fail_python="3.14")
    monkeypatch.setattr(check_locks, "SUPPORTED_PYTHONS", TWO_PYTHONS)                  # a second version, as in the test above
    assert check_locks.main(["--python", "3.13", "--artifacts", str(tmp_path / "a")]) == 0 and set(calls) == {"3.13"}
    calls.clear()
    with pytest.raises(_common.CiError, match=r"1 case.s. gave the wrong result: Intel lock on x86_64-apple-darwin \(target, unresolvable, Python 3\.14\)"):
        check_locks.main(["--no-controls", "--python", "3.14", "--artifacts", str(tmp_path / "b")])
    assert set(calls) == {"3.14"}
    calls.clear()
    with pytest.raises(_common.CiError, match=r"1 case.s. gave the wrong result: .*Python 3\.14\)$"):                 # both versions: only 3.14 fails, and the text says so
        check_locks.main(["--no-controls", "--artifacts", str(tmp_path / "c")])
    assert "3.13" in calls and "3.14" in calls
    monkeypatch.undo()
    for old in ("3.11", "3.12"):
        with pytest.raises(SystemExit):                                                  # argparse: a version that is not supported
            check_locks.main(["--python", old, "--artifacts", str(tmp_path / "d")])


def test_run_compile_gives_uv_the_python_version_it_is_asked_for(monkeypatch, tmp_path):
    seen = []

    class Done:
        returncode, stdout, stderr = 0, "", ""
    monkeypatch.setattr(check_locks.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or Done())
    for v in _common.SUPPORTED_PYTHONS:
        check_locks.run_compile("uv", tmp_path / "x.lock", "x86_64-unknown-linux-gnu", tmp_path / "o.txt", v)
    assert [c[c.index("--python-version") + 1] for c in seen] == list(_common.SUPPORTED_PYTHONS)
    assert all("--no-build" in c and "--python-platform" in c for c in seen)


def test_main_without_uv_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(check_locks.shutil, "which", lambda name: None)
    with pytest.raises(_common.CiError, match="uv is not on PATH"):
        check_locks.main(["--artifacts", str(tmp_path)])


# --- check_locks.py: the real uv and the package index --------------------------------------------

def _real_runner(python=_common.SUPPORTED_PYTHONS[0]):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is not on PATH")
    return lambda lock, platform, out: check_locks.run_compile(uv, lock, platform, out, python)


@pytest.mark.parametrize("python", _common.SUPPORTED_PYTHONS)
def test_the_real_locks_resolve_for_their_platforms_and_the_controls_fail(tmp_path, python):
    """The check of the CI step, with uv and the package index (about 5 s with a warm uv cache), for each supported Python."""
    rows = check_locks.check(_real_runner(python), ROOT, tmp_path)
    if any(r["state"] == "error" for r in rows):
        pytest.skip("the package index is not reachable: " + "; ".join(r["detail"] for r in rows if r["state"] == "error")[:200])
    assert all(r["ok"] for r in rows), [(r["name"], r["state"], r["detail"]) for r in rows if not r["ok"]]


@pytest.mark.parametrize("python", _common.SUPPORTED_PYTHONS)
def test_the_controls_do_not_pass_on_a_platform_where_the_wheels_exist(tmp_path, python):
    """The controls on x86_64 Linux, where every wheel exists: each must resolve (or fail for another reason), so it is not ok.
    On x86_64-apple-darwin they pass only because a pin has no wheel there. The old numba control (numba 0.67.0 alone) stayed ok here, as a version conflict."""
    uv = _real_runner(python)

    def on_linux(lock, platform, out):
        return uv(lock, "x86_64-unknown-linux-gnu" if platform == check_locks.INTEL_PLATFORM else platform, out)
    rows = [r for r in check_locks.check(on_linux, ROOT, tmp_path) if r["kind"] == "control"]
    if any(r["state"] == "error" for r in rows):
        pytest.skip("the package index is not reachable: " + "; ".join(r["detail"] for r in rows if r["state"] == "error")[:200])
    assert len(rows) == 3 and not any(r["ok"] for r in rows), [(r["name"], r["state"], r["ok"]) for r in rows]
    assert {r["state"] for r in rows} == {"resolved"}


def test_a_drifted_intel_lock_fails_with_the_real_uv(tmp_path):
    """The Intel lock with a cryptography that has no macOS x86_64 wheel is a target that fails: the guard sees the drift.
    cryptography is one package: with numba 0.67.0 alone the failure would be a conflict with llvmlite 0.45.1, not a missing wheel."""
    for name in (check_locks.MAIN_LOCK, check_locks.INTEL_LOCK):
        (tmp_path / name).write_text((ROOT / name).read_text())
    (tmp_path / check_locks.INTEL_LOCK).write_text(check_locks.with_pin(INTEL_LOCK, "cryptography", "50.0.2"))
    rows = check_locks.check(_real_runner(), tmp_path, tmp_path / "work", controls=False)
    if rows[0]["state"] == "error":
        pytest.skip("the package index is not reachable: " + rows[0]["detail"])
    assert (rows[0]["state"], rows[0]["ok"]) == ("unresolvable", False) and "cryptography==50.0.2 has no usable wheels" in rows[0]["detail"]
    assert rows[1]["ok"] and rows[2]["ok"]


def test_an_intel_lock_without_a_package_that_another_pin_needs_fails_with_the_real_uv(tmp_path):
    """The Intel lock without its scipy line: uv still resolves scipy (scikit-image needs it), so the old check said ok (49 in the lock, 50 resolved)."""
    for name in (check_locks.MAIN_LOCK, check_locks.INTEL_LOCK):
        (tmp_path / name).write_text((ROOT / name).read_text())
    (tmp_path / check_locks.INTEL_LOCK).write_text("\n".join(l for l in INTEL_LOCK.splitlines() if not l.startswith("scipy==")) + "\n")
    rows = check_locks.check(_real_runner(), tmp_path, tmp_path / "work", controls=False)
    if rows[0]["state"] == "error":
        pytest.skip("the package index is not reachable: " + rows[0]["detail"])
    assert (rows[0]["state"], rows[0]["ok"]) == ("differs", False) and "added (resolved, not in the lock): scipy==" in rows[0]["detail"]
    assert rows[1]["ok"] and rows[2]["ok"]
