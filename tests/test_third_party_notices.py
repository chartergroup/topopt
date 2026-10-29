"""Test that THIRD_PARTY_NOTICES.md names every package that pyproject.toml declares.

The notices file has one table for the core packages (`## Core packages`) and one for each
extra (`## Extra: <name>`). The first column of a row is the package name, the second is the
pin, the third is the licence, the fourth is the source of the licence fact.

A package can have more than one pin in `pyproject.toml`: one for each environment marker (for
example `numpy==2.5.3; ...` and `numpy==2.3.5; ... macOS x86_64`). The pin cell of its row names
every pin. Format: the versions in a list, separated by `;`. A version can have a note in
brackets. Example: `2.5.3; 2.3.5 (macOS x86_64)`.

The test fails when:
- a package of `dependencies` or of an extra has no row in the table of its group,
- a row has no licence or no source,
- the pin cell of a row does not name every pin of `pyproject.toml` for that package, or names a
  pin that `pyproject.toml` does not have,
- a package with GPL or LGPL in its licence is not named in "Components whose licences matter",
- `pyproject.toml` lists licence files for the build and the notices file is not among them,
- a pin of `requirements.lock` or `requirements-macos-x86_64.lock` has no row in any table (REL-1),
- a vendored three.js file (the dashboard copy, REL-1; the refine copy, #396) has no path, revision or licence text in the notices,
- the two data files and the files with copied code are not named in the notices (REL-1),
- README.md has no "License" section with MIT and the links to LICENSE and the notices (REL-1),
- the extras table of docs/reference/install.md does not name a package that is not permissive, with its licence, in the row of its extra,
- the CI licence policy has no entry for a package whose metadata has no licence field (REL-1).

A package that the notices file lists and that `pyproject.toml` does not declare is not an error.
"""
import fnmatch
import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
NOTICES_NAME = "THIRD_PARTY_NOTICES.md"
NOTICES = ROOT / NOTICES_NAME
MATTER_TITLE = "Components whose licences matter"


def norm(name):
    """PEP 503 name: `Pillow` and `pillow`, `typing_extensions` and `typing-extensions` match."""
    return re.sub(r"[-_.]+", "-", name).lower()


def split_requirement(req):
    """Return (name, pin) of a requirement text. The pin is the `==` version, or None.

    The marker after `;` is ignored: a package with a marker still needs a row. The pins of the
    other markers of the same package are the work of `declared_packages`.
    """
    spec = req.split(";")[0].strip()
    m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$", spec)
    name, rest = m.group(1), m.group(2)
    pin = re.match(r"==\s*([^\s,]+)$", rest)
    return name, (pin.group(1) if pin else None)


def collect_pins(requirements):
    """Return {name: [pin, ...]}. A package keeps one pin for each requirement, in file order.

    Two requirements with the same name (the same PEP 503 name) and different markers give two
    pins. A pin that is repeated is kept once. A requirement without `==` gives None.
    """
    packages, first_name = {}, {}
    for req in requirements:
        name, pin = split_requirement(req)
        pins = packages.setdefault(first_name.setdefault(norm(name), name), [])
        if pin not in pins:
            pins.append(pin)
    return packages


def declared_packages(pyproject_text):
    """Return {group: {name: [pins]}}. The group is "core" or the name of an extra."""
    cfg = tomllib.loads(pyproject_text)["project"]
    groups = {"core": collect_pins(cfg.get("dependencies", []))}
    for extra, reqs in cfg.get("optional-dependencies", {}).items():
        groups[extra] = collect_pins(reqs)
    return groups


PIN_ENTRY = r"[^\s;()]+(?:\s*\([^();]*\))?"
PIN_CELL = re.compile(rf"{PIN_ENTRY}(?:\s*;\s*{PIN_ENTRY})*")


def pin_cell_versions(cell):
    """Return the versions that a pin cell names, in order, or None if the cell has a wrong format.

    Format: `VERSION` or `VERSION (note)`, and more of them separated by `;`.
    Example: `2.5.3; 2.3.5 (macOS x86_64)` gives ["2.5.3", "2.3.5"]. `none (9.1.1 checked)` gives ["none"].
    """
    if not PIN_CELL.fullmatch(cell):
        return None
    return [re.match(r"[^\s;()]+", entry).group() for entry in re.split(r"\s*;\s*", cell)]


def parse_notices(notices_text):
    """Return ({group: {normalised name: cells of the row}}, text of the "matter" section)."""
    rows, matter, group, in_matter = {}, [], None, False
    for line in notices_text.splitlines():
        heading = re.match(r"##\s+(.*?)\s*$", line)
        if heading:
            title = heading.group(1)
            extra = re.match(r"Extra:\s+(\S+)$", title)
            group = "core" if title == "Core packages" else (extra.group(1) if extra else None)
            in_matter = title == MATTER_TITLE
            continue
        if in_matter:
            matter.append(line)
        if group and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells[0] == "Package" or not set(cells[0]) - set("-: "):
                continue  # the header row and the separator row
            rows.setdefault(group, {})[norm(cells[0])] = cells
    return rows, "\n".join(matter)


def find_problems(pyproject_text, notices_text):
    """Return a list of texts, one for each problem. An empty list means that the file is in sync.

    A package that has a pin in `pyproject.toml` needs every pin in its row. A requirement of the
    same package without `==` does not count as a pin then.
    """
    declared = declared_packages(pyproject_text)
    rows, matter = parse_notices(notices_text)
    problems = []
    for group, packages in declared.items():
        label = "`dependencies`" if group == "core" else f"extra `{group}`"
        for name, all_pins in packages.items():
            pins = [p for p in all_pins if p is not None]
            cells = rows.get(group, {}).get(norm(name))
            if cells is None:
                problems.append(f"{name} ({label}) has no row in the table of its group")
                continue
            if len(cells) < 4 or not cells[2] or not cells[3]:
                problems.append(f"{name} ({label}) has a row without a licence or without a source")
                continue
            if pins:
                problems.extend(pin_problems(name, label, pins, cells[1]))
            elif not cells[1].lower().startswith("none"):
                problems.append(f"{name} ({label}) has no pin in pyproject.toml, so the row must start with "
                                f"`none`, not `{cells[1]}`")
            if "GPL" in cells[2] and norm(name) not in norm(matter):
                problems.append(f"{name} ({label}) has the licence `{cells[2]}` and is not named in "
                                f"'{MATTER_TITLE}'")
    return problems


def pin_problems(name, label, pins, cell):
    """Return a list with one text if the pin cell does not name exactly the pins, else []."""
    versions = pin_cell_versions(cell)
    if versions is None:
        return [f"{name} ({label}): the pin cell `{cell}` has a wrong format; write the versions "
                f"separated by `;`, for example `2.5.3; 2.3.5 (macOS x86_64)`"]
    missing = [p for p in pins if p not in versions]
    unknown = [v for v in versions if v not in pins]
    if not missing and not unknown:
        return []
    detail = []
    if missing:
        detail.append("the row does not name " + ", ".join(f"`{p}`" for p in missing))
    if unknown:
        detail.append("pyproject.toml does not have " + ", ".join(f"`{v}`" for v in unknown))
    return [f"{name} ({label}): the row says pin `{cell}`, pyproject.toml says `{'; '.join(pins)}` "
            f"({'; '.join(detail)})"]


def licence_file_problem(pyproject_text, root_names):
    """Return a text if the build ships a licence file and not the notices file, else None.

    `[project] license-files` (setuptools 77 and later) lists the files that go into the sdist and
    the wheel. Without that key, setuptools ships only files named LICEN[CS]E*, COPYING*, NOTICE*
    and AUTHORS*, and the notices file is none of them. `root_names` is the list of file names
    in the repository root.
    """
    patterns = tomllib.loads(pyproject_text)["project"].get("license-files")
    if patterns is None:
        if not any(fnmatch.fnmatchcase(n, "LICEN[CS]E*") for n in root_names):
            return None  # the build ships no licence file, so the notices file has no peer to follow
        return (f"the build ships the LICENSE file and {NOTICES_NAME} is not in the same list: set "
                f'`license-files = ["LICENSE", "{NOTICES_NAME}"]` in [project] (setuptools 77 or later)')
    if any(fnmatch.fnmatchcase(NOTICES_NAME, p) for p in patterns):
        return None
    return f"`license-files` in pyproject.toml is {patterns} and does not include {NOTICES_NAME}"


def test_notices_file_names_every_package():
    problems = find_problems(PYPROJECT.read_text(), NOTICES.read_text())
    assert not problems, "\n".join(problems)


def test_fake_packages_are_reported(tmp_path):
    """Add a fake package to a copy of pyproject.toml: the check must name it."""
    text = PYPROJECT.read_text()
    fake_core = text.replace("dependencies = [", 'dependencies = [\n  "fake-core-package==1.0",', 1)
    fake_extra = text.replace('dev = ["pytest", "pytest-xdist"]', 'dev = ["pytest", "pytest-xdist", "fake-dev-package>=2"]', 1)
    fake_group = text.replace("[project.optional-dependencies]",
                              '[project.optional-dependencies]\nfakeextra = ["fake-extra-package"]', 1)
    for name, changed in (("fake-core-package", fake_core), ("fake-dev-package", fake_extra),
                          ("fake-extra-package", fake_group)):
        assert changed != text, name
        copy = tmp_path / f"{name}.toml"
        copy.write_text(changed)
        problems = find_problems(copy.read_text(), NOTICES.read_text())
        assert any(name in p and "no row" in p for p in problems), (name, problems)


def test_changed_pin_is_reported():
    text = PYPROJECT.read_text()
    changed = text.replace("numpy==2.5.3", "numpy==9.9.9", 1)
    assert changed != text
    problems = find_problems(changed, NOTICES.read_text())
    assert len(problems) == 1 and "numpy" in problems[0] and "9.9.9" in problems[0], problems


def test_second_pin_is_reported():
    """numpy has two pins in pyproject.toml (one for macOS x86_64). A change of the second pin is a problem too."""
    text = PYPROJECT.read_text()
    assert "numpy==2.3.5;" in text
    changed = text.replace("numpy==2.3.5;", "numpy==9.9.9;", 1)
    problems = find_problems(changed, NOTICES.read_text())
    assert len(problems) == 1 and "numpy" in problems[0] and "9.9.9" in problems[0] and "2.3.5" in problems[0], problems


def test_row_without_a_marker_pin_is_reported():
    """Take a pin out of the real row: the check names the package and the pin. One problem for each package."""
    lines = NOTICES.read_text().splitlines()
    for name, pin in (("numpy", "2.3.5"), ("numba", "0.62.1")):
        cut = f"; {pin} (macOS x86_64)"
        # The first column is the package name. (The text before the tables has the same words.)
        index = [i for i, line in enumerate(lines) if line.startswith(f"| {name} |") and cut in line]
        assert len(index) == 1, (name, index)
        changed = list(lines)
        changed[index[0]] = lines[index[0]].replace(cut, "", 1)
        problems = find_problems(PYPROJECT.read_text(), "\n".join(changed))
        assert len(problems) == 1 and name in problems[0] and pin in problems[0], (name, problems)


def test_marker_pins_in_every_group():
    """The real pyproject.toml: each package with more than one pin, and the pins that it has."""
    declared = declared_packages(PYPROJECT.read_text())
    assert declared["core"]["numpy"] == ["2.5.3", "2.3.5"]
    assert declared["core"]["numba"] == ["0.67.0", "0.62.1"]
    assert declared["core"]["scipy"] == ["1.18.1"]
    assert declared["mcp"]["cryptography"] == ["48.0.1"]


HEAD = "| Package | Pin | Licence (SPDX) | Checked from |\n|---|---|---|---|\n"


def test_package_with_two_marker_pins():
    pyproject = ('[project]\nname = "x"\n'
                 'dependencies = ["Eps==1.0; sys_platform == \'linux\'", "eps==0.9; sys_platform == \'darwin\'"]\n'
                 '[project.optional-dependencies]\n'
                 'more = ["zeta==2.0; python_version < \'3.13\'", "zeta==1.5; python_version >= \'3.13\'"]\n')

    def notices(core_pin, more_pin="2.0; 1.5 (Python 3.13 or later)"):
        return (f"## Core packages\n\n{HEAD}| eps | {core_pin} | MIT | meta |\n\n"
                f"## Extra: more\n\n{HEAD}| zeta | {more_pin} | MIT | meta |\n")

    assert declared_packages(pyproject) == {"core": {"Eps": ["1.0", "0.9"]}, "more": {"zeta": ["2.0", "1.5"]}}
    # Both pins in the row are good, in any order, with or without a note.
    for cell in ("1.0; 0.9", "1.0; 0.9 (macOS)", "0.9 (macOS); 1.0", "1.0 (Linux);0.9 (macOS)"):
        assert find_problems(pyproject, notices(cell)) == [], cell
    # One pin missing is a problem. The text names the package and the pin.
    for cell, missing in (("1.0", "0.9"), ("0.9 (macOS)", "1.0")):
        problems = find_problems(pyproject, notices(cell))
        assert len(problems) == 1 and "Eps" in problems[0] and f"`{missing}`" in problems[0], (cell, problems)
    problems = find_problems(pyproject, notices("1.0; 0.9", more_pin="2.0"))
    assert len(problems) == 1 and "zeta" in problems[0] and "`1.5`" in problems[0] and "extra `more`" in problems[0]
    # A pin that pyproject.toml does not have is a problem.
    problems = find_problems(pyproject, notices("1.0; 0.9; 0.8"))
    assert len(problems) == 1 and "Eps" in problems[0] and "`0.8`" in problems[0], problems
    # A wrong format (a comma, a note without brackets, an empty cell) is a problem. No crash.
    for cell in ("1.0, 0.9", "1.0; 0.9 macOS", "1.0 (a; b); 0.9", "1.0;", ";", "none (1.0 checked)"):
        problems = find_problems(pyproject, notices(cell))
        assert len(problems) == 1 and "Eps" in problems[0], (cell, problems)
    # A requirement without `==` beside the pins is not a pin.
    loose = pyproject.replace('"eps==0.9; sys_platform == \'darwin\'"', '"eps>=0.5; sys_platform == \'win32\'"')
    assert find_problems(loose, notices("1.0")) == []
    assert any("`1.0`" in p for p in find_problems(loose, notices("0.9")))
    # Names match as PEP 503 names: `Eps` and `eps` are one package.
    assert declared_packages(pyproject)["core"].keys() == {"Eps"}


def test_pin_cell_format():
    assert pin_cell_versions("2.5.3") == ["2.5.3"]
    assert pin_cell_versions("2.5.3; 2.3.5 (macOS x86_64)") == ["2.5.3", "2.3.5"]
    assert pin_cell_versions("48.0.1 (macOS x86_64 only)") == ["48.0.1"]
    assert pin_cell_versions("none (9.1.1 checked)") == ["none"]
    assert pin_cell_versions("2.5.3, 2.3.5") is None
    assert pin_cell_versions("") is None


def test_rules_on_a_small_example():
    pyproject = ('[project]\nname = "x"\ndependencies = ["Alpha_Pkg==1.0", "beta[extra]==2.0; sys_platform == \'linux\'"]\n'
                 '[project.optional-dependencies]\nfree = ["gamma"]\ncopyleft = ["delta==3.1"]\n')
    good = ("## Components whose licences matter\n\n- delta is named here.\n\n"
            "## Core packages\n\n| Package | Pin | Licence (SPDX) | Checked from |\n|---|---|---|---|\n"
            "| alpha-pkg | 1.0 | MIT | meta |\n| beta | 2.0 | BSD-3-Clause | meta |\n\n"
            "## Extra: free\n\n| Package | Pin | Licence (SPDX) | Checked from |\n|---|---|---|---|\n"
            "| gamma | none (4.0 checked) | MIT | meta |\n\n"
            "## Extra: copyleft\n\n| Package | Pin | Licence (SPDX) | Checked from |\n|---|---|---|---|\n"
            "| delta | 3.1 | GPL-2.0-or-later | meta |\n")
    assert find_problems(pyproject, good) == []
    # The same row in the wrong group does not count.
    assert any("gamma" in p for p in find_problems(pyproject.replace('free = ["gamma"]', 'free = []\ndev = ["gamma"]'), good))
    # A GPL package that the "matter" section does not name.
    assert any("delta" in p and "Components" in p for p in find_problems(pyproject, good.replace("delta is named", "x is named")))
    # A row without a source, and an unpinned package with a pinned row.
    assert any("Alpha_Pkg" in p for p in find_problems(pyproject, good.replace("| alpha-pkg | 1.0 | MIT | meta |", "| alpha-pkg | 1.0 | MIT | |")))
    assert any("gamma" in p for p in find_problems(pyproject, good.replace("none (4.0 checked)", "4.0")))


def test_licence_file_rule():
    base = '[project]\nname = "x"\n'
    assert licence_file_problem(base, ["README.md", "pyproject.toml"]) is None
    assert NOTICES_NAME in licence_file_problem(base, ["LICENSE", "pyproject.toml"])
    assert NOTICES_NAME in licence_file_problem(base + 'license-files = ["LICENSE"]\n', ["LICENSE"])
    assert licence_file_problem(base + f'license-files = ["LICENSE", "{NOTICES_NAME}"]\n', ["LICENSE"]) is None
    assert licence_file_problem(base + 'license-files = ["LICENSE", "THIRD_PARTY_*"]\n', ["LICENSE"]) is None


def test_notices_file_goes_into_the_build_like_the_licence_file():
    problem = licence_file_problem(PYPROJECT.read_text(), [p.name for p in ROOT.iterdir()])
    assert problem is None, problem


# --- REL-1: the lock files, the vendored file, the data files, the README and the policy -----------------------

LOCK_FILES = ("requirements.lock", "requirements-macos-x86_64.lock")
README = ROOT / "README.md"
THREE_JS = ROOT / "topopt" / "dashboard" / "vendor" / "three.module.min.js"
REFINE_THREE_JS = ROOT / "topopt" / "refine" / "vendor" / "three.min.js"   # r128 for the refine page (#396)
POLICY = ROOT / "scripts" / "ci" / "license_policy.toml"


def lock_pins(lock_text):
    """Return [(name, version)] for the `name==version` lines of a lock file (comments and blank lines skipped)."""
    pins = []
    for line in lock_text.splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins.append((name.strip(), version.strip()))
    return pins


def table_row_names(notices_text):
    """Return the set of normalised names in the first cell of every table row of the notices file.

    The cell can hold a version after the name (`mkl 2026.2.0`). A mention in running text does not count.
    """
    names = set()
    for line in notices_text.splitlines():
        if not line.startswith("|"):
            continue
        cell = line.strip().strip("|").split("|")[0].strip()
        m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", cell)
        if m and cell != "Package" and set(cell) - set("-: "):
            names.add(norm(m.group(1)))
    return names


def lock_problems(lock_texts, notices_text):
    """Return a text for each lock pin that has no row in any table of the notices file."""
    rows = table_row_names(notices_text)
    problems = []
    for lock_name, lock_text in lock_texts.items():
        for name, version in lock_pins(lock_text):
            if norm(name) not in rows:
                problems.append(f"{name} (pin {version} in {lock_name}) has no row in any table of {NOTICES_NAME}")
    return problems


def test_every_lock_pin_has_a_row_in_the_notices():
    """The pip install and the .mcpb bundle install the pins of the lock files, not only the packages of pyproject.toml."""
    locks = {name: (ROOT / name).read_text() for name in LOCK_FILES}
    problems = lock_problems(locks, NOTICES.read_text())
    assert not problems, "\n".join(problems)


def test_a_lock_pin_without_a_row_is_reported():
    notices = f"{HEAD}| alpha 1.0 | MIT | meta |\n"
    assert lock_problems({"a.lock": "alpha==1.0\n"}, notices) == []
    assert lock_problems({"a.lock": "# comment\nAlpha_x==2.0  # note\n"}, notices) == [
        f"Alpha_x (pin 2.0 in a.lock) has no row in any table of {NOTICES_NAME}"]
    assert lock_problems({"a.lock": "alpha==1.0\n"}, "A mention of alpha in running text does not count.\n") != []
    assert table_row_names(f"{HEAD}| mkl 2026.2.0 | pypardiso | x | y |\n") == {"mkl"}


def test_the_lock_files_pin_the_same_packages_except_the_pyopencl_ones():
    """The notices need one row for each name. The two lock files differ in versions, and in pyopencl and the three packages that
    only it needs: an Intel Mac has no requirement for pyopencl (the marker in pyproject.toml)."""
    names = [{norm(n) for n, _ in lock_pins((ROOT / f).read_text())} for f in LOCK_FILES]
    assert names[0] - names[1] == {"pyopencl", "pytools", "platformdirs", "siphash24"} and names[1] <= names[0]


def third_party_files_section(notices_text):
    m = re.search(r"^## Third-party files in this repository\n(.*?)(?=^## )", notices_text, re.S | re.M)
    assert m, "the notices file has no section 'Third-party files in this repository'"
    return m.group(1)


@pytest.mark.parametrize("path", [THREE_JS, REFINE_THREE_JS], ids=["dashboard", "refine"])
def test_vendored_three_js_has_its_version_and_licence_text_in_the_notices(path):
    """The vendored file carries a one-line SPDX header. The notices hold its path, the version and the full licence text."""
    head = path.read_text(encoding="utf-8")[:600]
    years = re.search(r"Copyright (\d{4}-\d{4}) Three\.js Authors", head)
    revision = re.search(r'const \w+="(\d+)"', head)
    assert years and revision and "SPDX-License-Identifier: MIT" in head, head
    section = third_party_files_section(NOTICES.read_text())
    assert f"`{path.relative_to(ROOT)}`" in section, f"the section does not name {path.relative_to(ROOT)}"
    assert f"r{revision.group(1)}" in section, f"the section does not name the revision r{revision.group(1)}"
    assert f"Copyright © {years.group(1)} three.js authors" in section
    assert "Permission is hereby granted, free of charge" in section
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in section


@pytest.mark.parametrize("path", [THREE_JS, REFINE_THREE_JS, REFINE_THREE_JS.parent / "OrbitControls.js"],
                         ids=["dashboard", "refine", "refine-orbit-controls"])
def test_the_notices_give_the_sha256_of_each_copied_file(path):
    """A copied file that changes needs a new source check and a new sum in the notices."""
    section = third_party_files_section(NOTICES.read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() in section, f"{path.relative_to(ROOT)}: its sha256 is not in the notices"


def test_the_data_files_and_the_ported_code_are_named_in_the_notices():
    section = third_party_files_section(NOTICES.read_text())
    for path in ("topopt/data/printers.yaml", "topopt_mcp/materials/reference.md"):
        assert (ROOT / path).is_file(), path
        assert f"`{path}`" in section, f"{path} is not named in 'Third-party files in this repository'"
    for path in ("topopt/filter.py", "topopt/refine/build.py"):   # the files that say "copied/adapted from an earlier project"
        assert f"`{path}`" in section, path


def readme_license_section():
    m = re.search(r"^## License\n(.*?)(?=^## |\Z)", README.read_text(), re.S | re.M)
    assert m, "README.md has no '## License' section"
    return m.group(1)


def test_readme_has_a_license_section_that_matches_the_files():
    body = readme_license_section()
    cfg = tomllib.loads(PYPROJECT.read_text())["project"]
    assert cfg["license"] == "MIT" and "MIT" in body
    assert "(LICENSE)" in body and f"({NOTICES_NAME})" in body


def test_the_extras_table_names_each_package_that_is_not_permissive_with_its_licence():
    """The warning stands where a user installs an extra: the row of the extra in the extras table of docs/reference/install.md names each
    optional package that is not permissive, with its licence (the README keeps only the licence line and the link to the notices)."""
    text = (ROOT / "docs" / "reference" / "install.md").read_text()
    rows = {m.group(1): m.group(0) for m in re.finditer(r"^\| `([a-z-]+)` \|.*$", text, re.M)}
    expected = {"tet": [("gmsh", "GPL-2.0-or-later")],
                "verify": [("gmsh", "GPL-2.0-or-later"), ("cholespy", "GPL-2.0-or-later in effect")],
                "pardiso": [("pypardiso", "Intel license terms")],
                "step": [("cadquery-ocp-novtk", "LGPL")]}
    for extra, pairs in expected.items():
        for name, licence in pairs:
            assert name in rows[extra] and licence in rows[extra], (extra, name, licence)
    assert "Install an extra only if its license fits your use." in text


def test_the_licence_fields_of_the_files_that_we_ship_agree():
    """pyproject.toml, LICENSE, the plugin manifest and the .mcpb manifest all say MIT, and LICENSE has a copyright line."""
    cfg = tomllib.loads(PYPROJECT.read_text())["project"]
    assert cfg["license"] == "MIT" and "LICENSE" in cfg["license-files"]
    for path in (".claude-plugin/plugin.json", "packaging/mcpb/manifest.json"):
        assert json.loads((ROOT / path).read_text())["license"] == cfg["license"], path
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License\n")
    assert re.search(r"^Copyright \(c\) \d{4} \S.*$", text, re.M), "LICENSE has no 'Copyright (c) YEAR HOLDER' line"
    assert "Permission is hereby granted, free of charge" in text


def test_every_policy_entry_is_named_in_the_notices():
    """license_policy.toml says: every package of `allow` must also be named in the notices. The same holds for `override`."""
    policy = tomllib.loads(POLICY.read_text())
    rows = table_row_names(NOTICES.read_text())
    bold = {norm(m.group(1)) for m in re.finditer(r"\*\*([A-Za-z0-9][A-Za-z0-9._-]*)", NOTICES.read_text())}
    named = rows | bold
    missing = [e["package"] for e in policy.get("allow", []) + policy.get("override", []) if norm(e["package"]) not in named]
    assert not missing, f"named in license_policy.toml and not in {NOTICES_NAME}: {missing}"


def test_the_policy_covers_a_package_whose_metadata_has_no_licence_field():
    """pocl-binary-distribution (extra gpu-test) has no licence field; the notices read its licence from a wheel file (MIT)."""
    policy = tomllib.loads(POLICY.read_text())
    overrides = {norm(e["package"]): e for e in policy["override"]}
    assert overrides["pocl-binary-distribution"]["category"] == "permissive"
    assert "POCL.COPYING" in overrides["pocl-binary-distribution"]["why"]
