"""#394: the README and the install doc say what the setup downloads and what topopt sends. Their facts follow the code.

The section "What the setup downloads, and what topopt sends" is in both files. It names numbers and hosts that live in other files:

  the pinned uv version      `PINNED_UV_VERSION` in `scripts/topopt-env.sh`
  the package count          the `==` pins of `requirements.lock`
  the build backend          `[build-system] requires` of `pyproject.toml` (a pin of the lock; the setup builds topopt with it to link the code)

So a new uv, a changed lock or another build backend is a changed test. The section also says that the pages of topopt load nothing
from the internet: `tests/test_pages_offline.py` keeps that true.
"""
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
INSTALL = ROOT / "docs" / "reference" / "install.md"
LAUNCHER = ROOT / "scripts" / "topopt-env.sh"
LOCK = ROOT / "requirements.lock"
PYPROJECT = ROOT / "pyproject.toml"
HEADING = "### What the setup downloads, and what topopt sends"
ANCHOR = "what-the-setup-downloads-and-what-topopt-sends"


def section(path: Path) -> str:
    """The text of the section, from its heading to the next heading of the same or a higher level. A `#` line inside a code
    block is not a heading."""
    text = path.read_text()
    assert text.count(HEADING) == 1, f"{path.name}: the heading must be there once"
    lines = text.split(HEADING, 1)[1].splitlines()
    fence = None
    for i, line in enumerate(lines):
        mark = re.match(r"\s*(```|~~~)", line)
        if mark:
            fence = None if fence == mark.group(1) else (fence or mark.group(1))
        elif fence is None and re.match(r"#{1,3} ", line):
            return "\n".join(lines[:i])
    return "\n".join(lines)


def pinned_uv_version() -> str:
    found = re.findall(r"^PINNED_UV_VERSION=(\S+)$", LAUNCHER.read_text(), re.MULTILINE)
    assert len(found) == 1
    return found[0]


def lock_pins() -> int:
    return sum(1 for line in LOCK.read_text().splitlines() if re.match(r"^[A-Za-z0-9_.-]+==\S+", line))


def build_backends() -> list[str]:
    requires = tomllib.loads(PYPROJECT.read_text())["build-system"]["requires"]
    return [re.split(r"[<>=!~ ;\[]", r, maxsplit=1)[0] for r in requires]


@pytest.mark.parametrize("path", [README, INSTALL], ids=["README", "install"])
def test_the_section_names_the_uv_that_the_launcher_downloads(path):
    assert f"uv {pinned_uv_version()}" in section(path)


@pytest.mark.parametrize("path", [README, INSTALL], ids=["README", "install"])
def test_the_section_gives_the_package_count_of_the_lock(path):
    count = lock_pins()
    assert count > 0
    assert f"{count} packages" in section(path)


@pytest.mark.parametrize("path", [README, INSTALL], ids=["README", "install"])
def test_the_section_names_the_build_backend_that_links_the_code(path):
    names = build_backends()
    assert names, "pyproject.toml has no build backend"
    for name in names:
        assert name in section(path), f"{name}: the lock pins it and step 3 links the code with it (#397); the section must name it"


@pytest.mark.parametrize("path", [README, INSTALL], ids=["README", "install"])
def test_the_section_names_no_cdn_for_the_pages(path):
    """The pages hold their scripts (#396). The texts of #395 named the CDN of the refine page as an exception."""
    assert "cdn.jsdelivr.net" not in section(path)


def test_the_readme_links_the_section_of_the_install_doc_from_the_first_start_line():
    readme = README.read_text()
    first_start = next(line for line in readme.splitlines() if line.startswith("The first start sets up Python for the plugin"))
    assert f"(#{ANCHOR})" in first_start
    assert f"docs/reference/install.md#{ANCHOR}" in section(README)


def test_the_readme_section_stays_short():
    lines = [line for line in section(README).splitlines() if line.strip()]
    assert len(lines) <= 10, f"{len(lines)} lines: the details belong in the install doc"
