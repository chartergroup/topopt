"""The README and the reference pages: every link to a repo file resolves, every anchor exists, every image file is there (REL-2).

The reference sections of the README live in `docs/reference/`. A move can leave a link that points at the old place.
GitHub shows a broken relative link as a page that does not exist, and a broken anchor as the top of the page, with no error.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PAGES = sorted((REPO / "docs" / "reference").glob("*.md"))
FILES = [REPO / "README.md", REPO / "docs" / "clients.md", REPO / "docs" / "media" / "README.md", REPO / "examples" / "README.md", *PAGES]

LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")


def prose(text: str) -> str:
    """The text without fenced code blocks and inline code: a link in a code sample is not a link."""
    out, fence = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
            continue
        if not fence:
            out.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def slug(heading: str) -> str:
    """The anchor that GitHub makes from a heading."""
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).replace("`", "").strip().lower()
    return re.sub(r"[^\w\- ]", "", t).replace(" ", "-")


def anchors(path: Path) -> set:
    seen, out, fence = {}, set(), False
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"^#{1,6} (.*)$", line)
        if m:
            s = slug(m.group(1))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else f"{s}-{n}")
    return out


def links(path: Path):
    for m in LINK.finditer(prose(path.read_text(encoding="utf-8"))):
        yield m.group(1) == "!", m.group(3)


def test_the_reference_pages_exist():
    assert len(PAGES) >= 5, "docs/reference/ holds the sections that moved out of the README"


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_every_relative_link_and_image_resolves(path):
    bad = []
    for is_image, target in links(path):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):
            continue                                    # a web address
        file_part, _, frag = target.partition("#")
        dest = path if not file_part else (path.parent / file_part).resolve()
        if not dest.exists():
            bad.append(f"{target}: no such file")
            continue
        if is_image and dest.stat().st_size == 0:
            bad.append(f"{target}: the image file is empty")
        if frag:
            if dest.suffix != ".md":
                bad.append(f"{target}: an anchor in a file that is not Markdown")
            elif frag not in anchors(dest):
                bad.append(f"{target}: no heading makes the anchor #{frag} in {dest.relative_to(REPO)}")
    assert not bad, f"{path.relative_to(REPO)}:\n" + "\n".join(bad)


def test_the_readme_links_to_every_reference_page():
    targets = {t.partition("#")[0] for _, t in links(REPO / "README.md")}
    for page in PAGES:
        assert f"docs/reference/{page.name}" in targets, f"README.md does not link to {page.name}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_a_reference_page_has_a_title_and_a_way_back(page):
    lines = page.read_text(encoding="utf-8").split("\n")
    assert lines[0].startswith("# ") and not lines[0].startswith("## "), f"{page.name}: line 1 must be the page title"
    assert "[Back to the README](../../README.md)" in lines[:4], f"{page.name}: no link back to the README"


def test_the_readme_has_its_first_screen():
    """The hero animation is above the first section, and the before and after strip is on the page. Each picture has an alt text."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    first = text.split("\n## ", 1)[0]
    assert "(docs/media/hero.gif)" in first
    for m in re.finditer(r"!\[([^\]]*)\]\(docs/media/", text):
        assert len(m.group(1)) > 20, "an image needs a real alt text"
    assert "<summary><b>Claude Code</b></summary>" in text and "docs/media/before-after-under.png" in text


def test_the_readme_uses_no_html_that_github_removes():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"<(video|audio|script|style|iframe|object|embed|source)\b", prose(text), re.I)
