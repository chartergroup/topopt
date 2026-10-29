"""The examples gallery (`examples/README.md`): one tile for each folder of `examples/`, each with a README, a link and a PNG picture.

A new example folder with no tile fails here, so the gallery cannot fall behind. The main README and the gallery link to each other.
`EXAMPLES` is a parameter of the checks, so a test can run them on a copy of the tree.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def prose(text: str) -> str:
    """The text without fenced code blocks: a link in a code sample is not a tile (the same rule as `prose` in test_readme_links.py;
    `tests/` is not a package, so the small stripper is copied here)."""
    out, fence = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
            continue
        if not fence:
            out.append(line)
    return "\n".join(out)


def folders(examples: Path) -> list:
    """The example folders: a folder of `examples/` is an example, a script or `__pycache__` is not."""
    return sorted(p.name for p in examples.iterdir() if p.is_dir() and p.name != "__pycache__" and not p.name.startswith("."))


def problems(examples: Path, main_readme: Path) -> list:
    """Every way that the gallery can be wrong, as a list of texts (empty: the gallery is right)."""
    out = []
    gallery = examples / "README.md"
    if not gallery.is_file():
        return [f"{gallery.name} is missing"]
    text = gallery.read_text(encoding="utf-8")
    pairs = [(bang == "!", target) for bang, _, target in LINK.findall(prose(text))]
    for name in folders(examples):
        if f"{name}/README.md" not in [t for img, t in pairs if not img]:
            out.append(f"{name}: no tile link to {name}/README.md in the gallery")
        readme = examples / name / "README.md"
        if not readme.is_file():
            out.append(f"{name}: examples/{name}/README.md does not exist")
        elif "](../README.md)" not in readme.read_text(encoding="utf-8"):
            out.append(f"{name}: examples/{name}/README.md does not link back to ../README.md")
        if f"{name}/result.png" not in [t for img, t in pairs if img]:
            out.append(f"{name}: no picture {name}/result.png in the gallery")
    for img, target in pairs:
        if img and not target.startswith("http"):
            path = examples / target
            if not path.is_file():
                out.append(f"picture {target} does not exist")
            elif not path.read_bytes().startswith(PNG_MAGIC):
                out.append(f"picture {target} is not a PNG")
    if "](../README.md)" not in prose(text):
        out.append("the gallery does not link back to ../README.md")
    if "](examples/README.md)" not in main_readme.read_text(encoding="utf-8"):
        out.append("README.md does not link to examples/README.md")
    return out


def test_the_gallery_has_a_tile_for_every_example_folder():
    assert problems(REPO / "examples", REPO / "README.md") == []


def test_the_guard_sees_the_examples_that_exist_today():
    assert {"hook", "drone_arm", "interfaces", "ladle"} <= set(folders(REPO / "examples"))


def test_a_folder_with_no_tile_fails_and_is_named(tmp_path):
    shutil_tree = tmp_path / "examples"
    shutil_tree.mkdir()
    (shutil_tree / "README.md").write_text("[back](../README.md)\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("[gallery](examples/README.md)\n", encoding="utf-8")
    (shutil_tree / "zz_new").mkdir()
    (shutil_tree / "motion_rows.py").write_text("", encoding="utf-8")
    found = problems(shutil_tree, tmp_path / "README.md")
    assert found and all("zz_new" in f for f in found), found


def test_a_picture_that_is_not_a_png_fails(tmp_path):
    ex = tmp_path / "examples"
    (ex / "a").mkdir(parents=True)
    (ex / "a" / "README.md").write_text("[back](../README.md)", encoding="utf-8")
    (ex / "a" / "result.png").write_bytes(b"GIF89a")
    (ex / "README.md").write_text("[a](a/README.md) ![p](a/result.png) [back](../README.md)\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("[g](examples/README.md)\n", encoding="utf-8")
    assert problems(ex, tmp_path / "README.md") == ["picture a/result.png is not a PNG"]


def _tree(tmp_path, gallery: str, readme_text: str = "[back](../README.md)"):
    ex = tmp_path / "examples"
    (ex / "a").mkdir(parents=True)
    (ex / "a" / "README.md").write_text(readme_text, encoding="utf-8")
    (ex / "a" / "result.png").write_bytes(PNG_MAGIC)
    (ex / "README.md").write_text(gallery, encoding="utf-8")
    (tmp_path / "README.md").write_text("[g](examples/README.md)\n", encoding="utf-8")
    return ex, tmp_path / "README.md"


GOOD = "[a](a/README.md) ![p](a/result.png) [back](../README.md)\n"


def test_a_good_tree_has_no_problem(tmp_path):
    assert problems(*_tree(tmp_path, GOOD)) == []


def test_an_example_readme_with_no_back_link_fails(tmp_path):
    assert problems(*_tree(tmp_path, GOOD, "no link")) == ["a: examples/a/README.md does not link back to ../README.md"]


def test_a_tile_link_only_in_a_code_block_fails(tmp_path):
    gallery = "```\n[a](a/README.md) ![p](a/result.png)\n```\n[back](../README.md)\n"
    found = problems(*_tree(tmp_path, gallery))
    assert len(found) == 2 and all(f.startswith("a:") for f in found), found
