"""The hook prompt guide (examples/hook/PROMPTS.md): every link and picture exists, and each prompt folder holds a problem YAML that loads.

No run is made here. The guide gives numbers of real runs; this test only checks that the files that the guide names are in the repo.
"""
import re
from pathlib import Path

import pytest

from topopt.config import load_config

HOOK = Path(__file__).resolve().parent.parent / "examples" / "hook"
GUIDE = HOOK / "PROMPTS.md"
FOLDERS = sorted(p for p in (HOOK / "prompts").iterdir() if p.is_dir())
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s#]+)(?:#[^)]*)?\)")


def test_the_guide_exists_and_has_prompts():
    text = GUIDE.read_text(encoding="utf-8")
    assert len(FOLDERS) >= 4
    assert text.count("\n> ") >= 3, "each prompt is a quoted line"


def test_every_link_and_picture_resolves():
    text = GUIDE.read_text(encoding="utf-8")
    found = LINK.findall(text)
    assert any(img for img, _, _ in found), "the guide has pictures"
    for img, _, target in found:
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (GUIDE.parent / target).exists(), f"{target} is missing"
        if img:
            assert (GUIDE.parent / target).stat().st_size < 200_000, f"{target} is too big for the repo"


def test_every_prompt_folder_is_named_in_the_guide():
    text = GUIDE.read_text(encoding="utf-8")
    for folder in FOLDERS:
        assert f"prompts/{folder.name}/" in text, f"{folder.name} is not linked from the guide"


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: p.name)
def test_problem_yaml_loads(folder):
    assert (folder / "spec.yaml").is_file()
    cfg = load_config(folder / "problem.yaml")
    assert cfg.load_cases, "the problem has a load case"


def test_the_guide_is_linked_from_the_readmes():
    for readme in (HOOK / "README.md", HOOK.parent / "README.md"):
        assert "PROMPTS.md" in readme.read_text(encoding="utf-8")
