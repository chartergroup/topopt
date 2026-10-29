"""Shared fixtures and helpers of tests/test_part_card.py and tests/test_part_card_wiring.py (the part card, CONTRACTS.md section 9.2).

The fixtures in tests/fixtures/part_card/ are copies of real files: `run.json`, `metrics.json` and two rows of `history.jsonl` of a
real run of examples/hook/hook_short.yaml (CPU, 2026-10-23), and a `verify.json` of a real `topopt verify` run (another hook run,
cut down to the keys that the card reads).
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from topopt import part_card

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures" / "part_card"
STL_BYTES = b"solid x\nendsolid x\n"
HEADINGS = [title for title, _ in part_card.SECTIONS if title != "Fine pass"]   # "Fine pass" is in the card of a fine run, or of a run whose part needs a finer voxel (#318): tests/test_fine_cli.py


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def section(text: str, title: str) -> str:
    """The lines under the heading `## title`, up to the next heading."""
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert m, f"no section {title!r} in the card"
    return m.group(1)


@pytest.fixture
def work(tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / "hook_short.yaml").write_text("name: x\n")      # a YAML that does not load is fine for the case file
    return folder


@pytest.fixture
def run_dir(work):
    """A finished run folder made from the real sample files."""
    d = work / "runs" / "hook"
    (d / "result").mkdir(parents=True)
    run = json.loads((FIX / "run.json").read_text())
    run["yaml_path"] = str(work / "hook_short.yaml")
    (d / "run.json").write_text(json.dumps(run, indent=1))
    shutil.copy(FIX / "metrics.json", d / "result" / "metrics.json")
    shutil.copy(FIX / "history_first_last.jsonl", d / "history.jsonl")
    (d / "result" / "final.stl").write_bytes(STL_BYTES)
    (d / "status.json").write_text(json.dumps({"state": "done", "iteration": 40}))
    return d


def add_verify(d: Path, stl_sha=None, **edit):
    v = json.loads((FIX / "verify.json").read_text())
    v["run"]["final_stl_sha256"] = stl_sha or sha(STL_BYTES)
    v.update(edit)
    (d / "verify").mkdir(exist_ok=True)
    (d / "verify" / "verify.json").write_text(json.dumps(v))
    return v


def add_pngs(d: Path):
    (d / "renders").mkdir(exist_ok=True)
    for name in ("final_iso.png", "final_front.png", "final_montage.png"):
        (d / "renders" / name).write_bytes(b"\x89PNG")
