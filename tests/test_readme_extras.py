"""Guards for the text of the section "After the install: update, extras, check" (L1c). The text is in `docs/reference/install.md`
(it was in the README before the reference sections moved: REL-2). `README` below is that page.

Two facts live in two places, and a change in one place can leave the README wrong with no other test failing:
the extras of `pyproject.toml` (the "Extras" table), and the measured figures of `environment.GPU_MG_MEASURED`
(the table of "The GPU solver gpu-mg"). These tests compare the README with the code.
"""
import re
import tomllib
from pathlib import Path

import pytest

from topopt import environment

REPO = Path(__file__).resolve().parent.parent
README = (REPO / "docs" / "reference" / "install.md").read_text(encoding="utf-8")


def section(start: str, end: str | None = None) -> str:
    """The page text from the heading `start` up to the heading `end` (both are full heading lines), or to the end of the page."""
    a = README.index(f"\n{start}\n")
    b = README.index(f"\n{end}\n", a + 1) if end else len(README)
    return README[a:b]


def reference_page(name: str) -> str:
    """The text of `docs/reference/<name>`. The ratchet, the check of the printed part and the acceptance reports have their own pages."""
    return (REPO / "docs" / "reference" / name).read_text(encoding="utf-8")


def test_the_extras_table_names_every_extra_with_its_pins_and_install_command():
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    text = section("### Extras", "### Check your machine")
    assert extras, "pyproject.toml has no extras: the test reads the wrong key"
    for name, requirements in extras.items():
        rows = [line for line in text.splitlines() if line.startswith(f"| `{name}` |")]
        assert len(rows) == 1, f"the Extras table needs one row for the extra `{name}`"
        assert f'uv pip install -e ".[{name}]"' in rows[0], f"the row of `{name}` needs its install command"
        for requirement in requirements:
            package = re.match(r"[A-Za-z0-9_.-]+(?:==[^\s;]+)?", requirement).group(0)
            if "==" in package:
                assert package in rows[0], f"the row of `{name}` needs the pin {package}"
            else:
                assert f"`{package}`" in rows[0], f"the row of `{name}` needs the package {package}"


def test_the_gpu_mg_table_of_the_readme_equals_the_measured_figures_of_the_code():
    m = environment.GPU_MG_MEASURED
    text = section("### The GPU solver gpu-mg")
    assert "one machine" in text and m["machine"].split(",")[0] in text
    rows = re.findall(r"^\| ([\d.]+) mm \| ([\d,]+) \| ([\d.]+) s \| ([\d.]+) s \|", text, re.M)
    readme_rows = [(float(a), int(b.replace(",", "")), float(c), float(d)) for a, b, c, d in rows]
    code_rows = [(r["voxel_mm"], r["free_dofs"], r["default_s"], r["gpu_mg_s"]) for r in m["rows"]]
    assert readme_rows == code_rows
    f = m["finest"]
    assert f"{f['free_dofs']:,}" in text and f"{f['gpu_mg_s']:g} s" in text
    assert f"{f['default_memory_estimate_gib']:g} GiB" in text and f"{f['ram_gib']} GiB" in text
    assert m["compliance_agreement"] in text


def tracked_yaml_files(root: Path) -> list[Path]:
    """The YAML files under `root/examples` that git tracks: the shipped ones. A file that the user made (the README tells the user to
    copy `hook.yaml` to `examples/hook/hook_ratchet.yaml`) is not one of them. With no git, or no git repository around `root` (an
    sdist or a copy of the folder), every YAML file on disk is a shipped file."""
    import subprocess
    try:
        done = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "examples"], capture_output=True, text=True,
                              check=True, timeout=60)
        names = [n for n in done.stdout.split("\0") if n.endswith(".yaml")]
    except (OSError, subprocess.SubprocessError):
        names = []
    if names:
        return sorted(root / n for n in names)
    return sorted((root / "examples").rglob("*.yaml"))


def test_the_ratchet_section_says_that_no_shipped_yaml_has_the_acceptance_block_and_this_is_true():
    """Q1-09: the commands of the ratchet section (`docs/reference/verify.md`) pass a copy of the YAML with `--yaml`. If a shipped YAML gets
    the block, that section needs a new look.
    Only the tracked files count (Q1 R1 m3): the copy that the README tells the user to make must not fail this test."""
    import yaml
    files = tracked_yaml_files(REPO)
    assert REPO / "examples" / "hook" / "hook.yaml" in files and len(files) >= 5, files               # the guard reads the shipped files
    shipped = [p for p in files if "acceptance" in (yaml.safe_load(p.read_text()) or {})]
    assert shipped == [], f"{shipped} has an acceptance block: change the ratchet section of docs/reference/verify.md"
    page = reference_page("verify.md")
    text = page[page.index("\n## Ratchet (keep a candidate edit only if the rules pass)\n"):page.index("\n## Cleanup list (find spurs, bumps and loose bodies)\n")]
    assert "No shipped YAML has the block `acceptance`" in text and "examples/hook/hook_ratchet.yaml" in text
    commands = [x for x in page.splitlines() if re.match(r"python -m topopt\.ratchet (check|run)\s+runs/hook\b", x)]
    assert len(commands) == 3 and all("--yaml " in x for x in commands), commands


def test_no_reason_of_an_allowed_skip_names_an_open_pr_and_section_38_does_not_say_other_open_branches():
    """X7 and X8 of the Q1 fix: the earlier PRs and the sections 36 and 37 are in main."""
    legs = tomllib.loads((REPO / "scripts" / "ci" / "legs.toml").read_text())["legs"]
    whys = [s["why"] for leg in legs.values() for s in leg.get("allowed_skip", [])]
    assert whys and not [w for w in whys if "(open)" in w or "no PR number" in w], whys
    contracts = (REPO / "CONTRACTS.md").read_text()
    assert "are in other open branches" not in contracts and '`why` ends in "(open)"' not in contracts
    section_38 = contracts[contracts.index("## 38. Joints"):][:900]
    assert "Sections 35 (A7a) and 36 (D3) are in main" in section_38


def test_the_check_part_sample_has_no_reference_and_the_reference_example_uses_the_mesh_method_of_the_recording():
    """Q1-02: the Quick start run is `sdf`, the recording is `blur`: a `--ref` against it needs a `blur` run, and the page says so."""
    import json
    page = reference_page("print-step-tet.md")
    text = page[page.index("\n## Check the printed part\n"):page.index("\n## Your printer (the catalog)\n")]
    first = re.search(r"```\n(.*?)\n```", text, re.S).group(1)
    assert first == ".venv/bin/python tools/check_part.py runs/hook examples/hook/hook.yaml"          # exit code 0 as written
    assert "`blur` run" in text and "--mesh-method blur" in text and "metrics.json" in text
    recorded = json.loads((REPO / "tests" / "fixtures" / "part_card" / "older_build" / "run.json").read_text())   # the run.json of the recording
    assert recorded.get("mesh_method", "blur") == "blur"            # a run from before `sdf` has no key and is a `blur` run
    assert tomllib.loads((REPO / "pyproject.toml").read_text())                    # the file reads
    from topopt.config import Settings
    assert Settings.__dataclass_fields__["mesh_method"].default == "sdf"           # the default method of a new run



def test_the_guard_reads_tracked_yaml_files_only(tmp_path):
    """Q1 R1 m3: a YAML that the user made after the checkout (the README copy) is not a shipped file."""
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("needs git")
    repo = tmp_path / "repo"
    (repo / "examples" / "hook").mkdir(parents=True)
    (repo / "examples" / "hook" / "hook.yaml").write_text("settings: {}\n")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "examples"], check=True)
    (repo / "examples" / "hook" / "hook_ratchet.yaml").write_text("settings: {}\nacceptance: {rules: []}\n")      # made after the `git add`
    assert tracked_yaml_files(repo) == [repo / "examples" / "hook" / "hook.yaml"]


def test_the_guard_reads_every_yaml_file_when_there_is_no_git(tmp_path, monkeypatch):
    """Q1 R1 m3: an sdist or a copy of the folder has no git: the files on disk are the shipped files."""
    def no_git(*_a, **_k):
        raise FileNotFoundError("git")
    (tmp_path / "examples" / "a").mkdir(parents=True)
    (tmp_path / "examples" / "a" / "x.yaml").write_text("a: 1\n")
    monkeypatch.setattr("subprocess.run", no_git)
    assert tracked_yaml_files(tmp_path) == [tmp_path / "examples" / "a" / "x.yaml"]


def test_git_ignores_the_copy_that_the_ratchet_section_of_the_readme_asks_for():
    """Q1 R1 m3: the copy is not an untracked file in `git status` (the friction that Q1-14 removes for the acceptance reports)."""
    import shutil
    import subprocess
    if shutil.which("git") is None or subprocess.run(["git", "-C", str(REPO), "rev-parse"], capture_output=True).returncode != 0:
        pytest.skip("needs git and a git repository")
    name = "examples/hook/hook_ratchet.yaml"
    assert name in reference_page("verify.md")
    assert subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", name]).returncode == 0
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "examples"], capture_output=True, text=True).stdout
    assert "examples/hook/hook.yaml" in tracked and "_ratchet.yaml" not in tracked           # no shipped file has an ignored name
