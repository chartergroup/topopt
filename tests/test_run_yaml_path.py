"""One rule for the `yaml_path` of run.json (item R2): a relative path starts at the run folder, from any current folder.

The rule is `topopt.runinfo.run_yaml_path`. Every reader of the key uses it: `topopt refine`, `topopt buckling`, `topopt verify`,
the post step (the region files next to the YAML), the render step, the MCP checkpoint facts. `refine` and `buckling` have
`--yaml FILE` like `verify`. The recording of this repo (`recordings/hook_*`) names its YAML relative to its folder, so a
copy of the repo tree on another machine finds it.
"""
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).parent))

from helpers import recorded_hook_run  # noqa: E402
from topopt import post, runinfo  # noqa: E402
from topopt.cli import main  # noqa: E402
from topopt.config import UserError  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RECORDING = recorded_hook_run()
HOOK_DIR = REPO / "examples" / "hook"
RECORDED_PATH = "../../examples/hook/hook.yaml"     # what the committed run.json holds

needs_recording = pytest.mark.skipif(not (RECORDING / "result" / "final.stl").is_file(),
                                     reason="needs the recorded hook run recordings/hook_*")


# --------------------------------------------------------------------------- #
# The resolver
# --------------------------------------------------------------------------- #

def test_a_relative_path_starts_at_the_run_folder_from_any_current_folder(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "a"
    run_dir.mkdir(parents=True)
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "c.yaml").write_text("name: x\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    run = {"yaml_path": "../../cases/c.yaml"}
    for cwd in (elsewhere, tmp_path, run_dir):
        monkeypatch.chdir(cwd)
        p = runinfo.run_yaml_path(run, run_dir)
        assert p == run_dir / "../../cases/c.yaml" and p.is_file()
        assert p.resolve() == (tmp_path / "cases" / "c.yaml").resolve()
    monkeypatch.chdir(elsewhere)
    rel_run_dir = Path(os.path.relpath(run_dir, elsewhere))      # a run folder given as a relative path
    assert runinfo.run_yaml_path(run, rel_run_dir).is_file()


def test_an_absolute_path_keeps_its_meaning(tmp_path):
    absolute = tmp_path / "somewhere" / "c.yaml"
    run = {"yaml_path": str(absolute)}
    assert runinfo.run_yaml_path(run, tmp_path / "runs" / "a") == absolute
    assert runinfo.run_yaml_path(run) == absolute
    assert runinfo.run_yaml_path(run, None) == absolute


def test_home_path_and_missing_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    p = runinfo.run_yaml_path({"yaml_path": "~/c.yaml"}, tmp_path / "runs" / "a")
    assert p == tmp_path / "home" / "c.yaml"                      # `~` is absolute: not joined to the run folder
    for run in ({}, {"yaml_path": ""}, {"yaml_path": None}, None):
        assert runinfo.run_yaml_path(run, tmp_path) is None
    assert runinfo.run_yaml_path({"yaml_path": "c.yaml"}) == Path("c.yaml")    # no run folder: as stored


def test_resolved_run_is_a_copy_with_the_path_as_text(tmp_path):
    run = {"yaml_path": "../c.yaml", "h": 1.0}
    out = runinfo.resolved_run(run, tmp_path / "a")
    assert out == {"yaml_path": str(tmp_path / "a" / "../c.yaml"), "h": 1.0} and run["yaml_path"] == "../c.yaml"
    no_key = {"h": 1.0}
    assert runinfo.resolved_run(no_key, tmp_path) is no_key


def test_the_old_private_name_is_the_same_function():
    from topopt import feedback
    assert feedback._run_yaml_path is runinfo.run_yaml_path


def test_missing_yaml_text_gives_facts_and_the_option(tmp_path):
    tried = tmp_path / "a" / "../../x.yaml"
    text = runinfo.missing_yaml_text(tried, {"yaml_path": "../../x.yaml"}, "--yaml")
    assert os.path.realpath(tried) in text and ".." not in os.path.realpath(tried) and "'../../x.yaml'" in text and "relative path starts at the run folder" in text
    assert text.endswith("Give its new place with --yaml")
    absolute = runinfo.missing_yaml_text(Path("/x/c.yaml"), {"yaml_path": "/x/c.yaml"}, "--yaml")
    assert absolute == "the YAML file of the run is missing: /x/c.yaml. Give its new place with --yaml"
    assert "given with --yaml is missing: /y.yaml" in runinfo.missing_yaml_text(Path("/y.yaml"), {}, "--yaml", given=True)
    assert "no yaml_path" in runinfo.missing_yaml_text(None, {}, "yaml_path") and "yaml_path" in runinfo.missing_yaml_text(None, {}, "yaml_path")


# --------------------------------------------------------------------------- #
# The committed recording
# --------------------------------------------------------------------------- #

HOME_PATHS = (b"/Users/", b"/home/", b"C:\\Users")           # the start of a home folder: macOS, Linux, Windows


def home_paths_in_tracked_files(root: Path, tops=("recordings", "examples")) -> list[str] | None:
    """The tracked text files under `tops` that hold a home path, as `name: needle` lines (the publish gate).

    The files are those of `git ls-files`, so an untracked local run (`examples/**/runs/`) does not count, and every tracked
    text file counts (run.json, logs, .html, .yaml, .md, .csv, .txt ...). A file with a NUL byte is binary and is not read.
    Return None if there is no git or no `.git` folder (an sdist), so that the caller can skip.
    """
    import subprocess
    if shutil.which("git") is None or not (root / ".git").exists():
        return None
    done = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", *tops], capture_output=True)
    if done.returncode != 0:
        return None
    found = []
    for name in sorted(n for n in done.stdout.decode("utf-8", "replace").split("\0") if n):
        path = root / name
        if not path.is_file():
            continue                                                   # a tracked file that is deleted in the work tree
        data = path.read_bytes()
        if b"\0" in data:
            continue
        found += [f"{name}: {needle.decode()}" for needle in HOME_PATHS if needle in data]
    return found


def test_no_tracked_file_of_recordings_or_examples_holds_a_home_path():
    """`recordings/` and `examples/` must not show a user name (a publish gate). Every tracked text file, not the disk."""
    found = home_paths_in_tracked_files(REPO)
    if found is None:
        pytest.skip("needs git and the .git folder (the tracked files); this is a tree without them")
    assert found == []


@pytest.fixture
def scratch_repo(tmp_path):
    """A git repo in the temporary folder with a tracked recording and an example (nothing is committed: the index counts)."""
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("needs git")
    repo = tmp_path / "repo"
    (repo / "recordings" / "r1" / "result").mkdir(parents=True)
    (repo / "examples" / "hook").mkdir(parents=True)
    (repo / "recordings" / "r1" / "run.json").write_text('{"yaml_path": "../../examples/hook/hook.yaml"}\n')
    (repo / "examples" / "hook" / "hook.yaml").write_text("name: x\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    return repo


def add(repo: Path, name: str, data: bytes, track: bool = True) -> None:
    import subprocess
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if track:
        subprocess.run(["git", "-C", str(repo), "add", "--", name], check=True)


def test_the_gate_passes_a_clean_tree_and_ignores_an_untracked_local_run(scratch_repo):
    assert home_paths_in_tracked_files(scratch_repo) == []
    add(scratch_repo, "examples/hook/runs/local/run.json", b'{"yaml_path": "/Users/x/hook.yaml"}', track=False)   # a local run: untracked
    add(scratch_repo, "examples/hook/runs/local/result/refine.html", b"<p>/home/x/run</p>", track=False)
    assert home_paths_in_tracked_files(scratch_repo) == []


@pytest.mark.parametrize("name, data, needle", [
    ("recordings/r1/run.json", b'{"yaml_path": "/Users/x/hook.yaml"}', "/Users/"),
    ("recordings/r1/result/refine.html", b"<script>var META = {\"source_stl\": \"/home/x/final.stl\"}</script>", "/home/"),
    ("examples/hook/notes.md", b"see C:\\Users\\x\\hook.yaml", "C:\\Users"),
    ("recordings/r1/run.log", b"started in /home/x/runs\n", "/home/"),
    ("examples/hook/data.csv", b"a,b\n/Users/x,1\n", "/Users/"),
    ("examples/hook/notes.txt", b"/home/x", "/home/"),
    ("examples/hook/other.yaml", b"yaml_path: /Users/x/a.yaml\n", "/Users/"),
])
def test_the_gate_fails_for_a_planted_tracked_text_file(scratch_repo, name, data, needle):
    add(scratch_repo, name, data)
    assert home_paths_in_tracked_files(scratch_repo) == [f"{name}: {needle}"]


def test_the_gate_does_not_read_a_binary_file_and_gives_none_without_git_folder(scratch_repo, tmp_path):
    add(scratch_repo, "recordings/r1/result/final.stl", b"\0\0/Users/x\0")
    assert home_paths_in_tracked_files(scratch_repo) == []
    bare = tmp_path / "no_git"
    (bare / "recordings").mkdir(parents=True)
    assert home_paths_in_tracked_files(bare) is None


@needs_recording
def test_the_recording_names_its_yaml_relative_to_its_folder():
    run = json.loads((RECORDING / "run.json").read_text())
    assert run["yaml_path"] == RECORDED_PATH
    p = runinfo.run_yaml_path(run, RECORDING)
    assert p.is_file() and p.resolve() == (HOOK_DIR / "hook.yaml").resolve()


def portable_tree(dst: Path, run_name: str = "hook_recorded", yaml_ok: bool = True) -> tuple[Path, Path]:
    """A copy of the recording and of the example outside the repo, placed as `yaml_path` says.

    `dst/recordings/<run_name>` has the files that refine, buckling and verify read (a copy of the 15 MB folder is not needed),
    `dst/examples/hook` has the YAML and the region STL files. Return (run folder, YAML)."""
    run = dst / "recordings" / run_name
    (run / "result").mkdir(parents=True)
    (run / "snapshots").mkdir()
    for name in ("run.json", "labels.bin", "status.json"):
        shutil.copy(RECORDING / name, run / name)
    for name in ("final.stl", "metrics.json"):
        shutil.copy(RECORDING / "result" / name, run / "result" / name)
    shutil.copy(RECORDING / "snapshots" / "it_0080.bin", run / "snapshots" / "it_0080.bin")
    if yaml_ok:
        shutil.copytree(HOOK_DIR, dst / "examples" / "hook", ignore=shutil.ignore_patterns("__pycache__"))
    return run, dst / "examples" / "hook" / "hook.yaml"


class Found(UserError):
    """What a stub raises after the YAML was found and the grid matched: no solve is needed to see that."""


@pytest.fixture
def stop_after_the_problem(monkeypatch):
    """Replace the stress solve and the buckling check by a stop that keeps the `Problem` they get."""
    got = []

    def stress(problem, *a, **k):
        got.append(problem)
        raise Found("stop: the YAML was found")

    def check(run_dir, problem, *a, **k):
        got.append(problem)
        return {"status": "done"}

    monkeypatch.setattr("topopt.refine.stress.compute_stress", stress)
    monkeypatch.setattr("topopt.buckling_voxel.run_check", check)
    return got


@needs_recording
def test_refine_finds_the_yaml_of_a_copied_recording_from_another_folder(tmp_path, monkeypatch, capsys, stop_after_the_problem):
    run, _yaml = portable_tree(tmp_path / "tree")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(["refine", str(run)]) == 1                       # our stub stops the solve: the problem was built
    err = capsys.readouterr().err
    assert "stop: the YAML was found" in err and "missing" not in err
    (problem,) = stop_after_the_problem
    assert tuple(problem.shape) == (68, 30, 55) and problem.h == 1.2     # the grid of the recording: it matched labels.bin


@needs_recording
def test_buckling_finds_the_yaml_of_a_copied_recording_from_another_folder(tmp_path, monkeypatch, stop_after_the_problem):
    run, _yaml = portable_tree(tmp_path / "tree")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(["buckling", str(run)]) == 0
    (problem,) = stop_after_the_problem
    assert tuple(problem.shape) == (68, 30, 55)


@needs_recording
def test_verify_finds_the_yaml_of_a_copied_recording_from_another_folder(tmp_path, monkeypatch):
    from topopt.verify.run import read_run
    run, yaml = portable_tree(tmp_path / "tree")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    info = read_run(run)
    assert info.yaml_path == yaml.resolve() and info.cfg.name == "tshirt_hook"


@needs_recording
@pytest.mark.parametrize("command", ["refine", "buckling"])
def test_yaml_option_gives_the_place_of_the_yaml(tmp_path, monkeypatch, capsys, stop_after_the_problem, command):
    """A run folder moved to a place where the relative path is not valid: `--yaml` gives the YAML (like `verify --yaml`)."""
    run, _ = portable_tree(tmp_path / "tree", yaml_ok=False)
    spot = tmp_path / "somewhere" / "hook.yaml"
    shutil.copytree(HOOK_DIR, tmp_path / "somewhere")
    monkeypatch.chdir(tmp_path)
    assert not runinfo.run_yaml_path(json.loads((run / "run.json").read_text()), run).is_file()
    code = main([command, str(run), "--yaml", str(spot)])
    assert code == (1 if command == "refine" else 0)
    assert "missing" not in capsys.readouterr().err
    (problem,) = stop_after_the_problem
    assert tuple(problem.shape) == (68, 30, 55)


@needs_recording
@pytest.mark.parametrize("command", ["refine", "buckling"])
def test_a_missing_yaml_gives_the_path_it_tried_and_the_option(tmp_path, monkeypatch, capsys, command):
    run, _ = portable_tree(tmp_path / "tree", yaml_ok=False)
    monkeypatch.chdir(tmp_path)
    assert main([command, str(run)]) == 1
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("error: the YAML file of the run is missing: ")
    assert os.path.realpath(run / RECORDED_PATH) in lines[0]                       # the path that was tried (absolute)
    assert f"'{RECORDED_PATH}'" in lines[0] and "starts at the run folder" in lines[0]
    assert "Give its new place with --yaml" in lines[0]
    assert main([command, str(run), "--yaml", str(tmp_path / "none.yaml")]) == 1    # a wrong --yaml is a fact too
    err = capsys.readouterr().err
    assert f"the YAML file given with --yaml is missing: {os.path.realpath(tmp_path / 'none.yaml')}" in err


@needs_recording
def test_verify_text_is_the_same_text(tmp_path):
    from topopt.verify.run import read_run
    run, _ = portable_tree(tmp_path / "tree", yaml_ok=False)
    with pytest.raises(UserError) as e:
        read_run(run)
    assert str(e.value) == runinfo.missing_yaml_text(run / RECORDED_PATH, json.loads((run / "run.json").read_text()), "--yaml")
    with pytest.raises(UserError, match="given with --yaml is missing"):
        read_run(run, yaml_path=tmp_path / "none.yaml")


def test_a_run_json_without_yaml_path_says_so(tmp_path, capsys):
    from topopt.verify.run import read_run
    run = tmp_path / "r"
    (run / "result").mkdir(parents=True)
    (run / "result" / "final.stl").write_bytes(b"x")
    (run / "run.json").write_text("{}")
    with pytest.raises(UserError, match="run.json has no yaml_path"):
        read_run(run)


@pytest.mark.parametrize("command", ["verify", "refine", "buckling"])
def test_options_are_in_the_help_like_verify(capsys, command):
    with pytest.raises(SystemExit) as e:
        main([command, "--help"])
    assert e.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "--yaml FILE the problem YAML, if the path in run.json is not valid" in text


# --------------------------------------------------------------------------- #
# The other readers
# --------------------------------------------------------------------------- #

def hole_run(run_dir: Path, case: Path):
    """A run folder (`run_dir`) of a plate with a round hole as a keep-out STL (`case/hole.stl`); its run.json holds `yaml_path`."""
    from test_post import make_hole_run
    run_dir.mkdir(parents=True, exist_ok=True)
    run, _labels = make_hole_run(run_dir)
    h, org = run["h"], np.asarray(run["origin"])
    cx, cz = org[0] + 20 * h, org[2] + 12 * h
    cyl = trimesh.creation.cylinder(radius=3.5, height=60, sections=64)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    cyl.apply_translation([cx, org[1] + 12 * h, cz])
    case.mkdir(parents=True, exist_ok=True)
    cyl.export(case / "hole.stl")
    (case / "x.yaml").write_text("name: x\n")
    run["regions"] = {"keep_out": ["hole.stl"]}
    return run, cx, cz


def test_the_post_step_finds_the_keep_out_next_to_a_relative_yaml(tmp_path, monkeypatch):
    run_dir, case = tmp_path / "runs" / "a", tmp_path / "cases"
    run, cx, cz = hole_run(run_dir, case)
    run["yaml_path"] = os.path.relpath(case / "x.yaml", run_dir)           # ../../cases/x.yaml
    (run_dir / "run.json").write_text(json.dumps(run))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    m = post.finish(run_dir)
    assert m["watertight"] is True
    v = trimesh.load(run_dir / "result" / "final.stl").vertices
    d = np.hypot(v[:, 0] - cx, v[:, 2] - cz)
    assert abs(d[d < 3.7].min() - 3.5) < 0.05           # the hole has the radius of the STL (the label hole is a 4.8 mm square)


def test_the_post_step_with_the_same_run_json_and_an_absolute_path_is_the_same(tmp_path, monkeypatch):
    run_dir, case = tmp_path / "runs" / "a", tmp_path / "cases"
    run, _cx, _cz = hole_run(run_dir, case)
    run["yaml_path"] = str(case / "x.yaml")
    (run_dir / "run.json").write_text(json.dumps(run))
    monkeypatch.chdir(tmp_path)
    assert post.finish(run_dir)["watertight"] is True
    abs_stl = (run_dir / "result" / "final.stl").read_bytes()
    run["yaml_path"] = os.path.relpath(case / "x.yaml", run_dir)
    (run_dir / "run.json").write_text(json.dumps(run))
    post.finish(run_dir)
    assert (run_dir / "result" / "final.stl").read_bytes() == abs_stl           # same mesh, byte for byte


def test_render_finds_the_region_files_next_to_a_relative_yaml(tmp_path, monkeypatch):
    from topopt import render_input
    run_dir, case = tmp_path / "runs" / "a", tmp_path / "cases"
    (run_dir / "result").mkdir(parents=True)
    (run_dir / "result" / "final.stl").write_bytes(b"x")
    case.mkdir()
    for n in ("d.stl", "f.stl"):
        (case / n).write_bytes(b"x")
    (case / "c.yaml").write_text("name: x\n")
    (run_dir / "run.json").write_text(json.dumps({"yaml_path": "../../cases/c.yaml",
                                                  "regions": {"design": "d.stl", "fixed": ["f.stl"], "keep_out": [], "load": {}}}))
    monkeypatch.chdir(tmp_path)
    items, missing, notes = render_input._run_inputs(run_dir)
    assert [r for _p, r in items] == ["part", "design", "fixed"] and not missing and not notes
    (case / "f.stl").unlink()
    items, missing, notes = render_input._run_inputs(run_dir)
    assert [m["role"] for m in missing] == ["fixed"] and missing[0]["path"] == str(case / "f.stl")   # shown without `..`
    assert ".." not in missing[0]["path"]


def test_interface_specs_use_the_folder_of_the_resolved_yaml(tmp_path, monkeypatch):
    from topopt import interfaces
    run_dir, case = tmp_path / "runs" / "a", tmp_path / "cases"
    run_dir.mkdir(parents=True)
    case.mkdir()
    trimesh.creation.box(extents=[1, 1, 1]).export(case / "boss.stl")
    run = {"yaml_path": "../../cases/c.yaml",
           "interfaces": [{"name": "boss", "kind": "stl", "op": "union", "file": "/nowhere/boss.stl"}]}
    monkeypatch.chdir(tmp_path)
    (spec,) = interfaces.specs_from_run(runinfo.resolved_run(run, run_dir))
    assert Path(spec.path).is_file() and Path(spec.path).name == "boss.stl"


def test_mcp_checkpoint_facts_give_the_yaml_path_as_a_usable_argument(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from topopt_mcp import server
    monkeypatch.setattr("topopt.checkpoint.describe", lambda d: {"file": "checkpoint.npz"})
    run_dir = tmp_path / "runs" / "a"
    got = server._checkpoint_facts(run_dir, {"yaml_path": "../../cases/c.yaml"})
    assert got["yaml_path"] == os.path.realpath(run_dir / "../../cases/c.yaml") and ".." not in got["yaml_path"]   # no `..` in the text
    assert server._checkpoint_facts(run_dir, {"yaml_path": "/x/c.yaml"})["yaml_path"] == "/x/c.yaml"
    assert server._checkpoint_facts(run_dir, {})["yaml_path"] is None


def test_the_keep_out_warning_of_the_post_step_shows_the_path_without_dots(tmp_path, caplog):
    stl = tmp_path / "runs" / "a" / ".." / ".." / "cases" / "hole.stl"
    verts, faces = np.zeros((3, 3)), np.array([[0, 1, 2]])
    with caplog.at_level("WARNING"):
        out = post.subtract_keep_out(verts, faces, [stl])
    assert out[2]["keep_out_cut"] is False
    assert f"no exact cut: {tmp_path / 'cases' / 'hole.stl'}" in caplog.text and ".." not in caplog.text


def feedback_case(tmp_path):
    """The small case of test_feedback with a run whose run.json holds a RELATIVE yaml_path (`../../in/<name>.yaml`)."""
    import test_feedback as tf
    ypath = tf.make_case(tmp_path / "in")
    run = tf.fake_run(tmp_path / "runs" / "a", ypath)
    doc = json.loads((run / "run.json").read_text())
    doc["yaml_path"] = os.path.relpath(ypath, run)
    (run / "run.json").write_text(json.dumps(doc))
    return tf, run, ypath


def test_feedback_shows_the_yaml_path_without_dots_and_reads_the_file(tmp_path, monkeypatch):
    from topopt import feedback
    tf, run, ypath = feedback_case(tmp_path)
    monkeypatch.chdir(tmp_path)
    state = feedback._yaml_state(json.loads((run / "run.json").read_text()), None, run)
    assert state["path"] == str(ypath) and state["exists"] is True                  # found, and shown as an absolute path
    ypath.rename(ypath.with_name("moved.yaml"))
    result = feedback.make_hints(run, tf.marks_doc(tf.mark(1, "remove", [(10, 0, 6)])), tmp_path / "hints")
    assert result["ok"] is False                                                   # the YAML is not at the place any more
    assert f"the YAML of the run is not on disk: {ypath}." in result["error"] and ".." not in result["error"]


def test_check_hints_script_finds_the_yaml_of_a_relative_yaml_path_from_another_folder(tmp_path, monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_hints", REPO / "tools" / "check_hints.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    tf, run, _ypath = feedback_case(tmp_path)
    marks = tmp_path / "marks.json"
    marks.write_text(json.dumps(tf.marks_doc(tf.mark(1, "remove", [(10, 3, 6)], mirror=True))))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert script.main([str(run), str(marks), str(tmp_path / "hints"), "--make"]) == 0
    text = capsys.readouterr().out
    assert "RESULT: OK" in text and "[FAIL]" not in text and "unchanged" in text


@needs_recording
def test_check_final_stl_script_finds_the_region_files_of_a_relative_yaml_path_from_another_folder(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import check_final_stl
    finally:
        sys.path.remove(str(REPO / "tools"))
    run, _yaml = portable_tree(tmp_path / "tree")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    out = check_final_stl.check_run(run)
    assert out["regions"] is not None and set(out["regions"]) == {"fixed:wall_plate.stl", "load:tip"}, out["problems"]    # the files are found
    assert set(out["keep_out"]) == {"screw_hole_top.stl", "screw_hole_bottom.stl"}
    assert all(v["intersect_mm3"] <= 1e-3 for v in out["keep_out"].values())       # the exact cut: nothing of the mesh is in a hole
    (tmp_path / "tree" / "examples" / "hook" / "wall_plate.stl").unlink()
    out = check_final_stl.check_run(run)                                           # the region file is not found: no region numbers
    assert out["regions"] is None


# --------------------------------------------------------------------------- #
# Replay of the recording
# --------------------------------------------------------------------------- #

@needs_recording
def test_replay_of_the_recording_works_and_serves_the_relative_path():
    """The replay server reads the committed folder. `/api/run` holds the relative `yaml_path`: no home path goes to the browser."""
    from topopt import dashboard_server as ds

    class Clock:
        t = 0.0

        def __call__(self):
            return self.t

    server = ds.start_in_thread(RECORDING, port=0, replay=True, speed=1.0)
    clock = Clock()
    server.source.replay_state.now = clock
    base = server.url.rstrip("/")
    try:
        def jget(path):
            with urllib.request.urlopen(base + path) as r:
                return json.loads(r.read())
        clock.t = 100.0
        urllib.request.urlopen(base + "/api/replay/start").read()
        run = jget("/api/run")
        assert run["name"] == "tshirt_hook" and run["yaml_path"] == RECORDED_PATH
        assert "/Users/" not in json.dumps(run) and "/home/" not in json.dumps(run)
        assert jget("/api/status")["mode"] == "replay"
        clock.t = 100.0 + 200.0                                               # past the end (147.63 s)
        assert jget("/api/status")["state"] == "done"
        assert jget("/api/history")[-1]["iteration"] == 80
        assert jget("/api/metrics")["iterations"] == 80
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# Review M1: --yaml with a YAML that does not fit the run gives facts (the rule of `verify`)
# --------------------------------------------------------------------------- #

NOTE = "note: the YAML differs from run.json: "


def forces_x10(yaml_file: Path) -> Path:
    """The hook YAML with the forces times ten (the same STL files, the same grid)."""
    text = yaml_file.read_text().replace("[0.0, 0.0, -1.96]", "[0.0, 0.0, -19.6]").replace("[20.89, 5.21, -20.89]", "[208.9, 52.1, -208.9]")
    out = yaml_file.with_name("hook_forces10.yaml")
    out.write_text(text)
    return out


@needs_recording
def test_refine_notes_a_yaml_with_other_forces_and_the_same_yaml_gives_no_note(tmp_path, monkeypatch, capsys, stop_after_the_problem):
    run, yaml = portable_tree(tmp_path / "tree")
    other = forces_x10(yaml)
    monkeypatch.chdir(tmp_path)
    assert main(["refine", str(run), "--yaml", str(other)]) == 1            # the stub stops the solve, after the notes
    err = capsys.readouterr().err
    assert err.count(NOTE) == 2
    assert f"{NOTE}case hang, region tip: force [0.0, 0.0, -1.96] in run.json, [0.0, 0.0, -19.6] in the YAML" in err
    assert f"{NOTE}case pull_off, region tip: force [20.89, 5.21, -20.89] in run.json, [208.9, 52.1, -208.9] in the YAML" in err
    assert main(["refine", str(run), "--yaml", str(yaml)]) == 1
    assert NOTE not in capsys.readouterr().err


@needs_recording
def test_buckling_notes_a_yaml_with_other_forces_and_the_same_yaml_gives_no_note(tmp_path, monkeypatch, capsys, stop_after_the_problem):
    run, yaml = portable_tree(tmp_path / "tree")
    other = forces_x10(yaml)
    monkeypatch.chdir(tmp_path)
    assert main(["buckling", str(run), "--yaml", str(other)]) == 0
    out = capsys.readouterr().out
    assert out.count(NOTE) == 2 and "force [0.0, 0.0, -1.96] in run.json, [0.0, 0.0, -19.6] in the YAML" in out
    assert out.count(NOTE) == (run / "run.log").read_text().count(NOTE)      # the notes are in run.log too
    assert main(["buckling", str(run), "--yaml", str(yaml)]) == 0
    assert NOTE not in capsys.readouterr().out


def amg_run(tmp_path: Path):
    """A finished small run with `solver: amg`: `topopt buckling` gives a fact and no solve (the key `buckling` is written)."""
    from test_buckling import run_cli
    code, out, _ = run_cli(tmp_path, solver="amg")
    assert code == 0
    return out, tmp_path / "c.yaml"


def test_buckling_stores_the_differences_in_metrics_json(tmp_path, capsys):
    import yaml
    out, case = amg_run(tmp_path)
    doc = yaml.safe_load(case.read_text())
    for c in doc["load_cases"]:
        for f in c["forces"]:
            f["force"] = [10 * x for x in f["force"]]
    other = tmp_path / "forces10.yaml"
    other.write_text(yaml.safe_dump(doc))
    assert main(["buckling", str(out), "--yaml", str(other)]) == 3                   # amg: a fact, exit 3
    key = json.loads((out / "result" / "metrics.json").read_text())["buckling"]
    assert key["status"] == "skipped"
    cmp = key["yaml_vs_run"]
    assert cmp["matches"] is False and len(cmp["differences"]) == 2 and "force" in cmp["differences"][0]
    assert "E, nu" in cmp["note"] and set(cmp) == {"matches", "differences", "note"}     # no path in the record
    assert str(tmp_path) not in json.dumps(cmp)
    assert capsys.readouterr().out.count(NOTE) == 2
    assert main(["buckling", str(out)]) == 3                                        # the YAML of the run: no difference
    cmp = json.loads((out / "result" / "metrics.json").read_text())["buckling"]["yaml_vs_run"]
    assert cmp["matches"] is True and cmp["differences"] == []
    assert NOTE not in capsys.readouterr().out


def test_refine_page_data_holds_the_differences(tmp_path, capsys):
    import re
    import yaml
    out, case = amg_run(tmp_path)

    def meta():
        m = re.search(r"var META = (\{.*?\});\n", (out / "result" / "refine.html").read_text(encoding="utf-8"))
        return json.loads(m.group(1))
    assert "yaml_vs_run" not in meta()                                              # the page of the run itself: no comparison
    doc = yaml.safe_load(case.read_text())
    doc["load_cases"][0]["weight"] = 7.0
    other = tmp_path / "weight7.yaml"
    other.write_text(yaml.safe_dump(doc))
    capsys.readouterr()
    assert main(["refine", str(out), "--yaml", str(other)]) == 0
    cap = capsys.readouterr()
    assert cap.out.strip().splitlines()[-1] == str(out / "result" / "refine.html")       # stdout is still the path
    assert cap.err.count(NOTE) == 1 and "weight" in cap.err
    cmp = meta()["yaml_vs_run"]
    assert cmp["matches"] is False and "weight" in cmp["differences"][0] and str(tmp_path) not in json.dumps(cmp)
    assert main(["refine", str(out)]) == 0
    assert meta()["yaml_vs_run"]["matches"] is True and NOTE not in capsys.readouterr().err


def test_the_note_lines_are_one_text_for_verify_refine_and_buckling():
    from topopt.verify.report import yaml_note_lines
    assert yaml_note_lines({"differences": ["a", "b"]}) == [NOTE + "a", NOTE + "b"]
    assert yaml_note_lines({"matches": True, "differences": []}) == [] and yaml_note_lines({}) == []


@needs_recording
def test_verify_prints_the_same_note_lines_as_refine(tmp_path, monkeypatch, capsys, stop_after_the_problem):
    """`topopt verify --estimate-only` (the lines of verify) and `topopt refine` (the lines of this round) give the same notes."""
    run, yaml = portable_tree(tmp_path / "tree")
    other = forces_x10(yaml)
    monkeypatch.chdir(tmp_path)
    assert main(["verify", str(run), "--yaml", str(other), "--estimate-only"]) == 0
    verify_notes = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("note:")]
    assert len(verify_notes) == 2
    assert main(["refine", str(run), "--yaml", str(other)]) == 1                     # the stub stops the solve, after the notes
    assert [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("note:")] == verify_notes


# --------------------------------------------------------------------------- #
# Review m1: the grid-mismatch line
# --------------------------------------------------------------------------- #

@needs_recording
@pytest.mark.parametrize("command", ["refine", "buckling"])
def test_a_given_yaml_with_another_grid_says_it_does_not_fit_and_shows_the_path_once(tmp_path, monkeypatch, capsys, command):
    run, yaml = portable_tree(tmp_path / "tree")
    monkeypatch.chdir(tmp_path)
    short = yaml.with_name("hook_short.yaml")
    assert main([command, str(run), "--yaml", str(short)]) == 1
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    line = lines[0]
    assert "does not fit the grid of the run" in line and "changed after the run" not in line
    assert line.count(os.path.realpath(short)) == 1 and ".." not in line and "(file:" not in line
    assert "The run has the grid [68, 30, 55] at 1.2 mm" in line and "the YAML gives the grid [41, 18, 33] at 2 mm" in line
    assert "Option: give the YAML that made the run with --yaml" in line


@needs_recording
def test_a_changed_yaml_at_the_stored_path_says_changed_and_shows_the_path_once(tmp_path, monkeypatch, capsys):
    run, yaml = portable_tree(tmp_path / "tree")
    yaml.write_text(yaml.read_text().replace("voxel_size: 1.2", "voxel_size: 1.0"))
    monkeypatch.chdir(tmp_path)
    assert main(["refine", str(run)]) == 1
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1 and "changed after the run" in lines[0] and "given with" not in lines[0]
    assert lines[0].count(os.path.realpath(yaml)) == 1 and ".." not in lines[0] and "(file:" not in lines[0]
    assert "at 1.2 mm" in lines[0] and "at 1 mm" in lines[0] and "Option: give another YAML with --yaml, or run again" in lines[0]


@needs_recording
def test_labels_that_differ_are_counted(tmp_path, monkeypatch, capsys):
    """Same grid, other STL region (a moved keep-out hole): the line gives the number of voxel labels that differ."""
    run, yaml = portable_tree(tmp_path / "tree")
    plate = yaml.parent / "screw_hole_top.stl"
    mesh = trimesh.load(plate)
    mesh.apply_translation([0.0, 1.5, 0.0])                      # a shift of a keep-out region inside the part: labels move, the grid stays
    mesh.export(plate)
    monkeypatch.chdir(tmp_path)
    assert main(["refine", str(run)]) == 1
    line = capsys.readouterr().err.strip()
    import re
    assert line.count("grid [68, 30, 55] at 1.2 mm, origin [-1, -17.5, -30]") == 2, line     # the same grid and origin twice
    assert re.search(r"; [1-9]\d* of 112200 voxel labels differ\. Option", line), line
