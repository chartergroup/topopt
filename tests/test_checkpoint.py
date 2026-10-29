"""Tests for checkpoint, resume, stop rules and the verdict of a run (D1b, CONTRACTS.md section 24.2).

The runs that die are child processes: `tests/kill_run.py` sends SIGKILL to itself at a chosen point (no handler runs).
The child processes run with 1 thread, so a direct solver (PARDISO, Cholesky) gives the same bytes in each run. A resumed run
must then equal the run that was not stopped, byte for byte. AMG is not reproducible from run to run (the spectral radius
estimate of pyamg starts from random numbers), so the AMG test compares numbers with a tolerance.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from topopt import checkpoint as C
from topopt.cli import main
from topopt.config import load_config
from topopt.solver_choice import resolve_solver

from test_optimize import make_case
from test_stress_report import stress_yaml

KILL = Path(__file__).with_name("kill_run.py")
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
DIRECT = resolve_solver("auto") != "amg"
needs_direct = pytest.mark.skipif(not DIRECT, reason="needs a direct solver (PARDISO or Cholesky): AMG is not reproducible")
TIMING = {"run_time_s", "thresholded_solve_s", "resume_points", "t_iter_s", "elapsed_s", "written_at", "t_s",   # differ from run to run (t_s: the timings of the mesh step, C1)
          "robust_s", "robust_parts_s", "run_s"}      # the time of the robust part and the run time of the robust record (P4b-4: the robust stop test compares them)


def child(point, *args, timeout=600):
    """`topopt <args>` in a child process that dies at `point` (`never:-1`: it does not die). Return the CompletedProcess."""
    return subprocess.run([sys.executable, str(KILL), point, *args], env=ENV1, capture_output=True, text=True, timeout=timeout)


def run_args(yaml_path, out, *extra):
    return ["run", str(yaml_path), "--out", str(out), "--quiet", *extra]


def finish(yaml_path, out, *extra, point="never:-1"):
    done = child(point, *run_args(yaml_path, out, *extra))
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-2500:]
    return done


def killed(yaml_path, out, point, *extra):
    done = child(point, *run_args(yaml_path, out, *extra))
    assert done.returncode == -9, f"the child did not die at {point}: exit {done.returncode}\n" + done.stderr[-2500:]


def resume(yaml_path, out, *extra):
    return finish(yaml_path, out, "--resume", *extra)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k not in TIMING}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def history(d):
    return [json.loads(line) for line in (Path(d) / "history.jsonl").read_text().splitlines()]


def log(d):
    return (Path(d) / "run.log").read_text()


def same_run(a, b, exact=True, rtol=1e-6, run_json_drop=()):
    """The run folder `b` equals `a`, apart from the timing. `exact`: byte for byte. Else the numbers are close.
    `run_json_drop`: keys of run.json that are not compared (P4b-5: `continued_from`, which only a continuation has)."""
    a, b = Path(a), Path(b)
    ha, hb = history(a), history(b)
    assert [r["iteration"] for r in ha] == [r["iteration"] for r in hb]
    names = sorted(p.name for p in (a / "snapshots").iterdir())
    assert names == sorted(p.name for p in (b / "snapshots").iterdir())
    if exact:
        assert clean(ha) == clean(hb)
        for n in names:
            assert (a / "snapshots" / n).read_bytes() == (b / "snapshots" / n).read_bytes(), n
        for f in ("result/final.stl", "result/preview.bin", "labels.bin"):
            assert (a / f).read_bytes() == (b / f).read_bytes(), f
        ma, mb = (json.loads((d / "result" / "metrics.json").read_text()) for d in (a, b))
        assert clean(ma) == clean(mb)
    else:
        for x, y in zip(ha, hb):
            for key in ("compliance", "volume_fraction"):
                assert abs(x[key] - y[key]) <= rtol * abs(x[key]), (x["iteration"], key, x[key], y[key])
    ja, jb = (json.loads((d / "run.json").read_text()) for d in (a, b))
    ja.pop("started_at"), jb.pop("started_at")
    for key in run_json_drop:
        ja.pop(key, None), jb.pop(key, None)
    assert ja == jb


def tree_state(d):
    """The files of a folder with their bytes, except the job files (a job writes its own record)."""
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file() and not p.name.startswith("job.")}


def edit(path, fn):
    data = yaml.safe_load(Path(path).read_text())
    fn(data)
    Path(path).write_text(yaml.safe_dump(data))


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """The cantilever (1 mm voxels, 24 passes), and the run that was not stopped (1 thread)."""
    tmp = tmp_path_factory.mktemp("ck_case")
    yaml_path = make_case(tmp, h=1.0, iterations=24)
    base = tmp / "base"
    finish(yaml_path, base)
    return yaml_path, base


@pytest.fixture
def work(case, tmp_path):
    """A copy of the case in a new folder (the YAML and the STL files): a test can edit it."""
    import shutil
    src = case[0].parent
    dst = tmp_path / "case"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("base", "r*", "*.log"))
    return dst / case[0].name


# --- the checkpoint file -------------------------------------------------------------------

def meta_of(it, **extra):
    return dict({"kind": C.KIND, "format": C.FORMAT, "iteration": it, "n_design": 4, "n_iterations": 10, "complete": False,
                 "mma": {"it": it + 1, "f0_ref": 1.0, "rows": {}, "move": None, "x": "same"},
                 "inputs": {"grid": {"shape": [2, 2, 1], "n_design": 4}},
                 "topopt_version": "t", "written_at": "now", "elapsed_s": 1.5}, **extra)


def arrays_of(v):
    """The core arrays of a checkpoint with 4 design voxels on a grid of 2 x 2 x 1."""
    return {"x": np.full(4, float(v)), "mma_low": np.zeros((4, 1)), "mma_upp": np.ones((4, 1)),
            "mma_xold1": np.full(4, float(v)), "mma_xold2": np.full(4, float(v)), "design_mask": np.packbits(np.ones(4, bool))}


def test_write_read_rotate(tmp_path):
    size = C.write_checkpoint(tmp_path, arrays_of(1), meta_of(1))
    assert size == (tmp_path / C.CHECKPOINT_FILE).stat().st_size and not (tmp_path / C.TMP_FILE).exists()
    assert not (tmp_path / C.PREV_FILE).exists()
    C.write_checkpoint(tmp_path, arrays_of(2), meta_of(2))
    ck = C.read_checkpoint(tmp_path)
    assert (ck.file, ck.iteration, ck.facts, ck.bad) == (C.CHECKPOINT_FILE, 2, [], {})
    assert ck.arrays["x"].tolist() == [2.0] * 4 and ck.arrays["mma_low"].shape == (4, 1)
    assert C.read_checkpoint(tmp_path).meta["elapsed_s"] == 1.5
    prev = C._load_file(tmp_path / C.PREV_FILE)[0]
    assert prev["iteration"] == 1                                    # the pass before is the second file
    C.write_checkpoint(tmp_path, arrays_of(3), meta_of(3))
    assert C._load_file(tmp_path / C.PREV_FILE)[0]["iteration"] == 2   # two files only


def test_the_arrays_of_the_row_states_go_to_the_npz_and_come_back_exact_and_equal_arrays_share_one_name(tmp_path):
    """#155: `pack_rows` takes each numpy array of a row state out of the JSON; `unpack_rows` puts back a copy with the same bytes."""
    v, w = np.random.default_rng(1).standard_normal(7), np.random.default_rng(2).standard_normal(7)
    rows = {"buckling:a": {"n": 3, "v0": v, "last_info": {"x": 1.5}}, "buckling:b": {"v0": w}, "stress:c": {"c": 2.0, "beta": 1.0}, "odd": 5}
    before = {"buckling:a": {"n": 2, "v0": v.copy(), "last_info": None}, "buckling:b": {"v0": None}, "stress:c": {"c": 2.0, "beta": 1.0}, "odd": 5}
    arrays = {"x": np.zeros(4)}
    packed, packed_before = C.pack_rows(rows, arrays), C.pack_rows(before, arrays)
    names = sorted(n for n in arrays if n.startswith(C.ROW_ARRAY))
    assert names == ["rowvec_0", "rowvec_1"]                                         # `v` in both is stored once
    assert packed["buckling:a"] == {"n": 3, "v0": {C.ARRAY_REF: "rowvec_0"}, "last_info": {"x": 1.5}}
    assert packed["buckling:b"] == {"v0": {C.ARRAY_REF: "rowvec_1"}} and packed_before["buckling:a"]["v0"] == {C.ARRAY_REF: "rowvec_0"}
    assert packed_before["buckling:b"] == {"v0": None} and packed["stress:c"] == rows["stress:c"] and packed["odd"] == 5
    assert isinstance(rows["buckling:a"]["v0"], np.ndarray)                           # the argument is not changed
    assert C.pack_rows({"stress:c": {"c": 2.0}}, {}) == {"stress:c": {"c": 2.0}} and C.pack_rows(None, {}) is None
    assert sorted(C.row_array_refs(packed)) == ["rowvec_0", "rowvec_1"] and C.row_array_refs({"a": 1, "b": {"c": None}}) == []
    C.write_checkpoint(tmp_path, {**arrays_of(1), **{n: arrays[n] for n in names}},
                       meta_of(1, mma={"it": 2, "f0_ref": 1.0, "rows": packed, "move": None, "x": "same"}, rows_before=packed_before))
    ck = C.read_checkpoint(tmp_path)
    assert ck.facts == [] and ck.arrays["rowvec_0"].dtype == np.float64
    back = C.unpack_rows(ck.meta["mma"]["rows"], ck.arrays)
    assert back.keys() == rows.keys() and back["buckling:a"]["last_info"] == {"x": 1.5} and back["odd"] == 5 and back["stress:c"] == rows["stress:c"]
    assert back["buckling:a"]["v0"].tobytes() == v.tobytes() and back["buckling:b"]["v0"].tobytes() == w.tobytes()
    before_back = C.unpack_rows(ck.meta["rows_before"], ck.arrays)
    assert before_back["buckling:a"]["v0"].tobytes() == v.tobytes() and before_back["buckling:b"]["v0"] is None
    back["buckling:a"]["v0"][0] = 99.0                                                # a copy: the arrays of the file are not shared
    assert ck.arrays["rowvec_0"][0] == v[0]


def test_a_cut_current_file_gives_the_file_of_the_pass_before(tmp_path):
    C.write_checkpoint(tmp_path, arrays_of(1), meta_of(1))
    size = C.write_checkpoint(tmp_path, arrays_of(2), meta_of(2))
    cur = tmp_path / C.CHECKPOINT_FILE
    cur.write_bytes(cur.read_bytes()[: size // 2])
    ck = C.read_checkpoint(tmp_path)
    assert (ck.file, ck.iteration) == (C.PREV_FILE, 1) and list(ck.bad) == [C.CHECKPOINT_FILE]
    assert "checkpoint.npz cannot be used" in ck.facts[0] and "used checkpoint.prev.npz (iteration 1)" in ck.facts[1]


def test_a_flipped_byte_is_found_by_the_crc(tmp_path):
    C.write_checkpoint(tmp_path, arrays_of(1), meta_of(1))
    C.write_checkpoint(tmp_path, dict(arrays_of(2), big=np.zeros(5000)), meta_of(2))
    cur = tmp_path / C.CHECKPOINT_FILE
    raw = bytearray(cur.read_bytes())
    raw[len(raw) // 2] ^= 0xFF                                       # the middle of the file is inside the array 'big'
    cur.write_bytes(bytes(raw))
    assert C.read_checkpoint(tmp_path).file == C.PREV_FILE


def test_a_part_file_is_never_used(tmp_path):
    (tmp_path / C.TMP_FILE).write_bytes(b"half")
    with pytest.raises(C.ResumeRefused, match="no checkpoint"):
        C.read_checkpoint(tmp_path)


def test_no_valid_file_is_refused_with_the_reason_and_options(tmp_path):
    (tmp_path / C.CHECKPOINT_FILE).write_bytes(b"not a zip")
    with pytest.raises(C.ResumeRefused) as e:
        C.read_checkpoint(tmp_path)
    assert "checkpoint.npz:" in e.value.reason and any("without --resume" in o for o in e.value.options)
    assert "Options:" in str(e.value)
    with pytest.raises(C.ResumeRefused, match="no checkpoint"):
        C.read_checkpoint(tmp_path / "nowhere")


def test_an_unknown_version_is_refused_and_not_deleted(tmp_path):
    C.write_checkpoint(tmp_path, arrays_of(1), dict(meta_of(1), format=99, topopt_version="9.9"))
    with pytest.raises(C.ResumeRefused) as e:
        C.read_checkpoint(tmp_path)
    assert "unknown checkpoint version 99" in e.value.reason
    assert any("use the topopt version that wrote the checkpoint" in o for o in e.value.options)
    assert (tmp_path / C.CHECKPOINT_FILE).exists()


def test_a_file_with_missing_pieces_is_not_a_checkpoint(tmp_path):
    for name, meta, arrays in (("no kind", dict(meta_of(1), kind="other"), arrays_of(1)),
                               ("no x", meta_of(1), {"mma_low": np.zeros((4, 1))}),
                               ("x of the wrong size", meta_of(1), {"x": np.zeros(3)}),
                               ("no mma", {k: v for k, v in meta_of(1).items() if k != "mma"}, arrays_of(1))):
        d = tmp_path / name.replace(" ", "_")
        d.mkdir()
        C.write_checkpoint(d, arrays, meta)
        with pytest.raises(C.ResumeRefused):
            C.read_checkpoint(d)


def test_describe(tmp_path):
    assert C.describe(tmp_path) is None
    C.write_checkpoint(tmp_path, arrays_of(1), meta_of(1, stop={"rule": "converged"}, resume_points=[{}, {}]))
    d = C.describe(tmp_path)
    assert (d["file"], d["iteration"], d["iterations"], d["complete"], d["stop_rule"], d["resume_points"]) == (
        C.CHECKPOINT_FILE, 1, 10, False, "converged", 2)
    (tmp_path / C.CHECKPOINT_FILE).write_bytes(b"cut")
    assert C.describe(tmp_path) is None                              # the file does not read, and there is no file before


# --- input hashes and the comparison -------------------------------------------------------

def inputs(path, problem=None, solver=None):
    return C.input_hashes(load_config(path), problem, solver)


def test_the_comparison_names_each_changed_input(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=6)
    old = inputs(p)
    assert C.diff_inputs(old, inputs(p)) == ([], [])
    p.write_text("# a comment\n" + p.read_text())                    # text only
    changes, notes = C.diff_inputs(old, inputs(p))
    assert changes == [] and len(notes) == 1 and "comment" in notes[0]
    edit(p, lambda d: d["settings"].update(volume_fraction=0.35))
    changes, notes = C.diff_inputs(old, inputs(p))
    assert changes == ["settings.volume_fraction: 0.3 -> 0.35"] and notes == []
    edit(p, lambda d: d["load_cases"][0]["forces"][0].update(force=[0, 0, -3.0]))
    assert any(c.startswith("load_cases[0].forces[0].force: [0, 0, -2.0] -> [0, 0, -3.0]") for c in C.diff_inputs(old, inputs(p))[0])
    edit(p, lambda d: d["settings"].update(beta_schedule=[[0, 1], [10, 3]]))
    assert any(c.startswith("settings.beta_schedule") for c in C.diff_inputs(old, inputs(p))[0])


def test_the_guards_and_the_stop_rules_are_notes_and_not_changes(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=6)
    old = inputs(p)
    edit(p, lambda d: d["settings"].update(max_active_voxels=123456, stop_rules=True))
    changes, notes = C.diff_inputs(old, inputs(p))
    assert changes == []                                             # they do not change the numbers of a pass
    assert sorted(n.split(":")[0] for n in notes) == ["settings.max_active_voxels", "settings.stop_rules"]
    assert all("does not change the numbers of a pass" in n for n in notes)


def test_a_key_outside_the_settings_is_a_real_change_and_not_a_comment_note(tmp_path):
    """The whole parsed YAML is compared (A4a `mirror`, A5 `regions.measure`, and any later key can change a problem with the same grid)."""
    p = make_case(tmp_path, h=2.0, iterations=6)
    old = inputs(p)
    edit(p, lambda d: d.update(name="another name"))
    changes, notes = C.diff_inputs(old, inputs(p))
    assert changes == ["name: cantilever -> another name"] and notes == []
    edit(p, lambda d: d.update(name="cantilever", assumptions=["a new assumption"]))     # a key that the loader accepts and the solver ignores
    changes, notes = C.diff_inputs(old, inputs(p))
    assert changes == ["assumptions: <not set> -> [\"a new assumption\"]"] and notes == []
    new = json.loads(json.dumps(old))
    new["yaml_parsed"]["mirror"] = {"axis": "y", "at": 5.0}          # a key that this loader does not know: the compare still sees it
    changes, _ = C.diff_inputs(old, new)
    assert sorted(changes) == ["mirror.at: <not set> -> 5.0", "mirror.axis: <not set> -> y"]


def test_comments_and_the_order_of_keys_are_a_note_only(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=6)
    old = inputs(p)
    data = yaml.safe_load(p.read_text())
    p.write_text("# a comment\n" + yaml.safe_dump(dict(reversed(list(data.items()))), sort_keys=False))   # the same content, other order
    changes, notes = C.diff_inputs(old, inputs(p))
    assert changes == [] and len(notes) == 1 and "comments or format only" in notes[0]


def test_the_effective_settings_still_catch_a_default_that_changed(tmp_path):
    """The parsed YAML is the same but the effective settings differ (a default of topopt changed between two versions)."""
    p = make_case(tmp_path, h=2.0, iterations=6)
    old = inputs(p)
    new = json.loads(json.dumps(inputs(p)))
    new["effective"]["settings"]["move_limit"] = 0.1
    changes, _ = C.diff_inputs(old, new)
    assert len(changes) == 1 and changes[0].startswith("settings.move_limit:") and "a default of topopt changed" in changes[0]


def test_an_old_checkpoint_without_the_parsed_yaml_compares_the_effective_settings(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=6)
    old = json.loads(json.dumps(inputs(p)))
    del old["yaml_parsed"]
    edit(p, lambda d: d["settings"].update(volume_fraction=0.35))
    changes, _ = C.diff_inputs(old, inputs(p))
    assert changes == ["settings.volume_fraction: 0.3 -> 0.35"]


def test_the_comparison_sees_a_region_file_and_the_solver(tmp_path):
    from test_optimize import box
    p = make_case(tmp_path, h=2.0, iterations=6)
    edit(p, lambda d: d["settings"].update(solver="pardiso"))        # an explicit setting: another solver that runs is a change
    old = inputs(p, solver="pardiso")
    box(tmp_path / "tip.stl", (16, 4, 0), (20, 6, 3))                # a bigger load region
    changes, _ = C.diff_inputs(old, inputs(p, solver="amg"))
    assert any(c.startswith("region file load:tip.stl has other content") for c in changes)
    assert "the solver that runs: pardiso -> amg" in changes


def test_another_pick_of_auto_is_a_note_and_not_a_change(tmp_path):
    """P9-4d: `settings.solver: auto` on both sides can give gpu-mg in one run and a CPU solver in the other (a fallback, a GPU that is gone)."""
    p = make_case(tmp_path, h=2.0, iterations=6)
    old = inputs(p, solver="gpu-mg")                                 # settings.solver is `auto` (the default)
    changes, notes = C.diff_inputs(old, inputs(p, solver="cholesky"))
    assert changes == []
    assert len(notes) == 1 and notes[0].startswith("the solver that runs: gpu-mg -> cholesky (settings.solver is auto in both runs")
    assert C.diff_inputs(old, inputs(p, solver="gpu-mg")) == ([], [])                # the same pick: nothing
    edit(p, lambda d: d["settings"].update(solver="cholesky"))       # an explicit setting on one side: the setting is a change, and so is the solver
    changes, notes = C.diff_inputs(old, inputs(p, solver="cholesky"))
    assert "settings.solver: <not set> -> cholesky" in changes and "the solver that runs: gpu-mg -> cholesky" in changes and not notes
    edit(p, lambda d: d["settings"].update(solver="gpu-mg"))
    explicit = inputs(p, solver="gpu-mg")
    changes, notes = C.diff_inputs(explicit, inputs(p, solver="cholesky"))           # explicit on both sides: a change
    assert changes == ["the solver that runs: gpu-mg -> cholesky"] and not notes


def test_two_region_files_with_the_same_base_name_get_two_keys(tmp_path):
    """Two fixed STLs named clamp.stl in different folders: each has its own key and its own hash (no hash is lost without a fact)."""
    from test_optimize import box
    p = make_case(tmp_path, h=2.0, iterations=6)
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    box(tmp_path / "a" / "clamp.stl", (-2, 0, 0), (0, 5, 10))
    box(tmp_path / "b" / "clamp.stl", (-2, 5, 0), (0, 10, 10))
    edit(p, lambda d: d["regions"].update(fixed=["a/clamp.stl", "b/clamp.stl"]))
    keys = C.input_hashes(load_config(p))["regions"]
    assert {"fixed:a/clamp.stl", "fixed:b/clamp.stl", "design:design.stl", "load:tip.stl"} == set(keys)
    assert keys["fixed:a/clamp.stl"] != keys["fixed:b/clamp.stl"]
    old = inputs(p)
    box(tmp_path / "b" / "clamp.stl", (-2, 5, 0), (0, 10, 9))        # change the second file only
    changes, _ = C.diff_inputs(old, inputs(p))
    assert changes == [c for c in changes if c.startswith("region file fixed:b/clamp.stl has other content")] and len(changes) == 1


def test_the_region_hashes_cover_every_field_of_config_regions(tmp_path):
    """A field that a later item adds (for example `solid`) is hashed with no change of the code: keys by role and relative path."""
    import types
    from pathlib import Path
    for name in ("design.stl", "extra.stl", "sub/solid.stl", "load.stl"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(name.encode())
    regions = types.SimpleNamespace(design=tmp_path / "design.stl", fixed=(), solid=(tmp_path / "sub" / "solid.stl",),
                                    other=types.SimpleNamespace(path=tmp_path / "extra.stl"),
                                    load=(types.SimpleNamespace(name="tip", path=tmp_path / "load.stl"),), none=None)
    config = types.SimpleNamespace(regions=regions, yaml_path=tmp_path / "x.yaml")
    keys = C.region_hashes(config)
    assert sorted(keys) == ["design:design.stl", "load:load.stl", "other:extra.stl", "solid:sub/solid.stl"]
    assert keys["solid:sub/solid.stl"] == C.sha256_file(Path(tmp_path / "sub" / "solid.stl"))


def test_a_checkpoint_with_the_old_region_keys_gives_facts_and_not_a_crash(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=6)
    new = inputs(p)
    old = json.loads(json.dumps(new))
    old["regions"] = {"design:design.stl": "x", "fixed:plate.stl": "y", "load:tip": "z"}      # the keys of an earlier version
    changes, _ = C.diff_inputs(old, new)
    assert "region file load:tip is no longer in the YAML" in changes and "region file load:tip.stl is new in the YAML" in changes


def test_cut_history_keeps_the_bytes_and_drops_a_partial_line(tmp_path):
    lines = [json.dumps({"iteration": i, "x": 1.0 / 3}) + "\n" for i in range(5)]
    path = tmp_path / "history.jsonl"
    path.write_text("".join(lines))
    inode = path.stat().st_ino
    cut = C.cut_history(tmp_path, 4)
    assert (cut["kept"], cut["last_kept"], cut["dropped"]) == (5, 4, []) and path.stat().st_ino == inode   # no rewrite
    path.write_text("".join(lines) + '{"iteration": 5, "x"')       # a line that was cut
    cut = C.cut_history(tmp_path, 3)
    assert (cut["kept"], cut["dropped"], cut["bad_lines"]) == (4, [4], 1)
    assert path.read_text() == "".join(lines[:4])
    assert C.cut_history(tmp_path / "none", 3)["present"] is False


def test_cut_history_ends_the_last_line_with_a_newline(tmp_path):
    """A kill between the text of a line and its newline: the line parses. The next appended line must not join it (D1a follow-up)."""
    path = tmp_path / "history.jsonl"
    path.write_bytes((json.dumps({"iteration": 0}) + "\n" + json.dumps({"iteration": 1})).encode())
    cut = C.cut_history(tmp_path, 1)
    assert cut["kept"] == 2 and cut["newline_added"] and cut["dropped"] == [] and cut["bad_lines"] == 0
    assert path.read_bytes().endswith(b"}\n") and [json.loads(x)["iteration"] for x in path.read_text().splitlines()] == [0, 1]


# --- a killed run continues ----------------------------------------------------------------

@needs_direct
@pytest.mark.parametrize("point, first", [
    ("after_checkpoint:0", 1), ("after_checkpoint:10", 11), ("after_checkpoint:23", 24), ("after_checkpoint:24", 25),
    ("after_checkpoint:9", 10), ("after_checkpoint:19", 20),    # the pass before a beta stage start (the stages start at 10 and 20): the scale of the objective is set again (P3a)
    ("before_checkpoint:12", 12), ("partial_tmp:12", 12), ("between_renames:12", 12), ("partial_history:12", 12),
    ("corrupt_current:12", 12)])
def test_a_killed_run_resumes_to_the_same_bytes(case, tmp_path, point, first):
    """GOAL D1b 4: SIGKILL at each point of the write, then `--resume`: the run folder equals the run that was not stopped."""
    yaml_path, base = case
    out = tmp_path / "r"
    killed(yaml_path, out, point)
    done = resume(yaml_path, out)
    assert "resume: continue from" in log(out)
    text = log(out)
    assert f"The first pass to run is {first}." in text or first == 25
    if first == 25:
        assert "No pass is left to run" in text
    kind = point.split(":")[0]
    if kind == "partial_tmp":
        assert "Removed checkpoint.npz.tmp" in text
    if kind == "corrupt_current":
        assert "checkpoint.npz cannot be used" in text and "used checkpoint.prev.npz" in text
    if kind == "partial_history":
        assert "that do not parse (a partial last line" in text
    assert "verdict:" in text and (out / "job.DONE").exists() and not (out / "job.lock").exists()
    assert not (out / C.TMP_FILE).exists()
    same_run(base, out)
    # the resume is on record in the verdict, with the iteration, and the folder is the same size as the base
    entry = json.loads((out / "verdict.json").read_text())["resume_points"][-1]
    assert entry["iteration"] == first - 1 and entry["changes"] == []


@needs_direct
def test_two_kills_and_two_resumes(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "r"
    killed(yaml_path, out, "after_checkpoint:7")
    done = child("after_checkpoint:15", *run_args(yaml_path, out, "--resume"))
    assert done.returncode == -9
    resume(yaml_path, out)
    same_run(base, out)
    points = json.loads((out / "verdict.json").read_text())["resume_points"]
    assert [p["iteration"] for p in points] == [7, 15]
    assert json.loads((out / "result" / "metrics.json").read_text())["verdict"]["resume_points"] == points


@needs_direct
def test_the_run_time_goes_on_after_a_resume(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "r"
    killed(yaml_path, out, "after_checkpoint:12")
    saved = C.describe(out)["elapsed_s"]
    started = json.loads((out / "run.json").read_text())["started_at"]
    resume(yaml_path, out)
    rows = history(out)
    assert rows[13]["elapsed_s"] > rows[12]["elapsed_s"] and rows[13]["elapsed_s"] - saved < 30   # the clock goes on from the checkpoint
    assert all(a["elapsed_s"] <= b["elapsed_s"] for a, b in zip(rows, rows[1:]))
    assert json.loads((out / "run.json").read_text())["started_at"] == started                  # the start of the first run


@needs_direct
def test_the_stress_run_with_step_acceptance_resumes_to_the_same_bytes(tmp_path):
    """The stress row keeps a KS scale between passes, and step acceptance carries a design and its solve (A2)."""
    yaml_path = stress_yaml(tmp_path, iterations=24)
    base = tmp_path / "base"
    finish(yaml_path, base)
    for n, point in enumerate(("after_checkpoint:9", "before_checkpoint:12", "after_checkpoint:17")):
        out = tmp_path / f"r{n}"
        killed(yaml_path, out, point)
        resume(yaml_path, out)
        same_run(base, out)
        assert any("step_acceptance" in r for r in history(out))


def test_an_amg_run_resumes_to_the_same_numbers(tmp_path):
    """AMG is not reproducible bit for bit (random start of the spectral radius estimate), so the numbers are close, not equal."""
    yaml_path = make_case(tmp_path, h=1.0, iterations=24)
    edit(yaml_path, lambda d: d["settings"].update(solver="amg"))
    base, out = tmp_path / "base", tmp_path / "r"
    finish(yaml_path, base)
    killed(yaml_path, out, "after_checkpoint:12")
    resume(yaml_path, out)
    assert "solver amg" in log(out) and "resume: continue from checkpoint.npz, iteration 12" in log(out)
    same_run(base, out, exact=False, rtol=1e-6)   # measured: two AMG runs differ by about 1.5e-8 on this case


def rewrite_checkpoint(out, drop=(), add=None, meta_drop=()):
    """Write `checkpoint.npz` of a folder again without some arrays or meta keys, and with other arrays. No `.prev` file stays."""
    ck = C.read_checkpoint(out)
    arrays = {k: v for k, v in ck.arrays.items() if k not in drop}
    arrays.update(add or {})
    meta = {k: v for k, v in ck.meta.items() if k not in meta_drop}
    (out / C.PREV_FILE).unlink(missing_ok=True)
    (out / C.CHECKPOINT_FILE).unlink()
    C.write_checkpoint(out, arrays, meta)


def test_a_checkpoint_with_missing_or_extra_fields_resumes_with_facts(tmp_path):
    """A field that the checkpoint lacks gives a `resume:` fact and the run goes on. So does a field that this version does not know
    (a later item, for example the robust mode, adds fields through the hook)."""
    yaml_path = make_case(tmp_path, h=1.0, iterations=12)
    edit(yaml_path, lambda d: d["settings"].update(solver="amg"))
    out = tmp_path / "r"
    killed(yaml_path, out, "after_checkpoint:5")
    assert "solver__x_prev" in C.read_checkpoint(out).arrays
    rewrite_checkpoint(out, drop=("solver__x_prev",), add={"robust__projection_s": np.zeros(3)}, meta_drop=("step_acceptance", "prev"))
    resume(yaml_path, out)
    text = log(out)
    assert "resume: the checkpoint has no AMG warm start: the first solve starts from zero" in text
    assert "resume: the checkpoint has no step acceptance counters" in text
    assert "resume: the checkpoint has no history record" in text
    assert "resume: the checkpoint has fields that this topopt does not use: robust__projection_s" in text
    assert [r["iteration"] for r in history(out)] == list(range(13)) and (out / "job.DONE").exists()


def test_the_owners_of_state_are_the_model_and_its_solver():
    """The hook of later items: an object with `checkpoint_state` is an owner. The model and `model.solver` are looked at."""
    class Owner:
        def checkpoint_state(self):
            return {}

    class Plain:
        pass

    class Model(Owner):
        solver = Owner()

    assert [n for n, _ in C._hooks(Model())] == ["model", "solver"]
    plain = Plain()
    plain.solver = Plain()
    assert C._hooks(plain) == []
    plain.solver = Owner()
    assert [n for n, _ in C._hooks(plain)] == ["solver"]


# --- the refusals --------------------------------------------------------------------------

def test_resume_without_a_folder_or_a_checkpoint_is_refused_and_writes_nothing(work, tmp_path, capsys):
    assert main(run_args(work, tmp_path / "nowhere", "--resume")) == 1
    assert "run folder" in capsys.readouterr().err and not (tmp_path / "nowhere").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(run_args(work, empty, "--resume")) == 1
    err = capsys.readouterr().err
    assert "no checkpoint" in err and "without --resume" in err and len(err.strip().splitlines()) == 1
    assert tree_state(empty) == {}


def test_allow_changed_inputs_needs_resume(work, tmp_path, capsys):
    assert main(run_args(work, tmp_path / "r", "--allow-changed-inputs")) == 1
    assert "--allow-changed-inputs needs --resume" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


@pytest.fixture
def half(work, tmp_path):
    """A run that was killed after pass 6 (24 passes planned): `(yaml, folder)`."""
    out = tmp_path / "r"
    killed(work, out, "after_checkpoint:6")
    return work, out


def test_changed_inputs_are_refused_with_facts_and_options_and_nothing_is_written(half, capsys):
    yaml_path, out = half
    edit(yaml_path, lambda d: d["settings"].update(volume_fraction=0.35))
    before = tree_state(out)
    assert main(run_args(yaml_path, out, "--resume")) == 1
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert "settings.volume_fraction: 0.3 -> 0.35" in err and "--allow-changed-inputs" in err
    assert "restore the old inputs" in err and "without --resume" in err
    assert tree_state(out) == before                                 # no log line, no cut, no clean


def test_the_override_goes_on_with_the_new_inputs_and_says_so(half):
    yaml_path, out = half
    edit(yaml_path, lambda d: d["settings"].update(volume_fraction=0.35))
    finish(yaml_path, out, "--resume", "--allow-changed-inputs")
    text = log(out)
    assert "resume: the inputs changed since the checkpoint: settings.volume_fraction: 0.3 -> 0.35. The override is given" in text
    point = json.loads((out / "verdict.json").read_text())["resume_points"][0]
    assert point["changes"] == ["settings.volume_fraction: 0.3 -> 0.35"] and point["iteration"] == 6
    assert history(out)[-1]["volume_fraction"] > 0.32               # the run followed the new volume fraction


def test_a_change_of_the_yaml_text_only_is_a_note_and_not_a_refusal(half):
    yaml_path, out = half
    yaml_path.write_text("# only a comment changed\n" + yaml_path.read_text())
    finish(yaml_path, out, "--resume")
    assert "resume: the YAML text changed (comments or format only): the parsed content and the effective settings are the same" in log(out)
    assert json.loads((out / "verdict.json").read_text())["resume_points"][0]["changes"] == []


def test_a_changed_name_or_an_added_key_is_refused_and_the_override_goes_on(half, capsys):
    yaml_path, out = half
    edit(yaml_path, lambda d: d.update(name="renamed", assumptions=["a note"]))
    before = tree_state(out)
    assert main(run_args(yaml_path, out, "--resume")) == 1
    err = capsys.readouterr().err
    assert "name: cantilever -> renamed" in err and "assumptions:" in err and "--allow-changed-inputs" in err
    assert "comment" not in err and tree_state(out) == before
    finish(yaml_path, out, "--resume", "--allow-changed-inputs")
    changes = json.loads((out / "verdict.json").read_text())["resume_points"][0]["changes"]
    assert changes[0] == "assumptions: <not set> -> [\"a note\"]" and "name: cantilever -> renamed" in changes


def test_a_changed_region_file_is_refused(half, capsys):
    from test_optimize import box
    yaml_path, out = half
    box(yaml_path.parent / "tip.stl", (18.1, 4, 0), (20, 6, 3))      # other bytes, the same voxels
    before = tree_state(out)
    assert main(run_args(yaml_path, out, "--resume")) == 1
    err = capsys.readouterr().err
    assert "region file load:tip.stl has other content" in err and "--allow-changed-inputs" in err
    assert "the voxel labels are not the same" not in err
    assert tree_state(out) == before


def test_a_region_that_fills_other_voxels_is_refused_with_the_counts_and_the_override_maps_by_position(half, capsys):
    """A bigger load region leaves 12 fewer design voxels on the same grid: a change, with the counts. The override maps by position."""
    from test_optimize import box
    yaml_path, out = half
    box(yaml_path.parent / "tip.stl", (16, 4, 0), (20, 6, 3))
    before = tree_state(out)
    assert main(run_args(yaml_path, out, "--resume")) == 1
    err = capsys.readouterr().err
    assert "the design voxels are not the same: 1976 of the 1988 saved design voxels are design voxels in this run too" in err
    assert "0 design voxels are new" in err and "12 saved design voxels are no longer design voxels (dropped)" in err
    assert tree_state(out) == before
    finish(yaml_path, out, "--resume", "--allow-changed-inputs")
    assert "1976 of the 1988 saved design voxels" in log(out)
    assert [r["iteration"] for r in history(out)] == list(range(25)) and (out / "job.DONE").exists()


def test_another_grid_cannot_resume_with_any_override(half, capsys):
    yaml_path, out = half
    edit(yaml_path, lambda d: d["settings"].update(voxel_size=2.0))
    for extra in ([], ["--allow-changed-inputs"]):
        assert main(run_args(yaml_path, out, "--resume", *extra)) == 1
        err = capsys.readouterr().err
        assert "Resume cannot work" in err and "does not fit the new grid" in err and "restore the YAML" in err


def test_no_pass_left_cannot_resume(half, capsys):
    yaml_path, out = half
    edit(yaml_path, lambda d: d["settings"].update(iterations=6))
    assert main(run_args(yaml_path, out, "--resume", "--allow-changed-inputs")) == 1
    err = capsys.readouterr().err
    assert "Resume cannot work: the checkpoint is at pass 6, and settings.iterations is 6" in err
    assert "use settings.iterations above 6" in err


def test_a_shorter_unfinished_run_resumes_with_the_override(half):
    yaml_path, out = half
    edit(yaml_path, lambda d: d["settings"].update(iterations=12))
    finish(yaml_path, out, "--resume", "--allow-changed-inputs")
    assert history(out)[-1]["iteration"] == 12 and "settings.iterations: 24 -> 12" in log(out)


def test_a_finished_run_is_not_made_longer(case, tmp_path, capsys):
    yaml_path, base = case
    out = tmp_path / "r"
    killed(yaml_path, out, "after_checkpoint:24")
    edit(yaml_path, lambda d: d["settings"].update(iterations=30))
    try:
        assert main(run_args(yaml_path, out, "--resume", "--allow-changed-inputs")) == 1
        err = capsys.readouterr().err
        assert "finished at pass 24" in err and "This version does not make a finished run longer" in err and "no step after it" not in err
        assert "use settings.iterations equal to the old value" in err
    finally:
        edit(yaml_path, lambda d: d["settings"].update(iterations=24))


@needs_direct
def test_the_resume_of_a_finished_run_runs_only_the_post_step(case, tmp_path):
    yaml_path, base = case
    out = tmp_path / "r"
    shutil_copy(base, out)
    (out / "result" / "final.stl").unlink()                          # a person removed the result
    before = (out / "history.jsonl").read_bytes()
    resume(yaml_path, out)
    assert (out / "history.jsonl").read_bytes() == before            # no pass ran
    assert "No pass is left to run. The post step runs again." in log(out)
    same_run(base, out)


def shutil_copy(src, dst):
    import shutil
    shutil.copytree(src, dst)
    for name in ("job.DONE", "job.guard", "job.json"):
        (dst / name).unlink(missing_ok=True)


# --- the folder: lock, markers, files ------------------------------------------------------

def test_a_live_lock_stops_a_resume_and_a_stale_lock_does_not(half):
    yaml_path, out = half
    lock = json.loads((out / "job.lock").read_text()) if (out / "job.lock").exists() else None
    assert lock is not None and lock["pid"] > 0                      # the killed run left its lock
    finish(yaml_path, out, "--resume")
    assert "job lock: the process of the lock" in log(out) and "is not running" in log(out)
    assert (out / "job.DONE").exists() and not (out / "job.FAILED").exists() and not (out / "job.lock").exists()


def test_a_live_holder_stops_a_resume_without_a_cut(half):
    from test_runjob import put_lock
    yaml_path, out = half
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        put_lock(out, sleeper.pid)
        before = tree_state(out)
        done = child("never:-1", *run_args(yaml_path, out, "--resume"))
        assert done.returncode != 0 and "holds the job lock" in done.stderr + done.stdout
        assert tree_state(out) == before
    finally:
        sleeper.kill()
        sleeper.wait()


def test_a_failed_marker_does_not_block_a_resume(half, capsys):
    yaml_path, out = half
    (out / "job.FAILED").write_text("exit 1\n")
    (out / "job.lock").unlink(missing_ok=True)
    finish(yaml_path, out, "--resume")
    assert (out / "job.DONE").exists() and not (out / "job.FAILED").exists()


def test_the_resume_cuts_the_history_and_the_snapshots_to_the_checkpoint(half):
    yaml_path, out = half
    rows = history(out)
    assert rows[-1]["iteration"] == 6
    # a pass that was written after the checkpoint: a history line and a snapshot of pass 7 (as after "before_checkpoint")
    with open(out / "history.jsonl", "a") as f:
        f.write(json.dumps(dict(rows[-1], iteration=7)) + "\n" + '{"iteration": 8, "part')
    (out / "snapshots" / "it_0007.bin").write_bytes((out / "snapshots" / "it_0006.bin").read_bytes())
    (out / "snapshots" / "it_0008.bin.tmp").write_bytes(b"part")
    (out / "result").mkdir(exist_ok=True)
    (out / "result" / "final.stl").write_bytes(b"old")
    finish(yaml_path, out, "--resume")
    text = log(out)
    assert "Dropped history lines for iterations 7 to 7 and 1 line(s) of history.jsonl that do not parse" in text
    assert "dropped snapshots 7 to 7" in text.lower() or "Dropped snapshots 7 to 7" in text
    assert "Removed result/ of the earlier run" in text
    assert [r["iteration"] for r in history(out)] == list(range(25))
    assert sorted(p.name for p in (out / "snapshots").iterdir()) == [f"it_{i:04d}.bin" for i in range(25)]
    assert (out / "result" / "final.stl").read_bytes() != b"old"
    point = json.loads((out / "verdict.json").read_text())["resume_points"][0]
    assert point["dropped_history_iterations"] == [7] and point["dropped_snapshots"] == [7]


def test_the_checkpoint_is_in_the_run_files_and_the_next_fresh_run_removes_it(half, work):
    from topopt.snapshot import RUN_FILES
    assert {C.CHECKPOINT_FILE, C.PREV_FILE, "verdict.json"} <= set(RUN_FILES)
    yaml_path, out = half
    assert (out / C.CHECKPOINT_FILE).exists() and (out / C.PREV_FILE).exists()
    edit(yaml_path, lambda d: d["settings"].update(iterations=3))
    finish(yaml_path, out)                                           # no --resume: a new run, the old files go
    assert [r["iteration"] for r in history(out)] == [0, 1, 2, 3]
    assert C.describe(out)["iteration"] == 3 and C.describe(out)["complete"] is True
    assert json.loads((out / "verdict.json").read_text())["resume_points"] == []


def test_a_new_run_deletes_the_timing_file_of_the_solver_gpu_mg_whatever_its_own_solver(tmp_path):
    """G1e review m1. The names in `RUN_FILES` and their `.tmp` files go. No OpenCL is needed. Before the fix a fresh run with another
    solver left an old `gpu_mg.jsonl`, and a later resume with gpu-mg could take its lines into the record of the new run."""
    from topopt.snapshot import RUN_FILES, RunWriter
    assert "gpu_mg.jsonl" in RUN_FILES
    for name in ("gpu_mg.jsonl", "gpu_mg.jsonl.tmp"):
        (tmp_path / name).write_text('{"line": 0}\n')
    (tmp_path / "notes.txt").write_text("not a run file")
    RunWriter(tmp_path, quiet=True).clean()
    assert not (tmp_path / "gpu_mg.jsonl").exists() and not (tmp_path / "gpu_mg.jsonl.tmp").exists()
    assert (tmp_path / "notes.txt").exists()                           # no other file is touched


# --- the checkpoint of a run, and what it costs --------------------------------------------

def test_a_run_writes_one_checkpoint_per_pass_and_logs_the_cost(case):
    _, base = case
    text = log(base)
    assert "fact: 25 checkpoints written to checkpoint.npz" in text and "% of the run time)" in text
    d = C.describe(base)
    assert (d["iteration"], d["iterations"], d["complete"], d["file"]) == (24, 24, True, C.CHECKPOINT_FILE)
    ck = C.read_checkpoint(base)
    assert ck.meta["inputs"]["effective"]["settings"]["iterations"] == 24 and ck.meta["prev"]["iteration"] == 24
    assert ck.meta["mma"]["it"] == 24                                # the last pass measures only: no step after it
    assert ck.arrays["x"].shape == (ck.meta["n_design"],) and ck.arrays["mma_low"].shape == (ck.meta["n_design"], 1)
    assert "model__" not in "".join(ck.arrays)                       # the model has no state of its own here


def test_the_solver_state_is_in_the_checkpoint_when_the_solver_has_some(tmp_path):
    yaml_path = make_case(tmp_path, h=1.0, iterations=4)
    edit(yaml_path, lambda d: d["settings"].update(solver="amg"))
    out = tmp_path / "r"
    finish(yaml_path, out)
    ck = C.read_checkpoint(out)
    assert any(n.startswith("solver__") for n in ck.arrays)           # the AMG warm start


# --- stop rules and the verdict in a run ---------------------------------------------------

RULES = {"objective_change": 0.01, "grayness_max": 0.4}   # thresholds that the coarse case (2 mm voxels) meets about 10 passes after the last beta step


def stoppable(tmp_path, iterations=60, rules=True):
    p = make_case(tmp_path, h=2.0, iterations=iterations)
    edit(p, lambda d: d["settings"].update(stop_rules=dict(RULES, enabled=rules)))
    return p


def test_stop_rules_are_off_by_default_and_the_flag_turns_them_on(tmp_path):
    p = stoppable(tmp_path, rules=False)                             # thresholds in the YAML, the rules off
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(run_args(p, a)) == 0
    assert len(history(a)) == 61 and json.loads((a / "verdict.json").read_text())["stop_rules"]["enabled"] is False
    assert main(run_args(p, b, "--stop-rules")) == 0
    v = json.loads((b / "verdict.json").read_text())
    assert v["state"] == "stopped_by_rule" and v["stop"]["rule"] == "converged" and v["converged"] and v["feasible"]
    assert len(history(b)) == v["final_iteration"] + 1 < 61
    assert "stop rule 'converged' at pass" in log(b) and "verdict: stop rule 'converged'" in log(b)
    # the flag does not change the numbers of the passes that both runs make (the stop pass itself makes no step)
    n = v["final_iteration"]
    hb, ha = [r["compliance"] for r in history(b)][:n], [r["compliance"] for r in history(a)][:n]
    assert hb == ha if DIRECT else hb == pytest.approx(ha, rel=1e-9)     # AMG: not repeatable in the last digits


def test_no_stop_rules_flag_wins_over_the_yaml(tmp_path):
    p = stoppable(tmp_path)
    out = tmp_path / "r"
    assert main(run_args(p, out, "--no-stop-rules")) == 0
    assert len(history(out)) == 61 and json.loads((out / "verdict.json").read_text())["stop_rules"]["enabled"] is False


def test_the_stop_pass_only_measures_and_the_metrics_carry_the_verdict(tmp_path):
    p = stoppable(tmp_path)
    out = tmp_path / "r"
    assert main(run_args(p, out)) == 0
    v = json.loads((out / "verdict.json").read_text())
    last = history(out)[-1]
    assert last["iteration"] == v["final_iteration"] and last["change"] == 0.0        # no update step at the stop pass
    assert C.describe(out)["complete"] and C.describe(out)["stop_rule"] == "converged"
    assert json.loads((out / "result" / "metrics.json").read_text())["verdict"] == v
    assert f"checkpoint.npz" in log(out)


def test_a_run_that_a_rule_stopped_resumes_to_its_post_step_only(tmp_path):
    p = stoppable(tmp_path)
    out = tmp_path / "r"
    assert main(run_args(p, out)) == 0
    before = (out / "history.jsonl").read_bytes()
    verdict = json.loads((out / "verdict.json").read_text())
    assert main(run_args(p, out, "--resume")) == 0
    assert (out / "history.jsonl").read_bytes() == before
    again = json.loads((out / "verdict.json").read_text())
    assert again["state"] == "stopped_by_rule" and again["stop"] == verdict["stop"] and len(again["resume_points"]) == 1


def test_the_time_budget_stops_the_run_and_the_verdict_gives_the_convergence_tests(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=60)
    edit(p, lambda d: d["settings"].update(stop_rules={"time_budget_s": 0.05}))
    out = tmp_path / "r"
    assert main(run_args(p, out)) == 0
    v = json.loads((out / "verdict.json").read_text())
    assert v["state"] == "stopped_by_rule" and v["stop"]["rule"] == "time_budget" and not v["converged"]
    assert v["final_iteration"] < 60 and "Convergence tests at this pass" in v["text"] and "stop rule 'time_budget'" in log(out)


def test_a_run_with_no_rules_reports_the_verdict_of_its_last_pass(case):
    _, base = case
    v = json.loads((base / "verdict.json").read_text())
    assert v["state"] == "iteration_limit" and v["final_iteration"] == 24 and v["feasible"] is True and v["converged"] is False
    assert "(threshold " in v["text"] and "not met" in v["text"] and "verdict: iteration limit reached" in log(base)
    assert json.loads((base / "result" / "metrics.json").read_text())["verdict"] == v


def test_a_folder_without_verdict_json_gets_a_verdict_from_its_history(case, tmp_path):
    from topopt import post
    _, base = case
    out = tmp_path / "r"
    shutil_copy(base, out)
    (out / "verdict.json").unlink()
    post.finish(out)
    v = json.loads((out / "result" / "metrics.json").read_text())["verdict"]
    assert v["source"] == "history" and v["state"] in ("iteration_limit", "converged")
    assert any("no verdict.json" in n for n in v["notes"])


# --- review round 1 (M1, M2, m1, m4, m5, m7, m8, n1, n2, n5, n6, n7) -------------------------------

def _ids(yaml_path):
    """The flat grid ids of the design voxels of a YAML, from the problem (not from checkpoint code)."""
    from topopt.problem import build_problem
    from topopt.voxelize import DESIGN
    prob = build_problem(load_config(yaml_path))
    return prob, np.flatnonzero(prob.labels.reshape(-1) == DESIGN)


def test_the_design_variables_follow_the_flat_grid_ids_of_the_design_voxels(tmp_path):
    """`design_ids` is the order of the design variables of the model. The saved mask and the mapping rely on it."""
    from topopt.optimize import Model
    prob, ids = _ids(make_case(tmp_path, h=2.0))
    m = Model(prob)
    assert np.array_equal(m.active_index[m.design_pos], ids)
    bits = C.design_mask_bits(prob.labels)
    assert np.array_equal(C.design_ids(bits, prob.labels.size), ids)
    assert bits.dtype == np.uint8 and bits.size == (prob.labels.size + 7) // 8


def test_the_design_map_keeps_new_and_dropped_voxels_apart():
    dm = C.map_design(np.array([2, 3, 5, 8, 9]), np.array([1, 3, 4, 8, 9, 10]))
    assert (dm.kept, dm.new, dm.dropped) == (3, 3, 2) and dm.src.tolist() == [1, 3, 4] and dm.dst.tolist() == [1, 3, 4]
    assert dm.apply(np.array([20.0, 30, 50, 80, 90]), 0.5).tolist() == [0.5, 30.0, 0.5, 80.0, 90.0, 0.5]
    low = dm.apply(np.arange(10.0).reshape(5, 2), np.full((6, 2), -1.0))     # (n, 1) and (n, k) arrays go by the first axis
    assert low.tolist() == [[-1, -1], [2, 3], [-1, -1], [6, 7], [8, 9], [-1, -1]]
    assert "3 of the 5 saved design voxels" in dm.text() and "3 design voxels are new" in dm.text() and "2 saved design voxels" in dm.text()


def test_the_start_asymptotes_are_the_first_ones_of_the_mma_code():
    """The first asymptotes are x -/+ 0.5 * (xmax - xmin) ([S07] (3.11)): the constant of the resume check is the one of `mma_sub`."""
    from topopt import mma_sub
    assert C.MMA_ASYINIT == mma_sub.ASYINIT == 0.5


def test_the_grid_check_stops_at_a_shape_a_voxel_size_or_an_origin_that_moved_by_half_a_voxel():
    old = {"shape": [4, 3, 2], "n_design": 20, "origin": [0.0, 0.0, 0.0], "h": 2.0}
    assert C.grid_mismatch(old, dict(old)) is None
    assert C.grid_mismatch(old, dict(old, origin=[0.9, -0.5, 0.0])) is None             # less than half a voxel: the same grid
    assert C.origin_shift(old, dict(old, origin=[0.9, -0.5, 0.0])) == pytest.approx(0.45)
    assert "origin moved by 0.50 voxels" in C.grid_mismatch(old, dict(old, origin=[0.0, 1.0, 0.0]))
    assert "the checkpoint has 20 design voxels on the grid [4, 3, 2]" in C.grid_mismatch(old, dict(old, shape=[5, 3, 2], n_design=25))
    assert "voxel size is 2 mm in the checkpoint and 1 mm now" in C.grid_mismatch(old, dict(old, h=1.0))
    assert C.grid_mismatch({"shape": [4, 3, 2]}, dict(old)) is None                      # a checkpoint without origin and h: not compared


def test_a_moved_keep_out_with_the_same_voxel_count_maps_the_saved_design_by_position(tmp_path, capsys, monkeypatch):
    """Review M1: the same count of design voxels on other voxels. The saved arrays must go onto the same places, not the same indices."""
    from test_optimize import box
    p = make_case(tmp_path, h=1.0, iterations=12)
    box(p.parent / "hole.stl", (8, 3, 3), (12, 7, 7))
    edit(p, lambda d: d["regions"].update(keep_out=["hole.stl"]))
    out = tmp_path / "r"
    killed(p, out, "after_checkpoint:6")
    old_prob, old_ids = _ids(p)
    old = C.read_checkpoint(out)
    box(p.parent / "hole.stl", (10, 3, 3), (14, 7, 7))                     # the same size, 2 mm further: 64 of 2000 voxels differ
    new_prob, new_ids = _ids(p)
    assert old_ids.size == new_ids.size and not np.array_equal(old_ids, new_ids)
    kept = np.intersect1d(old_ids, new_ids).size
    assert 0 < kept < old_ids.size
    before = tree_state(out)
    assert main(run_args(p, out, "--resume")) == 1
    err = capsys.readouterr().err
    assert f"{kept} of the {old_ids.size} saved design voxels are design voxels in this run too" in err
    assert f"{new_ids.size - kept} design voxels are new" in err and "the voxel labels are not the same" in err
    assert tree_state(out) == before
    seen = {}
    real = C.Checkpointer.restore

    def spy(self):
        got = real(self)
        st = self.mma.state
        seen.update(x=got[0].copy(), low=st.low.copy(), upp=st.upp.copy(), xold1=st.xold1.copy(), xold2=st.xold2.copy(),
                    half=self.mma.xmax - self.mma.xmin)
        return got

    monkeypatch.setattr(C.Checkpointer, "restore", spy)
    assert main(run_args(p, out, "--resume", "--allow-changed-inputs")) == 0
    in_old = np.isin(new_ids, old_ids)
    at = np.searchsorted(old_ids, new_ids[in_old])
    assert (new_ids[~in_old].size, in_old.sum()) == (new_ids.size - kept, kept)
    for name, key, start in (("x", "x", 0.3), ("low", "mma_low", 0.3 - 0.5 * seen["half"]), ("upp", "mma_upp", 0.3 + 0.5 * seen["half"]),
                             ("xold1", "mma_xold1", 0.3), ("xold2", "mma_xold2", 0.3)):
        saved = old.arrays[key].reshape(old_ids.size, -1)[:, 0]
        got = np.asarray(seen[name]).reshape(new_ids.size, -1)[:, 0]
        assert np.array_equal(got[in_old], saved[at]), name                  # a voxel in both: its own saved value
        assert np.all(got[~in_old] == start), name                           # a new voxel: the start value
    assert "were not carried" not in log(out) and (out / "job.DONE").exists()
    assert [r["iteration"] for r in history(out)] == list(range(13))
    ck = C.read_checkpoint(out)
    assert np.array_equal(C.design_ids(ck.arrays["design_mask"], new_prob.labels.size), new_ids)   # the next checkpoint holds the new mask
    point = json.loads((out / "verdict.json").read_text())["resume_points"][0]
    assert any(c.startswith("the design voxels are not the same") for c in point["changes"])


@needs_direct
def test_a_changed_force_does_not_use_the_saved_step_acceptance_solve(tmp_path):
    """Review M2: with an input change the next pass solves the design again. The saved solve belongs to the old force."""
    yaml_path = stress_yaml(tmp_path, iterations=24)
    base, out = tmp_path / "base", tmp_path / "r"
    finish(yaml_path, base)
    killed(yaml_path, out, "after_checkpoint:12")                            # pass 13 has the same beta: the checkpoint has carry_U
    assert "carry_U" in C.read_checkpoint(out).arrays
    edit(yaml_path, lambda d: d["load_cases"][0]["forces"][0].update(force=[f * 1.5 for f in d["load_cases"][0]["forces"][0]["force"]]))
    finish(yaml_path, out, "--resume", "--allow-changed-inputs")
    assert "resume: the step acceptance solve of the saved design is not used: the inputs changed" in log(out)
    ratio = [a / b for a, b in zip(history(out)[13]["compliance_cases"], history(base)[13]["compliance_cases"])]
    # the same design (the saved one) with 1.5 times the force of case 0: compliance 1.5 * 1.5 times. The stale solve would give 1.5.
    assert ratio[0] == pytest.approx(2.25, rel=1e-6) and ratio[1] == pytest.approx(1.0, rel=1e-6)


@needs_direct
def test_a_resume_with_no_input_change_still_uses_the_saved_step_acceptance_solve(tmp_path):
    yaml_path = stress_yaml(tmp_path, iterations=24)
    out = tmp_path / "r"
    killed(yaml_path, out, "after_checkpoint:12")
    edit(yaml_path, lambda d: d["settings"].update(stop_rules={"enabled": False, "window": 9}))   # a note, not a change
    finish(yaml_path, out, "--resume")
    assert "the step acceptance solve of the saved design is not used" not in log(out)


def full_state(d):
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file()}


def test_a_refused_resume_keeps_the_job_record_of_a_finished_run(case, tmp_path, capsys):
    """Review m1: a refusal writes no job file. The done marker and `job.json` of a finished run stay as they were."""
    import shutil
    yaml_path, base = case
    out = tmp_path / "r"
    shutil.copytree(base, out)
    assert (out / "job.DONE").exists()
    edit(yaml_path, lambda d: d["settings"].update(iterations=30))
    try:
        before = full_state(out)
        assert main(run_args(yaml_path, out, "--resume", "--allow-changed-inputs")) == 1
        assert "finished run longer" in capsys.readouterr().err
        assert full_state(out) == before and not (out / "job.FAILED").exists()
        assert json.loads((out / "job.json").read_text())["state"] == "done"
    finally:
        edit(yaml_path, lambda d: d["settings"].update(iterations=24))


CORE_DEFECTS = {
    "no mma_low": lambda a, m: a.pop("mma_low"),
    "mma_upp of the wrong shape": lambda a, m: a.update(mma_upp=np.zeros(4)),
    "no mma_xold1 after the first step": lambda a, m: a.pop("mma_xold1"),
    "no mma_x though it is saved": lambda a, m: m["mma"].update(x="saved"),
    "no design_mask": lambda a, m: a.pop("design_mask"),
    "a design_mask with another count": lambda a, m: a.update(design_mask=np.packbits(np.array([1, 1, 1, 0], bool))),
    "no mma.f0_ref": lambda a, m: m["mma"].pop("f0_ref"),
    "no mma.rows": lambda a, m: m["mma"].pop("rows"),
    "no mma.move": lambda a, m: m["mma"].pop("move"),
    "a row array that the file does not have": lambda a, m: m["mma"]["rows"].update({"buckling:x": {"v0": {C.ARRAY_REF: "rowvec_0"}}}),
    "a row array of rows_before that the file does not have": lambda a, m: m.update(rows_before={"buckling:x": {"v0": {C.ARRAY_REF: "rowvec_3"}}}),
    "no grid shape": lambda a, m: m["inputs"].pop("grid"),
    "counters of the wrong length": lambda a, m: m.update(step_acceptance={"counters": [1, 2]}),
}


@pytest.mark.parametrize("name", sorted(CORE_DEFECTS))
def test_a_file_that_lacks_a_core_field_goes_to_the_older_file_or_is_refused_with_the_reason(tmp_path, name):
    """Review m4: the check is made when the file is read, so a missing core field never reaches the restore."""
    C.write_checkpoint(tmp_path, arrays_of(1), meta_of(1))
    arrays, meta = arrays_of(2), meta_of(2)
    CORE_DEFECTS[name](arrays, meta)
    C.write_checkpoint(tmp_path, arrays, meta)       # the pass before is `checkpoint.prev.npz` now
    ck = C.read_checkpoint(tmp_path)
    assert (ck.file, ck.iteration) == (C.PREV_FILE, 1) and list(ck.bad) == [C.CHECKPOINT_FILE]
    assert "cannot be used" in ck.facts[0]
    (tmp_path / C.PREV_FILE).unlink()
    with pytest.raises(C.ResumeRefused) as e:
        C.read_checkpoint(tmp_path)
    assert "checkpoint.npz:" in e.value.reason and any("without --resume" in o for o in e.value.options)


def test_a_checkpoint_without_a_core_array_is_refused_before_the_folder_changes(half, capsys):
    """Review m4 (the review's experiment): `mma_low` is missing. It was an internal error after the cut of the history."""
    yaml_path, out = half
    rewrite_checkpoint(out, drop=("mma_low",))
    before = full_state(out)
    assert main(run_args(yaml_path, out, "--resume")) == 1
    err = capsys.readouterr().err
    assert "the array 'mma_low' is missing" in err and "internal error" not in err and len(err.strip().splitlines()) == 1
    assert full_state(out) == before


def test_a_zero_pass_resume_sets_the_values_that_the_end_of_the_run_reads(case, tmp_path, monkeypatch):
    """Review m7: a resume of a finished run runs no pass. Code after the loop (A3b: `robust.finish_run(m, x, beta, c_cases, out)`) needs these."""
    import sys as _sys
    from topopt import optimize
    yaml_path, base = case
    out = tmp_path / "r"
    shutil_copy(base, out)
    last = history(out)[-1]
    seen = {}
    real = optimize.build_verdict

    def spy(*a, **k):
        names = _sys._getframe(1).f_locals          # the locals of `run()` at the end of the run
        seen.update({n: names.get(n) for n in ("beta", "c", "c_cases", "vol", "gray", "change")})
        return real(*a, **k)

    monkeypatch.setattr(optimize, "build_verdict", spy)
    assert main(run_args(yaml_path, out, "--resume")) == 0
    assert seen == {"beta": last["beta"], "c": last["compliance"], "c_cases": last["compliance_cases"], "vol": last["volume_fraction"],
                    "gray": last["grayness"], "change": last["change"]}


def test_a_fresh_run_says_that_it_deletes_the_checkpoint_of_a_run_that_did_not_finish(half):
    """Review m8: no `--resume` in a folder with an unfinished checkpoint. A warning in the log; the run goes on."""
    yaml_path, out = half
    finish(yaml_path, out)
    text = log(out)
    assert "warning: the run folder had a checkpoint at pass 6 of 24" in text and "This run deletes it" in text and "--resume" in text
    assert [r["iteration"] for r in history(out)] == list(range(25)) and C.read_checkpoint(out).meta["complete"] is True
    finish(yaml_path, out)                                  # the checkpoint is complete now: nothing to warn about
    assert "had a checkpoint" not in log(out)


def test_the_history_cut_does_not_add_a_newline_for_a_line_that_it_drops(tmp_path):
    """Review n1: `newline_added` is about the last line that is kept."""
    path = tmp_path / "history.jsonl"
    path.write_bytes((json.dumps({"iteration": 0}) + "\n" + json.dumps({"iteration": 1}) + "\n" + '{"iteration": 2, "part').encode())
    cut = C.cut_history(tmp_path, 5)
    assert (cut["kept"], cut["bad_lines"], cut["newline_added"]) == (2, 1, False)
    path.write_bytes((json.dumps({"iteration": 0}) + "\n" + json.dumps({"iteration": 1}) + "\n" + json.dumps({"iteration": 2})).encode())
    cut = C.cut_history(tmp_path, 1)                         # the line without a newline (pass 2) is dropped: no newline to add
    assert (cut["kept"], cut["dropped"], cut["newline_added"]) == (2, [2], False)
    path.write_bytes((json.dumps({"iteration": 0}) + "\n" + json.dumps({"iteration": 1})).encode())
    assert C.cut_history(tmp_path, 1)["newline_added"] is True


def test_the_plan_is_in_the_log_before_the_folder_changes(half, monkeypatch):
    """Review n2: a resume that stops inside the apply step has left its facts in run.log."""
    yaml_path, out = half

    def stop(*a, **k):
        raise RuntimeError("stopped inside the apply step")

    monkeypatch.setattr(C, "cut_history", stop)
    assert main(run_args(yaml_path, out, "--resume")) == 2          # the CLI reports an internal error
    text = log(out)
    assert "resume: continue from checkpoint.npz, iteration 6 of 24" in text
    assert "resume: the run folder changes now. history.jsonl and the snapshots are cut to iteration 6" in text
    assert text.index("continue from checkpoint.npz") < text.index("the run folder changes now")


def test_the_flush_goes_to_the_file_and_to_the_folder(tmp_path, monkeypatch):
    """Review n5: the file is flushed before the renames and the folder after them."""
    import fcntl
    import stat
    monkeypatch.delattr(fcntl, "F_FULLFSYNC", raising=False)   # macOS has it: then `fsync` is not the call (next test)
    calls = []
    real = os.fsync
    monkeypatch.setattr(C.os, "fsync", lambda fd: calls.append("folder" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file") or real(fd))
    C.write_checkpoint(tmp_path, arrays_of(1), meta_of(1))
    assert calls == ["file", "folder"]
    monkeypatch.setattr(C, "FSYNC", False)
    calls.clear()
    C.write_checkpoint(tmp_path, arrays_of(2), meta_of(2))
    assert calls == []


def test_the_flush_uses_f_fullfsync_where_fcntl_has_it(tmp_path, monkeypatch):
    """Review n5: on macOS `os.fsync` does not flush the drive cache; `fcntl.F_FULLFSYNC` does."""
    import fcntl
    asked = []
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, cmd, *a: asked.append(cmd) or 0)
    monkeypatch.setattr(C.os, "fsync", lambda fd: asked.append("fsync"))
    C.write_checkpoint(tmp_path, arrays_of(1), meta_of(1))
    assert asked == [51, 51]                                 # the file and the folder, and no `fsync`
    asked.clear()
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, cmd, *a: (_ for _ in ()).throw(OSError("not supported")))
    C.write_checkpoint(tmp_path, arrays_of(2), meta_of(2))
    assert asked == ["fsync", "fsync"]                       # a file system that refuses it gets `fsync`


def test_the_same_region_file_under_another_path_is_one_fact_with_the_same_content(tmp_path):
    """Review n6: the YAML moved one level deeper with the same STL files."""
    old = {"regions": {"design:a/design.stl": "a" * 64, "fixed:p.stl": "b" * 64}}
    new = {"regions": {"design:../a/design.stl": "a" * 64, "fixed:p.stl": "c" * 64}}
    changes, _ = C.diff_inputs(old, new)
    assert "region file design:a/design.stl is now design:../a/design.stl: the content is the same" in changes[0]
    assert "no longer in the YAML" not in " ".join(changes) and "is new in the YAML" not in " ".join(changes)
    assert any("region file fixed:p.stl has other content" in c for c in changes)
    changes, _ = C.diff_inputs({"regions": {"design:a.stl": "a" * 64}}, {"regions": {"load:a.stl": "a" * 64}})   # another role: two facts
    assert sorted(c.split(" is ")[0] for c in changes) == ["region file design:a.stl", "region file load:a.stl"]


def test_the_effective_settings_hold_every_field_of_a_load_case(tmp_path):
    """Review n7: a field that a later item adds to `LoadCase` is in the effective settings without a change here."""
    import dataclasses
    from topopt.config import LoadCase
    config = load_config(make_case(tmp_path, h=2.0))
    case = config.load_cases[0]
    got = C.effective_settings(config)["load_cases"][case.name]
    assert set(got) == {f.name for f in dataclasses.fields(LoadCase)} and got["forces"][0]["region"] == "tip"


def test_the_resume_removes_every_result_folder_of_the_earlier_run(half, monkeypatch):
    """After #21 `snapshot.RUN_DIRS` has `verify/`: the report of the old `final.stl` must not stay next to the new one."""
    from topopt import snapshot
    yaml_path, out = half
    monkeypatch.setattr(snapshot, "RUN_DIRS", ("snapshots", "result", "verify"))
    (out / "verify").mkdir()
    (out / "verify" / "verify.json").write_text("{}")
    finish(yaml_path, out, "--resume")
    assert not (out / "verify").exists() and "Removed verify/ of the earlier run" in log(out)


# --- a region file outside the work folder (the key rule of runinfo.region_files, added with the D2 refresh) ------------

def test_the_keys_of_a_region_file_outside_hold_no_folder_name_and_a_change_is_named(tmp_path, tmp_path_factory):
    from test_optimize import box
    away = tmp_path_factory.mktemp("jdoe-parts")
    p = make_case(tmp_path, h=2.0, iterations=6)
    shutil.copyfile(tmp_path / "tip.stl", away / "tip.stl")
    edit(p, lambda d: d["regions"].update(load={"tip": str(away / "tip.stl")}))        # an absolute path in the YAML
    old = inputs(p)
    (key,) = [k for k in old["regions"] if k.startswith("load:")]
    assert re.fullmatch(r"load:<outside>/[0-9a-f]{12}/tip\.stl", key) and old["regions"][key] == C.sha256_file(away / "tip.stl")
    assert not any("jdoe-parts" in k for k in old["regions"])
    assert C.diff_inputs(old, inputs(p)) == ([], [])
    box(away / "tip.stl", (16, 4, 0), (20, 6, 3))
    changes, _ = C.diff_inputs(old, inputs(p))
    assert len(changes) == 1 and changes[0].startswith(f"region file {key} has other content") and "jdoe-parts" not in changes[0]


def test_a_link_is_keyed_by_the_name_in_the_yaml(tmp_path):
    p = make_case(tmp_path, h=2.0, iterations=6)
    (tmp_path / "store").mkdir()
    (tmp_path / "plate.stl").rename(tmp_path / "store" / "plate1.stl")
    (tmp_path / "plate.stl").symlink_to(Path("store") / "plate1.stl")
    old = inputs(p)
    assert "fixed:plate.stl" in old["regions"] and not any("store" in k for k in old["regions"])
    (tmp_path / "store" / "plate2.stl").write_bytes((tmp_path / "store" / "plate1.stl").read_bytes())
    (tmp_path / "plate.stl").unlink()
    (tmp_path / "plate.stl").symlink_to(Path("store") / "plate2.stl")        # a new target with the same bytes
    assert C.diff_inputs(old, inputs(p)) == ([], [])


def test_a_resume_with_a_region_file_outside_goes_on_and_a_change_of_it_is_refused(work, tmp_path, tmp_path_factory, capsys):
    from test_optimize import box
    away = tmp_path_factory.mktemp("jdoe-parts")
    shutil.copyfile(work.parent / "tip.stl", away / "tip.stl")
    (work.parent / "tip.stl").unlink()
    edit(work, lambda d: d["regions"].update(load={"tip": str(away / "tip.stl")}))
    out = tmp_path / "r"
    killed(work, out, "after_checkpoint:6")
    regions = C.read_checkpoint(out).meta["inputs"]["regions"]
    (key,) = [k for k in regions if k.startswith("load:")]
    assert re.fullmatch(r"load:<outside>/[0-9a-f]{12}/tip\.stl", key) and not any("jdoe-parts" in k for k in regions)
    original = (away / "tip.stl").read_bytes()
    box(away / "tip.stl", (18.1, 4, 0), (20, 6, 3))                  # other bytes, the same voxels
    before = tree_state(out)
    assert main(run_args(work, out, "--resume")) == 1
    err = capsys.readouterr().err
    assert f"region file {key} has other content" in err and "--allow-changed-inputs" in err and "jdoe-parts" not in err
    assert tree_state(out) == before
    (away / "tip.stl").write_bytes(original)                         # the same file again: the resume goes on
    resume(work, out)
    assert "inputs changed" not in log(out)
    assert json.loads((out / "status.json").read_text())["state"] == "done"
