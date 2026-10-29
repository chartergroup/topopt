"""A3a x B2 merge gate (patch `a3a-x-b2`): `verify` compares the solid region files of the YAML with `run.json` like the other regions.

The check is about facts. A changed list of solid STLs means that the YAML describes another part than the run. A run.json
without a `solid` key (a run from before A3a) is skipped, as for the other keys that run.json does not hold.
"""
import json

import yaml

from test_solid_filter import CAP, RIB, write_case
from topopt.verify.run import compare_yaml_run, read_run


def _run_folder(tmp_path, solids):
    """A run folder with the `run.json` that `topopt run` writes for the keys that the comparison reads (no optimizer run)."""
    case = tmp_path / "case"
    case.mkdir()
    yaml_path = write_case(case, solids=solids)
    d = yaml.safe_load(yaml_path.read_text())
    run = tmp_path / "run"
    (run / "result").mkdir(parents=True)
    (run / "result" / "final.stl").write_bytes(b"solid x\nendsolid x\n")          # only its hash is read
    regions = {"design": d["regions"]["design"], "fixed": d["regions"]["fixed"], "keep_out": [],
               "solid": d["regions"].get("solid", []), "load": d["regions"]["load"]}
    (run / "run.json").write_text(json.dumps({"name": d["name"], "regions": regions, "yaml_path": str(yaml_path)}))
    return run, yaml_path


def test_the_same_solid_files_match(tmp_path):
    run, yaml_path = _run_folder(tmp_path, [CAP, RIB])
    out = compare_yaml_run(read_run(run, yaml_path=yaml_path))
    assert out["matches"] and out["differences"] == []


def test_a_changed_solid_list_is_a_fact(tmp_path):
    run, yaml_path = _run_folder(tmp_path, [CAP, RIB])
    d = json.loads((run / "run.json").read_text())
    d["regions"]["solid"] = ["solid0.stl"]                                        # the run had one solid file less
    (run / "run.json").write_text(json.dumps(d))
    out = compare_yaml_run(read_run(run, yaml_path=yaml_path))
    assert not out["matches"]
    assert out["differences"] == ["solid region files: run.json has ['solid0.stl'], the YAML has ['solid0.stl', 'solid1.stl']"]


def test_a_run_json_without_a_solid_key_is_skipped(tmp_path):
    run, yaml_path = _run_folder(tmp_path, [CAP])
    d = json.loads((run / "run.json").read_text())
    del d["regions"]["solid"]                                                     # a run from before A3a
    (run / "run.json").write_text(json.dumps(d))
    assert compare_yaml_run(read_run(run, yaml_path=yaml_path))["matches"]
