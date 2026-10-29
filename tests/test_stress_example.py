"""The committed stress example (examples/hook/hook_stress.yaml) runs to the end (A2).

The test uses 2.4 mm voxels and 20 iterations to keep the run short. It does not test the final
mass or the stress: at this size the design is not converged. It tests that the YAML loads, that the run
ends with exit 0, that the history has the stress numbers, that `metrics.json` has the stress section
for each case, and that a stage that advances while a row is not met is recorded as a fact.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from topopt.cli import main

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "hook"


def test_example_runs_to_the_end(tmp_path):
    work = tmp_path / "hook"
    shutil.copytree(EXAMPLE, work)
    d = yaml.safe_load((work / "hook_stress.yaml").read_text())
    assert d["settings"]["objective"] == "mass"
    d["settings"].update(voxel_size=2.4, iterations=20, beta_schedule=[[0, 1], [8, 2], [14, 4]])
    (work / "small.yaml").write_text(yaml.safe_dump(d))
    out = tmp_path / "run"
    assert main(["check", str(work / "small.yaml")]) == 0
    assert main(["run", str(work / "small.yaml"), "--out", str(out), "--quiet"]) == 0

    history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert len(history) == 21
    last = history[-1]
    assert last["objective"]["name"] == "mass"
    rows = {c["name"]: c for c in last["constraints"]}
    assert set(rows) == {"stress:hang", "stress:pull_off"}
    for c in rows.values():           # each case: the row is met or the unmet row is in the report
        assert c["relaxed_peak_MPa"] > 0 and c["stress_limit_MPa"] > 0
    stages = [r["stage"] for r in history if "stage" in r]
    assert [s["to_beta"] for s in stages] == [1.0, 2.0, 4.0]
    assert stages[0]["feasible"] is None
    assert all(isinstance(s["feasible"], bool) for s in stages[1:])   # a fact for each stage end

    st = json.loads((out / "result" / "metrics.json").read_text())["stress"]
    assert set(st["cases"]) == {"hang", "pull_off"}
    assert st["cases"]["pull_off"]["stress_limit_MPa"] == 0.5
    assert st["cases"]["pull_off"]["limit_from"] == "stress_limit"
    assert st["cases"]["pull_off"]["allowable_mpa"] == 40.0     # the report uses it for the achieved safety factor
    assert st["cases"]["hang"]["stress_limit_MPa"] == 1.5
    for name, c in st["cases"].items():
        assert c["relaxed"]["label"] == "relaxed, not physical"
        phys = c["physical"]   # after 20 iterations a coarse design can have no voxel with rho >= 0.5: the report says so
        assert (phys["voxels"] > 0 and phys["peak_MPa"] > 0) or "No design voxel" in c["achieved_safety_factor"]["text"]
        met = c["relaxed"]["met"]              # the definition of the report: g <= 0.005 (R1-4)
        unmet_logged = any(f"stress:{name}" in line and "not met" in line
                           for line in (out / "run.log").read_text().splitlines())
        assert met or unmet_logged, f"{name}: the row is neither met nor reported as unmet"
        assert c["relaxed"]["met_means"].endswith("Not a statement about the part")
        assert c["summary"].startswith("the optimizer row is") and "oxel estimate" in c["summary"]
        assert any(line.startswith(f"fact: stress {name}: the optimizer row is") for line in
                   (out / "run.log").read_text().splitlines()), f"{name}: no summary fact in the run log"


@pytest.mark.slow
def test_example_meets_both_rows_at_one_thread(tmp_path):
    """The example as it is (2.4 mm, 220 passes) ends with both stress rows met, at 1 thread (A2 R1, REVIEW M1).

    The example has the `pull_off` limit 0.5 MPa. With 5 MPa the row was met as well, but the members were about one voxel thick:
    `final.stl` was the wall plate alone (15 g) and the relaxed rows said "met" for it. With 1 MPa `final.stl` is one body, but a thin neck joins the tip to the plate (`topopt verify`:
    1 N/mm). So the test also asks for one body of `final.stl` that holds the load region and a fixed region, from the load path
    facts of the report (A2 F1). The row is met when `g <= 5e-3`. Without the step acceptance the same kind of run swung around the limit.
    The run is a child process, so that the thread variables apply before numpy and MKL load.
    """
    out = tmp_path / "run"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    done = subprocess.run([sys.executable, "-m", "topopt", "run", str(EXAMPLE / "hook_stress.yaml"), "--out", str(out),
                           "--quiet"], env=env, capture_output=True, text=True, timeout=1500)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert len(history) == 221
    end = {c["name"]: c["g"] for c in history[-1]["constraints"]}
    assert set(end) == {"stress:hang", "stress:pull_off"}
    for name, g in end.items():
        assert g <= 5e-3, f"{name}: end g {g:+.4f} is above the NOT_MET threshold 5e-3"
    steps = [r["step_acceptance"] for r in history if "step_acceptance" in r]
    assert steps and not any(not s["accepted"] for s in steps), "a pass took a step after all tries were rejected"
    st = json.loads((out / "result" / "metrics.json").read_text())["stress"]
    assert all(c["relaxed"]["met"] for c in st["cases"].values())
    assert all(e["feasible"] for e in st["stage_ends"])
    paths = st["load_paths"]                  # the part carries the load: one body joins the load region and the plate
    assert paths["final_stl"]["bodies"] == 1, paths["final_stl"]
    tip = paths["regions"]["tip"]
    assert tip["thresholded_design"]["connected_to_fixed"] and tip["final_stl"]["connected_to_fixed"], tip
    assert all(c["load_path"]["all_connected"] for c in st["cases"].values())
    assert not any("not a part that carries the load" in c["summary"] for c in st["cases"].values())
