"""The dashboard draws one box for each connected part of label 3 (load regions and solid regions, A3a) and of label 2 (fixed regions).

`boxes.js` has no DOM and no three.js, so Node runs it. The test is skipped where there is no `node`.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

BOXES = Path(__file__).resolve().parents[1] / "topopt" / "dashboard" / "js" / "boxes.js"
NODE = shutil.which("node")

def boxes_of(lab: np.ndarray, code: int = 3):
    nx, ny, nz = lab.shape
    script = ("import { componentBoxes } from %s;\n"
              "import fs from 'node:fs';\n"
              "const d = JSON.parse(fs.readFileSync(0, 'utf8'));\n"
              "console.log(JSON.stringify(componentBoxes(Uint8Array.from(d.lab), d.nx, d.ny, d.nz, d.code)));\n"
              % json.dumps(BOXES.as_uri()))
    data = json.dumps({"lab": lab.reshape(-1).tolist(), "nx": nx, "ny": ny, "nz": nz, "code": code})
    done = subprocess.run([NODE, "--input-type=module", "-e", script], input=data, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-500:]
    return json.loads(done.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_load_and_a_solid_apart_have_two_boxes_and_touching_ones_have_one():
    lab = np.zeros((8, 6, 5), np.uint8)
    lab[0:2, 1:3, 0:2] = 3          # a load region
    lab[5:8, 2:6, 1:4] = 3          # a solid region, apart
    lab[3, 3, 3] = 1                # other labels are not boxed
    assert boxes_of(lab) == [[0, 1, 0, 1, 2, 1], [5, 2, 1, 7, 5, 3]]
    lab[2:5, 2, 1] = 3              # a bridge: the two parts touch, one part
    assert boxes_of(lab) == [[0, 1, 0, 7, 5, 3]]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_only_a_face_neighbour_connects_and_no_voxel_means_no_box():
    lab = np.zeros((4, 4, 4), np.uint8)
    lab[0, 0, 0] = 3
    lab[1, 1, 1] = 3                # a corner neighbour only: a separate part
    assert boxes_of(lab) == [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
    assert boxes_of(lab, code=2) == []


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_two_fixed_feet_apart_have_two_boxes_and_not_one_slab_between_them():
    """A bridge on two end feet: the fixed regions (label 2) were one bounding box under the whole span."""
    lab = np.zeros((12, 4, 5), np.uint8)
    lab[0:2, :, 0:2] = 2            # the left foot
    lab[10:12, :, 0:2] = 2          # the right foot
    lab[0:12, :, 4] = 3             # the deck load, one part
    assert boxes_of(lab, code=2) == [[0, 0, 0, 1, 3, 1], [10, 0, 0, 11, 3, 1]]
    assert boxes_of(lab, code=3) == [[0, 0, 4, 11, 3, 4]]


def test_the_scene_boxes_the_fixed_regions_part_by_part():
    app = (BOXES.parent / "app.js").read_text()
    assert "componentBoxes(lab, nx, ny, nz, 2).forEach((b) => addBox(b, 0x3b82f6));" in app
    assert "addBox(bb[2]" not in app
