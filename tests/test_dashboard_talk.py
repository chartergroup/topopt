"""P9-8 (D12): the demo screen of the dashboard (the code panel and the row "Pull-off stiffness per gram") shows only when the run
has a load case named `pull_off`, or when the page address has `?demo=1`. The rule is in `topopt/dashboard/js/talk.js`, which has no
DOM and no three.js, so Node runs it. The five tests that run Node are skipped where there is no `node`; the text test of the page
(HTML and JS source) runs everywhere.

The page itself (app.js) needs WebGL; the manual check of the page is in the report of P9-8.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "topopt" / "dashboard"
TALK = DASH / "js" / "talk.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_js(expression: str):
    script = (f"import {{ talkScreen, partFileText }} from {json.dumps(TALK.as_uri())};\n"
              f"console.log(JSON.stringify({expression}));\n")
    done = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-500:]
    return json.loads(done.stdout)


def cases(*names):
    return {"load_cases": [{"name": n, "weight": 1.0, "forces": []} for n in names]}


def talk(search, run):
    return run_js(f"talkScreen({json.dumps(search)}, {json.dumps(run)})")


@needs_node
def test_a_run_with_a_pull_off_case_shows_the_talk_screen():
    assert talk("", cases("hang", "pull_off")) is True
    assert talk("", cases("pull_off")) is True


@needs_node
def test_a_run_without_a_pull_off_case_does_not():
    assert talk("", cases("hang", "side")) is False
    assert talk("", cases("pull_off_x", "Pull_Off", "pull-off")) is False          # the name is exact
    assert talk("", {"load_cases": []}) is False
    assert talk("", {}) is False
    assert talk("", None) is False                                                  # no run yet (the waiting screen)


@needs_node
def test_the_demo_switch_shows_it_for_any_run_and_for_no_run():
    assert talk("?demo=1", cases("hang")) is True
    assert talk("?demo=1", None) is True
    assert talk("?x=2&demo=1", cases("hang")) is True


@needs_node
def test_a_demo_value_other_than_1_is_not_a_switch():
    for search in ("?demo=0", "?demo=", "?demo", "?demo=true", "?demo=11", "?mode=demo"):
        assert talk(search, cases("hang")) is False, search


@needs_node
def test_the_part_file_line_names_the_run_folder_and_the_file_and_waits_for_the_file():
    info = {"file": "result/final.stl", "run_dir": "/w/runs/live", "run_name": "live", "exists": True}
    assert run_js(f"partFileText({json.dumps(info)})") == "live/result/final.stl"
    assert run_js(f"partFileText({json.dumps({**info, 'exists': False})})") is None
    assert run_js("partFileText(null)") is None
    assert run_js("partFileText({})") is None


def test_the_page_starts_with_the_pull_off_row_hidden_and_has_the_part_file_line():
    html = (DASH / "index.html").read_text()
    assert 'id="row-perg" hidden' in html and 'id="m-perg"' in html
    assert 'id="partfile" hidden' in html and 'id="m-file"' in html and "Part file" in html
    js = (DASH / "js" / "app.js").read_text()
    assert 'import { partFileText, talkScreen } from "./talk.js";' in js
