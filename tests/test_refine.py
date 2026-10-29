"""Tests for the refine step: stress, the markup page, CLI, run hook and dashboard routes."""

import json
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml
from scipy.spatial import cKDTree

import topopt.refine
from topopt import dashboard_server as ds
from topopt.cli import main
from topopt.config import load_config
from topopt.fe import LOCAL_NODES
from topopt.problem import build_problem
from topopt.refine.build import build_refine
from topopt.refine.page_scripts import HERE as REFINE_DIR, OLD_CDN_TAGS, VENDOR_SCRIPTS, script_text
from topopt.refine.stress import compute_stress, isotropic_D, strain_matrix_centre, von_mises
from test_optimize import box, make_case

ROOT = Path(__file__).resolve().parent.parent


def read_meta(page: Path) -> dict:
    m = re.search(r"var META = (\{.*?\});\n", page.read_text(encoding="utf-8"))
    assert m, "META not found in the page"
    return json.loads(m.group(1))


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A fast real run. The run hook makes the refine page."""
    tmp = tmp_path_factory.mktemp("refine")
    yaml_path = make_case(tmp, h=1.0, iterations=30)
    out = tmp / "run"
    assert main(["run", str(yaml_path), "--out", str(out), "--quiet"]) == 0
    return yaml_path, out


# -- stress maths --
def test_strain_matrix_recovers_a_uniform_strain():
    h = 1.7
    rng = np.random.default_rng(3)
    eps = rng.normal(size=(3, 3))
    eps = 0.5 * (eps + eps.T)
    x = LOCAL_NODES * h                      # node positions of one voxel
    u = x @ eps.T                            # u = eps x for each node
    got = strain_matrix_centre(h) @ u.reshape(-1)
    want = [eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[0, 1], 2 * eps[1, 2], 2 * eps[0, 2]]
    assert np.allclose(got, want, atol=1e-12)


def test_von_mises_known_states():
    uniaxial = np.array([[7.0, 0, 0, 0, 0, 0]])
    shear = np.array([[0, 0, 0, 4.0, 0, 0]])
    hydro = np.array([[5.0, 5.0, 5.0, 0, 0, 0]])
    assert np.allclose(von_mises(uniaxial), 7.0)
    assert np.allclose(von_mises(shear), 4.0 * np.sqrt(3.0))
    assert np.allclose(von_mises(hydro), 0.0)
    # D0 maps a uniaxial stress strain state back to the same stress.
    E, nu = 3500.0, 0.35
    eps = np.array([1.0, -nu, -nu, 0, 0, 0]) * 7.0 / E
    assert np.allclose(isotropic_D(E, nu) @ eps, [7.0, 0, 0, 0, 0, 0], atol=1e-9)


def test_bar_in_tension_gives_force_over_area(tmp_path):
    """Bar 4 x 4 mm, 16 N along x: sigma = 1 MPa away from the clamp and the load."""
    box(tmp_path / "design.stl", (0, 0, 0), (24, 4, 4))
    box(tmp_path / "plate.stl", (-2, 0, 0), (0, 4, 4))
    box(tmp_path / "tip.stl", (22, 0, 0), (24, 4, 4))
    d = {"name": "bar",
         "regions": {"design": "design.stl", "fixed": ["plate.stl"], "load": {"tip": "tip.stl"}},
         "load_cases": [{"name": "pull", "forces": [{"region": "tip", "force": [16.0, 0, 0]}]}],
         "settings": {"voxel_size": 1.0, "volume_fraction": 0.5, "iterations": 1}}
    (tmp_path / "bar.yaml").write_text(yaml.safe_dump(d))
    prob = build_problem(load_config(tmp_path / "bar.yaml"))
    res = compute_stress(prob, np.full(prob.labels.size, 255, np.uint8))
    x = res["centers"][:, 0]
    mid = res["vm_cases"][0][(x > 8) & (x < 16)]
    assert mid.size > 100
    assert abs(np.median(mid) - 1.0) < 0.02
    assert np.all(res["rho"] >= 0.5)
    assert np.array_equal(res["vm_max"], res["vm_cases"][0])


# -- the run hook and the files --
def test_run_makes_refine_page(tiny_run):
    _yaml, out = tiny_run
    page = out / "result" / "refine.html"
    assert page.is_file() and (out / "result" / "stress.npz").is_file()
    meta = read_meta(page)
    assert meta["stress_max_mpa"] > 0
    assert meta["stem"] == "cantilever" and meta["density_g_cm3"] == 1.24
    assert meta["faces_embedded"] == meta["faces_source"] > 0   # below 150k faces: no decimation
    text = page.read_text(encoding="utf-8")
    assert not any(p in text for p in ("%%META%%", "%%VERTS%%", "%%FACES%%", "%%STRESS%%", "%%TITLE%%"))
    # three.js r128 and its OrbitControls are in the page, byte for byte: no file comes from another host (#396).
    for rel in VENDOR_SCRIPTS.values():
        assert (REFINE_DIR / rel).read_text(encoding="utf-8") in text, rel
    assert "<script src=" not in text and "cdn.jsdelivr.net" not in text
    # Nothing of the steel scale (0 - 500 MPa, 7.8e-3 g/mm3) stays in the page.
    assert "7.8e-3" not in text and "65535*500" not in text and "stress[i]/500" not in text
    assert "frozen" not in text and "markup_tool" not in text
    assert not list(out.rglob("*.tmp")) and not list(out.rglob("*.tmp.npz"))
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "done" and status["step"] == "done"
    log = (out / "run.log").read_text().splitlines()
    i_done = next(i for i, t in enumerate(log) if t.startswith("done in"))
    i_ref = next(i for i, t in enumerate(log) if t.startswith("refine:"))
    assert i_done < i_ref and sum("done in" in t for t in log) == 1


def test_the_vendored_scripts_fit_inline_into_the_page():
    """The page holds three.js r128 and its OrbitControls inline (#396). The code of the page uses the r128 API (the global `THREE`,
    `THREE.OrbitControls`). In a script block, "</script" ends the block early, and "<!--" with "<script" can keep it open past its end
    tag; a file with a template mark would get the data of the run."""
    template = (REFINE_DIR / "template.html").read_text(encoding="utf-8")
    marks = set(re.findall(r"%%[A-Z_]+%%", template))
    assert set(VENDOR_SCRIPTS) <= marks
    for mark, rel in VENDOR_SCRIPTS.items():
        assert template.count(mark) == 1, mark
        text = (REFINE_DIR / rel).read_text(encoding="utf-8")
        assert not re.search(r"</?script|<!--", text, re.IGNORECASE), rel
        assert not any(m in text for m in marks), rel
    three = (REFINE_DIR / VENDOR_SCRIPTS["%%THREE_JS%%"]).read_text(encoding="utf-8")
    assert re.search(r'const \w+="128"', three[:600]), "the page code is written for three.js r128"
    assert "THREE.OrbitControls = OrbitControls" in (REFINE_DIR / VENDOR_SCRIPTS["%%ORBIT_CONTROLS_JS%%"]).read_text(encoding="utf-8")


def test_stress_npz_fits_the_mesh(tiny_run):
    _yaml, out = tiny_run
    z = np.load(out / "result" / "stress.npz")
    run = json.loads((out / "run.json").read_text())
    h = run["h"]
    assert list(z["case_names"]) == ["down", "side"]
    n = len(z["centers"])
    assert z["centers"].shape == (n, 3) and z["rho"].shape == (n,)
    assert np.all(z["rho"] >= 0.5)
    for k in ("vm_case_0", "vm_case_1", "vm_max"):
        assert z[k].shape == (n,) and np.all(np.isfinite(z[k])) and np.all(z[k] >= 0)
    assert np.array_equal(z["vm_max"], np.maximum(z["vm_case_0"], z["vm_case_1"]))
    # Same frame as final.stl: the centres lie in the mesh box (one voxel of margin).
    mesh = trimesh.load(str(out / "result" / "final.stl"), process=False)
    lo, hi = mesh.bounds
    assert np.all(z["centers"] > lo - h) and np.all(z["centers"] < hi + h)
    # Each mesh vertex has a voxel centre close by.
    d, _ = cKDTree(z["centers"]).query(np.asarray(mesh.vertices)[::20])
    assert np.median(d) < h and d.max() < 2.5 * h


def test_compliance_matches_the_optimizer(tiny_run):
    """One solve of the last snapshot gives the compliance of the last iteration (uint8 rounding)."""
    _yaml, out = tiny_run
    z = np.load(out / "result" / "stress.npz")
    last = json.loads((out / "history.jsonl").read_text().splitlines()[-1])["compliance_cases"]
    assert np.allclose(z["compliance_cases"], last, rtol=0.03)


def test_cli_refine_rebuilds_from_the_yaml(tiny_run, capsys):
    _yaml, out = tiny_run
    first = np.load(out / "result" / "stress.npz")["vm_max"]
    assert main(["refine", str(out)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(out / "result" / "refine.html")
    again = np.load(out / "result" / "stress.npz")["vm_max"]
    assert np.allclose(first, again, rtol=1e-9)


def test_cli_refine_stops_when_the_yaml_changed(tmp_path, capsys):
    p = make_case(tmp_path, h=2.0, iterations=2)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet"]) == 0
    d = yaml.safe_load(p.read_text())
    d["settings"]["voxel_size"] = 1.0
    p.write_text(yaml.safe_dump(d))
    capsys.readouterr()
    assert main(["refine", str(out)]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("error:") and "changed" in err[0]


def test_cli_refine_without_a_result(tmp_path, capsys):
    (tmp_path / "run.json").write_text("{}")
    assert main(["refine", str(tmp_path)]) == 1
    assert "final mesh" in capsys.readouterr().err


def test_refine_failure_is_one_warning_and_the_run_still_ends_ok(tmp_path, monkeypatch, opened_urls):
    def boom(*a, **k):
        raise RuntimeError("boom\nsecond line")

    monkeypatch.setattr(topopt.refine, "build_refine", boom)
    p = make_case(tmp_path, h=2.0, iterations=2)
    out = tmp_path / "r"
    assert main(["run", str(p), "--out", str(out), "--quiet", "--dashboard", "--port", "0",
                 "--no-wait"]) == 0
    log = (out / "run.log").read_text().splitlines()
    warn = [t for t in log if "refine failed" in t]
    assert len(warn) == 1 and warn[0].startswith("warning:") and "boom second line" in warn[0]
    assert json.loads((out / "status.json").read_text())["state"] == "done"
    assert (out / "result" / "final.stl").is_file()
    assert not (out / "result" / "refine.html").exists()
    assert not any(u.endswith("/refine") for u in opened_urls)  # a failed refine opens nothing


def test_dashboard_run_opens_the_refine_page(tmp_path, capsys, opened_urls):
    """After a --dashboard run the browser opens /refine, on the real port of the server."""
    p = make_case(tmp_path, h=2.0, iterations=2)
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    port = busy.getsockname()[1]
    try:
        assert main(["run", str(p), "--out", str(tmp_path / "r"), "--quiet", "--dashboard",
                     "--port", str(port), "--no-wait"]) == 0
    finally:
        busy.close()
    out = capsys.readouterr().out
    real = int(re.search(r"Dashboard: http://127\.0\.0\.1:(\d+)/", out).group(1))
    assert real != port                                   # the busy port was skipped
    assert opened_urls == [f"http://127.0.0.1:{real}/", f"http://127.0.0.1:{real}/refine"]
    # The server of the run is still on (daemon thread) and serves the page.
    code, headers, body = get(f"http://127.0.0.1:{real}/refine")
    assert code == 200 and headers["Content-Type"].startswith("text/html")
    assert body == (tmp_path / "r" / "result" / "refine.html").read_bytes()


def test_run_without_dashboard_opens_nothing(tiny_run, opened_urls):
    assert opened_urls == []


# -- dashboard server --
def test_refine_routes(tmp_path):
    server = ds.start_in_thread(tmp_path, port=0)
    base = server.url.rstrip("/")
    try:
        code, _h, _b = get(base + "/refine")
        assert code == 404
        assert json.loads(get(base + "/api/refine")[2]) == {"exists": False}
        (tmp_path / "result").mkdir()
        (tmp_path / "result" / "refine.html").write_bytes(b"<html>refine page</html>")
        code, headers, body = get(base + "/refine")
        assert code == 200 and body == b"<html>refine page</html>"
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert json.loads(get(base + "/api/refine")[2]) == {"exists": True}
        assert get(base + "/refine/x")[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_the_dashboard_serves_the_page_of_an_older_topopt_with_the_copied_scripts(tmp_path):
    """A page that an older topopt made has the two CDN tags. The dashboard puts the copied files in their place (#396)."""
    old = (b"<html><head><title>Refine</title></head><body><div id=\"err\"></div>\n" + OLD_CDN_TAGS["%%THREE_JS%%"] + b"\n"
           + OLD_CDN_TAGS["%%ORBIT_CONTROLS_JS%%"] + b"\n<script>\nvar META = {};\n</script>\n</body></html>\n")
    (tmp_path / "result").mkdir()
    (tmp_path / "result" / "refine.html").write_bytes(old)
    server = ds.start_in_thread(tmp_path, port=0)
    try:
        code, _h, body = get(server.url.rstrip("/") + "/refine")
    finally:
        server.shutdown()
        server.server_close()
    assert code == 200 and b"<script src=" not in body
    # No address but the namespace name of the copied three.js: the old page loads nothing from another host.
    assert set(re.findall(rb"https?://[^\s\"'<>()\\]+", body)) == {b"http://www.w3.org/1999/xhtml"}
    three, orbit = (b"<script>\n" + script_text(m).encode() + b"\n</script>" for m in ("%%THREE_JS%%", "%%ORBIT_CONTROLS_JS%%"))
    assert body.index(three) < body.index(orbit) < body.index(b"var META")
    assert (tmp_path / "result" / "refine.html").read_bytes() == old   # the file of the run stays as it is


def test_optimizer_and_post_never_run_the_stress_step():
    for name in ("optimize.py", "post.py"):
        text = (ROOT / "topopt" / name).read_text()
        assert "refine" not in text and "compute_stress" not in text
