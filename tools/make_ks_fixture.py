"""Make tests/data/ks_pipeline_fixture.json from the pipeline code (A2 equivalence fixture).

The pipeline file `pipeline/lib/pp_core_final.py` of the earlier pipeline project cannot be imported here
(it needs scikit-fem and the code of that project). So this script reads the lines of the file that make the
relaxed stress, the KS aggregate, the explicit term and the KS scale, checks that they read as expected,
and runs exactly those lines on random arrays. The result is the reference for
`tests/test_ks_stress.py::test_aggregation_equals_pipeline_fixture`.

Run:  python tools/make_ks_fixture.py <path-to-pipeline-checkout> [out.json]
Default out: tests/data/ks_pipeline_fixture.json. The fixture is committed; this script is the proof
of where it came from. It never writes into the pipeline folder.

The fixture holds the pipeline matrix `D0` (lines 312-314), the arrays of the relaxed stress, the KS aggregate,
the explicit term, the adjoint right-hand side `g6` (line 472), the row and the KS scale. All come from the
pipeline lines, run by `exec` after a text check. Nothing is typed by hand.

The equivalence is at the aggregation layer only. The pipeline has P1 tets (constant strain per tet),
the demo has hex8 voxels (strain at the centre). The arrays here are strain rows, not a mesh.
"""
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SRC = "pipeline/lib/pp_core_final.py"
REV = "d43bcf3"
# (first line, last line, a text that the first line must contain)
BLOCKS = {
    "d0": (312, 314, "self.D0 = cfg.E0 * np.block("),
    "relaxed": (406, 414, "fac = np.maximum(rho_full, 1e-12) ** self.cfg.stress_power"),
    "ks": (466, 470, "z = vm[self.active] / lim"),
    "g6": (472, 472, "g6 = (w / lim)[:, None] * fs[:, None] * (dvm_ds @ self.D0)"),
    "explicit": (474, 474, "explicit = w * vm / lim * cfg.stress_power / np.maximum(rho, 1e-12)"),
    "row": (477, 477, '{"value": c * S - 1.0}'),
    "scale": (607, 607, 't["zmax"] / t["ks"]'),
}


def block(lines, name):
    a, b, must = BLOCKS[name]
    text = "\n".join(lines[a - 1:b])
    if must not in text:
        raise SystemExit(f"{SRC} lines {a}-{b} do not contain {must!r}: the pipeline code changed")
    return textwrap.dedent(text)


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "tests/data/ks_pipeline_fixture.json"
    lines = (root / SRC).read_text().splitlines()
    code = {k: block(lines, k) for k in BLOCKS}
    print("row line:", code["row"])
    E0, nu = 3500.0, 0.35
    ns0 = {"np": np, "cfg": SimpleNamespace(E0=E0, nu=nu), "self": SimpleNamespace()}
    exec(code["d0"], ns0)                       # lam0, mu0 and self.D0 of the pipeline
    D0 = ns0["self"].D0
    cases = []
    spec = [  # name, n, seed, strain scale, limit, rho kind, scale c (None: the update rule)
        ("random_near_limit", 60, 1, 2e-3, 8.0, "uniform", None),
        ("peak_far_above_limit", 60, 2, 2e-2, 8.0, "uniform", None),
        ("far_below_limit_c_small", 100, 3, 1e-4, 40.0, "uniform", None),
        ("tiny_rho_floor", 60, 4, 3e-3, 10.0, "tiny", None),
        ("scale_one", 60, 5, 2e-3, 8.0, "uniform", 1.0),
        ("q_20_power_0p6", 60, 6, 2e-3, 8.0, "uniform", None),
    ]
    for name, n, seed, es, lim, rho_kind, c_fixed in spec:
        rng = np.random.default_rng(seed)
        eps = rng.normal(0.0, es, (n, 6))
        eps[:, 3:] *= 0.5
        rho = rng.uniform(0.05, 1.0, n) if rho_kind == "uniform" else np.concatenate(
            [rng.uniform(0.0, 1e-10, n // 2), rng.uniform(0.3, 1.0, n - n // 2)])
        q = 20.0 if name.startswith("q_20") else 40.0
        power = 0.6 if name.endswith("0p6") else 0.5
        ns = {"np": np, "rho_full": rho, "rho": rho, "eps": eps,
              "self": SimpleNamespace(cfg=SimpleNamespace(stress_power=power, ks_q=q), D0=D0,
                                      active=np.ones(n, bool), n_elem=n, ks_scale={}),
              "cfg": SimpleNamespace(stress_power=power, ks_q=q), "lim": lim, "n": n}
        exec(code["relaxed"].replace("return vm, dvm_ds, fac", "pass"), ns)   # vm, dvm_ds, fac
        exec(code["ks"], ns)                                                  # z, a, ez, S, w
        exec(code["explicit"], ns)
        ns["fs"] = ns["fac"]                                                  # the call site names it fs
        exec(code["g6"], ns)                                                  # the adjoint right-hand side
        ns.update(t={"zmax": ns["a"], "ks": ns["S"]}, name="case")
        exec(code["scale"], ns)
        c = c_fixed if c_fixed is not None else ns["self"].ks_scale["case"]
        ns["c"] = c
        ns["out"] = {"constraints": {}}
        exec(code["row"], ns)                                                 # the row value c*S - 1
        row = ns["out"]["constraints"]["stress_case"]["value"]
        cases.append({
            "name": name, "n": n, "E": E0, "nu": nu, "limit": lim, "q": q, "power": power,
            "eps": eps.tolist(), "rho": rho.tolist(),
            "vm": ns["vm"].tolist(), "dvm_ds": ns["dvm_ds"].tolist(), "fac": ns["fac"].tolist(),
            "S": float(ns["S"]), "zmax": float(ns["a"]), "w": ns["w"].tolist(),
            "c": float(c), "c_from_update_rule": c_fixed is None,
            "explicit": ns["explicit"].tolist(), "g6": ns["g6"].tolist(), "row": float(row)})
    doc = {"source": f"earlier pipeline@{REV} {SRC}", "lines": {k: list(v[:2]) for k, v in BLOCKS.items()},
           "note": "made by tools/make_ks_fixture.py from the pipeline lines; aggregation layer only (P1 tets in the "
                   "pipeline, hex8 voxels in the demo)", "E": E0, "nu": nu, "D0": D0.tolist(), "cases": cases}
    out.write_text(json.dumps(doc))
    print("wrote", out, f"{out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
