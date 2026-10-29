#!/usr/bin/env python
"""Make the golden fixture of `topopt verify`: numbers of the pipeline P2 code on a small fixed mesh.

    python tools/make_verify_golden.py <pipeline-folder> [--rev d43bcf3] [--out tests/fixtures/verify_golden.npz.b64]

`<pipeline-folder>` is the `pipeline/` folder of the earlier pipeline project (it holds `lib/p2_core.py`). The script
imports `p2_core` from it (scikit-fem must be installed) and writes, for a small mesh and a fixed
displacement field: the stiffness matrix K, the stress at the quadrature points, the stress at the
vertices, the von Mises values, and weighted percentiles. `tests/test_verify_p2.py` compares
`topopt.verify` against these numbers at rtol 1e-9. The pipeline folder is only read.

The mesh: 2 x 2 x 2 cubes, each split into 6 tets (the same split everywhere), with a fixed jitter of the
nodes so that no tet is special. The field: a fixed random quadratic displacement. The revision and the
sha256 of `p2_core.py` go into the file, so the source of the numbers is on record.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import itertools
import sys
from pathlib import Path

import numpy as np

CUBE_TETS = ((0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7))   # around the diagonal 0-7


def small_mesh(n: int = 2, jitter: float = 0.12, seed: int = 7):
    """Return `p` (3, nv) and `t` (4, nel) of n x n x n cubes of side 1 mm, 6 tets each, with a jitter of the nodes."""
    idx = {c: i for i, c in enumerate(itertools.product(range(n + 1), repeat=3))}
    p = np.array(list(idx), float).T
    rng = np.random.default_rng(seed)
    p = p + jitter * rng.uniform(-1, 1, p.shape)
    t = []
    for cx, cy, cz in itertools.product(range(n), repeat=3):
        corner = [(cx + (v & 1), cy + ((v >> 1) & 1), cz + ((v >> 2) & 1)) for v in range(8)]
        for tet in CUBE_TETS:
            t.append([idx[corner[v]] for v in tet])
    t = np.array(t, np.int64).T
    return p, t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pipeline", help="the pipeline/ folder of the earlier pipeline project (with lib/p2_core.py)")
    ap.add_argument("--rev", default="d43bcf3", help="revision of the pipeline to record (default: d43bcf3)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "verify_golden.npz.b64"),
                    help="output: the .npz bytes as base64 text (the PR tool of this repo publishes text files only)")
    args = ap.parse_args(argv)
    lib = Path(args.pipeline) / "lib"
    core_py = lib / "p2_core.py"
    if not core_py.is_file():
        print(f"error: {core_py} not found", file=sys.stderr)
        return 1
    sys.path.insert(0, str(lib))
    import p2_core as pc                                  # the pipeline code (reads only)
    E, nu = 3500.0, 0.35
    p, t = small_mesh()
    model = pc.P2Model(p, t)
    K = model.stiffness(E, nu).tocsr()
    K.sort_indices()
    rng = np.random.default_rng(11)
    # a smooth displacement field: a random quadratic polynomial in x, y, z, evaluated at all P2 nodes
    X = model.node_xyz
    mono = np.stack([np.ones(X.shape[1]), X[0], X[1], X[2], X[0] * X[1], X[1] * X[2], X[0] * X[2],
                     X[0] ** 2, X[1] ** 2, X[2] ** 2])
    u = (rng.normal(size=(3, 10)) @ mono).T * 1e-3         # (nn, 3)
    u = u.reshape(-1)
    sig, lam, w = model.stress_qp(u, E, nu)
    sigv = model.stress_at_vertices(u, E, nu)
    vm = pc.von_mises(sig)
    wv = w[None, :] * model.vol[:, None]
    vals = rng.uniform(0, 300, 5000)
    wts = rng.uniform(0.1, 2.0, 5000)
    pct = np.array([0.0, 1.0, 50.0, 95.0, 99.0, 99.9, 100.0])
    buf = io.BytesIO()
    np.savez_compressed(
        buf, p=p, t=t, E=E, nu=nu, N=model.N, u=u,
        K_data=K.data, K_indices=K.indices, K_indptr=K.indptr,
        node_xyz=model.node_xyz, enodes=model.enodes, vol=model.vol, edges=model.edges,
        stress_qp=sig, lam=lam, w=w, stress_at_vertices=sigv, vm_qp=vm,
        wp_values=vals, wp_weights=wts, wp_pct=pct, wp_result=pc.weighted_percentile(vals, wts, pct),
        wp_result_vm=pc.weighted_percentile(vm, wv, pct),
        source_rev=args.rev, source_file="pipeline/lib/p2_core.py",
        source_sha256=hashlib.sha256(core_py.read_bytes()).hexdigest())
    Path(args.out).write_text(base64.encodebytes(buf.getvalue()).decode())
    print(f"wrote {args.out}: {p.shape[1]} vertices, {t.shape[1]} tets, {model.N} dofs, K nnz {K.nnz}; "
          f"pipeline rev {args.rev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
