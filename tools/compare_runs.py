"""Compare two run folders bit for bit: snapshots, labels, metrics, history, mesh (A2 default-path check).

Usage:  python tools/compare_runs.py <run_dir_a> <run_dir_b>

Exit 0 when all of these are equal: every snapshot file, labels.bin, result/final.stl, result/preview.bin,
result/metrics.json (without `run_time_s`) and history.jsonl (without `elapsed_s` and `t_iter_s`). Prints one line for each.

How the check of A2 was made: `git archive 47369f8 | tar -x -C base`, then the hook example in both trees with
`MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1` and PYTHONPATH set to the tree
(PARDISO at one thread, so the sums run in a fixed order).
"""
import hashlib
import json
import re
import sys
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snapshot_files(run):
    """The files of `<run>/snapshots` in the order of the pass number (P4b-5). The name is `it_` and the number with at least four digits:
    a text sort puts `it_10000.bin` before `it_9999.bin`."""
    def number(p):
        m = re.fullmatch(r"it_(\d+)\.bin", p.name)
        return (int(m.group(1)) if m else -1, p.name)
    return sorted((Path(run) / "snapshots").iterdir(), key=number)


def main(a, b):
    a, b = Path(a), Path(b)
    ok = True

    def report(name, same):
        nonlocal ok
        ok &= bool(same)
        print(f"{'equal    ' if same else 'DIFFERENT'} {name}")

    sa, sb = snapshot_files(a), snapshot_files(b)
    report(f"snapshots ({len(sa)} and {len(sb)} files)",
           [p.name for p in sa] == [p.name for p in sb] and all(sha(x) == sha(y) for x, y in zip(sa, sb)))
    for name in ("labels.bin", "result/final.stl", "result/preview.bin"):
        report(name, sha(a / name) == sha(b / name))
    ma, mb = (json.loads((d / "result" / "metrics.json").read_text()) for d in (a, b))
    for m in (ma, mb):
        m.pop("run_time_s", None)
    report("result/metrics.json (without run_time_s)", ma == mb)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("elapsed_s", "t_iter_s")}
    ha, hb = ([strip(json.loads(x)) for x in (d / "history.jsonl").read_text().splitlines()] for d in (a, b))
    report("history.jsonl (without elapsed_s, t_iter_s)", ha == hb)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
