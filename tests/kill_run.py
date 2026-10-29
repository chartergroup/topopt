"""A `topopt run` that kills itself with SIGKILL at a chosen point (a helper of tests/test_checkpoint.py).

    python kill_run.py <point>:<pass> <topopt run arguments...>

`point` is where the process dies (it is a real SIGKILL, no handler runs, no file is closed):

    after_checkpoint      the checkpoint of pass <pass> is written, the next pass has not started
    before_checkpoint     the snapshot and the history line of pass <pass> are written, the checkpoint is not
    partial_tmp           half of the checkpoint file of pass <pass> is in `checkpoint.npz.tmp`, nothing was renamed
    between_renames       the old checkpoint moved to `checkpoint.prev.npz`, the new file is not renamed yet
    partial_history       half of the history line of pass <pass> is on disk (no newline)
    corrupt_current       the checkpoint of pass <pass> is written, then cut to half its size, then the process dies
"""
import io
import os
import signal
import sys

import numpy as np


def die():
    os.kill(os.getpid(), signal.SIGKILL)


def main():
    point, _, number = sys.argv[1].partition(":")
    at = int(number)
    from topopt import checkpoint
    from topopt.snapshot import RunWriter

    real_write = checkpoint.write_checkpoint
    real_add = RunWriter.add_history

    def write(run_dir, arrays, meta):
        if int(meta["iteration"]) != at:
            return real_write(run_dir, arrays, meta)
        d = checkpoint.Path(run_dir)
        if point == "before_checkpoint":
            die()
        if point in ("partial_tmp", "between_renames"):
            buf = io.BytesIO()
            np.savez(buf, **{checkpoint.META: np.frombuffer(checkpoint.json.dumps(meta).encode(), np.uint8)}, **arrays)
            data = buf.getvalue()
            if point == "partial_tmp":
                (d / checkpoint.TMP_FILE).write_bytes(data[: len(data) // 2])
            else:
                (d / checkpoint.TMP_FILE).write_bytes(data)
                if (d / checkpoint.CHECKPOINT_FILE).exists():
                    os.replace(d / checkpoint.CHECKPOINT_FILE, d / checkpoint.PREV_FILE)
            die()
        size = real_write(run_dir, arrays, meta)
        if point == "corrupt_current":
            path = d / checkpoint.CHECKPOINT_FILE
            path.write_bytes(path.read_bytes()[: size // 2])
        die()
        return size

    def add(self, rec):
        if point == "partial_history" and int(rec["iteration"]) == at:
            import json
            data = (json.dumps(rec) + "\n").encode()
            with open(self.dir / "history.jsonl", "ab") as f:
                f.write(data[: len(data) // 2])
                f.flush()
            die()
        return real_add(self, rec)

    checkpoint.write_checkpoint = write
    RunWriter.add_history = add
    from topopt.cli import main as cli_main
    raise SystemExit(cli_main(sys.argv[2:]))


if __name__ == "__main__":
    main()
