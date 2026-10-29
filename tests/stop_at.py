"""A `topopt run` that writes its own request to stop at a chosen point (a helper of tests/test_run_stop.py).

    python stop_at.py <point> <topopt run arguments...>

The request is the file `job.stop` that `runjob.request_stop` writes. The helper writes it from inside the run, at a fixed point, so a test has
no sleep and no timing. `point`:

    pass:<k>      the checkpoint of pass <k> is written; the request comes right after it (the run reads it at the end of that pass)
    hold:<k>      the checkpoint of pass <k> is written; the run then waits (a blocking read of the FIFO `hold.fifo` in the run folder, after it
                  has made the file `hold.reached`). The test writes the request itself (`topopt stop`, MCP `stop_run`), then writes to the FIFO.
                  So the request is in the folder before the run reads it, with no sleep and no timing.
    hold:<k>:<t>  the same with the tag <t>: the files are `hold-<t>.fifo` and `hold-<t>.reached`, so two runs in one folder (`--override-lock`)
                  can wait at different passes
    other:<k>     the checkpoint of pass <k> is written; the request comes right after it, but it names another pid (a request for another process)
    post          the post step starts; the request comes before it (the post step never reads it)
    lock          the process is about to take the job lock; the request comes before it and names this pid (a request between spawn and lock)
    lock_other    the same, but the request names another pid (an old request: the lock step removes it)
    never         no request (the run goes on as a plain `topopt run`)

The exit code is the exit code of `topopt.cli.main`.
"""
import os
import sys
from pathlib import Path


def main():
    point, _, number = sys.argv[1].partition(":")
    number, _, tag = number.partition(":")
    hold_name = f"hold-{tag}" if tag else "hold"
    from topopt import checkpoint, post, runjob

    if point == "pass":
        real_write = checkpoint.write_checkpoint

        def write(run_dir, arrays, meta):
            size = real_write(run_dir, arrays, meta)
            if int(meta["iteration"]) == int(number):
                runjob.request_stop(run_dir, "tests/stop_at.py", os.getpid())
            return size

        checkpoint.write_checkpoint = write
    elif point == "hold":
        real_write = checkpoint.write_checkpoint

        def write(run_dir, arrays, meta):
            size = real_write(run_dir, arrays, meta)
            if int(meta["iteration"]) == int(number):
                fifo = Path(run_dir) / f"{hold_name}.fifo"
                os.mkfifo(fifo)
                (Path(run_dir) / f"{hold_name}.reached").write_text(str(os.getpid()))
                with open(fifo, "rb") as f:     # blocks until the test writes to the FIFO
                    f.read()
            return size

        checkpoint.write_checkpoint = write
    elif point == "other":
        real_write = checkpoint.write_checkpoint

        def write(run_dir, arrays, meta):
            size = real_write(run_dir, arrays, meta)
            if int(meta["iteration"]) == int(number):
                runjob.request_stop(run_dir, "tests/stop_at.py", os.getpid() + 1000003)
            return size

        checkpoint.write_checkpoint = write
    elif point == "post":
        real_finish = post.finish

        def finish(run_dir, *args, **kwargs):
            runjob.request_stop(run_dir, "tests/stop_at.py", os.getpid())
            return real_finish(run_dir, *args, **kwargs)

        post.finish = finish
    elif point in ("lock", "lock_other"):
        real_start = runjob.RunJob.start

        def start(self):
            self.dir.mkdir(parents=True, exist_ok=True)
            runjob.request_stop(self.dir, "tests/stop_at.py", os.getpid() if point == "lock" else os.getpid() + 1000003)
            return real_start(self)

        runjob.RunJob.start = start
    elif point != "never":
        raise SystemExit(f"stop_at.py: unknown point {point!r}")
    from topopt.cli import main as cli_main
    raise SystemExit(cli_main(sys.argv[2:]))


if __name__ == "__main__":
    main()
