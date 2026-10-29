"""Shared parts of the fixture recorders: start the real server, call a tool, save the answer."""
import contextlib
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# The fixtures hold these paths in place of the paths of the machine that recorded them.
WORK_PLACEHOLDER = "/Users/me/topopt-work"
CODE_PLACEHOLDER = "/Users/me/topopt-demo"
HOME_PLACEHOLDER = "/Users/me"


def text_of(res):
    return "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")


class Recorder:
    """One server session. `call` returns the parsed answer, `save` writes it as a fixture."""

    def __init__(self, session, launcher, work):
        self.s = session
        self.subs = []
        for path, placeholder in ((work, WORK_PLACEHOLDER),
                                  (Path(launcher).resolve().parents[1], CODE_PLACEHOLDER),
                                  (Path.home(), HOME_PLACEHOLDER)):
            for p in {str(path), os.path.realpath(path)}:
                self.subs.append((p, placeholder))

    async def call(self, tool, **args):
        res = await self.s.call_tool(tool, args)
        if res.is_error:
            raise SystemExit(f"{tool} failed: {text_of(res)[:500]}")
        text = text_of(res)
        try:
            return json.loads(text)
        except ValueError:
            return text

    def save(self, out_dir, name, answer):
        text = answer if isinstance(answer, str) else json.dumps(answer, indent=1, ensure_ascii=False)
        for p, placeholder in self.subs:
            text = text.replace(p, placeholder)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.json").write_text(text + "\n")
        print(f"saved {out_dir / name}.json ({len(text)} chars)", file=sys.stderr)


@contextlib.asynccontextmanager
async def recorder(launcher, work):
    """Start the server with TOPOPT_WORKSPACE=work (runs/ go there, not in the repo)."""
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TOPOPT_WORKSPACE=str(work))
    params = StdioServerParameters(command="sh", args=[launcher, "mcp"], env=env, cwd=str(work))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            yield Recorder(s, launcher, work)
