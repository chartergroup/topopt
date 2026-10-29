"""Print the tools/list response of the topopt MCP server as JSON.

Usage (in the repo root):
    sh scripts/topopt-env.sh python evals/tools/dump_tools.py \
        "$PWD/scripts/topopt-env.sh" > evals/mocks/topopt/_tools.json
"""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(launcher):
    params = StdioServerParameters(command="sh", args=[launcher, "mcp"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.list_tools()
            print(json.dumps(res.model_dump(mode="json", by_alias=True, exclude_none=True),
                             indent=1, ensure_ascii=False))


asyncio.run(main(sys.argv[1]))
