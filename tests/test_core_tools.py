"""#174 step 2: the core set of MCP tools is the same in INSTRUCTIONS, CONTRACTS.md and topopt_mcp/README.md, and every tool is accounted for."""
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topopt_mcp.server import INSTRUCTIONS, TOOL_SURFACE, mcp  # noqa: E402

CONTRACTS = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
README = (ROOT / "topopt_mcp" / "README.md").read_text(encoding="utf-8")


def live_tools():
    return {tool.name for tool in asyncio.run(mcp.list_tools())}


def named_in(text):
    """The live tools that a text names, in order of first mention."""
    found = [(m.start(), m.group(0)) for name in live_tools() for m in [re.search(r"(?<![\w.])" + name + r"(?!\w)", text)] if m]
    return [name for _, name in sorted(found)]


def core_of_instructions():
    line = next(l for l in INSTRUCTIONS.splitlines() if l.startswith("Core tools:"))
    return named_in(line.split(". ")[0])


def core_of_doc(text):
    line = next(l for l in text.splitlines() if l.startswith("Core tools: "))
    return re.findall(r"`(\w+)`", line)


def last_line_tools():
    last = INSTRUCTIONS.splitlines()[-1]
    assert last.startswith("Descriptions say when to use:")
    return named_in(last)


def test_the_core_set_is_the_same_in_the_instructions_the_contracts_and_the_readme():
    core = core_of_instructions()
    assert len(core) == len(set(core)) and core
    assert core_of_doc(CONTRACTS) == core
    assert core_of_doc(README) == core


def test_every_core_tool_and_every_tool_of_the_last_line_is_a_live_tool():
    live = live_tools()
    assert live == set(TOOL_SURFACE)
    last_words = re.findall(r"\w+", INSTRUCTIONS.splitlines()[-1].split(":", 1)[1])
    core_line = next(l for l in INSTRUCTIONS.splitlines() if l.startswith("Core tools:")).split(". ")[0]
    tool_like = re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", core_line)                 # every snake_case name of the core line is a live tool
    assert tool_like and set(tool_like) <= live, [w for w in tool_like if w not in live]
    assert all(word in live for word in last_words), [w for w in last_words if w not in live]


def test_no_core_tool_is_in_the_last_line():
    assert not set(core_of_instructions()) & set(last_line_tools())


def test_every_live_tool_is_named_in_the_instructions_or_listed_as_not_named_in_the_contracts():
    line = next(l for l in CONTRACTS.splitlines() if l.startswith("Not named in INSTRUCTIONS: "))
    listed = set(re.findall(r"`(\w+)`", line))
    assert listed <= live_tools()
    assert live_tools() - set(named_in(INSTRUCTIONS)) == listed


def test_the_core_line_is_the_first_line_of_the_instructions_that_names_a_tool():
    lines = INSTRUCTIONS.splitlines()
    first = next(i for i, line in enumerate(lines) if named_in(line))
    assert lines[first].startswith("Core tools:")


def test_the_counts_in_the_texts_follow_the_live_tools():
    live = live_tools()
    not_named = live - set(named_in(INSTRUCTIONS))
    assert f"The server has {len(live)} tools" in CONTRACTS and f"cannot name all {len(live)} tools" in CONTRACTS
    assert f"The core set has {len(core_of_instructions())} tools" in CONTRACTS
    assert f"except these {len(not_named)}, which it does not" in CONTRACTS
    assert f"The section Tools names all {len(live)} tools." in README
    section = README.split("\n## Tools\n", 1)[1].split("\n## ", 1)[0]
    assert {name for name in live if f"`{name}(" not in section} == set()     # every tool has its entry (verify_status shares one)
