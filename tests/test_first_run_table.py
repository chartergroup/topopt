"""LAUNCH-2 (#227): the table "First-run problems" (in docs/reference/install.md since #294) (message, cause, fix) and the skill that points to it.

A first-time user (or Claude) who meets a common error finds its message, its cause and its fix in one table. A message in the
first column is a part of the real output of the tool. So the test looks for a key phrase of each message in the code
(`topopt/`, `topopt_mcp/` and `scripts/`). When a message changes in the code, the row breaks, and the table gets the new text.

The key phrase of a message:
- A message is a backtick span of the first column with 3 words or more and 15 characters or more. (A short span such as
  `/mcp` or `failed` is a name, not a message.)
- The parts that change from run to run (numbers, a name in quotes) are cut out. The longest part that is left is the key phrase.
- The code builds a message from pieces (`f"... {n} separate pieces ..."`), so a variable can sit at the start or the end of the
  key phrase. The key phrase is also tried without its first word and without its last word.
- Python files are read as strings of the syntax tree (an f-string or a string in two lines gives one piece). Shell files are read as text.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTION = "First-run problems"
PAGE = "docs/reference/install.md"       # the table was a README section before #294 made the README short
SKILL_COPIES = ("skills/start-here/SKILL.md", ".agents/skills/start-here/SKILL.md")
MIN_ROWS = 6
MIN_PHRASE = 15

VARIABLE = re.compile(r"\d+(?:[.,]\d+)*|'[^']*'|<[^>]*>")


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def section_lines() -> list[str]:
    """The lines of the section `### First-run problems` of the install page, up to the next heading."""
    lines = read(PAGE).splitlines()
    heading = f"### {SECTION}"
    assert heading in lines, f"{PAGE} has no section '{heading}'"
    start = lines.index(heading) + 1
    out = []
    for line in lines[start:]:
        if line.startswith("#"):
            break
        out.append(line)
    return out


def table_rows() -> list[list[str]]:
    """The body rows of the table of the section: a list of the cells (stripped). The header and the separator are not in it."""
    table = [ln for ln in section_lines() if ln.startswith("|")]
    assert len(table) >= 2, "the section has no table"
    rows = []
    for line in table[2:]:
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def messages(cell: str) -> list[str]:
    """The messages in a cell: the backtick spans that have 3 words or more and 15 characters or more."""
    spans = re.findall(r"`([^`]+)`", cell)
    return [s for s in spans if len(s) >= MIN_PHRASE and len(s.split()) >= 3]


def key_phrase(message: str) -> str:
    """The longest part of a message that has no number, no quoted name and no <placeholder>."""
    return max((p.strip() for p in VARIABLE.split(message)), key=len)


def candidates(phrase: str) -> list[str]:
    """The phrase, and the phrase without its first word, without its last word and without both (a variable can sit there)."""
    words = phrase.split()
    found = [phrase]
    if len(words) > 2:
        found += [" ".join(words[1:]), " ".join(words[:-1]), " ".join(words[1:-1])]
    return [c for c in found if len(c) >= MIN_PHRASE]


def strings_of(source: str) -> list[str]:
    """The strings of a Python source: an f-string gives its fixed parts, and strings in two lines give one string."""
    return [n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def code_text() -> str:
    """All the strings of the code, with a NUL byte between two strings (a phrase cannot sit across two strings)."""
    pieces = []
    for folder in ("topopt", "topopt_mcp"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            pieces += strings_of(path.read_text(encoding="utf-8"))
    for path in sorted((ROOT / "scripts").glob("*.sh")):
        pieces.append(path.read_text(encoding="utf-8"))
    return "\0".join(pieces)


def in_code(message: str, text: str) -> bool:
    return any(c in text for c in candidates(key_phrase(message)))


# --- the table ---------------------------------------------------------------------------------------------------

def test_the_install_page_has_the_section_after_check_your_machine_and_a_table_of_enough_rows():
    page = read(PAGE)
    assert f"\n### {SECTION}\n" in page, f"{PAGE} has no section '### {SECTION}'"
    assert page.index("### Check your machine") < page.index(f"### {SECTION}")
    rows = table_rows()
    assert len(rows) >= MIN_ROWS, f"the table has {len(rows)} rows; it needs {MIN_ROWS} or more"
    header = [ln for ln in section_lines() if ln.startswith("|")][0]
    assert [c.strip() for c in header.strip().strip("|").split("|")] == ["Message or symptom", "Cause", "Fix"]


def test_each_row_has_a_message_a_cause_and_a_fix():
    bad = []
    for n, row in enumerate(table_rows(), 1):
        if len(row) != 3 or not all(row):
            bad.append(f"row {n}: needs 3 cells, none empty: {row}")
            continue
        if not messages(row[0]) and "no message" not in row[0]:
            bad.append(f"row {n}: the first cell has no message in backticks and does not say 'no message': {row[0][:80]}")
    assert not bad, "\n".join(bad)


def test_each_message_of_the_table_is_found_in_the_code():
    text = code_text()
    missing = []
    for n, row in enumerate(table_rows(), 1):
        for message in messages(row[0]):
            if not in_code(message, text):
                missing.append(f"row {n}: the key phrase {key_phrase(message)!r} of the message {message!r} is not in the code")
    assert not missing, "the code prints another text now; change the row to the new message:\n" + "\n".join(missing)


def test_the_message_matcher_finds_a_message_of_the_code_and_refuses_an_unknown_one():
    """A matcher that accepts everything would make the test above empty. The sample code has the shapes of the real messages."""
    sample = (
        'raise UserError(f"STL is not watertight (not closed): {path}")\n'
        'raise UserError(f"model has {active} active voxels, the limit is {limit} "\n'
        '                f"(max_active_voxels). Try voxel_size {size} or more")\n'
        'text = f"load path: {what} not joined to a fixed, spring or joint region in the thresholded design (density {t:g}){at}"\n'
        'text = f"grid would have {n} voxels at voxel_size {h}, the limit is {m} (max_grid_voxels)."\n'
    )
    text = "\0".join(strings_of(sample))
    assert in_code("STL is not watertight (not closed)", text)
    assert in_code("model has 564298 active voxels, the limit is 100000 (max_active_voxels).", text)    # numbers are cut out
    assert in_code("load region 'tip' is not joined to a fixed, spring or joint region in the thresholded design (density 0.5) at beta 8.", text)
    assert not in_code("this message is in no file of the code at all", text)
    assert not in_code("STL is closed and all is well", text)
    assert key_phrase("grid would have 23388750 voxels at voxel_size 0.2, the limit is 5000000 (max_grid_voxels).") == "voxels at voxel_size"
    assert messages("`topopt` shows `failed` in `/mcp`. `topopt printcheck` on the run folder") == []
    assert len(code_text()) > 100_000, "the code folders are empty or not found: the test above would find nothing to refuse"


# --- the links ---------------------------------------------------------------------------------------------------

def test_the_tools_do_not_show_bullet_links_the_section():
    page = read(PAGE)      # the bullet moved from the README to the install page with #385
    start = page.index("- **If the tools do not show.**")
    bullet = page[start:page.index("\n\n", start)]
    assert "(#first-run-problems)" in bullet, "the bullet 'If the tools do not show' needs a link to the section"


def test_the_start_here_skill_and_its_copy_send_the_agent_to_the_table():
    texts = {rel: read(rel) for rel in SKILL_COPIES}
    assert len(set(texts.values())) == 1, "the copy in .agents/skills/ must equal skills/start-here/ (python3 scripts/sync_agent_skills.py)"
    for rel, text in texts.items():
        assert SECTION in text, f"{rel} does not name the table '{SECTION}'"
        at = text.index(SECTION)
        assert PAGE in text[max(0, at - 400):at + 400], f"{rel}: the table is not tied to {PAGE}"
        assert "${CLAUDE_PLUGIN_ROOT}/" + PAGE in text, f"{rel} does not give the path of the install page in the plugin"
