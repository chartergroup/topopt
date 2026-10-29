"""Role playbooks of the topopt team (CONTRACTS.md section 25).

The source is one set of files: `topopt_mcp/roles/<id>.md`, shipped as package data. Four places carry the same
text: the tool `team_guide`, the MCP prompts, the MCP resources, and the body of `agents/<id>-engineer.md`.
Tests compare them with the role files.

This module needs only the standard library. The build script of the `.mcpb` bundle loads it too.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

# Order of the list that `team_guide` returns: the order of the work (`finish` is last: it follows `print`, after a run).
# Every id has a file `roles/<id>.md` and an agent `<id>-engineer`.
ROLE_IDS = ("lead-design", "cad", "fea", "print", "finish")
URI_PREFIX = "topopt://roles/"
AGENT_SUFFIX = "-engineer"


class RoleFilesError(Exception):
    """A role file is missing or has no usable header."""


@dataclass(frozen=True)
class Role:
    id: str
    title: str
    summary: str  # one line, for the list and for the descriptions of the prompt and the resource
    text: str     # the playbook: the body of the role file

    @property
    def uri(self) -> str:
        return URI_PREFIX + self.id

    @property
    def agent(self) -> str:
        """Name of the Claude Code plugin sub-agent that carries the same text."""
        return self.id + AGENT_SUFFIX


def parse_role(role_id: str, raw: str) -> Role:
    """Split a role file into its header (`title`, `summary`) and its body."""
    if not raw.startswith("---\n"):
        raise RoleFilesError(f"role {role_id}: the file has no header (it must start with ---)")
    head, sep, body = raw[4:].partition("\n---\n")
    if not sep:
        raise RoleFilesError(f"role {role_id}: the header has no closing ---")
    fields = {}
    for line in head.splitlines():
        key, colon, value = line.partition(":")
        if colon:
            fields[key.strip()] = value.strip()
    for key in ("title", "summary"):
        if not fields.get(key):
            raise RoleFilesError(f"role {role_id}: the header has no `{key}`")
    text = body.strip()
    if not text:
        raise RoleFilesError(f"role {role_id}: the body is empty")
    return Role(role_id, fields["title"], fields["summary"], text)


def read_roles(folder: Path | None = None) -> dict[str, Role]:
    """Read the role files. `folder` None: the package data of `topopt_mcp` (also from a wheel)."""
    base = resources.files("topopt_mcp") / "roles" if folder is None else folder
    roles = {}
    for role_id in ROLE_IDS:
        try:
            raw = (base / f"{role_id}.md").read_text(encoding="utf-8")
        except OSError as exc:
            raise RoleFilesError(f"role {role_id}: cannot read {base}/{role_id}.md ({exc})") from exc
        roles[role_id] = parse_role(role_id, raw)
    return roles


@functools.lru_cache(maxsize=1)
def load_roles() -> dict[str, Role]:
    """The roles of the installed package, in the order of ROLE_IDS (read once)."""
    return read_roles()


def resolve_role(name: str | None) -> Role | None:
    """Find a role by id. Also accepted: any case, `_` for `-`, the plugin agent name (`cad-engineer`,
    `topopt:cad-engineer`), the resource URI (`topopt://roles/cad`) and a prefix that names one role (`lead`).
    None if no role matches."""
    if not name or not name.strip():
        return None
    text = name.strip()
    if text.lower().startswith(URI_PREFIX):  # an agent that read the resource list can pass the URI
        text = text[len(URI_PREFIX):]
    key = text.rpartition(":")[2].lower().replace("_", "-").replace(" ", "-")
    roles = load_roles()
    if key in roles:
        return roles[key]
    if key.endswith(AGENT_SUFFIX) and key[: -len(AGENT_SUFFIX)] in roles:
        return roles[key[: -len(AGENT_SUFFIX)]]
    hits = [r for r in roles.values() if key and r.id.startswith(key)]
    return hits[0] if len(hits) == 1 else None


def roles_list(unknown: str | None = None) -> str:
    """The list of roles with one line each. `unknown` is a name that matched no role: the first line says so."""
    lines = []
    if unknown and unknown.strip():
        lines.append(f"No role is named {unknown.strip()!r}.")
    lines.append("Roles of the topopt team. team_guide(role) gives the playbook of one role:")
    for r in load_roles().values():
        lines.append(f"- {r.id}: {r.summary}")
    lines.append(
        "One agent can do all roles in one conversation. A client with sub-agents can give each role to one "
        "sub-agent (the Claude Code plugin has them as " + ", ".join(r.agent for r in load_roles().values()) + ").")
    lines.append("The same playbooks are MCP prompts (name = role id) and MCP resources (" + URI_PREFIX + "<id>).")
    return "\n".join(lines)


def guide(role: str | None = None) -> str:
    """The text that `team_guide` returns: the playbook of the role, else the list of roles.

    If the role files cannot be read (a broken install), the answer is the cause as text. The other tools work.
    """
    try:
        found = resolve_role(role)
        return found.text if found else roles_list(role)
    except RoleFilesError as exc:
        return f"The role playbooks are not available: {exc}. The other tools work."
