"""Case file, brief and soft gate for MCP runs (CONTRACTS.md section 30).

A case is one problem YAML with the region STL files that it names. The case folder is the folder of the YAML, inside the
work folder. A case has one brief and many runs. Two files sit next to the YAML:

- `<yaml stem>.case.json`: the brief, the input hashes of the brief and of each run, and one entry for each run.
- `<yaml stem>.case-log.jsonl`: one JSON line for each event (brief recorded, run started, gate returned, override used).

Rules:
- Every path in these files is relative. No absolute path and no user name is written.
- File hashes come from `topopt.runinfo.sha256_file` (the hash is of the content). The file keys come from
  `topopt.runinfo.region_files` with the work folder as `root` (the rule that the checkpoint of D1b uses too):
  `<role>:<path relative to the YAML folder>`, where the path is the one the YAML names (a symbolic link keeps its own name).
  A file outside the work folder has the key `<role>:<outside>/<12 hex of the SHA-256 of its resolved path>/<file name>`:
  no folder name of the outside place is written. A missing file has the hash `missing`. It is a fact, not an error.
- `yaml_content` is a digest of the parsed YAML (canonical JSON text, an integer and an equal float are one value). It is
  not a file hash. It tells a change of the content from a change of a comment or of the format.
- A brief also holds the split of the parsed YAML by the kind of each key (the cards of `topopt.levers_cards`, P9-4b):
  `yaml_requirements` (a digest of the values of the keys that are not `method`), `requirement_keys` (the 12 hex start of the
  digest of each such value, to name a changed key) and `method_values` (the values of the `method` keys that the YAML sets).
  The gate compares the brief with these, so a change of a method key only (for example `settings.solver`) is a fact and not
  stale. A run entry holds none of them: it is compared with the whole content.
- `case.json` is written with `runinfo.write_atomic`. The log gets one whole line for each event, then a flush.
  There is no lock: two servers on one work folder can lose a run entry of `case.json`. The log keeps every event, and
  `case_status` says when the log has a run start that `case.json` lacks.
- The CLI writes no case. Only the MCP tools write one. Since P9-7a the part card (`topopt.part_card`) reads the case file, read only,
  when `topopt run` ends and when `topopt report` runs; nothing else of the CLI reads it.
- This module imports `topopt.config`, `topopt.runinfo` and `topopt.levers_cards` (data only) and nothing else of the package.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import yaml as pyyaml

from .config import load_config
from .levers_cards import CARDS
from .runinfo import OUTSIDE, outside_key, read_history, region_files, region_names, sha256_file, write_atomic

FORMAT = 1
ORIGINS = ("user", "inferred", "default")   # where an assumption or a criterion comes from
MISSING = "missing"                         # the hash of a file that is not there
NO_CASE_FACT = "no case for this YAML; record_brief makes one"
OVERRIDE = "proceed_without_brief=true"
RESUME_OVERRIDE = "allow_changed_inputs=true"   # the override of the resume stop (CONTRACTS.md section 24.2)
SAME_RUN_S = 120                            # equal pids and start times this close: the same run (else a reused pid)
ITEM = "[]"                                 # the token of a list item in a key path (`load_cases[].name`)
OLD_RECORD_FACT = ("the brief has no record of the method keys; call record_brief again, and a later change of a method "
                   "key (for example settings.solver) will not make the brief stale")


# --- small helpers ---------------------------------------------------------------------------

def _now() -> str:
    """Local time with the offset, to the second (the form of `job.json`)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def manifest_path(yaml: Path) -> Path:
    return yaml.with_name(yaml.stem + ".case.json")


def log_path(yaml: Path) -> Path:
    return yaml.with_name(yaml.stem + ".case-log.jsonl")


def in_work_folder(path: Path, work: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(work).resolve())
    except ValueError:
        return False
    return True


def _wrel(path: Path, work: Path) -> str:
    """Path relative to the work folder. A path outside it gives the file name only (no absolute path)."""
    try:
        return Path(path).resolve().relative_to(Path(work).resolve()).as_posix() or "."
    except ValueError:
        return Path(path).name


def _scrub(text: str, base: Path) -> str:
    """Take the folder of the YAML out of a message, so that the message holds no absolute path."""
    for prefix in {str(base), str(base.resolve())}:
        text = text.replace(prefix + os.sep, "").replace(prefix, ".")
    return text


def _read_json(path: Path):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _hash_or_missing(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError:
        return MISSING


def _norm_numbers(value):
    """The parsed YAML with string keys and with each integer as a float, so `1` and `1.0` give one digest."""
    if isinstance(value, dict):
        return {str(k): _norm_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_norm_numbers(v) for v in value]
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) < 2 ** 53:
        return float(value)
    return value


def _canonical(value) -> str:
    """The canonical JSON text of a parsed YAML value (an integer and an equal float give one text)."""
    return json.dumps(_norm_numbers(value), sort_keys=True, separators=(",", ":"), default=str)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _content_digest(yaml_file: Path) -> str | None:
    """A digest of the parsed YAML (canonical JSON), or None if the YAML does not parse."""
    return _yaml_view(yaml_file)[0]


# --- method keys and requirement keys of the YAML (P9-4b) ----------------------------------------
#
# The lever cards (`topopt.levers_cards`) give each YAML key a kind. A `method` key is a setting of the method: the agent changes it
# without a new brief and reports it. Every other key counts as a requirement, and so does a key with no card; `both` counts as a
# requirement here, because the record of a brief does not say where a value comes from. The card with the longest key path
# that starts the path of a YAML key gives its kind (the card `settings.voxel_size` before the card `settings`).

@dataclass
class YamlSplit:
    """The parsed YAML split by the kind of each key. A part is one value that has no dict inside (a number, a text, a list)."""
    requirements: str            # a digest of the parts that are not method
    keys: dict[str, str]         # "<path>" -> the 12 hex start of the digest of the value, for the parts that are not method
    methods: dict[str, object]   # "<path>" -> the value (JSON form), for the method parts


def _key_tokens(path: str) -> tuple[str, ...]:
    """The tokens of a card key path: `load_cases[].forces` -> ("load_cases", "[]", "forces")."""
    out: list[str] = []
    for part in path.split("."):
        items = 0
        while part.endswith(ITEM):
            part, items = part[: -len(ITEM)], items + 1
        out += [part] + [ITEM] * items
    return tuple(out)


def _is_name(token: str) -> bool:
    """A card token like `<label>` stands for any name of a mapping."""
    return token.startswith("<") and token.endswith(">")


@lru_cache(maxsize=None)
def _kind_table() -> tuple[dict[tuple[str, ...], str], tuple[tuple[tuple[str, ...], str], ...]]:
    """The kind of each key path of the YAML cards: the exact paths, and the paths with a `<name>` token.

    The table is made once from `CARDS` and kept. A test that changes `CARDS` must call `_kind_table.cache_clear()`.
    """
    exact: dict[tuple[str, ...], str] = {}
    named: list[tuple[tuple[str, ...], str]] = []
    for card in CARDS:
        if card.group != "yaml":
            continue
        for key in (*card.also, card.key):      # the key of the card wins over an `also` key
            tokens = _key_tokens(key)
            if any(_is_name(t) for t in tokens):
                named.append((tokens, card.kind))
            else:
                exact[tokens] = card.kind
    return exact, tuple(named)


def key_kind(tokens: tuple[str, ...]) -> str:
    """The kind (`requirement`, `method` or `both`) of a YAML key path, from the most specific card. No card: `requirement`.

    `tokens` are the names of the mappings and `ITEM` for each list: `("load_cases", "[]", "name")`.
    """
    exact, named = _kind_table()
    for n in range(len(tokens), 0, -1):
        head = tokens[:n]
        if head in exact:
            return exact[head]
        for pattern, kind in named:
            if len(pattern) == n and all(p == t or (_is_name(p) and t != ITEM) for p, t in zip(pattern, head)):
                return kind
    return "requirement"


def _has_dict(node) -> bool:
    return isinstance(node, dict) or (isinstance(node, list) and any(_has_dict(x) for x in node))


def _parts(node, path: str = "", tokens: tuple[str, ...] = ()):
    """The parts of a parsed YAML as (path, tokens, value). A dict and a list that holds a dict are opened. A list of
    numbers or texts (a force, the beta schedule) is one part. The path has the list numbers: `load_cases[0].name`."""
    if isinstance(node, dict) and node:
        for k, v in node.items():
            yield from _parts(v, f"{path}.{k}" if path else str(k), (*tokens, str(k)))
    elif isinstance(node, list) and any(_has_dict(x) for x in node):
        for i, v in enumerate(node):
            yield from _parts(v, f"{path}[{i}]", (*tokens, ITEM))
    else:
        yield path, tokens, node


def _split(data) -> YamlSplit:
    requirements: dict[str, str] = {}
    methods: dict[str, object] = {}
    for path, tokens, value in _parts(data):
        if key_kind(tokens) == "method":
            methods[path] = json.loads(json.dumps(value, default=str))     # the form that case.json holds
        else:
            requirements[path] = _canonical(value)
    return YamlSplit(_digest(json.dumps(requirements, sort_keys=True, separators=(",", ":"))),
                     {p: _digest(c)[:12] for p, c in requirements.items()}, methods)


def _yaml_view(yaml_file: Path) -> tuple[str | None, YamlSplit | None]:
    """(the digest of the parsed YAML, its split by key kind). A part that cannot be made is None."""
    try:
        data = pyyaml.safe_load(yaml_file.read_bytes())
    except (OSError, pyyaml.YAMLError, TypeError, ValueError):
        return None, None
    try:
        content = _digest(_canonical(data))
    except (TypeError, ValueError, RecursionError):
        return None, None
    try:
        return content, _split(data)
    except (TypeError, ValueError, RecursionError):
        return content, None


def _under(path: Path, work: Path) -> bool:
    """True if `path` (not resolved through a link in its last part) lies inside the work folder."""
    try:
        Path(os.path.abspath(path)).relative_to(Path(work).resolve())
    except ValueError:
        return False
    return True


def _shown(rel: str) -> str:
    """The name to show for a key path: an outside file gives its file name only."""
    return rel.rsplit("/", 1)[-1] if rel.startswith(OUTSIDE + "/") else rel


# --- the inputs of a problem and their changes -------------------------------------------------

@dataclass
class Inputs:
    """The input files of a problem now: the YAML and every region file, with their hashes."""
    yaml_sha256: str
    yaml_content: str | None
    files: dict[str, str]                 # "<role>:<path relative to the YAML folder>" -> hash or "missing"
    assumptions: list[str] | None = None  # the `assumptions` of the YAML, or None if the YAML does not load
    facts: list[str] = field(default_factory=list)
    from_yaml: bool = True                # False: the YAML gave no file list, so `files` holds the recorded keys only
    split: YamlSplit | None = None        # the parsed YAML split by the kind of each key; None if the YAML does not parse

    def record(self) -> dict:
        """The hashes of a run entry. A run entry is compared with the whole content of the YAML, so it holds no split."""
        return {"yaml": self.yaml_sha256, "yaml_content": self.yaml_content, "files": dict(self.files)}

    def brief_record(self) -> dict:
        """The hashes of a brief: `record()` and the split of the YAML by key kind (the gate ignores a method key)."""
        rec = self.record()
        if self.split is not None:
            rec.update(yaml_requirements=self.split.requirements, requirement_keys=dict(self.split.keys),
                       method_values=dict(self.split.methods))
        return rec


def current_inputs(yaml: Path, work: Path, recorded_keys=()) -> Inputs:
    """Hash the YAML and every region file that it names.

    The keys come from `runinfo.region_files` with the work folder as `root` (the one key rule of the checkpoint and the
    case): the path as the YAML names it, and `<role>:<outside>/<12 hex>/<file name>` for a file outside the work folder.

    A YAML that does not load (a syntax error, a missing STL, a test stub) still has a hash. The region files then come from
    the text of the YAML. If the text gives none, they are the keys that an earlier record holds (`recorded_keys`), and
    a file that is not there is `missing`.
    """
    base = yaml.parent
    facts: list[str] = []
    assumptions = None
    cfg = None
    try:
        cfg = load_config(yaml)
        assumptions = list(cfg.assumptions)
    except Exception as exc:   # any problem with the YAML: the case still works with the file hashes
        reason = _scrub(str(exc).splitlines()[0] if str(exc) else type(exc).__name__, base)
        facts.append(f"{yaml.name} does not load as a problem ({reason})")
    from_yaml = cfg is not None or region_names(yaml) is not None
    files: dict[str, str] = {}
    if from_yaml:
        for key, path in region_files(cfg.regions if cfg is not None else None, base, root=work, yaml_path=yaml).items():
            role, _, rel = key.partition(":")
            if rel.startswith(OUTSIDE + "/"):
                facts.append(f"{path.name} ({role}) is outside the work folder; keep the STL files beside the YAML")
            elif not _under(Path(os.path.realpath(path)), work):
                facts.append(f"{path.name} ({role}) is a link to a file outside the work folder; the case keeps the "
                             "hash of its content")
            files[key] = _hash_or_missing(path)
    else:
        facts[-1] += "; only the files that an earlier record names are checked"
        for key in recorded_keys:
            rel = key.partition(":")[2]
            if not rel.startswith(OUTSIDE + "/"):
                files[key] = _hash_or_missing(base / rel)
    content, split = _yaml_view(yaml)
    return Inputs(_hash_or_missing(yaml), content, files, assumptions, facts, from_yaml, split)


@dataclass
class Changes:
    names: list[str]   # the files that changed (one name each, relative to the YAML folder); empty if nothing is stale
    facts: list[str]   # one sentence for each change, and the notes that are not a change
    method_facts: list[str] = field(default_factory=list)   # the facts of `facts` that name a changed method key (P9-4b)

    @property
    def stale(self) -> bool:
        return bool(self.names)


_UNSET = object()    # a method key that the brief or the YAML does not set


def _show_value(value) -> str:
    """A value in a fact: a text as it is, anything else as compact JSON, in back quotes. `_UNSET` is a key that is not there."""
    if value is _UNSET:
        return "not set"
    return "`" + (value if isinstance(value, str) else json.dumps(value, default=str)) + "`"


def _method_facts(before: dict, after: dict, tail: str) -> list[str]:
    """One fact for each method key whose value differs between the brief (`before`) and now (`after`)."""
    facts = []
    for path in sorted(set(before) | set(after)):
        a, b = before.get(path, _UNSET), after.get(path, _UNSET)
        if a is not _UNSET and b is not _UNSET and _canonical(a) == _canonical(b):
            continue
        facts.append(f"`{path}`: a method key changed from {_show_value(a)} to {_show_value(b)}{tail}")
    return facts


def _requirement_fact(before: dict, after: dict, yaml_name: str) -> str | None:
    """One fact that names the requirement keys (and `both` keys) whose digest differs, or None if no key differs."""
    names = []
    for path in sorted(set(before) | set(after)):
        a, b = before.get(path), after.get(path)
        if a != b:
            names.append(f"`{path or yaml_name}`" + (" (new)" if a is None else " (removed)" if b is None else ""))
    if not names:
        return None
    return f"{yaml_name}: requirement keys changed since the brief: {', '.join(names)}"


def _yaml_change(recorded: dict, now: Inputs, yaml_name: str, method_free: bool) -> tuple[bool, list[str], list[str]]:
    """Is a change of the YAML content stale, the facts, and the part of the facts that names method keys. Called when the
    content differs from the record.

    Whole comparison (a run entry): stale. With `method_free` (the brief): only a change of a method key is not stale. A record
    of a brief from before the split (P9-4b) has no split: it is stale as before, with the fact that says how to get the rule.
    """
    if now.yaml_content is None:
        return True, [f"{yaml_name}: the YAML changed and it does not parse now"], []
    content = f"{yaml_name}: the YAML content changed"
    if not method_free or now.split is None:
        return True, [content], []
    old_requirements, old_methods = recorded.get("yaml_requirements"), recorded.get("method_values")
    if not isinstance(old_requirements, str) or not isinstance(old_methods, dict):
        return True, [content, OLD_RECORD_FACT], []
    if old_requirements == now.split.requirements:
        facts = _method_facts(old_methods, now.split.methods, "; the brief stays current")
        return False, facts or [f"{yaml_name}: only method keys changed; the brief stays current"], facts
    keys = recorded.get("requirement_keys")
    named = _requirement_fact(keys if isinstance(keys, dict) else {}, now.split.keys, yaml_name)
    method = _method_facts(old_methods, now.split.methods, "")
    return True, [content] + ([named] if named else []) + method, method


def compare(recorded: dict, now: Inputs, yaml_name: str, method_free: bool = False) -> Changes:
    """What changed in the inputs since `recorded` (the `inputs` of a brief or of a run entry).

    A region file is stale when its hash differs, it is missing, or it is new. The YAML is stale when its bytes differ
    and its parsed content differs. A change of a comment or of the format only is a fact, not stale. If the YAML gave no
    file list (`now.from_yaml` is false), only the files of `recorded` are compared, so a file of another record is
    not reported as new.

    `method_free` is for the brief: a change of the values of method keys only (the cards of `topopt.levers_cards`) is a fact
    for each key and not stale. A run entry is compared with the whole content (`method_free` false): another solver is another run.
    """
    names: list[str] = []
    facts: list[str] = []
    method_facts: list[str] = []
    if recorded.get("yaml") != now.yaml_sha256:
        old = recorded.get("yaml_content")
        if old is not None and old == now.yaml_content:
            facts.append(f"{yaml_name}: the text changed and the content is the same (a comment or the format); "
                         "this is not stale")
        else:
            stale, why, method_facts = _yaml_change(recorded, now, yaml_name, method_free)
            if stale:
                names.append(yaml_name)
            facts += why
    before = recorded.get("files") or {}
    for key in sorted(set(before) | set(now.files)):
        a, b = before.get(key), now.files.get(key)
        if a == b:
            continue
        if not now.from_yaml and (a is None or b is None):
            continue    # the YAML gave no list: a file of another record, or an outside file, cannot be judged
        role, _, rel = key.partition(":")
        rel = _shown(rel)
        if a is None:
            what = "is a new file of the problem"
        elif b is None:
            what = "is not named by the YAML now"
        elif b == MISSING:
            what = "is missing"
        elif a == MISSING:
            what = "was missing and is there now"
        else:
            what = "changed"
        if rel not in names:
            names.append(rel)
        facts.append(f"{rel} ({role}): the file {what}")
    return Changes(names, facts, method_facts)


# --- the case file and the log -----------------------------------------------------------------

class CaseRead(NamedTuple):
    doc: dict | None
    facts: list[str]
    kind: str    # "none" (no case file), "ok", "unreadable" (not a case file) or "other_yaml" (made for another YAML)


def read_case(manifest: Path, yaml_name: str | None = None) -> CaseRead:
    """Read a case file. With `yaml_name`, a file made for another YAML (`hook.yaml` and `hook.yml` share a stem) is
    `other_yaml`: it is not read as the case of this YAML."""
    if not manifest.exists():
        return CaseRead(None, [], "none")
    doc = _read_json(manifest)
    if (doc is None or not isinstance(doc.get("runs") or [], list)
            or not isinstance(doc.get("brief"), (dict, type(None)))):
        return CaseRead(None, [f"{manifest.name} cannot be read as a case file"], "unreadable")
    owner = doc.get("yaml")
    if yaml_name is not None and isinstance(owner, str) and Path(owner).name != yaml_name:
        return CaseRead(None, [f"{manifest.name} is the case of {Path(owner).name}, not of {yaml_name}. The two YAML files "
                               "have the same name before the extension"], "other_yaml")
    return CaseRead(doc, [], "ok")


def _same_stem_options(yaml: Path) -> list[str]:
    return [f"Rename {yaml.name} so that its name differs from the other YAML before the extension, then call record_brief."]


def _entries(doc: dict | None) -> list[dict]:
    """The run entries of a case document (an entry that is not an object is left out)."""
    return [e for e in (doc or {}).get("runs") or [] if isinstance(e, dict)]


def save_case(manifest: Path, doc: dict) -> None:
    """Write the case document atomically. If the write fails, the last valid file stays and no `.tmp` is left."""
    data = (json.dumps(doc, indent=1) + "\n").encode()
    try:
        write_atomic(manifest, data)
    except BaseException:
        try:
            Path(str(manifest) + ".tmp").unlink()
        except OSError:
            pass
        raise


def log_event(yaml: Path, event: str, **fields) -> None:
    """Add one line to the log: one write, then a flush. A line that an earlier write left open is closed first."""
    line = json.dumps({"ts": _now(), "event": event, **fields}, sort_keys=False) + "\n"
    path = log_path(yaml)
    with open(path, "ab+") as f:
        f.seek(0, os.SEEK_END)
        lead = b""
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            lead = b"" if f.read(1) == b"\n" else b"\n"
        f.write(lead + line.encode())
        f.flush()


def _log_rows(yaml: Path) -> list[dict]:
    """The log lines that parse, in order. A part line is skipped. No log: no rows."""
    try:
        return read_history(log_path(yaml), strict=False).rows
    except OSError:
        return []


def _highest_brief_id(rows: list[dict]) -> int:
    ids = [r["brief"].get("id") for r in rows
           if r.get("event") == "brief_recorded" and isinstance(r.get("brief"), dict)]
    return max([i for i in ids if isinstance(i, int)] or [0])


def _lost_runs(rows: list[dict], entries: list[dict]) -> int:
    """How many `run_started` events of the log have no entry in the case file (same pid and start time)."""
    have = {(e.get("pid"), e.get("started_at")) for e in entries}
    return sum(1 for r in rows if r.get("event") == "run_started" and (r.get("pid"), r.get("started_at")) not in have)


def _new_doc(yaml: Path) -> dict:
    return {"format": FORMAT, "yaml": yaml.name, "created_at": _now(), "brief_count": 0, "brief": None, "runs": []}


# --- the brief ---------------------------------------------------------------------------------

def _clean_items(items, name: str) -> tuple[list[dict], list[str]]:
    """The items as `[{"text", "origin"}]` and the problems that were found."""
    if items is None:
        return [], []
    if not isinstance(items, list):
        return [], [f"{name} needs a list of objects {{\"text\": ..., \"origin\": ...}}"]
    out, problems = [], []
    for i, item in enumerate(items):
        text = item.get("text") if isinstance(item, dict) else None
        origin = item.get("origin") if isinstance(item, dict) else None
        if not isinstance(text, str) or not text.strip():
            problems.append(f"{name}[{i}] needs a non-empty text")
        elif origin not in ORIGINS:
            problems.append(f"{name}[{i}] needs an origin: {', '.join(ORIGINS)}")
        else:
            out.append({"text": text.strip(), "origin": origin})
    return out, problems


def _assumption_facts(brief: dict, yaml_assumptions: list[str] | None) -> list[str]:
    """A fact when the assumptions of the brief and the `assumptions` list of the YAML differ. No block."""
    if yaml_assumptions is None:
        return []
    mine = [a["text"] for a in brief.get("assumptions") or [] if isinstance(a, dict) and "text" in a]
    theirs = [a.strip() for a in yaml_assumptions]
    only_brief = [t for t in mine if t not in theirs]
    only_yaml = [t for t in theirs if t not in mine]
    facts = []
    if only_brief:
        facts.append(f"{len(only_brief)} assumption(s) of the brief are not in the `assumptions` list of the YAML "
                     f"(run.json copies that list): {_quote(only_brief)}")
    if only_yaml:
        facts.append(f"{len(only_yaml)} assumption(s) of the YAML are not in the brief: {_quote(only_yaml)}")
    return facts


def _quote(texts: list[str], n: int = 3, width: int = 60) -> str:
    shown = [t if len(t) <= width else t[: width - 1] + "..." for t in texts[:n]]
    return "; ".join(f'"{t}"' for t in shown) + ("; ..." if len(texts) > n else "")


def _outside_reply(yaml: Path) -> dict:
    return {
        "recorded": False,
        "case": None,
        "facts": [f"{yaml.name} is outside the work folder, so no case file can go beside it. A file in the code "
                  "folder (an example, or the plugin copy) is replaced by an update."],
        "options": ["Copy the YAML and the STL files it names into the work folder (for example runs/<name>/), "
                    "then call record_brief for the copy.",
                    "Call build_regions: it writes the YAML and the STL files in the work folder."],
    }


def _in_code_folder(yaml: Path, code: Path | None) -> bool:
    """True if the YAML lies in the code folder (a source checkout), but not in its `runs/` folder."""
    if code is None:
        return False
    try:
        rel = yaml.resolve().relative_to(Path(code).resolve())
    except ValueError:
        return False
    return not (rel.parts and rel.parts[0] == "runs")


def record_brief(yaml: Path, work: Path, text, assumptions=None, criteria=None, code_folder: Path | None = None) -> dict:
    """Record the brief of the case of `yaml` (mode `state`: say it and go on). Make the case if it does not exist.

    `code_folder` is the folder of the installed code. A YAML inside it (a source checkout with no work folder set) is
    recorded, with a fact and an option.
    """
    problems = []
    if not isinstance(text, str) or not text.strip():
        problems.append("text needs a non-empty string")
    items_a, p = _clean_items(assumptions, "assumptions")
    problems += p
    items_c, p = _clean_items(criteria, "criteria")
    problems += p
    if problems:
        return {"error": "The brief was not recorded: " + "; ".join(problems) + ".", "recorded": False}
    if not in_work_folder(yaml, work):
        return _outside_reply(yaml)

    manifest = manifest_path(yaml)
    read = read_case(manifest, yaml.name)
    doc, facts, options = read.doc, list(read.facts), []
    if read.kind == "other_yaml":
        return {"recorded": False, "case": None, "facts": facts, "options": _same_stem_options(yaml)}
    if read.kind == "unreadable":     # keep a case file that cannot be read: move it aside
        n = 1
        while manifest.with_name(f"{manifest.name}.bad{n}").exists():
            n += 1
        bad = manifest.with_name(f"{manifest.name}.bad{n}")
        try:
            os.replace(manifest, bad)
        except OSError as exc:
            return {"error": f"The brief was not recorded: {manifest.name} cannot be read and cannot be moved aside "
                    f"({_scrub(str(exc), yaml.parent)}).", "recorded": False}
        facts.append(f"the old case file was moved to {bad.name}")
    doc = doc or _new_doc(yaml)
    now = current_inputs(yaml, work)
    brief_id = max(int(doc.get("brief_count") or 0), _highest_brief_id(_log_rows(yaml))) + 1
    replaced = (doc.get("brief") or {}).get("id")
    brief = {"id": brief_id, "recorded_at": _now(), "mode": "state", "text": text.strip(),
             "assumptions": items_a, "criteria": items_c, "inputs": now.brief_record()}
    doc.update(format=FORMAT, yaml=yaml.name, brief_count=brief_id, brief=brief)
    try:
        save_case(manifest, doc)
    except OSError as exc:
        return {"error": f"The brief was not recorded: cannot write {manifest.name} ({_scrub(str(exc), yaml.parent)}). "
                "The last valid case file is unchanged.", "recorded": False}
    try:
        log_event(yaml, "brief_recorded", brief=brief, replaced=replaced)
    except OSError as exc:
        facts.append(f"the log was not written ({_scrub(str(exc), yaml.parent)})")
    if _in_code_folder(yaml, code_folder):
        facts.append(f"{yaml.name} lies in the code folder of this installation: git does not list a case file there by "
                     "default, and an update of the code can replace the folder")
        from .workfolder import change_text   # the words for the surface that started the server
        options.append("Copy the YAML and the STL files into a work folder, then call record_brief for the copy. " + change_text())
    return {
        "recorded": True,
        "case": _wrel(manifest, work),
        "yaml_rel": _wrel(yaml, work),
        "brief": brief,
        "replaced": replaced,
        "facts": facts + now.facts + _assumption_facts(brief, now.assumptions),
        "options": options,
    }


# --- the status --------------------------------------------------------------------------------

def _epoch(text) -> float | None:
    try:
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def _newer(run_dir: Path, started: float) -> bool:
    """True if a run file of the folder was written at or after `started` (one second of slack)."""
    for name in ("status.json", "run.json"):
        try:
            if (run_dir / name).stat().st_mtime >= started - 1:
                return True
        except OSError:
            continue
    return False


def _job_matches(job: dict | None, entry: dict) -> bool | None:
    """Is `job.json` the job of this run entry? True: the same pid, and start times that are near (a pid can be used
    again, so equal pids are not enough). False: another run. None: there is no pid to compare."""
    if not job or job.get("pid") is None or entry.get("pid") is None:
        return None
    if job["pid"] != entry["pid"]:
        return False
    theirs, mine = _epoch(job.get("started_at")), _epoch(entry.get("started_at"))
    if theirs is not None and mine is not None and abs(theirs - mine) > SAME_RUN_S:
        return False
    return True


def _shows(entry: dict, run_dir: Path, job: dict | None, unknown_start: bool) -> bool:
    """Does the run folder show data of this run entry? By the job file when it has a pid, else by the file times."""
    m = _job_matches(job, entry)
    if m is not None:
        return m
    started = _epoch(entry.get("started_at"))
    return unknown_start if started is None else _newer(run_dir, started)


def _run_view(entry: dict, later: list[dict], yaml: Path, work: Path, now: Inputs) -> dict:
    """What `case_status` says about one run entry. The state comes from the run folder, not from a copy.

    `later` are the later entries of the case with the same run folder. An entry is `replaced` only when the folder shows
    the data of one of them: the pid of its `job.json`, or (no job) a run file that is newer than its start. A later run
    that wrote nothing (it stopped early) does not replace an earlier one.
    """
    rel = entry.get("run_dir")
    run_dir = (yaml.parent / rel).resolve() if isinstance(rel, str) else None
    facts: list[str] = []
    state, iteration = "unknown", None
    if run_dir is None:
        facts.append("the run folder is outside the work folder; its state is not read")
    elif not run_dir.is_dir():
        state = "missing"
        facts.append("the run folder is not there now")
    else:
        job = _read_json(run_dir / "job.json")
        status = _read_json(run_dir / "status.json")
        run = _read_json(run_dir / "run.json")
        mine = _job_matches(job, entry)
        successor = None
        if mine is not True:
            successor = next((e for e in later if _shows(e, run_dir, job, False)), None)
        started = _epoch(entry.get("started_at"))
        if successor is not None:
            state = "replaced"
            facts.append(f"a later run of this case (pid {successor.get('pid')}, started {successor.get('started_at')}) "
                         "wrote into the same folder, so the folder holds that run now")
        elif mine is False:
            theirs = _epoch(job.get("started_at"))
            if theirs is not None and started is not None and theirs < started - 1:
                state = "not_written"     # the job is older than this run: this run has written nothing yet (or stopped early)
                facts.append(f"the folder holds an earlier run (pid {job['pid']}, started {job.get('started_at')}) and no "
                             "data of this run yet")
            else:
                state = "replaced"
                facts.append(f"this folder now holds another run (pid {job['pid']}, started {job.get('started_at')})")
        elif mine is None and started is not None and not _newer(run_dir, started):
            state = "not_written"
            facts.append("the folder holds no data of this run yet (an earlier run, or none)")
        else:
            state = (status or {}).get("state") or (job or {}).get("state") or "unknown"
            iteration = (status or {}).get("iteration")
            if run is None and status is None:
                facts.append("the folder has no run.json and no status.json")
    changes = compare(entry.get("inputs") or {}, now, yaml.name)
    return {
        "run_dir_rel": _wrel(run_dir, work) if run_dir is not None else None,
        "started_at": entry.get("started_at"),
        "pid": entry.get("pid"),
        "brief": entry.get("brief"),
        "state": state,
        "iteration": iteration,
        "stale": changes.stale,
        "stale_files": changes.names,
        "facts": facts + changes.facts,
    }


def _recorded_keys(brief: dict | None, runs: list[dict]) -> list[str]:
    """The file keys that the brief and the run entries hold (used when the YAML gives no file list)."""
    keys = list(((brief or {}).get("inputs") or {}).get("files") or {})
    for entry in runs:
        keys += [k for k in ((entry.get("inputs") or {}).get("files") or {}) if k not in keys]
    return keys


def status(yaml: Path, work: Path) -> dict:
    """The case of `yaml`: the brief, the input hashes now, the runs from their folders, and what is stale.

    Every reply has the same keys (CONTRACTS.md section 30). A YAML outside the work folder has no case: the keys are null
    or empty.
    """
    reply: dict = {"yaml_rel": _wrel(yaml, work)}
    if not in_work_folder(yaml, work):
        out = _outside_reply(yaml)
        return {**reply, "exists": False, "case": None, "log": None, "brief": None, "brief_stated": False,
                "brief_stale": False, "stale_files": [], "inputs": None, "runs": [],
                "gate": "run_topopt starts; a YAML outside the work folder has no case",
                "facts": out["facts"], "options": out["options"]}
    manifest = manifest_path(yaml)
    read = read_case(manifest, yaml.name)
    doc, facts = read.doc, list(read.facts)
    if read.kind == "other_yaml":
        return {**reply, "exists": False, "case": None, "log": None, "brief": None, "brief_stated": False,
                "brief_stale": False, "stale_files": [], "inputs": current_inputs(yaml, work).record(), "runs": [],
                "gate": "run_topopt starts; this YAML has no case of its own",
                "facts": facts, "options": _same_stem_options(yaml)}
    brief = (doc or {}).get("brief")
    runs = _entries(doc)
    now = current_inputs(yaml, work, _recorded_keys(brief, runs))
    facts += now.facts
    options: list[str] = []
    changes = Changes([], [])
    if doc is None:
        if not manifest.exists():
            facts.append(NO_CASE_FACT)
        options.append("Call record_brief with the brief text and the assumptions. It makes the case.")
    elif brief is None:
        facts.append(f"no brief is recorded for {yaml.name}")
        options.append("Call record_brief with the brief text and the assumptions.")
    else:
        changes = compare(brief.get("inputs") or {}, now, yaml.name, method_free=True)
        if changes.stale:
            facts.append(f"the brief (number {brief.get('id')}, recorded {brief.get('recorded_at')}) is stale: "
                         f"{', '.join(changes.names)} changed since then")
            options.append("Call record_brief again with the new brief. It replaces this one.")
        facts += changes.facts
        facts += _assumption_facts(brief, now.assumptions)
    lost = _lost_runs(_log_rows(yaml), runs)
    if lost:
        facts.append(f"the log lists {lost} run start(s) that {manifest.name} does not hold (a lost write; two servers on one "
                     "work folder can do this). The log keeps them")
    stated = brief is not None and not changes.stale
    views = []
    for i in range(len(runs) - 1, -1, -1):    # newest first
        e = runs[i]
        later = [x for x in runs[i + 1:] if x.get("run_dir") is not None and x.get("run_dir") == e.get("run_dir")]
        views.append(_run_view(e, later, yaml, work, now))
    return {
        **reply,
        "exists": doc is not None,
        "case": _wrel(manifest, work) if manifest.exists() else None,
        "log": _wrel(log_path(yaml), work) if log_path(yaml).exists() else None,
        "brief": brief,
        "brief_stated": stated,
        "brief_stale": bool(brief is not None and changes.stale),
        "stale_files": changes.names,
        "inputs": now.record(),
        "runs": views,
        "gate": ("run_topopt starts" if stated or not manifest.exists() else
                 f"run_topopt returns awaiting_user; the override is {OVERRIDE}"),
        "facts": facts,
        "options": options,
    }


# --- the soft gate of run_topopt -----------------------------------------------------------------

@dataclass
class Gate:
    """The decision of the gate for one `run_topopt` call."""
    stop: dict | None = None                              # the reply of run_topopt when it stops (awaiting_user)
    reply: dict = field(default_factory=dict)             # keys for the success reply
    case: bool = False                                    # a case file exists for this YAML
    inputs: Inputs | None = None                          # the hashes at the start of the run
    brief_id: int | None = None                           # the brief that was current, or None
    override: dict | None = None                          # set when the override lets the run start
    method_facts: list[str] = field(default_factory=list)   # the facts about changed method keys (P9-4b); `run_started` logs them


def gate(yaml: Path, work: Path, proceed_without_brief: bool = False, resume: bool = False) -> Gate:
    """Decide whether `run_topopt` starts. It reads files and writes one log line when it stops. It stops nothing else.

    No case file (or a YAML outside the work folder): the run goes on as without D2, with `brief_stated: false`.
    A case with a current brief: the run goes on with `brief_stated: true`. A change of method keys only is not stale (P9-4b):
    `brief_facts` then has one fact for each changed key.
    A case with no brief or a stale brief: `awaiting_user` with the facts, the options and the override. With
    `proceed_without_brief` the run goes on, and `run_started` writes the override to the log.
    `resume` is the `resume` argument of `run_topopt`: a resume has a second possible stop (changed inputs, override
    `allow_changed_inputs=true`), so the stop of the gate names both overrides in one option.
    """
    try:
        return _gate(yaml, work, proceed_without_brief, resume)
    except Exception as exc:   # a failure of the case check does not stop a run
        return Gate(reply={"brief_stated": False,
                           "brief_facts": [f"the case check failed ({type(exc).__name__}: "
                                           f"{_scrub(str(exc), yaml.parent)}); the run goes on"]})


def _gate(yaml: Path, work: Path, proceed: bool, resume: bool = False) -> Gate:
    if not in_work_folder(yaml, work):
        return Gate(reply={"brief_stated": False, "brief_facts": [
            f"{yaml.name} is outside the work folder, so it has no case; copy it and its STL files into the work folder"]})
    manifest = manifest_path(yaml)
    if not manifest.exists():
        return Gate(reply={"brief_stated": False, "brief_facts": [NO_CASE_FACT]})
    read = read_case(manifest, yaml.name)
    if read.kind == "other_yaml":    # the case file is of another YAML: this YAML has no case, and the run starts
        return Gate(reply={"brief_stated": False, "brief_facts": read.facts + [NO_CASE_FACT]})
    doc, facts = read.doc, list(read.facts)
    brief = (doc or {}).get("brief")
    now = current_inputs(yaml, work, _recorded_keys(brief, []))
    facts = facts + now.facts
    method_facts: list[str] = []
    if brief is None:
        reason, names = "no_brief", []
        facts.append(f"no brief is recorded for {yaml.name}")
    else:
        changes = compare(brief.get("inputs") or {}, now, yaml.name, method_free=True)
        method_facts = changes.method_facts
        if not changes.stale:
            return Gate(case=True, inputs=now, brief_id=brief.get("id"), method_facts=method_facts,
                        reply={"brief_stated": True, "brief_facts": facts + changes.facts})
        reason, names = "stale", changes.names
        facts.append(f"the brief (number {brief.get('id')}, recorded {brief.get('recorded_at')}) is stale: "
                     f"{', '.join(names)} changed since then")
        facts += changes.facts
    if proceed:
        return Gate(case=True, inputs=now, reply={"brief_stated": False, "brief_facts": facts}, method_facts=method_facts,
                    override={"reason": reason, "stale_files": names})
    try:
        log_event(yaml, "gate_returned", state="awaiting_user", reason=reason, stale_files=names)
    except OSError as exc:
        facts.append(f"the log was not written ({_scrub(str(exc), yaml.parent)})")
    if resume:
        # A changed input file is also a change since the checkpoint: the resume has its own stop for that. One option
        # names both overrides, so the agent does not need two stops.
        last = (f"Resume without a brief: call run_topopt again with resume=true and {OVERRIDE}. If an input changed since the "
                f"checkpoint (the stale files above can be such a change), add {RESUME_OVERRIDE}: the resume then goes on with "
                "the new inputs. The case log records the first override; the run log and the verdict name the changes.")
    else:
        last = f"Run without a brief: call run_topopt again with {OVERRIDE}. The case log records it."
    return Gate(case=True, stop={
        "state": "awaiting_user",
        "facts": facts,
        "options": [
            "Tell the user the assumptions. Call record_brief with the brief text and the assumptions. "
            "Then call run_topopt again.",
            "Call case_status for the brief, the changed files and the earlier runs.",
            last,
        ],
        "override": OVERRIDE,
        "case": _wrel(manifest, work),
        "yaml_rel": _wrel(yaml, work),
    })


def run_started(g: Gate, yaml: Path, work: Path, run_dir: Path, pid: int, started: float | None = None) -> dict:
    """Record a run that has started (case mode only). Return the keys for the success reply of `run_topopt`.

    `started` is the epoch time of the start (the time when the process started). Default: now.

    It adds one run entry to `case.json` and the events `override_used` and `run_started` to the log. The `run_started` event also
    has `method_facts`: the facts about method keys that differ from the brief (an empty list if none). The entry has none. A failure
    does not stop the run: it is one more fact in `brief_facts`. A run folder outside the work folder has no path in
    the entry: `run_dir` is null and `run_dir_name` is `<outside>/<12 hex>` (no folder name).
    """
    reply = dict(g.reply)
    if not g.case or g.inputs is None:
        return reply
    facts = list(reply.get("brief_facts") or [])
    try:
        rel = None
        if in_work_folder(run_dir, work):
            rel = Path(os.path.relpath(run_dir, yaml.parent)).as_posix()
        stamp = _now() if started is None else datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds")
        entry = {"run_dir": rel, "started_at": stamp, "pid": pid, "brief": g.brief_id, "inputs": g.inputs.record()}
        if rel is None:
            entry["run_dir_name"] = outside_key(run_dir)
        manifest = manifest_path(yaml)
        read = read_case(manifest, yaml.name)
        if read.doc is None:
            facts.append("the run is not in the case file: " + (read.facts[0] if read.facts else "the case file is gone"))
        else:
            read.doc.setdefault("runs", []).append(entry)
            try:
                save_case(manifest, read.doc)
            except OSError as exc:
                facts.append(f"the run is not in {manifest.name}: {_scrub(str(exc), yaml.parent)}")
        if g.override:
            log_event(yaml, "override_used", override=OVERRIDE, **g.override)
        log_event(yaml, "run_started", **entry, method_facts=list(g.method_facts))
    except Exception as exc:   # the run has started: this only adds a fact
        facts.append(f"the case log was not written ({type(exc).__name__}: {_scrub(str(exc), yaml.parent)})")
    reply["brief_facts"] = facts
    return reply
