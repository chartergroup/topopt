"""The preferences file of the work folder (CONTRACTS.md section 10.2).

`<work folder>/topopt-preferences.yaml` keeps the answers that a user gives once: the print bed and the printer, the material,
the quality (the voxel size) and the outputs. The file is visible, so a person can find it and edit it. The MCP server reads it at
each call (no restart), and `save_preferences` writes it.

This module does not import `mcp`. It imports `topopt.config` (the rules of the problem YAML) only when a value is checked.

Format 1 (every section is optional):

- `print`: `printer_model`, `printer_type`, `bed_mm`, `build_direction`, `self_supporting_angle_deg`. The names, values and checks of
  the `print` block of the problem YAML (`topopt.config.parse_print_block`). `printer_model` is a printer of the catalog
  (`topopt.printers`): the file keeps the id, and a text that is not one printer is refused with the candidates. The user names only the
  printer; the bed and the type come from its profile when the YAML does not give them. A print LIMIT key (`min_wall_mm`, ...) belongs
  to one part, so it is not a preference.
- `material`: `E`, `nu`, `density`, `filament` with the checks of the `material` block (`topopt.config.parse_material_block`), and an
  optional free text `name`. `filament` is a filament of the catalog (`topopt.materials_catalog`): the file keeps the id, and a text that
  is not one filament is refused with the candidates. The user names only the filament; E and density come from its profile when the YAML
  does not give them.
- `filaments`: `available`, a list of the filaments that the user has (ids of the catalog or of the user filament file). The agent shows
  `printer_materials` for them and does not ask for a filament again.
- `quality`: `voxel_size` in mm, with the check of `settings.voxel_size`. Nothing else.
- `outputs`: a list of `renders` (`render_part`), `step` (`export_step`) and `verify` (`verify_run`). The final STL and the part
  card are made in every run, so they are not choices. `[]` means none of the three.

A key that this module does not know (a key that a person added, or a key of a later format) is kept in the file as it is. `read`
names it in `kept_not_used`, and it is not a problem of the file. An input key that is not known is a refusal.

A preference does not change a run by itself: `run_topopt`, `check_problem` and `topopt run` do not read this file to fill a YAML. Two
functions use it, and both say so in the reply: `fill_spec` (for `build_regions`: a block that the spec does not have at all) and
`facts_for_yaml` (for `check_problem`: a fact about a difference).
"""
from __future__ import annotations

import errno
import math
import re
import sys
from pathlib import Path

import yaml

from . import runinfo, workfolder

FILE_NAME = "topopt-preferences.yaml"
FORMAT = 1
PRINT_KEYS = ("printer_model", "printer_type", "bed_mm", "build_direction", "self_supporting_angle_deg")
MATERIAL_KEYS = ("E", "nu", "density", "filament", "name")
QUALITY_KEYS = ("voxel_size",)
FILAMENTS_KEYS = ("available",)
SECTION_KEYS = {"print": PRINT_KEYS, "material": MATERIAL_KEYS, "quality": QUALITY_KEYS, "filaments": FILAMENTS_KEYS}   # the sections that are mappings
SECTIONS = ("print", "material", "quality", "filaments", "outputs")
TOP_KEYS = ("format",) + SECTIONS
OUTPUT_TOOLS = {"renders": "render_part", "step": "export_step", "verify": "verify_run"}   # name -> the tool that makes it
OUTPUTS = tuple(OUTPUT_TOOLS)

HEADER = (f"# topopt preferences (format {FORMAT}). Edit this file by hand, or ask Claude to change it (the tool save_preferences).\n"
          "# The server reads it at each call. A preference does not change a run by itself: the problem YAML is the record of a run.\n"
          "# Sections: print, material, quality, filaments, outputs. A comment that you add here is not kept when save_preferences writes the file.\n")

NOTE_NONE = ("No preferences are saved in the work folder: save_preferences keeps the answers of the user (print bed, material, "
             "filaments, quality, outputs) for the next runs.")
NOTE_READ = ("The server reads this file at each call. A preference does not change a run by itself: build_regions writes it into a YAML "
             "only for a block that the spec does not have, and says so; check_problem names a difference. "
             "save_preferences changes the file, and a person can edit it.")
NOTE_PROBLEMS = ("The file has problems, so the server uses no value from it. Fix the file by hand, or give the wrong keys to "
                 "save_preferences (it does not change a file that is not valid YAML).")
NOTE_UNREADABLE = ("The server cannot read the file, so it uses no value from it. The cause is the rights of the file or the folder, "
                   "not the content of the file.")

_FILE_SUFFIX = re.compile(r" \(file: .*\)$", re.S)


class _Dumper(yaml.SafeDumper):
    """Mappings in block style (a person edits the file), lists of numbers and names in one line."""


_Dumper.add_representer(list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True))


class Parsed:
    """The result of the checks of a mapping: `values` (the sections that are valid), `unknown` (dotted key names) and `problems`."""

    def __init__(self):
        self.values: dict = {}
        self.unknown: list[str] = []
        self.problems: list[str] = []
        self.unused: list[str] = []          # a printer_model that is not found now: only this key is not used (it is not a problem of the file)


def _clean(exc: Exception) -> str:
    """The text of an error as one line, without the `(file: ...)` end that the loader adds."""
    return _FILE_SUFFIX.sub("", " ".join(str(exc).split()))


def _oneline(exc: Exception) -> str:
    return " ".join(str(exc).split())


# --- the checks of one value -------------------------------------------------------------------------------------------------

def _catalog(folder):
    """The shipped printers and the user printers of `folder`; None if the shipped catalog cannot be read (the loader then says so)."""
    from . import printers
    try:
        return printers.catalog_for(folder)
    except printers.CatalogError:
        return None


def _filament_catalog(folder):
    """The shipped filaments and the user filaments of `folder`; None if the shipped catalog cannot be read (the loader then says so)."""
    from . import materials_catalog
    try:
        return materials_catalog.catalog_for(folder)
    except materials_catalog.CatalogError:
        return None


def _check_print_key(key: str, value, path):
    from .config import PRINTER_TYPES, UserError, parse_print_block
    if key == "printer_type" and isinstance(value, (list, dict)):    # the loader fails on an unhashable value with a TypeError
        raise UserError(f"'print.printer_type' must be one of {', '.join(PRINTER_TYPES)}, got {value!r}")
    # the checks of the problem YAML (for printer_model: the resolver, with the printers of the folder of this file)
    ps = parse_print_block({key: value}, path, _catalog(Path(path).parent) if key == "printer_model" else None)
    if key == "printer_model":
        return ps.printer_model                      # the id, also when the user gave an alias
    if key == "printer_type":
        return ps.printer_type
    if key == "bed_mm":
        return [float(x) for x in ps.bed_mm]
    if key == "build_direction":
        return value.strip().lower() if isinstance(value, str) else [float(x) for x in value]
    return ps.self_supporting_angle_deg


def _check_material_key(key: str, value, path):
    from .config import UserError, parse_filament_name, parse_material_block
    if key == "name":
        if not isinstance(value, str) or not value.strip():
            raise UserError(f"'material.name' must be a non-empty text, got {value!r}")
        return value.strip()
    if key == "filament":                           # the resolver, with the filaments of the folder of this file; the file keeps the id
        return parse_filament_name(value, path, _filament_catalog(Path(path).parent)).id
    return getattr(parse_material_block({key: value}, path), key)


def _check_filaments_key(key: str, value, path, unused: list | None = None):
    """`filaments.available`: a list of filaments (ids, names or aliases). The file keeps the ids, once each, in the order given.

    With `unused` (a list: the read of a FILE), a filament that is not found now goes into it as a text and only that id is left out; the
    other ids stay. Without it (an input of `save_preferences`), the same text is an error."""
    from . import materials_catalog
    from .config import FilamentNotFound, UserError
    if not isinstance(value, list):
        raise UserError(f"'filaments.available' must be a list of filament names or ids, got {value!r}")
    catalog = _filament_catalog(Path(path).parent)
    ids: list[str] = []
    for i, item in enumerate(value):
        where = f"filaments.available[{i}]"
        if not isinstance(item, str) or not item.strip():
            raise UserError(f"'{where}' must be the name or the id of a filament as text, got {item!r}. {materials_catalog.WAYS_ON_LIST}")
        try:
            res = materials_catalog.resolve(item, catalog)
        except materials_catalog.CatalogError as exc:
            raise FilamentNotFound(f"'{where}' cannot be checked: {exc}")
        if res.status != "match":
            lost = FilamentNotFound(materials_catalog.refusal_text(res, key=where, ways_on=materials_catalog.WAYS_ON_LIST))
            if unused is None:
                raise lost
            unused.append(_clean(lost))
            continue
        if res.profile.id not in ids:
            ids.append(res.profile.id)
    return ids


def _check_quality_key(key: str, value, path):
    from .config import parse_voxel_size
    return parse_voxel_size(value, path, "quality.voxel_size")


_KEY_CHECKS = {"print": _check_print_key, "material": _check_material_key, "quality": _check_quality_key,
               "filaments": _check_filaments_key}


def _check_section(name: str, raw, path, allow_null: bool = False, unused: list | None = None):
    """Check one section that is a mapping. Returns `(values, unknown, errors, removals)`.

    `unknown` are the keys that this module does not know (the caller decides: kept in a file, refused in an input). With
    `allow_null` a key with the value None is a request to remove the key: it is in `removals` and is not checked. With `unused` (a
    list), a printer that is not found now goes into that list as a text and is left out of the values: the printer files change, and the
    rest of the file is still good (MIN-3); a filament that is not found now is treated the same way. Without it (an input), the same text is an error.
    """
    from .config import FilamentNotFound, PrinterNotFound, UserError
    values, unknown, errors, removals = {}, [], [], []
    if not isinstance(raw, dict):
        return values, unknown, [f"{name}: must be a mapping of keys ({', '.join(SECTION_KEYS[name])}), got {raw!r}"], removals
    for key, value in raw.items():
        if key not in SECTION_KEYS[name]:
            unknown.append(key)
            continue
        if value is None and allow_null:
            removals.append(key)
            continue
        try:
            values[key] = _check_filaments_key(key, value, path, unused) if name == "filaments" else _KEY_CHECKS[name](key, value, path)
        except (UserError, TypeError, ValueError, OverflowError) as exc:    # a file that a person edited can hold any value
            if unused is not None and isinstance(exc, (PrinterNotFound, FilamentNotFound)):
                unused.append(_clean(exc))
                continue
            text = _clean(exc) if isinstance(exc, UserError) else f"cannot be used, got {value!r}"
            errors.append(text if text.startswith(f"'{name}.{key}") else f"{name}.{key}: {text}")
    ordered = {k: values[k] for k in SECTION_KEYS[name] if k in values}
    return ordered, unknown, errors, removals


def _unknown_text(name: str, key) -> str:
    from .config import PRINT_LIMIT_KEYS
    allowed = ", ".join(SECTION_KEYS[name])
    if name == "print" and key in PRINT_LIMIT_KEYS:
        return (f"{name}.{key}: a print limit belongs to one part, so it is not a preference: write it in the print block of the "
                f"problem YAML. Allowed keys: {allowed}")
    return f"{name}.{key}: not a key of this section. Allowed keys: {allowed}"


def _check_outputs(raw, unknown: list | None = None):
    """Returns `(list, errors)`. A name is checked after the blanks are cut and the case is lowered; a repeated name is kept once.

    A name that is not known is an error (the input of `save_preferences`). With `unknown` (a list), a text that is not known goes into
    that list instead, as `outputs.<name>`: a name of a later format in a FILE is kept, not used, and the known names are used (R1 m4).
    """
    allowed = ", ".join(f"{name} ({tool})" for name, tool in OUTPUT_TOOLS.items())
    if not isinstance(raw, list):
        return [], [f"outputs: must be a list of names, got {raw!r}. Allowed names: {allowed}"]
    out, errors = [], []
    for item in raw:
        name = item.strip().lower() if isinstance(item, str) else item
        if name not in OUTPUTS:
            if unknown is not None and isinstance(name, str) and name:
                if f"outputs.{name}" not in unknown:
                    unknown.append(f"outputs.{name}")
                continue
            errors.append(f"outputs: {item!r} is not an output name. Allowed names: {allowed}")
        elif name not in out:
            out.append(name)
    return out, errors


def parse_mapping(raw: dict, path) -> Parsed:
    """Check the mapping of a file. A key that is not known is in `unknown` (dotted), not in `problems`."""
    out = Parsed()
    fmt = raw.get("format", FORMAT)
    if type(fmt) is not int or fmt != FORMAT:                        # the int 1: not 1.0, not True
        out.problems.append(f"format: is {fmt!r}, and this server reads format {FORMAT}")
    out.unknown += [str(k) for k in raw if k not in TOP_KEYS]
    for name in SECTION_KEYS:
        if raw.get(name) is None:           # no section, or a section with no keys (`print:` and nothing after it)
            continue
        values, unknown, errors, _ = _check_section(name, raw[name], path, unused=out.unused)
        out.problems += errors
        out.unknown += [f"{name}.{k}" for k in unknown]
        if values:
            out.values[name] = values
    if raw.get("outputs") is not None:
        outputs, errors = _check_outputs(raw["outputs"], unknown=out.unknown)
        out.problems += errors
        if not errors:
            out.values["outputs"] = outputs
    return out


def _load_file(path: Path):
    """Returns `(state, mapping, problems)`. `state` is `absent`, `unreadable` (the system does not let the server read the file: its
    content is not known), `broken` (the file is there but cannot be used as a mapping) or `ok`."""
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return "absent", None, []
    except IsADirectoryError:
        return "broken", None, ["cannot read the file: the name is a folder"]
    except UnicodeDecodeError as exc:
        return "broken", None, [f"not valid UTF-8 text: {_oneline(exc)}"]
    except OSError as exc:
        cause = f"cannot read the file: {exc.strerror or _oneline(exc)}"
        if exc.errno == errno.EPERM and sys.platform == "darwin":      # macOS asks the user before an app reads a folder such as ~/Documents
            cause += f". {workfolder.MAC_PRIVACY}"
        return "unreadable", None, [cause]
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return "broken", None, [f"not valid YAML: {_oneline(exc)}"]
    if data is None:                                    # an empty file
        data = {}
    if not isinstance(data, dict):
        return "broken", None, [f"the file must hold a mapping of keys ({', '.join(TOP_KEYS)}), got {type(data).__name__}"]
    return "ok", data, []


# --- reading -----------------------------------------------------------------------------------------------------------------

def path_of(folder) -> Path:
    return Path(folder) / FILE_NAME


def read(folder) -> dict:
    """The facts for the key `preferences` of `environment`: `path`, `exists`, `values` (when the file is valid), `problems` (when it
    is not), `kept_not_used` (keys that are not known; only when there are some) and `note`. It never raises: a broken file is a fact."""
    path = path_of(folder)
    state, raw, problems = _load_file(path)
    out = {"path": str(path), "exists": state != "absent"}
    if state == "absent":
        out["note"] = NOTE_NONE
        return out
    unknown: list[str] = []
    unused: list[str] = []
    values: dict = {}
    if state == "ok":
        parsed = parse_mapping(raw, path)
        problems, unknown, values, unused = parsed.problems, parsed.unknown, parsed.values, parsed.unused
    if problems:
        out["problems"] = problems
        note = NOTE_UNREADABLE if state == "unreadable" else NOTE_PROBLEMS
    else:
        out["values"] = values
        note = NOTE_READ
    if unknown:
        out["kept_not_used"] = unknown
        note += f" Kept, not used: {', '.join(unknown)}."
    if unused:
        out["not_used"] = unused
        what, fix = _unused_words(_problem_keys(unused))
        note += f" Not used now: {', '.join(_problem_keys(unused))} ({what} is not found; the rest of the file is used). {fix}"
    out["note"] = note
    return out


def usable(folder):
    """`(path, values, problems)` for a tool that may use the file. `values` is None when there is no file or the file has problems;
    `problems` is [] unless the file is there and has problems. A printer_model that is not found now is left out of `values`."""
    path, values, problems, _ = _usable(folder)
    return path, values, problems


def _usable(folder):
    """`usable` and a fourth item: the texts of the keys that the file has but the server does not use now (see `not_used` of `read`)."""
    facts = read(folder)
    if not facts["exists"]:
        return facts["path"], None, [], []
    if "problems" in facts:
        return facts["path"], None, facts["problems"], []
    return facts["path"], facts["values"], [], facts.get("not_used", [])


# --- writing -----------------------------------------------------------------------------------------------------------------

def _problem_keys(problems: list[str]) -> list[str]:
    """The key that each problem text starts with (`'print.bed_mm' must be ...` gives `print.bed_mm`), once each, in order."""
    keys: list[str] = []
    for text in problems:
        found = re.match(r"'?([A-Za-z_][\w.]*)", text)
        key = found.group(1) if found else text
        if key not in keys:
            keys.append(key)
    return keys


def _unused_words(keys: list[str]) -> tuple[str, str]:
    """For the keys that are not used now: what is not found (`the printer`, `the filament`) and how to fix it."""
    printer = any(k.startswith("print.") for k in keys)
    filament = any(k.startswith(("material.filament", "filaments.")) for k in keys)
    what = "the printer" if printer and not filament else "the filament" if filament and not printer else "the printer or the filament"
    fixes = []
    if printer:
        fixes.append("Save the printer again with save_printer, or give another printer to save_preferences.")
    if filament:
        fixes.append("Save the filament again with save_material, or give another filament to save_preferences.")
    return what, " ".join(fixes)


def _refuse(text: str) -> dict:
    return {"ok": False, "error": text}


def save(folder, print=None, material=None, quality=None, outputs=None, filaments=None) -> dict:    # noqa: A002 (the names are the arguments of the tool)
    """Merge the given sections into `<folder>/topopt-preferences.yaml` and write the file.

    A section that is not given stays as it is. A mapping section is merged key by key: a given key replaces the key, a key with the
    value None is removed, and `{}` removes the section (a key that this module does not know stays). `outputs` is a list that
    replaces the list; `{}` removes it. Every given value is checked first; one wrong value refuses the call and nothing is written.
    With no section at all, the call makes the file if it is not there (the user kept the defaults) and changes nothing else.
    A file that is not valid YAML, is not a mapping, or has another `format` is not changed.
    """
    path = path_of(folder)
    given = {k: v for k, v in (("print", print), ("material", material), ("quality", quality), ("filaments", filaments), ("outputs", outputs))
             if v is not None}
    state, raw, problems = _load_file(path)
    if state == "unreadable":                                           # the content is not known: it may be a good file
        return _refuse(f"save_preferences did not change {path}: {problems[0]}. It does not write over a file that it cannot read.")
    if state == "broken":
        return _refuse(f"save_preferences did not change {path}: {problems[0]}. Fix the file by hand, or remove it.")
    old = raw if raw is not None else {}
    if "format" in old and (type(old["format"]) is not int or old["format"] != FORMAT):
        return _refuse(f"save_preferences did not change {path}: format is {old['format']!r}, and this server writes format {FORMAT}. "
                       "Fix the file by hand, or remove it.")

    errors: list[str] = []
    checked: dict = {}
    for name, value in given.items():
        if name == "outputs":
            if isinstance(value, dict) and not value:
                checked[name] = None                                   # {}: remove the section
                continue
            listed, errs = _check_outputs(value)
            errors += errs
            checked[name] = listed
            continue
        if isinstance(value, dict) and not value:
            checked[name] = None
            continue
        values, unknown, errs, removals = _check_section(name, value, path, allow_null=True)
        errors += errs + [_unknown_text(name, k) for k in unknown]
        checked[name] = (values, removals)
    if errors:
        return _refuse("save_preferences wrote nothing. " + "; ".join(errors) + ".")

    doc: dict = {"format": FORMAT}
    for name in SECTIONS:
        current = old.get(name)
        if name not in checked:
            if name in old:
                doc[name] = current
            continue
        change = checked[name]
        if name == "outputs":
            if change is not None:
                doc[name] = change
            continue
        kept = {k: v for k, v in current.items() if k not in SECTION_KEYS[name]} if isinstance(current, dict) else {}
        if change is None:                                              # remove the section; the keys that are not known stay
            merged = {}
        else:
            values, removals = change
            known = {k: v for k, v in current.items() if k in SECTION_KEYS[name]} if isinstance(current, dict) else {}
            for key in removals:
                known.pop(key, None)
            known.update(values)
            merged = {k: known[k] for k in SECTION_KEYS[name] if k in known}
        merged.update(kept)
        if merged:
            doc[name] = merged
    doc.update({k: v for k, v in old.items() if k not in TOP_KEYS})    # keys that are not known stay, in their order

    changed = [name for name in SECTIONS if old.get(name) != doc.get(name)]
    made = state == "absent"
    if made or changed:
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            text = HEADER + yaml.dump(doc, Dumper=_Dumper, sort_keys=False, default_flow_style=False, allow_unicode=True)
            runinfo.write_atomic(path, text.encode("utf-8"))
        except OSError as exc:
            text = f"save_preferences could not write {path}: {exc.strerror or exc}."
            if exc.errno == errno.EPERM and sys.platform == "darwin":      # macOS asks the user before an app writes in a folder such as ~/Documents
                text += f" {workfolder.MAC_PRIVACY}."
            return _refuse(text)

    parsed = parse_mapping(doc, path)
    reply = {"ok": True, "path": str(path), "values": parsed.values, "changed": changed, "created": made}
    if parsed.unknown:
        reply["kept_not_used"] = parsed.unknown
    if parsed.problems:
        reply["problems"] = parsed.problems
    if parsed.unused:
        reply["not_used"] = parsed.unused
    if made and not changed:
        note = "The file is made with no value: the answer was to keep the defaults, so the question is not asked again."
    elif not changed:
        note = "Nothing changed: the file already has these values."
    elif parsed.problems:
        note = "Saved."                                              # not "live now": the server uses nothing from a file with problems
    else:
        note = "Saved. The server reads the file at each call, so the values are live now."
    if parsed.problems:
        note += (f" The file still has problems with {', '.join(_problem_keys(parsed.problems))}. The server uses no value from the file "
                 "until they are fixed, so none of these values is live; `values` shows only the part that is valid.")
    if parsed.unused:
        keys = _problem_keys(parsed.unused)
        what = _unused_words(keys)[0].replace("the ", "a ")
        note += (f" The file has {what} that is not found now ({', '.join(keys)}), so "
                 f"{'that key is' if len(keys) == 1 else 'those keys are'} not live; the other values are.")
    reply["note"] = note
    return reply


# --- use: build_regions ------------------------------------------------------------------------------------------------------

def _num(x) -> str:
    return f"{x:g}"


def _bed_text(bed) -> str:
    return f"{_num(bed[0])} x {_num(bed[1])} mm"


PRINT_LABELS = {"printer_model": "printer model", "printer_type": "printer", "bed_mm": "bed", "build_direction": "build direction",
                "self_supporting_angle_deg": "self-supporting angle"}


def _model_text(value, catalog=None) -> str:
    """A printer model as text: the id, with the name and the numbers that the profile gives (`catalog`: default the shipped one)."""
    if value is None:
        return "none"
    from . import printers
    try:
        profile = printers.profile_for(value, catalog)
    except printers.CatalogError:
        profile = None
    return f"{value} ({profile.name}: {printers.facts_text(profile)})" if profile is not None else str(value)


def _print_value_text(key: str, value, catalog=None) -> str:
    if key == "printer_model":
        return _model_text(value, catalog)
    if key == "bed_mm":
        return _bed_text(value)
    if key == "build_direction":
        return value if isinstance(value, str) else "(" + ", ".join(_num(x) for x in value) + ")"
    if key == "self_supporting_angle_deg":
        return f"{_num(value)} deg"
    return str(value)


def print_text(values: dict, catalog=None) -> str:
    """The `print` preferences as a short text."""
    return ", ".join(f"{PRINT_LABELS[k]} {_print_value_text(k, values[k], catalog)}" for k in PRINT_KEYS if k in values)


def _filament_text(value, catalog=None) -> str:
    """A filament as text: the id, with the name that the profile gives (`catalog`: default the shipped one)."""
    from . import materials_catalog
    try:
        profile = materials_catalog.profile_for(value, catalog)
    except materials_catalog.CatalogError:
        profile = None
    return f"{value} ({profile.name})" if profile is not None else str(value)


def material_text(values: dict, catalog=None) -> str:
    """The `material` preferences as a short text, with the units (`catalog`: where a filament is looked up; default the shipped one)."""
    parts = []
    if "name" in values:
        parts.append(values["name"])
    if "filament" in values:
        parts.append(f"filament {_filament_text(values['filament'], catalog)}")
    for key, label, unit in (("E", "E", " MPa"), ("nu", "nu", ""), ("density", "density", " g/cm3")):
        if key in values:
            parts.append(f"{label} {_num(values[key])}{unit}")
    return ", ".join(parts)


def filaments_text(values: dict, catalog=None) -> str:
    """The `filaments` preferences as a short text."""
    ids = values.get("available") or []
    return ", ".join(_filament_text(i, catalog) for i in ids) if ids else "none"


def _filament_fill(material: dict, path, folder) -> tuple[str, bool]:
    """The words about the filament of a `material` block that is filled from the preferences, and if the filament can be used.

    The numbers are the ones that the loader takes (`config.parse_filament_block`), so the fact shows what the run will use. A key that
    the file gives wins over the filament.
    """
    from .config import UserError, parse_filament_block
    try:
        _material, use = parse_filament_block(material, path, _filament_catalog(folder))
    except UserError as exc:
        why = _clean(exc)
        return f"The filament of the file is not used: {why}{'' if why.endswith('.') else '.'}", False
    units = {"E": " MPa", "nu": "", "density": " g/cm3"}
    parts = [f"{k} {_num(v)}{units[k]} ({'your preferences file' if s == 'yaml' else s}{f', {r}' if r else ''})" for k, v, s, r in use.keys]
    return "The run takes " + ", ".join(parts) + ". A key that the file gives wins over the filament.", True


def _missing(value) -> bool:
    """A block that the spec does not have: absent, null or an empty mapping (it gives no value)."""
    return value is None or value == {}


def fill_spec(spec: dict, folder) -> list[str]:
    """Fill a block that the spec does not have at all from the preferences, and return one fact for each block that was filled.

    `spec` is changed in place at its top level; the nested mappings that it gets are new, so the caller's nested data stays as it is.
    The blocks: `material` (E, nu, density, filament), `print`, `settings.voxel_size`. A block that the spec has, also a part of one, is not
    changed: the spec of this request always wins, and a value of a preference is never mixed into it. A file with problems is not
    used, and one fact says so.
    """
    settings = spec.get("settings")
    wants = {"material": _missing(spec.get("material")), "print": _missing(spec.get("print")),
             "voxel_size": (settings is None or (isinstance(settings, dict) and settings.get("voxel_size") is None))}
    if not any(wants.values()):
        return []
    path, values, problems, not_used = _usable(folder)
    if problems:
        return [f"The preferences file {path} has a problem, so build_regions did not use it: {problems[0]}."]
    if values is None:
        return []
    facts = []
    given = values.get("material", {})
    material = {k: given[k] for k in ("E", "nu", "density", "filament") if k in given}
    if wants["material"] and material:
        from .config import Material
        fcat = _filament_catalog(folder)
        filament_words = ""
        if "filament" in material:
            filament_words, usable_filament = _filament_fill(material, path, folder)
            if not usable_filament:
                del material["filament"]
        fact = ""
        if material:
            shown = {k: v for k, v in given.items() if k in material or k == "name"}
            fact = f"material from your preferences file ({path}): {material_text(shown, fcat)}. The spec had no material block."
            rest = [k for k in ("E", "nu", "density") if k not in material]
            if "filament" in material:
                fact += " " + filament_words
            elif rest:
                d = Material()
                fact += (" The file has no " + ", ".join(rest) + ", so the default of the problem YAML applies ("
                         + ", ".join(f"{k} {_num(getattr(d, k))}" for k in rest) + ").")
            spec["material"] = material
        if filament_words and "filament" not in material:
            fact = (fact + " " if fact else f"The preferences file {path} names a filament. ") + filament_words
        facts.append(fact)
    if wants["print"] and values.get("print"):
        spec["print"] = {k: (list(v) if isinstance(v, list) else v) for k, v in values["print"].items()}
        facts.append(f"print settings from your preferences file ({path}): {print_text(values['print'], _catalog(folder))}. The spec had no print block, "
                     "so the print report uses these values and not the assumed ones."
                     + (" The printer model sets the bed and the printer type, unless the print block also gives them; the build height "
                        "of the profile is compared with the part." if "printer_model" in values["print"] else ""))
    if wants["voxel_size"] and values.get("quality", {}).get("voxel_size") is not None:
        voxel = values["quality"]["voxel_size"]
        spec["settings"] = {**(settings or {}), "voxel_size": voxel}
        facts.append(f"voxel_size from your preferences file ({path}): {_num(voxel)} mm. The spec had no settings.voxel_size; "
                     "without a preference, build_regions picks the size from the part (auto).")
    lost_printer = [t for t in not_used if t.startswith("'print.")]
    lost_filament = [t for t in not_used if t.startswith("'material.filament")]
    if wants["print"] and lost_printer:
        facts.append(f"The preferences file {path} names a printer that is not found now, so build_regions did not use it: {lost_printer[0]}")
    if wants["material"] and lost_filament:
        facts.append(f"The preferences file {path} names a filament that is not found now, so build_regions did not use it: {lost_filament[0]}")
    return facts


# --- use: check_problem ------------------------------------------------------------------------------------------------------

def _same(a, b) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)


def facts_for_yaml(folder, yaml_path) -> list[str]:
    """One fact when the printer type, the bed, another print value or the material of a problem YAML differ from the preferences,
    or when the YAML has no print block (so the print report assumes the bed) and the preferences have one. [] when they agree, when
    there is no file, or when the file has problems. It reads the YAML only; it changes nothing. Offer, do not decide."""
    from .config import UserError, parse_filament_block, parse_print_block
    path, values, _ = usable(folder)
    if not values:
        return []
    catalog = _catalog(folder)
    fcat = _filament_catalog(folder)
    try:
        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
        block = data.get("print")
        yaml_print = parse_print_block(block or {}, yaml_path, catalog)
        raw_material = data.get("material")
        yaml_material, yaml_use = parse_filament_block(raw_material or {}, yaml_path, fcat)
    except (OSError, ValueError, yaml.YAMLError, UserError):
        return []
    gaps = []
    wanted = values.get("print") or {}
    if wanted and not block:
        gaps.append(f"the YAML has no print block, so the print report assumes {print_text(_effective_print(yaml_print, tuple(wanted)), catalog)}; "
                    f"the preferences say {print_text(wanted, catalog)}")
    elif wanted:
        for key in PRINT_KEYS:
            if key in wanted and not _print_same(key, wanted[key], yaml_print):
                have = _effective_print(yaml_print, (key,))[key]
                how = "" if key in yaml_print.given else (f" (from the printer profile {yaml_print.printer_model})" if key in yaml_print.derived
                                                          else "" if key == "printer_model" else " (assumed: the YAML does not give it)")
                gaps.append(f"{PRINT_LABELS[key]}: {_print_value_text(key, have, catalog)} in the YAML{how}, "
                            f"{_print_value_text(key, wanted[key], catalog)} in the preferences")
    given = values.get("material") or {}
    if "filament" in given:
        have_id = yaml_use.profile.id if yaml_use is not None else None
        if have_id != given["filament"]:
            gaps.append(f"material filament: {f'{have_id} in the YAML' if have_id else 'none in the YAML'}, "
                        f"{_filament_text(given['filament'], fcat)} in the preferences")
    for key, unit in (("E", " MPa"), ("nu", ""), ("density", " g/cm3")):
        if key in given and not _same(given[key], getattr(yaml_material, key)):
            named = f" ({given['name']})" if "name" in given else ""
            have = f"{_num(getattr(yaml_material, key))}{unit}"
            source = yaml_use.source(key) if yaml_use is not None else None
            if isinstance(raw_material, dict) and raw_material.get(key) is not None:
                have += " in the YAML"
            elif source is not None and source.startswith("filament:"):      # the filament of the YAML gives it
                have += f" from the filament {source.split(':', 1)[1]} of the YAML"
            else:                                           # not a value of the YAML: the default of the loader (R1 n2)
                have = f"the default {have} ({'the YAML does not give it' if raw_material else 'the YAML has no material block'})"
            gaps.append(f"material {key}: {have}, {_num(given[key])}{unit}{named} in the preferences")
    if not gaps:
        return []
    return [f"The preferences file {path} differs from this YAML: " + "; ".join(gaps) + ". The YAML is the record of this run and is not "
            "changed. If the preferences are right for this part, edit the YAML (build_regions fills a block that a spec does not have)."]


def _effective_print(ps, keys) -> dict:
    """The values that the print report uses for `keys`: the YAML value, or the assumed one."""
    out = {}
    for key in keys:
        if key == "printer_model":
            out[key] = ps.printer_model
        elif key == "printer_type":
            out[key] = ps.printer_type
        elif key == "bed_mm":
            out[key] = list(ps.bed_mm)
        elif key == "build_direction":
            out[key] = list(ps.build_direction)
        elif key == "self_supporting_angle_deg":
            out[key] = ps.angle_deg
    return out


def _print_same(key: str, wanted, ps) -> bool:
    """True when the preference `wanted` for `key` gives the value that the print report uses for the YAML."""
    from .config import UserError, parse_print_block
    have = _effective_print(ps, (key,))[key]
    if key in ("printer_model", "printer_type"):
        return wanted == have
    if key == "self_supporting_angle_deg":
        return _same(wanted, have)
    try:
        asked = parse_print_block({key: wanted}, "<preferences>")
    except UserError:
        return False
    asked = _effective_print(asked, (key,))[key]
    return len(asked) == len(have) and all(_same(a, b) for a, b in zip(asked, have))
