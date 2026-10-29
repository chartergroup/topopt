"""The printer catalog and the lookup of a printer by name (CONTRACTS.md section 18.7).

The user names the printer; the tool finds the model in a catalog that ships with topopt and fills the print parameters
itself. The catalog is the package data file `topopt/data/printers.yaml`: our own list of facts from maker pages (a source
URL, the date it was opened and a tag on every row). No slicer profile data is in it (see the header of the file).

- `load_catalog()` reads and checks the file once (cached). A broken file raises `CatalogError` with the place of the fault.
- `resolve(text)` finds a printer: `match` (one printer), `ambiguous` (the printers whose names contain the words of the text),
  or `none` (with "did you mean" suggestions). A suggestion is never taken by itself: the caller asks.
- `PrinterProfile.facts()` is the record that a run keeps (`run.json`, key `print.profile`).
- A printer that is not in the catalog can be a USER profile: `<work folder>/topopt-printers.yaml` has the same entries (the same
  `format: 1`), and `catalog_for(folder)` returns the shipped catalog and that file as one `Catalog`. Every profile says its
  `origin` (`catalog` or `user`). A user entry never replaces a shipped one (no shadowing: not by id, not by name). A fault in the
  user file is a fact (`Catalog.user_file.problems`), never an exception. `save_user_printer` writes the file (the MCP tool
  `save_printer`).

This module does not import `mcp`, and it imports `topopt.config` only inside a check (config imports this module).
"""
from __future__ import annotations

import datetime
import errno
import functools
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path

import yaml

from . import catalogs, workfolder
from .catalogs import (ID_PATTERN, CatalogError, Dumper as _Dumper, FileFault as _FileFault, NameIndex, UserFile,
                       canonical_vendor as _canonical_vendor, date_text as _date_text, fault as _fault, number as _number,
                       out_number as _out, read_user_file, required_number as _required_number, url as _url, words as _words)

FORMAT = 1
FILE = ("data", "printers.yaml")      # the place of the file in the package `topopt`
USER_FILE_NAME = "topopt-printers.yaml"   # in the work folder, beside topopt-preferences.yaml (CONTRACTS.md section 10.2)
ORIGINS = ("catalog", "user")

# The settings of the "did you mean" list (module constants, not YAML keys).
SUGGEST_CUTOFF = 0.6      # a name is suggested if its difflib ratio to the text is at least this
SUGGEST_MAX = 5           # and at most this many names are suggested

ENCLOSURES = ("open", "enclosed", "heated")
EXTRUDERS = ("direct", "bowden")
NOZZLE_MATERIALS = ("brass", "stainless", "hardened_steel", "tungsten_carbide")
TAGS = ("maker doc", "third party")
USER_TAGS = TAGS + ("user",)           # a user entry may also say that the user gave the values
USER_REQUIRED = ("id", "vendor", "model", "printer_type", "bed_mm")   # the rest of a user entry is optional; a value that is not given is null
# The facts of a printer, in the order that a reply shows them. `printer_type` is the first.
FACT_FIELDS = ("printer_type", "bed_mm", "build_height_mm", "nozzle_mm", "nozzle_options_mm", "max_nozzle_c", "max_bed_c",
               "enclosure", "chamber_max_c", "stock_nozzle_material", "extruder")
# The facts that decide what a printer can print; the ambiguous reply names those on which the candidates differ.
GATING_FIELDS = ("bed_mm", "build_height_mm", "nozzle_mm", "max_nozzle_c", "max_bed_c", "enclosure", "chamber_max_c",
                 "stock_nozzle_material", "extruder")
IDENTITY_FIELDS = ("id", "vendor", "model", "aliases")
SOURCE_FIELDS = ("source_url", "opened", "tag")
REQUIRED_FIELDS = IDENTITY_FIELDS + FACT_FIELDS + SOURCE_FIELDS
OPTIONAL_FIELDS = ("field_sources", "note")
USER_ONLY_FIELDS = ("source",)         # `source: user`: the user said the values (a user entry gives this or `source_url`)
NULL_WITHOUT_NOTE = ("chamber_max_c", "nozzle_options_mm")   # null is a plain fact here: no chamber heater stated, no sizes stated
TOP_KEYS = ("format", "catalog_version", "vendors", "models")
WAYS_ON = "Use an id from the catalog, or remove printer_model and give bed_mm and printer_type"
NOT_LISTED = ("If the printer is not listed: look up the page of its maker (a web search), show the user the values with the page address, "
              "and after the user agrees call save_printer with the values and the source_url; a value that you cannot find stays null "
              "(unknown), so ask the user for it: bed size (bed_mm), build height (build_height_mm), max nozzle temperature "
              "(max_nozzle_c), max bed temperature (max_bed_c), enclosure, nozzle material (stock_nozzle_material) and extruder. "
              "Then use the id of the saved printer. Or remove printer_model and give bed_mm and printer_type in the YAML.")
ADD_ON = (f"A printer that is not listed can be added to {USER_FILE_NAME} in the work folder (the MCP tool save_printer writes it)")


# --- the profile -------------------------------------------------------------------------------------------------------------

def _pos(value):
    """A number above 0 as a float, else None (a missing or bad value of a run.json that a person edited)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


@dataclass(frozen=True)
class PrinterProfile:
    """One printer of the catalog. A field that the catalog does not state is None (the `note` says why)."""
    id: str
    vendor: str
    model: str
    printer_type: str = "fdm"
    bed_mm: tuple[float, float] | None = None
    build_height_mm: float | None = None
    nozzle_mm: float | None = None
    nozzle_options_mm: tuple[float, ...] | None = None
    max_nozzle_c: float | None = None
    max_bed_c: float | None = None
    enclosure: str | None = None
    chamber_max_c: float | None = None
    stock_nozzle_material: str | None = None
    extruder: str | None = None
    aliases: tuple[str, ...] = ()
    source_url: str = ""
    opened: str = ""
    tag: str = ""
    field_sources: tuple[tuple[str, str], ...] = ()
    note: str | None = None
    catalog_version: str = ""
    origin: str = "catalog"                 # `catalog` (the file of the package) or `user` (topopt-printers.yaml in the work folder)
    source: str | None = None               # `user` when the user said the values; None otherwise (user entries only)

    @property
    def name(self) -> str:
        return f"{self.vendor} {self.model}".strip()

    @property
    def origin_text(self) -> str:
        """Where the profile comes from, for a line of text: `catalog 2026-10-24`, or `user printer file`."""
        if self.origin == "user":
            return "user printer file"
        return f"catalog {self.catalog_version}".rstrip()

    def facts(self) -> dict:
        """The record of the printer: the name, the capability facts, and where they come from. Lists, not tuples (JSON)."""
        out: dict = {"name": self.name, "vendor": self.vendor, "model": self.model, "origin": self.origin}
        for key in FACT_FIELDS:
            value = getattr(self, key)
            out[key] = list(value) if isinstance(value, tuple) else value
        out["source_url"], out["opened"], out["tag"] = self.source_url, self.opened, self.tag
        if self.source:
            out["source"] = self.source
        if self.field_sources:
            out["field_sources"] = dict(self.field_sources)
        if self.note:
            out["note"] = self.note
        return out

    def facts_sha256(self) -> str:
        """The sha256 (hex) of the normalized facts of the printer: its name and the capability facts (`FACT_FIELDS`) as JSON with sorted keys
        and no blanks. It leaves out the sources, the dates, the note and the origin, so a changed note or page does not change it and a
        changed number does. `run.json` keeps it (`print.profile_sha256`): two runs used the same printer facts if the digests are equal."""
        facts = self.facts()
        body = {"name": facts["name"], **{key: facts[key] for key in FACT_FIELDS}}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        """`facts()` with the id and the catalog version first: the form that `find_printer` returns."""
        return {"id": self.id, "catalog_version": self.catalog_version, **self.facts()}

    @classmethod
    def from_facts(cls, pid: str, catalog_version, facts) -> "PrinterProfile | None":
        """The profile that a run recorded (`run.json`, key `print`). A bad value reads as not stated; None if `facts` is no mapping.

        The catalog is not read: the record of the run is the truth for that run, also when the catalog changed since.
        """
        if not isinstance(pid, str) or not pid.strip() or not isinstance(facts, dict):
            return None

        def text(key):
            value = facts.get(key)
            return value if isinstance(value, str) and value.strip() else None

        bed = None
        raw_bed = facts.get("bed_mm")
        if isinstance(raw_bed, (list, tuple)) and len(raw_bed) == 2:
            x, y = _pos(raw_bed[0]), _pos(raw_bed[1])
            if x is not None and y is not None:
                bed = (x, y)
        options = None
        raw_options = facts.get("nozzle_options_mm")
        if isinstance(raw_options, (list, tuple)) and raw_options:
            numbers = [_pos(v) for v in raw_options]
            if all(v is not None for v in numbers):
                options = tuple(v for v in numbers if v is not None)
        sources = facts.get("field_sources")
        sources = tuple((str(k), str(v)) for k, v in sources.items()) if isinstance(sources, dict) else ()
        enclosure, extruder, material = text("enclosure"), text("extruder"), text("stock_nozzle_material")
        origin = text("origin")
        return cls(
            id=pid.strip(), vendor=text("vendor") or "", model=text("model") or pid.strip(),
            printer_type=text("printer_type") or "fdm", bed_mm=bed, build_height_mm=_pos(facts.get("build_height_mm")),
            nozzle_mm=_pos(facts.get("nozzle_mm")), nozzle_options_mm=options, max_nozzle_c=_pos(facts.get("max_nozzle_c")),
            max_bed_c=_pos(facts.get("max_bed_c")), enclosure=enclosure if enclosure in ENCLOSURES else None,
            chamber_max_c=_pos(facts.get("chamber_max_c")), stock_nozzle_material=material,
            extruder=extruder if extruder in EXTRUDERS else None, source_url=text("source_url") or "",
            opened=text("opened") or "", tag=text("tag") or "", field_sources=sources, note=text("note"),
            catalog_version=catalog_version if isinstance(catalog_version, str) else "",
            origin=origin if origin in ORIGINS else "catalog", source=text("source"))


def _n(x) -> str:
    return f"{x:g}"


def facts_text(p: PrinterProfile) -> str:
    """The numbers that the tool takes from the profile, as text: `bed 256 x 256 mm, build height 256 mm, fdm`."""
    parts = [f"bed {_n(p.bed_mm[0])} x {_n(p.bed_mm[1])} mm" if p.bed_mm else "bed not stated"]
    parts.append(f"build height {_n(p.build_height_mm)} mm" if p.build_height_mm is not None else "build height not stated")
    parts.append(p.printer_type)
    return ", ".join(parts)


def summary_text(p: PrinterProfile) -> str:
    """One line for a person: the name, the id, and the numbers that the tool takes from the profile."""
    return f"{p.name} (profile {p.id}): {facts_text(p)}"


# --- the file ----------------------------------------------------------------------------------------------------------------

def _entry(raw, label: str, vendors: dict, version: str, user: bool = False) -> PrinterProfile:
    """Check one entry of a file and return the profile. Raise `CatalogError`; the text starts with `label`.

    The shipped catalog is strict: every field is there (a fact that is not stated is null, and a null needs a note). A user entry
    needs `id`, `vendor`, `model`, `printer_type`, `bed_mm` and a source (`source_url` or `source: user`); a field that is not given is
    null, which means unknown.
    """
    from .config import PRINTER_TYPES      # lazy: config imports this module

    where = label + (f" ({raw['id']})" if isinstance(raw, dict) and isinstance(raw.get("id"), str) else "")
    if not isinstance(raw, dict):
        raise _fault(where, f"a model must be a mapping, got {type(raw).__name__}")
    allowed = REQUIRED_FIELDS + OPTIONAL_FIELDS + (USER_ONLY_FIELDS if user else ())
    unknown = [k for k in raw if k not in allowed]
    if unknown:
        raise _fault(where, f"unknown key {unknown[0]!r}")
    missing = [k for k in (USER_REQUIRED if user else REQUIRED_FIELDS) if k not in raw]
    if missing:
        raise _fault(where, f"missing key {missing[0]!r}" + ("" if user else " (a fact that is not stated is null, with a note)"))
    pid = raw["id"]
    if not isinstance(pid, str) or not ID_PATTERN.match(pid):
        raise _fault(where, f"id must be a lower-case slug (letters, digits, '-'), got {pid!r}")
    for key in ("vendor", "model"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise _fault(where, f"{key} must be a non-empty text")
    if user and not _words(raw["vendor"]):
        raise _fault(where, "vendor must have a letter or a digit")
    if not user and raw["vendor"] not in vendors:
        raise _fault(where, f"vendor {raw['vendor']!r} is not in the `vendors` table")
    aliases = raw.get("aliases") if "aliases" in raw or not user else []
    if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
        raise _fault(where, "aliases must be a list of non-empty texts")
    if raw["printer_type"] not in PRINTER_TYPES:
        raise _fault(where, f"printer_type must be one of {', '.join(PRINTER_TYPES)}, got {raw['printer_type']!r}")
    bed = None
    if raw.get("bed_mm") is not None or user:
        raw_bed = raw["bed_mm"]
        if not isinstance(raw_bed, list) or len(raw_bed) != 2:
            raise _fault(where, f"bed_mm must be [x, y]{'' if user else ' or null'}, got {raw_bed!r}")
        bed = (_required_number(where, "bed_mm", raw_bed[0]), _required_number(where, "bed_mm", raw_bed[1]))
    options = None
    raw_options = raw.get("nozzle_options_mm")
    if raw_options is not None:
        if not isinstance(raw_options, list) or not raw_options:
            raise _fault(where, f"nozzle_options_mm must be a list of numbers or null, got {raw_options!r}")
        options = tuple(_required_number(where, "nozzle_options_mm", v) for v in raw_options)
    for key, allowed_values in (("enclosure", ENCLOSURES), ("extruder", EXTRUDERS), ("stock_nozzle_material", NOZZLE_MATERIALS)):
        if raw.get(key) is not None and raw[key] not in allowed_values:
            raise _fault(where, f"{key} must be one of {', '.join(allowed_values)} or null, got {raw[key]!r}")
    chamber = _number(where, "chamber_max_c", raw.get("chamber_max_c"))
    if user:
        if chamber is not None and raw.get("enclosure") != "heated":
            raise _fault(where, "chamber_max_c needs enclosure: heated (a chamber heater)")
    elif (raw["enclosure"] == "heated") != (chamber is not None):
        raise _fault(where, "chamber_max_c is a number exactly when enclosure is heated (the maker states a chamber heater)")

    source = raw.get("source")
    url = raw.get("source_url")
    if user:
        if source is not None and source != "user":
            raise _fault(where, f"source must be the word user (the user said the values) or left out, got {source!r}")
        if url is None and source is None:
            raise _fault(where, "give source_url (the maker page that has the values) or source: user (the user said them)")
        url = _url(where, "source_url", url) if url is not None else ""
        opened = _date_text(where, raw.get("opened"), allow_none=True)
        tag = raw.get("tag")
        if tag is None:
            tag = "user" if not url else ""
        elif tag not in USER_TAGS:
            raise _fault(where, f"tag must be one of {', '.join(USER_TAGS)}, got {tag!r}")
    else:
        url = _url(where, "source_url", url)
        opened = _date_text(where, raw["opened"])
        tag = raw["tag"]
        if tag not in TAGS:
            raise _fault(where, f"tag must be one of {', '.join(TAGS)}, got {tag!r}")
    sources = raw.get("field_sources") or {}
    if not isinstance(sources, dict):
        raise _fault(where, "field_sources must be a mapping of a field to an address")
    for key, value in sources.items():
        if key not in FACT_FIELDS:
            raise _fault(where, f"field_sources names {key!r}, which is not a fact field")
        _url(where, f"field_sources.{key}", value, allow_user=user)
    note = raw.get("note")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise _fault(where, "note must be a non-empty text")
    if not user:
        for key in FACT_FIELDS:
            if raw[key] is None and key not in NULL_WITHOUT_NOTE and not note:
                raise _fault(where, f"{key} is null, so the model needs a note that says why")
    return PrinterProfile(
        id=pid, vendor=raw["vendor"].strip(), model=raw["model"].strip(), aliases=tuple(a.strip() for a in aliases),
        printer_type=raw["printer_type"], bed_mm=bed, build_height_mm=_number(where, "build_height_mm", raw.get("build_height_mm")),
        nozzle_mm=_number(where, "nozzle_mm", raw.get("nozzle_mm")), nozzle_options_mm=options,
        max_nozzle_c=_number(where, "max_nozzle_c", raw.get("max_nozzle_c")),
        max_bed_c=_number(where, "max_bed_c", raw.get("max_bed_c")),
        enclosure=raw.get("enclosure"), chamber_max_c=chamber, stock_nozzle_material=raw.get("stock_nozzle_material"),
        extruder=raw.get("extruder"), source_url=url, opened=opened, tag=tag,
        field_sources=tuple((k, str(v)) for k, v in sources.items()), note=note.strip() if note else None,
        catalog_version=version, origin="user" if user else "catalog", source=source)


class Catalog(NameIndex):
    """The checked catalog and its index of names. Build it with `parse_catalog` or `load_catalog`; `catalog_for` adds the user file.

    `version` is the version of the shipped catalog. `profiles` holds the shipped printers first, then the user printers.
    `vendors` is the vendor table (the shipped table, plus the words of the user file). `base_vendors` is the shipped table: a
    shipped printer is indexed with its OWN vendor words from that table, so a user file changes no shipped name. A user printer is
    indexed with its own vendor words from `vendors`. The words of ALL vendors still find the vendor in a text (`strip_vendors`).
    The index of the names is `topopt.catalogs.NameIndex`, the code that the filament catalog uses too.
    """

    def __init__(self, version: str, profiles: tuple[PrinterProfile, ...], vendors: dict[str, tuple[str, ...]],
                 user_file: UserFile | None = None, base_vendors: dict[str, tuple[str, ...]] | None = None):
        self.version = version
        self.user_file = user_file                              # None: the user file was not looked for
        super().__init__(profiles, vendors, base_vendors, label="printer catalog")


def parse_catalog(data, source: str = "printers.yaml") -> Catalog:
    """Check a loaded mapping and return the `Catalog`. Raise `CatalogError` at the first fault."""
    if not isinstance(data, dict):
        raise CatalogError(f"printer catalog {source}: the file must hold a mapping of keys ({', '.join(TOP_KEYS)})")
    unknown = [k for k in data if k not in TOP_KEYS]
    if unknown:
        raise CatalogError(f"printer catalog {source}: unknown key {unknown[0]!r}")
    if type(data.get("format")) is not int or data["format"] != FORMAT:
        raise CatalogError(f"printer catalog {source}: format is {data.get('format')!r}, and this code reads format {FORMAT}")
    version = data.get("catalog_version")
    if not isinstance(version, str) or not version.strip():
        raise CatalogError(f"printer catalog {source}: catalog_version must be a text, for example \"2026-10-24\"")
    raw_vendors = data.get("vendors")
    if not isinstance(raw_vendors, dict) or not raw_vendors:
        raise CatalogError(f"printer catalog {source}: vendors must be a mapping of a vendor name to a list of vendor words")
    vendors: dict[str, tuple[str, ...]] = {}
    for name, words in raw_vendors.items():
        if not isinstance(name, str) or not isinstance(words, list) or not all(isinstance(w, str) and _words(w) for w in words):
            raise CatalogError(f"printer catalog {source}: vendors.{name}: must be a list of vendor words")
        vendors[name] = tuple(words)
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise CatalogError(f"printer catalog {source}: models must be a non-empty list")
    profiles = tuple(_entry(raw, f"printer catalog, models[{i}]", vendors, version.strip()) for i, raw in enumerate(models))
    ids = [p.id for p in profiles]
    for pid in ids:
        if ids.count(pid) > 1:
            raise CatalogError(f"printer catalog {source}: the id {pid!r} is used twice")
    return Catalog(version.strip(), profiles, vendors)


def read_catalog_text() -> tuple[str, str]:
    """`(text, source)` of the file in the installed package. Raise `CatalogError` if it cannot be read or is empty."""
    place = resources.files("topopt").joinpath(*FILE)
    source = "topopt/" + "/".join(FILE)
    try:
        text = place.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"printer catalog: cannot read {source} ({exc})") from exc
    if not text.strip():
        raise CatalogError(f"printer catalog: {source} is empty")
    return text, source


def load_catalog(path=None) -> Catalog:
    """The catalog. With no `path`: the file of the package, read and checked once (cached). With a `path`: that file, not cached."""
    if path is None:
        return _default_catalog()
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"printer catalog: cannot read {p} ({exc})") from exc
    return _parse_text(text, str(p))


def _parse_text(text: str, source: str) -> Catalog:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CatalogError(f"printer catalog {source}: not valid YAML: {' '.join(str(exc).split())}") from exc
    return parse_catalog(data, source)


@functools.lru_cache(maxsize=1)
def _default_catalog() -> Catalog:
    text, source = read_catalog_text()
    return _parse_text(text, source)


# --- the user printer file ---------------------------------------------------------------------------------------------------

USER_HEADER = (
    f"# topopt user printers (format {FORMAT}): printers that are not in the catalog that ships with topopt.\n"
    "# The entries are the same as in the shipped catalog. A value that is not known is null (unknown), never a guess.\n"
    "# Every entry names its source: source_url (the maker page that has the values) or source: user (the user said them).\n"
    "# The server reads this file at each call. The MCP tool save_printer adds or replaces an entry, and you can edit the file by hand.\n"
    "# A comment that you add here is not kept when save_printer writes the file.\n")


def user_file_path(folder) -> Path:
    return Path(folder) / USER_FILE_NAME


def work_folder() -> Path:
    """The work folder by the one rule of `topopt.workfolder` (CONTRACTS.md section 10.1): the MCP server, the checkpoint and the CLI
    all take it from there. The CLI has no option of its own for the folder, so `topopt run` and `topopt check` read the user printer
    file of the folder that the rule gives (TOPOPT_WORKSPACE, else TOPOPT_DEFAULT_WORKSPACE, else the current directory)."""
    return workfolder.resolve(check=False).path


def work_catalog() -> Catalog:
    """The shipped catalog with the user printers of the work folder."""
    return catalog_for(work_folder())


def _read_user_file(path: Path):
    """`(state, mapping, problems)`. `state` is `absent`, `unreadable` (the system does not let the server read it), `broken` or `ok`."""
    return read_user_file(path, TOP_KEYS)


def _merge(base: Catalog, path: Path, data: dict) -> Catalog:
    """The shipped catalog and the entries of the user file `data`. A fault of the whole file raises `_FileFault`. An entry with a
    fault is left out and named in `user_file.problems`."""
    unknown = [k for k in data if k not in TOP_KEYS]
    if unknown:
        raise _FileFault(f"unknown key {unknown[0]!r} (the keys are {', '.join(TOP_KEYS)})")
    if "format" in data and (type(data["format"]) is not int or data["format"] != FORMAT):
        raise _FileFault(f"format is {data['format']!r}, and this code reads format {FORMAT}")
    version = data.get("catalog_version")
    if isinstance(version, datetime.date):
        version = version.isoformat()
    if version is not None and not isinstance(version, str):
        raise _FileFault("catalog_version must be a text")
    version = (version or "").strip()
    vendors = dict(base.vendors)
    raw_vendors = data.get("vendors") or {}
    if not isinstance(raw_vendors, dict):
        raise _FileFault("vendors must be a mapping of a vendor name to a list of vendor words")
    for name, words in raw_vendors.items():
        if not isinstance(name, str) or not _words(name) or not isinstance(words, list) \
                or not all(isinstance(w, str) and _words(w) for w in words):
            raise _FileFault(f"vendors.{name}: must be a list of vendor words")
        vendors[name] = (*vendors.get(name, ()), *words)
    models = data.get("models") or []
    if not isinstance(models, list):
        raise _FileFault("models must be a list of printers")

    accepted: list[PrinterProfile] = []
    problems: list[str] = []
    for i, raw in enumerate(models):
        label = f"{USER_FILE_NAME}, models[{i}]"
        try:
            profile = _entry(raw, label, vendors, version, user=True)
        except CatalogError as exc:
            problems.append(f"{exc}. This entry is not used.")
            continue
        where = f"{label} ({profile.id})"
        shipped = base.by_id.get(profile.id)
        if shipped is not None:
            problems.append(f"{where}: the id is the id of the shipped printer {shipped.name}. A user printer never replaces a shipped one, "
                            f"so this entry is not used. Use another id, for example {profile.id}-user.")
            continue
        if any(a.id == profile.id for a in accepted):
            problems.append(f"{where}: the id is used twice in this file. The first entry is used and this one is not.")
            continue
        profile = replace(profile, vendor=_canonical_vendor(vendors, profile.vendor))
        trial = dict(vendors)
        trial.setdefault(profile.vendor, ())
        try:
            Catalog(base.version, base.profiles + tuple(accepted) + (profile,), trial, base_vendors=base.vendors)
        except CatalogError as exc:
            problems.append(f"{where}: {_fault_text(exc)}. The name of a printer must differ from the names of the other printers of "
                            "its vendor, so this entry is not used. Give it another model name or alias.")
            continue
        vendors = trial
        accepted.append(profile)
    try:
        return Catalog(base.version, base.profiles + tuple(accepted), vendors, UserFile(str(path), True, len(accepted), tuple(problems)),
                       base_vendors=base.vendors)
    except CatalogError as exc:                         # the names cannot be indexed together: a fact about the file, not an exception
        raise _FileFault(f"the vendors and the printers of the file cannot be used together ({_fault_text(exc)})") from exc


def _fault_text(exc: Exception) -> str:
    return str(exc).removeprefix("printer catalog: ").removeprefix("printer catalog, ")


def catalog_for(folder) -> Catalog:
    """The shipped catalog with the user printers of `<folder>/topopt-printers.yaml`. `folder` None: the shipped catalog only.

    The file is read at each call (it is small, and `save_printer` changes it). A file that is not there is no problem. A file that
    cannot be used, or an entry with a fault, is a fact in `Catalog.user_file.problems`: the rest still works, nothing is raised.
    A user printer never replaces a shipped one. Raises `CatalogError` only if the shipped catalog is broken.
    """
    base = load_catalog()
    if folder is None:
        return base
    path = user_file_path(folder)
    state, data, found = _read_user_file(path)
    if state == "absent":
        return Catalog(base.version, base.profiles, base.vendors, UserFile(str(path), False))
    if state == "ok" and data is not None:
        try:
            return _merge(base, path, data)
        except _FileFault as exc:
            found = [str(exc)]
    problems = tuple(f"{USER_FILE_NAME}: {text}. The file is not used." for text in found)
    return Catalog(base.version, base.profiles, base.vendors, UserFile(str(path), True, 0, problems))


def entry_dict(p: PrinterProfile) -> dict:
    """The entry of a user profile as it is written in the file: every fact field is there, null when it is not known."""
    out: dict = {"id": p.id, "vendor": p.vendor, "model": p.model, "aliases": list(p.aliases)}
    if p.source_url:
        out["source_url"] = p.source_url
    if p.source:
        out["source"] = p.source
    if p.opened:
        out["opened"] = p.opened
    if p.tag:
        out["tag"] = p.tag
    for key in FACT_FIELDS:
        value = getattr(p, key)
        out[key] = [_out(v) for v in value] if isinstance(value, tuple) else _out(value)
    if p.field_sources:
        out["field_sources"] = dict(p.field_sources)
    if p.note:
        out["note"] = p.note
    return out


def _refuse(text: str) -> dict:
    return {"ok": False, "error": text}


def save_user_printer(folder, raw) -> dict:
    """Add or replace one printer in `<folder>/topopt-printers.yaml` (the MCP tool `save_printer`).

    `raw` has the fields of an entry (`USER_REQUIRED` and a source are needed). The entry is checked as the loader checks it. An entry
    with the same id in the file is replaced; a shipped id, or a name of another printer of the vendor, is refused. The file is written
    atomically; the other entries and keys stay as they are. A file that cannot be read or is not valid is not changed.
    Returns `{ok, path, profile, created, replaced, note}` (+ `problems` of the rest of the file), or `{ok: False, error}`.
    """
    from . import runinfo      # lazy: runinfo reads the work folder, and this module is read by config

    path = user_file_path(folder)
    if not isinstance(raw, dict):
        return _refuse(f"save_printer wrote nothing. profile must be a mapping with the fields of one printer ({', '.join(USER_REQUIRED)} "
                       "and a source), got " + type(raw).__name__ + ".")
    state, old, found = _read_user_file(path)
    if state == "unreadable":
        return _refuse(f"save_printer did not change {path}: {found[0]}. It does not write over a file that it cannot read.")
    if state == "broken":
        return _refuse(f"save_printer did not change {path}: {found[0]}. Fix the file by hand, or remove it.")
    old = old or {}
    try:
        base = load_catalog()
    except CatalogError as exc:
        return _refuse(f"save_printer wrote nothing. The shipped catalog cannot be read, so the name cannot be checked: {exc}.")
    try:
        profile = _entry(raw, "profile", base.vendors, "", user=True)
    except CatalogError as exc:
        return _refuse(f"save_printer wrote nothing. {exc}.")
    shipped = base.by_id.get(profile.id)
    if shipped is not None:
        return _refuse(f"save_printer wrote nothing. profile ({profile.id}): the id is the id of the shipped printer {shipped.name}. "
                       f"A user printer never replaces a shipped one. Use another id, for example {profile.id}-user. If the values of "
                       "the shipped printer are wrong for the machine of the user, save the machine with its own id and model name.")
    models = old.get("models") or []
    others = {**old, "models": [m for m in models if not (isinstance(m, dict) and m.get("id") == profile.id)]} \
        if isinstance(models, list) else old
    try:
        merged = _merge(base, path, others)
    except _FileFault as exc:
        return _refuse(f"save_printer did not change {path}: {exc}. Fix the file by hand, or remove it.")
    profile = replace(profile, vendor=_canonical_vendor(merged.vendors, profile.vendor))
    vendors = dict(merged.vendors)
    vendors.setdefault(profile.vendor, ())
    try:
        Catalog(base.version, merged.profiles + (profile,), vendors, base_vendors=base.vendors)
    except CatalogError as exc:
        return _refuse(f"save_printer wrote nothing. profile ({profile.id}): {_fault_text(exc)}. The name of a printer must differ from "
                       "the names of the other printers of its vendor. Give it another model name or alias.")

    entry = entry_dict(profile)
    replaced = False
    listed = []
    for m in (models if isinstance(models, list) else []):
        if isinstance(m, dict) and m.get("id") == profile.id:
            if not replaced:                                   # the first entry of the id is replaced; a later entry of the same id goes
                listed.append(entry)
            replaced = True
        else:
            listed.append(m)
    if not replaced:
        listed.append(entry)
    doc: dict = {"format": FORMAT}
    if old.get("catalog_version") is not None:
        doc["catalog_version"] = old["catalog_version"]
    if old.get("vendors") is not None:
        doc["vendors"] = old["vendors"]
    doc["models"] = listed
    made = state == "absent"
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        text = USER_HEADER + yaml.dump(doc, Dumper=_Dumper, sort_keys=False, default_flow_style=False, allow_unicode=True)
        runinfo.write_atomic(path, text.encode("utf-8"))
    except OSError as exc:
        text = f"save_printer could not write {path}: {exc.strerror or exc}."
        if exc.errno == errno.EPERM and sys.platform == "darwin":
            text += f" {workfolder.MAC_PRIVACY}."
        return _refuse(text)

    after = catalog_for(folder)
    saved = after.by_id.get(profile.id, profile)
    unknown = [k for k in FACT_FIELDS if getattr(saved, k) is None]
    note = (f"Saved {saved.name} as the user printer {saved.id} ({'replaced' if replaced else 'new entry'}"
            f"{', new file' if made else ''}). The server reads the file at each call, so the printer is live now: "
            f"print.printer_model: {saved.id} in a YAML sets the bed and the printer type from it, and save_preferences keeps "
            "printer_model for the next runs.")
    if replaced:
        note += (f" A YAML that names {saved.id} gets the new values at its next check or run (the YAML text did not change, so nothing "
                 "else notes it); old runs keep their record in run.json, and print.profile_sha256 there tells whether two runs used the "
                 "same printer facts.")
    if unknown:
        note += f" Not known (null): {', '.join(unknown)}. Ask the user for a value that matters, then save the printer again."
    reply = {"ok": True, "path": str(path), "profile": saved.summary(), "created": not replaced, "replaced": replaced, "note": note}
    if after.user_file is not None and after.user_file.problems:
        reply["problems"] = list(after.user_file.problems)
    return reply


# --- the lookup --------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Resolution:
    """The answer of `resolve`: `status` is `match`, `ambiguous` or `none`."""
    status: str
    query: str
    catalog_version: str
    profile: PrinterProfile | None = None                     # match
    matched_by: str | None = None                             # match: id | name | alias. ambiguous: partial | name | vendor
    candidates: tuple[PrinterProfile, ...] = ()               # ambiguous
    differs: tuple[str, ...] = ()                             # ambiguous: the gating facts on which the candidates differ
    suggestions: tuple[tuple[PrinterProfile, float], ...] = ()    # none: "did you mean", best first. Never taken by itself
    vendors: tuple[str, ...] = ()                             # none: the vendors of the catalog
    note: str = ""
    user_file: UserFile | None = None                         # the user printer file of the work folder (None: not looked for)

    def to_dict(self) -> dict:
        """The reply of the MCP tool `find_printer` (the same shape for the three statuses)."""
        return {
            "ok": True, "status": self.status, "query": self.query, "catalog_version": self.catalog_version,
            "matched_by": self.matched_by,
            "origin": self.profile.origin if self.profile else None,
            "profile": self.profile.summary() if self.profile else None,
            "derives": _derives(self.profile) if self.profile else None,
            "candidates": [p.summary() for p in self.candidates],
            "differs": list(self.differs),
            "suggestions": [{"id": p.id, "name": p.name, "score": round(score, 3)} for p, score in self.suggestions],
            "vendors": list(self.vendors),
            "user_file": self.user_file.facts() if self.user_file is not None else None,
            "note": self.note,
        }


def _derives(p: PrinterProfile) -> dict:
    """What `print.printer_model: <id>` sets in a run (a key that the YAML gives wins), and what the profile cannot give."""
    sets: dict[str, object] = {"print.printer_type": p.printer_type}
    missing: list[str] = []
    if p.bed_mm is not None:
        sets["print.bed_mm"] = list(p.bed_mm)
    else:
        missing.append("print.bed_mm")
    if p.build_height_mm is not None:
        sets["build height for the bed-fit check"] = p.build_height_mm
    else:
        missing.append("build height")
    return {"sets": sets, "missing": missing}


def _differing(candidates) -> tuple[str, ...]:
    return tuple(f for f in GATING_FIELDS if len({repr(getattr(p, f)) for p in candidates}) > 1)


def resolve(text, catalog: Catalog | None = None) -> Resolution:
    """Find the printer that `text` names. `catalog` (default: the shipped catalog) may hold user printers (`catalog_for`).

    - `match`: the text is the id, or the name or an alias of exactly one printer (case, punctuation, vendor words such as
      `bambu` or `original prusa`, and the words `my` and `printer` do not matter). A vendor word in the text is a hard filter. If that
      gives no match, the text is read again for each printer with only the words of its own vendor taken away (a vendor named `Plus`
      does not hide `MINI+`).
    - `ambiguous`: the text is not an exact name, or it is the exact name of printers of two vendors; `candidates` are the printers
      whose names contain all the words of the text (a text with only a vendor word gives all printers of the vendor). The caller
      asks the user; `differs` names the facts on which the candidates differ.
    - `none`: no printer fits. `suggestions` are the names close to the text by difflib; none is taken by itself.

    The lookup of a name is `topopt.catalogs.find`; the words of the answer are written here.
    """
    cat = catalog or load_catalog()
    query = text if isinstance(text, str) else ""
    shown = " ".join(query.split())

    def answer(status: str, **rest) -> Resolution:
        return Resolution(status, shown, cat.version, user_file=cat.user_file, **rest)

    if not _words(query):
        return answer("none", vendors=tuple(cat.vendors),
                      note="The name is empty. Ask the user which printer they have (vendor and model).")

    by_id = cat.by_id.get(query.strip().lower())
    if by_id is not None:
        return answer("match", profile=by_id, matched_by="id", note=_match_note(by_id, "id"))

    hit = catalogs.find(cat, query)
    if hit is not None and hit.status == "match":
        return answer("match", profile=hit.profile, matched_by=hit.matched_by, note=_match_note(hit.profile, hit.matched_by))
    if hit is not None:
        return _ambiguous(answer, list(hit.candidates), hit.matched_by,
                          {"name": "The name is the name of printers of more than one vendor.",
                           "vendor": "The text names only the vendor."}.get(hit.matched_by))
    suggestions = catalogs.suggest_names(cat, query, SUGGEST_CUTOFF, SUGGEST_MAX)
    where = "the catalog or the user printer file" if cat.user_file is not None else "the catalog"
    note = (f"No printer of {where} has this name. "
            + ("Ask the user if they mean one of the suggestions (a suggestion is never taken by itself). " if suggestions else "")
            + NOT_LISTED)
    if cat.user_file is not None and cat.user_file.problems:
        note += " The user printer file has problems: " + " ".join(cat.user_file.problems)
    return answer("none", suggestions=suggestions, vendors=tuple(cat.vendors), note=note)


def _ambiguous(answer, candidates: list[PrinterProfile], matched_by: str, reason: str | None) -> Resolution:
    """`matched_by` says why the answer is not a match: `partial` (the text is a part of a name), `name` (the exact name of printers of
    more than one vendor) or `vendor` (the text is only a vendor word)."""
    differs = _differing(candidates) if len(candidates) > 1 else ()
    if len(candidates) == 1:
        p = candidates[0]
        note = (f"The text is only part of the name of one printer in the catalog: {p.name} (id {p.id}). A part of a name is not the "
                "name: a similar model can have other numbers. Ask the user if this is their printer. If yes, use the id.")
    else:
        note = ((reason + " ") if reason else "") \
            + (f"The text is part of the names of {len(candidates)} printers. " if matched_by == "partial" else f"The text fits {len(candidates)} printers. ") \
            + "Ask the user which one" + (f"; they differ in {', '.join(differs)}" if differs else "") + ". Then use the id."
    return answer("ambiguous", matched_by=matched_by, candidates=tuple(candidates), differs=differs, note=note)


def _match_note(p: PrinterProfile, how: str) -> str:
    lack = _derives(p)["missing"]
    note = (f"Matched {p.name} by {how} ({p.origin_text}). print.printer_model: {p.id} sets print.printer_type and "
            "print.bed_mm when the YAML does not give them, and the build height for the bed-fit check.")
    if lack:
        note += f" The catalog does not state {', '.join(lack)} for this printer: ask the user for it if it matters. " \
                "See the note of the profile."
    return note


def profile_for(printer_id, catalog: Catalog | None = None) -> PrinterProfile | None:
    """The profile with this id, or None."""
    return (catalog or load_catalog()).by_id.get(printer_id) if isinstance(printer_id, str) else None


def refusal_text(res: Resolution, key: str = "print.printer_model") -> str:
    """One line for a `UserError` when `res` is not a match: the key, the text, the candidates or suggestions, and the ways on."""
    head = f"'{key}' {res.query!r}"

    def listed(profiles) -> str:
        return ", ".join(f"{p.id} ({p.name}{', user printer' if p.origin == 'user' else ''})" for p in profiles)

    if res.status == "ambiguous":
        n = len(res.candidates)
        where = "the catalog or the user printer file" if any(p.origin == "user" for p in res.candidates) else "the catalog"
        text = f"{head} fits {n} printer{'' if n == 1 else 's'} of {where} (version {res.catalog_version}): {listed(res.candidates)}. {WAYS_ON}"
    elif res.suggestions:
        text = (f"{head} is not in the printer catalog (version {res.catalog_version}). "
                f"Did you mean: {listed(p for p, _ in res.suggestions)}? {WAYS_ON}. {ADD_ON}")
    else:
        text = (f"{head} is not in the printer catalog (version {res.catalog_version}) and no name is close to it. "
                f"{WAYS_ON}. {ADD_ON}")
    uf = res.user_file
    if uf is not None and res.status != "ambiguous":
        text += f" (user printer file {uf.path}: " + (f"{uf.entries} printers" if uf.exists else "not there") + ")"
    if uf is not None and uf.problems:
        text += " The user printer file has problems: " + " ".join(uf.problems)
    return text
