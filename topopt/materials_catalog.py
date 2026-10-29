"""The filament catalog and the lookup of a filament by name (CONTRACTS.md section 18.8).

The user names the filament; the tool finds the product in a catalog that ships with topopt and fills the material numbers itself. The
catalog is the package data file `topopt/data/materials.yaml`: our own list of facts from maker data sheets (a source URL, the date it was
opened and a tag on every row; no price; no slicer profile data, see the header of the file). One row is one product and one sheet version.

- `load_catalog()` reads and checks the file once (cached). A broken file raises `CatalogError` with the place of the fault.
- `resolve(text)` finds a filament: `match` (one filament), `ambiguous` (the filaments whose names contain the words of the text) or `none`
  (with "did you mean" suggestions). A suggestion is never taken by itself: the caller asks. The name lookup is `topopt.catalogs`, the code
  that `topopt.printers` uses too.
- `FilamentProfile.facts()` is the record that a run keeps (`run.json`, key `material.filament`).
- A filament that is not in the catalog can be a USER profile: `<work folder>/topopt-filaments.yaml` has the same entries (the same
  `format: 1`), and `catalog_for(folder)` returns the shipped catalog and that file as one `Catalog`. Every profile says its `origin`
  (`catalog` or `user`). A user entry never replaces a shipped one (no shadowing: not by id, not by name). A fault in the user file is a fact
  (`Catalog.user_file.problems`), never an exception. `save_user_filament` writes the file (the MCP tool `save_material`).

This module does not import `mcp`, and it imports `topopt.config` only inside a check (config imports this module).
"""
from __future__ import annotations

import datetime
import errno
import functools
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path

import yaml

from . import catalogs, printers, workfolder
from .catalogs import (ID_PATTERN, CatalogError, Dumper as _Dumper, FileFault as _FileFault, NameIndex, UserFile,
                       canonical_vendor as _canonical_maker, date_text as _date_text, fault as _fault, number as _number,
                       out_number as _out, read_user_file, url as _url, words as _words)

FORMAT = 1
FILE = ("data", "materials.yaml")             # the place of the file in the package `topopt`
USER_FILE_NAME = "topopt-filaments.yaml"      # in the work folder, beside topopt-preferences.yaml and topopt-printers.yaml
ORIGINS = ("catalog", "user")

STRENGTH_KINDS = ("yield", "peak", "not stated")
ENCLOSURES = ("no", "recommended", "required")
TAGS = ("maker TDS", "maker page", "third party")
USER_TAGS = TAGS + ("user",)                  # a user entry may also say that the user gave the values
USER_REQUIRED = ("id", "maker", "product", "family")      # the rest of a user entry is optional; a value that is not given is null
NUMBER_FIELDS = ("E_xy_mpa", "E_z_mpa", "bending_modulus_xy_mpa", "bending_modulus_z_mpa", "strength_xy_mpa", "strength_z_mpa",
                 "elongation_xy_pct", "density_g_cm3", "hdt_18_c", "hdt_045_c", "chamber_c_min")
RANGE_FIELDS = ("nozzle_c", "bed_c", "density_range_g_cm3")      # the last one is optional (`OPTIONAL_FIELDS`)
FLAG_FIELDS = ("hardened_nozzle", "direct_drive", "abrasive", "soft")
CONDITIONED_KEYS = ("condition", "strength_xy_mpa", "strength_z_mpa", "E_xy_mpa", "E_z_mpa")
# The facts of a filament, in the order that a reply shows them.
FACT_FIELDS = ("E_xy_mpa", "E_z_mpa", "bending_modulus_xy_mpa", "bending_modulus_z_mpa", "strength_xy_mpa", "strength_z_mpa",
               "strength_kind", "elongation_xy_pct", "density_g_cm3", "hdt_18_c", "hdt_045_c", "conditioned", "nu", "nozzle_c", "bed_c",
               "enclosure", "chamber_c_min", "hardened_nozzle", "drying", "direct_drive", "abrasive", "soft")
# The facts on which the candidates of an ambiguous answer are compared.
CHOICE_FIELDS = ("E_xy_mpa", "E_z_mpa", "strength_xy_mpa", "strength_z_mpa", "density_g_cm3", "enclosure", "hardened_nozzle")
IDENTITY_FIELDS = ("id", "family", "maker", "product", "aliases", "sheet_version")
SOURCE_FIELDS = ("source_url", "opened", "tag")
REQUIRED_FIELDS = IDENTITY_FIELDS + FACT_FIELDS + SOURCE_FIELDS
OPTIONAL_FIELDS = ("field_sources", "note", "superseded_by", "enclosed_note", "density_range_g_cm3")      # `superseded_by`: the shipped catalog only
# A field that is not a fact but may name its page in `field_sources`.
SOURCED_FIELDS = FACT_FIELDS + ("enclosed_note",)
USER_ONLY_FIELDS = ("source",)                # `source: user`: the user said the values (a user entry gives this or `source_url`)
# A null is a plain fact here: no source states it, and the header of the catalog says so. Any other null needs a note that names the field.
NULL_WITHOUT_NOTE = ("nu", "chamber_c_min", "hardened_nozzle", "drying", "direct_drive", "conditioned", "bending_modulus_xy_mpa",
                     "bending_modulus_z_mpa")
TOP_KEYS = ("format", "catalog_version", "makers", "materials")
NO_VERSION = "none stated"               # the `sheet_version` of a sheet that prints no version
NU_REASON = "no data sheet gives it"
WAYS_ON = "Use an id from the catalog, or remove material.filament and give E and density in the material block"
WAYS_ON_LIST = "Use an id from the catalog, or leave the filament out of the list"
NOT_LISTED = ("If the filament is not listed: look up its data sheet (a web search), show the user the values with the address of the sheet, "
              "and after the user agrees call save_material with the values and the source_url; a value that you cannot find stays null "
              "(unknown). A run needs a modulus (the tensile E_xy_mpa or E_z_mpa; a sheet that gives only a bending modulus gives "
              "bending_modulus_xy_mpa or bending_modulus_z_mpa) and the density (density_g_cm3); the strengths, the "
              "temperatures and the print needs are facts for the checks. Then use the id of the saved filament. Or remove "
              "material.filament and give E and density in the material block.")
ADD_ON = (f"A filament that is not listed can be added to {USER_FILE_NAME} in the work folder (the MCP tool save_material writes it)")


# --- the profile -------------------------------------------------------------------------------------------------------------

def _pair(value):
    """A [min, max] list of 2 numbers above 0 as a tuple of floats, else None (a missing or bad value of a run.json)."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = (_pos(v) for v in value)
        if lo is not None and hi is not None and lo <= hi:
            return (lo, hi)
    return None


def _pos(value):
    """A number above 0 as a float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _flag(value):
    return value if isinstance(value, bool) else None


def _text_of(value):
    return value if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True)
class FilamentProfile:
    """One filament of the catalog: one product and one sheet version. A field that the catalog does not state is None (the `note` says why).

    XY is a load in the layer plane, Z is a load across the layers. `conditioned` holds the wet values of a polyamide as pairs.
    """
    id: str
    maker: str
    product: str
    family: str = ""
    aliases: tuple[str, ...] = ()
    sheet_version: str = ""
    E_xy_mpa: float | None = None
    E_z_mpa: float | None = None
    bending_modulus_xy_mpa: float | None = None       # only for a sheet that gives no tensile modulus; `modulus()` falls back to it
    bending_modulus_z_mpa: float | None = None
    strength_xy_mpa: float | None = None
    strength_z_mpa: float | None = None
    strength_kind: str = "not stated"
    elongation_xy_pct: float | None = None
    density_g_cm3: float | None = None
    density_range_g_cm3: tuple[float, float] | None = None      # [min, max] of a sheet that gives a range and no single density
    hdt_18_c: float | None = None
    hdt_045_c: float | None = None
    conditioned: tuple[tuple[str, object], ...] | None = None
    nu: float | None = None
    nozzle_c: tuple[float, float] | None = None
    bed_c: tuple[float, float] | None = None
    enclosure: str | None = None
    chamber_c_min: float | None = None
    hardened_nozzle: bool | None = None
    drying: str | None = None
    direct_drive: bool | None = None
    abrasive: bool | None = None
    soft: bool | None = None
    source_url: str = ""
    opened: str = ""
    tag: str = ""
    field_sources: tuple[tuple[str, str], ...] = ()
    note: str | None = None
    enclosed_note: str | None = None        # the text of the sheet about printing in an enclosed printer (a `care` reason on such a printer)
    superseded_by: str | None = None        # the id of the row of a newer sheet of the same product (shipped rows only); None: this is current
    catalog_version: str = ""
    origin: str = "catalog"                 # `catalog` (the file of the package) or `user` (topopt-filaments.yaml in the work folder)
    source: str | None = None               # `user` when the user said the values; None otherwise (user entries only)

    # The name index of `topopt.catalogs` reads a vendor and a model; for a filament these are the maker and the product.
    @property
    def vendor(self) -> str:
        return self.maker

    @property
    def model(self) -> str:
        return self.product

    @property
    def name(self) -> str:
        return f"{self.maker} {self.product}".strip()

    @property
    def origin_text(self) -> str:
        """Where the profile comes from, for a line of text: `catalog 2026-10-25`, or `user filament file`."""
        if self.origin == "user":
            return "user filament file"
        return f"catalog {self.catalog_version}".rstrip()

    @property
    def wet(self) -> dict | None:
        """The conditioned (wet) values as a mapping, or None."""
        return dict(self.conditioned) if self.conditioned is not None else None

    def facts(self) -> dict:
        """The record of the filament: the name, the facts, and where they come from. Lists and mappings, not tuples (JSON)."""
        out: dict = {"name": self.name, "maker": self.maker, "product": self.product, "family": self.family,
                     "sheet_version": self.sheet_version, "origin": self.origin}
        for key in FACT_FIELDS:
            value = getattr(self, key)
            out[key] = dict(value) if key == "conditioned" and value is not None else list(value) if isinstance(value, tuple) else value
        out["source_url"], out["opened"], out["tag"] = self.source_url, self.opened, self.tag
        if self.source:
            out["source"] = self.source
        if self.field_sources:
            out["field_sources"] = dict(self.field_sources)
        if self.note:
            out["note"] = self.note
        if self.enclosed_note:
            out["enclosed_note"] = self.enclosed_note
        if self.density_range_g_cm3 is not None:
            out["density_range_g_cm3"] = list(self.density_range_g_cm3)
        if self.superseded_by:
            out["superseded_by"] = self.superseded_by
        return out

    def facts_sha256(self) -> str:
        """The sha256 (hex) of the normalized facts of the filament: its name and the facts (`FACT_FIELDS`) as JSON with sorted keys and no
        blanks, and the `enclosed_note` and the `density_range_g_cm3` when the filament has them (the printer check and a run read them). It leaves out the sources, the dates, the
        note and the origin, so a changed note or page does not change it and a changed number does. `run.json` keeps it (`material.filament.profile_sha256`): two runs used the same filament facts if the digests are
        equal."""
        facts = self.facts()
        body = {"name": facts["name"], **{key: facts[key] for key in FACT_FIELDS}}
        for key in ("enclosed_note", "density_range_g_cm3"):
            if key in facts:
                body[key] = facts[key]
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        """`facts()` with the id and the catalog version first: the form that `find_material` returns."""
        return {"id": self.id, "catalog_version": self.catalog_version, **self.facts()}

    def short(self) -> dict:
        """The row of a pick list: the name, the sheet and the numbers that tell the rows of one name apart."""
        return {"id": self.id, "name": self.name, "family": self.family, "sheet_version": self.sheet_version, "origin": self.origin,
                "E_xy_mpa": self.E_xy_mpa, "E_z_mpa": self.E_z_mpa, "strength_xy_mpa": self.strength_xy_mpa,
                "strength_z_mpa": self.strength_z_mpa}

    @classmethod
    def from_facts(cls, fid: str, catalog_version, facts) -> "FilamentProfile | None":
        """The profile that a run recorded (`run.json`, key `material.filament`). A bad value reads as not stated; None if `facts` is no mapping.

        The catalog is not read: the record of the run is the truth for that run, also when the catalog changed since.
        """
        if not isinstance(fid, str) or not fid.strip() or not isinstance(facts, dict):
            return None

        def text(key):
            value = facts.get(key)
            return value if isinstance(value, str) and value.strip() else None

        wet = facts.get("conditioned")
        conditioned = None
        if isinstance(wet, dict):
            conditioned = tuple((k, (_pos(wet.get(k)) if k != "condition" else _text_of(wet.get(k)))) for k in CONDITIONED_KEYS)
        sources = facts.get("field_sources")
        sources = tuple((str(k), str(v)) for k, v in sources.items()) if isinstance(sources, dict) else ()
        enclosure, kind, origin = text("enclosure"), text("strength_kind"), text("origin")
        nu = facts.get("nu")
        return cls(
            id=fid.strip(), maker=text("maker") or "", product=text("product") or fid.strip(), family=text("family") or "",
            sheet_version=text("sheet_version") or "",
            E_xy_mpa=_pos(facts.get("E_xy_mpa")), E_z_mpa=_pos(facts.get("E_z_mpa")),
            bending_modulus_xy_mpa=_pos(facts.get("bending_modulus_xy_mpa")), bending_modulus_z_mpa=_pos(facts.get("bending_modulus_z_mpa")),
            strength_xy_mpa=_pos(facts.get("strength_xy_mpa")), strength_z_mpa=_pos(facts.get("strength_z_mpa")),
            strength_kind=kind if kind in STRENGTH_KINDS else "not stated", elongation_xy_pct=_pos(facts.get("elongation_xy_pct")),
            density_g_cm3=_pos(facts.get("density_g_cm3")), density_range_g_cm3=_pair(facts.get("density_range_g_cm3")), hdt_18_c=_pos(facts.get("hdt_18_c")), hdt_045_c=_pos(facts.get("hdt_045_c")),
            conditioned=conditioned, nu=float(nu) if isinstance(nu, (int, float)) and not isinstance(nu, bool) and -1 < nu < 0.5 else None,
            nozzle_c=_pair(facts.get("nozzle_c")), bed_c=_pair(facts.get("bed_c")),
            enclosure=enclosure if enclosure in ENCLOSURES else None, chamber_c_min=_pos(facts.get("chamber_c_min")),
            hardened_nozzle=_flag(facts.get("hardened_nozzle")), drying=text("drying"), direct_drive=_flag(facts.get("direct_drive")),
            abrasive=_flag(facts.get("abrasive")), soft=_flag(facts.get("soft")), source_url=text("source_url") or "",
            opened=text("opened") or "", tag=text("tag") or "", field_sources=sources, note=text("note"),
            enclosed_note=text("enclosed_note"), superseded_by=text("superseded_by"), catalog_version=catalog_version if isinstance(catalog_version, str) else "",
            origin=origin if origin in ORIGINS else "catalog", source=text("source"))


BENDING_REASON = "the bending modulus of the sheet; the sheet gives no tensile modulus"


def modulus(p: FilamentProfile):
    """`(E in MPa, reason)` that the engine takes from the profile, or None if the profile has no modulus at all.

    E is the lower of the XY and the Z tensile modulus (a part loaded across the layers is the weaker case, and the engine has one isotropic
    E). With one modulus only, that one is taken, and the reason says which. A profile with no tensile modulus uses the bending modulus of
    the same sheet in the same way (`bending_modulus_xy_mpa`, `bending_modulus_z_mpa`), and the reason says so. None: the caller asks for E.
    """
    if p.E_xy_mpa is not None and p.E_z_mpa is not None:
        return min(p.E_xy_mpa, p.E_z_mpa), "the lower of the XY and Z modulus"
    if p.E_xy_mpa is not None:
        return p.E_xy_mpa, "the XY modulus (the catalog has no Z modulus for this filament)"
    if p.E_z_mpa is not None:
        return p.E_z_mpa, "the Z modulus (the catalog has no XY modulus for this filament)"
    if p.bending_modulus_xy_mpa is not None and p.bending_modulus_z_mpa is not None:
        return min(p.bending_modulus_xy_mpa, p.bending_modulus_z_mpa), f"{BENDING_REASON}; the lower of the XY and Z value"
    if p.bending_modulus_xy_mpa is not None:
        return p.bending_modulus_xy_mpa, f"{BENDING_REASON}; the XY bending modulus, the sheet has no Z value"
    if p.bending_modulus_z_mpa is not None:
        return p.bending_modulus_z_mpa, f"{BENDING_REASON}; the Z bending modulus, the sheet has no XY value"
    return None


def no_modulus_text(p: FilamentProfile, key: str = "material.filament") -> str:
    """The reason a filament cannot set the modulus, as one text for an error. It names the fields that are null; the `note` of the profile
    (`find_material`) says why, so it is not copied here (a note can be long)."""
    return (f"'{key}' {p.id} ({p.name}, {p.origin_text}) has no tensile modulus (E_xy_mpa, E_z_mpa) and no bending modulus "
            f"(bending_modulus_xy_mpa, bending_modulus_z_mpa); find_material('{p.id}') shows the note of the sheet. Give E in the material "
            "block, or choose a filament that has one")


def _n(x) -> str:
    return f"{x:g}"


def sheet_density(p: FilamentProfile) -> tuple[float, str] | None:
    """`(density in g/cm3, reason)` that the sheet of the filament gives: its density, or the middle of the range when the sheet gives a
    range and no single value (`density_range_g_cm3`). None when the sheet gives neither (`family_density` is the fallback then)."""
    if p.density_g_cm3 is not None:
        return p.density_g_cm3, "the density of the sheet"
    if p.density_range_g_cm3 is not None:
        lo, hi = p.density_range_g_cm3
        return float(_n((lo + hi) / 2)), f"the middle of the sheet range {_n(lo)}-{_n(hi)}"
    return None


def family_density(p: FilamentProfile, catalog: "Catalog | None") -> tuple[float, str] | None:
    """`(density in g/cm3, reason)` for a filament that has no density: the median density of the shipped rows of the same family (read
    without case, as the resolver reads it) that state one (the filament itself, the user rows and the rows of an older sheet are left
    out: a list shows the current row only; a row that gives only a range is left out too, the median takes values that a sheet
    states), or None when no such row exists (the run then keeps the default of the
    problem file). A run labels it `assumed`: the sheet does not give it. The reason names the rows, for example
    `the median of 4 TPU rows of the catalog (bambu-lab-tpu-95a-hf, ...)`."""
    if catalog is None or not p.family:
        return None
    key = _family_key(p.family)
    rows = sorted((q.id, q.density_g_cm3) for q in catalog.profiles
                  if q.origin == "catalog" and not q.superseded_by and _family_key(q.family) == key and q.id != p.id
                  and q.density_g_cm3 is not None)
    if not rows:
        return None
    value = float(_n(statistics.median(d for _i, d in rows)))        # the same 6 digits that every text of this module shows
    return value, f"the median of {len(rows)} {p.family} row{'' if len(rows) == 1 else 's'} of the catalog ({', '.join(i for i, _d in rows)})"


def sheet_text(p: FilamentProfile) -> str:
    """The sheet version for a line of text, with a leading space; empty when the sheet has no version (`none stated`)."""
    return f" {p.sheet_version}" if p.sheet_version and p.sheet_version != NO_VERSION else ""


def facts_text(p: FilamentProfile) -> str:
    """The numbers that the tool takes from the profile, as text: `E 2080 MPa (the lower of the XY and Z modulus), density 1.04 g/cm3`."""
    m = modulus(p)
    parts = [f"E {_n(m[0])} MPa ({m[1]})" if m else "E not stated"]
    sheet = sheet_density(p)
    parts.append("density not stated" if sheet is None else f"density {_n(sheet[0])} g/cm3" + ("" if p.density_g_cm3 is not None else f" ({sheet[1]})"))
    return ", ".join(parts)


def summary_text(p: FilamentProfile) -> str:
    """One line for a person: the name, the id, the sheet and the numbers that the tool takes from the profile."""
    sheet = f", sheet{sheet_text(p)}" if sheet_text(p) else ""
    return f"{p.name} (filament {p.id}{sheet}): {facts_text(p)}"


# --- the file ----------------------------------------------------------------------------------------------------------------

def _text(where, key, value, allow_none=False):
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _fault(where, f"{key} must be a non-empty text{' or null' if allow_none else ''}, got {value!r}")
    return value.strip()


def _range(where, key, value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise _fault(where, f"{key} must be [min, max] or null, got {value!r}")
    lo, hi = (_number(where, key, v) for v in value)
    if lo is None or hi is None or lo > hi:
        raise _fault(where, f"{key} must be [min, max] with two numbers above 0 and min not above max, got {value!r}")
    return (lo, hi)


def _flag_value(where, key, value):
    if value is not None and not isinstance(value, bool):
        raise _fault(where, f"{key} must be true, false or null, got {value!r}")
    return value


def _nu(where, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 < value < 0.5:
        raise _fault(where, f"nu must be a number above -1 and below 0.5 or null, got {value!r}")
    return float(value)


def _conditioned(where, value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _fault(where, f"conditioned must be a mapping (condition and the wet values) or null, got {type(value).__name__}")
    unknown = [k for k in value if k not in CONDITIONED_KEYS]
    if unknown:
        raise _fault(where, f"conditioned has the unknown key {unknown[0]!r} (the keys are {', '.join(CONDITIONED_KEYS)})")
    condition = _text(where, "conditioned.condition", value.get("condition"))
    numbers = tuple((k, _number(where, f"conditioned.{k}", value.get(k))) for k in CONDITIONED_KEYS[1:])
    if all(n is None for _, n in numbers):
        raise _fault(where, "conditioned has no wet value; use null if the sheet gives none")
    return (("condition", condition), *numbers)


def _entry(raw, label: str, makers: dict, version: str, user: bool = False) -> FilamentProfile:
    """Check one entry of a file and return the profile. Raise `CatalogError`; the text starts with `label`.

    The shipped catalog is strict: every field is there (a fact that is not stated is null, and a null needs a note that names the field,
    except the plain nulls of `NULL_WITHOUT_NOTE`). A user entry needs `id`, `maker`, `product`, `family` and a source (`source_url` or
    `source: user`); a field that is not given is null, which means unknown.
    """
    where = label + (f" ({raw['id']})" if isinstance(raw, dict) and isinstance(raw.get("id"), str) else "")
    if not isinstance(raw, dict):
        raise _fault(where, f"a material must be a mapping, got {type(raw).__name__}")
    allowed = REQUIRED_FIELDS + tuple(k for k in OPTIONAL_FIELDS if not (user and k == "superseded_by")) + (USER_ONLY_FIELDS if user else ())
    unknown = [k for k in raw if k not in allowed]
    if unknown:
        raise _fault(where, f"unknown key {unknown[0]!r}")
    missing = [k for k in (USER_REQUIRED if user else REQUIRED_FIELDS) if k not in raw]
    if missing:
        raise _fault(where, f"missing key {missing[0]!r}" + ("" if user else " (a fact that is not stated is null, with a note)"))
    fid = raw["id"]
    if not isinstance(fid, str) or not ID_PATTERN.match(fid):
        raise _fault(where, f"id must be a lower-case slug (letters, digits, '-'), got {fid!r}")
    for key in ("maker", "product", "family"):
        _text(where, key, raw[key])
    if user and not _words(raw["maker"]):
        raise _fault(where, "maker must have a letter or a digit")
    if not user and raw["maker"] not in makers:
        raise _fault(where, f"maker {raw['maker']!r} is not in the `makers` table")
    aliases = raw.get("aliases") if "aliases" in raw or not user else []
    if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
        raise _fault(where, "aliases must be a list of non-empty texts")
    sheet = _text(where, "sheet_version", raw.get("sheet_version"), allow_none=user) or ""

    numbers = {key: _number(where, key, raw.get(key)) for key in NUMBER_FIELDS}
    ranges = {key: _range(where, key, raw.get(key)) for key in RANGE_FIELDS}
    flags = {key: _flag_value(where, key, raw.get(key)) for key in FLAG_FIELDS}
    kind = raw.get("strength_kind", "not stated" if user else None)
    if kind not in STRENGTH_KINDS:
        raise _fault(where, f"strength_kind must be one of {', '.join(STRENGTH_KINDS)}, got {kind!r}")
    if kind != "not stated" and numbers["strength_xy_mpa"] is None and numbers["strength_z_mpa"] is None:
        raise _fault(where, f"strength_kind is {kind}, but the entry has no strength")
    enclosure = raw.get("enclosure")
    if enclosure is False:                  # YAML 1.1 reads an unquoted no as false
        enclosure = "no"
    if enclosure is not None and enclosure not in ENCLOSURES:
        raise _fault(where, f"enclosure must be one of {', '.join(ENCLOSURES)} or null (write \"no\" with quotes), got {enclosure!r}")
    drying = _text(where, "drying", raw.get("drying"), allow_none=True)
    wet = _conditioned(where, raw.get("conditioned"))
    nu = _nu(where, raw.get("nu"))

    source = raw.get("source")
    url = raw.get("source_url")
    if user:
        if source is not None and source != "user":
            raise _fault(where, f"source must be the word user (the user said the values) or left out, got {source!r}")
        if url is None and source is None:
            raise _fault(where, "give source_url (the data sheet that has the values) or source: user (the user said them)")
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
        if key not in SOURCED_FIELDS:
            raise _fault(where, f"field_sources names {key!r}, which is not a fact field")
        _url(where, f"field_sources.{key}", value, allow_user=user)
    note = raw.get("note")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise _fault(where, "note must be a non-empty text")
    if ranges["density_range_g_cm3"] is not None and numbers["density_g_cm3"] is not None:
        raise _fault(where, "density_range_g_cm3 is for a sheet that gives a range and no single density, so density_g_cm3 must be null")
    enclosed_note = raw.get("enclosed_note")
    if enclosed_note is not None and (not isinstance(enclosed_note, str) or not enclosed_note.strip()):
        raise _fault(where, "enclosed_note must be a non-empty text")
    newer = raw.get("superseded_by")
    if newer is not None and (not isinstance(newer, str) or not ID_PATTERN.match(newer)):
        raise _fault(where, f"superseded_by must be the id of the row of the newer sheet (a lower-case slug), got {newer!r}")
    values = {**numbers, **ranges, **flags, "strength_kind": kind, "enclosure": enclosure, "drying": drying, "conditioned": wet, "nu": nu}
    if not user:
        for key in FACT_FIELDS:
            if values[key] is None and key not in NULL_WITHOUT_NOTE and not (note and key in note):
                raise _fault(where, f"{key} is null, so the material needs a note that names {key} and says why")
    return FilamentProfile(
        id=fid, maker=raw["maker"].strip(), product=raw["product"].strip(), family=raw["family"].strip(),
        aliases=tuple(a.strip() for a in aliases), sheet_version=sheet, **values, source_url=url, opened=opened, tag=tag,
        field_sources=tuple((k, str(v)) for k, v in sources.items()), note=note.strip() if note else None,
        enclosed_note=enclosed_note.strip() if enclosed_note else None, superseded_by=newer, catalog_version=version, origin="user" if user else "catalog", source=source)


class Catalog(NameIndex):
    """The checked catalog and its index of names. Build it with `parse_catalog` or `load_catalog`; `catalog_for` adds the user file.

    `version` is the version of the shipped catalog. `profiles` holds the shipped filaments first, then the user filaments. `makers` is the
    maker table (the shipped table, plus the words of the user file); `base_makers` is the shipped table: a shipped filament is indexed with
    its OWN maker words from that table, so a user file changes no shipped name. The index of the names is `topopt.catalogs.NameIndex`, the
    code that the printer catalog uses too (a maker is the vendor of the index, a product is its model).
    """

    def __init__(self, version: str, profiles: tuple[FilamentProfile, ...], makers: dict[str, tuple[str, ...]],
                 user_file: UserFile | None = None, base_makers: dict[str, tuple[str, ...]] | None = None):
        self.version = version
        self.user_file = user_file                              # None: the user file was not looked for
        super().__init__(profiles, makers, base_makers, label="filament catalog")

    @property
    def makers(self) -> dict[str, tuple[str, ...]]:
        return self.vendors


def parse_catalog(data, source: str = "materials.yaml") -> Catalog:
    """Check a loaded mapping and return the `Catalog`. Raise `CatalogError` at the first fault."""
    if not isinstance(data, dict):
        raise CatalogError(f"filament catalog {source}: the file must hold a mapping of keys ({', '.join(TOP_KEYS)})")
    unknown = [k for k in data if k not in TOP_KEYS]
    if unknown:
        raise CatalogError(f"filament catalog {source}: unknown key {unknown[0]!r}")
    if type(data.get("format")) is not int or data["format"] != FORMAT:
        raise CatalogError(f"filament catalog {source}: format is {data.get('format')!r}, and this code reads format {FORMAT}")
    version = data.get("catalog_version")
    if not isinstance(version, str) or not version.strip():
        raise CatalogError(f"filament catalog {source}: catalog_version must be a text, for example \"2026-10-25\"")
    raw_makers = data.get("makers")
    if not isinstance(raw_makers, dict) or not raw_makers:
        raise CatalogError(f"filament catalog {source}: makers must be a mapping of a maker name to a list of maker words")
    makers: dict[str, tuple[str, ...]] = {}
    for name, ws in raw_makers.items():
        if not isinstance(name, str) or not isinstance(ws, list) or not all(isinstance(w, str) and _words(w) for w in ws):
            raise CatalogError(f"filament catalog {source}: makers.{name}: must be a list of maker words")
        makers[name] = tuple(ws)
    rows = data.get("materials")
    if not isinstance(rows, list) or not rows:
        raise CatalogError(f"filament catalog {source}: materials must be a non-empty list")
    profiles = tuple(_entry(raw, f"filament catalog, materials[{i}]", makers, version.strip()) for i, raw in enumerate(rows))
    ids = [p.id for p in profiles]
    for fid in ids:
        if ids.count(fid) > 1:
            raise CatalogError(f"filament catalog {source}: the id {fid!r} is used twice")
    older = {p.id: p.superseded_by for p in profiles if p.superseded_by}
    for fid, newer in older.items():
        where = f"filament catalog {source}: {fid}"
        if newer == fid:
            raise CatalogError(f"{where}: a sheet cannot be older than itself (superseded_by {newer!r})")
        if newer not in ids:
            raise CatalogError(f"{where}: superseded_by {newer!r} is not a filament of the catalog")
        if newer in older:
            raise CatalogError(f"{where}: superseded_by {newer!r} is itself superseded; name the current row")
    return Catalog(version.strip(), profiles, makers)


def read_catalog_text() -> tuple[str, str]:
    """`(text, source)` of the file in the installed package. Raise `CatalogError` if it cannot be read or is empty."""
    place = resources.files("topopt").joinpath(*FILE)
    source = "topopt/" + "/".join(FILE)
    try:
        text = place.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"filament catalog: cannot read {source} ({exc})") from exc
    if not text.strip():
        raise CatalogError(f"filament catalog: {source} is empty")
    return text, source


def load_catalog(path=None) -> Catalog:
    """The catalog. With no `path`: the file of the package, read and checked once (cached). With a `path`: that file, not cached."""
    if path is None:
        return _default_catalog()
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"filament catalog: cannot read {p} ({exc})") from exc
    return _parse_text(text, str(p))


def _parse_text(text: str, source: str) -> Catalog:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CatalogError(f"filament catalog {source}: not valid YAML: {' '.join(str(exc).split())}") from exc
    return parse_catalog(data, source)


@functools.lru_cache(maxsize=1)
def _default_catalog() -> Catalog:
    text, source = read_catalog_text()
    return _parse_text(text, source)


# --- the user filament file --------------------------------------------------------------------------------------------------

USER_HEADER = (
    f"# topopt user filaments (format {FORMAT}): filaments that are not in the catalog that ships with topopt.\n"
    "# The entries are the same as in the shipped catalog. A value that is not known is null (unknown), never a guess.\n"
    "# Every entry names its source: source_url (the data sheet that has the values) or source: user (the user said them).\n"
    "# The server reads this file at each call. The MCP tool save_material adds or replaces an entry, and you can edit the file by hand.\n"
    "# A comment that you add here is not kept when save_material writes the file.\n")


class _FileDumper(_Dumper):
    """Block style for the entries (one fact on a line, a person edits the file); a list of plain values stays in one line."""


_FileDumper.add_representer(
    list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data,
                                                         flow_style=not any(isinstance(v, (dict, list)) for v in data)))


def user_file_path(folder) -> Path:
    return Path(folder) / USER_FILE_NAME


def work_folder() -> Path:
    """The work folder by the one rule of `topopt.workfolder` (CONTRACTS.md section 10.1), the same as for the printers."""
    return workfolder.resolve(check=False).path


def work_catalog() -> Catalog:
    """The shipped catalog with the user filaments of the work folder."""
    return catalog_for(work_folder())


def _read_user_file(path: Path):
    """`(state, mapping, problems)`. `state` is `absent`, `unreadable` (the system does not let the server read it), `broken` or `ok`."""
    return read_user_file(path, TOP_KEYS)


def _fault_text(exc: Exception) -> str:
    return str(exc).removeprefix("filament catalog: ").removeprefix("filament catalog, ")


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
    makers = dict(base.makers)
    raw_makers = data.get("makers") or {}
    if not isinstance(raw_makers, dict):
        raise _FileFault("makers must be a mapping of a maker name to a list of maker words")
    for name, ws in raw_makers.items():
        if not isinstance(name, str) or not _words(name) or not isinstance(ws, list) or not all(isinstance(w, str) and _words(w) for w in ws):
            raise _FileFault(f"makers.{name}: must be a list of maker words")
        makers[name] = (*makers.get(name, ()), *ws)
    rows = data.get("materials") or []
    if not isinstance(rows, list):
        raise _FileFault("materials must be a list of filaments")

    accepted: list[FilamentProfile] = []
    problems: list[str] = []
    for i, raw in enumerate(rows):
        label = f"{USER_FILE_NAME}, materials[{i}]"
        try:
            profile = _entry(raw, label, makers, version, user=True)
        except CatalogError as exc:
            problems.append(f"{exc}. This entry is not used.")
            continue
        where = f"{label} ({profile.id})"
        shipped = base.by_id.get(profile.id)
        if shipped is not None:
            problems.append(f"{where}: the id is the id of the shipped filament {shipped.name}. A user filament never replaces a shipped one, "
                            f"so this entry is not used. Use another id, for example {profile.id}-user.")
            continue
        if any(a.id == profile.id for a in accepted):
            problems.append(f"{where}: the id is used twice in this file. The first entry is used and this one is not.")
            continue
        profile = replace(profile, maker=_canonical_maker(makers, profile.maker))
        trial = dict(makers)
        trial.setdefault(profile.maker, ())
        try:
            Catalog(base.version, base.profiles + tuple(accepted) + (profile,), trial, base_makers=base.makers)
        except CatalogError as exc:
            problems.append(f"{where}: {_fault_text(exc)}. The name of a filament must differ from the names of the other filaments of "
                            "its maker, so this entry is not used. Give it another product name or alias.")
            continue
        makers = trial
        accepted.append(profile)
    try:
        return Catalog(base.version, base.profiles + tuple(accepted), makers, UserFile(str(path), True, len(accepted), tuple(problems)),
                       base_makers=base.makers)
    except CatalogError as exc:                         # the names cannot be indexed together: a fact about the file, not an exception
        raise _FileFault(f"the makers and the filaments of the file cannot be used together ({_fault_text(exc)})") from exc


def catalog_for(folder) -> Catalog:
    """The shipped catalog with the user filaments of `<folder>/topopt-filaments.yaml`. `folder` None: the shipped catalog only.

    The file is read at each call (it is small, and `save_material` changes it). A file that is not there is no problem. A file that
    cannot be used, or an entry with a fault, is a fact in `Catalog.user_file.problems`: the rest still works, nothing is raised.
    A user filament never replaces a shipped one. Raises `CatalogError` only if the shipped catalog is broken.
    """
    base = load_catalog()
    if folder is None:
        return base
    path = user_file_path(folder)
    state, data, found = _read_user_file(path)
    if state == "absent":
        return Catalog(base.version, base.profiles, base.makers, UserFile(str(path), False))
    if state == "ok" and data is not None:
        try:
            return _merge(base, path, data)
        except _FileFault as exc:
            found = [str(exc)]
    problems = tuple(f"{USER_FILE_NAME}: {text}. The file is not used." for text in found)
    return Catalog(base.version, base.profiles, base.makers, UserFile(str(path), True, 0, problems))


def entry_dict(p: FilamentProfile) -> dict:
    """The entry of a user profile as it is written in the file: every fact field is there, null when it is not known."""
    out: dict = {"id": p.id, "family": p.family, "maker": p.maker, "product": p.product, "aliases": list(p.aliases)}
    if p.sheet_version:
        out["sheet_version"] = p.sheet_version
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
        if key == "conditioned" and value is not None:
            out[key] = {k: _out(v) for k, v in value}
        elif isinstance(value, tuple):
            out[key] = [_out(v) for v in value]
        else:
            out[key] = _out(value)
    if p.field_sources:
        out["field_sources"] = dict(p.field_sources)
    if p.note:
        out["note"] = p.note
    if p.enclosed_note:
        out["enclosed_note"] = p.enclosed_note
    if p.density_range_g_cm3 is not None:
        out["density_range_g_cm3"] = list(p.density_range_g_cm3)
    return out


def _refuse(text: str) -> dict:
    return {"ok": False, "error": text}


def save_user_filament(folder, raw) -> dict:
    """Add or replace one filament in `<folder>/topopt-filaments.yaml` (the MCP tool `save_material`).

    `raw` has the fields of an entry (`USER_REQUIRED` and a source are needed). The entry is checked as the loader checks it. An entry
    with the same id in the file is replaced; a shipped id, or a name of another filament of the maker, is refused. The file is written
    atomically; the other entries and keys stay as they are. A file that cannot be read or is not valid is not changed.
    Returns `{ok, path, profile, created, replaced, note}` (+ `problems` of the rest of the file), or `{ok: False, error}`.
    """
    from . import runinfo      # lazy: runinfo reads the work folder, and this module is read by config

    path = user_file_path(folder)
    if not isinstance(raw, dict):
        return _refuse(f"save_material wrote nothing. profile must be a mapping with the fields of one filament ({', '.join(USER_REQUIRED)} "
                       "and a source), got " + type(raw).__name__ + ".")
    state, old, found = _read_user_file(path)
    if state == "unreadable":
        return _refuse(f"save_material did not change {path}: {found[0]}. It does not write over a file that it cannot read.")
    if state == "broken":
        return _refuse(f"save_material did not change {path}: {found[0]}. Fix the file by hand, or remove it.")
    old = old or {}
    try:
        base = load_catalog()
    except CatalogError as exc:
        return _refuse(f"save_material wrote nothing. The shipped catalog cannot be read, so the name cannot be checked: {exc}.")
    try:
        profile = _entry(raw, "profile", base.makers, "", user=True)
    except CatalogError as exc:
        return _refuse(f"save_material wrote nothing. {exc}.")
    shipped = base.by_id.get(profile.id)
    if shipped is not None:
        return _refuse(f"save_material wrote nothing. profile ({profile.id}): the id is the id of the shipped filament {shipped.name}. "
                       f"A user filament never replaces a shipped one. Use another id, for example {profile.id}-user. If the values of "
                       "the shipped filament are wrong for the spool of the user, save the spool with its own id and product name.")
    rows = old.get("materials") or []
    others = {**old, "materials": [m for m in rows if not (isinstance(m, dict) and m.get("id") == profile.id)]} \
        if isinstance(rows, list) else old
    try:
        merged = _merge(base, path, others)
    except _FileFault as exc:
        return _refuse(f"save_material did not change {path}: {exc}. Fix the file by hand, or remove it.")
    profile = replace(profile, maker=_canonical_maker(merged.makers, profile.maker))
    makers = dict(merged.makers)
    makers.setdefault(profile.maker, ())
    try:
        Catalog(base.version, merged.profiles + (profile,), makers, base_makers=base.makers)
    except CatalogError as exc:
        return _refuse(f"save_material wrote nothing. profile ({profile.id}): {_fault_text(exc)}. The name of a filament must differ from "
                       "the names of the other filaments of its maker. Give it another product name or alias.")

    entry = entry_dict(profile)
    replaced = False
    listed = []
    for m in (rows if isinstance(rows, list) else []):
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
    if old.get("makers") is not None:
        doc["makers"] = old["makers"]
    doc["materials"] = listed
    made = state == "absent"
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        text = USER_HEADER + yaml.dump(doc, Dumper=_FileDumper, sort_keys=False, default_flow_style=False, allow_unicode=True,
                                                       width=1000)
        runinfo.write_atomic(path, text.encode("utf-8"))
    except OSError as exc:
        text = f"save_material could not write {path}: {exc.strerror or exc}."
        if exc.errno == errno.EPERM and sys.platform == "darwin":
            text += f" {workfolder.MAC_PRIVACY}."
        return _refuse(text)

    after = catalog_for(folder)
    saved = after.by_id.get(profile.id, profile)
    unknown = [k for k in ("E_xy_mpa", "E_z_mpa", "density_g_cm3") if getattr(saved, k) is None and not (k == "density_g_cm3" and sheet_density(saved))]
    note = (f"Saved {saved.name} as the user filament {saved.id} ({'replaced' if replaced else 'new entry'}"
            f"{', new file' if made else ''}). The server reads the file at each call, so the filament is live now: "
            f"material.filament: {saved.id} in a YAML sets material.E and material.density from it, and save_preferences keeps "
            "material.filament for the next runs.")
    if replaced:
        note += (f" A YAML that names {saved.id} gets the new values at its next check or run (the YAML text did not change, so nothing "
                 "else notes it); old runs keep their record in run.json, and material.filament.profile_sha256 there tells whether two runs "
                 "used the same filament facts.")
    found = modulus(saved)
    if found is None:
        note += (" The filament has no tensile modulus (E_xy_mpa, E_z_mpa) and no bending modulus (bending_modulus_xy_mpa, "
                 "bending_modulus_z_mpa), so material.filament cannot set E: ask the user for it or find it on the sheet.")
    elif found[1].startswith(BENDING_REASON):
        note += " material.E comes from the bending modulus of the entry (E_xy_mpa and E_z_mpa are null)."
    elif unknown:
        note += f" Not known (null): {', '.join(unknown)}. Ask the user for a value that matters, then save the filament again."
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
    profile: FilamentProfile | None = None                    # match
    matched_by: str | None = None                             # match: id | name | alias. ambiguous: partial | name | vendor | family
    candidates: tuple[FilamentProfile, ...] = ()              # ambiguous
    differs: tuple[str, ...] = ()                             # ambiguous: the facts on which the candidates differ
    suggestions: tuple[tuple[FilamentProfile, float], ...] = ()   # none: "did you mean", best first. Never taken by itself
    makers: tuple[str, ...] = ()                              # none: the makers of the catalog
    note: str = ""
    user_file: UserFile | None = None                         # the user filament file of the work folder (None: not looked for)
    catalog: "Catalog | None" = None                          # the catalog that was searched (the family density of a match comes from it)

    def to_dict(self) -> dict:
        """The reply of the MCP tool `find_material` (the same shape for the three statuses)."""
        return {
            "ok": True, "status": self.status, "query": self.query, "catalog_version": self.catalog_version,
            "matched_by": self.matched_by,
            "origin": self.profile.origin if self.profile else None,
            "profile": self.profile.summary() if self.profile else None,
            "derives": derives(self.profile, self.catalog) if self.profile else None,
            "candidates": [p.short() for p in self.candidates],
            "differs": list(self.differs),
            "suggestions": [{"id": p.id, "name": p.name, "score": round(score, 3)} for p, score in self.suggestions],
            "makers": list(self.makers),
            "user_file": self.user_file.facts() if self.user_file is not None else None,
            "note": self.note,
        }


def derives(p: FilamentProfile, catalog: "Catalog | None" = None) -> dict:
    """What `material.filament: <id>` sets in a run (a key that the YAML gives wins), what stays assumed, and what the profile cannot give.

    The density is the one of the sheet, or the middle of the range that the sheet gives (`sheet_density`). With a `catalog`, a filament
    whose sheet gives neither gets the median density of its family as an assumed value (`family_density`)."""
    from .config import Material      # lazy: config imports this module

    sets: dict[str, object] = {}
    missing: list[str] = []
    m = modulus(p)
    if m is not None:
        sets["material.E"] = m[0]
    else:
        missing.append("material.E")
    sheet = sheet_density(p)
    if sheet is not None:
        sets["material.density"] = sheet[0]
    else:
        missing.append("material.density")
    assumed = {"material.nu": Material().nu} if p.nu is None else {}
    fallback = family_density(p, catalog) if sheet is None else None
    if fallback is not None:
        assumed["material.density"] = fallback[0]
    if p.nu is not None:
        sets["material.nu"] = p.nu
    return {"sets": sets, "assumed": assumed, "missing": missing}


def _differing(candidates) -> tuple[str, ...]:
    return tuple(f for f in CHOICE_FIELDS if len({repr(getattr(p, f)) for p in candidates}) > 1)


def resolve(text, catalog: Catalog | None = None) -> Resolution:
    """Find the filament that `text` names. `catalog` (default: the shipped catalog) may hold user filaments (`catalog_for`).

    - `match`: the text is the id, or the name or an alias of exactly one filament (case, punctuation, maker words such as `bambu` or
      `prusament`, and the word `my` do not matter). A maker word in the text is a hard filter.
    - `ambiguous`: the text is not an exact name, or it is the exact name of filaments of two makers; `candidates` are the filaments whose
      names contain all the words of the text (a text with only a maker word gives all filaments of the maker). The caller asks the
      user; `differs` names the facts on which the candidates differ.
    - `ambiguous` with `matched_by: family`: the text is only the name of a material family (`PLA`, `PETG-CF`) and has no maker word.
      A family is not a product, so every filament of the family is a candidate.
    - `none`: no filament fits. `suggestions` are the names close to the text by difflib; none is taken by itself.

    The lookup of a name is `topopt.catalogs.find`; the words of the answer are written here. The two "did you mean" numbers are the ones of
    the printer catalog (`topopt.printers.SUGGEST_CUTOFF`, `SUGGEST_MAX`).
    """
    cat = catalog or load_catalog()
    query = text if isinstance(text, str) else ""
    shown = " ".join(query.split())

    def answer(status: str, **rest) -> Resolution:
        return Resolution(status, shown, cat.version, user_file=cat.user_file, catalog=cat, **rest)

    if not _words(query):
        return answer("none", makers=tuple(cat.makers),
                      note="The name is empty. Ask the user which filament they have (maker and product, for example the label of the spool).")

    by_id = cat.by_id.get(query.strip().lower())
    if by_id is not None:
        return answer("match", profile=by_id, matched_by="id", note=_match_note(by_id, "id", cat))

    family = _family(cat, query)
    if family:
        current = current_rows(family)
        return _ambiguous(answer, current, "family", f"The text names a material family ({family[0].family}), not one product: "
                          f"the catalog has {len(current)} current filaments of it.", [p for p in family if p not in current])
    hit = catalogs.find(cat, query)
    if hit is not None and hit.status == "match":
        return answer("match", profile=hit.profile, matched_by=hit.matched_by,
                      note=_match_note(hit.profile, hit.matched_by, cat))
    if hit is not None:
        current = current_rows(list(hit.candidates))
        return _ambiguous(answer, current, hit.matched_by,
                          {"name": "The name is the name of filaments of more than one maker.",
                           "vendor": "The text names only the maker."}.get(hit.matched_by), [p for p in hit.candidates if p not in current])
    suggestions = catalogs.suggest_names(cat, query, printers.SUGGEST_CUTOFF, printers.SUGGEST_MAX)
    where = "the catalog or the user filament file" if cat.user_file is not None else "the catalog"
    note = (f"No filament of {where} has this name. "
            + ("Ask the user if they mean one of the suggestions (a suggestion is never taken by itself). " if suggestions else "")
            + NOT_LISTED)
    if cat.user_file is not None and cat.user_file.problems:
        note += " The user filament file has problems: " + " ".join(cat.user_file.problems)
    return answer("none", suggestions=suggestions, makers=tuple(cat.makers), note=note)


def current_rows(rows) -> list[FilamentProfile]:
    """`rows` without the rows of an older sheet whose newer row is in `rows` too: a list shows the current row of a product."""
    ids = {p.id for p in rows}
    return [p for p in rows if not (p.superseded_by and p.superseded_by in ids)]


def _older_text(older) -> str:
    return (f" Older sheets of listed products are not listed ({', '.join(p.id for p in older)}); the id of one selects it." if older else "")


def _family_key(text) -> str:
    return " ".join(catalogs.parts([w for w in _words(text) if w not in catalogs.FILLER_WORDS]))


def _family(cat: Catalog, query: str) -> list[FilamentProfile]:
    """The filaments of a family when the text is only the name of a family with two filaments or more (`PLA`, `PETG-CF`) and has no
    maker word: a family is not a product, so the caller asks. Else an empty list."""
    rest, found = cat.strip_vendors(_words(query))
    key = " ".join(catalogs.parts(rest))
    if found or not key:
        return []
    same = [p for p in cat.profiles if _family_key(p.family) == key]
    return same if len(same) > 1 else []


def _ambiguous(answer, candidates: list[FilamentProfile], matched_by: str, reason: str | None, older=()) -> Resolution:
    """`matched_by` says why the answer is not a match: `partial` (the text is a part of a name), `name` (the exact name of filaments of
    more than one maker), `vendor` (the text is only a maker word) or `family` (the text is only the name of a material family)."""
    differs = _differing(candidates) if len(candidates) > 1 else ()
    if len(candidates) == 1:
        p = candidates[0]
        note = (f"The text is only part of the name of one filament in the catalog: {p.name} (id {p.id}). A part of a name is not the "
                "name: a similar product can have other numbers. Ask the user if this is their filament. If yes, use the id.")
    else:
        count = "" if matched_by == "family" else (f"The text is part of the names of {len(candidates)} filaments. "
                                                  if matched_by == "partial" else f"The text fits {len(candidates)} filaments. ")
        note = ((reason + " ") if reason else "") + count \
            + "Ask the user which one" + (f"; they differ in {', '.join(differs)}" if differs else "") + ". Then use the id."
    note += _older_text(older)
    return answer("ambiguous", matched_by=matched_by, candidates=tuple(candidates), differs=differs, note=note)


def _match_note(p: FilamentProfile, how: str, catalog: Catalog | None = None) -> str:
    d = derives(p, catalog)
    note = (f"Matched {p.name}{' sheet' + sheet_text(p) if sheet_text(p) else ''} by {how} ({p.origin_text}). "
            f"material.filament: {p.id} sets {facts_text(p)} when the YAML does not give them; material.nu stays assumed "
            f"({NU_REASON}).")
    if p.superseded_by:
        note += (f" This is an older sheet: a newer sheet of the same product exists (filament {p.superseded_by}); the spool of the user may "
                 "be of either, so ask which one.")
    if d["missing"]:
        note += f" The catalog does not state {', '.join(d['missing'])} for this filament: ask the user for it if it matters. See the note of the profile."
        fallback = family_density(p, catalog) if "material.density" in d["missing"] else None
        if fallback is not None:
            note += f" A run uses a density of {_n(fallback[0])} g/cm3 ({fallback[1]}), labelled assumed."
    return note


def profile_for(filament_id, catalog: Catalog | None = None) -> FilamentProfile | None:
    """The profile with this id, or None."""
    return (catalog or load_catalog()).by_id.get(filament_id) if isinstance(filament_id, str) else None


def refusal_text(res: Resolution, key: str = "material.filament", ways_on: str = WAYS_ON) -> str:
    """One line for a `UserError` when `res` is not a match: the key, the text, the candidates or suggestions, and the ways on."""
    head = f"'{key}' {res.query!r}"

    def listed(profiles) -> str:
        return ", ".join(f"{p.id} ({p.name}{sheet_text(p)}{', user filament' if p.origin == 'user' else ''})" for p in profiles)

    if res.status == "ambiguous":
        n = len(res.candidates)
        where = "the catalog or the user filament file" if any(p.origin == "user" for p in res.candidates) else "the catalog"
        text = f"{head} fits {n} filament{'' if n == 1 else 's'} of {where} (version {res.catalog_version}): {listed(res.candidates)}. {ways_on}"
    elif res.suggestions:
        text = (f"{head} is not in the filament catalog (version {res.catalog_version}). "
                f"Did you mean: {listed(p for p, _ in res.suggestions)}? {ways_on}. {ADD_ON}")
    else:
        text = (f"{head} is not in the filament catalog (version {res.catalog_version}) and no name is close to it. "
                f"{ways_on}. {ADD_ON}")
    uf = res.user_file
    if uf is not None and res.status != "ambiguous":
        text += f" (user filament file {uf.path}: " + (f"{uf.entries} filaments" if uf.exists else "not there") + ")"
    if uf is not None and uf.problems:
        text += " The user filament file has problems: " + " ".join(uf.problems)
    return text


def check_lines(facts: dict) -> list[str]:
    """Text lines for `topopt check`: the `material` facts (`config.FilamentUse.check_facts`). Facts only: nothing here stops a run."""
    parts = []
    for key, unit in (("E", " MPa"), ("nu", ""), ("density", " g/cm3")):
        e = facts["material"].get(key)
        if e is not None:
            why = f", {e['reason']}" if e.get("reason") else ""
            parts.append(f"{key} {_n(e['value'])}{unit} ({e['source']}{why})")
    sheet = f", sheet {facts['sheet_version']}" if facts.get("sheet_version") and facts["sheet_version"] != NO_VERSION else ""
    lines = [f"material:       {facts['name']} (filament {facts['filament']}, {facts['profile_from']}{sheet}): " + ", ".join(parts)]
    if facts.get("superseded_by"):
        lines.append(f"material sheet: a newer sheet of this filament exists: {facts['superseded_by']}; the spool may be of either")
    c = facts.get("compat")
    if c is None:
        lines.append("material fit:   not checked: the YAML has no print.printer_model, so there is no printer to compare with")
        return lines
    lines.append(f"material fit:   {c['state']}: {c['filament']} on {c['printer']}")
    lines += [f"                {r['state']}: {r['text']}" for r in c["reasons"] if r["state"] != "ok"]
    return lines



GROUPS = ("ok", "care", "no", "unknown")      # the states of `topopt.compat`, in the order that `printer_fit` lists them


def printer_fit(printer, catalog: Catalog, available=()) -> dict:
    """Every filament of `catalog` against `printer` (a `topopt.printers.PrinterProfile`), grouped by the state of `topopt.compat.check`.

    This is the answer of the MCP tool `printer_materials`. A state is a fact about stated values, not a stop. A filament with state `ok` has no
    reason (every check fits); the others list the reasons that are not `ok`, each with both values; the source of the printer is in `printer.from` and the source of each filament in its `source`. `available` are the ids of
    the filaments that the user has (`filaments.available` of the preferences): they are marked, and listed in the order given with their state.
    """
    from . import compat      # lazy: a run without a filament never loads it

    have = list(dict.fromkeys(a for a in available if isinstance(a, str)))
    groups: dict[str, list] = {g: [] for g in GROUPS}
    state_of: dict[str, str] = {}
    older = [p for p in catalog.profiles if p.superseded_by and p.superseded_by in catalog.by_id]
    for p in sorted(catalog.profiles, key=lambda q: (q.name.lower(), q.id)):
        answer = compat.check(printer, p)
        state_of[p.id] = answer.state
        if p in older:
            continue                    # a list shows the current row of a product; an older sheet is counted in the note
        printer_ref, filament_ref = compat.refs(printer, p)      # the reasons repeat these two; the list gives each source once
        shown = [{**r.to_dict(), "text": r.text.replace(filament_ref, p.name).replace(printer_ref, printer.name)}
                 for r in answer.reasons if r.state != "ok"]
        groups[answer.state].append({"id": p.id, "name": p.name, "family": p.family, "origin": p.origin, "available": p.id in have,
                                     "source": filament_ref.removeprefix(p.name).strip().strip("()"), "reasons": shown})
    listed = []
    for fid in have:
        p = catalog.by_id.get(fid)
        if p is None:
            listed.append({"id": fid, "name": None, "state": None, "note": "this filament is not in the catalog or the user filament file now"})
        elif p.superseded_by:
            listed.append({"id": fid, "name": p.name, "state": state_of[fid],
                           "note": f"a newer sheet exists: {p.superseded_by}; this older sheet is not in the groups"})
        else:
            listed.append({"id": fid, "name": p.name, "state": state_of[fid]})
    return {"printer": {"id": printer.id, "name": printer.name, "origin": printer.origin, "from": printer.origin_text},
            "catalog_version": catalog.version, "counts": {g: len(groups[g]) for g in GROUPS}, "groups": groups, "available": listed,
            "user_file": catalog.user_file.facts() if catalog.user_file is not None else None,
            "note": ("A state is a fact about the values that the sources state, not a stop. ok: every stated value fits. care: it can work, and "
                     "the person has to do or know something (the reasons say what). no: a stated need is above a stated limit of the printer as "
                     "it ships (the user may have changed the printer). unknown: a value is missing on one side (the reason names it). The data "
                     "sheet of a filament may be for a version of the product that differs from the spool of the user."
                     + (f" {len(older)} filament{' with an older sheet is' if len(older) == 1 else 's with an older sheet are'} not listed "
                        f"({', '.join(p.id for p in older)}): the list shows the row of the newer sheet. Ask about the id of one to see it."
                        if older else ""))}
