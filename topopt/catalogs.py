"""The parts that the printer catalog and the filament catalog share (CONTRACTS.md sections 18.7 and 18.8).

Both catalogs are a list of named products with a maker (a vendor) and a user file in the work folder. This module holds the code that is
the same for both, so that there is one copy of it:

- `CatalogError` and the checks of one value of a catalog file (`number`, `url`, `date_text`).
- The name index (`NameIndex`) and the lookup of a text in it (`find`, `suggest_names`): the words of a name, the vendor words as a hard filter,
  the exact id, the model name, the aliases, and the part of a name. The caller adds the texts of its own answer.
- The reading of a user file (`read_user_file`) and the YAML dumper of the file that a tool writes.

It imports no part of topopt except `workfolder`, and not `mcp`.
"""
from __future__ import annotations

import datetime
import difflib
import errno
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import workfolder

# Words that do not name a model: the person words, and the words of a shop name ("P1S Combo", "A1 mini AMS bundle", "MK4S kit").
FILLER_WORDS = frozenset({"my", "the", "printer", "3d", "combo", "bundle", "ams", "cfs", "kit"})
ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class CatalogError(Exception):
    """A catalog file is missing, cannot be read, or breaks a rule of its format."""


# --- one value of a file -----------------------------------------------------------------------------------------------------

def fault(where: str, text: str) -> CatalogError:
    return CatalogError(f"{where}: {text}")


def number(where, key, value):
    """A number above 0, or None (the fact is not stated)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        hint = ""
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+(\.\d*)?[eE][+-]?\d+", value.strip()):
            hint = " (YAML reads a number such as 1e9 as text: write 1.0e+9, or the digits)"
        raise fault(where, f"{key} must be a number above 0 or null, got {value!r}{hint}")
    return float(value)


def required_number(where, key, value) -> float:
    n = number(where, key, value)
    if n is None:
        raise fault(where, f"{key} must be a number")
    return n


def url(where, key, value, allow_user=False):
    if allow_user and value == "user":
        return value
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        raise fault(where, f"{key} must be an http(s) address{' or the word user' if allow_user else ''}, got {value!r}")
    return value


def date_text(where, value, allow_none=False) -> str:
    """A date as text, YYYY-MM-DD. A user file may hold an unquoted date, which YAML reads as a date: that is the same text."""
    if allow_none and value is None:
        return ""
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, str):
        try:
            datetime.date.fromisoformat(value)
            return value
        except ValueError:
            pass
    raise fault(where, f"opened must be a date as text, YYYY-MM-DD, got {value!r}")


def out_number(value):
    """A number for the file: 220.0 is written as 220."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# --- names -------------------------------------------------------------------------------------------------------------------

def words(text) -> list[str]:
    """Lower-case words of a text: letters and digits only. `+` reads as the word `plus` (MINI+, K2+); accents are dropped."""
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()
    return re.findall(r"[a-z0-9]+", ascii_text.replace("+", " plus "))


def parts(ws) -> list[str]:
    """Each word cut at the border of a letter run and a digit run: `p1s` -> p 1 s, so `p1 s` and `p1s` are one name."""
    return [part for w in ws for part in re.findall(r"[a-z]+|[0-9]+", w)]


def name_parts(ws: list[str]) -> tuple[tuple[str, ...], frozenset[int]]:
    """The parts of a name (`parts`) and the index in them where each word starts."""
    out: list[str] = []
    starts: set[int] = set()
    for w in ws:
        starts.add(len(out))
        out.extend(parts([w]))
    return tuple(out), frozenset(starts)


class NameIndex:
    """The index of the names of a list of profiles. A profile has `id`, `vendor`, `model`, `aliases`, `origin` and `name`.

    `vendors` is the vendor table (a vendor name mapped to its words). `base_vendors` is the shipped table: a shipped profile (origin
    `catalog`) is indexed with its OWN vendor words from that table, so a user file changes no shipped name. A user profile is indexed
    with its own vendor words from `vendors`. The words of ALL vendors still find the vendor in a text (`strip_vendors`).
    `label` starts the text of a fault (`printer catalog`).
    """

    def __init__(self, profiles: tuple, vendors: dict[str, tuple[str, ...]], base_vendors: dict[str, tuple[str, ...]] | None = None,
                 label: str = "catalog"):
        self.profiles = profiles
        self.vendors = vendors                                  # vendor name -> the vendor words (the name itself is one too)
        self.base_vendors = vendors if base_vendors is None else base_vendors
        self._phrases = self._phrase_list(vendors)
        self.by_id = {p.id: p for p in profiles}
        self._keys: dict[str, frozenset[str]] = {}               # id -> the exact names (as keys)
        self._model_key: dict[str, str] = {}                     # id -> the key of the model name
        self._own: dict[str, list] = {}                          # id -> the vendor phrases of this profile's own vendor
        self._names: dict[str, list[tuple[tuple[str, ...], frozenset[int]]]] = {}     # id -> each name as (its parts, where its words start)
        seen: dict[tuple[str, str], str] = {}
        for p in profiles:
            table = self.base_vendors if p.origin == "catalog" else vendors
            own = self._phrase_list({p.vendor: table.get(p.vendor, ())})        # only the vendor words of THIS profile
            self._own[p.id] = own
            keys, names = set(), []
            for text in (p.model, *p.aliases):
                rest, _ = self._strip(words(text), own)
                if not rest:
                    raise CatalogError(f"{label}, {p.id}: the name {text!r} has no word besides the vendor words")
                key = " ".join(parts(rest))
                other = seen.setdefault((p.vendor, key), p.id)
                if other != p.id:
                    raise CatalogError(f"{label}: the name {text!r} of {p.id} is also a name of {other} ({self.by_id[other].name})")
                if text == p.model:
                    self._model_key[p.id] = key
                keys.add(key)
                names.append(name_parts(rest))
            self._keys[p.id] = frozenset(keys)
            self._names[p.id] = names

    def holds(self, pid: str, ps: list[str]) -> bool:
        """True if `ps` are a run of the parts of one name of the profile that starts at the start of a word of the name (it may end
        inside the last word): `x1` and `x1 carbon` are parts of `X1 Carbon`; `p1p`, `s1`, `1s` and `carbon x1` are not."""
        n = len(ps)
        return any(whole[i:i + n] == tuple(ps) for whole, starts in self._names[pid] for i in starts)

    @staticmethod
    def _phrase_list(table: dict[str, tuple[str, ...]]):
        """The vendor phrases of a vendor table as `(words, vendor name)`, the longest phrase first."""
        phrases: dict[tuple[str, ...], str] = {}
        for name, ws in table.items():
            for text in (name, *ws):
                phrases.setdefault(tuple(words(text)), name)
        return sorted(phrases.items(), key=lambda kv: -len(kv[0]))

    @staticmethod
    def _strip(ws: list[str], phrases):
        """`(rest, vendors)`: the words without the phrases and filler words, and the vendor names found, in order."""
        rest, found, i = [], [], 0
        while i < len(ws):
            for phrase, name in phrases:
                if tuple(ws[i:i + len(phrase)]) == phrase:
                    if name not in found:
                        found.append(name)
                    i += len(phrase)
                    break
            else:
                if ws[i] not in FILLER_WORDS:
                    rest.append(ws[i])
                i += 1
        return rest, found

    def strip_vendors(self, ws: list[str]):
        """`(rest, vendors)`: the words without the vendor phrases of all vendors and filler words, and the vendor names found."""
        return self._strip(ws, self._phrases)

    def names_of(self, p) -> list[str]:
        """The texts that a person may type for `p`, as compact strings (for the "did you mean" list)."""
        texts = [p.id, f"{p.vendor} {p.model}", p.model, *p.aliases, *(f"{p.vendor} {a}" for a in p.aliases)]
        return ["".join(w for w in words(t) if w not in FILLER_WORDS) for t in texts]


@dataclass(frozen=True)
class Hit:
    """A profile that a text names: `status` is `match` (one profile; `matched_by` is `name` or `alias`) or `ambiguous` (`candidates`;
    `matched_by` is `name` (the exact name of profiles of more than one vendor), `partial` (the text is a part of a name) or `vendor`
    (the text is only a vendor word)). The words of the answer are the caller's."""
    status: str
    profile: Any = None            # the profile of a `match` (the type is the caller's: a printer or a filament), None otherwise
    matched_by: str = ""           # `name` or `alias` for a match; `name`, `partial` or `vendor` for an ambiguous answer; always set
    candidates: tuple = ()


def find(index: NameIndex, query: str) -> Hit | None:
    """The `Hit` for a text by name (the exact id is the caller's first step), or None if no profile fits.

    - A vendor word in the text is a hard filter: the text is read with the words of all vendors taken away, and only the profiles of the
      vendors found are looked at. One profile with that exact name is a `match`; the exact name of profiles of two vendors is `ambiguous`;
      the profiles with a name that holds the text as a part are the candidates of an `ambiguous` answer; a text of a vendor word only gives
      all profiles of the vendor.
    - If that gives no match, the text is read again for each profile with only the words of ITS OWN vendor taken away (a vendor named
      `Plus` does not hide `MINI+`).
    """
    first = _lookup(index, query)
    if first is not None and first.status == "match":
        return first
    second = _lookup_own_vendor(index, query) if index.strip_vendors(words(query))[1] else None
    if second is not None and second.status == "match":
        return second
    return first if first is not None else second


def _lookup(index: NameIndex, query: str) -> Hit | None:
    """`match` or `ambiguous` for the text with the vendor words of all vendors taken away (a vendor word is a hard filter), or None."""
    rest, found = index.strip_vendors(words(query))
    pool = [p for p in index.profiles if not found or p.vendor in found]
    ps = parts(rest)
    key = " ".join(ps)
    if ps:
        exact = [p for p in pool if key in index._keys[p.id]]
        if len(exact) == 1:
            p = exact[0]
            return Hit("match", p, "name" if key == index._model_key[p.id] else "alias")
        if exact:
            return Hit("ambiguous", None, "name", tuple(exact))
        contained = [p for p in pool if index.holds(p.id, ps)]
        if contained:
            return Hit("ambiguous", None, "partial", tuple(contained))
    elif found and pool:
        return Hit("ambiguous", None, "vendor", tuple(pool))
    return None


def _lookup_own_vendor(index: NameIndex, query: str) -> Hit | None:
    """Like `_lookup`, but each profile drops only the words of its own vendor from the text. A vendor word of another vendor stays in the
    text, so it must be a word of the name. `match` or `ambiguous`, or None."""
    ws = words(query)
    exact, contained = [], []
    for p in index.profiles:
        ps = parts(index._strip(ws, index._own[p.id])[0])
        if not ps:
            continue
        key = " ".join(ps)
        if key in index._keys[p.id]:
            exact.append((p, key))
        elif index.holds(p.id, ps):
            contained.append(p)
    if len(exact) == 1:
        p, key = exact[0]
        return Hit("match", p, "name" if key == index._model_key[p.id] else "alias")
    if exact:
        return Hit("ambiguous", None, "name", tuple(p for p, _ in exact))
    if contained:
        return Hit("ambiguous", None, "partial", tuple(contained))
    return None


def suggest_names(index: NameIndex, query: str, cutoff: float, maximum: int) -> tuple[tuple[object, float], ...]:
    """The "did you mean" list for a text that `find` could not place: the profiles whose compact names have a difflib `SequenceMatcher`
    ratio of at least `cutoff` to the text, at most `maximum`, the best first. A suggestion is never taken by itself."""
    rest, found = index.strip_vendors(words(query))
    ps = parts(rest)
    variants = ["".join(w for w in words(query) if w not in FILLER_WORDS)]
    if found and ps:
        variants.append("".join(ps))
    variants = [v for v in variants if v]
    scored = []
    for p in index.profiles:
        names = index.names_of(p)
        score = max((difflib.SequenceMatcher(None, q, n).ratio() for q in variants for n in names), default=0.0)
        if score >= cutoff:
            scored.append((p, score))
    scored.sort(key=lambda ps_: -ps_[1])        # a stable sort: the catalog order breaks a tie
    return tuple(scored[:maximum])


# --- the user file -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class UserFile:
    """The facts about a user file of a work folder (`Catalog.user_file`)."""
    path: str
    exists: bool
    entries: int = 0                       # the user entries that are in use
    problems: tuple[str, ...] = ()         # what is wrong with the file or with an entry (an entry with a fault is not used)

    def facts(self) -> dict:
        return {"path": self.path, "exists": self.exists, "entries": self.entries, "problems": list(self.problems)}


class FileFault(CatalogError):
    """The user file as a whole cannot be used (not an entry: the entries have their own problem texts)."""


class Dumper(yaml.SafeDumper):
    """Mappings in block style (a person edits the file), lists of numbers and names in one line."""


Dumper.add_representer(list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True))


def read_user_file(path: Path, top_keys: tuple[str, ...]):
    """`(state, mapping, problems)`. `state` is `absent`, `unreadable` (the system does not let the server read it), `broken` or `ok`."""
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return "absent", None, []
    except IsADirectoryError:
        return "broken", None, ["cannot read the file: the name is a folder"]
    except UnicodeDecodeError as exc:
        return "broken", None, [f"not valid UTF-8 text: {' '.join(str(exc).split())}"]
    except OSError as exc:
        cause = f"cannot read the file: {exc.strerror or ' '.join(str(exc).split())}"
        if exc.errno == errno.EPERM and sys.platform == "darwin":      # macOS asks the user before an app reads a folder such as ~/Documents
            cause += f". {workfolder.MAC_PRIVACY}"
        return "unreadable", None, [cause]
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return "broken", None, [f"not valid YAML: {' '.join(str(exc).split())}"]
    if data is None:                                    # an empty file
        data = {}
    if not isinstance(data, dict):
        return "broken", None, [f"the file must hold a mapping of keys ({', '.join(top_keys)}), got {type(data).__name__}"]
    return "ok", data, []


def canonical_vendor(vendors: dict, text: str) -> str:
    """The name of the vendor table that `text` is (the name or one of its words, case and punctuation do not matter), else `text`."""
    key = tuple(words(text))
    for name, ws in vendors.items():
        if any(tuple(words(t)) == key for t in (name, *ws)):
            return name
    return text.strip()
