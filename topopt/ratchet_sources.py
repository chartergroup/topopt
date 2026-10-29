"""Candidate sources of the ratchet (CONTRACTS.md section 36).

A source makes candidate edits of a finished part. The ratchet (`topopt/ratchet.py`) judges each candidate and keeps it only if
the rules of the case pass. A source decides nothing: it writes a candidate and gives facts about it.

The interface is small, so that a later item can add a generator without a change to the ratchet or to CONTRACTS:

    class CandidateSource:
        kind: str                      # "marks", "stl", or the name of a new generator
        needs_labels: bool             # True: the parent must hold labels.bin (the source reads the frozen regions)
        def describe(self) -> dict     # facts for the log: kind, file, sha256, count
        def propose(self, parent_dir, out_dir, history) -> Candidate | None

`propose` returns the next candidate for the CURRENT parent, or None when the source has no more. `history` is the list of
log rows read from disk. A source skips every `ref` that it finds there with the same content (`judged_refs` gives the content
hashes that the log holds for each ref), so a restart goes on with the next candidate. A ref with other content is a new
candidate: the same name does not hide a new file.
`out_dir` is a new folder that the source may write into (it does not exist when `propose` starts).

Two sources are here: `MarksSource` (one candidate for each mark of a marks file, made by `topopt.apply_marks`) and
`StlFolderSource` (each STL file of a folder is one candidate: a part that another tool made). The tests also have a source that
cuts a box out of the parent (`tests/ratchet_helpers.py`).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .config import UserError
from .runinfo import sha256_file


@dataclass
class Candidate:
    """One candidate edit. `stl` is None when the source made no change (the runner logs it as a REJECT)."""
    kind: str
    ref: str                             # names the candidate in the source, for example "marks.json#3" or "part_b.stl"
    stl: Path | None
    dir: Path | None = None              # the folder the source wrote (None when it wrote nothing)
    id: str = ""                         # set by the runner ("c007")
    facts: dict = field(default_factory=dict)      # facts of the source, for the log (JSON-safe)
    note: str = ""                       # why there is no candidate, or a remark
    error: dict | None = None            # {"exit_code", "message"} when the source itself failed
    seconds: float | None = None         # time the source needed


class CandidateSource(Protocol):
    """The interface of a source (see the module text)."""
    kind: str
    needs_labels: bool

    def describe(self) -> dict: ...

    def propose(self, parent_dir: Path, out_dir: Path, history: list[dict]) -> "Candidate | None": ...


def tried_refs(history: list[dict]) -> set[str]:
    """The refs of the candidates that the log already holds (every decision counts)."""
    refs = ((r.get("source") or {}).get("ref") for r in history if r.get("type") == "candidate")
    return {ref for ref in refs if isinstance(ref, str)}


def judged_refs(history: list[dict]) -> dict[str, set]:
    """`{ref: the content hashes judged under that ref}`. The hash is `inputs.marks_sha256` (a marks file: the whole file) or
    `inputs.candidate_sha256` (an STL file). A source offers a ref again when its content is not in the set."""
    out: dict[str, set] = {}
    for r in history:
        if r.get("type") != "candidate":
            continue
        ref = (r.get("source") or {}).get("ref")
        if isinstance(ref, str):
            inputs = r.get("inputs") or {}
            out.setdefault(ref, set()).add(inputs.get("marks_sha256") or inputs.get("candidate_sha256"))
    return out


def promoted_before(history: list[dict]) -> bool:
    """True if a candidate was promoted: the parent is not the start part any more."""
    return any(r.get("promoted") and r.get("type") in ("candidate", "approval") for r in history)


def natural_key(name: str) -> list:
    """A sort key that reads the numbers in a name as numbers: part_2 comes before part_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


class MarksSource:
    """One candidate for each mark of a marks file (the Refine page exports it), made by `apply_marks` on the current parent.

    The marks are tried in file order. After the first promotion the parent is not the part that the marks were made for:
    the hash `source_sha256` of the file does not match, and the source applies the mark anyway (`allow_stale`). The facts of
    the candidate say so (`source_sha256_stale`). For the start part, a hash that does not match is a refusal of
    `apply_marks`, and the record says so.
    """
    kind = "marks"
    needs_labels = True

    def __init__(self, marks_path, yaml_path=None, options=None):
        from .feedback import read_marks
        self.path = Path(marks_path).expanduser().resolve()
        self.doc = read_marks(str(self.path))
        self.yaml_path = Path(yaml_path).resolve() if yaml_path else None
        self.options = dict(options or {})
        self.sha256 = sha256_file(self.path)

    def describe(self) -> dict:
        return {"kind": self.kind, "file": self.path.name, "sha256": self.sha256, "count": len(self.doc["marks"])}

    def _ref(self, index: int, mark) -> str:
        mid = mark.get("id", index + 1) if isinstance(mark, dict) else index + 1
        return f"{self.path.name}#{mid}"

    def propose(self, parent_dir, out_dir, history):
        done = judged_refs(history)
        for i, mark in enumerate(self.doc["marks"]):
            ref = self._ref(i, mark)
            if self.sha256 not in done.get(ref, ()):        # the same ref with another marks file is a new candidate
                cand = self._apply(Path(parent_dir), Path(out_dir), ref, mark, stale_ok=promoted_before(history))
                if ref in done:
                    cand.facts["same_ref_as_earlier"] = True
                    cand.note = (cand.note + "; " if cand.note else "") + "the log holds this ref for another marks file"
                return cand
        return None

    def _apply(self, parent_dir: Path, out_dir: Path, ref: str, mark, stale_ok: bool) -> Candidate:
        from .apply_marks import apply_marks
        t0 = time.perf_counter()
        doc = dict(self.doc, marks=[mark])
        rep = apply_marks(parent_dir, doc, out_dir=out_dir, options=self.options, allow_stale=stale_ok,
                          yaml_path=self.yaml_path)
        secs = round(time.perf_counter() - t0, 2)
        if not rep.get("ok"):
            msg = str(rep.get("error") or "apply_marks failed")
            return Candidate(self.kind, ref, None, None, note=msg, error={"exit_code": 1, "message": msg}, seconds=secs)
        entry = (rep.get("marks") or [{}])[0]
        totals = rep.get("totals") or {}
        edited = (rep.get("files") or {}).get("edited_stl")
        exact = totals.get("frozen_exact") or {}
        facts = {"marks_status": entry.get("status"), "marks_match_final_stl": rep.get("marks_match_final_stl"),
                 "source_sha256_stale": rep.get("marks_match_final_stl") is False,
                 "mass_after_g": totals.get("mass_after_g"), "volume_after_mm3": totals.get("volume_after_mm3"),
                 "change_mm3": totals.get("change_mm3"), "bodies_after": totals.get("bodies_after"),
                 "closed_after": totals.get("closed_after"), "frozen_unchanged": exact.get("unchanged"),
                 "report": "apply_marks.json", "problems": list(rep.get("problems") or [])}
        if not edited:
            why = entry.get("reason") or entry.get("status") or "the mark changed nothing"
            return Candidate(self.kind, ref, None, Path(out_dir), facts=facts,
                             note=f"apply_marks made no edited.stl: {entry.get('status')}: {why}", seconds=secs)
        return Candidate(self.kind, ref, Path(out_dir) / edited, Path(out_dir), facts=facts, seconds=secs)


class StlFolderSource:
    """Each STL file of a folder is one candidate: a whole part that another tool made (CAD, a script, a generator).

    The files are tried in natural name order (part_2 before part_10). A file is judged against the current parent,
    whatever the parent was when the file was made. The source writes nothing.
    """
    kind = "stl"
    needs_labels = False

    def __init__(self, folder):
        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.is_dir():
            raise UserError(f"the candidate folder {self.folder} does not exist")
        self.files = sorted((p for p in self.folder.iterdir() if p.is_file() and p.suffix.lower() == ".stl"),
                            key=lambda p: natural_key(p.name))
        if not self.files:
            raise UserError(f"the candidate folder {self.folder} has no .stl file")

    def describe(self) -> dict:
        return {"kind": self.kind, "folder": self.folder.name, "count": len(self.files)}

    def propose(self, parent_dir, out_dir, history):
        done = judged_refs(history)
        for p in self.files:
            if p.name not in done:
                return Candidate(self.kind, p.name, p, None, facts={"file": p.name})
            if sha256_file(p) not in done[p.name]:          # the same name with other content is a new candidate
                return Candidate(self.kind, p.name, p, None, facts={"file": p.name, "same_ref_as_earlier": True},
                                 note="the log holds a file with this name and other content")
        return None


def parse_source(spec: str, yaml_path=None):
    """`marks:FILE` or `stl:FOLDER` to a source. Raise `UserError` for any other text."""
    kind, sep, rest = str(spec).partition(":")
    if not sep or not rest.strip():
        raise UserError(f"the source {spec!r} is not understood. Use marks:FILE (a marks file of the Refine page) or "
                        "stl:FOLDER (a folder of candidate STL files)")
    if kind == "marks":
        return MarksSource(rest, yaml_path=yaml_path)
    if kind == "stl":
        return StlFolderSource(rest)
    raise UserError(f"the source kind {kind!r} is not known. Use marks:FILE or stl:FOLDER")
