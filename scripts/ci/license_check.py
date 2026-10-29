#!/usr/bin/env python3
"""The licence step of CI: read the licence of every package that is installed in the venv of the leg.

    .venv/bin/python scripts/ci/license_check.py --leg linux-x86_64

The set is every installed distribution of the venv except the project itself (`importlib.metadata`). The leg
venv is clean, so this holds the core dependencies, the installed extras and everything that they pull in,
also a package that a requirement with an extra marker brings (for example `pyjwt[crypto]` of `mcp`). For each
package the script reads the installed metadata (`License-Expression`, then the `License ::` classifiers, then the `License` text) and sorts the
licence into `permissive`, `copyleft`, `proprietary` or `unknown`. An `override` entry of the policy replaces the
category for a package whose metadata does not match its files (each override states why).

It fails when:
1. a package is not permissive and `license_policy.toml` has no `allow` entry for it (the entries are the
   decided exceptions, each with its decision), or
2. a core dependency or an installed extra package, or a policy exception, has no row or mention in
   THIRD_PARTY_NOTICES.md.

The notices file comes with PR #27. The naming rule (2) runs when the file exists, or when pyproject.toml
lists it in `license-files`. Before PR #27 merges, the script says that the rule did not run.
The test tests/test_third_party_notices.py (PR #27) checks the declared packages and their pins. This script
checks the installed set and the licence text of each package. The two do not replace each other.
Standard library only (Python 3.11 or later).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, get_leg, md_table, run_main, write_summary  # noqa: E402

POLICY_FILE = Path(__file__).resolve().with_name("license_policy.toml")
NOTICES_NAME = "THIRD_PARTY_NOTICES.md"
PROJECT_NAME = "topopt"

RANK = {"permissive": 0, "copyleft": 1, "proprietary": 2, "unknown": 3}
NAMES = {v: k for k, v in RANK.items()}

PERMISSIVE_IDS = {
    "mit", "mit-0", "mit-cmu", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-open-mpi", "bsd-4-clause", "0bsd",
    "apache-2.0", "isc", "psf-2.0", "python-2.0", "zlib", "cc0-1.0", "unlicense", "hpnd", "bsl-1.0", "ftl",
    "curl", "x11", "libtiff", "ncsa",
}
COPYLEFT_PREFIXES = ("gpl-", "lgpl-", "agpl-", "mpl-", "epl-", "eupl-", "cddl-", "osl-", "cc-by-sa", "sleepycat")
EXCEPTION_IDS = ("llvm-exception", "gcc-exception", "classpath-exception")

CLASSIFIER_WORDS = [      # (text in the classifier, category). The first match counts.
    ("other/proprietary", "proprietary"), ("proprietary", "proprietary"),
    ("lesser general public", "copyleft"), ("general public license", "copyleft"), ("gpl", "copyleft"),
    ("mozilla public", "copyleft"), ("eclipse public", "copyleft"), ("common development and distribution", "copyleft"),
    ("mit", "permissive"), ("bsd", "permissive"), ("apache", "permissive"), ("isc", "permissive"),
    ("python software foundation", "permissive"), ("zlib", "permissive"), ("unlicense", "permissive"),
    ("public domain", "permissive"), ("cc0", "permissive"), ("historical permission notice", "permissive"),
]


# --- classify one licence ---------------------------------------------------------------------

def _classify_id(token: str) -> str:
    t = token.lower().rstrip("+")
    if t in PERMISSIVE_IDS or t.rstrip("+") in PERMISSIVE_IDS:
        return "permissive"
    if t.startswith(COPYLEFT_PREFIXES):
        return "copyleft"
    return "unknown"


def classify_expression(expr: str) -> str:
    """The category of an SPDX expression. `A OR B` takes the better one, `A AND B` the worse one.

    `WITH <exception>` does not change the category of the licence before it.
    """
    tokens = re.findall(r"\(|\)|[^\s()]+", expr)
    pos = 0

    def parse_or() -> int:
        nonlocal pos
        best = parse_and()
        while pos < len(tokens) and tokens[pos].upper() == "OR":
            pos += 1
            best = min(best, parse_and())
        return best

    def parse_and() -> int:
        nonlocal pos
        worst = parse_atom()
        while pos < len(tokens) and tokens[pos].upper() == "AND":
            pos += 1
            worst = max(worst, parse_atom())
        return worst

    def parse_atom() -> int:
        nonlocal pos
        if pos >= len(tokens):
            return RANK["unknown"]
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            val = parse_or()
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1
            return val
        rank = RANK[_classify_id(tok)]
        if pos < len(tokens) and tokens[pos].upper() == "WITH":
            pos += 1
            pos += 1          # the exception id
        return rank

    return NAMES[parse_or()]


def classify_classifiers(classifiers: list[str]) -> str | None:
    """The category of the `License ::` classifiers (the worst one counts), or None if there is none that tells."""
    cats = []
    for c in classifiers:
        if not c.startswith("License ::"):
            continue
        low = c.lower()
        for word, cat in CLASSIFIER_WORDS:
            if word in low:
                cats.append(RANK[cat])
                break
    return NAMES[max(cats)] if cats else None


def classify_text(text: str) -> str:
    """The category of a free `License` text. A rough rule: the text names a family."""
    t = text[:4000]
    if re.search(r"intel|proprietary|end user license", t, re.I):
        return "proprietary"
    if re.search(r"\bA?GPL\b|GNU (Lesser |Affero )?General Public|Mozilla Public", t, re.I):
        return "copyleft"
    if re.search(r"\bMIT\b|BSD|Apache|\bISC\b|\bPSF\b|Python Software Foundation|Permission is hereby granted|Zlib", t, re.I):
        return "permissive"
    return "unknown"


def classify_package(md) -> tuple[str, str]:
    """Return (category, the text that decided it) for the metadata of one installed package."""
    expr = (md.get("License-Expression") or "").strip()
    if expr:
        return classify_expression(expr), f"License-Expression: {expr}"
    cls = classify_classifiers(md.get_all("Classifier") or [])
    if cls:
        names = [c.split("::")[-1].strip() for c in md.get_all("Classifier") if c.startswith("License ::")]
        return cls, "classifier: " + "; ".join(names)
    text = (md.get("License") or "").strip()
    if text:
        return classify_text(text), "License: " + text.splitlines()[0][:70]
    return "unknown", "no licence field"


# --- the package set --------------------------------------------------------------------------

def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def req_name(req: str) -> str:
    return norm(re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req).group(1))


def declared_packages(pyproject_text: str, extras: list[str]) -> dict[str, str]:
    """{normalised name: group} of the core packages and of the packages of the given extras."""
    cfg = tomllib.loads(pyproject_text)["project"]
    out = {req_name(r): "core" for r in cfg.get("dependencies", [])}
    for e in extras:
        for r in cfg.get("optional-dependencies", {}).get(e, []):
            out.setdefault(req_name(r), f"extra {e}")
    return out


def installed_packages(exclude: tuple[str, ...] = (PROJECT_NAME,)) -> dict[str, object]:
    """{normalised name: metadata} of every distribution that is installed in this Python, except `exclude`.

    A distribution that appears twice on `sys.path` counts once (the first one wins, as for an import).
    """
    out: dict[str, object] = {}
    skip = {norm(n) for n in exclude}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name or norm(name) in skip or norm(name) in out:
            continue
        out[norm(name)] = dist.metadata
    return out


# --- the notices file -----------------------------------------------------------------------------

def notices_names(text: str) -> set[str]:
    """Normalised names in the first column of each table row, and in bold text (`**gmsh 4.15.2 (extra `tet`)**`)."""
    names = set()
    for line in text.splitlines():
        if line.startswith("|"):
            cell = line.strip().strip("|").split("|")[0].strip()
            m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", cell)
            if m and cell not in ("Package",) and set(cell) - set("-: "):
                names.add(norm(m.group(1)))
        for bold in re.findall(r"\*\*([^*]+)\*\*", line):
            m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", bold.strip())
            if m:
                names.add(norm(m.group(1)))
    return names


def notices_required(pyproject_text: str, root: Path) -> tuple[bool, str]:
    """Whether the naming rule runs, and why (a fact for the log)."""
    if (root / NOTICES_NAME).is_file():
        return True, f"{NOTICES_NAME} is in the repo"
    patterns = tomllib.loads(pyproject_text)["project"].get("license-files") or []
    if any(fnmatch.fnmatchcase(NOTICES_NAME, p) for p in patterns):
        return True, f"pyproject.toml lists {NOTICES_NAME} in license-files, and the file is missing"
    return False, f"{NOTICES_NAME} is not in the repo and pyproject.toml does not list it, so there is nothing to check"


# --- the check ---------------------------------------------------------------------------------------

def check_licenses(declared: dict[str, str], packages: dict[str, object], policy: dict, named: set[str] | None) -> list[dict]:
    """One row for each package: category, status and the problems. `named` is None if the naming rule did not run."""
    allow = {norm(e["package"]): e for e in policy.get("allow", [])}
    override = {norm(e["package"]): e for e in policy.get("override", [])}
    rows = []
    for name in sorted(packages):
        md = packages[name]
        cat, source = classify_package(md)
        if name in override:
            cat, source = override[name]["category"], f"override ({source}): {override[name]['why']}"
        row = {"package": name, "version": md.get("Version"), "category": cat, "source": source,
               "group": declared.get(name, "installed, not declared"), "problems": [], "exception": None}
        if cat != "permissive":
            if name in allow:
                row["exception"] = allow[name]["decision"]
            else:
                row["problems"].append(f"licence is {cat} ({source}) and license_policy.toml has no entry for it")
        if named is not None and (name in declared or row["exception"]) and name not in named:
            row["problems"].append(f"{NOTICES_NAME} does not name it")
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--leg", required=True)
    ap.add_argument("--pyproject", default=str(ROOT / "pyproject.toml"))
    ap.add_argument("--policy", default=str(POLICY_FILE))
    ap.add_argument("--root", default=str(ROOT), help="the repo folder (where THIRD_PARTY_NOTICES.md is)")
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args(argv)

    leg = get_leg(args.leg)
    pyproject_text = Path(args.pyproject).read_text()
    declared_extras = tomllib.loads(pyproject_text)["project"].get("optional-dependencies", {})
    extras = [e for e in leg["extras"] if e in declared_extras]
    declared = declared_packages(pyproject_text, extras)
    packages = installed_packages()
    policy = tomllib.loads(Path(args.policy).read_text())

    root = Path(args.root)
    run_naming, why = notices_required(pyproject_text, root)
    named = notices_names((root / NOTICES_NAME).read_text()) if run_naming and (root / NOTICES_NAME).is_file() else (set() if run_naming else None)
    rows = check_licenses(declared, packages, policy, named)

    missing = [n for n in declared if n not in packages]
    out = Path(args.artifacts).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "licenses.json").write_text(json.dumps(rows, indent=2) + "\n")
    table = md_table(["package", "version", "category", "from", "group", "state"],
                     [[r["package"], r["version"], r["category"], r["source"][:60], r["group"],
                       "; ".join(r["problems"]) or (f"exception: {r['exception'][:70]}" if r["exception"] else "ok")]
                      for r in rows])
    bad = [r for r in rows if r["problems"]]
    note = (f"Naming rule: {'ran' if run_naming else 'did not run'} ({why}).")
    if missing:
        note += f" Declared but not installed here (platform markers): {', '.join(missing)}."
    write_summary(f"### Licences ({args.leg}): {len(rows)} packages, {len(bad)} with a problem\n\n{note}\n\n{table}")
    print(f"{len(rows)} packages; extras {extras}; {note}")
    counts = {c: sum(1 for r in rows if r["category"] == c) for c in RANK}
    print("categories: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for r in rows:
        if r["category"] != "permissive" or r["problems"]:
            print(f"  {r['package']} {r['version']}: {r['category']} ({r['source']}) -> "
                  + ("; ".join(r["problems"]) or f"exception: {r['exception']}"))
    if bad:
        raise CiError(f"{len(bad)} package(s) with a licence problem: " + "; ".join(f"{r['package']}: {' / '.join(r['problems'])}" for r in bad))
    naming = ("the notices name the declared packages and the exceptions" if run_naming
              else "the naming rule against the notices did not run")
    print(f"OK: every package is permissive or a recorded exception; {naming}")
    return 0


if __name__ == "__main__":
    run_main(main)
